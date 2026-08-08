"""
Embedding index for semantic search.

Opt-in only (controlled by config.embeddings.enabled). Stores page embeddings
as JSON in .state/embeddings.json. Lazy indexing: pages are indexed after writes
(organizer/curate) and can be bulk-rebuilt via CLI.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Optional

from graybox.config import Config
from graybox.models import Page

logger = logging.getLogger(__name__)

_EMBEDDING_CACHE: dict[str, "EmbeddingIndex"] = {}
_NOISE_FLOOR = 0.05
# Fixed high-end anchor for calibrating raw cosine similarity onto the 0-1
# scale (see EmbeddingIndex.search). The low end comes from
# cfg.retrieval.semantic_min_score so it's tunable per embedding model;
# this constant is a reasonable default ceiling for the OpenAI-style
# small embedding models this project defaults to. If you swap embedding
# providers and semantic scores feel systematically too low/high, this is
# the number to adjust.
_CALIBRATION_HIGH = 0.5

def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingIndex:
    """Simple JSON-backed embedding store. Scales to thousands of pages."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.path = cfg.state_dir / "embeddings.json"
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def _search_blob(self, page: Page) -> str:
        """Mirror of search_engine._PageDoc.search_blob for consistency."""
        return " ".join([
            page.title, page.summary, " ".join(page.notes), " ".join(page.aliases),
            " ".join(page.tags), page.type, page.status, page.owner, page.due, page.date,
            " ".join(page.attendees), page.id,
        ])

    def index_page(self, page: Page, embedding: list[float]) -> None:
        """Store embedding for a page. Call after write_page."""
        blob = self._search_blob(page)
        self._data[page.ref] = {
            "embedding": embedding,
            "text_hash": _text_hash(blob),
            "indexed_at": __import__("graybox.models", fromlist=["now_iso"]).now_iso(),
        }
        self._save()
        logger.debug("Indexed embedding for %s", page.ref)

    def needs_reindex(self, page: Page) -> bool:
        """Check if page content changed since last index."""
        entry = self._data.get(page.ref)
        if not entry:
            return True
        blob = self._search_blob(page)
        return entry.get("text_hash") != _text_hash(blob)

    def search(self, query_embedding: list[float], *, top_k: int = 10, min_score: float = 0.5) -> list[tuple[str, float]]:
        """Return (ref, similarity) tuples, sorted descending, on the SAME
        0-1 scale as coverage_scorer/name_scorer: closer to 1.0 = stronger
        match, closer to 0.0 = weak/no match.

        Raw cosine similarity from the embedding model runs on its own,
        much lower and less intuitive range (genuine matches often sit
        ~0.15-0.35, never near 1.0). We rescale it here once using FIXED,
        query-independent anchors (`semantic_min_score` as the low end,
        `_CALIBRATION_HIGH` as the high end) so a page's normalized score
        reflects how similar it actually is to the query - not how it
        compares to whatever else happens to be in this result set.

        NOTE: this used to be a per-query min-max normalization (best
        candidate -> 1.0, worst -> 0.0). That made min_score meaningless
        whenever few pages were indexed: with only one candidate, spread
        was always 0 and it was unconditionally normalized to 1.0 no
        matter how weak the actual match was. Fixed anchors fix that -
        a genuinely weak match now stays low even when it's the "best
        available" one.
        """
        low = self.cfg.retrieval.semantic_min_score
        span = max(_CALIBRATION_HIGH - low, 1e-9)

        raw: list[tuple[str, float]] = []
        for ref, entry in self._data.items():
            emb = entry.get("embedding")
            if not emb:
                continue
            sim = _cosine_similarity(query_embedding, emb)
            if sim >= _NOISE_FLOOR:
                raw.append((ref, sim))

        if not raw:
            return []

        normalized: list[tuple[str, float]] = []
        for ref, sim in raw:
            norm = (sim - low) / span
            normalized.append((ref, round(max(0.0, min(1.0, norm)), 4)))

        normalized.sort(key=lambda x: x[1], reverse=True)
        results = [(ref, s) for ref, s in normalized if s >= min_score]
        return results[:top_k]

    def drop(self, ref: str) -> None:
        """Remove a page from the index (e.g. after delete)."""
        if ref in self._data:
            del self._data[ref]
            self._save()

    def stats(self) -> dict:
        return {"indexed_pages": len(self._data), "path": str(self.path)}


def _get_index(cfg: Config) -> Optional[EmbeddingIndex]:
    if not getattr(cfg.embeddings, "enabled", False):
        return None
    key = str(cfg.workspace)
    if key not in _EMBEDDING_CACHE:
        _EMBEDDING_CACHE[key] = EmbeddingIndex(cfg)
    return _EMBEDDING_CACHE[key]


def ensure_indexed(cfg: Config, page: Page, embed_model) -> bool:
    """Index a page if embeddings are enabled and it needs reindexing.
    Returns True if indexed successfully.
    """
    idx = _get_index(cfg)
    if idx is None:
        return False
    if not idx.needs_reindex(page):
        return True
    try:
        blob = idx._search_blob(page)
        result = embed_model.embedding_call(blob)
        if result and result.get("embedding"):
            idx.index_page(page, result["embedding"])
            return True
    except Exception as e:
        logger.warning("Failed to index embedding for %s: %s", page.ref, e)
    return False


def search_embeddings(cfg: Config, query_embedding: list[float], *, top_k: int = 10, min_score: float = 0.5) -> list[tuple[str, float]]:
    """Semantic search over indexed pages."""
    idx = _get_index(cfg)
    if idx is None:
        return []
    return idx.search(query_embedding, top_k=top_k, min_score=min_score)