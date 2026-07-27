from __future__ import annotations

from graybox.config import Config
from graybox.search_engine import _PageDoc, Engine, Hit, Query
from graybox.storage import list_inbox_items, list_pages


def _build_engine(cfg: Config, page_type: str | None = None) -> Engine:
    engine = Engine()
    engine.add_wiki(list_pages(cfg, page_type), workspace_id=cfg.workspace_id, workspace_name=cfg.workspace_name)
    return engine


def search(cfg: Config, query: str, page_type: str | None = None,
           top_k: int = 6, all_workspaces: bool = False) -> list[Hit]:
    if not all_workspaces:
        return _build_engine(cfg, page_type).search(
            query, Engine.coverage_scorer, top_k=top_k, min_score=cfg.retrieval.min_score, kind="wiki"
        )

    engine = Engine()
    for ws in cfg.workspace_manager.list():
        ws_cfg = cfg.for_workspace(ws)
        engine.add_wiki(list_pages(ws_cfg, page_type), workspace_id=ws.id, workspace_name=ws.name)
    return engine.search(query, Engine.coverage_scorer, top_k=top_k, min_score=cfg.retrieval.min_score, kind="wiki")


def search_inbox(cfg: Config, query: str, top_k: int = 6,
                 all_workspaces: bool = False) -> list[Hit]:
    if not all_workspaces:
        engine = Engine()
        engine.add_inbox(list_inbox_items(cfg), workspace_id=cfg.workspace_id, workspace_name=cfg.workspace_name)
        return engine.search(query, Engine.coverage_scorer, top_k=top_k, min_score=cfg.retrieval.min_score, kind="inbox")

    engine = Engine()
    for ws in cfg.workspace_manager.list():
        ws_cfg = cfg.for_workspace(ws)
        engine.add_inbox(list_inbox_items(ws_cfg), workspace_id=ws.id, workspace_name=ws.name)
    return engine.search(query, Engine.coverage_scorer, top_k=top_k, min_score=cfg.retrieval.min_score, kind="inbox")


def search_all(cfg: Config, query: str, page_type: str | None = None,
               top_k: int = 6, all_workspaces: bool = False,
               min_score: float | None = None) -> tuple[list[Hit], list[Hit]]:
    """Unified search over both corpora. Returns (wiki_hits, inbox_hits)."""
    threshold = min_score if min_score is not None else cfg.retrieval.min_score
    engine = Engine()
    workspaces = cfg.workspace_manager.list() if all_workspaces else [cfg.workspace_manager.current()]
    for ws in workspaces:
        ws_cfg = cfg.for_workspace(ws)
        engine.add_wiki(list_pages(ws_cfg, page_type), workspace_id=ws.id, workspace_name=ws.name)
        engine.add_inbox(list_inbox_items(ws_cfg), workspace_id=ws.id, workspace_name=ws.name)

    q = Query.parse(query)
    wiki = engine.search(q, Engine.coverage_scorer, top_k=top_k, min_score=threshold, kind="wiki")
    inbox = engine.search(q, Engine.coverage_scorer, top_k=top_k, min_score=threshold, kind="inbox")
    return wiki, inbox


def find_duplicates(cfg: Config, page_type: str | None = None,
                    threshold: float = 0.84) -> list[tuple[Hit, Hit, float, str]]:
    """Universal duplicate detection using the same name scorer."""
    pages = list_pages(cfg, page_type)
    hits: list[tuple[Hit, Hit, float, str]] = []
    seen = set()

    for i, a in enumerate(pages):
        for b in pages[i + 1:]:
            pair = tuple(sorted((a.ref, b.ref)))
            if pair in seen:
                continue
            if a.type != b.type:
                continue
            score_a = Engine.name_scorer(Query.parse(a.title), _PageDoc(b))
            score_b = Engine.name_scorer(Query.parse(b.title), _PageDoc(a))
            score = max(score_a, score_b)
            if score >= threshold:
                seen.add(pair)
                reason = "fuzzy name match"
                hits.append((Hit(_PageDoc(a), score), Hit(_PageDoc(b), score), score, reason))

    hits.sort(key=lambda x: x[2], reverse=True)
    return hits


# Re-export for backward compat
from graybox.search_engine import Engine as SearchEngine, Query as SearchQuery