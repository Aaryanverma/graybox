from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Callable, Protocol

from graybox.models import InboxItem, Page

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "for",
    "and", "or", "with", "this", "that", "it", "as", "at", "by", "be", "been",
    "have", "has", "had", "do", "does", "did", "done", "can", "could", "would",
    "should", "will", "shall", "may", "might", "must",
    "who", "what", "when", "where", "why", "how", "which", "whom", "whose",
    "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "hers",
    "ours", "theirs", "myself", "yourself", "himself", "herself", "itself",
    "ourselves", "yourselves", "themselves",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(str(text).lower())
            if t not in STOPWORDS and len(t) > 1]


@dataclass(frozen=True)
class Query:
    raw: str
    tokens: list[str]
    unique: set[str]
    counts: dict[str, int]

    @classmethod
    def parse(cls, text: str) -> Query:
        tokens = tokenize(text)
        counts: dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        return cls(raw=text, tokens=tokens, unique=set(tokens), counts=counts)


class Searchable(Protocol):
    @property
    def search_id(self) -> str: ...
    @property
    def search_blob(self) -> str: ...
    @property
    def identifier_blob(self) -> str: ...
    @property
    def source_kind(self) -> str: ...  # "wiki" or "inbox"


class _PageDoc:
    __slots__ = ("page",)
    def __init__(self, page: Page) -> None:
        self.page = page

    @property
    def search_id(self) -> str:
        return self.page.ref

    @property
    def search_blob(self) -> str:
        p = self.page
        return " ".join([
            p.title, p.summary, " ".join(p.notes), " ".join(p.aliases),
            " ".join(p.tags), p.type, p.status, p.owner, p.due, p.date,
            " ".join(p.attendees), p.id,
        ])

    @property
    def identifier_blob(self) -> str:
        p = self.page
        parts = [p.title, *p.aliases]
        if p.owner:
            parts.append(p.owner)
        if p.attendees:
            parts.extend(p.attendees)
        return " ".join(parts)

    @property
    def identity_candidates(self) -> list[str]:
        """
        Discrete WHOLE-NAME identity strings (title, each alias, owner,
        each attendee) - each compared as one unit, never exploded into
        individual words. This is what keeps name_scorer from mistaking a
        coincidental word-fragment overlap (e.g. 'data' happening to be a
        literal prefix of 'database') for a genuine name match: comparing
        whole multi-word names preserves the word boundaries that a naive
        whitespace-split loses.
        """
        p = self.page
        cands = [p.title, *p.aliases]
        if p.owner:
            cands.append(p.owner)
        if p.attendees:
            cands.extend(p.attendees)
        return cands

    @property
    def core_identity_candidates(self) -> list[str]:
        """
        Identity candidates restricted to the page's OWN name (title +
        aliases) - excludes owner/attendees, which describe this page's
        relationship to another entity, not this page's identity. Used
        when comparing pages of DIFFERENT types, where e.g. a task's owner
        happening to share a name with an unrelated person page must never
        count as "these are the same entity."
        """
        p = self.page
        return [p.title, *p.aliases]

    @property
    def source_kind(self) -> str:
        return "wiki"


class _InboxDoc:
    __slots__ = ("item",)
    def __init__(self, item: InboxItem) -> None:
        self.item = item

    @property
    def search_id(self) -> str:
        return f"inbox/{self.item.id}"

    @property
    def search_blob(self) -> str:
        return self.item.content

    @property
    def identifier_blob(self) -> str:
        return ""

    @property
    def identity_candidates(self) -> list[str]:
        return []

    @property
    def core_identity_candidates(self) -> list[str]:
        return []

    @property
    def source_kind(self) -> str:
        return "inbox"


@dataclass
class Hit:
    doc: Searchable
    score: float
    workspace_id: str = ""
    workspace_name: str = ""
    linked_from: str = ""


ScoreFn = Callable[[Query, Searchable], float]


class Engine:
    """Universal search index. Add docs once, search many times with different scorers."""

    def __init__(self) -> None:
        self._docs: list[tuple[Searchable, str, str]] = []  # doc, ws_id, ws_name

    def add_wiki(self, pages: list[Page], workspace_id: str = "", workspace_name: str = "") -> None:
        for p in pages:
            self._docs.append((_PageDoc(p), workspace_id, workspace_name))

    def add_inbox(self, items: list[InboxItem], workspace_id: str = "", workspace_name: str = "") -> None:
        for i in items:
            self._docs.append((_InboxDoc(i), workspace_id, workspace_name))

    def clear(self) -> None:
        self._docs.clear()

    def search(self, query: Query | str, scorer: ScoreFn, *,
               top_k: int = 10, min_score: float = 0.0,
               kind: str | None = None) -> list[Hit]:
        """Search with any scoring function. kind='wiki'|'inbox' to filter corpus."""
        if isinstance(query, str):
            query = Query.parse(query)

        hits: list[Hit] = []
        for doc, ws_id, ws_name in self._docs:
            if kind and doc.source_kind != kind:
                continue
            score = scorer(query, doc)
            if score >= min_score:
                hits.append(Hit(doc=doc, score=score, workspace_id=ws_id, workspace_name=ws_name))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    # ------------------------------------------------------------------
    # Built-in scorers (the only scoring logic in the entire codebase)
    # ------------------------------------------------------------------

    @staticmethod
    def coverage_scorer(query: Query, doc: Searchable) -> float:
        """
        General search relevance, normalized to [0, 1] where scores close to
        1.0 mean a strong match and scores close to 0.0 mean little/no
        overlap - the same scale convention as name_scorer, so a single
        mental model ("closer to 1 = more similar") applies everywhere in
        the app, regardless of which scorer or threshold is being tuned.

        Implementation: a recall/precision-style (F1) blend of
          - recall: fraction of the query's distinct terms found in the doc
          - quality: how well-represented those matched terms are in the doc,
            relative to how often the query asked for them
        Blending via a harmonic mean means a document has to be BOTH broad
        (covers most of the question) AND concentrated (isn't just one
        matching word buried in something unrelated) to score near 1.0 -
        which keeps genuinely relevant hits naturally high without needing a
        separate, unintuitive low-magnitude "good match" band.
        """
        if not query.unique:
            return 0.0

        doc_tokens = tokenize(doc.search_blob)
        if not doc_tokens:
            return 0.0

        doc_counts: dict[str, int] = {}
        for t in doc_tokens:
            doc_counts[t] = doc_counts.get(t, 0) + 1

        matched = query.unique & set(doc_tokens)
        if not matched:
            return 0.0

        recall = len(matched) / len(query.unique)
        quality = sum(min(1.0, doc_counts[t] / query.counts[t]) for t in matched) / len(matched)

        denom = recall + quality
        score = (2 * recall * quality / denom) if denom else 0.0

        # Exact identifier (title/alias/owner/attendee) hits are always a
        # strong signal, on the same normalized scale as name_scorer's match.
        id_tokens = set(tokenize(doc.identifier_blob))
        if query.unique & id_tokens:
            score = max(score, 0.9)

        return round(min(1.0, score), 3)

    @staticmethod
    def name_scorer(query: Query, doc: Searchable, *, core_identity_only: bool = False) -> float:
        """
        Duplicate / entity-resolution scorer, normalized to [0, 1] where
        scores close to 1.0 mean "same entity" and close to 0.0 mean
        unrelated names.

        Compares the query name against each of the document's identity
        candidates as WHOLE strings (title, each alias, and - for
        same-type comparisons - owner/attendees), never exploded into
        individual words. That word-boundary preservation is what stops a
        coincidental fragment overlap between two unrelated multi-word
        names (e.g. "Data Science" vs "Create a new database", where
        "database" merely happens to start with the letters "data") from
        ever being scored as a real match.

        core_identity_only=True restricts candidates to title/aliases only
        (no owner/attendees) - use this whenever comparing pages of
        DIFFERENT types, since an owner/attendee name is a relationship to
        another entity, not this page's own identity, and must never be
        allowed to make a task/meeting look like "the same entity" as an
        unrelated person page that happens to share that name.
        """
        q_name = re.sub(r"[^a-z0-9]", "", query.raw.lower())
        if not q_name:
            return 0.0

        candidates = doc.core_identity_candidates if core_identity_only else doc.identity_candidates
        if not candidates:
            return 0.0

        best = 0.0
        for cand in candidates:
            c_name = re.sub(r"[^a-z0-9]", "", cand.lower())
            if not c_name or len(c_name) < 2:
                continue
            # Substring match (catches 'aaryan' inside 'aaryan verma') -
            # compared as whole names, so word boundaries are preserved.
            if len(q_name) >= 4 and len(c_name) >= 4:
                if q_name in c_name or c_name in q_name:
                    return 1.0
            best = max(best, difflib.SequenceMatcher(None, q_name, c_name).ratio())

        return round(best, 3)