"""
Retrieval Agent.
...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import re
from typing import Any
from collections import deque

from graybox.config import Config
from graybox.ai import AIService
from graybox.prompts import (
    RETRIEVAL_PROMPT_TMPL,
    RETRIEVAL_SYSTEM,
    CHAT_RETRIEVAL_PROMPT_TMPL,
    QUERY_REPHRASE_SYSTEM_PROMPT_TMPL,
    QUERY_REPHRASE_PROMPT,
    RETRIEVAL_COMPRESSION_PROMPT,
    HISTORY_COMPRESSION_PROMPT,
)
from graybox.search import search_all
from graybox.search_engine import Hit, _PageDoc
from graybox.workspace import workspace_context_block
from graybox.embedding_index import search_embeddings
from graybox.adaptive_compressor import compress_context
from graybox.storage import read_page

logger = logging.getLogger(__name__)

_NOTE_RE = re.compile(
    r"^- \((?P<ts>[^)]*)\)\s*(?P<body>.*?)(?:\s*_\(source:\s*(?P<src>[^)]*)\)_)?\s*$"
)


@dataclass
class ConversationTurn:
    question: str
    answer: str


@dataclass
class Answer:
    text: str
    sources: list[str]
    grounded: bool
    fallback: bool = False
    fallback_kind: str = ""

_REFUSAL_PREFIX = "I don't have enough information in the knowledge base to answer that."


def _is_refusal(response: str | None) -> bool:
    if not response:
        return True
    return response.strip().lower().startswith(_REFUSAL_PREFIX.lower())


NO_EVIDENCE_MSG = (
    "I don't have enough information in the knowledge base to answer that. "
    "Try capturing more notes about this topic, or rephrase the question."
)

_WEAK_INBOX_FLOOR = 0.05

# How many prior turns to keep threading into context. Capped so a long
# chat session doesn't quietly balloon prompt size/cost on every turn.
MAX_HISTORY_TURNS = 10


def _source_tag(hit, all_workspaces: bool) -> str:
    doc = hit.doc
    search_id = (
        getattr(doc, "search_id", None) or getattr(hit, "search_id", None) or "unknown"
    )

    if not all_workspaces:
        return search_id

    workspace_id, _workspace_name = _get_workspace_meta(hit)

    # Safe fallback: do not invent provenance if we do not have it.
    return f"{workspace_id}/{search_id}" if workspace_id else search_id


def _build_history_block(
    history: list[ConversationTurn],
    llm: AIService | None,
) -> str:
    lines = []
    for turn in history:
        if _is_refusal(turn.answer) or NO_EVIDENCE_MSG in turn.answer:
            continue
        lines.append(f"User: {turn.question}")
        lines.append(f"Assistant: {turn.answer}")

    final_history = "\n".join(lines)

    if not llm or len(history) <= MAX_HISTORY_TURNS:
        return final_history

    try:
        params = llm.get_llm_params()
        max_tokens = (
            params.get("max_tokens")
            or params.get("max_completion_tokens")
            or 2048
        )

        return compress_context(
            final_history,
            llm=llm,
            max_tokens=max_tokens,
            prompt=HISTORY_COMPRESSION_PROMPT,
        )

    except Exception as e:
        logger.info(f"History compression failed\n{e};\nfalling back to truncated history")
        return "\n".join(lines[-MAX_HISTORY_TURNS * 2 :])

def _render_note(note: str) -> str:
    note = note.strip()

    first_line, *rest = note.splitlines()

    m = _NOTE_RE.match(first_line)
    if not m:
        return note

    ts = m.group("ts").strip()
    body = m.group("body").strip()
    src = (m.group("src") or "").strip()

    out = [f"[{ts}]", body]

    continuation = "\n".join(
        line.strip().lstrip(">").strip() for line in rest if line.strip()
    )
    if continuation:
        out.append(continuation)

    if src:
        out.append(f"Source: {src}")

    return "\n".join(out)


def _build_context(hits: list[Hit], all_workspaces: bool = False) -> str:
    """Render retrieved pages into an LLM-friendly context."""

    blocks: list[str] = []

    for h in hits:

        if h.doc.source_kind == "wiki":
            p = h.doc.page

            tag = _source_tag(h, all_workspaces)

            header = [
                "=" * 80,
                f"Source Tag: [{tag}]",
            ]

            if h.linked_from:
                header.append(f"Retrieved via: {h.linked_from}")

            header.extend(
                [
                    f"Title: {p.title}",
                    f"Type: {p.type}",
                    f"Created: {p.created}",
                    f"Updated: {p.updated}",
                ]
            )

            summary = p.summary.strip() if p.summary else "None"

            notes = p.notes if p.notes else []

            timeline = "\n\n".join(_render_note(n) for n in notes) if notes else "None"

            related = (
                "\n".join(f"- {r}" for r in sorted(set(p.related)))
                if p.related
                else "None"
            )

            backlinks = (
                "\n".join(f"- {r}" for r in sorted(set(p.backlinks)))
                if p.backlinks
                else "None"
            )

            sources = (
                "\n".join(f"- inbox/{s}" for s in p.sources) if p.sources else "None"
            )

            block = "\n".join(header) + f"""

Summary
-------
{summary}

Timeline
--------
{timeline}

Related Pages
-------------
{related}

Backlinks
---------
{backlinks}

Original Sources
----------------
{sources}
"""

            blocks.append(block.strip())

        else:

            tag = _source_tag(h, all_workspaces)

            blocks.append(f"""
{"=" * 80}
Source Tag: [{tag}]
Raw Capture

Captured:
{h.doc.item.created}

Content
-------
{h.doc.item.content.strip()}
""".strip())

    return "\n\n".join(blocks)


def _system_prompt(cfg: Config) -> str:
    """Build the authoritative system prompt used by Ask/Chat answers.

    RETRIEVAL_SYSTEM is always the foundation. Workspace context is added
    next, and the optional answer-style configuration is appended last as a
    presentation-only preference. The style block is explicitly delimited
    and cannot change Gray Box's grounding contract.
    """
    parts = [RETRIEVAL_SYSTEM]

    ctx = workspace_context_block(cfg)
    if ctx:
        parts.append(f"Workspace context:\n{ctx}")

    style = (getattr(getattr(cfg, "prompts", None), "answer_style", "") or "").strip()
    if style:
        parts.append(
            "User-defined answer preferences (presentation only):\n"
            "<answer_style_preferences>\n"
            f"{style}\n"
            "</answer_style_preferences>\n\n"
            "These preferences control only tone, verbosity, structure, formatting, "
            "audience, personality, explanation style, and requested language. "
            "They do not override Gray Box's grounding, citation, timestamp, "
            "uncertainty, evidence, or refusal rules. If a preference conflicts "
            "with those rules, the Gray Box retrieval rules take precedence."
        )

    return "\n\n".join(parts)


def _get_workspace_meta(hit: Any) -> tuple[str | None, str | None]:
    doc = getattr(hit, "doc", None)
    workspace_id = getattr(doc, "workspace_id", None) or getattr(
        hit, "workspace_id", None
    )
    workspace_name = getattr(doc, "workspace_name", None) or getattr(
        hit, "workspace_name", None
    )
    return workspace_id, workspace_name


def _apply_workspace_meta(target: Any, source: Any) -> Any:
    target_doc = getattr(target, "doc", None)
    source_ws_id, source_ws_name = _get_workspace_meta(source)

    if target_doc is None:
        return target

    if source_ws_id and not getattr(target_doc, "workspace_id", None):
        setattr(target_doc, "workspace_id", source_ws_id)

    if source_ws_name and not getattr(target_doc, "workspace_name", None):
        setattr(target_doc, "workspace_name", source_ws_name)

    return target

def _semantic_hits_all_workspaces(cfg, query_embedding, *, top_k: int, min_score: float) -> list[Hit]:
    hits: list[Hit] = []
    workspaces = cfg.workspace_manager.list()

    for ws in workspaces:
        ws_cfg = cfg.for_workspace(ws)
        refs = search_embeddings(
            ws_cfg,
            query_embedding,
            top_k=top_k,
            min_score=min_score,
        )
        for ref, score in refs:
            page_type, slug = ref.split("/", 1)
            page = read_page(ws_cfg, page_type, slug)
            if page is None:
                continue

            hits.append(
                Hit(
                    doc=_PageDoc(page),
                    score=score,
                    workspace_id=ws.id,
                    workspace_name=ws.name,
                )
            )

    hits.sort(key=lambda h: h.score, reverse=True)

    # Deduplicate by workspace + ref, not just ref.
    seen: set[tuple[str, str]] = set()
    deduped: list[Hit] = []
    for h in hits:
        key = (h.workspace_id, h.doc.search_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(h)

    return deduped[:top_k]


def _blend_hits(keyword_hits: list[Hit], semantic_refs: list[tuple[str, float]], cfg) -> list[Hit]:
    """Merge keyword Hits with semantic search results.

    `semantic_refs` is whatever `search_embeddings()` returns:
    `list[tuple[ref, score]]` — plain (ref, score) pairs, not Hit objects.
    (embedding_index.EmbeddingIndex.search() has always returned this
    shape; nothing about it changed.)

    A page found by both keyword and semantic search gets a weighted blend
    favoring the keyword score (which reflects lexical/title relevance more
    reliably); a page found only semantically is included but damped, since
    it has no lexical corroboration.
    """
    from graybox.storage import read_page

    keyword_by_ref = {h.doc.search_id: h for h in keyword_hits}
    semantic_by_ref = {ref: score for ref, score in semantic_refs}
    all_refs = set(keyword_by_ref) | set(semantic_by_ref)

    blended: list[Hit] = []
    for ref in all_refs:
        kw_hit = keyword_by_ref.get(ref)
        sem_score = semantic_by_ref.get(ref, 0.0)

        if kw_hit and sem_score > 0:
            blended_score = round(0.6 * kw_hit.score + 0.4 * sem_score, 4)
            blended.append(Hit(
                doc=kw_hit.doc, score=blended_score,
                workspace_id=kw_hit.workspace_id, workspace_name=kw_hit.workspace_name,
                linked_from=kw_hit.linked_from,
            ))
        elif kw_hit:
            blended.append(kw_hit)
        else:
            page_type, slug = ref.split("/", 1)
            page = read_page(cfg, page_type, slug)
            if page is None:
                continue
            adjusted = round(sem_score * 0.85, 4)
            blended.append(Hit(doc=_PageDoc(page), score=adjusted))

    blended.sort(key=lambda h: h.score, reverse=True)
    return blended

def _expand_graph(
    cfg,
    hits,
    *,
    max_hops: int = 1,
    max_nodes: int = 25,
    max_neighbors_per_node: int = 5,
    decay: float = 0.65,
    graph_min_score: float = 0.35,
):
    from graybox.storage import read_page

    wiki_hits = [h for h in hits if h.doc.source_kind == "wiki"]
    if not wiki_hits:
        return hits

    # Keep the best hit per ref as the starting frontier.
    best_by_ref: dict[str, Hit] = {h.doc.search_id: h for h in wiki_hits}
    frontier = deque((h.doc.search_id, 0, h.score) for h in wiki_hits)

    seen: set[str] = set(best_by_ref.keys())
    expanded: list[Hit] = []

    while frontier and len(seen) < max_nodes:
        ref, hop, parent_score = frontier.popleft()
        if hop >= max_hops:
            continue

        page_type, slug = ref.split("/", 1)
        page = read_page(cfg, page_type, slug)
        if not page:
            continue

        # Deterministic fanout cap. No full-graph traversal.
        neighbors = sorted(set(page.related) | set(page.backlinks))
        neighbors = neighbors[:max_neighbors_per_node]

        for n_ref in neighbors:
            if n_ref in seen or len(seen) >= max_nodes:
                continue

            n_type, n_slug = n_ref.split("/", 1)
            n_page = read_page(cfg, n_type, n_slug)
            if not n_page:
                continue

            new_score = round(parent_score * (decay ** (hop + 1)), 4)
            if new_score < graph_min_score:
                continue

            hit = Hit(
                doc=_PageDoc(n_page),
                score=new_score,
                linked_from=getattr(page, "title", ref),
            )
            expanded.append(hit)
            best_by_ref[n_ref] = hit
            seen.add(n_ref)

            # Only expand the promising neighbors.
            frontier.append((n_ref, hop + 1, new_score))

    out = list(best_by_ref.values()) + expanded
    out.sort(key=lambda h: h.score, reverse=True)
    return out


def _prepare_search_query(
    llm: AIService,
    question: str,
    history: list[ConversationTurn],
    history_block: str,
) -> str:
    if not history:
        return question

    prompt = QUERY_REPHRASE_PROMPT.format(history=history_block, question=question)
    text = llm.llm_call(system_prompt=QUERY_REPHRASE_SYSTEM_PROMPT_TMPL, prompt=prompt)
    if text["response"] is None:
        logger.warning(
            "Query rewriter returned no response; falling back to original question"
        )
        return question
    return text["response"].strip()

def _dedupe_by_ref(hits: list[Hit]) -> list[Hit]:
    seen, out = set(), []
    for h in hits:
        key = (getattr(h, "workspace_id", None), h.doc.search_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out

def _answer_with(cfg, llm, hits, question, history_block, all_workspaces) -> dict:
    """Build context from `hits`, compress it, and call the LLM. Shared by
    every fallback path so context-building/compression/prompt-selection
    logic lives in exactly one place."""
    context = _build_context(hits, all_workspaces=all_workspaces)
    max_tokens = cfg.llm.max_tokens or cfg.llm.max_completion_tokens or 8196
    context = compress_context(context, llm, max_tokens=max_tokens,
                               prompt=RETRIEVAL_COMPRESSION_PROMPT)
    prompt = (CHAT_RETRIEVAL_PROMPT_TMPL if history_block else RETRIEVAL_PROMPT_TMPL).format(
        history=history_block, context=context, question=question)
    return llm.llm_call(system_prompt=_system_prompt(cfg), prompt=prompt)


def ask(cfg, llm, question, all_workspaces=False, history=None) -> Answer:
    history = history or []
    history_block = _build_history_block(history, llm)
    search_query = _prepare_search_query(llm, question, history, history_block)

    wiki_hits, inbox_hits = search_all(
        cfg, search_query, top_k=cfg.retrieval.top_k, all_workspaces=all_workspaces,
        inbox_min_score=_WEAK_INBOX_FLOOR,
    )

    # ---- semantic enrichment ----
    if getattr(cfg.embeddings, "enabled", False):
        try:
            emb_result = llm.embedding_call(search_query)
            if emb_result and emb_result.get("embedding"):
                if all_workspaces:
                    semantic_hits = _semantic_hits_all_workspaces(
                        cfg, emb_result["embedding"],
                        top_k=cfg.retrieval.top_k, min_score=cfg.retrieval.min_score,
                    )
                    if semantic_hits:
                        wiki_hits = _dedupe_by_ref(wiki_hits + semantic_hits)
                else:
                    semantic_refs = search_embeddings(
                        cfg, emb_result["embedding"],
                        top_k=cfg.retrieval.top_k, min_score=cfg.retrieval.min_score,
                    )
                    if semantic_refs:
                        wiki_hits = _blend_hits(wiki_hits, semantic_refs, cfg)
        except Exception as e:
            logger.warning("Semantic search failed (%s), falling back to keyword-only", e)

    strong_wiki = [h for h in wiki_hits if h.score >= cfg.retrieval.min_score]
    if strong_wiki and not all_workspaces:
        expanded = _expand_graph(
            cfg, strong_wiki,
            max_hops=getattr(cfg.retrieval, "graph_max_hops", 1),
            max_nodes=getattr(cfg.retrieval, "graph_max_nodes", cfg.retrieval.top_k * 3),
            max_neighbors_per_node=getattr(cfg.retrieval, "graph_max_neighbors_per_node", 5),
            decay=getattr(cfg.retrieval, "graph_decay", 0.65),
            graph_min_score=getattr(cfg.retrieval, "graph_min_score_ratio", 0.5) * cfg.retrieval.min_score,
        )
        wiki_hits = _dedupe_by_ref(expanded)
    else:
        wiki_hits = _dedupe_by_ref(wiki_hits)   # keep sub-threshold hits as fallback

    inbox_threshold = (
        cfg.retrieval.inbox_min_score
        if cfg.retrieval.inbox_min_score is not None
        else cfg.retrieval.min_score * 0.5
    )
    strong_inbox = [h for h in inbox_hits if h.score >= inbox_threshold]
    weak_inbox = [h for h in inbox_hits if h not in strong_inbox][:3]

    # ---- Path A: strong wiki hits -> normal grounded answer ----
    if strong_wiki:
        text = _answer_with(cfg, llm, wiki_hits, question, history_block, all_workspaces)
        response = text.get("response")
        if response and not _is_refusal(response):
            return Answer(text=response.strip(),
                          sources=[_source_tag(h, all_workspaces) for h in wiki_hits],
                          grounded=True, fallback=False, fallback_kind="")
        # else: the model honestly found nothing useful in these pages -
        # fall through rather than returning its refusal as final, since a
        # weaker-confidence tier below may hold the actual evidence.

    # ---- Path B: no strong wiki, but inbox has the raw note -> warn + answer ----
    if strong_inbox:
        warning = ("⚠️  These results come from raw, un-organized inbox captures — "
                   "the structured knowledge base has no page for this query yet. "
                   "Run the organizer to promote these notes into the wiki.\n\n")
        text = _answer_with(cfg, llm, strong_inbox, question, history_block, all_workspaces)
        response = text.get("response") if text else None
        if response and not _is_refusal(response):
            return Answer(
                text=warning + response.strip(),
                sources=[_source_tag(h, all_workspaces) for h in strong_inbox],
                grounded=True, fallback=True, fallback_kind="inbox",
            )

    # ---- Path C: weak wiki hits exist -> still try, but warn ----
    if not strong_wiki and wiki_hits:
        warning = ("⚠️  No high-confidence matches were found. The following is based "
                   "on loosely related pages and may be incomplete.\n\n")
        text = _answer_with(cfg, llm, wiki_hits, question, history_block, all_workspaces)
        response = text.get("response") if text else None
        if response and not _is_refusal(response):
            return Answer(text=warning + response.strip(),
                          sources=[_source_tag(h, all_workspaces) for h in wiki_hits],
                          grounded=True, fallback=True, fallback_kind="weak_wiki")

    # ---- Path D: absolute last resort - inbox hits too weak for Path B ----
    if weak_inbox:
        warning = ("⚠️  Only weak, un-organized inbox matches were found for this "
                   "question — treat the following as a rough lead, not a confident "
                   "answer. Run the organizer, or capture more detail on this topic, "
                   "for better results next time.\n\n")
        text = _answer_with(cfg, llm, weak_inbox, question, history_block, all_workspaces)
        response = text.get("response") if text else None
        if response and not _is_refusal(response):
            return Answer(
                text=warning + response.strip(),
                sources=[_source_tag(h, all_workspaces) for h in weak_inbox],
                grounded=True, fallback=True, fallback_kind="weak_inbox",
            )

    return Answer(text=NO_EVIDENCE_MSG, sources=[], grounded=False, fallback=False, fallback_kind="")