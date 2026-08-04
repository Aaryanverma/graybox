"""
Retrieval Agent.
...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import re
from typing import Any

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


NO_EVIDENCE_MSG = (
    "I don't have enough information in the knowledge base to answer that. "
    "Try capturing more notes about this topic, or rephrase the question."
)

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

    workspace_id, workspace_name = _get_workspace_meta(hit)
    workspace = workspace_name or workspace_id

    # Safe fallback: do not invent provenance if we do not have it.
    return f"{workspace}:{search_id}" if workspace else search_id


def _build_history_block(
    history: list[ConversationTurn],
    llm: AIService | None,
) -> str:
    lines = []
    for turn in history:
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

    except:
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
    ctx = workspace_context_block(cfg)
    if ctx:
        return f"{RETRIEVAL_SYSTEM}\n\nWorkspace context:\n{ctx}"
    return RETRIEVAL_SYSTEM


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


def _expand_graph(cfg, hits, *, max_hops=1, max_nodes=15, decay=0.65):
    from graybox.storage import read_page

    wiki_hits = [h for h in hits if h.doc.source_kind == "wiki"]
    if not wiki_hits:
        return hits

    seen = {h.doc.search_id for h in wiki_hits}
    score_by_ref = {h.doc.search_id: h.score for h in wiki_hits}
    title_by_ref = {h.doc.search_id: h.doc.page.title for h in wiki_hits}
    frontier = list(seen)
    expanded: list[Hit] = []

    for hop in range(max_hops):
        if len(seen) >= max_nodes:
            break
        next_frontier = []
        for ref in frontier:
            if len(seen) >= max_nodes:
                break
            page_type, slug = ref.split("/", 1)
            page = read_page(cfg, page_type, slug)
            if not page:
                continue
            neighbors = set(page.related) | set(page.backlinks)
            for n_ref in neighbors:
                if n_ref in seen or len(seen) >= max_nodes:
                    continue
                n_type, n_slug = n_ref.split("/", 1)
                n_page = read_page(cfg, n_type, n_slug)
                if not n_page:
                    continue
                base = score_by_ref.get(ref, 0.5)
                new_score = round(base * (decay ** (hop + 1)), 4)
                expanded.append(
                    Hit(
                        doc=_PageDoc(n_page),
                        score=new_score,
                        linked_from=title_by_ref.get(ref, ref),
                    )
                )
                score_by_ref[n_ref] = new_score
                title_by_ref[n_ref] = n_page.title
                seen.add(n_ref)
                next_frontier.append(n_ref)
        frontier = next_frontier

    out = hits + expanded
    out.sort(key=lambda h: h.score, reverse=True)
    return out


def _prepare_search_query(
    llm: AIService,
    question: str,
    history: list[ConversationTurn],
) -> str:
    if not history:
        return question

    history_block = _build_history_block(history, llm)
    prompt = QUERY_REPHRASE_PROMPT.format(history=history_block, question=question)
    text = llm.llm_call(system_prompt=QUERY_REPHRASE_SYSTEM_PROMPT_TMPL, prompt=prompt)
    if text["response"] is None:
        logger.warning(
            "Query rewriter returned no response; falling back to original question"
        )
        return question
    return text["response"].strip()


def ask(
    cfg: Config,
    llm: AIService,
    question: str,
    all_workspaces: bool = False,
    history: list[ConversationTurn] | None = None,
) -> Answer:
    history = history or []

    search_query = _prepare_search_query(
        llm,
        question,
        history,
    )

    wiki_hits, inbox_hits = search_all(
        cfg, search_query, top_k=cfg.retrieval.top_k, all_workspaces=all_workspaces
    )

    if getattr(cfg.embeddings, "enabled", False):
        try:
            emb_result = llm.embedding_call(search_query)
            if emb_result and emb_result.get("embedding"):
                semantic_refs = search_embeddings(
                    cfg,
                    emb_result["embedding"],
                    top_k=cfg.retrieval.top_k,
                    min_score=cfg.retrieval.min_score,
                )
                if semantic_refs:
                    wiki_hits = _blend_hits(wiki_hits, semantic_refs, cfg)
            elif emb_result is None:
                logger.warning(
                    "Embedding call returned no result — check embeddings.model_name in config.yaml"
                )
        except Exception as e:
            logger.warning(
                "Semantic search failed (%s), falling back to keyword-only", e
            )

    if wiki_hits and not all_workspaces:
        # Keep only strong search matches.
        primary_hits = [h for h in wiki_hits if h.score >= cfg.retrieval.min_score]

        # Expand only from high-confidence pages.
        wiki_hits = _expand_graph(
            cfg,
            primary_hits,
            max_hops=1,
            max_nodes=cfg.retrieval.top_k * 3,
        )

        # Remove duplicate pages.
        seen = set()
        deduped = []
        for h in wiki_hits:
            if h.score < cfg.retrieval.min_score:
                continue
            ref = h.doc.search_id
            if ref in seen:
                continue
            seen.add(ref)
            deduped.append(h)

        wiki_hits = deduped

    history_block = _build_history_block(history, llm)

    if wiki_hits:
        context = _build_context(wiki_hits, all_workspaces=all_workspaces)

        max_tokens = cfg.llm.max_tokens or cfg.llm.max_completion_tokens or 8196
        context = compress_context(
            context, llm, max_tokens=max_tokens, prompt=RETRIEVAL_COMPRESSION_PROMPT
        )

        if history_block:
            prompt = CHAT_RETRIEVAL_PROMPT_TMPL.format(
                history=history_block, context=context, question=question
            )
        else:
            prompt = RETRIEVAL_PROMPT_TMPL.format(context=context, question=question)

        text = llm.llm_call(system_prompt=_system_prompt(cfg), prompt=prompt)
        if text["response"] is not None:
            return Answer(
                text=text["response"].strip(),
                sources=[_source_tag(h, all_workspaces) for h in wiki_hits],
                grounded=True,
                fallback=False,
            )

    if inbox_hits:
        context = _build_context(inbox_hits, all_workspaces=all_workspaces)

        max_tokens = cfg.llm.max_tokens or cfg.llm.max_completion_tokens or 8196
        context = compress_context(
            context, llm, max_tokens=max_tokens, prompt=RETRIEVAL_COMPRESSION_PROMPT
        )

        if history_block:
            prompt = CHAT_RETRIEVAL_PROMPT_TMPL.format(
                history=history_block, context=context, question=question
            )
        else:
            prompt = RETRIEVAL_PROMPT_TMPL.format(context=context, question=question)

        text = llm.llm_call(system_prompt=_system_prompt(cfg), prompt=prompt)
        if not text or not text.get("response"):
            return Answer(text=NO_EVIDENCE_MSG, sources=[], grounded=False, fallback=False)

        return Answer(
            text=text["response"].strip(),
            sources=[_source_tag(h, all_workspaces) for h in inbox_hits],
            grounded=True,
            fallback=True,
        )

    return Answer(text=NO_EVIDENCE_MSG, sources=[], grounded=False, fallback=False)