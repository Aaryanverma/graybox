"""
Retrieval Agent.
...
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from graybox.config import Config
from graybox.ai import AIService
from graybox.prompts import (
    RETRIEVAL_PROMPT_TMPL, RETRIEVAL_SYSTEM,
    CHAT_RETRIEVAL_PROMPT_TMPL,
)
from graybox.search import search_all
from graybox.search_engine import Hit, _PageDoc
from graybox.workspace import workspace_context_block
from graybox.embedding_index import ensure_indexed, search_embeddings

logger = logging.getLogger(__name__)


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


def _source_tag(hit: Hit, all_workspaces: bool = False) -> str:
    if hit.workspace_id and all_workspaces:
        return f"{hit.workspace_id}/{hit.doc.search_id}"
    return hit.doc.search_id


def _search_query_with_context(question: str, history: list[ConversationTurn]) -> str:
    """Fold the immediately preceding question into the search query so
    pronoun/ellipsis-style follow-ups ("when's it due", "what about him")
    still retrieve the right pages - without full coreference resolution.
    The ORIGINAL question (not this augmented one) is still what gets asked
    of the LLM and shown to the user; this only widens what search matches on.
    """
    if not history:
        return question
    return f"{history[-1].question} {question}"


def _build_history_block(history: list[ConversationTurn]) -> str:
    lines = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        lines.append(f"User: {turn.question}")
        lines.append(f"Assistant: {turn.answer}")
    return "\n".join(lines)


def _build_context(hits: list[Hit], all_workspaces: bool = False) -> str:
    blocks = []
    for h in hits:
        if h.doc.source_kind == "wiki":
            p = h.doc.page
            note_text = "\n".join(p.notes) if p.notes else p.summary
            tag = _source_tag(h, all_workspaces)
            provenance = f" (linked from {h.linked_from})" if h.linked_from else ""
            blocks.append(f"[{tag}]{provenance} {p.title}\n{p.summary}\n{note_text}".strip())
        else:
            blocks.append(f"[{_source_tag(h, all_workspaces)}] (raw capture)\n{h.doc.item.content}".strip())
    return "\n\n---\n\n".join(blocks)


def _system_prompt(cfg: Config) -> str:
    ctx = workspace_context_block(cfg)
    if ctx:
        return f"{RETRIEVAL_SYSTEM}\n\nWorkspace context:\n{ctx}"
    return RETRIEVAL_SYSTEM


def _blend_hits(
    keyword_hits: list[Hit],
    semantic_refs: list[tuple[str, float]],
    cfg: Config,
) -> list[Hit]:
    from graybox.storage import read_page

    keyword_by_ref: dict[str, Hit] = {h.doc.search_id: h for h in keyword_hits}
    semantic_by_ref: dict[str, float] = {ref: score for ref, score in semantic_refs}

    all_refs = set(keyword_by_ref) | set(semantic_by_ref)
    blended: list[Hit] = []

    for ref in all_refs:
        kw_hit = keyword_by_ref.get(ref)
        sem_score = semantic_by_ref.get(ref, 0.0)

        if kw_hit and sem_score > 0:
            blended_score = 0.6 * kw_hit.score + 0.4 * sem_score
            blended.append(Hit(doc=kw_hit.doc, score=round(blended_score, 4), workspace_id=kw_hit.workspace_id, workspace_name=kw_hit.workspace_name))
        elif kw_hit:
            blended.append(kw_hit)
        else:
            page_type, slug = ref.split("/", 1)
            page = read_page(cfg, page_type, slug)
            if page:
                adjusted = round(sem_score * 0.85, 4)
                blended.append(Hit(doc=_PageDoc(page), score=adjusted, workspace_id="", workspace_name=""))

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
                expanded.append(Hit(doc=_PageDoc(n_page), score=new_score, linked_from=title_by_ref.get(ref, ref)))
                score_by_ref[n_ref] = new_score
                title_by_ref[n_ref] = n_page.title
                seen.add(n_ref)
                next_frontier.append(n_ref)
        frontier = next_frontier

    out = hits + expanded
    out.sort(key=lambda h: h.score, reverse=True)
    return out


def ask(
    cfg: Config,
    llm: AIService,
    question: str,
    all_workspaces: bool = False,
    history: list[ConversationTurn] | None = None,
) -> Answer:
    history = history or []
    search_query = _search_query_with_context(question, history)

    wiki_hits, inbox_hits = search_all(cfg, search_query, top_k=cfg.retrieval.top_k, all_workspaces=all_workspaces)

    if getattr(cfg.embeddings, "enabled", False):
        try:
            emb_result = llm.embedding_call(search_query)
            if emb_result and emb_result.get("embedding"):
                semantic_refs = search_embeddings(
                    cfg, emb_result["embedding"],
                    top_k=cfg.retrieval.top_k,
                    min_score=cfg.retrieval.min_score,
                )
                if semantic_refs:
                    wiki_hits = _blend_hits(wiki_hits, semantic_refs, cfg)
            elif emb_result is None:
                logger.warning("Embedding call returned no result — check embeddings.model_name in config.yaml")
        except Exception as e:
            logger.warning("Semantic search failed (%s), falling back to keyword-only", e)

    if wiki_hits and not all_workspaces:
        wiki_hits = _expand_graph(cfg, wiki_hits, max_hops=1, max_nodes=cfg.retrieval.top_k * 3)
        wiki_hits = [h for h in wiki_hits if h.score >= cfg.retrieval.min_score]

    history_block = _build_history_block(history)

    if wiki_hits:
        context = _build_context(wiki_hits, all_workspaces=all_workspaces)
        if history_block:
            prompt = CHAT_RETRIEVAL_PROMPT_TMPL.format(history=history_block, context=context, question=question)
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
        if history_block:
            prompt = CHAT_RETRIEVAL_PROMPT_TMPL.format(history=history_block, context=context, question=question)
        else:
            prompt = RETRIEVAL_PROMPT_TMPL.format(context=context, question=question)
        text = llm.llm_call(system_prompt=_system_prompt(cfg), prompt=prompt)
        if text["response"] is None:
            return Answer(text=NO_EVIDENCE_MSG, sources=[], grounded=False, fallback=False)

        return Answer(
            text=text["response"].strip(),
            sources=[_source_tag(h, all_workspaces) for h in inbox_hits],
            grounded=True,
            fallback=True,
        )

    return Answer(text=NO_EVIDENCE_MSG, sources=[], grounded=False, fallback=False)