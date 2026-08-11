"""
Organizer Agent.

Reads unprocessed inbox items, asks the LLM to extract entities/relations/tasks/
decisions as strict JSON, then deterministically (no LLM involved here)
creates or merges wiki pages, maintains bidirectional backlinks, and records
provenance.

Workspace context is injected into the system prompt so extraction is aware of
the currently active knowledge silo.
"""

from __future__ import annotations

import json
import re

from graybox.config import Config
from graybox.ai import AIService
from graybox.models import Page, TYPE_DIR, now_iso
from graybox.prompts import ORGANIZER_PROMPT_TMPL, ORGANIZER_SYSTEM
from graybox.storage import (
    find_page_by_name,
    list_pages,
    list_unprocessed,
    mark_processed,
    slugify,
    write_page,
)
from graybox.workspace import workspace_context_block
from graybox.summarizer import refresh_page_summary
from graybox.embedding_index import ensure_indexed
from graybox.search_engine import _PageDoc, Engine, Query, Hit
from graybox.search import search_all


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json(llm_output: str) -> dict:
    cleaned = _strip_code_fence(llm_output)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def _get_or_create_page(
    cfg: Config, page_type: str, name: str, aliases: list[str], summary: str
) -> tuple[Page, bool]:
    existing = find_page_by_name(cfg, page_type, name)
    if existing:
        return existing, False

    # Universal fuzzy match via the same engine everyone else uses.
    # Uses dedup_threshold (entity-identity similarity), not min_score
    # (search relevance) - the two measure different things even though
    # both are normalized 0-1 "closer to 1 = more similar" scores.
    # Take the BEST match above threshold, not just the first one found,
    # so results don't depend on directory listing order.
    q = Query.parse(name)
    best_page: Page | None = None
    best_score = 0.0
    for page in list_pages(cfg, page_type):
        score = Engine.name_scorer(q, _PageDoc(page))
        if score >= cfg.retrieval.dedup_threshold and score > best_score:
            best_page, best_score = page, score

    if best_page is not None:
        if name.lower() != best_page.title.lower() and name not in best_page.aliases:
            best_page.aliases.append(name)
        return best_page, False

    slug = slugify(name)
    return (
        Page(
            id=slug,
            type=page_type,
            title=name,
            created=now_iso(),
            updated=now_iso(),
            aliases=list(dict.fromkeys(aliases)),
            summary=summary,
        ),
        True,
    )


def _merge_unique(target: list[str], values: list[str]) -> None:
    seen = set(target)
    for raw in values or []:
        value = (raw or "").strip()
        if value and value not in seen:
            target.append(value)
            seen.add(value)


def _append_note(page: Page, text: str, source_id: str, raw: str = "") -> None:
    line = f"- ({now_iso()}) {text.strip()} _(source: inbox/{source_id})_"
    if raw and raw.strip() and raw.strip() not in text:
        line += f"\n  > {raw.strip()}"
    page.notes.append(line)
    if source_id not in page.sources:
        page.sources.append(source_id)
    page.updated = now_iso()
    if not page.summary:
        page.summary = text.strip()


def _link(a: Page, b_ref: str) -> None:
    if b_ref not in a.related and b_ref != a.ref:
        a.related.append(b_ref)


def _backlink(a: Page, from_ref: str) -> None:
    if from_ref not in a.backlinks and from_ref != a.ref:
        a.backlinks.append(from_ref)


def _system_prompt(cfg: Config) -> str:
    ctx = workspace_context_block(cfg)
    if ctx:
        return f"{ORGANIZER_SYSTEM}\n\nWorkspace context:\n{ctx}"
    return ORGANIZER_SYSTEM


def _format_existing_context(hits: list[Hit]) -> str:
    """Render retrieved existing pages (of ANY type) as compact reference
    lines the LLM can match a note against - enough to identify 'this note
    updates that page' without blowing up prompt size.
    """
    if not hits:
        return "(none found)"
    lines: list[str] = []
    for h in hits:
        p = h.doc.page
        parts = [f'- {p.ref} [{p.type}] "{p.title}"']
        if p.status:
            parts.append(f"status={p.status}")
        if p.owner:
            parts.append(f"owner={p.owner}")
        if p.due:
            parts.append(f"due={p.due}")
        if p.date:
            parts.append(f"date={p.date}")
        line = " ".join(parts)
        if p.summary:
            line += f"\n  summary: {p.summary.strip()}"
        lines.append(line)
    return "\n".join(lines)


def _gather_existing_context(cfg: Config, item_content: str, top_k: int = 10) -> str:
    """Retrieve existing wiki pages of ANY type related to this note BEFORE
    extraction, so the organizer can reconcile state changes against what
    already exists - regardless of whether the existing page is a task,
    project, decision, meeting, or any other entity type - instead of
    blindly creating disconnected duplicates with no memory of each other.

    Uses a lower, fixed min_score than normal retrieval (0.3 vs. the usual
    ~0.4+ answer-grounding floor) since this is candidate context for the
    LLM to reconcile against, not a final grounded answer - false
    positives here just mean an extra reference line, not a wrong claim.
    """
    wiki_hits, _ = search_all(cfg, item_content, top_k=top_k, wiki_min_score=0.3)
    return _format_existing_context(wiki_hits)


def process_item(
    cfg: Config,
    llm: AIService,
    item_id: str,
    item_content: str,
    dry_run: bool = False,
    item_extra: dict | None = None,
) -> list[str]:
    existing_context = _gather_existing_context(cfg, item_content)
    prompt = ORGANIZER_PROMPT_TMPL.format(
        note=item_content, existing_context=existing_context
    )
    raw = llm.llm_call(system_prompt=_system_prompt(cfg), prompt=prompt)
    if raw["response"] is None:
        raise RuntimeError(
            "LLM call failed — check logs/config for the underlying API error."
        )
    data = _extract_json(raw["response"])

    touched: dict[str, Page] = {}
    is_new: dict[str, bool] = {}

    def touch(page: Page, new: bool = False) -> Page:
        touched[page.ref] = page
        is_new.setdefault(page.ref, new)
        if item_extra:
            page.extra.update(item_extra)
        return page

    name_to_page: dict[str, Page] = {}

    for ent in data.get("entities", []):
        etype = ent.get("type", "topic")
        if etype not in TYPE_DIR:
            etype = "topic"

        name = ent.get("name", "").strip()
        if not name:
            continue

        page, new = _get_or_create_page(
            cfg,
            etype,
            name,
            ent.get("aliases", []) or [],
            "",
        )

        # Merge current-state fields that the Page model actually stores.
        if etype in ("project", "action") and ent.get("status"):
            page.status = ent["status"].strip()

        if etype == "meeting":
            if ent.get("date"):
                page.date = ent["date"].strip()
            _merge_unique(page.attendees, ent.get("attendees", []) or [])

        if etype == "action":
            if ent.get("owner"):
                page.owner = ent["owner"].strip()
            if ent.get("due"):
                page.due = ent["due"].strip()

        _merge_unique(page.aliases, ent.get("aliases", []) or [])
        _merge_unique(page.tags, ent.get("tags", []) or [])

        _append_note(
            page,
            ent.get("summary", "") or "Mentioned in note.",
            item_id,
            raw=item_content,
        )
        touch(page, new)
        name_to_page[name.lower()] = page

    for rel in data.get("relations", []):
        a_name, b_name = rel.get("a", "").strip(), rel.get("b", "").strip()
        note = rel.get("note", "")
        page_a = name_to_page.get(a_name.lower())
        page_b = name_to_page.get(b_name.lower())
        if page_a and page_b:
            _link(page_a, page_b.ref)
            _link(page_b, page_a.ref)
            _backlink(page_a, page_b.ref)
            _backlink(page_b, page_a.ref)
            if note:
                _append_note(page_a, f"Related to {page_b.title}: {note}", item_id)
                _append_note(page_b, f"Related to {page_a.title}: {note}", item_id)
        touch(page_a) if page_a else None
        touch(page_b) if page_b else None

    for task in data.get("tasks", []):
        title = task.get("title", "").strip()
        if not title:
            continue
        page, new = _get_or_create_page(cfg, "task", title, [], "")
        page.status = (task.get("status", "open") or "open").replace("_", "-").strip()
        owner = task.get("owner", "")
        due = task.get("due", "")
        if owner:
            page.owner = owner
        if due:
            page.due = due
        detail = f"Task: {title}"
        if owner:
            detail += f" (owner: {owner})"
        if due:
            detail += f" (due: {due})"
        _append_note(page, detail, item_id, raw=item_content)
        touch(page, new)
        if owner and owner.lower() in name_to_page:
            owner_page = name_to_page[owner.lower()]
            _link(page, owner_page.ref)
            _backlink(owner_page, page.ref)
            touch(owner_page)

    for act in data.get("actions", []):
        title = act.get("title", "").strip()
        if not title:
            continue
        page, new = _get_or_create_page(cfg, "action", title, [], "")
        page.status = (act.get("status", "open") or "open").replace("_", "-").strip()
        owner = act.get("owner", "")
        due = act.get("due", "")
        if owner:
            page.owner = owner
        if due:
            page.due = due
        detail = f"Action: {title}"
        if owner:
            detail += f" (owner: {owner})"
        if due:
            detail += f" (due: {due})"
        _append_note(page, detail, item_id, raw=item_content)
        touch(page, new)
        if owner and owner.lower() in name_to_page:
            owner_page = name_to_page[owner.lower()]
            _link(page, owner_page.ref)
            _backlink(owner_page, page.ref)
            touch(owner_page)

    for dec in data.get("decisions", []):
        title = dec.get("title", "").strip()
        if not title:
            continue
        page, new = _get_or_create_page(cfg, "decision", title, [], "")
        page.status = "decided"
        decided_by = dec.get("decided_by", "")
        detail = dec.get("description", "") or title
        if decided_by:
            detail += f" (decided by: {decided_by})"
        _append_note(page, detail, item_id, raw=item_content)
        touch(page, new)
        if decided_by and decided_by.lower() in name_to_page:
            person_page = name_to_page[decided_by.lower()]
            _link(page, person_page.ref)
            _backlink(person_page, page.ref)
            touch(person_page)

    meeting_pages: list[Page] = []
    for meet in data.get("meetings", []):
        title = meet.get("title", "").strip()
        if not title:
            continue
        page, new = _get_or_create_page(cfg, "meeting", title, [], "")
        if meet.get("date"):
            page.date = meet["date"]
        for a in meet.get("attendees", []) or []:
            a = a.strip()
            if a and a not in page.attendees:
                page.attendees.append(a)
        agenda = meet.get("agenda", "") or f"Meeting: {title}"
        _append_note(page, agenda, item_id, raw=item_content)
        touch(page, new)
        meeting_pages.append(page)

        for a in meet.get("attendees", []) or []:
            person_page = name_to_page.get(a.strip().lower())
            if person_page:
                _link(page, person_page.ref)
                _backlink(person_page, page.ref)
                touch(person_page)

    for meeting_page in meeting_pages:
        for ref, other in list(touched.items()):
            if other is meeting_page or other.type == "meeting":
                continue
            _link(meeting_page, other.ref)
            _backlink(other, meeting_page.ref)

    for evt in data.get("events", []):
        title = evt.get("title", "").strip()
        if not title:
            continue
        page, new = _get_or_create_page(cfg, "event", title, [], "")
        if evt.get("date"):
            page.date = evt["date"]

        description = evt.get("description", "") or f"Event: {title}"
        location = evt.get("location", "")
        detail = description
        if location:
            detail += f" (location: {location})"

        _append_note(page, detail, item_id, raw=item_content)
        touch(page, new)

    if dry_run:
        return [f"{ref} ({'new' if is_new.get(ref) else 'updated'})" for ref in touched]

    for page in touched.values():
        write_page(cfg, page)
        ensure_indexed(cfg, page, llm)

    return list(touched.keys())


def organize_all(cfg: Config, llm: AIService, dry_run: bool = False) -> dict:
    report = {"processed": [], "errors": [], "summaries_refreshed": []}
    touched_refs: set[str] = set()

    for item in list_unprocessed(cfg):
        try:
            refs = process_item(
                cfg, llm, item.id, item.content, dry_run=dry_run, item_extra=item.extra
            )
            if not dry_run:
                mark_processed(cfg, item.id, refs)
            report["processed"].append({"item": item.id, "pages": refs})
            touched_refs.update(refs)
        except Exception as e:  # noqa: BLE001
            report["errors"].append({"item": item.id, "error": str(e)})

    # Auto-refresh summaries for pages that were touched and have enough notes
    if not dry_run and touched_refs and getattr(cfg, "auto_refresh_summaries", True):
        from graybox.storage import read_page

        for ref in touched_refs:
            try:
                page_type, slug = ref.split("/", 1)
                page = read_page(cfg, page_type, slug)
                if page and len(page.notes) >= 3:
                    refresh_report = refresh_page_summary(cfg, llm, page, dry_run=False)
                    if refresh_report:
                        report["summaries_refreshed"].append(ref)
            except Exception:
                pass  # Don't fail organize if summarization fails

    return report
