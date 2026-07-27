"""
Summarizer Agent.

Refreshes stale page summaries by re-synthesizing all notes into a concise,
current summary. The original summary (from first extraction) often becomes
outdated as a page accumulates notes over months. This is the fix.

- Never invents facts not present in the notes.
- Preserves the original summary in page history via git_tracker.
- Can be run explicitly (CLI) or automatically after organize.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from graybox.ai import AIService
from graybox.config import Config
from graybox.models import Page
from graybox.prompts import SUMMARIZER_SYSTEM, SUMMARIZER_PROMPT_TMPL
from graybox.storage import list_pages, read_page, write_page
from graybox.history_tracker import _maybe_record
from graybox.models import now_iso

logger = logging.getLogger(__name__)


@dataclass
class RefreshReport:
    ref: str
    old_summary: str
    new_summary: str
    notes_considered: int
    cost: float


def _build_prompt(page: Page) -> str:
    notes_text = "\n".join(page.notes) if page.notes else "(no notes)"
    return SUMMARIZER_PROMPT_TMPL.format(
        title=page.title,
        current_summary=page.summary or "(none)",
        notes=notes_text,
    )


def refresh_page_summary(
    cfg: Config,
    llm: AIService,
    page: Page,
    dry_run: bool = False,
) -> RefreshReport | None:
    """Re-summarize a single page from its notes. Returns None if skipped."""
    if not page.notes:
        logger.debug("Skipping %s: no notes to summarize", page.ref)
        return None

    # Skip if summary is already fresh (heuristic: summary was written after
    # the most recent note, or page has < 3 notes and already has a summary)
    if page.summary and len(page.notes) < 3:
        logger.debug("Skipping %s: few notes and summary exists", page.ref)
        return None

    prompt = _build_prompt(page)
    raw = llm.llm_call(system_prompt=SUMMARIZER_SYSTEM, prompt=prompt)
    if raw.get("response") is None:
        logger.warning("LLM failed for %s summary refresh", page.ref)
        return None

    new_summary = raw["response"].strip()
    if not new_summary:
        logger.debug("Summary empty for %s", page.ref)
        return None

    report = RefreshReport(
        ref=page.ref,
        old_summary=page.summary,
        new_summary=new_summary,
        notes_considered=len(page.notes),
        cost=float(raw.get("cost", 0.0)),
    )

    if dry_run:
        return report

    page.summary = new_summary
    page.summary_refreshed_at = now_iso()
    page.updated = now_iso()
    write_page(cfg, page)
    _maybe_record(cfg, page, f"wiki: refresh summary for {page.ref}")
    return report


def refresh_all_summaries(
    cfg: Config,
    llm: AIService,
    *,
    page_type: str | None = None,
    dry_run: bool = False,
    min_notes: int = 3,
) -> dict:
    """Refresh summaries for all pages matching criteria.

    Returns {"refreshed": [...], "skipped": int, "errors": [...], "total_cost": float}
    """
    pages = list_pages(cfg)
    refreshed: list[RefreshReport] = []
    skipped = 0
    errors: list[dict] = []
    total_cost = 0.0

    for page in pages:
        if page_type is not None and page.type != page_type:
            skipped += 1
            continue
        if len(page.notes) < min_notes:
            skipped += 1
            continue
        try:
            report = refresh_page_summary(cfg, llm, page, dry_run=dry_run)
            if report:
                refreshed.append(report)
                total_cost += report.cost
            else:
                skipped += 1
        except Exception as e:
            logger.exception("Error refreshing %s: %s", page.ref, e)
            errors.append({"ref": page.ref, "error": str(e)})

    return {
        "refreshed": refreshed,
        "skipped": skipped,
        "errors": errors,
        "total_cost": total_cost,
    }