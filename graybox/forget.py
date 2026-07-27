
"""
Forget Agent.

The Capture Agent's inbox is immutable by design - but "immutable" and
"can never be corrected" are not the same thing. This lets a person retract
a bad capture (typo'd note, accidental paste, something sensitive) with
three levels of severity:

  soft forget (default) - the raw file stays on disk untouched, but a
      tombstone in .state/forgotten.json excludes it from every read path
      (list_inbox_items, list_unprocessed, search_inbox, and therefore
      `organize`). Nothing is destroyed; it can be a simple accountability
      trail of "this was captured, then retracted, and why."
  --purge - additionally deletes the raw inbox file. Irreversible; for
      genuinely sensitive captures that shouldn't sit on disk at all.
  --scrub - if the item was already run through `organize`, also strips
      the notes it produced out of whatever wiki pages they landed in,
      deterministically (regex match on the exact "(source: inbox/<id>)"
      tag `_append_note` writes) - no LLM involved.
"""
from __future__ import annotations

import re
from pathlib import Path

from graybox.config import Config
from graybox.models import now_iso
from graybox.storage import list_pages, load_processed, mark_forgotten, read_inbox_item, write_page

_SOURCE_NOTE_RE_TMPL = r"^- \([^)]*\).*_\(source: inbox/{id}\)_(?:\n  >.*)?$"


def forget_item(
    cfg: Config, item_id: str, purge: bool = False, scrub: bool = False, reason: str = "",
) -> dict:
    item = read_inbox_item(cfg, item_id)
    if item is None:
        raise ValueError(f"No such inbox item: {item_id}")

    processed = load_processed(cfg)
    already_processed = item_id in processed
    touched_pages = processed.get(item_id, {}).get("pages", [])

    report = {
        "item_id": item_id,
        "purged": False,
        "already_processed": already_processed,
        "touched_pages": touched_pages,
        "scrubbed_pages": [],
    }

    mark_forgotten(cfg, item_id, reason=reason)

    if purge:
        path = Path(item.path)
        if path.exists():
            path.unlink()
        report["purged"] = True

    if scrub and already_processed:
        pattern = re.compile(_SOURCE_NOTE_RE_TMPL.format(id=re.escape(item_id)), re.MULTILINE)
        for page in list_pages(cfg):
            if item_id not in page.sources:
                continue
            new_notes = [n for n in page.notes if not pattern.match(n)]
            if len(new_notes) == len(page.notes):
                continue
            page.notes = new_notes
            page.sources = [s for s in page.sources if s != item_id]
            page.updated = now_iso()
            write_page(cfg, page)
            report["scrubbed_pages"].append(page.ref)

    return report
