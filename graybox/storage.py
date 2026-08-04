"""
Storage layer. Everything is plain files on disk.

Each workspace is isolated under:

workspace/
  inbox/<id>.md
  wiki/<type_dir>/<slug>.md
  .state/processed.json
  attachments/
  metadata.json
  wiki.db / graph.db / vectors.db (future-ready placeholders)

No database is used yet for the wiki content itself; the DB files are
workspace-scoped placeholders so future indexes can slot in cleanly.
"""
from __future__ import annotations

import json
import re
import secrets
from pathlib import Path
import yaml

from graybox.config import Config
from graybox.history_tracker import _maybe_record
from graybox.index import invalidate_page, invalidate_inbox
from graybox.models import TYPE_DIR, InboxItem, Page, now_iso, now_id_ts

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)

# A note entry starts with "- (<timestamp>)". Some entries also carry a
# continuation line (an indented "> quoted source text" block). Splitting
# naively on "\n" breaks that continuation into its own bogus entry on the
# next read - so instead we only split immediately *before* a new bullet.
NOTE_ENTRY_RE = re.compile(r"\n(?=- )")


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "untitled"


def ensure_workspace(cfg: Config) -> None:
    ws = cfg.workspace_manager.ensure_workspace()
    # Keep legacy properties valid too.
    ws.inbox_dir.mkdir(parents=True, exist_ok=True)
    ws.wiki_dir.mkdir(parents=True, exist_ok=True)
    ws.state_dir.mkdir(parents=True, exist_ok=True)
    ws.attachments_dir.mkdir(parents=True, exist_ok=True)
    ws.exports_dir.mkdir(parents=True, exist_ok=True)
    # ws.wiki_db_path.touch(exist_ok=True)
    # ws.graph_db_path.touch(exist_ok=True)
    # ws.vectors_db_path.touch(exist_ok=True)


# ---------------------------------------------------------------------------
# Inbox (append-only, immutable)
# ---------------------------------------------------------------------------

def write_inbox_item(cfg: Config, content: str) -> InboxItem:
    ensure_workspace(cfg)
    inbox_dir = cfg.inbox_dir
    item_id = f"{now_id_ts()}-{secrets.token_hex(2)}"
    body = (
        "---\n"
        + yaml.safe_dump({"id": item_id, "created": now_iso()}, sort_keys=False)
        + "---\n\n"
        + content.strip()
        + "\n"
    )
    path = inbox_dir / f"{item_id}.md"
    path.write_text(body, encoding="utf-8")
    invalidate_inbox(cfg, item_id)
    return InboxItem(id=item_id, created=now_iso(), content=content.strip(), path=str(path))

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm = yaml.safe_load(m.group(1)) or {}
    return fm, m.group(2)


def read_inbox_item(cfg: Config, item_id: str) -> InboxItem | None:
    path = cfg.inbox_dir / f"{item_id}.md"
    from graybox.index import cached_read_inbox
    return cached_read_inbox(path, cfg)


def list_inbox_items(cfg: Config, include_forgotten: bool = False) -> list[InboxItem]:
    from graybox.index import cached_list_inbox_items
    return cached_list_inbox_items(cfg, include_forgotten=include_forgotten)


# ---------------------------------------------------------------------------
# Processed-state tracking (kept OUTSIDE inbox/ so inbox stays immutable)
# ---------------------------------------------------------------------------

def _processed_path(cfg: Config) -> Path:
    return cfg.state_dir / "processed.json"


def load_processed(cfg: Config) -> dict:
    p = _processed_path(cfg)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_processed(cfg: Config, data: dict) -> None:
    ensure_workspace(cfg)
    _processed_path(cfg).write_text(json.dumps(data, indent=2), encoding="utf-8")


def mark_processed(cfg: Config, item_id: str, page_refs: list[str]) -> None:
    data = load_processed(cfg)
    data[item_id] = {"processed_at": now_iso(), "pages": page_refs}
    save_processed(cfg, data)


def list_unprocessed(cfg: Config) -> list[InboxItem]:
    processed = load_processed(cfg)
    return [i for i in list_inbox_items(cfg) if i.id not in processed]


# ---------------------------------------------------------------------------
# Forgotten-item tombstones (soft-delete for bad captures)
# ---------------------------------------------------------------------------

def _forgotten_path(cfg: Config) -> Path:
    return cfg.state_dir / "forgotten.json"


def load_forgotten(cfg: Config) -> dict:
    p = _forgotten_path(cfg)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_forgotten(cfg: Config, data: dict) -> None:
    ensure_workspace(cfg)
    _forgotten_path(cfg).write_text(json.dumps(data, indent=2), encoding="utf-8")


def mark_forgotten(cfg: Config, item_id: str, reason: str = "") -> None:
    data = load_forgotten(cfg)
    data[item_id] = {"forgotten_at": now_iso(), "reason": reason}
    save_forgotten(cfg, data)


def is_forgotten(cfg: Config, item_id: str) -> bool:
    return item_id in load_forgotten(cfg)


# ---------------------------------------------------------------------------
# Wiki pages
# ---------------------------------------------------------------------------

def page_path(cfg: Config, page_type: str, slug: str) -> Path:
    directory = TYPE_DIR.get(page_type, "topics")
    return cfg.wiki_dir / directory / f"{slug}.md"


def _render_page(page: Page) -> str:
    fm = yaml.safe_dump(page.frontmatter(), sort_keys=False).strip()
    lines = [f"---\n{fm}\n---\n", f"# {page.title}\n"]
    if page.summary:
        lines.append("## Summary\n" + page.summary.strip() + "\n")
    if page.attendees:
        lines.append("## Attendees\n" + "\n".join(f"- {a}" for a in page.attendees) + "\n")
    lines.append("## Notes\n" + ("\n".join(page.notes) if page.notes else "_No notes yet._") + "\n")
    if page.related:
        lines.append("## Related\n" + "\n".join(f"- [[{r}]]" for r in sorted(set(page.related))) + "\n")
    if page.backlinks:
        lines.append("## Backlinks\n" + "\n".join(f"- [[{r}]]" for r in sorted(set(page.backlinks))) + "\n")
    if page.sources:
        lines.append("## Sources\n" + "\n".join(f"- inbox/{s}" for s in page.sources) + "\n")
    return "\n".join(lines)


def write_page(cfg: Config, page: Page) -> Path:
    ensure_workspace(cfg)
    path = page_path(cfg, page.type, page.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_page(page), encoding="utf-8")
    page.path = str(path)
    invalidate_page(cfg, page.type, page.id)
    _maybe_record(cfg, page, f"wiki: update {page.ref}")
    return path


def read_page(cfg: Config, page_type: str, slug: str) -> Page | None:
    path = page_path(cfg, page_type, slug)
    from graybox.index import cached_read_page
    return cached_read_page(path, cfg)


def _split_notes(notes_raw: str) -> list[str]:
    """Split a rendered Notes section back into individual entries."""
    if not notes_raw or notes_raw.startswith("_No notes"):
        return []
    return [entry for entry in NOTE_ENTRY_RE.split(notes_raw) if entry.strip()]


def _parse_page(path: Path) -> Page:
    fm, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
    title_match = re.search(r"^#\s+(.*)$", body, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else fm.get("title", path.stem)

    def _section(name: str) -> str:
        m = re.search(rf"^##\s+{name}\s*\n(.*?)(?=^##\s+|\Z)", body, re.MULTILINE | re.DOTALL)
        return m.group(1).strip() if m else ""

    summary = _section("Summary")
    notes_raw = _section("Notes")
    notes = _split_notes(notes_raw)

    return Page(
        id=fm.get("id", path.stem),
        type=fm.get("type", "topic"),
        title=title,
        created=fm.get("created", now_iso()),
        updated=fm.get("updated", now_iso()),
        aliases=fm.get("aliases", []) or [],
        related=fm.get("related", []) or [],
        backlinks=fm.get("backlinks", []) or [],
        sources=fm.get("sources", []) or [],
        tags=fm.get("tags", []) or [],
        status=fm.get("status", "") or "",
        summary=summary,
        notes=notes,
        attendees=fm.get("attendees", []) or [],
        date=fm.get("date", "") or "",
        owner=fm.get("owner", "") or "",
        due=fm.get("due", "") or "",
        extra=fm.get("extra", {}) or {},
        summary_refreshed_at=fm.get("summary_refreshed_at", "") or "",
        path=str(path),
    )


def list_pages(cfg: Config, page_type: str | None = None) -> list[Page]:
    from graybox.index import cached_list_pages
    return cached_list_pages(cfg, page_type=page_type)


def find_page_by_name(cfg: Config, page_type: str, name: str) -> Page | None:
    """Match by slug first, then by alias, on an exact or near-exact basis."""
    slug = slugify(name)
    exact = read_page(cfg, page_type, slug)
    if exact:
        return exact
    for page in list_pages(cfg, page_type):
        if slug == page.id or name.lower() in [a.lower() for a in page.aliases]:
            return page
    return None


def _replace_ref(refs: list[str], old_ref: str, new_ref: str | None, self_ref: str) -> list[str]:
    out = []
    for r in refs:
        if r == old_ref:
            if new_ref is not None:
                out.append(new_ref)
        else:
            out.append(r)
    return [r for r in dict.fromkeys(out) if r != self_ref]


def _replace_wiki_link(match: re.Match, new_ref: str | None) -> str:
    """[[old_ref|Display]] → [[new_ref]]  or  just Display/old_ref on deletion."""
    inner = match.group(0)[2:-2]          # strip [[ and ]]
    display = inner.split('|', 1)[1] if '|' in inner else inner
    if new_ref:
        return f"[[{new_ref}]]"
    return display


def rewire_references(
    cfg: Config, old_ref: str, new_ref: str | None, exclude_refs: set[str] = frozenset()
) -> list[str]:
    """Replace every occurrence of `old_ref` in other pages' related/backlinks AND inline [[links]]."""
    changed: list[str] = []
    wiki_link_re = re.compile(rf"\[\[{re.escape(old_ref)}(?:\|[^\]]*)?\]\]")

    def repl(m: re.Match) -> str:
        return _replace_wiki_link(m, new_ref)

    for page in list_pages(cfg):
        if page.ref in exclude_refs or page.ref == old_ref:
            continue
        touched = False

        # Frontmatter lists
        if old_ref in page.related:
            page.related = _replace_ref(page.related, old_ref, new_ref, page.ref)
            touched = True
        if old_ref in page.backlinks:
            page.backlinks = _replace_ref(page.backlinks, old_ref, new_ref, page.ref)
            touched = True

        # Inline wiki links inside notes
        new_notes = []
        notes_touched = False
        for note in page.notes:
            new_note = wiki_link_re.sub(repl, note)
            if new_note != note:
                notes_touched = True
            new_notes.append(new_note)
        if notes_touched:
            page.notes = new_notes
            touched = True

        # Inline wiki links inside summary
        if page.summary:
            new_summary = wiki_link_re.sub(repl, page.summary)
            if new_summary != page.summary:
                page.summary = new_summary
                touched = True

        if touched:
            page.updated = now_iso()
            write_page(cfg, page)
            changed.append(page.ref)
    return changed