"""
Obsidian Vault Migration.

One-time, explicitly-invoked import of an existing Obsidian vault into
Gray Box's wiki/ format. NOT a sync feature - re-running against a vault
whose Gray Box pages have since been edited (via curate.py) or accumulated
organizer notes risks clobbering that state.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import yaml

from graybox.config import Config
from graybox.ai import AIService
from graybox.models import Page, TYPE_DIR, now_iso
from graybox.search_engine import Engine, Query, _PageDoc
from graybox.storage import (
    find_page_by_name,
    list_pages,
    slugify,
    write_inbox_item,
    write_page,
)
from graybox.prompts import (
    MIGRATE_CLASSIFY_PROMPT_TMPL,
    MIGRATE_CLASSIFY_SYSTEM,
)
from graybox.models import VaultNote, MigrationReport, MigratedPage

logger = logging.getLogger(__name__)

WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]")


# ---------------------------------------------------------------------------
# Pass 1: parse vault, build title -> slug map (no LLM involved)
# ---------------------------------------------------------------------------


def _parse_note(path: Path) -> VaultNote | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        logger.warning("Skipping unreadable note %s: %s", path, e)
        return None

    frontmatter: dict = {}
    body = text
    if text.startswith("---"):
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.DOTALL)
        if m:
            try:
                frontmatter = yaml.safe_load(m.group(1)) or {}
            except yaml.YAMLError:
                frontmatter = {}
            body = m.group(2)

    title = frontmatter.get("title") or path.stem
    links = [m.group(1).strip() for m in WIKILINK_RE.finditer(body)]

    return VaultNote(
        path=path, title=title, frontmatter=frontmatter, body=body, outgoing_links=links
    )


def scan_vault(vault_path: str | Path) -> list[VaultNote]:
    """Pass 1a: parse every .md file in the vault. Pure Python, no LLM."""
    root = Path(vault_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    notes: list[VaultNote] = []
    for path in sorted(root.rglob("*.md")):
        if ".obsidian" in path.parts:  # skip Obsidian's own bookkeeping folder
            continue
        note = _parse_note(path)
        if note is not None:
            notes.append(note)
    return notes


def build_title_slug_map(notes: list[VaultNote]) -> dict[str, str]:
    """Pass 1b: title (lowercased) -> bare slug, across the WHOLE vault,
    before any note is classified or written. Mirrors
    storage.rewire_references, which needs this same full-corpus view to
    rewrite links safely.

    Title collisions (two notes with the same title in different folders)
    are disambiguated by suffixing the slug with the parent folder name,
    rather than silently picking one - silent collision resolution here
    would produce confidently-wrong wikilinks downstream.
    """
    seen_slugs: dict[str, Path] = {}
    title_map: dict[str, str] = {}

    for note in notes:
        base_slug = slugify(note.title)
        slug = base_slug
        if slug in seen_slugs and seen_slugs[slug] != note.path:
            tag = slugify(note.path.parent.name) or "dup"
            slug = f"{base_slug}-{tag}"
            n = 2
            while slug in seen_slugs:
                slug = f"{base_slug}-{tag}-{n}"
                n += 1
            logger.info(
                "Title collision for %r: disambiguated %s -> %s",
                note.title,
                note.path,
                slug,
            )
        seen_slugs[slug] = note.path
        title_map[note.title.lower()] = slug

    return title_map


# ---------------------------------------------------------------------------
# Pass 2: classify (LLM), dedup, rewrite links, write (all deterministic
# except the one classify call)
# ---------------------------------------------------------------------------


def _classify_note(llm: AIService, note: VaultNote) -> dict:
    """The ONE deliberate LLM touchpoint in migration - one call per note.
    Falls back to a safe deterministic default ("topic", empty summary) on
    any failure; migration should never hard-fail on a single bad note.
    """
    prompt = MIGRATE_CLASSIFY_PROMPT_TMPL.format(
        title=note.title,
        frontmatter=json.dumps(note.frontmatter) if note.frontmatter else "(none)",
        body=note.body.strip()[:4000],  # guard against pathologically long notes
    )
    raw = llm.llm_call(system_prompt=MIGRATE_CLASSIFY_SYSTEM, prompt=prompt)
    if raw["response"] is None:
        logger.warning("Classification failed for %s; defaulting to topic", note.path)
        return {"type": "topic", "summary": "", "aliases": []}

    try:
        cleaned = re.sub(r"^```(json)?\s*|\s*```$", "", raw["response"].strip())
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(
            "Unparseable classification for %s; defaulting to topic", note.path
        )
        return {"type": "topic", "summary": "", "aliases": []}

    if data.get("type") not in TYPE_DIR:
        data["type"] = "topic"
    return data


def _rewrite_links(body: str, title_ref_map: dict[str, str]) -> str:
    """Rewrite [[Title]] / [[Title|Display]] into Gray Box's [[type/slug]]
    form. Runs AFTER classification, since a correct ref needs the note's
    resolved type, not just its slug. Unresolved targets (e.g. links into
    attachments, or notes that failed to parse) fall back to plain display
    text rather than being silently dropped.
    """

    def repl(m: re.Match) -> str:
        target_title = m.group(1).strip()
        display = m.group(2) or target_title
        ref = title_ref_map.get(target_title.lower())
        if ref:
            return f"[[{ref}]]" if display == target_title else f"[[{ref}|{display}]]"
        return display

    return WIKILINK_RE.sub(repl, body)


def _dedup_match(cfg: Config, page_type: str, title: str) -> Page | None:
    """Same fuzzy entity-resolution logic organizer.py's _get_or_create_page
    uses for native capture, so a vault note titled 'Aaryan Verma' merges
    into an existing person/aaryan-verma page instead of duplicating it.
    """
    existing = find_page_by_name(cfg, page_type, title)
    if existing:
        return existing

    q = Query.parse(title)
    best: Page | None = None
    best_score = 0.0
    for page in list_pages(cfg, page_type):
        score = Engine.name_scorer(q, _PageDoc(page))
        if score >= cfg.retrieval.dedup_threshold and score > best_score:
            best, best_score = page, score
    return best


def migrate_vault(
    cfg: Config,
    llm: AIService,
    vault_path: str | Path,
    dry_run: bool = False,
) -> MigrationReport:
    """One-time Obsidian vault import. NOT a sync.
    Safe to run once against a fresh vault; re-running against a vault
    whose migrated pages have since been edited via curate.py or
    accumulated organizer notes risks overwriting that state.
    """
    report = MigrationReport(vault_path=str(vault_path))

    notes = scan_vault(vault_path)
    report.total_notes = len(notes)
    if not notes:
        return report

    title_slug_map = build_title_slug_map(
        notes
    )  # title -> bare slug (type unknown yet)

    # Pass 2a: classify every note BEFORE rewriting any links - a link's
    # correct ref needs type/slug, not just slug.
    classifications: dict[str, dict] = {}
    for note in notes:
        try:
            classifications[note.title] = _classify_note(llm, note)
        except Exception as e:  # noqa: BLE001
            logger.exception("Classification error for %s", note.path)
            report.errors.append({"note": str(note.path), "error": str(e)})
            classifications[note.title] = {
                "type": "topic",
                "summary": "",
                "aliases": [],
            }

    title_ref_map = {
        note.title.lower(): f"{classifications[note.title]['type']}/{title_slug_map[note.title.lower()]}"
        for note in notes
    }

    # Pass 2b: dedup, rewrite links, write.
    for note in notes:
        cls = classifications[note.title]
        page_type = cls["type"]

        try:
            existing = _dedup_match(cfg, page_type, note.title)
            rewritten_body = _rewrite_links(note.body, title_ref_map)
            is_new = existing is None

            if existing:
                target = existing
            else:
                slug = title_slug_map[note.title.lower()]
                target = Page(
                    id=slug,
                    type=page_type,
                    title=note.title,
                    created=now_iso(),
                    updated=now_iso(),
                    aliases=list(dict.fromkeys(cls.get("aliases") or [])),
                    summary=cls.get("summary", ""),
                )

            if dry_run:
                bucket = report.created if is_new else report.merged
                bucket.append(
                    MigratedPage(
                        ref=target.ref,
                        title=note.title,
                        status="created" if is_new else "merged",
                    )
                )
                continue

            # Synthetic inbox entry: preserves the "every fact traces to a
            # source" guarantee for migrated pages too.
            inbox_item = write_inbox_item(
                cfg,
                f"(migrated from Obsidian note: {note.path.name})\n\n{note.body.strip()}",
                extra={"migrated_from": str(note.path), "vault_title": note.title},
            )

            target.notes.append(
                f"- ({now_iso()}) Migrated from Obsidian note {note.path.name!r}. "
                f"_(source: inbox/{inbox_item.id})_\n  > {rewritten_body.strip()[:500]}"
            )
            if inbox_item.id not in target.sources:
                target.sources.append(inbox_item.id)
            target.updated = now_iso()
            if not target.summary and cls.get("summary"):
                target.summary = cls["summary"]

            write_page(cfg, target)

            bucket = report.created if is_new else report.merged
            bucket.append(
                MigratedPage(
                    ref=target.ref,
                    title=note.title,
                    status="created" if is_new else "merged",
                )
            )

        except Exception as e:  # noqa: BLE001
            logger.exception("Migration error for %s", note.path)
            report.errors.append({"note": str(note.path), "error": str(e)})
            report.skipped.append(
                MigratedPage(ref="", title=note.title, status="skipped", reason=str(e))
            )

    return report
