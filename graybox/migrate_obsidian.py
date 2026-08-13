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

WIKILINK_RE = re.compile(
    r"\[\[([^\]|#]+)(?:#([^\]|]+))?(?:\|([^\]]+))?\]\]"
)


def _note_key(note: VaultNote) -> str:
    """Return a stable identity for a vault note, independent of its title."""
    return str(note.path)


def _as_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_string_list(value: object) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        return []
    return list(
        dict.fromkeys(
            item.strip()
            for item in values
            if isinstance(item, str) and item.strip()
        )
    )


def _normalise_link_key(value: str) -> str:
    value = value.strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    if value.lower().endswith(".md"):
        value = value[:-3]
    return value.strip("/").casefold()


def _add_link_mapping(mapping: dict[str, str | None], key: str, ref: str) -> None:
    key = _normalise_link_key(key)
    if not key:
        return
    if key in mapping and mapping[key] != ref:
        mapping[key] = None  # Ambiguous links must not resolve confidently.
    else:
        mapping[key] = ref


# ---------------------------------------------------------------------------
# Pass 1: parse vault, build note-identity -> slug map (no LLM involved)
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
                loaded = yaml.safe_load(m.group(1)) or {}
                if isinstance(loaded, dict):
                    frontmatter = loaded
                else:
                    logger.warning("Ignoring non-object frontmatter in %s", path)
            except yaml.YAMLError:
                logger.warning("Ignoring invalid frontmatter in %s", path)
            body = m.group(2)

    title = _as_string(frontmatter.get("title")) or path.stem
    links = [m.group(1).strip() for m in WIKILINK_RE.finditer(body)]

    return VaultNote(
        path=path,
        title=title,
        frontmatter=frontmatter,
        body=body,
        outgoing_links=links,
        source_text=text,
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
    """Pass 1b: note path -> bare slug, across the WHOLE vault,
    before any note is classified or written. Mirrors
    storage.rewire_references, which needs this same full-corpus view to
    rewrite links safely.

    Title collisions (two notes with the same title in different folders)
    are disambiguated by suffixing the slug with the parent folder name,
    rather than silently picking one - silent collision resolution here
    would produce confidently-wrong wikilinks downstream.
    """
    seen_slugs: dict[str, Path] = {}
    note_map: dict[str, str] = {}

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
        note_map[_note_key(note)] = slug

    return note_map


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
        frontmatter=json.dumps(note.frontmatter, default=str) if note.frontmatter else "(none)",
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

    if not isinstance(data, dict):
        logger.warning("Classification was not an object for %s; defaulting to topic", note.path)
        return {"type": "topic", "summary": "", "aliases": []}
    if data.get("type") not in TYPE_DIR:
        data["type"] = "topic"
    data["summary"] = _as_string(data.get("summary"))
    data["aliases"] = _as_string_list(data.get("aliases"))
    return data


def _rewrite_links(body: str, title_ref_map: dict[str, str | None]) -> str:
    """Rewrite [[Title]] / [[Title|Display]] into Gray Box's [[type/slug]]
    form. Runs AFTER classification, since a correct ref needs the note's
    resolved type, not just its slug. Unresolved targets (e.g. links into
    attachments, or notes that failed to parse) are preserved verbatim rather
    than being silently dropped.
    """

    def repl(m: re.Match) -> str:
        target_title = m.group(1).strip()
        heading = m.group(2)
        display = m.group(3) or target_title
        ref = title_ref_map.get(_normalise_link_key(target_title))
        if ref:
            anchor = f"#{heading}" if heading else ""
            return (
                f"[[{ref}{anchor}]]"
                if display == target_title
                else f"[[{ref}{anchor}|{display}]]"
            )
        # Preserve unresolved links, embeds, and ambiguous duplicate-title
        # links instead of silently destroying the user's Markdown.
        return m.group(0)

    return WIKILINK_RE.sub(repl, body)


def _build_link_map(
    cfg: Config,
    notes: list[VaultNote],
    note_targets: dict[str, Page],
    vault_root: Path,
) -> dict[str, str | None]:
    """Build only unambiguous title/path/alias mappings to real page refs."""
    mapping: dict[str, str | None] = {}
    for note in notes:
        target = note_targets[_note_key(note)]
        _add_link_mapping(mapping, note.title, target.ref)
        _add_link_mapping(mapping, note.path.stem, target.ref)
        try:
            relative = note.path.relative_to(vault_root).with_suffix("")
            _add_link_mapping(mapping, relative.as_posix(), target.ref)
        except ValueError:
            pass
        for alias in _as_string_list(note.frontmatter.get("aliases")):
            _add_link_mapping(mapping, alias, target.ref)

    # Also resolve links to pages already present in Gray Box when their
    # title/alias is unambiguous.
    for page in list_pages(cfg):
        _add_link_mapping(mapping, page.title, page.ref)
        for alias in page.aliases:
            _add_link_mapping(mapping, alias, page.ref)
    return mapping


def _quote_markdown(text: str) -> str:
    """Quote every source line so Gray Box's note parser keeps it together."""
    lines = text.strip().splitlines()
    return "\n".join(f"  > {line}" if line else "  >" for line in lines) or "  >"


def _new_page_from_note(note: VaultNote, cls: dict, slug: str) -> Page:
    frontmatter = note.frontmatter
    aliases = _as_string_list(frontmatter.get("aliases"))
    aliases.extend(cls.get("aliases", []))
    extra = {"obsidian_frontmatter": frontmatter} if frontmatter else {}

    def scalar(name: str) -> str:
        value = frontmatter.get(name)
        return (
            str(value)
            if value is not None and not isinstance(value, (dict, list, tuple, set))
            else ""
        )

    return Page(
        id=slug,
        type=cls["type"],
        title=note.title,
        created=scalar("created") or now_iso(),
        updated=scalar("updated") or now_iso(),
        aliases=list(dict.fromkeys(aliases)),
        tags=_as_string_list(frontmatter.get("tags")),
        status=scalar("status"),
        summary=cls.get("summary", ""),
        attendees=_as_string_list(frontmatter.get("attendees")),
        date=scalar("date"),
        owner=scalar("owner"),
        due=scalar("due"),
        extra=extra,
    )


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

    vault_root = Path(vault_path).expanduser().resolve()
    note_slug_map = build_title_slug_map(notes)

    # Pass 2a: classify every note BEFORE rewriting any links - a link's
    # correct ref needs type/slug, not just slug.
    classifications: dict[str, dict] = {}
    for note in notes:
        key = _note_key(note)
        try:
            classifications[key] = _classify_note(llm, note)
        except Exception as e:  # noqa: BLE001
            logger.exception("Classification error for %s", note.path)
            report.errors.append({"note": str(note.path), "error": str(e)})
            classifications[key] = {
                "type": "topic",
                "summary": "",
                "aliases": [],
            }

    # Resolve every note before rewriting links. This makes links point to
    # the actual existing page when fuzzy deduplication merges a note.
    note_targets: dict[str, Page] = {}
    note_is_new: dict[str, bool] = {}
    for note in notes:
        key = _note_key(note)
        cls = classifications[key]
        existing = _dedup_match(cfg, cls["type"], note.title)
        note_is_new[key] = existing is None
        note_targets[key] = existing or _new_page_from_note(
            note, cls, note_slug_map[key]
        )

    title_ref_map = _build_link_map(cfg, notes, note_targets, vault_root)

    # Pass 2b: dedup, rewrite links, write.
    for note in notes:
        key = _note_key(note)
        cls = classifications[key]
        try:
            rewritten_body = _rewrite_links(note.body, title_ref_map)
            target = note_targets[key]
            is_new = note_is_new[key]

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
                note.source_text.strip(),
                extra={"migrated_from": str(note.path), "vault_title": note.title},
            )

            target.notes.append(
                f"- ({now_iso()}) Migrated from Obsidian note {note.path.name!r}. "
                f"_(source: inbox/{inbox_item.id})_\n{_quote_markdown(rewritten_body)}"
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
