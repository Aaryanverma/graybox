"""
Curate Agent.

Human-in-the-loop remediation for organizer mistakes: duplicate pages, wrong
titles/types, and hallucinated entities. Unlike the Organizer, nothing here
runs automatically on a schedule or after every capture - every action is an
explicit command from a person. All writes are deterministic Python, the
same guarantee the rest of the storage layer makes; nothing here calls the
LLM.
"""
from __future__ import annotations

from dataclasses import dataclass

from graybox.config import Config
from graybox.ai import AIService
from graybox.models import Page, TYPE_DIR, now_iso
from graybox.history_tracker import _maybe_record, _maybe_record_deletion
from graybox.embedding_index import ensure_indexed, _get_index
from graybox.storage import (
    page_path, read_page, rewire_references, slugify, write_page,
)


def _parse_ref(ref: str) -> tuple[str, str]:
    if "/" not in ref:
        raise ValueError(f"Expected a 'type/slug' reference (e.g. person/aaryan), got: {ref!r}")
    page_type, slug = ref.split("/", 1)
    if page_type not in TYPE_DIR:
        raise ValueError(f"Unknown page type {page_type!r} in ref {ref!r}")
    return page_type, slug


def _get_page(cfg: Config, ref: str) -> Page:
    page_type, slug = _parse_ref(ref)
    page = read_page(cfg, page_type, slug)
    if page is None:
        raise ValueError(f"No such page: {ref}")
    return page


@dataclass
class DuplicateCandidate:
    page_a: Page
    page_b: Page
    similarity: float
    reason: str


def find_possible_duplicates(
    cfg: Config, page_type: str | None = None, threshold: float | None = None
) -> list[DuplicateCandidate]:
    threshold = threshold if threshold is not None else cfg.retrieval.dedup_threshold
    from graybox.search import find_duplicates
    hits = find_duplicates(cfg, page_type=page_type, threshold=threshold)
    return [
        DuplicateCandidate(
            page_a=h[0].doc.page, page_b=h[1].doc.page,
            similarity=h[2], reason=h[3]
        )
        for h in hits
    ]

def merge_pages(cfg: Config, primary_ref: str, secondary_ref: str, dry_run: bool = False) -> dict:
    if primary_ref == secondary_ref:
        raise ValueError("Cannot merge a page into itself.")

    primary = _get_page(cfg, primary_ref)
    secondary = _get_page(cfg, secondary_ref)

    merged_aliases = list(dict.fromkeys(primary.aliases + [secondary.title] + secondary.aliases))
    merged_aliases = [a for a in merged_aliases if a.lower() != primary.title.lower()]

    merged = Page(
        id=primary.id, type=primary.type, title=primary.title,
        created=primary.created, updated=now_iso(),
        aliases=merged_aliases,
        related=[r for r in dict.fromkeys(primary.related + secondary.related)
                 if r not in (primary.ref, secondary.ref)],
        backlinks=[r for r in dict.fromkeys(primary.backlinks + secondary.backlinks)
                   if r not in (primary.ref, secondary.ref)],
        sources=list(dict.fromkeys(primary.sources + secondary.sources)),
        tags=list(dict.fromkeys(primary.tags + secondary.tags)),
        status=primary.status or secondary.status,
        summary=primary.summary or secondary.summary,
        notes=list(dict.fromkeys(primary.notes + secondary.notes)),
        attendees=list(dict.fromkeys(primary.attendees + secondary.attendees)),
        date=primary.date or secondary.date,
        owner=primary.owner or secondary.owner,
        due=primary.due or secondary.due,
    )

    report = {
        "primary": primary.ref, "secondary": secondary.ref, "merged_into": merged.ref,
        "notes_before": len(primary.notes), "notes_after": len(merged.notes),
        "sources_before": len(primary.sources), "sources_after": len(merged.sources),
        "rewired_pages": [], "dry_run": dry_run,
    }
    if dry_run:
        return report

    write_page(cfg, merged)
    secondary_path = page_path(cfg, secondary.type, secondary.id)
    merged_path = page_path(cfg, merged.type, merged.id)
    if secondary_path.exists() and secondary_path != merged_path:
        secondary_path.unlink()

    report["rewired_pages"] = rewire_references(
        cfg, old_ref=secondary.ref, new_ref=merged.ref, exclude_refs={merged.ref}
    )
    _maybe_record(cfg, merged, f"wiki: merge {secondary_ref} into {primary_ref}")
    ensure_indexed(cfg, merged, AIService(cfg))
    # Drop old secondary from embedding index
    idx = _get_index(cfg)
    if idx:
        idx.drop(secondary.ref)
    return report


def edit_page(
    cfg: Config, ref: str, new_title: str | None = None, new_type: str | None = None,
    new_status: str | None = None, add_aliases: list[str] | None = None,
    dry_run: bool = False,
) -> dict:
    page = _get_page(cfg, ref)
    old_ref = page.ref
    old_path = page_path(cfg, page.type, page.id)

    if new_type and new_type not in TYPE_DIR:
        raise ValueError(f"Unknown page type: {new_type!r}")

    if new_title and new_title != page.title:
        if page.title not in page.aliases:
            page.aliases.append(page.title)
        page.title = new_title
        page.id = slugify(new_title)

    if new_type:
        page.type = new_type

    if new_status is not None:
        page.status = new_status

    for alias in (add_aliases or []):
        if alias and alias not in page.aliases and alias.lower() != page.title.lower():
            page.aliases.append(alias)

    page.updated = now_iso()
    new_ref = page.ref

    report = {
        "old_ref": old_ref, "new_ref": new_ref,
        "moved": old_ref != new_ref, "rewired_pages": [], "dry_run": dry_run,
    }
    if dry_run:
        return report

    write_page(cfg, page)
    new_path = page_path(cfg, page.type, page.id)
    if old_path.exists() and old_path != new_path:
        old_path.unlink()

    _maybe_record(cfg, page, f"wiki: edit {old_ref} -> {new_ref}")
    ensure_indexed(cfg, page, AIService(cfg))
    if old_ref != new_ref:
        report["rewired_pages"] = rewire_references(
            cfg, old_ref=old_ref, new_ref=new_ref, exclude_refs={new_ref}
        )
    return report


def delete_page(cfg: Config, ref: str, dry_run: bool = False) -> dict:
    page = _get_page(cfg, ref)
    path = page_path(cfg, page.type, page.id)

    report = {"ref": page.ref, "sources": list(page.sources), "rewired_pages": [], "dry_run": dry_run}
    if dry_run:
        return report

    if path.exists():
        path.unlink()

    report["rewired_pages"] = rewire_references(cfg, old_ref=page.ref, new_ref=None)
    _maybe_record_deletion(cfg, ref, f"wiki: delete {ref}")
    idx = _get_index(cfg)
    if idx:
        idx.drop(ref)
    return report