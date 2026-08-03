"""
Plain data structures for the knowledge store.

Everything is Markdown + YAML frontmatter on disk. These dataclasses are just
an in-memory convenience layer over that format.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

PAGE_TYPES = (
    "project",
    "person",
    "meeting",
    "technology",
    "company",
    "topic",
    "task",
    "decision",
    "action",
    "journal",
)

# Folder each page type lives in (plural directory names under wiki/)
TYPE_DIR = {
    "project": "projects",
    "person": "people",
    "meeting": "meetings",
    "technology": "technologies",
    "company": "technologies",  # companies filed alongside technologies/orgs
    "topic": "topics",
    "task": "tasks",
    "decision": "decisions",
    "action": "actions",
    "journal": "journal",
}


def now_iso() -> str:
    """UTC timestamp in ISO-8601 format, e.g. 2026-08-02T14:30:00Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_id_ts() -> str:
    """Compact UTC timestamp for building filesystem-safe ids, e.g. 20260802-143000.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

def now_readable() -> str:
    """Human-readable UTC timestamp, e.g. 2026-08-02 14:30:00."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

@dataclass
class InboxItem:
    id: str            # e.g. 20260724-153000-ab12
    created: str
    content: str
    path: str = ""      # populated on load


@dataclass
class Page:
    id: str             # slug, unique within its type
    type: str
    title: str
    created: str
    updated: str
    aliases: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)   # "type/slug" refs
    backlinks: list[str] = field(default_factory=list)  # "type/slug" refs
    sources: list[str] = field(default_factory=list)    # inbox item ids
    tags: list[str] = field(default_factory=list)
    status: str = ""     # used by tasks/decisions (open/done, proposed/decided)
    summary: str = ""
    notes: list[str] = field(default_factory=list)  # raw markdown note lines/blocks
    attendees: list[str] = field(default_factory=list)  # meeting pages only
    date: str = ""        # meeting/journal pages: the calendar date it covers (YYYY-MM-DD)
    owner: str = ""        # task pages: who owns it
    due: str = ""            # task pages: when it's due
    path: str = ""
    summary_refreshed_at: str = ""  # timestamp of last summarization; empty = never refreshed

    @property
    def ref(self) -> str:
        return f"{self.type}/{self.id}"

    def frontmatter(self) -> dict:
        fm = {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "created": self.created,
            "updated": self.updated,
            "aliases": self.aliases,
            "related": self.related,
            "backlinks": self.backlinks,
            "sources": self.sources,
            "tags": self.tags,
            "status": self.status,
        }
        # Only add these when actually used, so most pages stay uncluttered.
        if self.attendees:
            fm["attendees"] = self.attendees
        if self.date:
            fm["date"] = self.date
        if self.owner:
            fm["owner"] = self.owner
        if self.due:
            fm["due"] = self.due
        if self.summary_refreshed_at:
            fm["summary_refreshed_at"] = self.summary_refreshed_at
        return fm

    extra: dict[str, Any] = field(default_factory=dict)