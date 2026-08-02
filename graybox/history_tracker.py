"""
Pure-Python version history for wiki pages.

No external git binary. History is stored as JSON snapshots under
.state/history/ so every write, merge, edit, and delete is recoverable.
"""
from __future__ import annotations

import hashlib
import json
import logging
import difflib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from graybox.config import Config
from graybox.models import Page

logger = logging.getLogger(__name__)

_HISTORY_CACHE: dict[str, "HistoryTracker"] = {}


@dataclass
class Snapshot:
    ts: str
    message: str
    content_hash: str
    content: str

    def to_dict(self) -> dict:
        return {"ts": self.ts, "message": self.message, "hash": self.content_hash, "content": self.content}

    @classmethod
    def from_dict(cls, d: dict) -> "Snapshot":
        return cls(ts=d["ts"], message=d["message"], content_hash=d["hash"], content=d["content"])


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class HistoryTracker:
    """Records every version of every wiki page as a full snapshot.
    Storage cost is negligible for personal-scale use (thousands of pages).
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.root = cfg.state_dir / "history"
        self.root.mkdir(parents=True, exist_ok=True)

    def _history_path(self, ref: str) -> Path:
        """Map a page ref (e.g. person/alice) to a history file path."""
        safe = ref.replace("/", "--")
        # Shard by first char to avoid too many files in one dir
        shard = safe[0] if safe else "_"
        folder = self.root / shard
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{safe}.jsonl"

    def record(self, page: Page, message: str) -> None:
        """Append a snapshot of the current page content."""
        content = self._render_for_history(page)
        snap = Snapshot(
            ts=_now(),
            message=message,
            content_hash=_content_hash(content),
            content=content,
        )
        path = self._history_path(page.ref)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snap.to_dict(), ensure_ascii=False) + "\n")
        logger.debug("Recorded history for %s: %s", page.ref, message)

    def record_deletion(self, ref: str, message: str) -> None:
        """Record a deletion tombstone so undo knows the page existed."""
        snap = Snapshot(ts=_now(), message=message, content_hash="", content="")
        path = self._history_path(ref)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snap.to_dict(), ensure_ascii=False) + "\n")

    def history(self, ref: str, n: int = 20) -> list[Snapshot]:
        """Return the last n snapshots for a page, newest first."""
        path = self._history_path(ref)
        if not path.exists():
            return []
        snaps: list[Snapshot] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    snaps.append(Snapshot.from_dict(json.loads(line)))
                except Exception:
                    continue
        return list(reversed(snaps))[:n]

    def restore(self, ref: str, index: int = -1) -> Optional[str]:
        """Restore a page to a historical snapshot by index (0 = newest).
        Returns the restored content or None if no history exists."""
        snaps = self.history(ref, n=9999)
        if not snaps:
            return None
        try:
            snap = snaps[index]
        except IndexError:
            return None
        return snap.content

    def diff(self, ref: str, old_index: int = 1, new_index: int = 0) -> str:
        """Unified diff between two historical snapshots (default: previous vs current)."""
        snaps = self.history(ref, n=9999)
        if len(snaps) < 2:
            return "(not enough history to diff)"
        try:
            old = snaps[old_index].content.splitlines(keepends=True)
            new = snaps[new_index].content.splitlines(keepends=True)
        except IndexError:
            return "(invalid snapshot index)"
        return "".join(difflib.unified_diff(old, new, fromfile="before", tofile="after"))

    def undo(self, ref: str) -> Optional[str]:
        """Restore to the immediately previous snapshot (convenience wrapper)."""
        return self.restore(ref, index=1)

    def _render_for_history(self, page: Page) -> str:
        """Re-render the page exactly as it appears on disk."""
        from graybox.storage import _render_page
        return _render_page(page)


def _history_tracker(cfg: Config) -> HistoryTracker:
    ws_root = str(cfg.workspace)
    if ws_root not in _HISTORY_CACHE:
        _HISTORY_CACHE[ws_root] = HistoryTracker(cfg)
    return _HISTORY_CACHE[ws_root]


def _maybe_record(cfg: Config, page: Page, message: str) -> None:
    """Auto-record history on every wiki write."""
    try:
        _history_tracker(cfg).record(page, message)
    except Exception as e:
        logger.debug("History recording skipped: %s", e)


def _maybe_record_deletion(cfg: Config, ref: str, message: str) -> None:
    try:
        _history_tracker(cfg).record_deletion(ref, message)
    except Exception as e:
        logger.debug("History deletion recording skipped: %s", e)
