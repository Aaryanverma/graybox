"""Tests for the pure-Python history tracker (git-like, no external git)."""
from __future__ import annotations

from graybox.history_tracker import HistoryTracker, _history_tracker, _maybe_record, _maybe_record_deletion
from graybox.models import Page, now_iso
from graybox.storage import read_page, write_page


class TestHistoryTracker:
    def test_records_write(self, temp_cfg):
        tracker = _history_tracker(temp_cfg)
        page = Page(id="x", type="topic", title="X", created=now_iso(), updated=now_iso(), summary="First")
        tracker.record(page, "Initial write")
        history = tracker.history("topic/x")
        assert len(history) == 1
        assert history[0].message == "Initial write"
        assert "First" in history[0].content

    def test_records_multiple_versions(self, temp_cfg):
        tracker = _history_tracker(temp_cfg)
        page = Page(id="y", type="topic", title="Y", created=now_iso(), updated=now_iso(), summary="V1")
        tracker.record(page, "v1")
        page.summary = "V2"
        tracker.record(page, "v2")
        page.summary = "V3"
        tracker.record(page, "v3")

        history = tracker.history("topic/y")
        assert len(history) == 3
        assert history[0].message == "v3"  # newest first
        assert history[1].message == "v2"
        assert history[2].message == "v1"

    def test_undo_restores_previous(self, temp_cfg):
        tracker = _history_tracker(temp_cfg)
        page = Page(id="z", type="topic", title="Z", created=now_iso(), updated=now_iso(), summary="Original")
        tracker.record(page, "original")
        page.summary = "Changed"
        tracker.record(page, "changed")

        restored = tracker.undo("topic/z")
        assert restored is not None
        assert "Original" in restored
        assert "Changed" not in restored

    def test_diff_between_snapshots(self, temp_cfg):
        tracker = _history_tracker(temp_cfg)
        page = Page(id="w", type="topic", title="W", created=now_iso(), updated=now_iso(), summary="Alpha")
        tracker.record(page, "alpha")
        page.summary = "Beta"
        tracker.record(page, "beta")

        diff = tracker.diff("topic/w", old_index=1, new_index=0)
        assert "Alpha" in diff
        assert "Beta" in diff
        assert "--- before" in diff
        assert "+++ after" in diff

    def test_record_deletion_tombstone(self, temp_cfg):
        tracker = _history_tracker(temp_cfg)
        tracker.record_deletion("topic/gone", "deleted")
        history = tracker.history("topic/gone")
        assert len(history) == 1
        assert history[0].message == "deleted"
        assert history[0].content_hash == ""

    def test_no_history_returns_empty(self, temp_cfg):
        tracker = _history_tracker(temp_cfg)
        assert tracker.history("topic/nonexistent") == []
        assert tracker.undo("topic/nonexistent") is None

    def test_integration_with_write_page(self, temp_cfg):
        """write_page should auto-record via _maybe_record."""
        page = Page(id="auto", type="topic", title="Auto", created=now_iso(), updated=now_iso(), summary="Auto summary")
        write_page(temp_cfg, page)

        tracker = _history_tracker(temp_cfg)
        history = tracker.history("topic/auto")
        assert len(history) >= 1
        assert "Auto summary" in history[0].content