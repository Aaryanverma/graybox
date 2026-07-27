"""Tests for forget.py — soft/purge/scrub retraction of bad captures."""
from __future__ import annotations

import pytest

from graybox.capture import capture
from graybox.forget import forget_item
from graybox.models import Page, now_iso
from graybox.storage import (
    is_forgotten,
    list_inbox_items,
    load_forgotten,
    mark_processed,
    read_page,
    write_page,
)


class TestForgetSoft:
    def test_soft_forget_marks_tombstone(self, temp_cfg):
        item = capture(temp_cfg, "Bad note")
        report = forget_item(temp_cfg, item.id, reason="typo'd note")
        assert report["item_id"] == item.id
        assert report["purged"] is False
        assert is_forgotten(temp_cfg, item.id)
        forgotten = load_forgotten(temp_cfg)
        assert forgotten[item.id]["reason"] == "typo'd note"

    def test_soft_forget_excludes_from_list(self, temp_cfg):
        item = capture(temp_cfg, "Bad note")
        capture(temp_cfg, "Good note")
        forget_item(temp_cfg, item.id)
        items = list_inbox_items(temp_cfg)
        assert len(items) == 1
        assert items[0].content == "Good note"

    def test_soft_forget_keeps_raw_file(self, temp_cfg):
        item = capture(temp_cfg, "Bad note")
        forget_item(temp_cfg, item.id)
        from pathlib import Path
        assert Path(item.path).exists()

    def test_forget_unknown_item_raises(self, temp_cfg):
        with pytest.raises(ValueError, match="No such inbox item"):
            forget_item(temp_cfg, "does-not-exist")


class TestForgetPurge:
    def test_purge_deletes_raw_file(self, temp_cfg):
        item = capture(temp_cfg, "Sensitive note")
        report = forget_item(temp_cfg, item.id, purge=True)
        assert report["purged"] is True
        from pathlib import Path
        assert not Path(item.path).exists()


class TestForgetScrub:
    def test_scrub_strips_notes_from_touched_pages(self, temp_cfg):
        item = capture(temp_cfg, "Alice works on Atlas")
        page = Page(
            id="alice", type="person", title="Alice",
            created=now_iso(), updated=now_iso(),
            notes=[f"- (2026-07-26 10:00:00) Alice works on Atlas. _(source: inbox/{item.id})_"],
            sources=[item.id],
        )
        write_page(temp_cfg, page)
        mark_processed(temp_cfg, item.id, ["person/alice"])

        report = forget_item(temp_cfg, item.id, scrub=True)
        assert report["already_processed"] is True
        assert "person/alice" in report["scrubbed_pages"]

        reloaded = read_page(temp_cfg, "person", "alice")
        assert reloaded.notes == []
        assert item.id not in reloaded.sources

    def test_scrub_noop_if_not_processed(self, temp_cfg):
        item = capture(temp_cfg, "Unprocessed note")
        report = forget_item(temp_cfg, item.id, scrub=True)
        assert report["already_processed"] is False
        assert report["scrubbed_pages"] == []

    def test_scrub_leaves_unrelated_pages_alone(self, temp_cfg):
        item = capture(temp_cfg, "Alice works on Atlas")
        other = Page(
            id="bob", type="person", title="Bob",
            created=now_iso(), updated=now_iso(),
            notes=["- (2026-07-26 10:00:00) Unrelated note. _(source: inbox/other-item)_"],
            sources=["other-item"],
        )
        write_page(temp_cfg, other)
        mark_processed(temp_cfg, item.id, [])

        forget_item(temp_cfg, item.id, scrub=True)
        reloaded = read_page(temp_cfg, "person", "bob")
        assert len(reloaded.notes) == 1