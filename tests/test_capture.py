"""Tests for capture.py — the immutable inbox writer."""
from __future__ import annotations

import pytest

from graybox.capture import capture, capture_file
from graybox.storage import list_inbox_items, read_inbox_item


class TestCapture:
    def test_capture_writes_inbox_item(self, temp_cfg):
        item = capture(temp_cfg, "Talked to Alice about Atlas migration")
        assert item.content == "Talked to Alice about Atlas migration"
        loaded = read_inbox_item(temp_cfg, item.id)
        assert loaded is not None
        assert loaded.content == item.content

    def test_capture_strips_whitespace(self, temp_cfg):
        item = capture(temp_cfg, "  some note with padding  \n")
        assert item.content == "some note with padding"

    def test_capture_rejects_empty_text(self, temp_cfg):
        with pytest.raises(ValueError, match="Cannot capture empty content"):
            capture(temp_cfg, "")

    def test_capture_rejects_whitespace_only(self, temp_cfg):
        with pytest.raises(ValueError, match="Cannot capture empty content"):
            capture(temp_cfg, "   \n\t  ")

    def test_capture_appears_in_list(self, temp_cfg):
        capture(temp_cfg, "First note")
        capture(temp_cfg, "Second note")
        items = list_inbox_items(temp_cfg)
        assert len(items) == 2
        contents = {i.content for i in items}
        assert "First note" in contents
        assert "Second note" in contents


class TestCaptureFile:
    def test_capture_file_imports_text_file(self, temp_cfg, tmp_path):
        src = tmp_path / "note.txt"
        src.write_text("Imported content here", encoding="utf-8")
        item = capture_file(temp_cfg, str(src))
        assert "Imported content here" in item.content
        assert str(src) in item.content  # header records source path

    def test_capture_file_missing_raises(self, temp_cfg, tmp_path):
        missing = tmp_path / "nope.txt"
        with pytest.raises(ValueError, match="File not found"):
            capture_file(temp_cfg, str(missing))

    def test_capture_file_rejects_directory(self, temp_cfg, tmp_path):
        d = tmp_path / "adir"
        d.mkdir()
        with pytest.raises(ValueError, match="Not a file"):
            capture_file(temp_cfg, str(d))

    def test_capture_file_rejects_binary(self, temp_cfg, tmp_path):
        src = tmp_path / "bin.dat"
        src.write_bytes(b"\xff\xfe\x00\x01binarydata\x80\x81")
        with pytest.raises(ValueError, match="isn't valid UTF-8"):
            capture_file(temp_cfg, str(src))

class TestCaptureIdFormat:
    """Regression test for the space-in-inbox-id bug: now_iso()'s
    presentation format must never be able to leak into item.id, since the
    id is used unquoted in filenames and CLI commands."""

    def test_id_has_no_whitespace(self, temp_cfg):
        item = capture(temp_cfg, "Some note")
        assert " " not in item.id
        assert "\t" not in item.id

    def test_id_matches_documented_pattern(self, temp_cfg):
        import re
        item = capture(temp_cfg, "Some note")
        # Matches the format documented in InboxItem's docstring,
        # e.g. 20260724-153000-ab12
        assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{4}", item.id)