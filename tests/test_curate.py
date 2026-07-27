"""Tests for curate.py — merge, edit, delete, and reference rewiring."""
from __future__ import annotations
 
import pytest
 
from graybox.curate import merge_pages, edit_page, delete_page, find_possible_duplicates
from graybox.models import Page, now_iso
from graybox.storage import read_page, write_page, list_pages
 
 
class TestMergePages:
    def test_merge_unions_notes_sources_aliases(self, temp_cfg):
        primary = Page(
            id="alice", type="person", title="Alice",
            created=now_iso(), updated=now_iso(),
            aliases=["Ali"],
            notes=["- (2026-01-01 10:00:00) Note from primary. _(source: inbox/a)_"],
            sources=["inbox/a"],
            related=["project/x"],
            tags=["work"],
            status="active",
            summary="Primary summary",
        )
        secondary = Page(
            id="alice-smith", type="person", title="Alice Smith",
            created=now_iso(), updated=now_iso(),
            aliases=["A. Smith"],
            notes=["- (2026-02-01 11:00:00) Note from secondary. _(source: inbox/b)_"],
            sources=["inbox/b"],
            related=["project/y"],
            tags=["personal"],
            status="",
            summary="",
            owner="Bob",
        )
        write_page(temp_cfg, primary)
        write_page(temp_cfg, secondary)
 
        report = merge_pages(temp_cfg, "person/alice", "person/alice-smith", dry_run=False)
        assert report["merged_into"] == "person/alice"
        assert report["notes_after"] == 2
        assert report["sources_after"] == 2
 
        merged = read_page(temp_cfg, "person", "alice")
        assert "Alice Smith" in merged.aliases
        assert "A. Smith" in merged.aliases
        assert any("Note from primary" in n for n in merged.notes)
        assert any("Note from secondary" in n for n in merged.notes)
        assert "project/x" in merged.related
        assert "project/y" in merged.related
        assert "work" in merged.tags
        assert "personal" in merged.tags
        assert merged.summary == "Primary summary"
        assert merged.owner == "Bob"
        assert read_page(temp_cfg, "person", "alice-smith") is None
 
    def test_merge_rewires_references(self, temp_cfg):
        a = Page(id="a", type="topic", title="A", created=now_iso(), updated=now_iso(), related=["person/old"])
        old = Page(id="old", type="person", title="Old", created=now_iso(), updated=now_iso())
        new = Page(id="new", type="person", title="New", created=now_iso(), updated=now_iso())
        write_page(temp_cfg, a)
        write_page(temp_cfg, old)
        write_page(temp_cfg, new)
 
        report = merge_pages(temp_cfg, "person/new", "person/old", dry_run=False)
        assert "topic/a" in report["rewired_pages"]
 
        a_loaded = read_page(temp_cfg, "topic", "a")
        assert "person/new" in a_loaded.related
        assert "person/old" not in a_loaded.related
 
    def test_cannot_merge_into_self(self, temp_cfg):
        p = Page(id="x", type="topic", title="X", created=now_iso(), updated=now_iso())
        write_page(temp_cfg, p)
        with pytest.raises(ValueError, match="Cannot merge a page into itself"):
            merge_pages(temp_cfg, "topic/x", "topic/x")
 
    def test_dry_run_does_not_write(self, temp_cfg):
        primary = Page(id="a", type="topic", title="A", created=now_iso(), updated=now_iso())
        secondary = Page(id="b", type="topic", title="B", created=now_iso(), updated=now_iso())
        write_page(temp_cfg, primary)
        write_page(temp_cfg, secondary)
 
        report = merge_pages(temp_cfg, "topic/a", "topic/b", dry_run=True)
        assert report["dry_run"] is True
        assert read_page(temp_cfg, "topic", "b") is not None
 
 
class TestEditPage:
    def test_edit_title_moves_page(self, temp_cfg):
        page = Page(id="old-name", type="topic", title="Old Name", created=now_iso(), updated=now_iso())
        write_page(temp_cfg, page)
 
        report = edit_page(temp_cfg, "topic/old-name", new_title="New Name", dry_run=False)
        assert report["moved"] is True
        assert report["new_ref"] == "topic/new-name"
 
        assert read_page(temp_cfg, "topic", "old-name") is None
        loaded = read_page(temp_cfg, "topic", "new-name")
        assert loaded.title == "New Name"
        assert "Old Name" in loaded.aliases
 
    def test_edit_rewires_references(self, temp_cfg):
        a = Page(id="a", type="topic", title="A", created=now_iso(), updated=now_iso(), related=["topic/b"])
        b = Page(id="b", type="topic", title="B", created=now_iso(), updated=now_iso())
        write_page(temp_cfg, a)
        write_page(temp_cfg, b)
 
        report = edit_page(temp_cfg, "topic/b", new_title="Bee", dry_run=False)
        assert "topic/a" in report["rewired_pages"]
 
        a_loaded = read_page(temp_cfg, "topic", "a")
        assert "topic/bee" in a_loaded.related
        assert "topic/b" not in a_loaded.related
 
    def test_edit_adds_alias(self, temp_cfg):
        page = Page(id="x", type="topic", title="X", created=now_iso(), updated=now_iso())
        write_page(temp_cfg, page)
 
        edit_page(temp_cfg, "topic/x", add_aliases=["Ex", "X-ray"], dry_run=False)
        loaded = read_page(temp_cfg, "topic", "x")
        assert "Ex" in loaded.aliases
        assert "X-ray" in loaded.aliases
 
    def test_edit_type_moves_directory(self, temp_cfg):
        page = Page(id="x", type="topic", title="X", created=now_iso(), updated=now_iso())
        write_page(temp_cfg, page)
 
        edit_page(temp_cfg, "topic/x", new_type="person", dry_run=False)
        assert read_page(temp_cfg, "topic", "x") is None
        assert read_page(temp_cfg, "person", "x") is not None
 
 
class TestDeletePage:
    def test_delete_removes_file(self, temp_cfg):
        page = Page(id="x", type="topic", title="X", created=now_iso(), updated=now_iso())
        write_page(temp_cfg, page)
 
        report = delete_page(temp_cfg, "topic/x", dry_run=False)
        assert report["ref"] == "topic/x"
        assert read_page(temp_cfg, "topic", "x") is None
 
    def test_delete_rewires_references(self, temp_cfg):
        a = Page(id="a", type="topic", title="A", created=now_iso(), updated=now_iso(), related=["topic/b"])
        b = Page(id="b", type="topic", title="B", created=now_iso(), updated=now_iso())
        write_page(temp_cfg, a)
        write_page(temp_cfg, b)
 
        report = delete_page(temp_cfg, "topic/b", dry_run=False)
        assert "topic/a" in report["rewired_pages"]
 
        a_loaded = read_page(temp_cfg, "topic", "a")
        assert "topic/b" not in a_loaded.related
 
    def test_delete_traces_sources(self, temp_cfg):
        page = Page(id="x", type="topic", title="X", created=now_iso(), updated=now_iso(), sources=["inbox/123"])
        write_page(temp_cfg, page)
 
        report = delete_page(temp_cfg, "topic/x", dry_run=False)
        assert "inbox/123" in report["sources"]
 
    def test_dry_run_does_not_delete(self, temp_cfg):
        page = Page(id="x", type="topic", title="X", created=now_iso(), updated=now_iso())
        write_page(temp_cfg, page)
 
        report = delete_page(temp_cfg, "topic/x", dry_run=True)
        assert report["dry_run"] is True
        assert read_page(temp_cfg, "topic", "x") is not None
 
 
class TestFindPossibleDuplicates:
    def test_finds_similar_names(self, temp_cfg_low_threshold):
        write_page(temp_cfg_low_threshold, Page(id="alice", type="person", title="Alice", created=now_iso(), updated=now_iso()))
        write_page(temp_cfg_low_threshold, Page(id="alicee", type="person", title="Alicee", created=now_iso(), updated=now_iso()))
 
        dups = find_possible_duplicates(temp_cfg_low_threshold, threshold=0.85)
        assert len(dups) > 0
        assert any(d.page_a.id == "alice" and d.page_b.id == "alicee" for d in dups)
 
    def test_no_dupes_for_unrelated(self, temp_cfg):
        write_page(temp_cfg, Page(id="alice", type="person", title="Alice", created=now_iso(), updated=now_iso()))
        write_page(temp_cfg, Page(id="bob", type="person", title="Bob", created=now_iso(), updated=now_iso()))
 
        dups = find_possible_duplicates(temp_cfg, threshold=0.85)
        assert len(dups) == 0