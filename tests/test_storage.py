"""Tests for storage.py — markdown round-trip, frontmatter, note splitting, rewiring."""
from __future__ import annotations

from graybox.models import Page, now_iso
from graybox.storage import (
    write_page,
    read_page,
    list_pages,
    rewire_references,
    page_path,
    _split_notes,
    _replace_wiki_link,
)


class TestPageRoundTrip:
    def test_write_and_read_minimal_page(self, temp_cfg):
        page = Page(id="hello", type="topic", title="Hello", created=now_iso(), updated=now_iso())
        write_page(temp_cfg, page)
        loaded = read_page(temp_cfg, "topic", "hello")
        assert loaded is not None
        assert loaded.id == "hello"
        assert loaded.type == "topic"
        assert loaded.title == "Hello"

    def test_roundtrip_all_fields(self, temp_cfg):
        page = Page(
            id="test-page",
            type="person",
            title="Test Person",
            created="2026-01-01 00:00:00",
            updated="2026-07-26 12:00:00",
            aliases=["TP", "T-Person"],
            related=["project/alpha"],
            backlinks=["task/cleanup"],
            sources=["inbox/abc123"],
            tags=["important", "work"],
            status="active",
            summary="A test person for unit tests.",
            notes=["- (2026-07-26 10:00:00) First note. _(source: inbox/abc123)_"],
            attendees=["Alice", "Bob"],
            date="2026-07-26",
            owner="Alice",
            due="2026-08-01",
        )
        write_page(temp_cfg, page)
        loaded = read_page(temp_cfg, "person", "test-page")
        assert loaded.id == page.id
        assert loaded.title == page.title
        assert loaded.aliases == page.aliases
        assert loaded.related == page.related
        assert loaded.backlinks == page.backlinks
        assert loaded.sources == page.sources
        assert loaded.tags == page.tags
        assert loaded.status == page.status
        assert loaded.summary == page.summary
        assert loaded.notes == page.notes
        assert loaded.attendees == page.attendees
        assert loaded.date == page.date
        assert loaded.owner == page.owner
        assert loaded.due == page.due

    def test_note_continuation_preserved(self, temp_cfg):
        """Notes with continuation lines (indented '> ' blocks) must not be split incorrectly."""
        page = Page(
            id="notes-test",
            type="topic",
            title="Notes Test",
            created=now_iso(),
            updated=now_iso(),
            notes=[
                "- (2026-07-26 10:00:00) First note. _(source: inbox/a)_",
                """- (2026-07-26 11:00:00) Second note with continuation.
  > Quoted raw text here""",
                "- (2026-07-26 12:00:00) Third note. _(source: inbox/b)_",
            ],
        )
        write_page(temp_cfg, page)
        loaded = read_page(temp_cfg, "topic", "notes-test")
        assert len(loaded.notes) == 3
        assert "Quoted raw text here" in loaded.notes[1]

    def test_list_pages_filters_by_type(self, temp_cfg):
        write_page(temp_cfg, Page(id="p1", type="person", title="P1", created=now_iso(), updated=now_iso()))
        write_page(temp_cfg, Page(id="t1", type="task", title="T1", created=now_iso(), updated=now_iso()))
        write_page(temp_cfg, Page(id="p2", type="person", title="P2", created=now_iso(), updated=now_iso()))
        assert len(list_pages(temp_cfg, "person")) == 2
        assert len(list_pages(temp_cfg, "task")) == 1
        assert len(list_pages(temp_cfg)) == 3


class TestRewireReferences:
    def test_rewire_related_and_backlinks(self, temp_cfg):
        a = Page(id="a", type="topic", title="A", created=now_iso(), updated=now_iso(), related=["topic/b"])
        b = Page(id="b", type="topic", title="B", created=now_iso(), updated=now_iso(), backlinks=["topic/a"])
        write_page(temp_cfg, a)
        write_page(temp_cfg, b)

        changed = rewire_references(temp_cfg, old_ref="topic/b", new_ref="topic/c")
        assert "topic/a" in changed

        a_loaded = read_page(temp_cfg, "topic", "a")
        assert "topic/c" in a_loaded.related
        assert "topic/b" not in a_loaded.related

    def test_rewire_inline_wiki_links(self, temp_cfg):
        a = Page(
            id="a",
            type="topic",
            title="A",
            created=now_iso(),
            updated=now_iso(),
            notes=["See also [[topic/b]] for details."],
        )
        write_page(temp_cfg, a)
        rewire_references(temp_cfg, old_ref="topic/b", new_ref="topic/c")
        a_loaded = read_page(temp_cfg, "topic", "a")
        assert "[[topic/c]]" in a_loaded.notes[0]
        assert "[[topic/b]]" not in a_loaded.notes[0]

    def test_rewire_deletes_dangling_refs(self, temp_cfg):
        a = Page(id="a", type="topic", title="A", created=now_iso(), updated=now_iso(), related=["topic/b"])
        write_page(temp_cfg, a)
        rewire_references(temp_cfg, old_ref="topic/b", new_ref=None)
        a_loaded = read_page(temp_cfg, "topic", "a")
        assert "topic/b" not in a_loaded.related

    def test_rewire_self_ref_excluded(self, temp_cfg):
        a = Page(id="a", type="topic", title="A", created=now_iso(), updated=now_iso(), related=["topic/a"])
        write_page(temp_cfg, a)
        rewire_references(temp_cfg, old_ref="topic/a", new_ref="topic/z")
        a_loaded = read_page(temp_cfg, "topic", "a")
        # Self-refs should be removed, not rewired to point at themselves
        assert "topic/z" not in a_loaded.related


class TestSplitNotes:
    def test_split_simple_notes(self):
        raw = "- (2026-07-26 10:00:00) Note one\n- (2026-07-26 11:00:00) Note two"
        result = _split_notes(raw)
        assert len(result) == 2
        assert "Note one" in result[0]
        assert "Note two" in result[1]

    def test_split_ignores_placeholder(self):
        assert _split_notes("_No notes yet._") == []

    def test_split_preserves_continuations(self):
        raw = "- (2026-07-26 10:00:00) Note one\n  > continuation\n- (2026-07-26 11:00:00) Note two"
        result = _split_notes(raw)
        assert len(result) == 2
        assert "continuation" in result[0]


class TestReplaceWikiLink:
    def test_simple_link(self):
        import re
        match = re.search(r"\[\[topic/b\]\]", "See [[topic/b]]")
        assert _replace_wiki_link(match, "topic/c") == "[[topic/c]]"

    def test_piped_link(self):
        import re
        match = re.search(r"\[\[topic/b\|Display\]\]", "See [[topic/b|Display]]")
        assert _replace_wiki_link(match, "topic/c") == "[[topic/c]]"

    def test_delete_shows_display(self):
        import re
        match = re.search(r"\[\[topic/b\|Display\]\]", "See [[topic/b|Display]]")
        assert _replace_wiki_link(match, None) == "Display"