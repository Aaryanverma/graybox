"""Tests for organizer.py — entity extraction, dedup, and page merging logic."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from graybox.organizer import (
    _get_or_create_page,
    _append_note,
    _link,
    _backlink,
    process_item,
    _extract_json,
    organize_all,
)
from graybox.models import Page, now_iso
from graybox.storage import read_page, write_page, write_inbox_item


class TestExtractJson:
    def test_plain_json(self):
        assert _extract_json('{\"entities\": []}') == {"entities": []}

    def test_strips_markdown_fence(self):
        assert _extract_json("```json\n{\"entities\": []}\n```") == {"entities": []}

    def test_finds_json_in_noise(self):
        text = "Some intro text {\"entities\": []} trailing text"
        assert _extract_json(text) == {"entities": []}


class TestGetOrCreatePage:
    def test_creates_new_page_when_no_match(self, temp_cfg):
        page, is_new = _get_or_create_page(temp_cfg, "person", "Alice", [], "")
        assert is_new is True
        assert page.title == "Alice"
        assert page.type == "person"

    def test_returns_exact_match(self, temp_cfg):
        existing = Page(id="alice", type="person", title="Alice", created=now_iso(), updated=now_iso())
        write_page(temp_cfg, existing)
        page, is_new = _get_or_create_page(temp_cfg, "person", "Alice", [], "")
        assert is_new is False
        assert page.id == "alice"

    def test_returns_alias_match(self, temp_cfg):
        existing = Page(id="alice", type="person", title="Alice", created=now_iso(), updated=now_iso(), aliases=["Ali"])
        write_page(temp_cfg, existing)
        page, is_new = _get_or_create_page(temp_cfg, "person", "Ali", [], "")
        assert is_new is False
        assert page.id == "alice"

    def test_picks_best_fuzzy_match_not_first(self, temp_cfg_low_threshold):
        """BUG FIX: Previously picked the first fuzzy match above threshold.
        Now picks the BEST match above threshold."""
        write_page(temp_cfg_low_threshold, Page(
            id="alex", type="person", title="Alexandra Smith", created=now_iso(), updated=now_iso()
        ))
        write_page(temp_cfg_low_threshold, Page(
            id="alice", type="person", title="Alice Smith", created=now_iso(), updated=now_iso()
        ))
        page, is_new = _get_or_create_page(temp_cfg_low_threshold, "person", "Alice", [], "")
        assert is_new is False
        assert page.id == "alice"

    def test_cross_type_does_not_use_owner(self, temp_cfg_low_threshold):
        """BUG FIX: Owner names must not leak identity across types.
        A task owned by 'Alice' is NOT the same entity as person/Alice."""
        write_page(temp_cfg_low_threshold, Page(
            id="alice", type="person", title="Alice", created=now_iso(), updated=now_iso()
        ))
        page, is_new = _get_or_create_page(temp_cfg_low_threshold, "task", "Fix bug", [], "")
        assert is_new is True
        assert page.title == "Fix bug"

    def test_adds_name_as_alias_when_fuzzy_matched(self, temp_cfg_low_threshold):
        write_page(temp_cfg_low_threshold, Page(
            id="alice", type="person", title="Alice", created=now_iso(), updated=now_iso()
        ))
        page, is_new = _get_or_create_page(temp_cfg_low_threshold, "person", "Ali", [], "")
        assert is_new is False
        assert "Ali" in page.aliases


class TestAppendNote:
    def test_appends_note_with_source(self):
        page = Page(id="x", type="topic", title="X", created="", updated="")
        _append_note(page, "Something happened", "123")
        assert len(page.notes) == 1
        assert "Something happened" in page.notes[0]
        assert "source: inbox/123" in page.notes[0]
        assert "123" in page.sources

    def test_sets_summary_on_first_note(self):
        page = Page(id="x", type="topic", title="X", created="", updated="", summary="")
        _append_note(page, "First note text", "1")
        assert page.summary == "First note text"

    def test_preserves_existing_summary(self):
        page = Page(id="x", type="topic", title="X", created="", updated="", summary="Existing")
        _append_note(page, "New note", "1")
        assert page.summary == "Existing"


class TestLinkBacklink:
    def test_link_adds_related(self):
        a = Page(id="a", type="topic", title="A", created="", updated="")
        _link(a, "topic/b")
        assert "topic/b" in a.related

    def test_link_skips_self(self):
        a = Page(id="a", type="topic", title="A", created="", updated="")
        _link(a, "topic/a")
        assert "topic/a" not in a.related

    def test_backlink_adds_backlinks(self):
        a = Page(id="a", type="topic", title="A", created="", updated="")
        _backlink(a, "topic/b")
        assert "topic/b" in a.backlinks


class TestProcessItem:
    def test_extracts_entities_and_creates_pages(self, temp_cfg):
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [
                    {"type": "person", "name": "Alice", "aliases": [], "summary": "Engineer"}
                ],
                "relations": [],
                "tasks": [],
                "decisions": [],
                "meetings": []
            })
        }
        refs = process_item(temp_cfg, mock_llm, "item-1", "Alice is an engineer.", dry_run=False)
        assert "person/alice" in refs
        loaded = read_page(temp_cfg, "person", "alice")
        assert loaded is not None
        assert loaded.title == "Alice"
        assert "Engineer" in loaded.summary

    def test_extracts_task_with_owner(self, temp_cfg):
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [],
                "relations": [],
                "tasks": [
                    {"title": "Deploy app", "owner": "Alice", "due": "2026-08-01", "status": "open"}
                ],
                "decisions": [],
                "meetings": []
            })
        }
        refs = process_item(temp_cfg, mock_llm, "item-2", "Deploy app owned by Alice due Aug 1.", dry_run=False)
        assert "task/deploy-app" in refs
        loaded = read_page(temp_cfg, "task", "deploy-app")
        assert loaded.owner == "Alice"
        assert loaded.due == "2026-08-01"

    def test_dry_run_does_not_write(self, temp_cfg):
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [{"type": "topic", "name": "Test", "aliases": [], "summary": "X"}],
                "relations": [], "tasks": [], "decisions": [], "meetings": []
            })
        }
        refs = process_item(temp_cfg, mock_llm, "item-3", "Test topic.", dry_run=True)
        assert any("topic/test" in r for r in refs)
        assert read_page(temp_cfg, "topic", "test") is None

    def test_meeting_links_attendees(self, temp_cfg):
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [
                    {"type": "person", "name": "Alice", "aliases": [], "summary": "Attendee"},
                    {"type": "person", "name": "Bob", "aliases": [], "summary": "Attendee"},
                ],
                "relations": [],
                "tasks": [],
                "decisions": [],
                "meetings": [
                    {"title": "Sprint Planning", "date": "2026-07-26", "attendees": ["Alice", "Bob"], "agenda": "Plan sprint"}
                ]
            })
        }
        refs = process_item(temp_cfg, mock_llm, "item-4", "Sprint planning with Alice and Bob.", dry_run=False)
        assert "meeting/sprint-planning" in refs
        meeting = read_page(temp_cfg, "meeting", "sprint-planning")
        assert "Alice" in meeting.attendees
        assert "Bob" in meeting.attendees
        assert any("person/alice" in r or "person/bob" in r for r in meeting.related)


class TestAutoRefreshSummaries:
    def test_organize_auto_refreshes_when_enough_notes(self, temp_cfg):
        """After organizing, pages with >=3 notes should get auto-refreshed summaries."""
        # Pre-create page with 2 existing notes so after extraction it has 3
        existing = Page(
            id="bigproject", type="topic", title="BigProject",
            created=now_iso(), updated=now_iso(),
            summary="Old summary",
            notes=[
                "- (2026-01-01 10:00:00) Note one.",
                "- (2026-02-01 11:00:00) Note two.",
            ],
        )
        write_page(temp_cfg, existing)
        write_inbox_item(temp_cfg, "BigProject is now shipping v2.")

        mock_llm = MagicMock()
        mock_llm.llm_call.side_effect = [
            {
                "response": json.dumps({
                    "entities": [
                        {"type": "topic", "name": "BigProject", "aliases": [], "summary": "A project"}
                    ],
                    "relations": [], "tasks": [], "decisions": [], "meetings": []
                })
            },
            {"response": "Updated summary from all notes.", "cost": 0.001},
        ]

        report = organize_all(temp_cfg, mock_llm, dry_run=False)
        assert len(report["processed"]) == 1
        assert "topic/bigproject" in report["summaries_refreshed"]

        loaded = read_page(temp_cfg, "topic", "bigproject")
        assert loaded.summary == "Updated summary from all notes."
        assert loaded.summary_refreshed_at != ""

    def test_organize_skips_auto_refresh_when_few_notes(self, temp_cfg):
        """Pages with <3 notes should not trigger LLM summarization."""
        write_inbox_item(temp_cfg, "Small thing mentioned here.")

        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [
                    {"type": "topic", "name": "Small", "aliases": [], "summary": "Small thing"}
                ],
                "relations": [], "tasks": [], "decisions": [], "meetings": []
            })
        }

        report = organize_all(temp_cfg, mock_llm, dry_run=False)
        assert len(report["processed"]) == 1
        assert len(report["summaries_refreshed"]) == 0
        mock_llm.llm_call.assert_called_once()

    def test_dry_run_skips_auto_refresh(self, temp_cfg):
        """Dry-run organize should not call summarizer at all."""
        write_inbox_item(temp_cfg, "Test topic.")

        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [{"type": "topic", "name": "X", "aliases": [], "summary": "Y"}],
                "relations": [], "tasks": [], "decisions": [], "meetings": []
            })
        }

        report = organize_all(temp_cfg, mock_llm, dry_run=True)
        assert report["summaries_refreshed"] == []
        mock_llm.llm_call.assert_called_once()