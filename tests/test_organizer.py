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
    _merge_unique,
    _system_prompt,
    _format_existing_context,
    process_item,
    _extract_json,
    organize_all,
)
from graybox.models import Page, now_iso
from graybox.search_engine import Hit, _PageDoc
from graybox.storage import read_page, write_page, write_inbox_item


class TestExtractJson:
    def test_plain_json(self):
        assert _extract_json('{\"entities\": []}') == {"entities": []}

    def test_strips_markdown_fence(self):
        assert _extract_json("```json\n{\"entities\": []}\n```") == {"entities": []}

    def test_finds_json_in_noise(self):
        text = "Some intro text {\"entities\": []} trailing text"
        assert _extract_json(text) == {"entities": []}

    def test_raises_when_no_json_present(self):
        """No braces anywhere -> the original JSONDecodeError propagates."""
        with pytest.raises(json.JSONDecodeError):
            _extract_json("just plain prose, no JSON here")


class TestMergeUnique:
    def test_adds_new_values(self):
        target = ["a"]
        _merge_unique(target, ["b", "c"])
        assert target == ["a", "b", "c"]

    def test_skips_duplicates(self):
        target = ["a"]
        _merge_unique(target, ["a", "b"])
        assert target == ["a", "b"]

    def test_skips_blank_and_whitespace_only_values(self):
        target = []
        _merge_unique(target, ["", "   ", "real"])
        assert target == ["real"]

    def test_handles_none_values_list(self):
        target = ["a"]
        _merge_unique(target, None)
        assert target == ["a"]


class TestSystemPrompt:
    def test_includes_workspace_context_when_present(self, temp_cfg):
        prompt = _system_prompt(temp_cfg)
        assert "Workspace context:" in prompt
        assert temp_cfg.workspace_name in prompt

    def test_falls_back_to_plain_system_prompt_when_context_empty(self, temp_cfg, monkeypatch):
        monkeypatch.setattr(
            "graybox.organizer.workspace_context_block", lambda cfg: ""
        )
        prompt = _system_prompt(temp_cfg)
        assert "Workspace context:" not in prompt


class TestFormatExistingContext:
    def test_no_hits_returns_placeholder(self):
        assert _format_existing_context([]) == "(none found)"

    def test_includes_status_owner_due_date_and_summary(self):
        page = Page(
            id="atlas", type="project", title="Atlas", created="", updated="",
            status="in-progress", owner="Alice", due="2026-08-01", date="2026-07-01",
            summary="Migration project.",
        )
        hit = Hit(doc=_PageDoc(page), score=0.9)
        rendered = _format_existing_context([hit])
        assert "project/atlas" in rendered
        assert "status=in-progress" in rendered
        assert "owner=Alice" in rendered
        assert "due=2026-08-01" in rendered
        assert "date=2026-07-01" in rendered
        assert "summary: Migration project." in rendered

    def test_omits_empty_fields(self):
        page = Page(id="bare", type="topic", title="Bare", created="", updated="")
        hit = Hit(doc=_PageDoc(page), score=0.5)
        rendered = _format_existing_context([hit])
        assert "status=" not in rendered
        assert "owner=" not in rendered
        assert "summary:" not in rendered


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


class TestProcessItemErrorHandling:
    def test_raises_runtime_error_when_llm_response_is_none(self, temp_cfg):
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {"response": None}
        with pytest.raises(RuntimeError, match="LLM call failed"):
            process_item(temp_cfg, mock_llm, "item-err", "Some note.", dry_run=False)


class TestProcessItemEntityEdgeCases:
    def test_unknown_entity_type_falls_back_to_topic(self, temp_cfg):
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [
                    {"type": "not-a-real-type", "name": "Mystery", "aliases": [], "summary": "Unknown kind"}
                ],
                "relations": [], "tasks": [], "decisions": [], "meetings": []
            })
        }
        refs = process_item(temp_cfg, mock_llm, "item-5", "Mystery thing.", dry_run=False)
        assert "topic/mystery" in refs

    def test_entity_with_blank_name_is_skipped(self, temp_cfg):
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [
                    {"type": "topic", "name": "   ", "aliases": [], "summary": "blank"},
                    {"type": "topic", "name": "Real Topic", "aliases": [], "summary": "kept"},
                ],
                "relations": [], "tasks": [], "decisions": [], "meetings": []
            })
        }
        refs = process_item(temp_cfg, mock_llm, "item-6", "Real topic note.", dry_run=False)
        assert refs == ["topic/real-topic"]

    def test_project_status_is_merged(self, temp_cfg):
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [
                    {"type": "project", "name": "Atlas", "aliases": [], "summary": "Migration",
                     "status": "archived"}
                ],
                "relations": [], "tasks": [], "decisions": [], "meetings": []
            })
        }
        process_item(temp_cfg, mock_llm, "item-7", "Atlas has been archived.", dry_run=False)
        loaded = read_page(temp_cfg, "project", "atlas")
        assert loaded.status == "archived"

    def test_meeting_type_entity_merges_date_and_attendees(self, temp_cfg):
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [
                    {"type": "meeting", "name": "Standup", "aliases": [], "summary": "Daily",
                     "date": "2026-08-05", "attendees": ["Alice", "Bob"]}
                ],
                "relations": [], "tasks": [], "decisions": [], "meetings": []
            })
        }
        process_item(temp_cfg, mock_llm, "item-8", "Standup on Aug 5 with Alice and Bob.", dry_run=False)
        loaded = read_page(temp_cfg, "meeting", "standup")
        assert loaded.date == "2026-08-05"
        assert "Alice" in loaded.attendees and "Bob" in loaded.attendees

    def test_action_type_entity_merges_owner_and_due(self, temp_cfg):
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [
                    {"type": "action", "name": "Follow up", "aliases": [], "summary": "Ping client",
                     "owner": "Bob", "due": "2026-08-10", "status": "open"}
                ],
                "relations": [], "tasks": [], "decisions": [], "meetings": []
            })
        }
        process_item(temp_cfg, mock_llm, "item-9", "Follow up owned by Bob due Aug 10.", dry_run=False)
        loaded = read_page(temp_cfg, "action", "follow-up")
        assert loaded.owner == "Bob"
        assert loaded.due == "2026-08-10"

    def test_item_extra_is_merged_into_touched_pages(self, temp_cfg):
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [{"type": "topic", "name": "Tagged", "aliases": [], "summary": "x"}],
                "relations": [], "tasks": [], "decisions": [], "meetings": []
            })
        }
        process_item(
            temp_cfg, mock_llm, "item-10", "Tagged note.",
            dry_run=False, item_extra={"channel": "slack"},
        )
        loaded = read_page(temp_cfg, "topic", "tagged")
        assert loaded.extra.get("channel") == "slack"


class TestProcessItemRelations:
    def test_relation_links_and_notes_both_pages(self, temp_cfg):
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [
                    {"type": "person", "name": "Alice", "aliases": [], "summary": "Eng"},
                    {"type": "project", "name": "Atlas", "aliases": [], "summary": "Migration"},
                ],
                "relations": [
                    {"a": "Alice", "b": "Atlas", "note": "owns the migration"}
                ],
                "tasks": [], "decisions": [], "meetings": []
            })
        }
        process_item(temp_cfg, mock_llm, "item-11", "Alice owns Atlas migration.", dry_run=False)
        alice = read_page(temp_cfg, "person", "alice")
        atlas = read_page(temp_cfg, "project", "atlas")
        assert "project/atlas" in alice.related
        assert "person/alice" in atlas.related
        assert "person/alice" in atlas.backlinks
        assert any("owns the migration" in n for n in alice.notes)
        assert any("owns the migration" in n for n in atlas.notes)

    def test_relation_with_unknown_entity_names_is_ignored(self, temp_cfg):
        """If 'a'/'b' don't match any extracted entity, nothing should blow up."""
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [],
                "relations": [{"a": "Ghost One", "b": "Ghost Two", "note": "unrelated"}],
                "tasks": [], "decisions": [], "meetings": []
            })
        }
        refs = process_item(temp_cfg, mock_llm, "item-12", "Nothing real here.", dry_run=False)
        assert refs == []


class TestProcessItemTasksActionsDecisions:
    def test_task_with_blank_title_is_skipped(self, temp_cfg):
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [], "relations": [],
                "tasks": [{"title": "  ", "owner": "", "due": "", "status": "open"}],
                "decisions": [], "meetings": []
            })
        }
        refs = process_item(temp_cfg, mock_llm, "item-13", "Nothing.", dry_run=False)
        assert refs == []

    def test_action_list_creates_page_and_links_owner(self, temp_cfg):
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [{"type": "person", "name": "Alice", "aliases": [], "summary": "Eng"}],
                "relations": [],
                "tasks": [],
                "actions": [
                    {"title": "Review PR", "owner": "Alice", "due": "2026-08-02", "status": "in_progress"}
                ],
                "decisions": [], "meetings": []
            })
        }
        refs = process_item(temp_cfg, mock_llm, "item-14", "Alice to review PR by Aug 2.", dry_run=False)
        assert "action/review-pr" in refs
        action_page = read_page(temp_cfg, "action", "review-pr")
        assert action_page.owner == "Alice"
        assert action_page.due == "2026-08-02"
        assert action_page.status == "in-progress"
        alice = read_page(temp_cfg, "person", "alice")
        assert "action/review-pr" in alice.backlinks

    def test_action_with_blank_title_is_skipped(self, temp_cfg):
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [], "relations": [], "tasks": [],
                "actions": [{"title": "", "owner": "", "due": "", "status": "open"}],
                "decisions": [], "meetings": []
            })
        }
        refs = process_item(temp_cfg, mock_llm, "item-15", "Nothing.", dry_run=False)
        assert refs == []

    def test_decision_creates_page_and_links_decider(self, temp_cfg):
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [{"type": "person", "name": "Bob", "aliases": [], "summary": "Lead"}],
                "relations": [], "tasks": [], "actions": [],
                "decisions": [
                    {"title": "Adopt Postgres", "description": "Switch DB to Postgres.", "decided_by": "Bob"}
                ],
                "meetings": []
            })
        }
        refs = process_item(temp_cfg, mock_llm, "item-16", "Bob decided to adopt Postgres.", dry_run=False)
        assert "decision/adopt-postgres" in refs
        decision = read_page(temp_cfg, "decision", "adopt-postgres")
        assert decision.status == "decided"
        bob = read_page(temp_cfg, "person", "bob")
        assert "decision/adopt-postgres" in bob.backlinks

    def test_decision_with_blank_title_is_skipped(self, temp_cfg):
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [], "relations": [], "tasks": [], "actions": [],
                "decisions": [{"title": "", "description": "", "decided_by": ""}],
                "meetings": []
            })
        }
        refs = process_item(temp_cfg, mock_llm, "item-17", "Nothing.", dry_run=False)
        assert refs == []

    def test_meeting_with_blank_title_is_skipped(self, temp_cfg):
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [], "relations": [], "tasks": [], "actions": [], "decisions": [],
                "meetings": [{"title": "", "date": "", "attendees": [], "agenda": ""}]
            })
        }
        refs = process_item(temp_cfg, mock_llm, "item-18", "Nothing.", dry_run=False)
        assert refs == []

    def test_event_creates_page_with_location(self, temp_cfg):
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [], "relations": [], "tasks": [], "actions": [], "decisions": [], "meetings": [],
                "events": [
                    {"title": "Conference", "date": "2026-09-01",
                     "description": "Annual tech conference.", "location": "SF"}
                ]
            })
        }
        refs = process_item(temp_cfg, mock_llm, "item-19", "Conference in SF on Sep 1.", dry_run=False)
        assert "event/conference" in refs
        event = read_page(temp_cfg, "event", "conference")
        assert event.date == "2026-09-01"
        assert "SF" in event.notes[0]

    def test_event_with_blank_title_is_skipped(self, temp_cfg):
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [], "relations": [], "tasks": [], "actions": [], "decisions": [], "meetings": [],
                "events": [{"title": "", "date": "", "description": "", "location": ""}]
            })
        }
        refs = process_item(temp_cfg, mock_llm, "item-20", "Nothing.", dry_run=False)
        assert refs == []


class TestOrganizeAllErrorHandling:
    def test_error_in_one_item_is_captured_and_others_still_process(self, temp_cfg):
        write_inbox_item(temp_cfg, "This one will fail.")
        write_inbox_item(temp_cfg, "This one will succeed.")

        mock_llm = MagicMock()
        mock_llm.llm_call.side_effect = [
            {"response": None},  # triggers RuntimeError inside process_item
            {
                "response": json.dumps({
                    "entities": [{"type": "topic", "name": "OK", "aliases": [], "summary": "fine"}],
                    "relations": [], "tasks": [], "decisions": [], "meetings": []
                })
            },
        ]
        report = organize_all(temp_cfg, mock_llm, dry_run=False)
        assert len(report["errors"]) == 1
        assert len(report["processed"]) == 1

    def test_summary_refresh_exception_does_not_fail_organize(self, temp_cfg, monkeypatch):
        existing = Page(
            id="bigproject", type="topic", title="BigProject",
            created=now_iso(), updated=now_iso(), summary="Old",
            notes=["- (1) a.", "- (2) b."],
        )
        write_page(temp_cfg, existing)
        write_inbox_item(temp_cfg, "BigProject update.")

        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = {
            "response": json.dumps({
                "entities": [{"type": "topic", "name": "BigProject", "aliases": [], "summary": "A project"}],
                "relations": [], "tasks": [], "decisions": [], "meetings": []
            })
        }

        monkeypatch.setattr(
            "graybox.organizer.refresh_page_summary",
            MagicMock(side_effect=RuntimeError("boom")),
        )

        report = organize_all(temp_cfg, mock_llm, dry_run=False)
        assert report["errors"] == []
        assert report["summaries_refreshed"] == []


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


class TestAppendFollowUpReachesKnowledgeLayer:
    """Regression: a follow-up appended to an already-organized item must
    reach the wiki. append_inbox_item writes a brand-new inbox item linked
    to the original, so the next organize run picks it up — instead of
    silently dropping it because the original item id is already marked
    processed."""

    def test_append_after_organize_is_picked_up_by_next_organize(self, temp_cfg):
        from graybox.capture import capture
        from graybox.storage import append_inbox_item, read_page

        def llm_extract(entities):
            return {
                "response": json.dumps({
                    "entities": entities,
                    "relations": [],
                    "tasks": [],
                    "actions": [],
                    "decisions": [],
                    "meetings": [],
                })
            }

        mock_llm = MagicMock()
        original = capture(temp_cfg, "Alice started the Atlas migration.")

        mock_llm.llm_call.return_value = llm_extract([
            {"type": "person", "name": "Alice", "aliases": [], "summary": "Engineer"}
        ])
        report1 = organize_all(temp_cfg, mock_llm, dry_run=False)
        assert len(report1["processed"]) == 1

        page = read_page(temp_cfg, "person", "alice")
        assert page is not None
        assert len(page.notes) == 1

        follow_up = append_inbox_item(temp_cfg, original.id, "Alice shipped the migration.")

        mock_llm.llm_call.return_value = llm_extract([
            {"type": "person", "name": "Alice", "aliases": [], "summary": "Engineer"}
        ])
        report2 = organize_all(temp_cfg, mock_llm, dry_run=False)
        assert any(i["item"] == follow_up.id for i in report2["processed"]), report2

        page = read_page(temp_cfg, "person", "alice")
        assert page is not None
        assert len(page.notes) == 2
        assert any("shipped" in n for n in page.notes)
