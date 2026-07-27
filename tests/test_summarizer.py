"""Tests for summarizer.py — page re-summarization from notes."""
from __future__ import annotations

from unittest.mock import MagicMock

from graybox.summarizer import refresh_page_summary, refresh_all_summaries
from graybox.models import Page, now_iso
from graybox.storage import read_page, write_page


class TestRefreshPageSummary:
    def test_skips_page_with_no_notes(self, temp_cfg):
        page = Page(id="empty", type="topic", title="Empty", created=now_iso(), updated=now_iso(), summary="Old")
        write_page(temp_cfg, page)
        llm = MagicMock()
        result = refresh_page_summary(temp_cfg, llm, page)
        assert result is None
        llm.llm_call.assert_not_called()

    def test_skips_page_with_few_notes_and_existing_summary(self, temp_cfg):
        page = Page(
            id="small", type="topic", title="Small",
            created=now_iso(), updated=now_iso(),
            summary="Already good",
            notes=["- Note one"],
        )
        write_page(temp_cfg, page)
        llm = MagicMock()
        result = refresh_page_summary(temp_cfg, llm, page)
        assert result is None
        llm.llm_call.assert_not_called()

    def test_refreshes_when_many_notes(self, temp_cfg):
        page = Page(
            id="big", type="topic", title="Big Project",
            created=now_iso(), updated=now_iso(),
            summary="Old summary",
            notes=[
                "- (2026-01-01) Started project.",
                "- (2026-02-01) Hired Alice.",
                "- (2026-03-01) Shipped v1.",
                "- (2026-04-01) Pivoted to B2B.",
            ],
        )
        write_page(temp_cfg, page)

        llm = MagicMock()
        llm.llm_call.return_value = {
            "response": "B2B project started in Jan, shipped v1 in March, pivoted in April.",
            "cost": 0.001,
        }

        result = refresh_page_summary(temp_cfg, llm, page, dry_run=False)
        assert result is not None
        assert result.ref == "topic/big"
        assert result.old_summary == "Old summary"
        assert "B2B" in result.new_summary

        loaded = read_page(temp_cfg, "topic", "big")
        assert loaded.summary == "B2B project started in Jan, shipped v1 in March, pivoted in April."
        assert loaded.summary_refreshed_at != ''

    def test_dry_run_does_not_write(self, temp_cfg):
        page = Page(
            id="dry", type="topic", title="Dry",
            created=now_iso(), updated=now_iso(),
            summary="Old",
            notes=["- One", "- Two", "- Three"],
        )
        write_page(temp_cfg, page)

        llm = MagicMock()
        llm.llm_call.return_value = {"response": "New summary", "cost": 0.0}

        result = refresh_page_summary(temp_cfg, llm, page, dry_run=True)
        assert result is not None
        assert result.new_summary == "New summary"

        loaded = read_page(temp_cfg, "topic", "dry")
        assert loaded.summary == "Old"  # unchanged

    def test_handles_llm_failure_gracefully(self, temp_cfg):
        page = Page(
            id="fail", type="topic", title="Fail",
            created=now_iso(), updated=now_iso(),
            summary="Old",
            notes=["- One", "- Two", "- Three"],
        )
        write_page(temp_cfg, page)

        llm = MagicMock()
        llm.llm_call.return_value = {"response": None, "error": "timeout"}

        result = refresh_page_summary(temp_cfg, llm, page)
        assert result is None


class TestRefreshAllSummaries:
    def test_filters_by_page_type(self, temp_cfg):
        person = Page(id="alice", type="person", title="Alice", created=now_iso(), updated=now_iso(),
                      summary="Old", notes=["- N1", "- N2", "- N3"])
        task = Page(id="task1", type="task", title="Task 1", created=now_iso(), updated=now_iso(),
                    summary="Old", notes=["- N1", "- N2", "- N3"])
        write_page(temp_cfg, person)
        write_page(temp_cfg, task)

        llm = MagicMock()
        llm.llm_call.return_value = {"response": "Updated", "cost": 0.001}

        report = refresh_all_summaries(temp_cfg, llm, page_type="person", min_notes=1)
        assert len(report["refreshed"]) == 1
        assert report["refreshed"][0].ref == "person/alice"
        assert report["skipped"] == 1  # task skipped

    def test_respects_min_notes(self, temp_cfg):
        page = Page(id="min", type="topic", title="Min", created=now_iso(), updated=now_iso(),
                    summary="Old", notes=["- Only one note"])
        write_page(temp_cfg, page)

        llm = MagicMock()
        report = refresh_all_summaries(temp_cfg, llm, min_notes=3)
        assert len(report["refreshed"]) == 0
        assert report["skipped"] == 1