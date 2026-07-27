"""Smoke tests for dashboard.py — read-only HTML/data generation."""
from __future__ import annotations

import json

from graybox.dashboard import build_dashboard_data, build_dashboard_html, write_dashboard
from graybox.models import Page, now_iso
from graybox.storage import write_page, write_inbox_item, list_pages, list_inbox_items


class TestBuildDashboardData:
    def test_empty_workspace_has_zero_counts(self, temp_cfg):
        data = build_dashboard_data(temp_cfg)
        assert data["summary"]["total_pages"] == 0
        assert data["summary"]["task_total"] == 0
        assert data["pages"] == []

    def test_counts_pages_and_tasks(self, temp_cfg):
        write_page(temp_cfg, Page(
            id="t1", type="task", title="Ship feature", created=now_iso(), updated=now_iso(),
            status="open", owner="Alice", due="2026-08-01",
        ))
        write_page(temp_cfg, Page(
            id="p1", type="project", title="Atlas", created=now_iso(), updated=now_iso(),
        ))
        write_inbox_item(temp_cfg, "A raw capture")

        data = build_dashboard_data(temp_cfg)
        assert data["summary"]["total_pages"] == 2
        assert data["summary"]["task_total"] == 1
        assert data["summary"]["project_total"] == 1
        assert data["summary"]["inbox_total"] == 1
        assert data["summary"]["open_tasks"] == 1

    def test_graph_nodes_and_edges_from_related_pages(self, temp_cfg):
        write_page(temp_cfg, Page(
            id="a", type="topic", title="A", created=now_iso(), updated=now_iso(), related=["topic/b"],
        ))
        write_page(temp_cfg, Page(
            id="b", type="topic", title="B", created=now_iso(), updated=now_iso(), backlinks=["topic/a"],
        ))
        data = build_dashboard_data(temp_cfg)
        edges = data["graph"]["edges"]
        assert any({e["source"], e["target"]} == {"topic/a", "topic/b"} for e in edges)

    def test_never_writes_to_wiki_or_inbox(self, temp_cfg):
        write_page(temp_cfg, Page(id="x", type="topic", title="X", created=now_iso(), updated=now_iso()))
        before_pages = len(list_pages(temp_cfg))
        before_inbox = len(list_inbox_items(temp_cfg))
        build_dashboard_data(temp_cfg)
        assert len(list_pages(temp_cfg)) == before_pages
        assert len(list_inbox_items(temp_cfg)) == before_inbox


class TestBuildDashboardHtml:
    def test_produces_valid_html_with_embedded_json(self, temp_cfg):
        write_page(temp_cfg, Page(id="x", type="topic", title="X", created=now_iso(), updated=now_iso()))
        html = build_dashboard_html(temp_cfg)
        assert html.startswith("<!doctype html>")
        assert "Gray Box Dashboard" in html
        # embedded JSON payload must parse
        start = html.index('id=\'data\' type=\'application/json\'>') + len("id='data' type='application/json'>")
        end = html.index("</script>", start)
        payload = html[start:end]
        parsed = json.loads(payload)
        assert "pages" in parsed


class TestWriteDashboard:
    def test_writes_html_file_to_exports_dir(self, temp_cfg):
        path = write_dashboard(temp_cfg)
        assert path.exists()
        assert path.name == "dashboard.html"
        assert path.parent == temp_cfg.workspace / "exports"