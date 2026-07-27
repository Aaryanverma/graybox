"""Tests for search.py — search_all() and find_duplicates()."""
from __future__ import annotations

from graybox.models import Page, InboxItem, now_iso
from graybox.search import search_all, find_duplicates
from graybox.storage import write_page, write_inbox_item


class TestSearchAll:
    def test_finds_wiki_page_over_inbox(self, temp_cfg):
        write_page(temp_cfg, Page(
            id="atlas", type="project", title="Atlas Migration",
            created=now_iso(), updated=now_iso(), summary="Migrating the Atlas database",
        ))
        write_inbox_item(temp_cfg, "Random unrelated note about lunch")

        wiki_hits, inbox_hits = search_all(temp_cfg, "atlas migration", top_k=5)
        assert len(wiki_hits) == 1
        assert wiki_hits[0].doc.search_id == "project/atlas"

    def test_falls_back_to_inbox_when_no_wiki_match(self, temp_cfg):
        write_inbox_item(temp_cfg, "Talked about the quarterly roadmap today")
        wiki_hits, inbox_hits = search_all(temp_cfg, "quarterly roadmap", top_k=5)
        assert wiki_hits == []
        assert len(inbox_hits) == 1

    def test_respects_min_score_threshold(self, temp_cfg):
        write_page(temp_cfg, Page(
            id="x", type="topic", title="Something Else Entirely",
            created=now_iso(), updated=now_iso(), summary="Nothing related",
        ))
        wiki_hits, _ = search_all(temp_cfg, "completely unrelated query terms", top_k=5, min_score=0.4)
        assert wiki_hits == []

    def test_top_k_limits_results(self, temp_cfg):
        for i in range(5):
            write_page(temp_cfg, Page(
                id=f"proj{i}", type="project", title=f"Atlas Project {i}",
                created=now_iso(), updated=now_iso(), summary="Atlas migration work",
            ))
        wiki_hits, _ = search_all(temp_cfg, "atlas migration", top_k=2)
        assert len(wiki_hits) <= 2

    def test_all_workspaces_flag_searches_other_workspaces(self, temp_cfg):
        write_page(temp_cfg, Page(
            id="here", type="project", title="Local Project",
            created=now_iso(), updated=now_iso(), summary="A project in this workspace",
        ))
        other_ws = temp_cfg.workspace_manager.create("other")
        other_cfg = temp_cfg.for_workspace(other_ws)
        write_page(other_cfg, Page(
            id="there", type="project", title="Remote Project",
            created=now_iso(), updated=now_iso(), summary="A project in the other workspace",
        ))

        wiki_hits, _ = search_all(temp_cfg, "project", top_k=10, all_workspaces=True)
        refs = {h.doc.search_id for h in wiki_hits}
        assert "project/here" in refs
        assert "project/there" in refs


class TestFindDuplicates:
    def test_flags_similar_names_same_type(self, temp_cfg):
        write_page(temp_cfg, Page(id="alice", type="person", title="Alice Smith", created=now_iso(), updated=now_iso()))
        write_page(temp_cfg, Page(id="alicee", type="person", title="Alicee Smith", created=now_iso(), updated=now_iso()))
        dupes = find_duplicates(temp_cfg, threshold=0.85)
        assert len(dupes) >= 1
        pair = dupes[0]
        assert pair[2] >= 0.85
        assert pair[3] == "fuzzy name match"

    def test_ignores_different_types(self, temp_cfg):
        write_page(temp_cfg, Page(id="alice", type="person", title="Alice", created=now_iso(), updated=now_iso()))
        write_page(temp_cfg, Page(id="alice-task", type="task", title="Alice", created=now_iso(), updated=now_iso()))
        dupes = find_duplicates(temp_cfg, threshold=0.85)
        assert dupes == []

    def test_no_dupes_below_threshold(self, temp_cfg):
        write_page(temp_cfg, Page(id="alice", type="person", title="Alice", created=now_iso(), updated=now_iso()))
        write_page(temp_cfg, Page(id="bob", type="person", title="Bob", created=now_iso(), updated=now_iso()))
        dupes = find_duplicates(temp_cfg, threshold=0.85)
        assert dupes == []

    def test_restricts_to_page_type(self, temp_cfg):
        write_page(temp_cfg, Page(id="alice", type="person", title="Alice", created=now_iso(), updated=now_iso()))
        write_page(temp_cfg, Page(id="alicee", type="person", title="Alicee", created=now_iso(), updated=now_iso()))
        write_page(temp_cfg, Page(id="task-alice", type="task", title="Alice", created=now_iso(), updated=now_iso()))
        dupes = find_duplicates(temp_cfg, page_type="task", threshold=0.85)
        assert dupes == []