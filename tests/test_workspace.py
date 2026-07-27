"""Tests for workspace.py — multi-workspace registry and switching."""
from __future__ import annotations

import pytest

from graybox.workspace import WorkspaceManager, _slugify, workspace_context_block


class TestSlugify:
    def test_basic(self):
        assert _slugify("Work Stuff") == "work-stuff"

    def test_empty_defaults_to_untitled(self):
        assert _slugify("") == "untitled"

    def test_strips_punctuation(self):
        assert _slugify("Q3!! Report??") == "q3-report"


@pytest.fixture
def manager(tmp_path):
    root = tmp_path / ".graybox"
    return WorkspaceManager(root=root, active_workspace="personal", default_workspace="personal")


class TestWorkspaceManager:
    def test_current_creates_default_workspace(self, manager):
        ws = manager.current()
        assert ws.id == "personal"
        assert ws.inbox_dir.exists()
        assert ws.wiki_dir.exists()

    def test_create_new_workspace(self, manager):
        ws = manager.create("Work", description="Day job")
        assert ws.id == "work"
        assert ws.name == "Work"
        assert ws.description == "Day job"
        assert ws.root.exists()

    def test_switch_changes_current(self, manager):
        manager.create("Work")
        manager.switch("work")
        assert manager.current().id == "work"

    def test_switch_persists_across_new_manager(self, tmp_path):
        root = tmp_path / ".graybox"
        cfg_path = tmp_path / "config.yaml"
        m1 = WorkspaceManager(root=root, active_workspace="personal", default_workspace="personal", config_path=cfg_path)
        m1.current()
        m1.create("Work")
        m1.switch("work")

        m2 = WorkspaceManager(root=root, default_workspace="personal", config_path=cfg_path)
        assert m2.current().id == "work"

    def test_list_includes_all_workspaces(self, manager):
        manager.current()
        manager.create("Work")
        manager.create("Side Project")
        ids = {ws.id for ws in manager.list()}
        assert {"personal", "work", "side-project"} <= ids

    def test_resolve_by_name_case_insensitive(self, manager):
        manager.create("Work")
        ws = manager.resolve("WORK")
        assert ws.id == "work"

    def test_resolve_unknown_raises(self, manager):
        with pytest.raises(ValueError, match="Workspace not found"):
            manager.resolve("does-not-exist")

    def test_create_with_custom_path(self, manager, tmp_path):
        custom = tmp_path / "custom_location"
        ws = manager.create("Custom", path=custom)
        assert ws.root == custom.resolve()
        assert (custom / "inbox").exists()

    def test_isolation_between_workspaces(self, manager):
        from graybox.config import Config, LLMConfig, RetrievalConfig, EmbeddingsConfig
        from graybox.storage import write_inbox_item, list_inbox_items

        manager.current()
        manager.create("Work")

        cfg = Config(
            root=manager.root,
            workspace_manager=manager,
            llm=LLMConfig(model_name="test", base_url="", temperature=0.0),
            retrieval=RetrievalConfig(top_k=5, min_score=0.4, dedup_threshold=0.85),
            embeddings=EmbeddingsConfig(),
        )
        manager.switch("personal")
        write_inbox_item(cfg, "Personal note")

        manager.switch("work")
        assert list_inbox_items(cfg) == []

    def test_context_block_includes_name_and_id(self, manager):
        from graybox.config import Config, LLMConfig, RetrievalConfig, EmbeddingsConfig

        cfg = Config(
            root=manager.root,
            workspace_manager=manager,
            llm=LLMConfig(model_name="test", base_url="", temperature=0.0),
            retrieval=RetrievalConfig(top_k=5, min_score=0.4, dedup_threshold=0.85),
            embeddings=EmbeddingsConfig(),
        )
        block = workspace_context_block(cfg)
        assert "Workspace:" in block
        assert "personal" in block