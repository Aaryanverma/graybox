"""Tests for config.py — env-var > yaml > defaults precedence."""
from __future__ import annotations

import os
import textwrap

import pytest
from pathlib import Path
from graybox.config import load_config, _deep_merge, DEFAULTS

@pytest.fixture
def mock_home(monkeypatch, tmp_path):
    """Mocks Path.home() to a temporary directory to isolate tests from the user's real home config."""
    home_dir = tmp_path / "mock_home"
    home_dir.mkdir()
    
    # Mock pathlib.Path.home
    def mock_home_return(*args, **kwargs):
        return home_dir
    monkeypatch.setattr(Path, "home", mock_home_return)
    
    # Also mock the environment variable for completeness
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("USERPROFILE", str(home_dir)) # Windows fallback
    
    return home_dir

class TestDeepMerge:
    def test_override_wins(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}}
        merged = _deep_merge(base, override)
        assert merged == {"a": 1, "b": {"c": 99, "d": 3}}

    def test_non_dict_override_replaces(self):
        base = {"a": {"x": 1}}
        override = {"a": 5}
        assert _deep_merge(base, override) == {"a": 5}


@pytest.fixture
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("GRAYBOX_"):
            monkeypatch.delenv(key, raising=False)
    yield monkeypatch


class TestLoadConfig:
    def test_defaults_used_when_no_file(self, tmp_path, clean_env, mock_home, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = load_config(str(tmp_path / "nope.yaml"))
        assert cfg.llm.model_name == DEFAULTS["llm"]["model_name"]
        assert cfg.retrieval.top_k == 5
        assert cfg.retrieval.min_score == 0.4

    def test_yaml_overrides_defaults(self, tmp_path, clean_env, mock_home, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(textwrap.dedent("""
            root: ./.graybox
            default_workspace: personal
            llm:
              model_name: custom/model-x
              temperature: 0.7
            retrieval:
              top_k: 8
              min_score: 0.6
        """), encoding="utf-8")
        cfg = load_config(str(cfg_path))
        assert cfg.llm.model_name == "custom/model-x"
        assert cfg.llm.temperature == 0.7
        assert cfg.retrieval.top_k == 8
        assert cfg.retrieval.min_score == 0.6

    def test_env_var_overrides_yaml(self, tmp_path, clean_env, mock_home, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(textwrap.dedent("""
            root: ./.graybox
            default_workspace: personal
            llm:
              model_name: from-yaml
        """), encoding="utf-8")
        monkeypatch.setenv("GRAYBOX_LLM_MODEL", "from-env")
        monkeypatch.setenv("GRAYBOX_TOP_K", "12")
        cfg = load_config(str(cfg_path))
        assert cfg.llm.model_name == "from-env"
        assert cfg.retrieval.top_k == 12

    def test_root_defaults_to_cwd_graybox(self, tmp_path, clean_env, mock_home, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = load_config(str(tmp_path / "nope.yaml"))
        assert cfg.root == (tmp_path / ".graybox").resolve()

    def test_for_workspace_switches_without_mutating_original(self, tmp_path, clean_env, mock_home, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = load_config(str(tmp_path / "nope.yaml"))
        original_id = cfg.workspace_id
        other_ws = cfg.workspace_manager.create("Work")
        switched = cfg.for_workspace(other_ws)
        assert switched.workspace_id == "work"
        assert cfg.workspace_id == original_id


    def test_home_default_is_used_when_present(self, tmp_path, clean_env, mock_home, monkeypatch):
        # Setup: Move out of the home directory
        monkeypatch.chdir(tmp_path)
        
        # Create a config in the mocked home directory (~/.graybox/config.yaml)
        graybox_dir = mock_home / ".graybox"
        graybox_dir.mkdir()
        cfg_path = graybox_dir / "config.yaml"
        
        cfg_path.write_text(textwrap.dedent("""
            llm:
              model_name: home-dir-model
            retrieval:
              top_k: 42
        """), encoding="utf-8")
        
        # Call load_config without providing an explicit path
        cfg = load_config()
        
        # Assert the home configuration was loaded
        assert cfg.llm.model_name == "home-dir-model"
        assert cfg.retrieval.top_k == 42