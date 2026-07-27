"""Pytest fixtures for Gray Box test suite."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

# Ensure graybox is importable
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from graybox.config import Config, LLMConfig, RetrievalConfig, EmbeddingsConfig
from graybox.workspace import WorkspaceManager
from graybox.storage import ensure_workspace


@pytest.fixture
def temp_cfg():
    """Yield a fully configured temporary workspace."""
    tmp = tempfile.mkdtemp()
    root = Path(tmp) / ".graybox"
    manager = WorkspaceManager(
        root=root,
        active_workspace="test",
        default_workspace="test",
        config_path=None,
        app_config={},
    )
    cfg = Config(
        root=root,
        workspace_manager=manager,
        llm=LLMConfig(model_name="test-model", base_url="", temperature=0.0),
        retrieval=RetrievalConfig(top_k=5, min_score=0.4, dedup_threshold=0.85),
        embeddings=EmbeddingsConfig(),
    )
    ensure_workspace(cfg)
    yield cfg
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def temp_cfg_low_threshold(temp_cfg):
    """Same as temp_cfg but with a lower dedup threshold for fuzzy tests."""
    temp_cfg.retrieval.dedup_threshold = 0.6
    return temp_cfg