"""
Configuration loading for GRAY BOX.

Precedence (highest wins): environment variables > config.yaml > defaults.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
import yaml

from graybox.workspace import Workspace, WorkspaceManager


DEFAULTS = {
    "root": "",
    "default_workspace": "personal",
    "active_workspace": "",
    "env_file": "",
    "llm": {
        "model_name": "gpt-5.6",
        "base_url": "",
        "temperature": 0.0,
    },
    "auto_refresh_summaries": True,
    "retrieval": {
        "top_k": 5,
        # Both thresholds below are normalized to [0, 1] on the SAME scale:
        # closer to 1.0 always means "more similar / more confident match",
        # closer to 0.0 always means "weak or no match". They differ only in
        # what they threshold, not in how to read the number.
        #
        # min_score: how relevant a wiki/inbox page must be to answer a
        #   question (Engine.coverage_scorer). A strong true match on a
        #   focused question typically lands ~0.6-1.0; ~0.35-0.5 is a
        #   reasonable "loosely relevant" floor.
        "min_score": 0.4,
        # dedup_threshold: how similar two names must be (Engine.name_scorer,
        # difflib-based) before the Organizer treats them as the same
        # real-world entity (or `dupes` flags them as likely duplicates).
        # This wants to be high - typos/nickname variants of the same name
        # score ~0.85-1.0; unrelated names score well below that.
        "dedup_threshold": 0.85,
    },
    # Optional - only used if you opt into embedding-based search.
    "embeddings": {
        "enabled": False,
        "model_name": "text-embedding-3-small",
        "base_url": "",
        "api_base": "",
        "input_type": "",
    },
}

def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class LLMConfig:
    model_name: str
    base_url: str
    temperature: float
    api_key: str = ""
    api_base: str = ""
    azure_ad_token: str = ""
    api_version: str = ""
    deployment_id: str = ""
    api_type: str = "chat_completion"
    max_tokens: int = 1024
    max_completion_tokens: int = 1024
    top_p: float = 0.95
    top_k: int = 40
    kwargs: dict = field(default_factory=dict)


@dataclass
class RetrievalConfig:
    top_k: int
    min_score: float
    dedup_threshold: float = 0.85
    semantic_min_score: float = 0.15


@dataclass
class EmbeddingsConfig:
    enabled: bool = False
    model_name: str = "text-embedding-3-small"
    api_key: str = ""
    base_url: str = ""
    api_base: str = ""
    input_type: str = ""
    azure_ad_token: str = ""
    api_version: str = ""
    deployment_id: str = ""
    kwargs: dict = field(default_factory=dict)


@dataclass
class Config:
    root: Path
    workspace_manager: WorkspaceManager
    llm: LLMConfig
    retrieval: RetrievalConfig
    embeddings: EmbeddingsConfig
    auto_refresh_summaries: bool = True
    raw: dict = field(default_factory=dict)
    config_path: Path | None = None

    @property
    def workspace(self) -> Path:
        return self.workspace_manager.current().root

    @property
    def workspace_id(self) -> str:
        return self.workspace_manager.current().id

    @property
    def workspace_name(self) -> str:
        return self.workspace_manager.current().name

    @property
    def workspace_description(self) -> str:
        return self.workspace_manager.current().description

    @property
    def inbox_dir(self) -> Path:
        return self.workspace / "inbox"

    @property
    def wiki_dir(self) -> Path:
        return self.workspace / "wiki"

    @property
    def state_dir(self) -> Path:
        return self.workspace / ".state"

    @property
    def attachments_dir(self) -> Path:
        return self.workspace / "attachments"

    @property
    def metadata_path(self) -> Path:
        return self.workspace / "metadata.json"

    @property
    def wiki_db_path(self) -> Path:
        return self.workspace / "wiki.db"

    @property
    def graph_db_path(self) -> Path:
        return self.workspace / "graph.db"

    @property
    def vectors_db_path(self) -> Path:
        return self.workspace / "vectors.db"

    @property
    def exports_dir(self) -> Path:
        return self.workspace / "exports"

    def for_workspace(self, workspace: Workspace | str) -> "Config":
        if isinstance(workspace, str):
            ws = self.workspace_manager.resolve(workspace)
        else:
            ws = workspace
        manager = WorkspaceManager(
            root=self.root,
            active_workspace=ws.id,
            default_workspace=self.raw.get("default_workspace", "personal"),
            config_path=self.config_path,
            app_config=copy.deepcopy(self.raw),
        )
        manager._cache[ws.id] = ws  # type: ignore[attr-defined]
        manager._active_workspace = ws.id  # type: ignore[attr-defined]
        return Config(
            root=self.root,
            workspace_manager=manager,
            llm=self.llm,
            retrieval=self.retrieval,
            embeddings=self.embeddings,
            raw=copy.deepcopy(self.raw),
            config_path=self.config_path,
        )

    def with_workspace_id(self, workspace_id: str) -> "Config":
        return self.for_workspace(workspace_id)


def _candidate_config_paths(path: str | None = None) -> list[Path]:
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    env_path = os.environ.get("GRAYBOX_CONFIG")
    if env_path and (not path or Path(env_path) != Path(path)):
        candidates.append(Path(env_path))
    local_default = Path("config.yaml")
    cwd_default = Path.cwd() / ".graybox" / "config.yaml"
    if local_default not in candidates:
        candidates.append(local_default)
    if cwd_default not in candidates:
        candidates.append(cwd_default)
    # A stray ~/.graybox/config.yaml on a dev/CI machine shouldn't silently
    # override project-local config, so this fallback is opt-in only.
    if os.environ.get("GRAYBOX_USE_GLOBAL_CONFIG"):
        home_default = Path.home() / ".graybox" / "config.yaml"
        if home_default not in candidates:
            candidates.append(home_default)
    return candidates


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_config(path: str | None = None) -> Config:
    cfg = dict(DEFAULTS)

    candidate = None
    for p in _candidate_config_paths(path):
        if p.exists():
            candidate = p
            break
    if candidate is None:
        candidate = _candidate_config_paths(path)[0]

    user_cfg = _load_yaml(candidate)
    cfg = _deep_merge(cfg, user_cfg)

    if cfg.get("env_file"):
        env_path = Path(cfg["env_file"]).expanduser()

        if not env_path.is_absolute():
            env_path = (candidate.parent / env_path).resolve()

        load_dotenv(env_path, override=True)

    env_map = {
        "GRAYBOX_ROOT": ("root",),
        "GRAYBOX_WORKSPACE": ("root",),  # legacy alias for the app root
        "GRAYBOX_ACTIVE_WORKSPACE": ("active_workspace",),
        "GRAYBOX_DEFAULT_WORKSPACE": ("default_workspace",),
        "GRAYBOX_LLM_MODEL": ("llm", "model_name"),
        "GRAYBOX_LLM_BASE_URL": ("llm", "base_url"),
        "GRAYBOX_LLM_API_KEY": ("llm", "api_key"),
        "GRAYBOX_TEMPERATURE": ("llm", "temperature"),
        "GRAYBOX_TOP_K": ("retrieval", "top_k"),
        "GRAYBOX_MIN_SCORE": ("retrieval", "min_score"),
        "GRAYBOX_DEDUP_THRESHOLD": ("retrieval", "dedup_threshold"),
        "GRAYBOX_SEMANTIC_MIN_SCORE": ("retrieval", "semantic_min_score"),
    }
    for env_var, path_keys in env_map.items():
        if env_var in os.environ:
            val = os.environ[env_var]
            node = cfg
            for key in path_keys[:-1]:
                node = node[key]
            last = path_keys[-1]
            if last in ("temperature", "min_score", "dedup_threshold", "semantic_min_score"):
                val = float(val)
            elif last in ("top_k",):
                val = int(val)
            node[last] = val

    if cfg.get("workspace") and not cfg.get("root"):
        cfg["root"] = cfg["workspace"]
    cfg.pop("workspace", None)
    if not cfg.get("root"):
        cfg["root"] = str(Path.cwd() / ".graybox")
    if not cfg.get("default_workspace"):
        cfg["default_workspace"] = "personal"

    root = Path(cfg["root"]).expanduser().resolve()
    manager = WorkspaceManager(
        root=root,
        active_workspace=cfg.get("active_workspace") or None,
        default_workspace=cfg.get("default_workspace", "personal"),
        config_path=candidate,
        app_config=copy.deepcopy(cfg),
    )

    manager.ensure_workspace(cfg.get("active_workspace") or None)

    return Config(
        root=root,
        workspace_manager=manager,
        llm=LLMConfig(**cfg["llm"]),
        retrieval=RetrievalConfig(**cfg["retrieval"]),
        embeddings=EmbeddingsConfig(**cfg.get("embeddings", {})),
        auto_refresh_summaries=cfg.get("auto_refresh_summaries", True),
        raw=cfg,
        config_path=candidate,
    )