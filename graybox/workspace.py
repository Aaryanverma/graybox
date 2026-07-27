from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
import json
import re
import shutil

import yaml


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _slugify(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "untitled"


def _safe_read_structured(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass
    try:
        data = yaml.safe_load(text) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@dataclass
class Workspace:
    id: str
    name: str
    description: str
    created_at: str
    last_used: str
    root: Path
    tags: list[str] = field(default_factory=list)
    registry_root: Path | None = None

    @property
    def path(self) -> Path:
        return self.root

    @property
    def metadata_path(self) -> Path:
        return self.root / "metadata.json"

    @property
    def registry_metadata_path(self) -> Path:
        if self.registry_root is None or self.registry_root == self.root:
            return self.metadata_path
        return self.registry_root / "metadata.json"

    @property
    def inbox_dir(self) -> Path:
        return self.root / "inbox"

    @property
    def wiki_dir(self) -> Path:
        return self.root / "wiki"

    @property
    def state_dir(self) -> Path:
        return self.root / ".state"

    @property
    def attachments_dir(self) -> Path:
        return self.root / "attachments"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def wiki_db_path(self) -> Path:
        return self.root / "wiki.db"

    @property
    def graph_db_path(self) -> Path:
        return self.root / "graph.db"

    @property
    def vectors_db_path(self) -> Path:
        return self.root / "vectors.db"

    def metadata(self) -> dict:
        data = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "tags": self.tags,
            "root": str(self.root),
        }
        if self.registry_root is not None and self.registry_root != self.root:
            data["registry_root"] = str(self.registry_root)
        return data

    def ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.attachments_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        # self.wiki_db_path.touch(exist_ok=True)
        # self.graph_db_path.touch(exist_ok=True)
        # self.vectors_db_path.touch(exist_ok=True)
        self.write_metadata()

    def write_metadata(self) -> None:
        payload = yaml.safe_dump(self.metadata(), sort_keys=False, allow_unicode=True)
        self.metadata_path.write_text(payload, encoding="utf-8")
        if self.registry_root is not None and self.registry_root != self.root:
            self.registry_root.mkdir(parents=True, exist_ok=True)
            self.registry_metadata_path.write_text(payload, encoding="utf-8")

    @classmethod
    def from_metadata_file(cls, path: Path) -> "Workspace":
        data = _safe_read_structured(path)
        root_value = data.get("root") or data.get("path") or path.parent
        root = Path(root_value).expanduser().resolve()
        registry_root = path.parent.resolve()
        if registry_root == root:
            registry_root = None
        ws_id = str(data.get("id") or root.name)
        name = str(data.get("name") or ws_id.replace("-", " ").title())
        description = str(data.get("description") or "")
        created_at = str(data.get("created_at") or now_iso())
        last_used = str(data.get("last_used") or created_at)
        tags = list(data.get("tags") or [])
        return cls(
            id=ws_id,
            name=name,
            description=description,
            created_at=created_at,
            last_used=last_used,
            root=root,
            tags=tags,
            registry_root=registry_root,
        )


class WorkspaceManager:
    """
    Central workspace registry and selector.

    The rest of the application should only ever interact with the current
    workspace through Config.workspace / Config.workspace_manager.
    """

    def __init__(
        self,
        root: Path,
        *,
        active_workspace: str | None = None,
        default_workspace: str = "personal",
        config_path: Path | None = None,
        app_config: dict | None = None,
    ) -> None:
        # Root is the app directory, e.g. <current terminal dir>/.graybox
        self.root = Path(root).expanduser().resolve()
        self.workspaces_root = self.root / "workspaces"
        self.default_workspace = _slugify(default_workspace)
        self._active_workspace = _slugify(active_workspace) if active_workspace else None
        self.config_path = Path(config_path).expanduser().resolve() if config_path else None
        self.app_config = dict(app_config or {})
        self._cache: dict[str, Workspace] = {}
        self._bootstrap_legacy_layout()

    def _bootstrap_legacy_layout(self) -> None:
        """
        Migrate the old single-workspace layout if the app root already has raw
        inbox/wiki/state folders directly beneath it.
        """
        if self.workspaces_root.exists():
            return

        legacy_dirs = [
            self.root / "inbox",
            self.root / "wiki",
            self.root / ".state",
            self.root / "attachments",
            self.root / "exports",
        ]
        legacy_files = [
            self.root / "wiki.db",
            self.root / "graph.db",
            self.root / "vectors.db",
            self.root / "metadata.json",
        ]
        if not any(p.exists() for p in legacy_dirs + legacy_files):
            return

        target = self.workspaces_root / self.default_workspace
        target.mkdir(parents=True, exist_ok=True)

        for src in legacy_dirs:
            if src.exists():
                dst = target / src.name
                if not dst.exists():
                    shutil.move(str(src), str(dst))

        for src in legacy_files:
            if src.exists():
                dst = target / src.name
                if not dst.exists():
                    shutil.move(str(src), str(dst))

        ws = self._load_workspace(target)
        ws.ensure_layout()
        self._cache[ws.id] = ws
        self._set_active_workspace_id(ws.id, persist=False)

    def _read_config(self) -> dict:
        if self.config_path and self.config_path.exists():
            try:
                return yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
            except Exception:
                return dict(self.app_config)
        return dict(self.app_config)

    def _write_config(self, data: dict) -> None:
        if not self.config_path:
            return
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def _set_active_workspace_id(self, workspace_id: str, *, persist: bool = True) -> None:
        self._active_workspace = _slugify(workspace_id)
        if persist:
            data = self._read_config()
            data["root"] = str(self.root)
            data["active_workspace"] = self._active_workspace
            if "default_workspace" not in data:
                data["default_workspace"] = self.default_workspace
            self._write_config(data)

    def _load_workspace(self, workspace_root: Path, registry_root: Path | None = None) -> Workspace:
        meta_path = workspace_root / "metadata.json"
        if meta_path.exists():
            try:
                ws = Workspace.from_metadata_file(meta_path)
            except Exception:
                ws = Workspace(
                    id=workspace_root.name,
                    name=workspace_root.name.replace("-", " ").title(),
                    description="",
                    created_at=now_iso(),
                    last_used=now_iso(),
                    root=workspace_root,
                    registry_root=registry_root,
                )
        else:
            ws = Workspace(
                id=workspace_root.name,
                name=workspace_root.name.replace("-", " ").title(),
                description="",
                created_at=now_iso(),
                last_used=now_iso(),
                root=workspace_root,
                registry_root=registry_root,
            )
        ws.root = workspace_root.resolve()
        if registry_root is not None:
            ws.registry_root = registry_root.resolve()
        return ws

    def _registry_paths(self) -> list[Path]:
        self.ensure_root()
        if not self.workspaces_root.exists():
            return []
        return sorted(self.workspaces_root.glob("*/metadata.json"))

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.workspaces_root.mkdir(parents=True, exist_ok=True)

    def ensure_workspace(self, workspace_id: str | None = None) -> Workspace:
        ws = self.resolve(workspace_id) if workspace_id else self.current()
        ws.ensure_layout()
        return ws

    def current(self) -> Workspace:
        self.ensure_root()

        if self._active_workspace is None:
            data = self._read_config()
            active = str(data.get("active_workspace") or data.get("default_workspace") or self.default_workspace)
            self._active_workspace = _slugify(active)

        try:
            ws = self.resolve(self._active_workspace)
        except ValueError:
            try:
                ws = self.resolve(self.default_workspace)
            except ValueError:
                candidates = self.list()
                if candidates:
                    ws = candidates[0]
                else:
                    ws = self.create(self.default_workspace, description="", path=None)
        ws.ensure_layout()
        self._touch(ws, persist=False)
        return ws

    def _touch(self, ws: Workspace, *, persist: bool = True) -> Workspace:
        ws.last_used = now_iso()
        ws.ensure_layout()
        self._cache[ws.id] = ws
        if persist:
            self._set_active_workspace_id(ws.id, persist=True)
        return ws

    def resolve(self, identifier: str | None) -> Workspace:
        if not identifier:
            raise ValueError("Workspace identifier is required.")
        ident = str(identifier).strip()
        slug = _slugify(ident)

        for ws in self.list(include_missing=True):
            if ws.id == slug or ws.name.lower() == ident.lower():
                return ws

        path_candidate = Path(ident).expanduser()
        if path_candidate.exists():
            meta = path_candidate / "metadata.json"
            if meta.exists():
                ws = Workspace.from_metadata_file(meta)
                self._cache[ws.id] = ws
                return ws

        # Raw registry lookup by slug in case metadata is missing.
        registry_meta = self.workspaces_root / slug / "metadata.json"
        if registry_meta.exists():
            ws = Workspace.from_metadata_file(registry_meta)
            self._cache[ws.id] = ws
            return ws

        raise ValueError(f"Workspace not found: {identifier}")

    def list(self, include_missing: bool = False) -> list[Workspace]:
        self.ensure_root()
        workspaces: list[Workspace] = []

        for meta in self._registry_paths():
            try:
                ws = Workspace.from_metadata_file(meta)
            except Exception:
                if not include_missing:
                    continue
                ws = self._load_workspace(meta.parent, registry_root=meta.parent)
            else:
                ws.registry_root = meta.parent.resolve()
            self._cache[ws.id] = ws
            workspaces.append(ws)

        if not workspaces:
            ws = self.create(self.default_workspace, description="", path=None)
            workspaces = [ws]

        workspaces.sort(key=lambda w: ((w.last_used or ""), (w.name or "").lower()), reverse=True)
        return workspaces

    def create(
        self,
        name: str,
        description: str = "",
        *,
        path: str | Path | None = None,
        tags: Sequence[str] | None = None,
    ) -> Workspace:
        self.ensure_root()
        ws_id = _slugify(name)
        registry_root = self.workspaces_root / ws_id

        if path is None:
            data_root = registry_root
        else:
            data_root = Path(path).expanduser().resolve()

        existing: Workspace | None = None
        registry_meta = registry_root / "metadata.json"
        data_meta = data_root / "metadata.json"
        if registry_meta.exists():
            existing = Workspace.from_metadata_file(registry_meta)
        elif data_meta.exists():
            existing = Workspace.from_metadata_file(data_meta)

        created_at = existing.created_at if existing else now_iso()
        last_used = now_iso()
        resolved_tags = list(tags or (existing.tags if existing else []))
        ws = Workspace(
            id=ws_id,
            name=name.strip() or ws_id.replace("-", " ").title(),
            description=description.strip() or (existing.description if existing else ""),
            created_at=created_at,
            last_used=last_used,
            root=data_root,
            tags=resolved_tags,
            registry_root=registry_root,
        )
        ws.ensure_layout()
        self._cache[ws.id] = ws
        return ws

    def switch(self, identifier: str) -> Workspace:
        ws = self.resolve(identifier)
        return self._touch(ws, persist=True)

    def path(self, *parts: str) -> Path:
        return self.current().root.joinpath(*parts)

    def current_metadata(self) -> dict:
        return self.current().metadata()

    def context(self) -> str:
        ws = self.current()
        lines = [f"Workspace: {ws.name}"]
        if ws.description:
            lines.append(f"Description: {ws.description}")
        if ws.tags:
            lines.append(f"Tags: {', '.join(ws.tags)}")
        lines.append(f"Workspace ID: {ws.id}")
        lines.append(f"Root: {ws.root}")
        return "\n".join(lines)

    def save_current_as_active(self) -> None:
        ws = self.current()
        self._set_active_workspace_id(ws.id, persist=True)

    def config_snapshot(self) -> dict:
        data = self._read_config()
        data["root"] = str(self.root)
        data["default_workspace"] = self.default_workspace
        data["active_workspace"] = self.current().id
        return data


def workspace_context_block(cfg) -> str:
    """
    Small helper for prompt injection.
    Accepts a Config-like object with a workspace_manager attribute.
    """
    try:
        return cfg.workspace_manager.context()
    except Exception:
        return ""
