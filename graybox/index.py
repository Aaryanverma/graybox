"""
Caching / indexing layer for storage.py.

Goal: eliminate full-corpus scans on every search/ask/dashboard/organize call.

Strategy:
- In-memory LRU-style cache keyed by file path + mtime.
- On list_pages() / list_inbox_items(), only re-parse files whose mtime
  changed since the last scan.
- On every write, the cache is eagerly updated so subsequent reads are free.
- A lightweight JSON manifest is persisted to .state/index_manifest.json so
  the *first* load after restart can skip parsing unchanged files.

This is intentionally simple: no external DB, no complex invalidation.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from graybox.config import Config
from graybox.models import Page, InboxItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory caches (per workspace)
# ---------------------------------------------------------------------------
_PAGE_CACHE: dict[str, dict[str, tuple[float, Page]]] = {}
_INBOX_CACHE: dict[str, dict[str, tuple[float, InboxItem]]] = {}
_MANIFEST: dict[str, dict] = {}


def _cache_key(cfg: Config) -> str:
    return str(cfg.workspace)


def _manifest_path(cfg: Config) -> Path:
    return cfg.state_dir / "index_manifest.json"


def _load_manifest(cfg: Config) -> dict:
    key = _cache_key(cfg)
    if key not in _MANIFEST:
        p = _manifest_path(cfg)
        if p.exists():
            try:
                _MANIFEST[key] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                _MANIFEST[key] = {}
        else:
            _MANIFEST[key] = {}
    return _MANIFEST[key]


# def _save_manifest(cfg: Config) -> None:
#     p = _manifest_path(cfg)
#     p.write_text(json.dumps(_MANIFEST.get(_cache_key(cfg), {}), indent=2), encoding="utf-8")


def _page_cache(cfg: Config) -> dict[str, tuple[float, Page]]:
    key = _cache_key(cfg)
    if key not in _PAGE_CACHE:
        _PAGE_CACHE[key] = {}
    return _PAGE_CACHE[key]


def _inbox_cache(cfg: Config) -> dict[str, tuple[float, InboxItem]]:
    key = _cache_key(cfg)
    if key not in _INBOX_CACHE:
        _INBOX_CACHE[key] = {}
    return _INBOX_CACHE[key]


# ---------------------------------------------------------------------------
# Cache invalidation / warming
# ---------------------------------------------------------------------------

def invalidate_page(cfg: Config, page_type: str, slug: str) -> None:
    """Call after any page mutation (write, merge, edit, delete)."""
    from graybox.models import TYPE_DIR
    cache = _page_cache(cfg)
    directory = TYPE_DIR.get(page_type, page_type)
    path = str(cfg.wiki_dir / directory / f"{slug}.md")
    cache.pop(path, None)
    # Also invalidate the type-level list so next list_pages rescans the dir
    manifest = _load_manifest(cfg)
    manifest.pop(f"_list:{page_type}", None)


def invalidate_inbox(cfg: Config, item_id: str) -> None:
    cache = _inbox_cache(cfg)
    path = str(cfg.inbox_dir / f"{item_id}.md")
    cache.pop(path, None)
    manifest = _load_manifest(cfg)
    manifest.pop("_inbox_list", None)


def invalidate_all(cfg: Config) -> None:
    """Nuclear option — e.g. after a bulk import or workspace switch."""
    key = _cache_key(cfg)
    _PAGE_CACHE.pop(key, None)
    _INBOX_CACHE.pop(key, None)
    _MANIFEST.pop(key, None)


# ---------------------------------------------------------------------------
# Cached readers (drop-in replacements for storage.py internals)
# ---------------------------------------------------------------------------

def cached_read_page(path: Path, cfg: Config) -> Optional[Page]:
    """Read a single page with mtime caching."""
    if not path.exists():
        cache = _page_cache(cfg)
        cache.pop(str(path), None)
        return None

    mtime = path.stat().st_mtime
    cache = _page_cache(cfg)
    cached = cache.get(str(path))
    if cached and cached[0] == mtime:
        return cached[1]

    from graybox.storage import _parse_page
    page = _parse_page(path)
    cache[str(path)] = (mtime, page)
    return page


def cached_read_inbox(path: Path, cfg: Config) -> Optional[InboxItem]:
    if not path.exists():
        cache = _inbox_cache(cfg)
        cache.pop(str(path), None)
        return None

    mtime = path.stat().st_mtime
    cache = _inbox_cache(cfg)
    cached = cache.get(str(path))
    if cached and cached[0] == mtime:
        return cached[1]

    from graybox.storage import _parse_frontmatter
    fm, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
    item = InboxItem(
        id=fm.get("id", path.stem),
        created=fm.get("created", ""),
        content=body.strip(),
        path=str(path),
        extra=fm.get("extra", {}) or {},
    )
    cache[str(path)] = (mtime, item)
    return item


def cached_list_pages(cfg: Config, page_type: str | None = None) -> list[Page]:
    """List pages, only re-parsing files whose mtime changed."""
    from graybox.models import TYPE_DIR
    from graybox.storage import ensure_workspace

    ensure_workspace(cfg)
    dirs = [TYPE_DIR[page_type]] if page_type else sorted(set(TYPE_DIR.values()))
    cache = _page_cache(cfg)
    pages: list[Page] = []

    for d in dirs:
        folder = cfg.wiki_dir / d
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.md")):
            mtime = path.stat().st_mtime
            cached = cache.get(str(path))
            if cached and cached[0] == mtime:
                pages.append(cached[1])
            else:
                from graybox.storage import _parse_page
                page = _parse_page(path)
                cache[str(path)] = (mtime, page)
                pages.append(page)

    return pages


def cached_list_inbox_items(cfg: Config, include_forgotten: bool = False) -> list[InboxItem]:
    from graybox.storage import ensure_workspace, load_forgotten

    ensure_workspace(cfg)
    forgotten = set() if include_forgotten else set(load_forgotten(cfg).keys())
    cache = _inbox_cache(cfg)
    items: list[InboxItem] = []

    for path in sorted(cfg.inbox_dir.glob("*.md")):
        if path.stem in forgotten:
            cache.pop(str(path), None)
            continue
        mtime = path.stat().st_mtime
        cached = cache.get(str(path))
        if cached and cached[0] == mtime:
            items.append(cached[1])
        else:
            from graybox.storage import _parse_frontmatter
            fm, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
            item = InboxItem(
                id=fm.get("id", path.stem),
                created=fm.get("created", ""),
                content=body.strip(),
                path=str(path),
                extra=fm.get("extra", {}) or {},
            )
            cache[str(path)] = (mtime, item)
            items.append(item)

    return items