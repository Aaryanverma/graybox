"""Tests for index.py — mtime-based caching eliminates full-corpus scans."""
from __future__ import annotations

import time

from graybox.index import (
    cached_list_pages,
    cached_list_inbox_items,
    cached_read_page,
    invalidate_page,
    invalidate_inbox,
    invalidate_all,
    _page_cache,
    _inbox_cache,
)
from graybox.models import Page, InboxItem, now_iso
from graybox.storage import write_page, write_inbox_item, read_page, list_pages, list_inbox_items


class TestPageCache:
    def test_first_read_populates_cache(self, temp_cfg):
        page = Page(id="cached", type="topic", title="Cached", created=now_iso(), updated=now_iso())
        write_page(temp_cfg, page)

        # First read should populate cache
        loaded = read_page(temp_cfg, "topic", "cached")
        assert loaded is not None
        cache = _page_cache(temp_cfg)
        assert any("cached" in k for k in cache)

    def test_second_read_uses_cache(self, temp_cfg):
        page = Page(id="fast", type="topic", title="Fast", created=now_iso(), updated=now_iso())
        write_page(temp_cfg, page)

        # Prime cache
        _ = read_page(temp_cfg, "topic", "fast")
        cache = _page_cache(temp_cfg)
        key = [k for k in cache if "fast" in k][0]
        mtime_before = cache[key][0]

        # Second read should hit cache (same mtime)
        loaded = read_page(temp_cfg, "topic", "fast")
        assert loaded.title == "Fast"
        assert cache[key][0] == mtime_before

    def test_write_invalidates_cache(self, temp_cfg):
        page = Page(id="inv", type="topic", title="Inv", created=now_iso(), updated=now_iso())
        write_page(temp_cfg, page)
        _ = read_page(temp_cfg, "topic", "inv")  # prime cache

        # Overwrite
        page.summary = "Updated"
        write_page(temp_cfg, page)

        # Cache should be invalidated and re-read
        loaded = read_page(temp_cfg, "topic", "inv")
        assert loaded.summary == "Updated"

    def test_list_pages_uses_cache(self, temp_cfg):
        for i in range(5):
            write_page(temp_cfg, Page(id=f"p{i}", type="topic", title=f"P{i}", created=now_iso(), updated=now_iso()))

        # First scan populates cache
        pages1 = list_pages(temp_cfg)
        assert len(pages1) == 5

        # Second scan should be instant (all cached)
        pages2 = list_pages(temp_cfg)
        assert len(pages2) == 5

    def test_invalidate_all_clears_everything(self, temp_cfg):
        page = Page(id="gone", type="topic", title="Gone", created=now_iso(), updated=now_iso())
        write_page(temp_cfg, page)
        _ = read_page(temp_cfg, "topic", "gone")

        invalidate_all(temp_cfg)
        cache = _page_cache(temp_cfg)
        assert len(cache) == 0


class TestInboxCache:
    def test_inbox_items_cached(self, temp_cfg):
        item = write_inbox_item(temp_cfg, "Test note content")
        items1 = list_inbox_items(temp_cfg)
        assert len(items1) == 1

        # Second call should hit cache
        items2 = list_inbox_items(temp_cfg)
        assert items2[0].content == "Test note content"

    def test_new_inbox_item_invalidates(self, temp_cfg):
        write_inbox_item(temp_cfg, "First")
        _ = list_inbox_items(temp_cfg)

        write_inbox_item(temp_cfg, "Second")
        items = list_inbox_items(temp_cfg)
        assert len(items) == 2
        contents = {i.content for i in items}
        assert "First" in contents
        assert "Second" in contents