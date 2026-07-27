"""Tests for embedding_index.py — semantic search storage and retrieval."""
from __future__ import annotations

from graybox.embedding_index import (
    EmbeddingIndex,
    _get_index,
    ensure_indexed,
    search_embeddings,
    _cosine_similarity,
    _text_hash,
)
from graybox.models import Page, now_iso
from graybox.storage import write_page


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == 1.0

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == -1.0

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestTextHash:
    def test_deterministic(self):
        assert _text_hash("hello") == _text_hash("hello")

    def test_different_inputs(self):
        assert _text_hash("hello") != _text_hash("world")


class TestEmbeddingIndex:
    def test_index_and_retrieve(self, temp_cfg):
        idx = EmbeddingIndex(temp_cfg)
        page = Page(id="test", type="topic", title="Test", created=now_iso(), updated=now_iso(), summary="A test page")
        idx.index_page(page, [1.0, 0.0, 0.0])

        results = idx.search([1.0, 0.0, 0.0], top_k=5, min_score=0.5)
        assert len(results) == 1
        assert results[0][0] == "topic/test"
        assert results[0][1] == 1.0

    def test_search_respects_min_score(self, temp_cfg):
        idx = EmbeddingIndex(temp_cfg)
        page = Page(id="a", type="topic", title="A", created=now_iso(), updated=now_iso())
        idx.index_page(page, [1.0, 0.0, 0.0])

        # Orthogonal vector should score 0.0
        results = idx.search([0.0, 1.0, 0.0], min_score=0.1)
        assert len(results) == 0

    def test_needs_reindex_on_content_change(self, temp_cfg):
        idx = EmbeddingIndex(temp_cfg)
        page = Page(id="change", type="topic", title="Change", created=now_iso(), updated=now_iso(), summary="Old")
        idx.index_page(page, [1.0, 0.0])

        assert not idx.needs_reindex(page)

        page.summary = "New"
        assert idx.needs_reindex(page)

    def test_drop_removes_entry(self, temp_cfg):
        idx = EmbeddingIndex(temp_cfg)
        page = Page(id="gone", type="topic", title="Gone", created=now_iso(), updated=now_iso())
        idx.index_page(page, [1.0, 0.0])
        assert len(idx.search([1.0, 0.0], top_k=5)) == 1

        idx.drop("topic/gone")
        assert len(idx.search([1.0, 0.0], top_k=5)) == 0

    def test_stats(self, temp_cfg):
        idx = EmbeddingIndex(temp_cfg)
        assert idx.stats()["indexed_pages"] == 0
        page = Page(id="s", type="topic", title="S", created=now_iso(), updated=now_iso())
        idx.index_page(page, [1.0])
        assert idx.stats()["indexed_pages"] == 1


class TestEnsureIndexed:
    def test_skips_when_disabled(self, temp_cfg):
        temp_cfg.embeddings.enabled = False
        page = Page(id="skip", type="topic", title="Skip", created=now_iso(), updated=now_iso())
        from unittest.mock import MagicMock
        llm = MagicMock()
        assert ensure_indexed(temp_cfg, page, llm) is False
        llm.embedding_call.assert_not_called()

    def test_indexes_when_enabled(self, temp_cfg):
        temp_cfg.embeddings.enabled = True
        page = Page(id="idx", type="topic", title="Idx", created=now_iso(), updated=now_iso())
        from unittest.mock import MagicMock
        llm = MagicMock()
        llm.embedding_call.return_value = {"embedding": [0.5, 0.5]}
        assert ensure_indexed(temp_cfg, page, llm) is True
        llm.embedding_call.assert_called_once()

    def test_skips_if_already_indexed(self, temp_cfg):
        temp_cfg.embeddings.enabled = True
        page = Page(id="cached", type="topic", title="Cached", created=now_iso(), updated=now_iso())
        from unittest.mock import MagicMock
        llm = MagicMock()
        llm.embedding_call.return_value = {"embedding": [0.5, 0.5]}
        ensure_indexed(temp_cfg, page, llm)
        llm.embedding_call.assert_called_once()

        # Second call should skip
        llm.reset_mock()
        assert ensure_indexed(temp_cfg, page, llm) is True
        llm.embedding_call.assert_not_called()


class TestSearchEmbeddings:
    def test_returns_empty_when_disabled(self, temp_cfg):
        temp_cfg.embeddings.enabled = False
        results = search_embeddings(temp_cfg, [1.0, 0.0])
        assert results == []