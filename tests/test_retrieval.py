"""Tests for retrieval.py — hybrid keyword + semantic search."""
from __future__ import annotations

from unittest.mock import MagicMock

from graybox.retrieval import ask, _blend_hits
from graybox.search_engine import Hit, _PageDoc
from graybox.models import Page, now_iso
from graybox.storage import write_page


class TestBlendHits:
    def test_keyword_only_no_change(self, temp_cfg):
        page = Page(id="kw", type="topic", title="KW", created=now_iso(), updated=now_iso())
        write_page(temp_cfg, page)
        hit = Hit(doc=_PageDoc(page), score=0.8)
        blended = _blend_hits([hit], [], temp_cfg)
        assert len(blended) == 1
        assert blended[0].score == 0.8

    def test_semantic_boosts_existing_keyword_hit(self, temp_cfg):
        page = Page(id="boost", type="topic", title="Boost", created=now_iso(), updated=now_iso())
        write_page(temp_cfg, page)
        hit = Hit(doc=_PageDoc(page), score=0.5)
        blended = _blend_hits([hit], [("topic/boost", 0.9)], temp_cfg)
        assert len(blended) == 1
        # 0.6 * 0.5 + 0.4 * 0.9 = 0.66
        assert blended[0].score == 0.66

    def test_semantic_only_adds_new_hit(self, temp_cfg):
        page = Page(id="sem", type="topic", title="Sem", created=now_iso(), updated=now_iso())
        write_page(temp_cfg, page)
        blended = _blend_hits([], [("topic/sem", 0.8)], temp_cfg)
        assert len(blended) == 1
        # 0.8 * 0.85 = 0.68
        assert blended[0].score == 0.68

    def test_dedupes_multiple_sources(self, temp_cfg):
        page = Page(id="dup", type="topic", title="Dup", created=now_iso(), updated=now_iso())
        write_page(temp_cfg, page)
        hit = Hit(doc=_PageDoc(page), score=0.7)
        blended = _blend_hits([hit], [("topic/dup", 0.9), ("topic/dup", 0.9)], temp_cfg)
        # Should dedupe by ref
        assert len(blended) == 1


class TestAskHybrid:
    def test_falls_back_to_keyword_when_embedding_fails(self, temp_cfg):
        """If embedding call throws, ask() should still work with keywords."""
        temp_cfg.embeddings.enabled = True
        page = Page(id="fallback", type="topic", title="Fallback", created=now_iso(), updated=now_iso(), summary="Test page content")
        write_page(temp_cfg, page)

        llm = MagicMock()
        llm.embedding_call.side_effect = RuntimeError("API down")
        llm.llm_call.return_value = {"response": "Answer from keywords."}

        answer = ask(temp_cfg, llm, "test page")
        assert answer.grounded is True
        assert "Answer from keywords" in answer.text

    def test_uses_semantic_when_enabled(self, temp_cfg):
        """When embeddings enabled and semantic finds a match, it should be included."""
        temp_cfg.embeddings.enabled = True
        page = Page(id="semantic", type="topic", title="Semantic", created=now_iso(), updated=now_iso(), summary="Deep learning concepts")
        write_page(temp_cfg, page)

        # Pre-index the page
        from graybox.embedding_index import EmbeddingIndex
        idx = EmbeddingIndex(temp_cfg)
        idx.index_page(page, [1.0, 0.0, 0.0])

        llm = MagicMock()
        llm.embedding_call.return_value = {"embedding": [1.0, 0.0, 0.0]}
        llm.llm_call.return_value = {"response": "Found via semantic search."}

        answer = ask(temp_cfg, llm, "neural networks")  # paraphrase of "deep learning"
        assert answer.grounded is True
        assert "Found via semantic" in answer.text