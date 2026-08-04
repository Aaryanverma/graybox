"""Tests for retrieval.py — hybrid keyword + semantic search."""
from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from graybox.retrieval import ask, _blend_hits, _render_note, _build_context
from graybox.search_engine import Hit, _PageDoc
from graybox.models import Page, now_iso
from graybox.storage import write_page, write_inbox_item

class TestBlendHitsAcceptsSemanticTuples:
    """BUG: `search_embeddings()` (embedding_index.py) returns
    list[tuple[str, float]] — it always has, and `ask()` still calls it
    that way. But `_blend_hits()` was rewritten to expect Hit-like objects
    (`sem.doc.search_id`), so every real call raises AttributeError. In
    `ask()` this is swallowed by the outer try/except around the semantic
    search block, which silently disables semantic blending entirely and
    logs a misleading "Semantic search failed" warning instead of ever
    boosting/adding results.
    """
 
    def test_blend_hits_accepts_ref_score_tuples(self, temp_cfg):
        page = Page(id="boost", type="topic", title="Boost", created=now_iso(), updated=now_iso())
        write_page(temp_cfg, page)
        hit = Hit(doc=_PageDoc(page), score=0.5)
 
        # This is the actual shape search_embeddings() returns.
        blended = _blend_hits([hit], [("topic/boost", 0.9)], temp_cfg)
 
        assert len(blended) == 1
        assert blended[0].score >= 0.5  # should be boosted, not crash
 
    def test_ask_actually_uses_semantic_hits(self, temp_cfg):
        """End-to-end: with embeddings enabled and a semantic match indexed,
        ask() should incorporate it rather than silently falling back to
        keyword-only search."""
        temp_cfg.embeddings.enabled = True
        page = Page(
            id="semantic", type="topic", title="Semantic",
            created=now_iso(), updated=now_iso(), summary="Deep learning concepts",
        )
        write_page(temp_cfg, page)
 
        from graybox.embedding_index import EmbeddingIndex
        idx = EmbeddingIndex(temp_cfg)
        idx.index_page(page, [1.0, 0.0, 0.0])
 
        llm = MagicMock()
        llm.embedding_call.return_value = {"embedding": [1.0, 0.0, 0.0]}
        llm.llm_call.return_value = {"response": "Found via semantic search."}
        llm.get_llm_params.return_value = {"model": "gpt-4o-mini"}
 
        answer = ask(temp_cfg, llm, "neural networks")
        assert answer.grounded is True
        assert "Found via semantic" in answer.text
 
 
class TestInboxFallbackLogic:
    """
    A successful call should return a grounded, fallback answer built from
    the inbox context; a failed call should return NO_EVIDENCE_MSG safely.
    """
 
    def test_successful_inbox_fallback_returns_grounded_answer(self, temp_cfg):
        write_inbox_item(temp_cfg, "Talked about the quarterly roadmap today")
 
        llm = MagicMock()
        llm.embedding_call.return_value = None
        llm.llm_call.return_value = {"response": "Roadmap discussion happened today."}
        llm.get_llm_params.return_value = {"model": "gpt-4o-mini"}
 
        answer = ask(temp_cfg, llm, "quarterly roadmap")
 
        assert answer.grounded is True
        assert answer.fallback is True
        assert "Roadmap discussion" in answer.text
 
    def test_failed_inbox_llm_call_does_not_crash(self, temp_cfg):
        write_inbox_item(temp_cfg, "Talked about the quarterly roadmap today")
 
        llm = MagicMock()
        llm.embedding_call.return_value = None
        llm.llm_call.return_value = {"response": None}
        llm.get_llm_params.return_value = {"model": "gpt-4o-mini"}
 
        answer = ask(temp_cfg, llm, "quarterly roadmap")  # must not raise
 
        assert answer.grounded is False
        assert answer.text == (
            "I don't have enough information in the knowledge base to answer that. "
            "Try capturing more notes about this topic, or rephrase the question."
        )
 
 
class TestCompressContextDoesNotCrashAsk:
 
    def test_ask_survives_unmapped_model_name(self, temp_cfg):
        temp_cfg.llm.model_name = "totally-not-a-real-model-xyz"
        page = Page(
            id="atlas", type="project", title="Atlas Migration",
            created=now_iso(), updated=now_iso(), summary="Migrating the Atlas database",
        )
        write_page(temp_cfg, page)
 
        llm = MagicMock()
        llm.embedding_call.return_value = None
        llm.llm_call.return_value = {"response": "Answer about Atlas."}
        llm.get_llm_params.return_value = {"model": "totally-not-a-real-model-xyz"}
 
        # Should not raise, even though get_max_tokens() will throw internally.
        answer = ask(temp_cfg, llm, "atlas migration")
        assert answer.grounded is True
 
 
class TestRenderNoteDropsContinuationLines:
 
    def test_continuation_text_survives_context_rendering(self, temp_cfg):
        note = (
            "- (2026-08-02T14:30:00Z) Alice works on Atlas. _(source: inbox/abc)_\n"
            "  > Alice mentioned she is now leading the Atlas migration end to end."
        )
        page = Page(
            id="alice", type="person", title="Alice",
            created=now_iso(), updated=now_iso(), notes=[note],
        )
        write_page(temp_cfg, page)
        hit = Hit(doc=_PageDoc(page), score=0.9)
 
        context = _build_context([hit])
        assert "leading the Atlas migration end to end" in context
 
    def test_render_note_preserves_continuation(self):
        note = (
            "- (2026-08-02T14:30:00Z) Alice works on Atlas. _(source: inbox/abc)_\n"
            "  > Alice mentioned she is now leading the Atlas migration end to end."
        )
        rendered = _render_note(note)
        assert "leading the Atlas migration end to end" in rendered

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