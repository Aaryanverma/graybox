"""Tests for retrieval.py — hybrid keyword + semantic search."""
from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from graybox.retrieval import ask, _blend_hits, _render_note, _build_context
from graybox.search_engine import Hit, _PageDoc, _InboxDoc
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

    def test_wiki_refusal_falls_through_to_matching_inbox(self, temp_cfg, monkeypatch):
        """A non-empty abstention from Path A is not a successful answer.

        Strong-but-irrelevant wiki retrieval can happen for common phrasing.
        A matching raw capture must still be offered to the LLM before ask()
        returns the final no-evidence response.
        """
        wiki_page = Page(
            id="unrelated",
            type="topic",
            title="Unrelated",
            created=now_iso(),
            updated=now_iso(),
        )
        inbox_item = write_inbox_item(
            temp_cfg, "Going to the movie on Friday at 7 PM."
        )
        strong_wiki = Hit(doc=_PageDoc(wiki_page), score=0.8)
        strong_inbox = Hit(doc=_InboxDoc(inbox_item), score=1.0)

        monkeypatch.setattr(
            "graybox.retrieval.search_all",
            lambda cfg, q, top_k, all_workspaces, inbox_min_score: (
                [strong_wiki], [strong_inbox]
            ),
        )

        llm = MagicMock()
        llm.embedding_call.return_value = None
        llm.llm_call.side_effect = [
            {"response": "I don't have enough information in the knowledge base to answer that."},
            {"response": "You are going to the movie on Friday at 7 PM."},
        ]

        answer = ask(temp_cfg, llm, "When am I going for a movie?")

        assert answer.grounded is True
        assert answer.fallback is True
        assert answer.fallback_kind == "inbox"
        assert "Friday at 7 PM" in answer.text
        assert answer.sources == [f"inbox/{inbox_item.id}"]
        assert llm.llm_call.call_count == 2
 
 
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

class TestFallbackKindTagging:
    """Answer.fallback_kind must accurately describe which path produced
    an answer, since cli.py shows different follow-up guidance per kind
    and previously relied on a single `fallback` bool that couldn't tell
    "raw inbox" apart from "low-confidence wiki page" apart from "nothing
    at all"."""

    def test_strong_wiki_path_has_no_fallback_kind(self, temp_cfg):
        write_page(temp_cfg, Page(
            id="atlas", type="project", title="Atlas Migration",
            created=now_iso(), updated=now_iso(), summary="Migrating the Atlas database",
        ))
        llm = MagicMock()
        llm.embedding_call.return_value = None
        llm.llm_call.return_value = {"response": "Atlas is a database migration."}

        answer = ask(temp_cfg, llm, "atlas migration")
        assert answer.grounded is True
        assert answer.fallback is False
        assert answer.fallback_kind == ""

    def test_inbox_fallback_path_tagged_inbox(self, temp_cfg):
        write_inbox_item(temp_cfg, "Talked about the quarterly roadmap today")
        llm = MagicMock()
        llm.embedding_call.return_value = None
        llm.llm_call.return_value = {"response": "Roadmap discussion happened today."}

        answer = ask(temp_cfg, llm, "quarterly roadmap")
        assert answer.fallback is True
        assert answer.fallback_kind == "inbox"

    def test_no_evidence_has_no_fallback_kind(self, temp_cfg):
        llm = MagicMock()
        llm.embedding_call.return_value = None
        llm.llm_call.return_value = {"response": "should never be reached"}

        answer = ask(temp_cfg, llm, "completely unrelated nonexistent topic xyz")
        assert answer.grounded is False
        assert answer.fallback is False
        assert answer.fallback_kind == ""
        # Empty workspace -> zero hits at every tier -> the LLM should
        # never even be called for an answer.
        assert llm.llm_call.call_count == 0


class TestWeakWikiFallback:
    """Path C: organized wiki pages exist, but nothing clears min_score.
    Reachable today only via semantic blending, since keyword search_all()
    already filters wiki hits to >= min_score before ask() sees them."""

    def test_semantic_only_hit_below_min_score_triggers_weak_wiki_path(self, temp_cfg, monkeypatch):
        temp_cfg.embeddings.enabled = True
        write_page(temp_cfg, Page(
            id="dim", type="topic", title="Dim Match",
            created=now_iso(), updated=now_iso(), summary="Loosely related content",
        ))

        # 0.3 * 0.85 (the damping _blend_hits applies to semantic-only
        # hits) = 0.255, below temp_cfg's min_score of 0.4 -> excluded from
        # strong_wiki, but should still surface via Path C.
        monkeypatch.setattr(
            "graybox.retrieval.search_embeddings",
            lambda cfg, emb, top_k, min_score: [("topic/dim", 0.3)],
        )

        llm = MagicMock()
        llm.embedding_call.return_value = {"embedding": [1.0, 0.0, 0.0]}
        llm.llm_call.return_value = {"response": "Loosely related answer."}

        answer = ask(temp_cfg, llm, "something with no keyword overlap")

        assert answer.grounded is True
        assert answer.fallback is True
        assert answer.fallback_kind == "weak_wiki"
        assert "No high-confidence matches" in answer.text
        assert "Loosely related answer" in answer.text
        # Path C's warning must NOT claim this came from raw captures -
        # it came from an actual wiki page, just a weak match.


class TestPathCSkippedWhenAlreadyTriedInPathA:
    """Regression test: when strong_wiki fires (Path A) but the LLM
    honestly refuses given that context, Path C used to re-run with the
    *same* wiki_hits (graph expansion runs off strong_wiki, so Path C's
    wiki_hits was already a superset Path A had already tried) - a
    guaranteed-repeat LLM call for nothing. Path C must only fire when
    strong_wiki was empty to begin with."""

    def test_refused_strong_wiki_does_not_retrigger_identical_call_in_path_c(self, temp_cfg):
        from graybox.retrieval import NO_EVIDENCE_MSG

        write_page(temp_cfg, Page(
            id="camera", type="task", title="Buy new camera",
            created=now_iso(), updated=now_iso(),
            summary="Irrelevant page that happens to clear min_score for this query.",
        ))

        prompts_seen = []

        def fake_llm_call(system_prompt=None, prompt=None, **kw):
            prompts_seen.append(prompt)
            return {"response": NO_EVIDENCE_MSG}

        llm = MagicMock()
        llm.embedding_call.return_value = None
        llm.llm_call.side_effect = fake_llm_call

        answer = ask(temp_cfg, llm, "buy new camera")

        assert len(prompts_seen) == 1, (
            "Path C re-ran with the same wiki_hits Path A already refused on"
        )
        assert answer.grounded is False
        assert answer.fallback_kind == ""
        assert "raw" not in answer.text.lower()


class TestWeakInboxLastResort:
    """Path D: regression coverage for the previously-dead last-ditch
    inbox fallback. A weak (below inbox_threshold) inbox hit must now
    actually be used as a final resort instead of being computed and then
    silently discarded."""

    def test_weak_inbox_hit_used_when_nothing_else_matches(self, temp_cfg, monkeypatch):
        item = write_inbox_item(temp_cfg, "A barely related note.")
        weak_hit = Hit(doc=_InboxDoc(item), score=0.1)  # below default inbox_threshold (0.2)

        monkeypatch.setattr(
            "graybox.retrieval.search_all",
            lambda cfg, q, top_k, all_workspaces, inbox_min_score: ([], [weak_hit]),
        )

        llm = MagicMock()
        llm.embedding_call.return_value = None
        llm.llm_call.return_value = {"response": "Rough lead based on a weak note."}

        answer = ask(temp_cfg, llm, "anything")

        assert answer.grounded is True
        assert answer.fallback is True
        assert answer.fallback_kind == "weak_inbox"
        assert "Rough lead based on a weak note" in answer.text
        assert "Only weak, un-organized inbox matches" in answer.text

    def test_weak_inbox_reachable_through_real_search(self, temp_cfg):
        """End-to-end, no mocking of search_all: proves the companion fix
        (passing inbox_min_score=_WEAK_INBOX_FLOOR into search_all) is
        actually wired through, not just present in a unit-test mock.
        Without it, search_all()'s own default threshold would already
        have filtered this hit out before Path D ever saw it."""
        write_inbox_item(temp_cfg, "Something about zebras crossing rivers in Kenya sometimes")

        llm = MagicMock()
        llm.embedding_call.return_value = None
        llm.llm_call.return_value = {"response": "A rough lead about zebras."}

        # Empirically ~0.18 against the note above: below the 0.2
        # inbox_threshold, above the 0.05 weak-inbox search floor.
        answer = ask(
            temp_cfg, llm,
            "zebras oranges bicycles telephones mountains architecture violins guitars pianos drums",
        )

        assert answer.grounded is True
        assert answer.fallback_kind == "weak_inbox"
        assert "A rough lead about zebras" in answer.text

    def test_no_hits_at_any_tier_still_returns_no_evidence(self, temp_cfg):
        """Sanity check the floor doesn't turn genuinely-empty results
        into a hallucinated last resort."""
        llm = MagicMock()
        llm.embedding_call.return_value = None
        llm.llm_call.return_value = {"response": "should never be reached"}

        answer = ask(temp_cfg, llm, "nothing exists about this at all")

        assert answer.grounded is False
        assert answer.fallback_kind == ""
        assert answer.text.startswith("I don't have enough information")
        assert llm.llm_call.call_count == 0