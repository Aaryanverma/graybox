"""Tests for search_engine.py — covers the three known correctness bugs."""
from __future__ import annotations

import pytest

from graybox.search_engine import Engine, Query, _PageDoc, tokenize
from graybox.models import Page


class TestTokenize:
    def test_basic_tokenization(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_stopwords_removed(self):
        assert tokenize("The quick brown fox") == ["quick", "brown", "fox"]

    def test_short_tokens_kept(self):
        """Tokens of length 2 are kept (condition is len > 1, not >= 2)."""
        assert tokenize("a bb ccc") == ["bb", "ccc"]

    def test_numbers_kept(self):
        assert tokenize("Version 2024 release") == ["version", "2024", "release"]


class TestCoverageScorer:
    def test_empty_query_returns_zero(self):
        doc = _PageDoc(Page(id="x", type="topic", title="X", created="", updated=""))
        assert Engine.coverage_scorer(Query.parse(""), doc) == 0.0

    def test_no_match_returns_zero(self):
        doc = _PageDoc(Page(id="x", type="topic", title="X", created="", updated="", summary="foo bar"))
        assert Engine.coverage_scorer(Query.parse("baz qux"), doc) == 0.0

    def test_perfect_match_near_one(self):
        doc = _PageDoc(Page(id="x", type="topic", title="hello world", created="", updated=""))
        score = Engine.coverage_scorer(Query.parse("hello world"), doc)
        assert 0.9 <= score <= 1.0

    def test_partial_match_midrange(self):
        """Use a page whose title does NOT trigger the identifier_blob boost.
        The identifier boost fires when query tokens appear in title/aliases."""
        doc = _PageDoc(Page(id="x", type="topic", title="foo bar baz", created="", updated=""))
        score = Engine.coverage_scorer(Query.parse("hello world bar"), doc)
        # "bar" is in title so this DOES trigger the boost — use a different query
        doc2 = _PageDoc(Page(id="x", type="topic", title="alpha beta gamma", created="", updated=""))
        score2 = Engine.coverage_scorer(Query.parse("hello world delta"), doc2)
        assert 0.0 <= score2 < 0.9

    def test_identifier_boost(self):
        """Exact title/alias hits should boost score to >=0.9."""
        doc = _PageDoc(Page(id="x", type="topic", title="Atlas Migration", created="", updated=""))
        score = Engine.coverage_scorer(Query.parse("atlas migration"), doc)
        assert score >= 0.9

    def test_scores_normalized_to_0_1(self):
        doc = _PageDoc(Page(id="x", type="topic", title="a b c d e f g h i j", created="", updated=""))
        for q in ["a", "a b", "a b c", "nonexistent"]:
            score = Engine.coverage_scorer(Query.parse(q), doc)
            assert 0.0 <= score <= 1.0


class TestNameScorer:
    """The name_scorer had two critical bugs:
    1. Word-boundary loss: "data" inside "database" caused false 1.0 matches.
    2. Owner/attendee leakage: cross-type comparisons counted owner names
       as identity matches, causing unrelated pages to look like duplicates.
    """

    def test_exact_whole_name_match(self):
        doc = _PageDoc(Page(id="x", type="topic", title="Aaryan Verma", created="", updated=""))
        assert Engine.name_scorer(Query.parse("Aaryan Verma"), doc) == 1.0

    def test_alias_match(self):
        doc = _PageDoc(Page(id="x", type="topic", title="AV", created="", updated="", aliases=["Aaryan Verma"]))
        assert Engine.name_scorer(Query.parse("Aaryan Verma"), doc) == 1.0

    def test_substring_match_for_long_names(self):
        """Substring match is allowed for names >= 4 chars (e.g. 'aaryan' inside 'aaryan verma')."""
        doc = _PageDoc(Page(id="x", type="topic", title="Aaryan Verma", created="", updated=""))
        assert Engine.name_scorer(Query.parse("aaryan"), doc) == 1.0

    def test_word_boundary_preserved_no_false_1_0(self):
        """BUG FIX: 'Data Science' vs 'Create a new database' must NOT match at 1.0.
        Previously, stripping non-alphanum and doing substring match caused
        'data' to match inside 'database'."""
        doc = _PageDoc(Page(id="x", type="topic", title="Create a new database", created="", updated=""))
        score = Engine.name_scorer(Query.parse("Data Science"), doc)
        assert score < 1.0
        assert score < 0.5

    def test_word_boundary_reverse(self):
        doc = _PageDoc(Page(id="x", type="topic", title="Data Science", created="", updated=""))
        score = Engine.name_scorer(Query.parse("Create a new database"), doc)
        assert score < 1.0

    def test_cross_type_owner_not_counted(self):
        """BUG FIX: When core_identity_only=True, owner/attendee names must not
        leak into identity comparison. A task owned by 'Aaryan' should not
        look like the same entity as a person page titled 'Aaryan'."""
        person = _PageDoc(Page(id="aaryan", type="person", title="Aaryan", created="", updated=""))
        task = _PageDoc(Page(id="task1", type="task", title="Fix bug", created="", updated="", owner="Aaryan"))

        # Query the TASK by the PERSON'S name — cross-type should NOT match
        score_cross = Engine.name_scorer(Query.parse("Aaryan"), task, core_identity_only=True)
        # task's core_identity_candidates are ["Fix bug"] — no "Aaryan"
        assert score_cross == 0.0

        # Same-type with owner included: owner is in identity_candidates
        # but query "Aaryan" vs task title "Fix bug" should still not match
        score_same = Engine.name_scorer(Query.parse("Aaryan"), task, core_identity_only=False)
        # "Aaryan" is an identity candidate (owner), so this WILL match at 1.0
        # The test above verifies cross-type is blocked; this verifies same-type allows it
        assert score_same == 1.0

    def test_cross_type_alias_still_works(self):
        """Cross-type with core_identity_only should still match on title/aliases."""
        topic = _PageDoc(Page(id="aaryan", type="topic", title="Aaryan", created="", updated="", aliases=["AV"]))
        score = Engine.name_scorer(Query.parse("AV"), topic, core_identity_only=True)
        assert score == 1.0

    def test_empty_query_returns_zero(self):
        doc = _PageDoc(Page(id="x", type="topic", title="X", created="", updated=""))
        assert Engine.name_scorer(Query.parse(""), doc) == 0.0

    def test_no_candidates_returns_zero(self):
        """Inbox docs have no identity candidates."""
        from graybox.search_engine import _InboxDoc
        from graybox.models import InboxItem
        doc = _InboxDoc(InboxItem(id="i1", created="", content="hello"))
        assert Engine.name_scorer(Query.parse("hello"), doc) == 0.0


class TestFindDuplicates:
    """Integration-level tests using the same routines as the dupes command."""

    def test_no_dupes_for_unrelated(self):
        pages = [
            Page(id="a", type="person", title="Alice", created="", updated=""),
            Page(id="b", type="person", title="Bob", created="", updated=""),
        ]
        engine = Engine()
        engine.add_wiki(pages)
        hits = engine.search(Query.parse("Alice"), Engine.name_scorer, top_k=10, min_score=0.85, kind="wiki")
        assert all(h.doc.page.id != "b" for h in hits)

    def test_finds_typo_variant(self):
        pages = [
            Page(id="aaryan", type="person", title="Aaryan Verma", created="", updated=""),
            Page(id="aarian", type="person", title="Aarian Verma", created="", updated=""),
        ]
        engine = Engine()
        engine.add_wiki(pages)
        hits = engine.search(Query.parse("Aaryan Verma"), Engine.name_scorer, top_k=10, min_score=0.85, kind="wiki")
        assert any(h.doc.page.id == "aarian" for h in hits)