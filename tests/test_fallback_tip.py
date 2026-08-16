"""Tests for cli._fallback_tip() — the follow-up guidance line shown under
a fallback Answer. Regression coverage for the bug where cli.py showed one
hardcoded "came from raw captures" message regardless of which fallback
path (inbox / weak_wiki / weak_inbox) actually produced the answer."""
from __future__ import annotations

from graybox.cli import _fallback_tip, FALLBACK_TIPS


class TestFallbackTip:
    def test_inbox_kind_mentions_organize(self):
        tip = _fallback_tip("inbox")
        assert "organize" in tip.lower()

    def test_weak_wiki_kind_does_not_claim_raw_captures(self):
        """This is the exact bug being regression-tested: Path C's answer
        comes from an actual wiki page, so its tip must not tell the user
        it came from raw captures."""
        tip = _fallback_tip("weak_wiki")
        assert "raw capture" not in tip.lower()

    def test_weak_inbox_kind_mentions_organize(self):
        tip = _fallback_tip("weak_inbox")
        assert "organize" in tip.lower()

    def test_all_three_kinds_produce_distinct_tips(self):
        tips = {_fallback_tip(k) for k in ("inbox", "weak_wiki", "weak_inbox")}
        assert len(tips) == 3

    def test_unrecognized_kind_falls_back_to_default_rather_than_crashing(self):
        assert _fallback_tip("some_future_kind_nobody_wired_yet") == FALLBACK_TIPS["inbox"]

    def test_empty_kind_falls_back_to_default(self):
        assert _fallback_tip("") == FALLBACK_TIPS["inbox"]