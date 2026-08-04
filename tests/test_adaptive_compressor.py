"""Tests for adaptive_compressor.py — this module had zero test coverage."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from graybox.adaptive_compressor import compress_context, estimate_tokens


class TestEstimateTokens:
    def test_roughly_four_chars_per_token(self):
        assert estimate_tokens("a" * 400) == 100

    def test_never_returns_zero(self):
        assert estimate_tokens("") == 1
        assert estimate_tokens("hi") == 1


class TestCompressContext:
    def test_short_context_returned_unchanged(self):
        llm = MagicMock()
        llm.get_llm_params.return_value = {"model": "gpt-4o-mini"}
        short_text = "This is a short piece of context."

        result = compress_context(short_text, llm, max_tokens=1024)

        assert result == short_text
        llm.llm_call.assert_not_called()

    def test_long_context_gets_compressed_via_llm(self):
        llm = MagicMock()
        llm.get_llm_params.return_value = {"model": "gpt-4o-mini"}
        llm.llm_call.return_value = {"response": "Compressed summary."}
        long_text = "word " * 200_000  # far beyond any max_tokens budget

        result = compress_context(long_text, llm, max_tokens=1024)

        assert result == "Compressed summary."
        llm.llm_call.assert_called_once()

    def test_falls_back_to_original_when_llm_fails(self):
        llm = MagicMock()
        llm.get_llm_params.return_value = {"model": "gpt-4o-mini"}
        llm.llm_call.return_value = {"response": None}
        long_text = "word " * 200_000

        result = compress_context(long_text, llm, max_tokens=1024)

        assert result == long_text

    def test_default_prompt_used_when_none_given(self):
        llm = MagicMock()
        llm.get_llm_params.return_value = {"model": "gpt-4o-mini"}
        llm.llm_call.return_value = {"response": "Compressed."}
        long_text = "word " * 200_000

        compress_context(long_text, llm, max_tokens=1024, prompt=None)

        called_kwargs = llm.llm_call.call_args
        sent_prompt = called_kwargs.kwargs.get("prompt") or called_kwargs.args[0]
        assert "compress" in sent_prompt.lower()

    def test_unmapped_model_name_does_not_raise(self):
        """get_max_tokens() raises for any model litellm doesn't recognize
        (most local/Ollama model strings, custom LiteLLM proxy aliases, or
        a typo'd name). compress_context() must guard against that and
        fall back to a conservative default window rather than raising,
        since graybox is meant to support pluggable/local LLMs."""
        llm = MagicMock()
        llm.get_llm_params.return_value = {"model": "not-a-real-model-xyz"}
        llm.llm_call.return_value = {"response": "Compressed."}
        long_text = "word " * 200_000

        result = compress_context(long_text, llm, max_tokens=1024)

        assert result == "Compressed."
        llm.llm_call.assert_called_once()

    def test_compression_llm_call_failure_falls_back_to_original(self):
        """If the compression call itself raises (e.g. transient API
        error), the original context should be returned, not propagated."""
        llm = MagicMock()
        llm.get_llm_params.return_value = {"model": "gpt-4o-mini"}
        llm.llm_call.side_effect = RuntimeError("API down")
        long_text = "word " * 200_000

        result = compress_context(long_text, llm, max_tokens=1024)

        assert result == long_text