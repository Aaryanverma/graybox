"""Tests for graybox.ai.ai_service.AIService — mocks litellm entirely."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from graybox.ai.ai_service import AIService
from graybox.config import Config, LLMConfig, RetrievalConfig, EmbeddingsConfig
from graybox.workspace import WorkspaceManager


@pytest.fixture
def cfg(tmp_path):
    manager = WorkspaceManager(
        root=tmp_path / ".graybox", active_workspace="test", default_workspace="test",
    )
    return Config(
        root=tmp_path / ".graybox",
        workspace_manager=manager,
        llm=LLMConfig(model_name="test/model", base_url="", temperature=0.0, api_key="k"),
        retrieval=RetrievalConfig(top_k=5, min_score=0.4, dedup_threshold=0.85),
        embeddings=EmbeddingsConfig(model_name="test/embed", api_key="k"),
    )


def _fake_chat_response(text: str):
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message, logprobs=None)
    return SimpleNamespace(choices=[choice])


class TestLocalModelCostMap:
    """Regression: litellm's import fetches its model-cost map over the
    network, which hangs ~20s on broken-IPv6 networks. Setting
    LITELLM_LOCAL_MODEL_COST_MAP before the import skips the fetch entirely
    and uses litellm's bundled local map — fast for everyone."""

    def test_env_var_set_before_litellm_import(self):
        import inspect

        import graybox.ai.ai_service as ai_service

        src = inspect.getsource(ai_service)
        setdefault = 'os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")'
        litellm_import = "from litellm import ("
        assert setdefault in src and litellm_import in src
        assert src.index(setdefault) < src.index(litellm_import)



class TestGetLlmParams:
    def test_builds_expected_params(self, cfg):
        svc = AIService(cfg)
        params = svc.get_llm_params()
        assert params["model"] == "test/model"
        assert params["api_key"] == "k"
        assert params["api_type"] == "chat_completion"

    def test_kwargs_merge_into_final_kwargs(self, cfg):
        svc = AIService(cfg)
        params = svc.get_llm_params(top_p=0.9)
        assert params["final_kwargs"]["top_p"] == 0.9


class TestLlmCall:
    def test_successful_call_extracts_response(self, cfg):
        svc = AIService(cfg)
        with patch("graybox.ai.ai_service.completion", return_value=_fake_chat_response("Hello there")) as mock_completion, \
             patch("graybox.ai.ai_service.completion_cost", return_value=0.001):
            result = svc.llm_call(system_prompt="sys", prompt="hi")
        assert result["response"] == "Hello there"
        assert mock_completion.called

    def test_non_retryable_exception_returns_error_dict(self, cfg):
        svc = AIService(cfg)
        with patch("graybox.ai.ai_service.completion", side_effect=ValueError("boom")):
            result = svc.llm_call(system_prompt="sys", prompt="hi")
        assert result["response"] is None
        assert "boom" in result["error"]

    def test_messages_built_with_system_and_user(self, cfg):
        svc = AIService(cfg)
        messages = svc._build_messages("system text", "user text")
        assert messages[0] == {"role": "system", "content": "system text"}
        assert messages[1] == {"role": "user", "content": "user text"}

    def test_default_system_prompt_when_none(self, cfg):
        svc = AIService(cfg)
        messages = svc._build_messages(None, "user text")
        assert messages[0]["content"] == "You are a helpful assistant."

    def test_streaming_call_returns_stream_wrapper(self, cfg):
        svc = AIService(cfg)
        fake_stream = SimpleNamespace()
        with patch("graybox.ai.ai_service.completion", return_value=fake_stream):
            result = svc.llm_call(system_prompt="sys", prompt="hi", stream=True)

        assert result["response"] is fake_stream
        assert result["logprobs"] is None
        assert result["cost"] == 0.0
        assert result["streaming"] is True


class TestLlmCallBatch:
    def test_batch_call_extracts_all_responses(self, cfg):
        svc = AIService(cfg)
        fake_responses = [_fake_chat_response("one"), _fake_chat_response("two")]
        with patch("graybox.ai.ai_service.batch_completion", return_value=fake_responses), \
             patch("graybox.ai.ai_service.completion_cost", return_value=0.001):
            result = svc.llm_call_batch(system_prompt="sys", prompts=["p1", "p2"])
        assert result["response"] == ["one", "two"]
        assert result["cost"] == pytest.approx(0.002, rel=1e-3)

    def test_batch_rejects_non_chat_api_type(self, cfg):
        cfg.llm.api_type = "responses"
        svc = AIService(cfg)
        with pytest.raises(ValueError, match="Batch calls only support"):
            svc.llm_call_batch(system_prompt="sys", prompts=["p1"])

    def test_batch_failure_path_does_not_crash_logging(self, cfg):
        """Regression test: the except-block used to call
        logger.exception(msg, exc) with no '%s' placeholder in msg, which
        raised a TypeError from inside logging itself and masked the real
        error. It must now log cleanly and return a graceful fallback."""
        svc = AIService(cfg)
        with patch("graybox.ai.ai_service.batch_completion", side_effect=ValueError("batch boom")):
            result = svc.llm_call_batch(system_prompt="sys", prompts=["p1", "p2"])
        assert result["response"] == [None, None]
        assert result["cost"] == 0.0


class TestEmbeddingCall:
    def test_successful_embedding_call(self, cfg):
        svc = AIService(cfg)
        fake_response = SimpleNamespace(data=[{"embedding": [0.1, 0.2, 0.3]}])
        with patch("graybox.ai.ai_service.embedding", return_value=fake_response), \
             patch("graybox.ai.ai_service.completion_cost", return_value=0.0001):
            result = svc.embedding_call("some text")
        assert result["embedding"] == [0.1, 0.2, 0.3]

    def test_embedding_failure_returns_none(self, cfg):
        svc = AIService(cfg)
        with patch("graybox.ai.ai_service.embedding", side_effect=ValueError("down")):
            result = svc.embedding_call("some text")
        assert result is None

    def test_embedding_batch_returns_empty_for_no_texts(self, cfg):
        svc = AIService(cfg)
        assert svc.embedding_call_batch([]) == []

    def test_embedding_batch_extracts_all(self, cfg):
        svc = AIService(cfg)
        fake_response = SimpleNamespace(data=[{"embedding": [1.0]}, {"embedding": [2.0]}])
        with patch("graybox.ai.ai_service.embedding", return_value=fake_response), \
             patch("graybox.ai.ai_service.completion_cost", return_value=0.0):
            result = svc.embedding_call_batch(["a", "b"])
        assert result["embedding"] == [[1.0], [2.0]]