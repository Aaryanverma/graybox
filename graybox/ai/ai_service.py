# services.py
"""
Service layer for External APIs (LLMs, Embeddings, Rerankers).
"""

import os
from typing import Optional, List, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from graybox.ai import AuthenticationManager
from litellm import (
    completion,
    embedding,
    acompletion,
    batch_completion,
    aembedding,
    completion_cost,
    responses,
    aresponses,
)
import litellm
import requests
import logging

logger = logging.getLogger(__name__)

litellm.drop_params = True

RETRYABLE_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    OSError,
    requests.Timeout,
    litellm.exceptions.Timeout,
    litellm.exceptions.APIConnectionError,
)


def empty_decorator(fn):
    return fn


class AIService:
    def __init__(self, config):
        self.config = config
        self._authentication = AuthenticationManager()

    def get_llm_params(self, **kwargs):
        """Update LLM parameters in config at runtime"""
        cfg = self.config.llm
        model = cfg.model_name
        base_url = cfg.base_url or os.environ.get("GRAYBOX_LLM_BASE_URL")
        api_base = cfg.api_base or os.environ.get("GRAYBOX_LLM_API_BASE")
        api_key = cfg.api_key or os.environ.get("GRAYBOX_LLM_API_KEY")
        api_version = cfg.api_version
        deployment_id = cfg.deployment_id
        api_type = cfg.api_type or "chat_completion"

        final_kwargs = cfg.kwargs.copy()
        final_kwargs.update(kwargs)

        final_kwargs.update(self._authentication.get_auth_kwargs(cfg))

        if api_version:
            final_kwargs["api_version"] = api_version
        if deployment_id:
            final_kwargs["deployment_id"] = deployment_id

        params = {
            "model": model,
            "base_url": base_url or None,
            "api_base": api_base or None,
            "api_type": api_type,
            "api_key": api_key,
            "final_kwargs": final_kwargs,
        }
        return params

    def get_embedding_params(self, **kwargs):
        """Update embedding parameters in config at runtime"""
        cfg = self.config.embeddings
        model = cfg.model_name
        api_base = (
            cfg.api_base
            or cfg.base_url
            or os.environ.get("GRAYBOX_EMBEDDING_API_BASE")
            or os.environ.get("GRAYBOX_EMBEDDING_BASE_URL")
        )
        api_key = cfg.api_key or os.environ.get("GRAYBOX_EMBEDDING_API_KEY")
        input_type = cfg.input_type or None
        api_version = cfg.api_version
        deployment_id = cfg.deployment_id
        final_kwargs = cfg.kwargs.copy()
        final_kwargs.update(kwargs)

        final_kwargs.update(self._authentication.get_auth_kwargs(cfg))

        if api_version:
            final_kwargs["api_version"] = api_version
        if deployment_id:
            final_kwargs["deployment_id"] = deployment_id
        params = {
            "model": model,
            "api_base": api_base,
            "api_key": api_key,
            "input_type": input_type,
            "final_kwargs": final_kwargs,
        }
        return params

    def _build_messages(self, system_prompt: str, prompt: str) -> list[dict]:
        return [
            {
                "role": "system",
                "content": system_prompt or "You are a helpful assistant.",
            },
            {"role": "user", "content": prompt},
        ]

    def _extract_response(self, response, api_type: str) -> dict:
        try:
            cost = completion_cost(completion_response=response) or 0.0
        except Exception:
            cost = 0.0

        if api_type == "chat_completion":
            content = response.choices[0].message.content
            logprobs = None
            if (
                hasattr(response.choices[0], "logprobs")
                and response.choices[0].logprobs
            ):
                logprobs = [
                    token.logprob for token in response.choices[0].logprobs.content
                ]
            return {"response": content, "logprobs": logprobs, "cost": f"{cost:.6f}"}
        else:
            return {
                "response": response.output_text,
                "logprobs": None,
                "cost": f"{cost:.6f}",
            }

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=500),
        reraise=True,
    )
    def llm_call(self, system_prompt: str = None, prompt: str = None, **kwargs) -> dict:
        stream = kwargs.pop("stream", False)
        params = self.get_llm_params(**kwargs)
        messages = self._build_messages(system_prompt, prompt)

        try:
            if params["api_type"] == "responses":
                response = responses(
                    model=params["model"],
                    base_url=params["base_url"],
                    api_base=params["api_base"],
                    api_key=params["api_key"],
                    api_version=params.get("api_version"),
                    deployment_id=params.get("deployment_id"),
                    seed=42,
                    input=messages,
                    stream=stream,
                    **params["final_kwargs"],
                )
            else:
                response = completion(
                    model=params["model"],
                    base_url=params["base_url"],
                    api_base=params["api_base"],
                    api_key=params["api_key"],
                    seed=42,
                    messages=messages,
                    stream=stream,
                    **params["final_kwargs"],
                )
            if stream:
                return {
                    "response": response,
                    "logprobs": None,
                    "cost": 0.0,
                    "streaming": True,
                }
            return self._extract_response(response, params["api_type"])

        except RETRYABLE_EXCEPTIONS:
            raise
        except Exception as e:
            logger.exception(f"Error calling LLM: {e}")
            return {"response": None, "logprobs": None, "cost": 0.0, "error": str(e)}

    def llm_call_stream(
        self, system_prompt: str = None, prompt: str = None, **kwargs
    ) -> dict:
        params = self.get_llm_params(**kwargs)
        messages = self._build_messages(system_prompt, prompt)

        try:
            if params["api_type"] == "responses":
                stream = responses(
                    model=params["model"],
                    base_url=params["base_url"],
                    api_base=params["api_base"],
                    api_key=params["api_key"],
                    api_version=params.get("api_version"),
                    deployment_id=params.get("deployment_id"),
                    seed=42,
                    input=messages,
                    stream=True,
                    **params["final_kwargs"],
                )
            else:
                stream = completion(
                    model=params["model"],
                    base_url=params["base_url"],
                    api_base=params["api_base"],
                    api_key=params["api_key"],
                    seed=42,
                    messages=messages,
                    stream=True,
                    **params["final_kwargs"],
                )
            return {
                "response": stream,
                "logprobs": None,
                "cost": 0.0,
                "streaming": True,
            }

        except RETRYABLE_EXCEPTIONS:
            raise
        except Exception as e:
            logger.exception(f"Error calling LLM streaming: {e}")
            return {"response": None, "logprobs": None, "cost": 0.0, "error": str(e)}

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=500),
        reraise=True,
    )
    def llm_call_batch(
        self, system_prompt: str = None, prompts: list[str] = None, **kwargs
    ) -> dict:
        params = self.get_llm_params(**kwargs)
        if params["api_type"] != "chat_completion":
            raise ValueError("Batch calls only support chat_completion")

        batched_messages = [
            self._build_messages(system_prompt, p) for p in (prompts or [])
        ]
        try:
            responses_list = batch_completion(
                model=params["model"],
                base_url=params["base_url"],
                api_base=params["api_base"],
                seed=42,
                messages=batched_messages,
                **params["final_kwargs"],
            )
            extracted = [
                self._extract_response(r, "chat_completion") for r in responses_list
            ]
            return {
                "response": [e["response"] for e in extracted],
                "logprobs": (
                    [e["logprobs"] for e in extracted]
                    if any(e["logprobs"] for e in extracted)
                    else None
                ),
                "cost": sum(float(e["cost"]) for e in extracted),
            }
        except RETRYABLE_EXCEPTIONS:
            raise
        except Exception as e:
            logger.exception(f"Error calling LLM batch: {e}")
            return {
                "response": [None] * len(prompts) if prompts else [],
                "logprobs": None,
                "cost": 0.0,
            }

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=500),
        reraise=True,
    )
    async def llm_call_async(
        self, system_prompt: str = None, prompt: str = None, **kwargs
    ) -> dict:
        params = self.get_llm_params(**kwargs)
        messages = self._build_messages(system_prompt, prompt)

        try:
            if params["api_type"] == "responses":
                response = await aresponses(
                    model=params["model"],
                    base_url=params["base_url"],
                    api_base=params["api_base"],
                    seed=42,
                    messages=messages,
                    **params["final_kwargs"],
                )
            else:
                response = await acompletion(
                    model=params["model"],
                    base_url=params["base_url"],
                    api_base=params["api_base"],
                    seed=42,
                    messages=messages,
                    **params["final_kwargs"],
                )
            return self._extract_response(response, params["api_type"])
        except RETRYABLE_EXCEPTIONS:
            raise
        except Exception as e:
            logger.exception(f"Error calling LLM asynchronously: {e}")
            return {"response": None, "logprobs": None, "cost": 0.0}

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=500),
        reraise=True,
    )
    def embedding_call(self, text: str, **kwargs) -> Optional[dict]:
        """Safely call embedding model"""
        params = self.get_embedding_params(**kwargs)

        try:
            response = embedding(
                model=params["model"],
                input=[text],
                api_base=params["api_base"],
                api_key=params["api_key"],
                input_type=params["input_type"],
                **params["final_kwargs"],
            )

            try:
                cost = completion_cost(completion_response=response) or 0.0
            except Exception:
                cost = 0.0
            return {"embedding": response.data[0]["embedding"], "cost": cost}

        except RETRYABLE_EXCEPTIONS as e:
            logger.exception(f"Retryable error (will retry): {e}")
            raise
        except Exception as e:
            logger.exception(f"Error calling embedding: {e}")
            return None

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=500),
        reraise=True,
    )
    async def embedding_call_async(self, text: str, **kwargs) -> Optional[dict]:
        """Safely call embedding model"""
        params = self.get_embedding_params(**kwargs)

        try:
            response = await aembedding(
                model=params["model"],
                input=[text],
                api_base=params["api_base"],
                api_key=params["api_key"],
                input_type=params["input_type"],
                **params["final_kwargs"],
            )
            try:
                cost = completion_cost(completion_response=response) or 0.0
            except Exception:
                cost = 0.0
            return {"embedding": response.data[0]["embedding"], "cost": cost}
        except RETRYABLE_EXCEPTIONS as e:
            logger.exception(f"Retryable error (will retry): {e}")
            raise
        except Exception as e:
            logger.exception(f"Error calling embedding asynchronously: {e}")
            return None

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=500),
        reraise=True,
    )
    def embedding_call_batch(self, texts: List[str], **kwargs) -> Optional[dict]:
        """Safely call embedding model with a batch of texts"""
        if not texts:
            return []

        params = self.get_embedding_params(**kwargs)

        try:
            # litellm handles batching when input is a list
            response = embedding(
                model=params["model"],
                input=texts,
                api_base=params["api_base"],
                api_key=params["api_key"],
                input_type=params["input_type"],
                **params["final_kwargs"],
            )
            try:
                cost = completion_cost(completion_response=response) or 0.0
            except Exception:
                cost = 0.0
            # Extract list of embeddings in order
            return {
                "embedding": [data["embedding"] for data in response.data],
                "cost": cost,
            }
        except RETRYABLE_EXCEPTIONS as e:
            logger.exception(f"Retryable error (will retry): {e}")
            raise
        except Exception as e:
            logger.exception(f"Error calling embedding batch: {e}")
            return []
