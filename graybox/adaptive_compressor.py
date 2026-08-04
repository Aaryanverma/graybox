from __future__ import annotations
from graybox.ai import AIService
from litellm.utils import get_max_tokens
import logging

logger = logging.getLogger(__name__)

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

_FALLBACK_MODEL_MAX_TOKENS = 32_000

def compress_context(
    context: str,
    llm: AIService,
    max_tokens: int = 8196,
    prompt: str = None
) -> str:
    """
    Compresses the context to fit within the max_tokens limit using the provided LLM service.
    If the context is already within the limit, it is returned unchanged.

    Never raises: if the model's context window can't be determined, or the
    compression LLM call itself fails, the original context is returned
    unchanged rather than propagating the error to the caller.
    """
    if not prompt:
        logger.info("No compression prompt provided. Using default prompt.")
        prompt = "Please compress the following text while preserving the most important information:\n\n"

    try:
        model_max_tokens = get_max_tokens(llm.get_llm_params()["model"])
    except Exception as e:
        logger.warning(
            "Could not determine max tokens for model (%s); using a conservative default of %d.",
            e, _FALLBACK_MODEL_MAX_TOKENS,
        )
        model_max_tokens = None

    if not model_max_tokens:
        model_max_tokens = _FALLBACK_MODEL_MAX_TOKENS

    available_context = model_max_tokens - max_tokens - 1000  # safety buffer
    if estimate_tokens(context) <= available_context:
        return context

    logger.info("Context exceeds context limit. Compressing...")

    try:
        result = llm.llm_call(prompt=prompt + context)
    except Exception as e:
        logger.warning("Compression LLM call failed (%s); returning original context.", e)
        return context

    if result and result.get("response"):
        return result["response"]

    return context