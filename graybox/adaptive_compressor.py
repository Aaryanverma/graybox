from __future__ import annotations
from graybox.ai import AIService
import litellm
litellm.suppress_debug_info = True
import logging

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


_FALLBACK_MODEL_CONTEXT_WINDOW = 32_000


def compress_context(
    context: str, llm: AIService, max_tokens: int = 8196, prompt: str = None
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
        model_context_window = litellm.get_model_info(llm.get_llm_params()["model"]).get(
            "max_input_tokens"
        )
    except Exception as e:
        logger.info(
            "Could not determine max tokens for model (%s); using a conservative default of %d.",
            e,
            _FALLBACK_MODEL_CONTEXT_WINDOW,
        )
        model_context_window = None

    if not model_context_window:
        model_context_window = _FALLBACK_MODEL_CONTEXT_WINDOW

    available_context = min(max_tokens, model_context_window // 2)
    if estimate_tokens(context) <= available_context:
        return context

    logger.info("Context exceeds context limit. Compressing...")

    try:
        result = llm.llm_call(prompt=prompt + context)
    except Exception as e:
        logger.info("Compression LLM call failed (%s); returning original context.", e)
        return context

    if result and result.get("response"):
        return result["response"]

    return context
