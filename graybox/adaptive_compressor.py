from __future__ import annotations
from graybox.ai import AIService
from litellm.utils import get_max_tokens
import logging

logger = logging.getLogger(__name__)

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def compress_context(
    context: str,
    llm: AIService,
    max_tokens: int = 8196,
    prompt: str = None
) -> str:
    """
    Compresses the context to fit within the max_tokens limit using the provided LLM service.
    If the context is already within the limit, it is returned unchanged.
    """
    if not prompt:
        logger.info("No compression prompt provided. Using default prompt.")
        prompt = "Please compress the following text while preserving the most important information:\n\n"
        
    available_context = (
        get_max_tokens(llm.get_llm_params()["model"])
        - max_tokens
        - 1000  # safety buffer
    )
    if estimate_tokens(context) <= available_context:
        return context
    else:
        logger.info(f"Context exceeds context limit. Compressing...")

    result = llm.llm_call(prompt=prompt + context)
    if result and result["response"]:
        return result["response"]

    return context
