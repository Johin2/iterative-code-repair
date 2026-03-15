"""Shared Groq API client with retry logic."""
from __future__ import annotations

import time
from typing import Optional

from groq import Groq

from experiments.config import GROQ_API_KEY, TEMPERATURE, MAX_TOKENS, REQUEST_DELAY


def create_client() -> Groq:
    """Create a Groq API client."""
    return Groq(api_key=GROQ_API_KEY)


def call_model(
    client: Groq,
    model_id: str,
    messages: list[dict[str, str]],
    max_retries: int = 6,
) -> tuple[Optional[str], dict[str, int]]:
    """Call the Groq API with automatic retry on rate limits.

    Returns:
        (response_text, usage_dict) on success, or (None, {}) after all retries fail.
    """
    time.sleep(REQUEST_DELAY)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            text = response.choices[0].message.content or ""
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            return text, usage
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate" in err_str:
                wait = 60 * (attempt + 1)
                print(f" [rl {wait}s]", end="", flush=True)
                time.sleep(wait)
            else:
                print(f" [err: {type(e).__name__}]", end="", flush=True)
                time.sleep(10)

    return None, {}
