"""Vertex AI API client for Gemini and Model Garden models.

Supports:
  - Gemini models (native Vertex AI): gemini-2.5-flash, gemini-2.5-pro, etc.
  - Model Garden deployments: DeepSeek, Mistral, Llama via OpenAI-compatible endpoints

Usage:
  client = create_vertex_client()
  text, usage = call_vertex_model(client, "gemini-2.5-flash", messages)
"""
from __future__ import annotations

import time
from typing import Optional

from experiments.config import TEMPERATURE, MAX_TOKENS, VERTEX_PROJECT, VERTEX_LOCATION


# ---------------------------------------------------------------------------
# Gemini via google-genai SDK (unified client)
# ---------------------------------------------------------------------------

def create_vertex_client():
    """Create a google-genai client configured for Vertex AI."""
    import os
    from google import genai

    # Avoid conflicts with stale GOOGLE_APPLICATION_CREDENTIALS pointing
    # to files that may not exist; ADC login credentials take priority.
    old_creds = os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
    if old_creds and not os.path.exists(old_creds):
        pass  # removed stale env var
    elif old_creds:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = old_creds

    client = genai.Client(
        vertexai=True,
        project=VERTEX_PROJECT,
        location=VERTEX_LOCATION,
    )
    return client


def _messages_to_genai(messages: list[dict[str, str]]) -> tuple[str | None, list[dict]]:
    """Convert OpenAI-style messages to google-genai format.

    Returns (system_instruction, contents) where contents is a list of
    role/parts dicts for the genai API.
    """
    system_instruction = None
    contents = []

    for msg in messages:
        role = msg["role"]
        text = msg["content"]

        if role == "system":
            system_instruction = text
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": text}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": text}]})

    return system_instruction, contents


def call_vertex_model(
    client,
    model_id: str,
    messages: list[dict[str, str]],
    max_retries: int = 6,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[Optional[str], dict[str, int]]:
    """Call a Gemini model via Vertex AI.

    Uses the same interface as api_client.call_model() for drop-in compatibility.

    Returns:
        (response_text, usage_dict) on success, or (None, {}) after retries fail.
    """
    from google.genai import types

    _temperature = temperature if temperature is not None else TEMPERATURE
    _max_tokens = max_tokens if max_tokens is not None else MAX_TOKENS

    system_instruction, contents = _messages_to_genai(messages)

    config = types.GenerateContentConfig(
        temperature=_temperature,
        max_output_tokens=_max_tokens,
    )
    if system_instruction:
        config.system_instruction = system_instruction

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=config,
            )

            text = response.text or ""

            # Extract usage from response metadata
            usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                um = response.usage_metadata
                usage = {
                    "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
                    "completion_tokens": getattr(um, "candidates_token_count", 0) or 0,
                    "total_tokens": getattr(um, "total_token_count", 0) or 0,
                }

            return text, usage

        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resource" in err_str or "quota" in err_str:
                wait = 30 * (attempt + 1)
                print(f" [quota {wait}s]", end="", flush=True)
                time.sleep(wait)
            elif "500" in err_str or "503" in err_str or "unavailable" in err_str:
                wait = 15 * (attempt + 1)
                print(f" [server {wait}s]", end="", flush=True)
                time.sleep(wait)
            else:
                print(f" [err: {type(e).__name__}: {str(e)[:80]}]", end="", flush=True)
                time.sleep(10)

    return None, {}


# ---------------------------------------------------------------------------
# Model Garden via OpenAI-compatible endpoint
# ---------------------------------------------------------------------------

def create_model_garden_client(endpoint_id: str):
    """Create an OpenAI-compatible client for a Model Garden endpoint.

    Args:
        endpoint_id: The full endpoint resource name, e.g.
            'projects/PROJECT/locations/LOCATION/endpoints/ENDPOINT_ID'
            or just the endpoint ID if project/location are in config.
    """
    import google.auth
    from openai import OpenAI

    credentials, project = google.auth.default()
    credentials.refresh(google.auth.transport.requests.Request())

    # Build the endpoint URL
    if "/" not in endpoint_id:
        base_url = (
            f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1/"
            f"projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}/"
            f"endpoints/{endpoint_id}"
        )
    else:
        base_url = f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1/{endpoint_id}"

    client = OpenAI(
        base_url=base_url,
        api_key=credentials.token,
    )
    return client


def call_model_garden(
    client,
    model_id: str,
    messages: list[dict[str, str]],
    max_retries: int = 6,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[Optional[str], dict[str, int]]:
    """Call a Model Garden endpoint (OpenAI-compatible).

    Same interface as the Groq call_model() for drop-in compatibility.
    """
    _temperature = temperature if temperature is not None else TEMPERATURE
    _max_tokens = max_tokens if max_tokens is not None else MAX_TOKENS

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=_temperature,
                max_tokens=_max_tokens,
            )
            text = response.choices[0].message.content or ""
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }
            return text, usage
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str:
                wait = 30 * (attempt + 1)
                print(f" [quota {wait}s]", end="", flush=True)
                time.sleep(wait)
            else:
                print(f" [err: {type(e).__name__}]", end="", flush=True)
                time.sleep(10)

    return None, {}
