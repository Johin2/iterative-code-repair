"""
Unified LLM client for OpenAI, Anthropic, and Google APIs.
Tracks token usage and cost for each call.
"""

import os
import time
import json
from dataclasses import dataclass, field
from typing import Optional

from config import MODEL_TIERS


@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    cost: float
    model_id: str
    tier: str
    latency_ms: float
    success: bool = True
    error: Optional[str] = None


@dataclass
class TokenTracker:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    calls: list = field(default_factory=list)

    def record(self, response: LLMResponse):
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_cost += response.cost
        self.calls.append({
            "tier": response.tier,
            "model_id": response.model_id,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost": response.cost,
            "latency_ms": response.latency_ms,
            "success": response.success,
        })

    def summary(self) -> dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": round(self.total_cost, 6),
            "num_calls": len(self.calls),
        }


def compute_cost(tier: str, input_tokens: int, output_tokens: int) -> float:
    cfg = MODEL_TIERS[tier]
    input_cost = (input_tokens / 1_000_000) * cfg["input_cost_per_1m"]
    output_cost = (output_tokens / 1_000_000) * cfg["output_cost_per_1m"]
    return input_cost + output_cost


def call_openai(model_id: str, system_prompt: str, user_prompt: str,
                temperature: float = 0.0, max_tokens: int = 4096) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    start = time.time()
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = (time.time() - start) * 1000

    return {
        "content": response.choices[0].message.content or "",
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "latency_ms": latency,
    }


def call_anthropic(model_id: str, system_prompt: str, user_prompt: str,
                   temperature: float = 0.0, max_tokens: int = 4096) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    start = time.time()
    response = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    latency = (time.time() - start) * 1000

    return {
        "content": response.content[0].text if response.content else "",
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "latency_ms": latency,
    }


def call_google(model_id: str, system_prompt: str, user_prompt: str,
                temperature: float = 0.0, max_tokens: int = 4096) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    start = time.time()
    response = client.models.generate_content(
        model=model_id,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    latency = (time.time() - start) * 1000

    usage = response.usage_metadata
    return {
        "content": response.text if response.text else "",
        "input_tokens": usage.prompt_token_count,
        "output_tokens": usage.candidates_token_count,
        "latency_ms": latency,
    }


PROVIDER_DISPATCH = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_google,
}


def call_llm(tier: str, system_prompt: str, user_prompt: str,
             temperature: float = 0.0, max_tokens: int = 4096) -> LLMResponse:
    """Call an LLM at the specified tier. Returns LLMResponse with cost tracking."""
    cfg = MODEL_TIERS[tier]
    provider = cfg["provider"]
    model_id = cfg["model_id"]
    call_fn = PROVIDER_DISPATCH[provider]

    try:
        result = call_fn(model_id, system_prompt, user_prompt, temperature, max_tokens)
        cost = compute_cost(tier, result["input_tokens"], result["output_tokens"])
        return LLMResponse(
            content=result["content"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            cost=cost,
            model_id=model_id,
            tier=tier,
            latency_ms=result["latency_ms"],
            success=True,
        )
    except Exception as e:
        return LLMResponse(
            content="",
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            model_id=model_id,
            tier=tier,
            latency_ms=0.0,
            success=False,
            error=str(e),
        )
