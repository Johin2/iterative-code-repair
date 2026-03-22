"""Experiment configuration: models, paths, and hyperparameters."""
from __future__ import annotations

import os
from pathlib import Path

# Load .env file from project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _value = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _value.strip())

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")

# Vertex AI configuration
VERTEX_PROJECT: str = os.environ.get("VERTEX_PROJECT", "")
VERTEX_LOCATION: str = os.environ.get("VERTEX_LOCATION", "us-central1")

# Models to evaluate: display_name -> Groq model ID
MODELS: dict[str, str] = {
    "Llama-3.1-8B": "llama-3.1-8b-instant",
    "Llama-3.3-70B": "llama-3.3-70b-versatile",
    "Llama-4-Scout-17B": "meta-llama/llama-4-scout-17b-16e-instruct",
    "Llama-4-Maverick-17B": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "Qwen3-32B": "qwen/qwen3-32b",
}

# Vertex AI models: display_name -> model ID
# Native Gemini models use the model ID directly.
# Model Garden models require an endpoint_id (set via --endpoint flag or env var).
VERTEX_MODELS: dict[str, dict] = {
    "Gemini-2.5-Flash": {
        "model_id": "gemini-2.5-flash",
        "provider": "gemini",
        "max_tokens": 4096,
    },
    "Gemini-2.5-Pro": {
        "model_id": "gemini-2.5-pro",
        "provider": "gemini",
        "max_tokens": 4096,
    },
}

# Experiment parameters
MAX_REPAIR_ROUNDS: int = 5    # 1 initial + 4 repair attempts
EXECUTION_TIMEOUT: int = 15   # seconds per code execution
REQUEST_DELAY: float = 5.0    # seconds between API calls (rate limiting)
TEMPERATURE: float = 0.0      # greedy decoding for reproducibility
MAX_TOKENS: int = 2048

# Paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = BASE_DIR / "results"
DATA_DIR: Path = Path(__file__).resolve().parent / "data"
