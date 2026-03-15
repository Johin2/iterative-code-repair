import os
from pathlib import Path

# Load .env file
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _value = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _value.strip())

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Models to evaluate: display_name -> groq model ID
MODELS = {
    "Llama-3.1-8B": "llama-3.1-8b-instant",
    "Llama-3.3-70B": "llama-3.3-70b-versatile",
    "Llama-4-Scout-17B": "meta-llama/llama-4-scout-17b-16e-instruct",
    "Llama-4-Maverick-17B": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "Qwen3-32B": "qwen/qwen3-32b",
}

MAX_REPAIR_ROUNDS = 5       # 1 initial + 4 repair attempts
EXECUTION_TIMEOUT = 15      # seconds per code execution
REQUEST_DELAY = 5.0         # seconds between API calls (rate limiting)
TEMPERATURE = 0.0
MAX_TOKENS = 2048

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = Path(__file__).parent / "data"
