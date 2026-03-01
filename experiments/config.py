"""
Configuration for TaskRouter experiments.
"""

# ── Model Tiers ──────────────────────────────────────────────
MODEL_TIERS = {
    "T1": {
        "name": "Gemini 2.5 Flash",
        "provider": "google",
        "model_id": "gemini-2.5-flash",
        "input_cost_per_1m": 0.15,
        "output_cost_per_1m": 0.60,
    },
    "T2": {
        "name": "GPT-4o-mini",
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "input_cost_per_1m": 0.15,
        "output_cost_per_1m": 0.60,
    },
    "T3": {
        "name": "Claude 3.5 Haiku",
        "provider": "anthropic",
        "model_id": "claude-haiku-4-5-20251001",
        "input_cost_per_1m": 0.80,
        "output_cost_per_1m": 4.00,
    },
    "T4": {
        "name": "GPT-4o",
        "provider": "openai",
        "model_id": "gpt-4o",
        "input_cost_per_1m": 2.50,
        "output_cost_per_1m": 10.00,
    },
}

# ── Sub-Task Types ───────────────────────────────────────────
SUBTASK_TYPES = ["EXPL", "COMP", "LOC", "PATCH", "TEST", "VER"]

# ── Routing Priors (bias toward cheaper tiers for easy tasks) ─
# Values represent the default tier assignment when difficulty is unknown.
TYPE_DEFAULT_TIER = {
    "EXPL": "T1",
    "COMP": "T2",
    "LOC": "T2",
    "PATCH": "T4",
    "TEST": "T3",
    "VER": "T1",
}

# ── Difficulty Thresholds ────────────────────────────────────
# Each tier can handle sub-tasks up to this difficulty level (0-1).
# These are initial estimates; calibrated from experiment data.
DEFAULT_CAPABILITY_THRESHOLDS = {
    "T1": {"EXPL": 0.8, "COMP": 0.4, "LOC": 0.3, "PATCH": 0.2, "TEST": 0.3, "VER": 0.7},
    "T2": {"EXPL": 0.9, "COMP": 0.6, "LOC": 0.5, "PATCH": 0.3, "TEST": 0.5, "VER": 0.85},
    "T3": {"EXPL": 0.95, "COMP": 0.8, "LOC": 0.7, "PATCH": 0.6, "TEST": 0.7, "VER": 0.95},
    "T4": {"EXPL": 1.0, "COMP": 1.0, "LOC": 1.0, "PATCH": 1.0, "TEST": 1.0, "VER": 1.0},
}

# ── Experiment Settings ──────────────────────────────────────
MAX_AGENT_TURNS = 30
MAX_TOKENS_PER_TURN = 4096
TEMPERATURE = 0.0
NUM_TASKS = 50  # Start with 50 for feasibility; scale to 300 for full paper
RESULTS_DIR = "results"
