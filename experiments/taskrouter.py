"""
TaskRouter: Cost-Aware Sub-Task Model Routing.

Core routing logic that selects the cheapest model for each sub-task
based on type and estimated difficulty, with cascading fallback.
"""

import re
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from config import (
    MODEL_TIERS, SUBTASK_TYPES, TYPE_DEFAULT_TIER,
    DEFAULT_CAPABILITY_THRESHOLDS, MAX_TOKENS_PER_TURN, TEMPERATURE,
)
from llm_client import call_llm, LLMResponse, TokenTracker
from subtask_classifier import SUBTASK_PROMPTS


@dataclass
class RoutingDecision:
    subtask_type: str
    difficulty: float
    initial_tier: str
    final_tier: str
    cascaded: bool
    cascade_depth: int
    response: LLMResponse
    wasted_tokens: int = 0  # tokens from failed cascade attempts
    wasted_cost: float = 0.0


@dataclass
class TaskResult:
    task_id: str
    decisions: list = field(default_factory=list)
    patch_code: str = ""
    test_passed: bool = False
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    wasted_cost: float = 0.0


def estimate_difficulty(task: dict, subtask_type: str) -> float:
    """
    Estimate the difficulty of a sub-task based on code features.
    Returns a score in [0, 1].
    """
    code = task.get("code", "")
    test_code = task.get("test_code", "")

    # Structural features
    code_lines = len(code.strip().split("\n")) if code.strip() else 0
    test_lines = len(test_code.strip().split("\n")) if test_code.strip() else 0
    num_functions = len(re.findall(r"def \w+", code))
    num_classes = len(re.findall(r"class \w+", code))
    has_imports = 1 if re.search(r"^(import|from)", code, re.MULTILINE) else 0
    nesting_depth = max(
        (len(line) - len(line.lstrip())) // 4
        for line in code.split("\n") if line.strip()
    ) if code.strip() else 0

    # Complexity proxies
    num_conditionals = len(re.findall(r"\b(if|elif|else|while|for)\b", code))
    num_test_asserts = len(re.findall(r"assert", test_code))

    # Task metadata
    base_difficulty = task.get("difficulty", 0.5)

    # Type-specific adjustments
    type_multiplier = {
        "EXPL": 0.3,   # exploration is usually easy
        "COMP": 0.5,   # comprehension scales with code size
        "LOC": 0.6,    # localization is moderately hard
        "PATCH": 1.0,  # patch generation is the hardest
        "TEST": 0.7,   # test generation is medium
        "VER": 0.4,    # verification is usually easy
    }

    # Combine features into difficulty score
    raw = (
        0.3 * base_difficulty +
        0.1 * min(code_lines / 50, 1.0) +
        0.1 * min(num_functions / 5, 1.0) +
        0.1 * min(nesting_depth / 5, 1.0) +
        0.1 * min(num_conditionals / 8, 1.0) +
        0.1 * min(num_test_asserts / 6, 1.0) +
        0.1 * has_imports +
        0.1 * min(num_classes / 2, 1.0)
    )

    difficulty = min(raw * type_multiplier.get(subtask_type, 1.0), 1.0)
    return round(difficulty, 3)


def select_tier(subtask_type: str, difficulty: float,
                thresholds: dict = None) -> str:
    """Select the cheapest model tier that can handle this difficulty."""
    if thresholds is None:
        thresholds = DEFAULT_CAPABILITY_THRESHOLDS

    tier_order = ["T1", "T2", "T3", "T4"]
    for tier in tier_order:
        if thresholds[tier][subtask_type] >= difficulty:
            return tier

    return "T4"  # fallback to frontier


def confidence_check(subtask_type: str, response: LLMResponse,
                     task: dict) -> bool:
    """
    Check if the model's response meets quality thresholds.
    Returns True if the response is acceptable.
    """
    if not response.success or not response.content.strip():
        return False

    content = response.content.strip()

    if subtask_type == "EXPL":
        # Must contain some analysis (at least 20 chars)
        return len(content) > 20

    elif subtask_type == "COMP":
        # Must mention something specific about the code
        return len(content) > 50

    elif subtask_type == "LOC":
        # Must identify at least one specific line or location
        return ("line" in content.lower() or "bug" in content.lower() or
                "issue" in content.lower() or "error" in content.lower())

    elif subtask_type == "PATCH":
        # Must contain Python code
        has_code = "```python" in content or "def " in content or "class " in content
        # Must be substantially different from just echoing the input
        return has_code and len(content) > 30

    elif subtask_type == "TEST":
        # Must contain test functions
        return "def test_" in content or "assert" in content

    elif subtask_type == "VER":
        # Must contain a clear verdict
        return "VERDICT" in content.upper() or "PASS" in content.upper() or "FAIL" in content.upper()

    return True


def extract_code_from_response(response_text: str) -> str:
    """Extract Python code from markdown code fences."""
    # Try to find ```python ... ``` blocks
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try generic ``` blocks
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # If no code fences, try to find code by looking for def/class
    lines = response_text.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith(("def ", "class ", "import ", "from ")):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip()

    return response_text.strip()


def run_subtask(task: dict, subtask_type: str, tier: str,
                extra_context: dict = None) -> LLMResponse:
    """Run a single sub-task on a specific model tier."""
    prompt_info = SUBTASK_PROMPTS[subtask_type]
    extra = extra_context or {}

    system_prompt = (
        "You are an expert software engineer. Be precise and concise. "
        "Follow the instructions exactly."
    )

    # Build the user prompt from template
    user_prompt = prompt_info["prompt_template"].format(
        task_description=task.get("description", ""),
        code=task.get("code", ""),
        test_code=task.get("test_code", ""),
        original_code=extra.get("original_code", task.get("code", "")),
        fixed_code=extra.get("fixed_code", ""),
    )

    return call_llm(
        tier=tier,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS_PER_TURN,
    )


def run_task_with_routing(task: dict, strategy: str = "taskrouter",
                          fixed_tier: str = None) -> TaskResult:
    """
    Run a complete task through the agentic workflow with the specified strategy.

    Strategies:
    - "taskrouter": Full TaskRouter with difficulty estimation and cascading
    - "type_only": Fixed tier per sub-task type (no difficulty estimation)
    - "single": All sub-tasks on the same tier (specify fixed_tier)
    - "random": Random tier assignment
    """
    import random

    result = TaskResult(task_id=task["id"])
    subtask_sequence = ["EXPL", "COMP", "LOC", "PATCH", "TEST", "VER"]
    extra_context = {}

    for subtask_type in subtask_sequence:
        # ── Select tier based on strategy ──
        if strategy == "single":
            initial_tier = fixed_tier or "T4"
        elif strategy == "random":
            initial_tier = random.choice(["T1", "T2", "T3", "T4"])
        elif strategy == "type_only":
            initial_tier = TYPE_DEFAULT_TIER[subtask_type]
        elif strategy == "taskrouter":
            difficulty = estimate_difficulty(task, subtask_type)
            initial_tier = select_tier(subtask_type, difficulty)
        else:
            initial_tier = "T4"

        # ── Execute with cascading ──
        current_tier = initial_tier
        cascade_depth = 0
        wasted_tokens = 0
        wasted_cost = 0.0
        tier_order = ["T1", "T2", "T3", "T4"]
        start_idx = tier_order.index(current_tier)

        final_response = None
        for idx in range(start_idx, len(tier_order)):
            current_tier = tier_order[idx]
            response = run_subtask(task, subtask_type, current_tier, extra_context)

            if strategy in ("single", "random") or confidence_check(subtask_type, response, task):
                final_response = response
                break
            else:
                # Failed confidence check, cascade up
                wasted_tokens += response.input_tokens + response.output_tokens
                wasted_cost += response.cost
                cascade_depth += 1

        if final_response is None:
            # All tiers exhausted, use the last response
            final_response = response

        # ── Record decision ──
        decision = RoutingDecision(
            subtask_type=subtask_type,
            difficulty=estimate_difficulty(task, subtask_type) if strategy == "taskrouter" else -1,
            initial_tier=initial_tier,
            final_tier=current_tier,
            cascaded=cascade_depth > 0,
            cascade_depth=cascade_depth,
            response=final_response,
            wasted_tokens=wasted_tokens,
            wasted_cost=wasted_cost,
        )
        result.decisions.append(decision)
        result.total_cost += final_response.cost + wasted_cost
        result.total_input_tokens += final_response.input_tokens + wasted_tokens
        result.total_output_tokens += final_response.output_tokens
        result.wasted_cost += wasted_cost

        # ── Pass context to next sub-task ──
        if subtask_type == "PATCH":
            result.patch_code = extract_code_from_response(final_response.content)
            extra_context["fixed_code"] = result.patch_code
            extra_context["original_code"] = task.get("code", "")

    return result


def evaluate_patch(task: dict, patch_code: str) -> bool:
    """
    Evaluate whether a generated patch passes the task's tests.
    Executes the patched code + tests in a sandboxed exec().
    """
    if not patch_code.strip():
        return False

    test_code = task.get("test_code", "")
    if not test_code.strip():
        return False

    # Combine patched code with test code
    full_code = patch_code + "\n\n" + test_code

    # Find all test function names
    test_funcs = re.findall(r"def (test_\w+)", test_code)

    # Execute in isolated namespace
    namespace = {}
    try:
        exec(full_code, namespace)
        for func_name in test_funcs:
            if func_name in namespace:
                namespace[func_name]()
        return True
    except Exception:
        return False
