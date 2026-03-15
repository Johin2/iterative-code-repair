#!/usr/bin/env python3
"""Prompt ablation experiment: compare repair prompt strategies.

Tests three repair prompt variants on a subset of models/problems:
  1. minimal   — current approach (error message only)
  2. explain   — ask model to explain the bug first, then fix
  3. cot       — chain-of-thought: think step by step before fixing
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    MODELS, MAX_REPAIR_ROUNDS, REQUEST_DELAY,
    RESULTS_DIR, GROQ_API_KEY, TEMPERATURE, MAX_TOKENS,
)
from data_loader import load_humaneval
from code_executor import execute_solution
from self_repair import build_initial_prompt, extract_code

from groq import Groq

ABLATION_DIR = RESULTS_DIR / "ablation"

# --- Prompt variants ---

def build_repair_minimal(error_message):
    """Current approach: just the error message."""
    if len(error_message) > 1500:
        error_message = error_message[:1500] + "\n... (truncated)"
    return (
        f"Your code produced an error when tested:\n\n"
        f"{error_message}\n\n"
        f"Please fix the code. Return ONLY the corrected Python function, "
        f"no explanations, no markdown formatting."
    )


def build_repair_explain(error_message):
    """Explain-then-fix: ask model to explain the bug before fixing."""
    if len(error_message) > 1500:
        error_message = error_message[:1500] + "\n... (truncated)"
    return (
        f"Your code produced an error when tested:\n\n"
        f"{error_message}\n\n"
        f"First, explain in 1-2 sentences what went wrong and why. "
        f"Then provide the corrected Python function. "
        f"Format your response as:\n"
        f"Bug: <your explanation>\n"
        f"```python\n<corrected code>\n```"
    )


def build_repair_cot(error_message):
    """Chain-of-thought: think step by step."""
    if len(error_message) > 1500:
        error_message = error_message[:1500] + "\n... (truncated)"
    return (
        f"Your code produced an error when tested:\n\n"
        f"{error_message}\n\n"
        f"Let's think step by step to fix this:\n"
        f"1. What does the error message tell us?\n"
        f"2. What is the root cause in the code?\n"
        f"3. What is the correct fix?\n\n"
        f"After your analysis, provide the corrected Python function in a code block."
    )


PROMPT_STRATEGIES = {
    "minimal": build_repair_minimal,
    "explain": build_repair_explain,
    "cot": build_repair_cot,
}


def run_single_problem(client, model_id, problem, max_rounds, strategy_name):
    """Run self-repair loop for one problem with a specific prompt strategy."""
    entry_point = problem["entry_point"]
    test_code = problem["test"]
    prompt = problem["prompt"]
    build_repair = PROMPT_STRATEGIES[strategy_name]

    messages = build_initial_prompt(problem)

    if "qwen" in model_id.lower():
        messages[-1]["content"] += "\n/no_think"

    rounds = []

    for round_num in range(max_rounds):
        time.sleep(REQUEST_DELAY)

        raw_response = None
        usage = {}
        for attempt in range(6):
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                raw_response = response.choices[0].message.content or ""
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                break
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str:
                    wait = 60 * (attempt + 1)
                    print(f" [rl {wait}s]", end="", flush=True)
                    time.sleep(wait)
                else:
                    print(f" [err: {type(e).__name__}]", end="", flush=True)
                    time.sleep(10)

        if raw_response is None:
            rounds.append({
                "round": round_num,
                "passed": False,
                "error_type": "api_error",
                "error_message": "API call failed after retries",
                "usage": {},
            })
            break

        code = extract_code(raw_response, entry_point, prompt)
        exec_result = execute_solution(code, test_code, entry_point)

        rounds.append({
            "round": round_num,
            "passed": exec_result["passed"],
            "error_type": exec_result["error_type"],
            "error_message": exec_result["error_message"][:500],
            "usage": usage,
        })

        if exec_result["passed"]:
            return {
                "task_id": problem["task_id"],
                "final_passed": True,
                "rounds_to_pass": round_num,
                "total_rounds": round_num + 1,
                "rounds": rounds,
                "strategy": strategy_name,
            }

        messages.append({"role": "assistant", "content": raw_response})
        messages.append({"role": "user", "content": build_repair(exec_result["error_message"])})

    return {
        "task_id": problem["task_id"],
        "final_passed": False,
        "rounds_to_pass": -1,
        "total_rounds": len(rounds),
        "rounds": rounds,
        "strategy": strategy_name,
    }


def run_ablation(model_name, model_id, problems, max_rounds, strategy_name):
    """Run ablation for one model + one strategy."""
    client = Groq(api_key=GROQ_API_KEY)
    results_file = ABLATION_DIR / f"{model_name}_{strategy_name}.json"

    existing = {}
    if results_file.exists():
        with open(results_file, "r") as f:
            for r in json.load(f):
                existing[r["task_id"]] = r

    results = []

    for i, problem in enumerate(problems):
        task_id = problem["task_id"]

        if task_id in existing:
            results.append(existing[task_id])
            continue

        print(f"  [{i+1:3d}/{len(problems)}] {task_id}", end="", flush=True)

        result = run_single_problem(client, model_id, problem, max_rounds, strategy_name)
        result["model"] = model_name
        results.append(result)

        status = f"PASS (R{result['rounds_to_pass']})" if result["final_passed"] else f"FAIL"
        print(f" {status}")

        seen = {r["task_id"] for r in results}
        save_data = results + [existing[tid] for tid in existing if tid not in seen]
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Prompt Ablation Experiment")
    parser.add_argument(
        "--models", nargs="+", default=["Llama-3.3-70B", "Llama-4-Scout-17B"],
        help="Models to test",
    )
    parser.add_argument(
        "--strategies", nargs="+", default=list(PROMPT_STRATEGIES.keys()),
        help=f"Strategies: {list(PROMPT_STRATEGIES.keys())}",
    )
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="Max rounds for ablation (default 3, faster)")
    args = parser.parse_args()

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading HumanEval dataset...")
    problems = load_humaneval()
    print(f"Loaded {len(problems)} problems\n")

    summary = {}

    for model_name in args.models:
        if model_name not in MODELS:
            print(f"Unknown model: {model_name}. Skipping.")
            continue

        model_id = MODELS[model_name]

        for strategy in args.strategies:
            if strategy not in PROMPT_STRATEGIES:
                print(f"Unknown strategy: {strategy}. Skipping.")
                continue

            print(f"\n{'='*60}")
            print(f" {model_name} | Strategy: {strategy}")
            print(f"{'='*60}")

            start = time.time()
            results = run_ablation(model_name, model_id, problems, args.max_rounds, strategy)
            elapsed = time.time() - start

            passed = sum(1 for r in results if r["final_passed"])
            total = len(results)
            key = f"{model_name}_{strategy}"
            summary[key] = {
                "model": model_name,
                "strategy": strategy,
                "passed": passed,
                "total": total,
                "rate": round(passed / total * 100, 1) if total > 0 else 0,
                "time_seconds": round(elapsed, 1),
            }
            print(f"\n  Result: {passed}/{total} ({100*passed/total:.1f}%) in {elapsed:.0f}s")

    # Print summary
    print(f"\n{'='*60}")
    print(" ABLATION SUMMARY")
    print(f"{'='*60}")
    for key, s in summary.items():
        print(f"  {key:40s}: {s['passed']:3d}/{s['total']:3d} ({s['rate']:.1f}%)")

    with open(ABLATION_DIR / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {ABLATION_DIR / 'ablation_summary.json'}")


if __name__ == "__main__":
    main()
