#!/usr/bin/env python3
"""Run Qwen3-32B with thinking mode ENABLED.

The main experiments disable Qwen3's thinking mode via /no_think to ensure
fair comparison with non-reasoning models. This script runs Qwen3 with
thinking enabled to characterize the effect on both initial generation
and self-repair quality.
"""
from __future__ import annotations

import argparse
import json
import time

from experiments.config import MODELS, MAX_REPAIR_ROUNDS, RESULTS_DIR
from experiments.data_loader import load_humaneval, load_mbpp
from experiments.code_executor import execute_solution
from experiments.self_repair import build_initial_prompt, build_repair_prompt, extract_code
from experiments.api_client import create_client, call_model

QWEN_THINKING_DIR = RESULTS_DIR / "qwen_thinking"

QWEN_MODEL_NAME = "Qwen3-32B"
QWEN_MODEL_ID = MODELS[QWEN_MODEL_NAME]


def run_single_problem(
    client,
    model_id: str,
    problem: dict,
    max_rounds: int,
    max_tokens: int,
) -> dict:
    """Run self-repair loop for one problem WITHOUT disabling thinking."""
    entry_point = problem["entry_point"]
    test_code = problem["test"]
    prompt = problem["prompt"]

    # No /no_think appended — thinking mode stays enabled
    messages = build_initial_prompt(problem)

    rounds = []

    for round_num in range(max_rounds):
        raw_response, usage = call_model(
            client, model_id, messages, max_tokens=max_tokens,
        )

        if raw_response is None:
            rounds.append({
                "round": round_num,
                "passed": False,
                "error_type": "api_error",
                "error_message": "API call failed after retries",
                "usage": {},
            })
            break

        # extract_code already strips <think>...</think> tags
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
                "thinking_enabled": True,
            }

        # Build repair context for next round
        messages.append({"role": "assistant", "content": raw_response})
        messages.append({"role": "user", "content": build_repair_prompt(exec_result["error_message"])})

    return {
        "task_id": problem["task_id"],
        "final_passed": False,
        "rounds_to_pass": -1,
        "total_rounds": len(rounds),
        "rounds": rounds,
        "thinking_enabled": True,
    }


def run_experiment(
    problems: list[dict],
    max_rounds: int,
    max_tokens: int,
    benchmark: str = "humaneval",
) -> list[dict]:
    """Run Qwen3 thinking experiment across all problems."""
    client = create_client()

    suffix = f"_{benchmark}" if benchmark != "humaneval" else ""
    results_file = QWEN_THINKING_DIR / f"Qwen3-32B_thinking{suffix}.json"

    # Resume support
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

        result = run_single_problem(
            client, QWEN_MODEL_ID, problem, max_rounds, max_tokens,
        )
        result["model"] = QWEN_MODEL_NAME
        results.append(result)

        if result["final_passed"]:
            print(f" PASS (round {result['rounds_to_pass']})")
        else:
            print(f" FAIL (all {max_rounds} rounds)")

        # Save incrementally
        seen = {r["task_id"] for r in results}
        save_data = results + [existing[tid] for tid in existing if tid not in seen]
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-32B Thinking Mode Experiment",
    )
    parser.add_argument(
        "--benchmark", type=str, default="both",
        choices=["humaneval", "mbpp", "both"],
    )
    parser.add_argument("--max-rounds", type=int, default=MAX_REPAIR_ROUNDS)
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens (higher for thinking traces, default: 4096)")
    parser.add_argument("--num-problems", type=int, default=None)
    args = parser.parse_args()

    QWEN_THINKING_DIR.mkdir(parents=True, exist_ok=True)

    benchmarks = ["humaneval", "mbpp"] if args.benchmark == "both" else [args.benchmark]

    for benchmark in benchmarks:
        if benchmark == "mbpp":
            print("Loading MBPP sanitized dataset...")
            problems = load_mbpp()
        else:
            print("Loading HumanEval dataset...")
            problems = load_humaneval()
        if args.num_problems:
            problems = problems[:args.num_problems]
        print(f"Loaded {len(problems)} problems\n")

        print(f"{'='*60}")
        print(f" Qwen3-32B (thinking ENABLED) | {benchmark.upper()}")
        print(f"{'='*60}")

        start = time.time()
        results = run_experiment(
            problems, args.max_rounds, args.max_tokens, benchmark=benchmark,
        )
        elapsed = time.time() - start

        passed = sum(1 for r in results if r["final_passed"])
        total = len(results)
        r0_passed = sum(
            1 for r in results
            if r["rounds"] and r["rounds"][0]["passed"]
        )
        total_tokens = sum(
            u.get("total_tokens", 0)
            for r in results for rd in r.get("rounds", [])
            for u in [rd.get("usage", {})]
        )

        print(f"\n  R0: {r0_passed}/{total} ({100*r0_passed/total:.1f}%)")
        print(f"  Final: {passed}/{total} ({100*passed/total:.1f}%)")
        print(f"  Gain: +{100*(passed-r0_passed)/total:.1f}pp")
        print(f"  Tokens: {total_tokens:,}  Time: {elapsed:.0f}s\n")

        summary = {
            "model": QWEN_MODEL_NAME,
            "benchmark": benchmark,
            "thinking_enabled": True,
            "r0_passed": r0_passed,
            "final_passed": passed,
            "total": total,
            "r0_rate": round(100 * r0_passed / total, 1),
            "final_rate": round(100 * passed / total, 1),
            "gain_pp": round(100 * (passed - r0_passed) / total, 1),
            "total_tokens": total_tokens,
            "time_seconds": round(elapsed, 1),
        }
        suffix = f"_{benchmark}" if benchmark != "humaneval" else ""
        with open(QWEN_THINKING_DIR / f"summary{suffix}.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
