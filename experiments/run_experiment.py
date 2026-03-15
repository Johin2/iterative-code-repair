#!/usr/bin/env python3
"""Run self-repair experiments on HumanEval and MBPP benchmarks."""
from __future__ import annotations

import argparse
import json
import time

from experiments.config import MODELS, MAX_REPAIR_ROUNDS, RESULTS_DIR
from experiments.data_loader import load_humaneval, load_mbpp
from experiments.code_executor import execute_solution
from experiments.self_repair import build_initial_prompt, build_repair_prompt, extract_code
from experiments.api_client import create_client, call_model


def run_single_problem(
    client,
    model_id: str,
    problem: dict,
    max_rounds: int,
) -> dict:
    """Run self-repair loop for one problem."""
    entry_point = problem["entry_point"]
    test_code = problem["test"]
    prompt = problem["prompt"]

    messages = build_initial_prompt(problem)

    # Disable thinking for Qwen3 to avoid thinking-trace token overhead
    if "qwen" in model_id.lower():
        messages[-1]["content"] += "\n/no_think"

    rounds = []

    for round_num in range(max_rounds):
        raw_response, usage = call_model(client, model_id, messages)

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
    }


def run_model_experiment(
    model_name: str,
    model_id: str,
    problems: list[dict],
    max_rounds: int,
    benchmark: str = "humaneval",
) -> list[dict]:
    """Run experiment for one model across all problems."""
    client = create_client()

    if benchmark == "humaneval":
        results_file = RESULTS_DIR / f"{model_name}.json"
    else:
        results_file = RESULTS_DIR / f"{model_name}_{benchmark}.json"

    # Load existing results for resume
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

        result = run_single_problem(client, model_id, problem, max_rounds)
        result["model"] = model_name
        results.append(result)

        if result["final_passed"]:
            print(f" PASS (round {result['rounds_to_pass']})")
        else:
            print(f" FAIL (all {max_rounds} rounds)")

        # Save incrementally to prevent data loss
        seen = {r["task_id"] for r in results}
        save_data = results + [existing[tid] for tid in existing if tid not in seen]
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Self-Repair Experiment Runner")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Models to run. Choices: {list(MODELS.keys())}",
    )
    parser.add_argument("--max-rounds", type=int, default=MAX_REPAIR_ROUNDS)
    parser.add_argument("--num-problems", type=int, default=None)
    parser.add_argument(
        "--benchmark", type=str, default="humaneval",
        choices=["humaneval", "mbpp"],
        help="Benchmark dataset to evaluate on (default: humaneval)",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.benchmark == "mbpp":
        print("Loading MBPP sanitized dataset...")
        problems = load_mbpp()
    else:
        print("Loading HumanEval dataset...")
        problems = load_humaneval()
    if args.num_problems:
        problems = problems[:args.num_problems]
    print(f"Loaded {len(problems)} problems\n")

    model_names = args.models or list(MODELS.keys())

    summary = {}

    for model_name in model_names:
        if model_name not in MODELS:
            print(f"Unknown model: {model_name}. Skipping.")
            continue

        model_id = MODELS[model_name]
        print(f"{'='*60}")
        print(f" {model_name} ({model_id})")
        print(f"{'='*60}")

        start = time.time()
        results = run_model_experiment(
            model_name, model_id, problems, args.max_rounds, benchmark=args.benchmark,
        )
        elapsed = time.time() - start

        passed = sum(1 for r in results if r["final_passed"])
        total = len(results)
        summary[model_name] = {
            "passed": passed,
            "total": total,
            "rate": round(passed / total, 4) if total > 0 else 0,
            "time_seconds": round(elapsed, 1),
        }

        print(f"\n  Result: {passed}/{total} ({100*passed/total:.1f}%) in {elapsed:.0f}s\n")

    # Print summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    for name, s in summary.items():
        print(f"  {name:20s}: {s['passed']:3d}/{s['total']:3d} ({100*s['rate']:.1f}%)")

    # Save summary
    summary_name = "summary.json" if args.benchmark == "humaneval" else f"summary_{args.benchmark}.json"
    with open(RESULTS_DIR / summary_name, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
