#!/usr/bin/env python3
"""Run LiveCodeBench self-repair experiments using Vertex AI.

Usage:
    python -m experiments.run_vertex_livecodebench --models Gemini-2.5-Flash
    python -m experiments.run_vertex_livecodebench --models Gemini-2.5-Flash --num-problems 50
"""
from __future__ import annotations

import argparse
import json
import time

from experiments.config import VERTEX_MODELS, MAX_REPAIR_ROUNDS, RESULTS_DIR, VERTEX_PROJECT
from experiments.run_livecodebench import (
    load_livecodebench,
    execute_livecodebench,
    build_initial_prompt_lcb,
    build_repair_prompt_lcb,
    _extract_code_lcb,
    LIVECODEBENCH_DIR,
)
from experiments.vertex_client import create_vertex_client, call_vertex_model


VERTEX_LCB_DIR = RESULTS_DIR / "vertex" / "livecodebench"


def run_single_problem(
    client,
    model_id: str,
    problem: dict,
    max_rounds: int,
) -> dict:
    """Run self-repair loop for one LiveCodeBench problem via Vertex AI."""
    entry_point = problem["entry_point"]
    is_function = problem["is_function"]
    test_cases = problem["test_cases"]

    messages = build_initial_prompt_lcb(problem)
    rounds = []

    for round_num in range(max_rounds):
        raw_response, usage = call_vertex_model(client, model_id, messages)

        if raw_response is None:
            rounds.append({
                "round": round_num,
                "passed": False,
                "error_type": "api_error",
                "error_message": "API call failed after retries",
                "usage": {},
            })
            break

        code = _extract_code_lcb(raw_response)
        exec_result = execute_livecodebench(
            code, test_cases, is_function, entry_point,
        )

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
                "difficulty": problem["difficulty"],
                "final_passed": True,
                "rounds_to_pass": round_num,
                "total_rounds": round_num + 1,
                "rounds": rounds,
            }

        messages.append({"role": "assistant", "content": raw_response})
        messages.append({"role": "user", "content": build_repair_prompt_lcb(exec_result["error_message"])})

    return {
        "task_id": problem["task_id"],
        "difficulty": problem["difficulty"],
        "final_passed": False,
        "rounds_to_pass": -1,
        "total_rounds": len(rounds),
        "rounds": rounds,
    }


def run_model_experiment(
    model_name: str,
    model_config: dict,
    problems: list[dict],
    max_rounds: int,
) -> list[dict]:
    """Run LiveCodeBench experiment for one Vertex AI model."""
    model_id = model_config["model_id"]
    client = create_vertex_client()
    results_file = VERTEX_LCB_DIR / f"{model_name}_livecodebench.json"

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
            print(f" FAIL (all {result['total_rounds']} rounds)")

        # Save incrementally
        seen = {r["task_id"] for r in results}
        save_data = results + [existing[tid] for tid in existing if tid not in seen]
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Vertex AI LiveCodeBench Experiment")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Models to run. Choices: {list(VERTEX_MODELS.keys())}",
    )
    parser.add_argument("--max-rounds", type=int, default=MAX_REPAIR_ROUNDS)
    parser.add_argument("--num-problems", type=int, default=None)
    args = parser.parse_args()

    if not VERTEX_PROJECT:
        print("ERROR: VERTEX_PROJECT not set. Add to .env or set environment variable.")
        return

    VERTEX_LCB_DIR.mkdir(parents=True, exist_ok=True)
    model_names = args.models or list(VERTEX_MODELS.keys())

    print("Loading LiveCodeBench dataset...")
    problems = load_livecodebench(max_problems=args.num_problems)
    print(f"Total problems: {len(problems)}")

    diff_counts: dict[str, int] = {}
    for p in problems:
        d = p.get("difficulty", "unknown")
        diff_counts[d] = diff_counts.get(d, 0) + 1
    for d, c in sorted(diff_counts.items()):
        print(f"  {d}: {c}")
    print()

    summary = {}

    for model_name in model_names:
        if model_name not in VERTEX_MODELS:
            print(f"Unknown model: {model_name}. Skipping.")
            continue

        model_config = VERTEX_MODELS[model_name]
        print(f"{'='*60}")
        print(f" {model_name} | LiveCodeBench ({len(problems)} problems)")
        print(f"{'='*60}")

        start = time.time()
        results = run_model_experiment(model_name, model_config, problems, args.max_rounds)
        elapsed = time.time() - start

        passed = sum(1 for r in results if r["final_passed"])
        total = len(results)
        r0_passed = sum(
            1 for r in results
            if r["rounds"] and r["rounds"][0]["passed"]
        )

        diff_stats: dict[str, dict] = {}
        for r in results:
            d = r.get("difficulty", "unknown")
            if d not in diff_stats:
                diff_stats[d] = {"total": 0, "r0": 0, "final": 0}
            diff_stats[d]["total"] += 1
            if r["rounds"] and r["rounds"][0]["passed"]:
                diff_stats[d]["r0"] += 1
            if r["final_passed"]:
                diff_stats[d]["final"] += 1

        summary[model_name] = {
            "r0_passed": r0_passed,
            "final_passed": passed,
            "total": total,
            "r0_rate": round(100 * r0_passed / total, 1) if total else 0,
            "final_rate": round(100 * passed / total, 1) if total else 0,
            "gain_pp": round(100 * (passed - r0_passed) / total, 1) if total else 0,
            "by_difficulty": {
                d: {
                    "total": s["total"],
                    "r0_rate": round(100 * s["r0"] / s["total"], 1) if s["total"] else 0,
                    "final_rate": round(100 * s["final"] / s["total"], 1) if s["total"] else 0,
                }
                for d, s in diff_stats.items()
            },
            "time_seconds": round(elapsed, 1),
        }

        print(f"\n  R0: {r0_passed}/{total} ({100*r0_passed/total:.1f}%)")
        print(f"  Final: {passed}/{total} ({100*passed/total:.1f}%)")
        print(f"  Gain: +{100*(passed-r0_passed)/total:.1f}pp")
        for d, s in sorted(diff_stats.items()):
            print(f"    {d}: {s['r0']}/{s['total']} -> {s['final']}/{s['total']}")
        print(f"  Time: {elapsed:.0f}s\n")

    print(f"\n{'='*60}")
    print(" LIVECODEBENCH SUMMARY (Vertex AI)")
    print(f"{'='*60}")
    for name, s in summary.items():
        print(f"  {name:25s}: R0={s['r0_rate']:5.1f}% -> Final={s['final_rate']:5.1f}% "
              f"(+{s['gain_pp']:.1f}pp)")

    with open(VERTEX_LCB_DIR / "vertex_livecodebench_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
