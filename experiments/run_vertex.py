#!/usr/bin/env python3
"""Run self-repair experiments using Vertex AI (Gemini + Model Garden).

This script mirrors run_experiment.py but uses the Vertex AI client instead
of Groq. It supports all three benchmarks: HumanEval, MBPP, and LiveCodeBench.

Usage:
    # Run Gemini 2.5 Flash on HumanEval
    python -m experiments.run_vertex --models Gemini-2.5-Flash --benchmark humaneval

    # Run on all benchmarks
    python -m experiments.run_vertex --models Gemini-2.5-Flash --benchmark all

    # Run with cost estimation first (dry run)
    python -m experiments.run_vertex --models Gemini-2.5-Flash --benchmark humaneval --dry-run

    # Run just a few problems to verify setup
    python -m experiments.run_vertex --models Gemini-2.5-Flash --benchmark humaneval --num-problems 5

Prerequisites:
    pip install google-genai
    gcloud auth application-default login
    Set VERTEX_PROJECT and VERTEX_LOCATION in .env
"""
from __future__ import annotations

import argparse
import json
import time

from experiments.config import (
    VERTEX_MODELS, MAX_REPAIR_ROUNDS, RESULTS_DIR,
    VERTEX_PROJECT, VERTEX_LOCATION,
)
from experiments.data_loader import load_humaneval, load_mbpp
from experiments.code_executor import execute_solution
from experiments.self_repair import build_initial_prompt, build_repair_prompt, extract_code
from experiments.vertex_client import create_vertex_client, call_vertex_model


VERTEX_RESULTS_DIR = RESULTS_DIR / "vertex"


def run_single_problem(
    client,
    model_id: str,
    problem: dict,
    max_rounds: int,
    model_config: dict | None = None,
) -> dict:
    """Run self-repair loop for one problem using Vertex AI."""
    entry_point = problem["entry_point"]
    test_code = problem["test"]
    prompt = problem["prompt"]

    messages = build_initial_prompt(problem)
    rounds = []

    max_tokens = model_config.get("max_tokens") if model_config else None

    for round_num in range(max_rounds):
        raw_response, usage = call_vertex_model(
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
    model_config: dict,
    problems: list[dict],
    max_rounds: int,
    benchmark: str = "humaneval",
) -> list[dict]:
    """Run experiment for one Vertex AI model across all problems."""
    model_id = model_config["model_id"]
    client = create_vertex_client()

    suffix = f"_{benchmark}" if benchmark != "humaneval" else ""
    results_file = VERTEX_RESULTS_DIR / f"{model_name}{suffix}.json"

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

        result = run_single_problem(client, model_id, problem, max_rounds, model_config)
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


def estimate_cost(model_name: str, benchmark: str, num_problems: int):
    """Print estimated cost for a run (rough approximation).

    Based on average token usage from Groq experiments (~600 tokens/problem
    for initial, ~1500 tokens total with repairs).
    """
    # Approximate pricing per 1M tokens (as of early 2025)
    pricing = {
        "Gemini-2.5-Flash": {"input": 0.15, "output": 0.60},      # very cheap
        "Gemini-2.5-Pro": {"input": 1.25, "output": 10.00},        # expensive
    }

    p = pricing.get(model_name, {"input": 1.0, "output": 5.0})

    # Rough estimates: ~500 input tokens + ~300 output tokens per round,
    # average ~2.5 rounds per problem (some pass R0, some need all 5)
    avg_input_per_problem = 500 * 2.5   # 1250 tokens
    avg_output_per_problem = 300 * 2.5  # 750 tokens

    total_input = num_problems * avg_input_per_problem
    total_output = num_problems * avg_output_per_problem

    cost_input = (total_input / 1_000_000) * p["input"]
    cost_output = (total_output / 1_000_000) * p["output"]
    total_cost = cost_input + cost_output

    # Convert to INR (approximate)
    inr = total_cost * 85

    print(f"\n  Estimated cost for {model_name} on {benchmark} ({num_problems} problems):")
    print(f"    Input:  ~{total_input:,.0f} tokens  = ${cost_input:.3f}")
    print(f"    Output: ~{total_output:,.0f} tokens  = ${cost_output:.3f}")
    print(f"    Total:  ~${total_cost:.3f} (~{inr:.1f} INR)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Vertex AI Self-Repair Experiment Runner")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Models to run. Choices: {list(VERTEX_MODELS.keys())}",
    )
    parser.add_argument("--max-rounds", type=int, default=MAX_REPAIR_ROUNDS)
    parser.add_argument("--num-problems", type=int, default=None)
    parser.add_argument(
        "--benchmark", type=str, default="humaneval",
        choices=["humaneval", "mbpp", "all"],
        help="Benchmark to evaluate on",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Only estimate cost, don't run")
    args = parser.parse_args()

    if not VERTEX_PROJECT:
        print("ERROR: VERTEX_PROJECT not set. Add to .env or set environment variable.")
        print("  Example: VERTEX_PROJECT=my-gcp-project-id")
        return

    VERTEX_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_names = args.models or list(VERTEX_MODELS.keys())
    benchmarks = ["humaneval", "mbpp"] if args.benchmark == "all" else [args.benchmark]

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

        summary = {}

        for model_name in model_names:
            if model_name not in VERTEX_MODELS:
                print(f"Unknown model: {model_name}. Available: {list(VERTEX_MODELS.keys())}")
                continue

            model_config = VERTEX_MODELS[model_name]

            if args.dry_run:
                estimate_cost(model_name, benchmark, len(problems))
                continue

            print(f"{'='*60}")
            print(f" {model_name} ({model_config['model_id']})")
            print(f" Benchmark: {benchmark} | Project: {VERTEX_PROJECT}")
            print(f"{'='*60}")

            start = time.time()
            results = run_model_experiment(
                model_name, model_config, problems, args.max_rounds,
                benchmark=benchmark,
            )
            elapsed = time.time() - start

            passed = sum(1 for r in results if r["final_passed"])
            total = len(results)

            # Compute per-round pass rates
            round_pass = {}
            for r in results:
                for rd in r["rounds"]:
                    rn = rd["round"]
                    if rn not in round_pass:
                        round_pass[rn] = 0
                    if rd["passed"] and r["rounds_to_pass"] == rn:
                        round_pass[rn] += 1

            cumulative = 0
            print(f"\n  Per-round cumulative pass rates:")
            for rn in sorted(round_pass.keys()):
                cumulative += round_pass[rn]
                print(f"    R{rn}: {cumulative}/{total} ({100*cumulative/total:.1f}%)")

            total_tokens = sum(
                rd.get("usage", {}).get("total_tokens", 0)
                for r in results for rd in r["rounds"]
            )

            summary[model_name] = {
                "passed": passed,
                "total": total,
                "rate": round(passed / total, 4) if total > 0 else 0,
                "total_tokens": total_tokens,
                "time_seconds": round(elapsed, 1),
            }

            print(f"\n  Result: {passed}/{total} ({100*passed/total:.1f}%)")
            print(f"  Tokens: {total_tokens:,}")
            print(f"  Time: {elapsed:.0f}s\n")

        if not args.dry_run and summary:
            print("\n" + "=" * 60)
            print(f" SUMMARY ({benchmark.upper()})")
            print("=" * 60)
            for name, s in summary.items():
                print(f"  {name:20s}: {s['passed']:3d}/{s['total']:3d} "
                      f"({100*s['rate']:.1f}%)  tokens={s['total_tokens']:,}")

            summary_name = f"vertex_summary_{benchmark}.json"
            with open(VERTEX_RESULTS_DIR / summary_name, "w") as f:
                json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
