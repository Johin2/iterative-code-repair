#!/usr/bin/env python3
"""Resampling baseline using Vertex AI models.

Usage:
    python -m experiments.run_vertex_resampling --models Gemini-2.5-Flash --benchmark humaneval
    python -m experiments.run_vertex_resampling --models Gemini-2.5-Flash --benchmark both
"""
from __future__ import annotations

import argparse
import json
import math
import time

from experiments.config import VERTEX_MODELS, RESULTS_DIR, VERTEX_PROJECT
from experiments.data_loader import load_humaneval, load_mbpp
from experiments.code_executor import execute_solution
from experiments.self_repair import build_initial_prompt, extract_code
from experiments.vertex_client import create_vertex_client, call_vertex_model

VERTEX_RESAMPLING_DIR = RESULTS_DIR / "vertex" / "resampling"


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from Chen et al. (2021)."""
    if n < k:
        return 1.0 if c > 0 else 0.0
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def run_single_problem(
    client,
    model_id: str,
    problem: dict,
    num_samples: int,
    temperature: float,
    max_tokens: int,
) -> dict:
    """Generate num_samples independent solutions for one problem."""
    entry_point = problem["entry_point"]
    test_code = problem["test"]
    prompt = problem["prompt"]

    samples = []

    for sample_idx in range(num_samples):
        messages = build_initial_prompt(problem)

        raw_response, usage = call_vertex_model(
            client, model_id, messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if raw_response is None:
            samples.append({
                "sample": sample_idx,
                "passed": False,
                "error_type": "api_error",
                "error_message": "API call failed after retries",
                "usage": {},
            })
            continue

        code = extract_code(raw_response, entry_point, prompt)
        exec_result = execute_solution(code, test_code, entry_point)

        samples.append({
            "sample": sample_idx,
            "passed": exec_result["passed"],
            "error_type": exec_result["error_type"],
            "error_message": exec_result["error_message"][:500],
            "usage": usage,
        })

    n = len(samples)
    c = sum(1 for s in samples if s["passed"])
    total_tokens = sum(s.get("usage", {}).get("total_tokens", 0) for s in samples)

    return {
        "task_id": problem["task_id"],
        "model": "",
        "num_samples": n,
        "num_correct": c,
        "pass_at_1": pass_at_k(n, c, 1),
        "pass_at_2": pass_at_k(n, c, 2),
        "pass_at_3": pass_at_k(n, c, 3),
        "pass_at_5": pass_at_k(n, c, 5) if n >= 5 else pass_at_k(n, c, n),
        "total_tokens": total_tokens,
        "samples": samples,
    }


def run_model_resampling(
    model_name: str,
    model_config: dict,
    problems: list[dict],
    num_samples: int,
    temperature: float,
    max_tokens: int,
    benchmark: str = "humaneval",
) -> list[dict]:
    """Run resampling for one Vertex AI model."""
    model_id = model_config["model_id"]
    client = create_vertex_client()
    results_file = VERTEX_RESAMPLING_DIR / f"{model_name}_resampling_{benchmark}.json"

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
            client, model_id, problem, num_samples, temperature, max_tokens,
        )
        result["model"] = model_name
        results.append(result)

        print(f" {result['num_correct']}/{num_samples} correct")

        seen = {r["task_id"] for r in results}
        save_data = results + [existing[tid] for tid in existing if tid not in seen]
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Vertex AI Resampling Baseline")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Models to run. Choices: {list(VERTEX_MODELS.keys())}",
    )
    parser.add_argument(
        "--benchmark", type=str, default="humaneval",
        choices=["humaneval", "mbpp", "both"],
    )
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--num-problems", type=int, default=None)
    args = parser.parse_args()

    if not VERTEX_PROJECT:
        print("ERROR: VERTEX_PROJECT not set.")
        return

    VERTEX_RESAMPLING_DIR.mkdir(parents=True, exist_ok=True)
    model_names = args.models or list(VERTEX_MODELS.keys())
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

        summary = {}

        for model_name in model_names:
            if model_name not in VERTEX_MODELS:
                print(f"Unknown model: {model_name}. Skipping.")
                continue

            model_config = VERTEX_MODELS[model_name]
            print(f"{'='*60}")
            print(f" {model_name} | Resampling (n={args.num_samples}, t={args.temperature})")
            print(f" Benchmark: {benchmark}")
            print(f"{'='*60}")

            start = time.time()
            results = run_model_resampling(
                model_name, model_config, problems,
                args.num_samples, args.temperature, args.max_tokens,
                benchmark=benchmark,
            )
            elapsed = time.time() - start

            avg_pass1 = sum(r["pass_at_1"] for r in results) / len(results) * 100
            avg_pass5 = sum(r["pass_at_5"] for r in results) / len(results) * 100
            total_tokens = sum(r["total_tokens"] for r in results)

            summary[model_name] = {
                "pass_at_1": round(avg_pass1, 1),
                "pass_at_5": round(avg_pass5, 1),
                "total_tokens": total_tokens,
                "num_samples": args.num_samples,
                "temperature": args.temperature,
                "time_seconds": round(elapsed, 1),
            }

            print(f"\n  pass@1={avg_pass1:.1f}%  pass@5={avg_pass5:.1f}%")
            print(f"  Tokens: {total_tokens:,}  Time: {elapsed:.0f}s\n")

        print(f"\n{'='*60}")
        print(f" RESAMPLING SUMMARY ({benchmark.upper()}) - Vertex AI")
        print(f"{'='*60}")
        for name, s in summary.items():
            print(f"  {name:25s}: pass@1={s['pass_at_1']:5.1f}%  "
                  f"pass@5={s['pass_at_5']:5.1f}%  tokens={s['total_tokens']:,}")

        summary_file = VERTEX_RESAMPLING_DIR / f"vertex_resampling_summary_{benchmark}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
