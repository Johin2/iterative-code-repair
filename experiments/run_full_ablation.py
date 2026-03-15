#!/usr/bin/env python3
"""Extended prompt ablation: run all 3 strategies on all 5 models.

The original ablation (run_ablation.py) covered only Llama-3.3-70B and
Llama-4-Scout-17B. This script extends it to all 5 models to provide
complete coverage. Results are stored in the same results/ablation/
directory, so analyze_ablation.py picks them up automatically.
"""
from __future__ import annotations

import argparse
import time

from experiments.config import MODELS
from experiments.data_loader import load_humaneval
from experiments.run_ablation import (
    ABLATION_DIR,
    PROMPT_STRATEGIES,
    run_ablation,
)


def main():
    parser = argparse.ArgumentParser(description="Full Prompt Ablation (All 5 Models)")
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS.keys()),
        help=f"Models to test (default: all). Choices: {list(MODELS.keys())}",
    )
    parser.add_argument(
        "--strategies", nargs="+", default=list(PROMPT_STRATEGIES.keys()),
        help=f"Strategies: {list(PROMPT_STRATEGIES.keys())}",
    )
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="Max rounds for ablation (default 3: R0-R2)")
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

            # Check if results already exist and are complete
            results_file = ABLATION_DIR / f"{model_name}_{strategy}.json"
            if results_file.exists():
                import json
                with open(results_file) as f:
                    existing = json.load(f)
                if len(existing) >= 164:
                    passed = sum(1 for r in existing if r["final_passed"])
                    print(f"  {model_name}/{strategy}: already complete "
                          f"({passed}/{len(existing)}), skipping.")
                    key = f"{model_name}_{strategy}"
                    summary[key] = {
                        "model": model_name,
                        "strategy": strategy,
                        "passed": passed,
                        "total": len(existing),
                        "rate": round(passed / len(existing) * 100, 1),
                    }
                    continue

            print(f"\n{'='*60}")
            print(f" {model_name} | Strategy: {strategy}")
            print(f"{'='*60}")

            start = time.time()
            results = run_ablation(
                model_name, model_id, problems, args.max_rounds, strategy,
            )
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
    print(" FULL ABLATION SUMMARY")
    print(f"{'='*60}")
    for key, s in summary.items():
        print(f"  {key:40s}: {s['passed']:3d}/{s['total']:3d} ({s['rate']:.1f}%)")

    import json
    with open(ABLATION_DIR / "full_ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {ABLATION_DIR / 'full_ablation_summary.json'}")


if __name__ == "__main__":
    main()
