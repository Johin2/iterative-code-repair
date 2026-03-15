#!/usr/bin/env python3
"""Analyze resampling results and compare with self-repair."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.config import MODELS, RESULTS_DIR

RESAMPLING_DIR = RESULTS_DIR / "resampling"
FIGURES_DIR = RESULTS_DIR.parent / "paper" / "figures"


def load_resampling_results(benchmark: str = "humaneval") -> dict[str, list[dict]]:
    results = {}
    for model_name in MODELS:
        path = RESAMPLING_DIR / f"{model_name}_resampling_{benchmark}.json"
        if path.exists():
            with open(path) as f:
                results[model_name] = json.load(f)
    return results


def load_repair_results(benchmark: str = "humaneval") -> dict[str, list[dict]]:
    suffix = f"_{benchmark}" if benchmark == "mbpp" else ""
    results = {}
    for model_name in MODELS:
        path = RESULTS_DIR / f"{model_name}{suffix}.json"
        if path.exists():
            with open(path) as f:
                results[model_name] = json.load(f)
    return results


def compute_repair_stats(results: list[dict], max_rounds: int = 5) -> dict:
    """Compute cumulative pass rates and total tokens for self-repair."""
    total = len(results)
    cumulative = []
    for r in range(max_rounds):
        passed = sum(
            1 for res in results
            if res["final_passed"] and res["rounds_to_pass"] <= r
        )
        cumulative.append(passed / total * 100)

    total_tokens = sum(
        u.get("total_tokens", 0)
        for res in results
        for rd in res.get("rounds", [])
        for u in [rd.get("usage", {})]
    )
    return {
        "cumulative": cumulative,
        "total_tokens": total_tokens,
        "total_problems": total,
    }


def compute_resampling_stats(results: list[dict]) -> dict:
    """Compute pass@k and total tokens for resampling."""
    total = len(results)
    avg_pass1 = sum(r["pass_at_1"] for r in results) / total * 100
    avg_pass2 = sum(r.get("pass_at_2", 0) for r in results) / total * 100
    avg_pass3 = sum(r.get("pass_at_3", 0) for r in results) / total * 100
    avg_pass5 = sum(r["pass_at_5"] for r in results) / total * 100
    total_tokens = sum(r.get("total_tokens", 0) for r in results)

    return {
        "pass_at_k": {1: avg_pass1, 2: avg_pass2, 3: avg_pass3, 5: avg_pass5},
        "total_tokens": total_tokens,
        "total_problems": total,
    }


def plot_repair_vs_resampling(
    repair_data: dict[str, dict],
    resample_data: dict[str, dict],
    benchmark: str = "humaneval",
):
    """Side-by-side bar chart: self-repair final rate vs resampling pass@5."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    models = [m for m in repair_data if m in resample_data]
    if not models:
        print("No overlapping models for comparison")
        return

    repair_rates = [repair_data[m]["cumulative"][-1] for m in models]
    resample_rates = [resample_data[m]["pass_at_k"][5] for m in models]
    repair_tokens = [repair_data[m]["total_tokens"] for m in models]
    resample_tokens = [resample_data[m]["total_tokens"] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Pass rate comparison
    x = np.arange(len(models))
    width = 0.35
    ax1.bar(x - width/2, repair_rates, width, label="Self-Repair (final)", color="#3498db")
    ax1.bar(x + width/2, resample_rates, width, label="Resampling (pass@5)", color="#e74c3c")
    ax1.set_ylabel("Pass Rate (%)", fontsize=12)
    ax1.set_title(f"Pass Rate: Self-Repair vs Resampling ({benchmark.upper()})", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim(0, 105)

    # Token efficiency comparison
    repair_eff = [r / max(t, 1) for r, t in zip(repair_rates, repair_tokens)]
    resample_eff = [r / max(t, 1) for r, t in zip(resample_rates, resample_tokens)]
    ax2.bar(x - width/2, [t/1000 for t in repair_tokens], width,
            label="Self-Repair", color="#3498db")
    ax2.bar(x + width/2, [t/1000 for t in resample_tokens], width,
            label="Resampling", color="#e74c3c")
    ax2.set_ylabel("Total Tokens (K)", fontsize=12)
    ax2.set_title("Token Usage Comparison", fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha="right")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    tag = f"_{benchmark}" if benchmark != "humaneval" else ""
    plt.savefig(FIGURES_DIR / f"repair_vs_resampling{tag}.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / f"repair_vs_resampling{tag}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved repair_vs_resampling{tag}")


def print_comparison_table(
    repair_data: dict[str, dict],
    resample_data: dict[str, dict],
    benchmark: str = "humaneval",
):
    """Print comparison table."""
    models = [m for m in MODELS if m in repair_data and m in resample_data]

    bname = "HumanEval" if benchmark == "humaneval" else "MBPP"
    print(f"\n{'='*80}")
    print(f" SELF-REPAIR vs RESAMPLING ({bname})")
    print(f"{'='*80}")
    print(f"{'Model':25s} | {'Repair Final':>12s} | {'Resample@5':>10s} | "
          f"{'Repair Tok':>10s} | {'Resamp Tok':>10s} | {'Winner':>8s}")
    print("-" * 80)

    for model in models:
        rep = repair_data[model]
        res = resample_data[model]
        repair_rate = rep["cumulative"][-1]
        resample_rate = res["pass_at_k"][5]
        repair_tok = rep["total_tokens"]
        resample_tok = res["total_tokens"]

        # Winner = higher rate; if tied, lower tokens
        if repair_rate > resample_rate + 0.5:
            winner = "Repair"
        elif resample_rate > repair_rate + 0.5:
            winner = "Resample"
        else:
            winner = "Tie"

        print(f"{model:25s} | {repair_rate:11.1f}% | {resample_rate:9.1f}% | "
              f"{repair_tok:>10,} | {resample_tok:>10,} | {winner:>8s}")

    # LaTeX table
    print(f"\n% -- LaTeX: Self-Repair vs Resampling ({bname}) --")
    print("\\begin{table}[t]")
    print("\\centering")
    print(f"\\caption{{Self-repair vs.\\ independent resampling on {bname}. "
          "Self-repair uses 5 sequential attempts (greedy); resampling uses 5 "
          "independent samples (temperature 0.8).}}")
    print(f"\\label{{tab:resampling_{benchmark}}}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("\\textbf{Model} & \\textbf{Repair} & \\textbf{Resample} & "
          "\\textbf{Rep.\\,Tok} & \\textbf{Res.\\,Tok} \\\\")
    print(" & \\textbf{Final \\%} & \\textbf{pass@5 \\%} & & \\\\")
    print("\\midrule")
    for model in models:
        rep = repair_data[model]
        res = resample_data[model]
        rr = rep["cumulative"][-1]
        rs = res["pass_at_k"][5]
        rt = rep["total_tokens"]
        st = res["total_tokens"]
        rr_s = f"\\textbf{{{rr:.1f}}}" if rr > rs else f"{rr:.1f}"
        rs_s = f"\\textbf{{{rs:.1f}}}" if rs > rr else f"{rs:.1f}"
        print(f"{model} & {rr_s} & {rs_s} & {rt//1000}K & {st//1000}K \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    for benchmark in ["humaneval", "mbpp"]:
        resample_results = load_resampling_results(benchmark)
        repair_results = load_repair_results(benchmark)

        if not resample_results:
            print(f"No resampling results for {benchmark}")
            continue

        # Compute stats
        repair_stats = {m: compute_repair_stats(r) for m, r in repair_results.items()}
        resample_stats = {m: compute_resampling_stats(r) for m, r in resample_results.items()}

        print_comparison_table(repair_stats, resample_stats, benchmark)
        plot_repair_vs_resampling(repair_stats, resample_stats, benchmark)


if __name__ == "__main__":
    main()
