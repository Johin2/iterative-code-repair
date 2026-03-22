#!/usr/bin/env python3
"""Combined analysis: merge Groq + Vertex AI results into unified tables/figures.

This script loads results from both the original Groq experiments and the new
Vertex AI experiments, producing unified comparison tables and updated figures
for the paper.

Usage:
    python -m experiments.analyze_combined
    python -m experiments.analyze_combined --benchmark humaneval
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

from experiments.config import MODELS, VERTEX_MODELS, RESULTS_DIR, MAX_REPAIR_ROUNDS

FIGURES_DIR = RESULTS_DIR.parent / "paper" / "figures"
VERTEX_RESULTS_DIR = RESULTS_DIR / "vertex"

# Unified color/marker scheme for all 7 models
MODEL_ORDER = [
    "Llama-3.1-8B",
    "Llama-3.3-70B",
    "Llama-4-Scout-17B",
    "Llama-4-Maverick-17B",
    "Qwen3-32B",
    "Gemini-2.5-Flash",
    "Gemini-2.5-Pro",
]

MODEL_DISPLAY = {
    "Llama-3.1-8B": "Llama 3.1 8B",
    "Llama-3.3-70B": "Llama 3.3 70B",
    "Llama-4-Scout-17B": "Scout 17B (16E)",
    "Llama-4-Maverick-17B": "Maverick 17B (128E)",
    "Qwen3-32B": "Qwen3 32B",
    "Gemini-2.5-Flash": "Gemini 2.5 Flash",
    "Gemini-2.5-Pro": "Gemini 2.5 Pro",
}

MODEL_SIZES = {
    "Llama-3.1-8B": "8B",
    "Llama-3.3-70B": "70B",
    "Llama-4-Scout-17B": "17B (16E)",
    "Llama-4-Maverick-17B": "17B (128E)",
    "Qwen3-32B": "32B",
    "Gemini-2.5-Flash": "~undisclosed",
    "Gemini-2.5-Pro": "~undisclosed",
}

MODEL_FAMILY = {
    "Llama-3.1-8B": "Meta",
    "Llama-3.3-70B": "Meta",
    "Llama-4-Scout-17B": "Meta",
    "Llama-4-Maverick-17B": "Meta",
    "Qwen3-32B": "Alibaba",
    "Gemini-2.5-Flash": "Google",
    "Gemini-2.5-Pro": "Google",
}

MODEL_ARCH = {
    "Llama-3.1-8B": "Dense",
    "Llama-3.3-70B": "Dense",
    "Llama-4-Scout-17B": "MoE (16E)",
    "Llama-4-Maverick-17B": "MoE (128E)",
    "Qwen3-32B": "Dense",
    "Gemini-2.5-Flash": "MoE",
    "Gemini-2.5-Pro": "Dense",
}

COLORS = {
    "Llama-3.1-8B": "#e74c3c",
    "Llama-3.3-70B": "#3498db",
    "Llama-4-Scout-17B": "#2ecc71",
    "Llama-4-Maverick-17B": "#9b59b6",
    "Qwen3-32B": "#f39c12",
    "Gemini-2.5-Flash": "#1abc9c",
    "Gemini-2.5-Pro": "#e67e22",
}

MARKERS = {
    "Llama-3.1-8B": "o",
    "Llama-3.3-70B": "s",
    "Llama-4-Scout-17B": "^",
    "Llama-4-Maverick-17B": "D",
    "Qwen3-32B": "v",
    "Gemini-2.5-Flash": "P",
    "Gemini-2.5-Pro": "*",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(benchmark="humaneval") -> dict[str, list[dict]]:
    """Load results from both Groq and Vertex AI directories."""
    suffix = f"_{benchmark}" if benchmark == "mbpp" else ""
    all_results = {}

    # Groq results
    for model_name in MODELS:
        path = RESULTS_DIR / f"{model_name}{suffix}.json"
        if path.exists():
            with open(path) as f:
                all_results[model_name] = json.load(f)

    # Vertex AI results
    for model_name in VERTEX_MODELS:
        path = VERTEX_RESULTS_DIR / f"{model_name}{suffix}.json"
        if path.exists():
            with open(path) as f:
                all_results[model_name] = json.load(f)

    return all_results


# ---------------------------------------------------------------------------
# Metrics (same as analyze_results.py)
# ---------------------------------------------------------------------------

def compute_cumulative_pass_rates(all_results, max_rounds):
    rates = {}
    for model_name, results in all_results.items():
        total = len(results)
        cumulative = []
        for r in range(max_rounds):
            passed = sum(
                1 for res in results
                if res["final_passed"] and res["rounds_to_pass"] <= r
            )
            cumulative.append(passed / total * 100)
        rates[model_name] = cumulative
    return rates


def compute_error_distribution(all_results):
    dist = {}
    for model_name, results in all_results.items():
        errors = defaultdict(int)
        for res in results:
            if res["rounds"] and not res["rounds"][0]["passed"]:
                etype = res["rounds"][0].get("error_type", "unknown")
                errors[etype] += 1
        dist[model_name] = dict(errors)
    return dist


def compute_repair_success_by_error(all_results):
    stats = defaultdict(lambda: {"repaired": 0, "total": 0})
    for results in all_results.values():
        for res in results:
            if res["rounds"] and not res["rounds"][0]["passed"]:
                etype = res["rounds"][0].get("error_type", "unknown")
                stats[etype]["total"] += 1
                if res["final_passed"]:
                    stats[etype]["repaired"] += 1
    return dict(stats)


def compute_token_usage(all_results):
    usage = {}
    for model_name, results in all_results.items():
        total_prompt = 0
        total_completion = 0
        for res in results:
            for r in res.get("rounds", []):
                u = r.get("usage", {})
                total_prompt += u.get("prompt_tokens", 0)
                total_completion += u.get("completion_tokens", 0)
        usage[model_name] = {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
        }
    return usage


def compute_round_gain(all_results, max_rounds):
    gains = {}
    for model_name, results in all_results.items():
        total = len(results)
        per_round = []
        for r in range(max_rounds):
            newly = sum(
                1 for res in results
                if res["final_passed"] and res["rounds_to_pass"] == r
            )
            per_round.append(newly / total * 100)
        gains[model_name] = per_round
    return gains


# ---------------------------------------------------------------------------
# Plotting (updated for 7 models)
# ---------------------------------------------------------------------------

def _ordered_models(results_dict):
    """Return models in canonical order."""
    return [m for m in MODEL_ORDER if m in results_dict]


def plot_cumulative_pass_rates(rates, max_rounds, benchmark="humaneval", tag=""):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name in _ordered_models(rates):
        cumul = rates[name]
        ax.plot(
            range(max_rounds), cumul,
            marker=MARKERS[name], color=COLORS[name],
            linewidth=2.2, markersize=8,
            label=MODEL_DISPLAY.get(name, name),
        )

    bname = "HumanEval" if benchmark == "humaneval" else "MBPP"
    ax.set_xlabel("Repair Round", fontsize=12)
    ax.set_ylabel("Cumulative Pass Rate (%)", fontsize=12)
    ax.set_title(f"Cumulative Pass@1 by Repair Round ({bname})", fontsize=13)
    ax.set_xticks(range(max_rounds))
    ax.set_xticklabels([f"R{i}" for i in range(max_rounds)])
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fname = f"cumulative_pass_rate_combined{tag}"
    plt.savefig(FIGURES_DIR / f"{fname}.png", dpi=300)
    plt.savefig(FIGURES_DIR / f"{fname}.pdf")
    plt.close()
    print(f"  Saved {fname}")


def plot_error_distribution(error_dist, benchmark="humaneval", tag=""):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    models = _ordered_models(error_dist)
    all_types = sorted({t for d in error_dist.values() for t in d})

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(models))
    n = len(all_types)
    width = 0.8 / max(n, 1)
    cmap = plt.cm.Set3(np.linspace(0, 1, max(n, 1)))

    for j, etype in enumerate(all_types):
        vals = [error_dist[m].get(etype, 0) for m in models]
        ax.bar(x + j * width, vals, width, label=etype, color=cmap[j])

    bname = "HumanEval" if benchmark == "humaneval" else "MBPP"
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Error Type Distribution at R0 ({bname})", fontsize=13)
    ax.set_xticks(x + width * n / 2)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], rotation=20, ha="right")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fname = f"error_distribution_combined{tag}"
    plt.savefig(FIGURES_DIR / f"{fname}.png", dpi=300)
    plt.savefig(FIGURES_DIR / f"{fname}.pdf")
    plt.close()
    print(f"  Saved {fname}")


def plot_cross_benchmark(he_rates, mbpp_rates, max_rounds):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    common_models = [m for m in MODEL_ORDER if m in he_rates and m in mbpp_rates]
    if not common_models:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)

    for name in common_models:
        ax1.plot(range(max_rounds), he_rates[name],
                 marker=MARKERS[name], color=COLORS[name],
                 linewidth=2.2, markersize=8,
                 label=MODEL_DISPLAY.get(name, name))
        ax2.plot(range(max_rounds), mbpp_rates[name],
                 marker=MARKERS[name], color=COLORS[name],
                 linewidth=2.2, markersize=8,
                 label=MODEL_DISPLAY.get(name, name))

    for ax, title in [(ax1, "HumanEval"), (ax2, "MBPP")]:
        ax.set_xlabel("Repair Round", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(range(max_rounds))
        ax.set_xticklabels([f"R{i}" for i in range(max_rounds)])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

    ax1.set_ylabel("Cumulative Pass Rate (%)", fontsize=12)
    plt.suptitle("Cross-Benchmark Self-Repair Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cross_benchmark_combined.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "cross_benchmark_combined.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved cross_benchmark_combined")


def plot_improvement_per_round(gains, max_rounds, benchmark="humaneval", tag=""):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    models = _ordered_models(gains)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    n = len(models)
    width = 0.8 / n

    for i, name in enumerate(models):
        ax.bar(
            [x + i * width for x in range(max_rounds)],
            gains[name], width=width,
            label=MODEL_DISPLAY.get(name, name),
            color=COLORS[name], edgecolor="white",
        )

    bname = "HumanEval" if benchmark == "humaneval" else "MBPP"
    ax.set_xlabel("Repair Round", fontsize=12)
    ax.set_ylabel("Newly Passing (%)", fontsize=12)
    ax.set_title(f"New Problems Solved Per Round ({bname})", fontsize=13)
    ax.set_xticks([x + width * n / 2 for x in range(max_rounds)])
    ax.set_xticklabels([f"R{i}" for i in range(max_rounds)])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fname = f"improvement_per_round_combined{tag}"
    plt.savefig(FIGURES_DIR / f"{fname}.png", dpi=300)
    plt.savefig(FIGURES_DIR / f"{fname}.pdf")
    plt.close()
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def print_main_results_table(all_results, rates, max_rounds, benchmark="humaneval"):
    """Print the main cumulative pass rate table (Table I / II equivalent)."""
    bname = "HumanEval" if benchmark == "humaneval" else "MBPP Sanitized"
    n_problems = len(next(iter(all_results.values())))

    print(f"\n% -- Table: Cumulative Pass Rates ({bname}, {n_problems} problems) --")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(f"\\caption{{Cumulative pass@1 (\\%) on {bname} ({n_problems} problems) by repair round. "
          f"$R_0$ is the initial attempt; $R_1$--$R_4$ are repair rounds. "
          f"$\\Delta$ denotes the total self-repair gain ($R_4 - R_0$) in percentage points.}}")
    print(f"\\label{{tab:{benchmark}_combined}}")
    print(r"\begin{tabular}{llcccccc}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Family} & $R_0$ & $R_1$ & $R_2$ & $R_3$ & $R_4$ & $\Delta$ \\")
    print(r"\midrule")

    for name in _ordered_models(rates):
        cumul = rates[name]
        family = MODEL_FAMILY.get(name, "?")
        delta = cumul[-1] - cumul[0]
        vals = " & ".join(f"{v:.1f}" for v in cumul)
        print(f"{MODEL_DISPLAY.get(name, name)} & {family} & {vals} & +{delta:.1f} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def print_cross_benchmark_table(he_results, he_rates, mbpp_results, mbpp_rates):
    """Print cross-benchmark summary table."""
    all_models = [m for m in MODEL_ORDER if m in he_results or m in mbpp_results]

    print("\n% -- Table: Cross-Benchmark Summary (7 models) --")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Cross-benchmark self-repair summary across seven models and three families.}")
    print(r"\label{tab:cross_combined}")
    print(r"\resizebox{\columnwidth}{!}{%")
    print(r"\begin{tabular}{llccccccc}")
    print(r"\toprule")
    print(r" & & \multicolumn{3}{c}{\textbf{HumanEval (164)}} & "
          r"\multicolumn{3}{c}{\textbf{MBPP (257)}} \\")
    print(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}")
    print(r"\textbf{Model} & \textbf{Family} & "
          r"$R_0$ & Final & $\Delta$ & "
          r"$R_0$ & Final & $\Delta$ \\")
    print(r"\midrule")

    for name in all_models:
        family = MODEL_FAMILY.get(name, "?")
        display = MODEL_DISPLAY.get(name, name)

        if name in he_results:
            he = he_results[name]
            he_total = len(he)
            he_r0 = sum(1 for r in he if r["rounds"] and r["rounds"][0]["passed"])
            he_final = sum(1 for r in he if r["final_passed"])
            he_str = (f"{100*he_r0/he_total:.1f} & {100*he_final/he_total:.1f} & "
                      f"+{100*(he_final-he_r0)/he_total:.1f}")
        else:
            he_str = "-- & -- & --"

        if name in mbpp_results:
            mb = mbpp_results[name]
            mb_total = len(mb)
            mb_r0 = sum(1 for r in mb if r["rounds"] and r["rounds"][0]["passed"])
            mb_final = sum(1 for r in mb if r["final_passed"])
            mb_str = (f"{100*mb_r0/mb_total:.1f} & {100*mb_final/mb_total:.1f} & "
                      f"+{100*(mb_final-mb_r0)/mb_total:.1f}")
        else:
            mb_str = "-- & -- & --"

        print(f"{display} & {family} & {he_str} & {mb_str} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}%")
    print(r"}")
    print(r"\end{table}")


def print_token_usage_table(he_usage, mbpp_usage=None):
    """Print token usage comparison table."""
    all_models = [m for m in MODEL_ORDER if m in he_usage]

    print("\n% -- Table: Token Usage --")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Total token usage across all problems by model and benchmark.}")
    print(r"\label{tab:tokens_combined}")

    if mbpp_usage:
        print(r"\begin{tabular}{lrrr}")
        print(r"\toprule")
        print(r"\textbf{Model} & \textbf{HumanEval} & \textbf{MBPP} & \textbf{Total} \\")
        print(r"\midrule")
        for name in all_models:
            he_t = he_usage.get(name, {}).get("total_tokens", 0)
            mb_t = mbpp_usage.get(name, {}).get("total_tokens", 0) if mbpp_usage else 0
            display = MODEL_DISPLAY.get(name, name)
            print(f"{display} & {he_t:,} & {mb_t:,} & {he_t+mb_t:,} \\\\")
    else:
        print(r"\begin{tabular}{lr}")
        print(r"\toprule")
        print(r"\textbf{Model} & \textbf{Tokens} \\")
        print(r"\midrule")
        for name in all_models:
            t = he_usage.get(name, {}).get("total_tokens", 0)
            display = MODEL_DISPLAY.get(name, name)
            print(f"{display} & {t:,} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_benchmark(benchmark, max_rounds):
    tag = f"_{benchmark}" if benchmark == "mbpp" else ""

    print(f"\n{'='*60}")
    print(f" {benchmark.upper()} Analysis (Combined)")
    print(f"{'='*60}")

    all_results = load_all_results(benchmark)
    if not all_results:
        print(f"No {benchmark} results found.")
        return None, None, None

    print(f"Loaded {len(all_results)} models: {', '.join(_ordered_models(all_results))}")
    for name in _ordered_models(all_results):
        res = all_results[name]
        print(f"  {name}: {len(res)} problems")

    rates = compute_cumulative_pass_rates(all_results, max_rounds)
    error_dist = compute_error_distribution(all_results)
    repair_stats = compute_repair_success_by_error(all_results)
    token_usage = compute_token_usage(all_results)
    gains = compute_round_gain(all_results, max_rounds)

    print("\nGenerating combined figures...")
    plot_cumulative_pass_rates(rates, max_rounds, benchmark, tag)
    plot_error_distribution(error_dist, benchmark, tag)
    plot_improvement_per_round(gains, max_rounds, benchmark, tag)

    print(f"\n QUICK SUMMARY ({benchmark.upper()})")
    print("-" * 70)
    for name in _ordered_models(all_results):
        results = all_results[name]
        total = len(results)
        r0 = sum(1 for r in results if r["rounds"] and r["rounds"][0]["passed"])
        final = sum(1 for r in results if r["final_passed"])
        family = MODEL_FAMILY.get(name, "?")
        print(f"  {MODEL_DISPLAY.get(name, name):25s} ({family:7s}): "
              f"R0={100*r0/total:5.1f}% -> Final={100*final/total:5.1f}% "
              f"(+{100*(final-r0)/total:.1f}pp)")

    return all_results, rates, token_usage


def main():
    parser = argparse.ArgumentParser(description="Combined Groq + Vertex AI Analysis")
    parser.add_argument("--benchmark", default="all",
                        choices=["humaneval", "mbpp", "all"])
    args = parser.parse_args()

    max_rounds = MAX_REPAIR_ROUNDS

    he_results, he_rates, he_tokens = None, None, None
    mbpp_results, mbpp_rates, mbpp_tokens = None, None, None

    if args.benchmark in ("humaneval", "all"):
        he_results, he_rates, he_tokens = analyze_benchmark("humaneval", max_rounds)

    if args.benchmark in ("mbpp", "all"):
        mbpp_results, mbpp_rates, mbpp_tokens = analyze_benchmark("mbpp", max_rounds)

    # Cross-benchmark plot
    if he_rates and mbpp_rates:
        print("\nGenerating cross-benchmark figure...")
        plot_cross_benchmark(he_rates, mbpp_rates, max_rounds)

    # LaTeX tables
    print("\n" + "=" * 60)
    print(" LaTeX Tables (Combined)")
    print("=" * 60)

    if he_results and he_rates:
        print_main_results_table(he_results, he_rates, max_rounds, "humaneval")

    if mbpp_results and mbpp_rates:
        print_main_results_table(mbpp_results, mbpp_rates, max_rounds, "mbpp")

    if he_results and mbpp_results:
        print_cross_benchmark_table(he_results, he_rates, mbpp_results, mbpp_rates)

    if he_tokens:
        print_token_usage_table(he_tokens, mbpp_tokens)


if __name__ == "__main__":
    main()
