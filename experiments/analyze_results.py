#!/usr/bin/env python3
"""Analyze self-repair experiment results and generate figures/tables."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.config import MODELS, RESULTS_DIR, MAX_REPAIR_ROUNDS

FIGURES_DIR = RESULTS_DIR.parent / "paper" / "figures"

COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
MARKERS = ["o", "s", "^", "D", "v"]

MODEL_SIZES = {
    "Llama-3.1-8B": "8B",
    "Llama-3.3-70B": "70B",
    "Llama-4-Scout-17B": "17B (16E)",
    "Llama-4-Maverick-17B": "17B (128E)",
    "Qwen3-32B": "32B",
}


def load_all_results(benchmark="humaneval"):
    suffix = "_mbpp" if benchmark == "mbpp" else ""
    all_results = {}
    for model_name in MODELS:
        path = RESULTS_DIR / f"{model_name}{suffix}.json"
        if path.exists():
            with open(path) as f:
                all_results[model_name] = json.load(f)
    return all_results


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


# -- Plotting --

def plot_cumulative_pass_rates(rates, max_rounds, benchmark="humaneval", tag=""):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (name, cumul) in enumerate(rates.items()):
        ax.plot(
            range(max_rounds), cumul,
            marker=MARKERS[i % len(MARKERS)],
            color=COLORS[i % len(COLORS)],
            linewidth=2.2, markersize=8, label=name,
        )

    bname = "HumanEval" if benchmark == "humaneval" else "MBPP"
    ax.set_xlabel("Repair Round", fontsize=12)
    ax.set_ylabel("Cumulative Pass Rate (%)", fontsize=12)
    ax.set_title(f"Cumulative Pass@1 by Repair Round ({bname})", fontsize=13)
    ax.set_xticks(range(max_rounds))
    ax.set_xticklabels([f"R{i}" for i in range(max_rounds)])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fname = f"cumulative_pass_rate{tag}"
    plt.savefig(FIGURES_DIR / f"{fname}.png", dpi=300)
    plt.savefig(FIGURES_DIR / f"{fname}.pdf")
    plt.close()
    print(f"  Saved {fname}")


def plot_error_distribution(error_dist, benchmark="humaneval", tag=""):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    all_types = sorted({t for d in error_dist.values() for t in d})
    models = list(error_dist.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
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
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fname = f"error_distribution{tag}"
    plt.savefig(FIGURES_DIR / f"{fname}.png", dpi=300)
    plt.savefig(FIGURES_DIR / f"{fname}.pdf")
    plt.close()
    print(f"  Saved {fname}")


def plot_repair_success(repair_stats, tag=""):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    types = sorted(repair_stats.keys())
    rates_list = []
    totals = []
    for t in types:
        s = repair_stats[t]
        rate = s["repaired"] / s["total"] * 100 if s["total"] > 0 else 0
        rates_list.append(rate)
        totals.append(s["total"])

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(types)), rates_list, color="#3498db", edgecolor="white")
    for bar, total in zip(bars, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"n={total}", ha="center", fontsize=9,
        )

    ax.set_xlabel("Error Type", fontsize=12)
    ax.set_ylabel("Repair Success Rate (%)", fontsize=12)
    ax.set_title("Self-Repair Success Rate by Initial Error Type", fontsize=13)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, rotation=30, ha="right")
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fname = f"repair_success_by_error{tag}"
    plt.savefig(FIGURES_DIR / f"{fname}.png", dpi=300)
    plt.savefig(FIGURES_DIR / f"{fname}.pdf")
    plt.close()
    print(f"  Saved {fname}")


def plot_improvement_per_round(gains, max_rounds, benchmark="humaneval", tag=""):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(gains.keys())
    n = len(models)
    width = 0.8 / n

    for i, (name, per_round) in enumerate(gains.items()):
        ax.bar(
            [x + i * width for x in range(max_rounds)],
            per_round, width=width,
            label=name, color=COLORS[i % len(COLORS)], edgecolor="white",
        )

    bname = "HumanEval" if benchmark == "humaneval" else "MBPP"
    ax.set_xlabel("Repair Round", fontsize=12)
    ax.set_ylabel("Newly Passing (%)", fontsize=12)
    ax.set_title(f"New Problems Solved Per Round ({bname})", fontsize=13)
    ax.set_xticks([x + width * n / 2 for x in range(max_rounds)])
    ax.set_xticklabels([f"R{i}" for i in range(max_rounds)])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fname = f"improvement_per_round{tag}"
    plt.savefig(FIGURES_DIR / f"{fname}.png", dpi=300)
    plt.savefig(FIGURES_DIR / f"{fname}.pdf")
    plt.close()
    print(f"  Saved {fname}")


def plot_cross_benchmark(he_rates, mbpp_rates, max_rounds):
    """Side-by-side comparison of HumanEval vs MBPP final pass rates."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    common_models = [m for m in he_rates if m in mbpp_rates]
    if not common_models:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for i, name in enumerate(common_models):
        ax1.plot(range(max_rounds), he_rates[name],
                 marker=MARKERS[i % len(MARKERS)], color=COLORS[i % len(COLORS)],
                 linewidth=2.2, markersize=8, label=name)
        ax2.plot(range(max_rounds), mbpp_rates[name],
                 marker=MARKERS[i % len(MARKERS)], color=COLORS[i % len(COLORS)],
                 linewidth=2.2, markersize=8, label=name)

    for ax, title in [(ax1, "HumanEval"), (ax2, "MBPP")]:
        ax.set_xlabel("Repair Round", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(range(max_rounds))
        ax.set_xticklabels([f"R{i}" for i in range(max_rounds)])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

    ax1.set_ylabel("Cumulative Pass Rate (%)", fontsize=12)
    plt.suptitle("Cross-Benchmark Self-Repair Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cross_benchmark.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "cross_benchmark.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved cross_benchmark")


# -- LaTeX Tables --

def print_latex_tables(all_results, rates, max_rounds, token_usage,
                       benchmark="humaneval", mbpp_results=None, mbpp_rates=None):
    bname = "HumanEval" if benchmark == "humaneval" else "MBPP"
    n_problems = next(iter(all_results.values()), [])
    n_problems = len(n_problems) if n_problems else "?"

    # Table 1: Cumulative pass rates
    print(f"\n% -- Table: Cumulative Pass Rates ({bname}) --")
    print("\\begin{table}[htbp]")
    print(f"\\caption{{Cumulative Pass@1 (\\%) by Repair Round on {bname}}}")
    print(f"\\label{{tab:cumulative_{benchmark}}}")
    print("\\centering")
    cols = "l" + "r" * (max_rounds + 1)
    print(f"\\begin{{tabular}}{{{cols}}}")
    print("\\toprule")
    hdrs = " & ".join([f"$R_{{{i}}}$" for i in range(max_rounds)])
    print(f"\\textbf{{Model}} & {hdrs} & $\\Delta$ \\\\")
    print("\\midrule")
    for name, cumul in rates.items():
        vals = " & ".join([f"{v:.1f}" for v in cumul])
        delta = cumul[-1] - cumul[0]
        print(f"{name} & {vals} & +{delta:.1f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}\n")

    # Table 2: Summary with both benchmarks if available
    if mbpp_results and mbpp_rates:
        print("% -- Table: Cross-Benchmark Summary --")
        print("\\begin{table}[htbp]")
        print("\\caption{Self-Repair Summary: HumanEval and MBPP}")
        print("\\label{tab:summary}")
        print("\\centering")
        print("\\begin{tabular}{llrrrrrr}")
        print("\\toprule")
        print(" & & \\multicolumn{3}{c}{\\textbf{HumanEval}} & "
              "\\multicolumn{3}{c}{\\textbf{MBPP}} \\\\")
        print("\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}")
        print("\\textbf{Model} & \\textbf{Size} & "
              "$R_0$ & Final & $\\Delta$ & "
              "$R_0$ & Final & $\\Delta$ \\\\")
        print("\\midrule")

        all_models = list(dict.fromkeys(
            list(all_results.keys()) + list(mbpp_results.keys())
        ))
        for name in all_models:
            sz = MODEL_SIZES.get(name, "?")
            # HumanEval
            if name in all_results:
                he = all_results[name]
                he_total = len(he)
                he_r0 = sum(1 for r in he if r["rounds"] and r["rounds"][0]["passed"])
                he_final = sum(1 for r in he if r["final_passed"])
                he_r0p = 100 * he_r0 / he_total
                he_fp = 100 * he_final / he_total
                he_delta = he_fp - he_r0p
                he_str = f"{he_r0p:.1f} & {he_fp:.1f} & +{he_delta:.1f}"
            else:
                he_str = "-- & -- & --"
            # MBPP
            if name in mbpp_results:
                mb = mbpp_results[name]
                mb_total = len(mb)
                mb_r0 = sum(1 for r in mb if r["rounds"] and r["rounds"][0]["passed"])
                mb_final = sum(1 for r in mb if r["final_passed"])
                mb_r0p = 100 * mb_r0 / mb_total
                mb_fp = 100 * mb_final / mb_total
                mb_delta = mb_fp - mb_r0p
                mb_str = f"{mb_r0p:.1f} & {mb_fp:.1f} & +{mb_delta:.1f}"
            else:
                mb_str = "-- & -- & --"

            print(f"{name} & {sz} & {he_str} & {mb_str} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")
    else:
        # Single benchmark summary
        print("% -- Table: Overall Summary --")
        print("\\begin{table}[htbp]")
        print(f"\\caption{{Self-Repair Summary on {bname}}}")
        print("\\label{tab:summary}")
        print("\\centering")
        print("\\begin{tabular}{lrrrrr}")
        print("\\toprule")
        print("\\textbf{Model} & \\textbf{Size} & $R_0$ & Final & $\\Delta$ & Tokens \\\\")
        print("\\midrule")
        for name, results in all_results.items():
            total = len(results)
            r0 = sum(1 for r in results if r["rounds"] and r["rounds"][0]["passed"])
            final = sum(1 for r in results if r["final_passed"])
            gain = final - r0
            tokens = token_usage.get(name, {}).get("total_tokens", 0)
            sz = MODEL_SIZES.get(name, "?")
            print(f"{name} & {sz} & {100*r0/total:.1f} & "
                  f"{100*final/total:.1f} & +{100*gain/total:.1f} & "
                  f"{tokens:,} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")


def analyze_benchmark(benchmark, max_rounds):
    tag = f"_{benchmark}" if benchmark == "mbpp" else ""
    print(f"\n{'='*60}")
    print(f" {benchmark.upper()} Analysis")
    print(f"{'='*60}")

    all_results = load_all_results(benchmark)
    if not all_results:
        print(f"No {benchmark} results found.")
        return None, None, None

    print(f"Loaded: {', '.join(all_results.keys())}")
    for name, res in all_results.items():
        print(f"  {name}: {len(res)} problems")

    rates = compute_cumulative_pass_rates(all_results, max_rounds)
    error_dist = compute_error_distribution(all_results)
    repair_stats = compute_repair_success_by_error(all_results)
    token_usage = compute_token_usage(all_results)
    gains = compute_round_gain(all_results, max_rounds)

    print("\nGenerating figures...")
    plot_cumulative_pass_rates(rates, max_rounds, benchmark, tag)
    plot_error_distribution(error_dist, benchmark, tag)
    plot_repair_success(repair_stats, tag)
    plot_improvement_per_round(gains, max_rounds, benchmark, tag)

    print(f"\n QUICK SUMMARY ({benchmark.upper()})")
    print("-" * 60)
    for name, results in all_results.items():
        total = len(results)
        r0 = sum(1 for r in results if r["rounds"] and r["rounds"][0]["passed"])
        final = sum(1 for r in results if r["final_passed"])
        print(f"  {name:25s}: R0={100*r0/total:5.1f}% -> Final={100*final/total:5.1f}% "
              f"(+{100*(final-r0)/total:.1f}pp)  [{total} problems]")

    return all_results, rates, token_usage


def main():
    parser = argparse.ArgumentParser()
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
    if he_results:
        print("\n" + "=" * 60)
        print(" LaTeX Tables")
        print("=" * 60)
        print_latex_tables(he_results, he_rates, max_rounds, he_tokens,
                           "humaneval", mbpp_results, mbpp_rates)

    if mbpp_results and mbpp_rates:
        mbpp_token_usage = compute_token_usage(mbpp_results) if mbpp_results else {}
        print_latex_tables(mbpp_results, mbpp_rates, max_rounds, mbpp_token_usage, "mbpp")


if __name__ == "__main__":
    main()
