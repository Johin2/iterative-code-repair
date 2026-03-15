#!/usr/bin/env python3
"""Analyze prompt ablation results and generate comparison figure + LaTeX table."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.config import RESULTS_DIR

ABLATION_DIR = RESULTS_DIR / "ablation"
FIGURES_DIR = RESULTS_DIR.parent / "paper" / "figures"

STRATEGY_LABELS = {
    "minimal": "Minimal (error only)",
    "explain": "Explain-then-fix",
    "cot": "Chain-of-thought",
}

COLORS = {
    "minimal": "#e74c3c",
    "explain": "#3498db",
    "cot": "#2ecc71",
}


def load_ablation_results():
    """Load all ablation result files.

    Also loads original experiment results as the 'minimal' baseline
    for any model that has ablation data but no explicit minimal run.
    """
    results = {}
    if not ABLATION_DIR.exists():
        print("No ablation directory found.")
        return results

    # Load ablation-specific results
    for path in sorted(ABLATION_DIR.glob("*.json")):
        if path.name == "ablation_summary.json":
            continue
        stem = path.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        model_name, strategy = parts
        with open(path) as f:
            data = json.load(f)
        results[(model_name, strategy)] = data

    # Always prefer original experiment results as 'minimal' baseline
    # (they use the exact same minimal prompt and are complete)
    models_with_ablation = set(m for m, s in results if s != "minimal")
    for model_name in models_with_ablation:
        orig_path = RESULTS_DIR / f"{model_name}.json"
        if orig_path.exists():
            with open(orig_path) as f:
                data = json.load(f)
            results[(model_name, "minimal")] = data
            print(f"  Using original results as minimal baseline for {model_name}")

    # Filter out incomplete runs (< 164 problems for HumanEval)
    to_remove = []
    for key, data in results.items():
        if len(data) < 164:
            print(f"  WARNING: {key[0]}/{key[1]} only has {len(data)}/164 — skipping")
            to_remove.append(key)
    for key in to_remove:
        del results[key]

    return results


def compute_pass_rates(results):
    """Compute cumulative pass rates for each (model, strategy) pair."""
    rates = {}
    for (model, strategy), data in results.items():
        total = len(data)
        max_rounds = max(r["total_rounds"] for r in data) if data else 3
        cumulative = []
        for rd in range(max_rounds):
            passed = sum(
                1 for res in data
                if res["final_passed"] and res["rounds_to_pass"] <= rd
            )
            cumulative.append(passed / total * 100)
        rates[(model, strategy)] = cumulative
    return rates


def plot_ablation_comparison(rates):
    """Bar chart comparing strategies per model."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Group by model
    models = sorted(set(m for m, s in rates))
    strategies = sorted(set(s for m, s in rates))

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        x = np.arange(3)  # R0, R1, R2
        width = 0.25
        for j, strat in enumerate(strategies):
            key = (model, strat)
            if key not in rates:
                continue
            vals = rates[key][:3]  # up to 3 rounds
            bars = ax.bar(
                x + j * width, vals, width,
                label=STRATEGY_LABELS.get(strat, strat),
                color=COLORS.get(strat, f"C{j}"),
                edgecolor="white",
            )
            # Add value labels on top
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(model, fontsize=12)
        ax.set_xlabel("Round", fontsize=11)
        ax.set_xticks(x + width)
        ax.set_xticklabels(["R0", "R1", "R2"])
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Cumulative Pass@1 (%)", fontsize=11)
    plt.suptitle("Prompt Strategy Ablation on HumanEval", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ablation_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "ablation_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("Saved ablation_comparison figure")


def print_ablation_table(rates):
    """Print LaTeX table for ablation results."""
    models = sorted(set(m for m, s in rates))
    strategies = sorted(set(s for m, s in rates))

    print("\n% -- Table: Prompt Ablation Results --")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Prompt strategy ablation on HumanEval. Cumulative pass@1 (\\%) at each round for three repair prompt strategies.}")
    print("\\label{tab:ablation}")
    n_rounds = 3
    cols = "ll" + "c" * n_rounds + "c"
    print(f"\\begin{{tabular}}{{{cols}}}")
    print("\\toprule")
    round_hdrs = " & ".join([f"$R_{{{i}}}$" for i in range(n_rounds)])
    print(f"\\textbf{{Model}} & \\textbf{{Strategy}} & {round_hdrs} & $\\Delta$ \\\\")
    print("\\midrule")

    for model in models:
        first = True
        for strat in strategies:
            key = (model, strat)
            if key not in rates:
                continue
            vals = rates[key][:n_rounds]
            delta = vals[-1] - vals[0]
            val_str = " & ".join([f"{v:.1f}" for v in vals])
            model_str = model if first else ""
            strat_label = STRATEGY_LABELS.get(strat, strat)
            print(f"{model_str} & {strat_label} & {val_str} & +{delta:.1f} \\\\")
            first = False
        print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    results = load_ablation_results()
    if not results:
        print("No ablation results found. Run run_ablation.py first.")
        return

    print(f"Loaded {len(results)} (model, strategy) combinations:")
    for (model, strategy), data in sorted(results.items()):
        passed = sum(1 for r in data if r["final_passed"])
        total = len(data)
        print(f"  {model} / {strategy}: {passed}/{total} ({100*passed/total:.1f}%)")

    rates = compute_pass_rates(results)
    plot_ablation_comparison(rates)
    print_ablation_table(rates)


if __name__ == "__main__":
    main()
