"""
Analyze experiment results and generate LaTeX tables + figures for the paper.
"""

import json
import os
import sys

RESULTS_DIR = "results"
OUTPUT_DIR = "paper_outputs"


def load_results() -> dict:
    """Load all experiment results."""
    path = os.path.join(RESULTS_DIR, "all_results.json")
    if not os.path.exists(path):
        print(f"Results file not found: {path}")
        print("Run the experiments first: python run_experiment.py")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def compute_cost_reduction(results: dict) -> dict:
    """Compute cost reduction relative to single_frontier."""
    baseline_cost = results["single_frontier"]["summary"]["total_cost"]
    reductions = {}
    for name, data in results.items():
        cost = data["summary"]["total_cost"]
        reduction = ((baseline_cost - cost) / baseline_cost * 100) if baseline_cost > 0 else 0
        reductions[name] = round(reduction, 1)
    return reductions


def generate_overall_table(results: dict) -> str:
    """Generate Table II: Overall Performance Comparison."""
    reductions = compute_cost_reduction(results)

    display_names = {
        "single_frontier": "Single-Frontier (T4)",
        "single_small": "Single-Small (T1)",
        "single_medium": "Single-Medium (T2)",
        "single_large": "Single-Large (T3)",
        "random_routing": "Random Routing",
        "type_only_routing": "Type-Only Routing",
        "taskrouter": "\\textbf{TaskRouter}",
    }

    row_order = [
        "single_frontier", "single_small", "single_medium", "single_large",
        "random_routing", "type_only_routing", "taskrouter",
    ]

    rows = []
    for name in row_order:
        if name not in results:
            continue
        s = results[name]["summary"]
        reduction = reductions[name]
        reduction_str = "---" if name == "single_frontier" else f"{reduction:.1f}\\%"
        rows.append(
            f"    {display_names.get(name, name)} & "
            f"{s['resolve_rate']:.1f}\\% & "
            f"\\${s['total_cost']:.2f} & "
            f"\\${s['cost_per_resolved']:.2f} & "
            f"{reduction_str} \\\\"
        )

    table = r"""\begin{table}[htbp]
\caption{Overall Performance Comparison (%d tasks)}
\label{tab:overall}
\centering
\begin{tabular}{lrrrr}
\toprule
\textbf{Method} & \textbf{Resolve \%%} & \textbf{Cost (\$)} & \textbf{\$/Resolved} & \textbf{Reduction} \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}""" % (
        results[row_order[0]]["num_tasks"],
        "\n".join(rows),
    )
    return table


def generate_pertask_table(results: dict) -> str:
    """Generate Table III: Per-Sub-Task Routing Behavior."""
    if "taskrouter" not in results:
        return "% TaskRouter results not available"

    per_type = results["taskrouter"]["summary"]["per_type"]
    subtask_order = ["EXPL", "COMP", "LOC", "PATCH", "TEST", "VER"]

    rows = []
    for st in subtask_order:
        if st not in per_type:
            continue
        p = per_type[st]
        rows.append(
            f"    {st} & {p['pct_T1']:.1f}\\% & {p['pct_T2']:.1f}\\% & "
            f"{p['pct_T3']:.1f}\\% & {p['pct_T4']:.1f}\\% & {p['cascade_pct']:.1f}\\% \\\\"
        )

    table = r"""\begin{table}[htbp]
\caption{Per-Sub-Task Routing Behavior (TaskRouter)}
\label{tab:pertask}
\centering
\begin{tabular}{lrrrrr}
\toprule
\textbf{Type} & \textbf{\%% T1} & \textbf{\%% T2} & \textbf{\%% T3} & \textbf{\%% T4} & \textbf{Cascade \%%} \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}""" % "\n".join(rows)
    return table


def generate_cascading_table(results: dict) -> str:
    """Generate Table IV: Cascading Statistics."""
    if "taskrouter" not in results:
        return "% TaskRouter results not available"

    s = results["taskrouter"]["summary"]

    table = r"""\begin{table}[htbp]
\caption{Cascading Statistics (TaskRouter)}
\label{tab:cascading}
\centering
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Total sub-tasks & %d \\
Sub-tasks requiring cascade & %d \\
Cascade rate & %.1f\%% \\
Wasted cost from failed cascades & \$%.4f \\
Net cost savings (vs.\ Frontier) & \$%.2f \\
\bottomrule
\end{tabular}
\end{table}""" % (
        s["total_subtasks"],
        s["total_cascades"],
        s["cascade_rate"],
        s["total_wasted_cost"],
        results["single_frontier"]["summary"]["total_cost"] - s["total_cost"],
    )
    return table


def generate_token_distribution_table(results: dict) -> str:
    """Generate Table I: Token Distribution by Sub-Task Type."""
    if "taskrouter" not in results:
        return "% TaskRouter results not available"

    # Aggregate tokens per sub-task type
    type_tokens = {}
    total_all = 0
    for task_result in results["taskrouter"]["task_results"]:
        for decision in task_result["decisions"]:
            st = decision["subtask_type"]
            inp = decision["input_tokens"]
            out = decision["output_tokens"]
            if st not in type_tokens:
                type_tokens[st] = {"input": 0, "output": 0, "total": 0}
            type_tokens[st]["input"] += inp
            type_tokens[st]["output"] += out
            type_tokens[st]["total"] += inp + out
            total_all += inp + out

    subtask_order = ["EXPL", "COMP", "LOC", "PATCH", "TEST", "VER"]
    rows = []
    for st in subtask_order:
        if st not in type_tokens:
            continue
        t = type_tokens[st]
        total_input_all = sum(v["input"] for v in type_tokens.values())
        total_output_all = sum(v["output"] for v in type_tokens.values())
        inp_pct = (t["input"] / total_input_all * 100) if total_input_all > 0 else 0
        out_pct = (t["output"] / total_output_all * 100) if total_output_all > 0 else 0
        tot_pct = (t["total"] / total_all * 100) if total_all > 0 else 0
        rows.append(f"    {st} & {inp_pct:.1f}\\% & {out_pct:.1f}\\% & {tot_pct:.1f}\\% \\\\")

    table = r"""\begin{table}[htbp]
\caption{Token Distribution by Sub-Task Type (TaskRouter)}
\label{tab:token_distribution}
\centering
\begin{tabular}{lrrr}
\toprule
\textbf{Sub-Task Type} & \textbf{Input \%%} & \textbf{Output \%%} & \textbf{Total \%%} \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}""" % "\n".join(rows)
    return table


def generate_ablation_table(results: dict) -> str:
    """Generate Table V: Ablation Study.
    Compares taskrouter vs type_only (no difficulty) vs single_frontier (no routing)."""
    rows = []
    configs = [
        ("taskrouter", "Full TaskRouter"),
        ("type_only_routing", "-- w/o difficulty estimator"),
        ("single_frontier", "-- w/o routing (baseline)"),
    ]
    for name, label in configs:
        if name not in results:
            continue
        s = results[name]["summary"]
        rows.append(f"    {label} & {s['resolve_rate']:.1f}\\% & \\${s['total_cost']:.2f} \\\\")

    table = r"""\begin{table}[htbp]
\caption{Ablation Study Results}
\label{tab:ablation}
\centering
\begin{tabular}{lrr}
\toprule
\textbf{Variant} & \textbf{Resolve \%%} & \textbf{Cost (\$)} \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}""" % "\n".join(rows)
    return table


def generate_matplotlib_figure(results: dict) -> str:
    """Generate Python code for the Pareto frontier figure."""
    data_points = []
    display_names = {
        "single_frontier": "Frontier (T4)",
        "single_small": "Small (T1)",
        "single_medium": "Medium (T2)",
        "single_large": "Large (T3)",
        "random_routing": "Random",
        "type_only_routing": "Type-Only",
        "taskrouter": "TaskRouter",
    }
    for name, data in results.items():
        s = data["summary"]
        data_points.append({
            "name": display_names.get(name, name),
            "cost": s["total_cost"],
            "resolve_rate": s["resolve_rate"],
        })

    code = '''import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 12, "font.family": "serif"})

data = %s

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

for d in data:
    marker = "*" if d["name"] == "TaskRouter" else "o"
    size = 200 if d["name"] == "TaskRouter" else 80
    color = "#e74c3c" if d["name"] == "TaskRouter" else "#3498db"
    ax.scatter(d["cost"], d["resolve_rate"], s=size, marker=marker,
               color=color, zorder=5, edgecolors="black", linewidths=0.5)
    ax.annotate(d["name"], (d["cost"], d["resolve_rate"]),
                textcoords="offset points", xytext=(8, 5), fontsize=9)

ax.set_xlabel("Total Cost ($)", fontsize=13)
ax.set_ylabel("Resolve Rate (%%)", fontsize=13)
ax.set_title("Cost-Quality Tradeoff: TaskRouter vs Baselines", fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig("pareto_frontier.png", dpi=300, bbox_inches="tight")
plt.savefig("pareto_frontier.pdf", bbox_inches="tight")
print("Saved: pareto_frontier.png and pareto_frontier.pdf")
plt.show()
''' % json.dumps(data_points, indent=2)
    return code


def generate_all():
    """Generate all paper outputs."""
    results = load_results()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate LaTeX tables
    tables = {
        "table_token_distribution.tex": generate_token_distribution_table(results),
        "table_overall.tex": generate_overall_table(results),
        "table_pertask.tex": generate_pertask_table(results),
        "table_cascading.tex": generate_cascading_table(results),
        "table_ablation.tex": generate_ablation_table(results),
    }

    for filename, content in tables.items():
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Generated: {filepath}")

    # Generate figure script
    fig_code = generate_matplotlib_figure(results)
    fig_path = os.path.join(OUTPUT_DIR, "plot_pareto.py")
    with open(fig_path, "w") as f:
        f.write(fig_code)
    print(f"Generated: {fig_path}")

    # Print summary comparison
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    reductions = compute_cost_reduction(results)

    fmt = "{:<25} {:>12} {:>12} {:>12}"
    print(fmt.format("Strategy", "Resolve %", "Cost ($)", "Reduction %"))
    print("-" * 70)
    for name in ["single_frontier", "single_small", "random_routing",
                 "type_only_routing", "taskrouter"]:
        if name not in results:
            continue
        s = results[name]["summary"]
        red = reductions[name]
        print(fmt.format(
            name, f"{s['resolve_rate']:.1f}%",
            f"${s['total_cost']:.2f}",
            f"{red:.1f}%" if name != "single_frontier" else "---"
        ))

    print("\nThese values can be copied directly into the paper's placeholder fields.")


if __name__ == "__main__":
    generate_all()
