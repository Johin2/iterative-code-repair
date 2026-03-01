"""
Main experiment runner for TaskRouter evaluation.
Runs all strategies across all tasks and saves results.
"""

import json
import os
import time
from datetime import datetime

from config import MODEL_TIERS, NUM_TASKS, RESULTS_DIR
from tasks import get_tasks
from taskrouter import run_task_with_routing, evaluate_patch


STRATEGIES = [
    {"name": "single_frontier", "strategy": "single", "fixed_tier": "T4"},
    {"name": "single_small", "strategy": "single", "fixed_tier": "T1"},
    {"name": "single_medium", "strategy": "single", "fixed_tier": "T2"},
    {"name": "single_large", "strategy": "single", "fixed_tier": "T3"},
    {"name": "random_routing", "strategy": "random", "fixed_tier": None},
    {"name": "type_only_routing", "strategy": "type_only", "fixed_tier": None},
    {"name": "taskrouter", "strategy": "taskrouter", "fixed_tier": None},
]


def run_single_strategy(strategy_config: dict, tasks: list) -> dict:
    """Run one strategy across all tasks."""
    strategy_name = strategy_config["name"]
    print(f"\n{'='*60}")
    print(f"Running strategy: {strategy_name}")
    print(f"{'='*60}")

    results = {
        "strategy": strategy_name,
        "num_tasks": len(tasks),
        "start_time": datetime.now().isoformat(),
        "task_results": [],
        "summary": {},
    }

    total_cost = 0.0
    total_resolved = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_wasted_cost = 0.0
    total_cascades = 0
    total_subtasks = 0

    per_type_stats = {
        t: {"count": 0, "to_T1": 0, "to_T2": 0, "to_T3": 0, "to_T4": 0, "cascaded": 0}
        for t in ["EXPL", "COMP", "LOC", "PATCH", "TEST", "VER"]
    }

    for i, task in enumerate(tasks):
        print(f"\n  [{i+1}/{len(tasks)}] {task['id']}: {task['title']}")

        task_result = run_task_with_routing(
            task,
            strategy=strategy_config["strategy"],
            fixed_tier=strategy_config.get("fixed_tier"),
        )

        # Evaluate the patch
        task_result.test_passed = evaluate_patch(task, task_result.patch_code)

        # Collect stats
        total_cost += task_result.total_cost
        total_input_tokens += task_result.total_input_tokens
        total_output_tokens += task_result.total_output_tokens
        total_wasted_cost += task_result.wasted_cost
        if task_result.test_passed:
            total_resolved += 1

        # Per-decision stats
        for decision in task_result.decisions:
            st = decision.subtask_type
            per_type_stats[st]["count"] += 1
            per_type_stats[st][f"to_{decision.final_tier}"] += 1
            if decision.cascaded:
                per_type_stats[st]["cascaded"] += 1
                total_cascades += 1
            total_subtasks += 1

        # Serialize task result
        task_data = {
            "task_id": task_result.task_id,
            "test_passed": task_result.test_passed,
            "total_cost": round(task_result.total_cost, 6),
            "total_input_tokens": task_result.total_input_tokens,
            "total_output_tokens": task_result.total_output_tokens,
            "wasted_cost": round(task_result.wasted_cost, 6),
            "decisions": [
                {
                    "subtask_type": d.subtask_type,
                    "difficulty": d.difficulty,
                    "initial_tier": d.initial_tier,
                    "final_tier": d.final_tier,
                    "cascaded": d.cascaded,
                    "cascade_depth": d.cascade_depth,
                    "tier_model": d.response.model_id,
                    "input_tokens": d.response.input_tokens,
                    "output_tokens": d.response.output_tokens,
                    "cost": round(d.response.cost, 6),
                    "latency_ms": round(d.response.latency_ms, 1),
                    "success": d.response.success,
                }
                for d in task_result.decisions
            ],
        }
        results["task_results"].append(task_data)

        status = "PASS" if task_result.test_passed else "FAIL"
        print(f"    {status} | Cost: ${task_result.total_cost:.4f} | "
              f"Tokens: {task_result.total_input_tokens + task_result.total_output_tokens}")

    # Compute summary
    resolve_rate = (total_resolved / len(tasks)) * 100 if tasks else 0
    cost_per_resolved = total_cost / total_resolved if total_resolved > 0 else float('inf')
    cascade_rate = (total_cascades / total_subtasks) * 100 if total_subtasks > 0 else 0

    # Per-type percentages
    per_type_summary = {}
    for st, stats in per_type_stats.items():
        count = stats["count"]
        if count == 0:
            continue
        per_type_summary[st] = {
            "count": count,
            "pct_T1": round(stats["to_T1"] / count * 100, 1),
            "pct_T2": round(stats["to_T2"] / count * 100, 1),
            "pct_T3": round(stats["to_T3"] / count * 100, 1),
            "pct_T4": round(stats["to_T4"] / count * 100, 1),
            "cascade_pct": round(stats["cascaded"] / count * 100, 1),
        }

    results["summary"] = {
        "resolve_rate": round(resolve_rate, 1),
        "total_resolved": total_resolved,
        "total_cost": round(total_cost, 4),
        "cost_per_resolved": round(cost_per_resolved, 4),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_wasted_cost": round(total_wasted_cost, 4),
        "cascade_rate": round(cascade_rate, 1),
        "total_subtasks": total_subtasks,
        "total_cascades": total_cascades,
        "per_type": per_type_summary,
    }
    results["end_time"] = datetime.now().isoformat()

    print(f"\n  Summary: {resolve_rate:.1f}% resolved | ${total_cost:.4f} total cost | "
          f"{cascade_rate:.1f}% cascade rate")

    return results


def run_all_experiments(num_tasks: int = None):
    """Run all strategies and save results."""
    if num_tasks is None:
        num_tasks = NUM_TASKS

    tasks = get_tasks(num_tasks)
    print(f"Running experiments on {len(tasks)} tasks")
    print(f"Strategies: {[s['name'] for s in STRATEGIES]}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    for strategy_config in STRATEGIES:
        try:
            result = run_single_strategy(strategy_config, tasks)
            all_results[strategy_config["name"]] = result

            # Save individual result
            filepath = os.path.join(RESULTS_DIR, f"{strategy_config['name']}.json")
            with open(filepath, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {filepath}")
        except Exception as e:
            print(f"  ERROR in {strategy_config['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Save combined results
    combined_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined_path}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run TaskRouter experiments")
    parser.add_argument("--num-tasks", type=int, default=NUM_TASKS,
                        help="Number of tasks to run")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Run only this strategy (e.g., 'taskrouter')")
    args = parser.parse_args()

    if args.strategy:
        tasks = get_tasks(args.num_tasks)
        matching = [s for s in STRATEGIES if s["name"] == args.strategy]
        if matching:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            result = run_single_strategy(matching[0], tasks)
            filepath = os.path.join(RESULTS_DIR, f"{args.strategy}.json")
            with open(filepath, "w") as f:
                json.dump(result, f, indent=2)
        else:
            print(f"Unknown strategy: {args.strategy}")
            print(f"Available: {[s['name'] for s in STRATEGIES]}")
    else:
        run_all_experiments(args.num_tasks)
