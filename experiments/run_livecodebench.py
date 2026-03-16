#!/usr/bin/env python3
"""Run self-repair experiments on LiveCodeBench.

LiveCodeBench provides contamination-free coding problems that are harder
than HumanEval/MBPP. This addresses the reviewer concern about benchmark
saturation.

Problems are competitive-programming style (stdin/stdout or function-based).
We adapt them to function-completion format where possible.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

from experiments.config import MODELS, MAX_REPAIR_ROUNDS, RESULTS_DIR
from experiments.self_repair import extract_code
from experiments.api_client import create_client, call_model

LIVECODEBENCH_DIR = RESULTS_DIR / "livecodebench"
DATA_CACHE = LIVECODEBENCH_DIR / "problems.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

LCB_JSONL_FILES = [
    "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test.jsonl",
    "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test2.jsonl",
    "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test3.jsonl",
    "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test4.jsonl",
    "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test5.jsonl",
    "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test6.jsonl",
]


def _download_livecodebench_jsonl() -> list[dict]:
    """Download LiveCodeBench JSONL files from HuggingFace."""
    import urllib.request

    all_items = []
    for url in LCB_JSONL_FILES:
        fname = url.rsplit("/", 1)[-1]
        local_path = LIVECODEBENCH_DIR / fname

        if not local_path.exists():
            print(f"  Downloading {fname}...", end="", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                data = urllib.request.urlopen(req, timeout=120).read()
                with open(local_path, "wb") as f:
                    f.write(data)
                print(" done")
            except Exception as e:
                print(f" FAILED: {e}")
                continue

        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_items.append(json.loads(line))

    print(f"  Total raw problems: {len(all_items)}")
    return all_items


def load_livecodebench(max_problems: int | None = None) -> list[dict]:
    """Load LiveCodeBench problems via the datasets library, with caching.

    Each problem is normalised to a dict with keys:
        task_id, prompt, entry_point, test_cases, difficulty
    """
    if DATA_CACHE.exists():
        with open(DATA_CACHE, "r", encoding="utf-8") as f:
            problems = json.load(f)
        if max_problems:
            problems = problems[:max_problems]
        print(f"Loaded {len(problems)} LiveCodeBench problems from cache")
        return problems

    LIVECODEBENCH_DIR.mkdir(parents=True, exist_ok=True)

    # Download JSONL files directly from HuggingFace (avoids datasets library
    # compatibility issues with custom loading scripts)
    raw_items = _download_livecodebench_jsonl()

    problems = []
    for raw in raw_items:

        question_id = raw.get("question_id", raw.get("id", f"lcb_{len(problems)}"))
        title = raw.get("question_title", "")
        content = raw.get("question_content", "")
        starter = raw.get("starter_code", "")
        difficulty = raw.get("difficulty", "unknown")

        # Parse test cases — LiveCodeBench uses public_test_cases / private_test_cases
        # Each is a JSON string containing a list of {input, output, testtype} dicts
        test_cases_raw = raw.get("public_test_cases", "[]")
        if isinstance(test_cases_raw, str):
            try:
                test_cases_list = json.loads(test_cases_raw)
            except json.JSONDecodeError:
                test_cases_list = []
        else:
            test_cases_list = test_cases_raw if isinstance(test_cases_raw, list) else []

        if not test_cases_list:
            continue

        # Convert to inputs/outputs lists
        inputs = [tc.get("input", "") for tc in test_cases_list if isinstance(tc, dict)]
        outputs = [tc.get("output", "") for tc in test_cases_list if isinstance(tc, dict)]

        if not inputs or not outputs:
            continue

        # Determine if this is function-based or stdin/stdout
        import re
        is_function = bool(starter and "def " in starter)

        # Build prompt
        if is_function:
            prompt = f"{content}\n\n{starter}"
            match = re.search(r"def\s+(\w+)\s*\(", starter)
            entry_point = match.group(1) if match else "solution"
        else:
            # Competitive programming: stdin/stdout
            prompt = (
                f"{content}\n\n"
                f"Write a complete Python program that reads from stdin and "
                f"prints to stdout. Do NOT define a main function — just write "
                f"the solution code directly.\n"
            )
            if starter:
                prompt += f"\nStarter code:\n{starter}\n"
            entry_point = "__main__"  # signals stdin/stdout execution

        problems.append({
            "task_id": f"LCB/{question_id}",
            "title": title,
            "prompt": prompt,
            "entry_point": entry_point,
            "is_function": is_function,
            "test_cases": {"inputs": inputs, "outputs": outputs},
            "difficulty": difficulty,
        })

    # Cache to disk
    with open(DATA_CACHE, "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)

    print(f"Loaded {len(problems)} LiveCodeBench problems via datasets library")
    if max_problems:
        problems = problems[:max_problems]
    return problems


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execute_livecodebench(
    code: str,
    test_cases: dict,
    is_function: bool,
    entry_point: str,
    timeout: int = 30,
) -> dict[str, str | bool]:
    """Execute a LiveCodeBench solution against its test cases.

    For stdin/stdout problems (the vast majority): write code to a temp file,
    run it with each input via subprocess stdin, compare stdout to expected.

    Returns dict with keys: passed, error_message, error_type.
    """
    inputs = test_cases.get("inputs", [])
    outputs = test_cases.get("outputs", [])

    if not inputs or not outputs:
        return {
            "passed": False,
            "error_message": "No test cases available",
            "error_type": "no_tests",
        }

    # Write the solution to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8",
    ) as f:
        f.write(code)
        temp_path = f.name

    try:
        for i, (inp, expected) in enumerate(zip(inputs, outputs)):
            inp_str = inp if isinstance(inp, str) else str(inp)
            exp_str = expected if isinstance(expected, str) else str(expected)

            try:
                result = subprocess.run(
                    [sys.executable, temp_path],
                    input=inp_str,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return {
                    "passed": False,
                    "error_message": f"Execution timed out on test case {i}",
                    "error_type": "timeout",
                }

            if result.returncode != 0:
                return {
                    "passed": False,
                    "error_message": result.stderr.strip()[:500],
                    "error_type": _classify_error(result.stderr),
                }

            got = result.stdout.strip()
            want = exp_str.strip()
            if got != want:
                return {
                    "passed": False,
                    "error_message": f"Test {i}: expected {want[:200]!r}, got {got[:200]!r}",
                    "error_type": "assertion",
                }

        return {
            "passed": True,
            "error_message": "",
            "error_type": "none",
        }
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass




def _classify_error(stderr: str) -> str:
    """Classify error type from stderr."""
    if not stderr:
        return "unknown"
    if "SyntaxError" in stderr or "IndentationError" in stderr:
        return "syntax"
    if "AssertionError" in stderr:
        return "assertion"
    if "TypeError" in stderr:
        return "type_error"
    if "NameError" in stderr:
        return "name_error"
    if "IndexError" in stderr or "KeyError" in stderr:
        return "index_key_error"
    if "ValueError" in stderr:
        return "value_error"
    if "RecursionError" in stderr:
        return "recursion"
    return "runtime_other"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def build_initial_prompt_lcb(problem: dict) -> list[dict[str, str]]:
    """Build initial chat messages for a LiveCodeBench problem."""
    return [
        {
            "role": "system",
            "content": (
                "You are an expert Python programmer. Solve the given problem. "
                "Return ONLY the Python code, no explanations, no markdown formatting."
            ),
        },
        {
            "role": "user",
            "content": f"Solve the following problem:\n\n{problem['prompt']}",
        },
    ]


def build_repair_prompt_lcb(error_message: str) -> str:
    """Build repair prompt for LiveCodeBench."""
    if len(error_message) > 1500:
        error_message = error_message[:1500] + "\n... (truncated)"
    return (
        f"Your code produced an error when tested:\n\n"
        f"{error_message}\n\n"
        f"Please fix the code. Return ONLY the corrected Python code, "
        f"no explanations, no markdown formatting."
    )


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------

def run_single_problem(
    client,
    model_id: str,
    problem: dict,
    max_rounds: int,
) -> dict:
    """Run self-repair loop for one LiveCodeBench problem."""
    entry_point = problem["entry_point"]
    is_function = problem["is_function"]
    test_cases = problem["test_cases"]
    prompt_text = problem["prompt"]

    messages = build_initial_prompt_lcb(problem)

    # Disable thinking for Qwen3
    if "qwen" in model_id.lower():
        messages[-1]["content"] += "\n/no_think"

    rounds = []

    for round_num in range(max_rounds):
        raw_response, usage = call_model(client, model_id, messages)

        if raw_response is None:
            rounds.append({
                "round": round_num,
                "passed": False,
                "error_type": "api_error",
                "error_message": "API call failed after retries",
                "usage": {},
            })
            break

        code = extract_code(raw_response, entry_point, prompt_text)
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
    model_id: str,
    problems: list[dict],
    max_rounds: int,
) -> list[dict]:
    """Run LiveCodeBench experiment for one model."""
    client = create_client()
    results_file = LIVECODEBENCH_DIR / f"{model_name}_livecodebench.json"

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
            print(f" FAIL (all {max_rounds} rounds)")

        # Save incrementally
        seen = {r["task_id"] for r in results}
        save_data = results + [existing[tid] for tid in existing if tid not in seen]
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="LiveCodeBench Self-Repair Experiment")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Models to run. Choices: {list(MODELS.keys())}",
    )
    parser.add_argument("--max-rounds", type=int, default=MAX_REPAIR_ROUNDS)
    parser.add_argument("--num-problems", type=int, default=None,
                        help="Limit number of problems (useful for testing)")
    args = parser.parse_args()

    LIVECODEBENCH_DIR.mkdir(parents=True, exist_ok=True)
    model_names = args.models or list(MODELS.keys())

    print("Loading LiveCodeBench dataset...")
    problems = load_livecodebench(max_problems=args.num_problems)
    print(f"Total problems: {len(problems)}")

    # Print difficulty distribution
    diff_counts: dict[str, int] = {}
    for p in problems:
        d = p.get("difficulty", "unknown")
        diff_counts[d] = diff_counts.get(d, 0) + 1
    for d, c in sorted(diff_counts.items()):
        print(f"  {d}: {c}")
    print()

    summary = {}

    for model_name in model_names:
        if model_name not in MODELS:
            print(f"Unknown model: {model_name}. Skipping.")
            continue

        model_id = MODELS[model_name]
        print(f"{'='*60}")
        print(f" {model_name} | LiveCodeBench ({len(problems)} problems)")
        print(f"{'='*60}")

        start = time.time()
        results = run_model_experiment(model_name, model_id, problems, args.max_rounds)
        elapsed = time.time() - start

        passed = sum(1 for r in results if r["final_passed"])
        total = len(results)
        r0_passed = sum(
            1 for r in results
            if r["rounds"] and r["rounds"][0]["passed"]
        )

        # Breakdown by difficulty
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

    # Print summary
    print(f"\n{'='*60}")
    print(" LIVECODEBENCH SUMMARY")
    print(f"{'='*60}")
    for name, s in summary.items():
        print(f"  {name:25s}: R0={s['r0_rate']:5.1f}% -> Final={s['final_rate']:5.1f}% "
              f"(+{s['gain_pp']:.1f}pp)")

    with open(LIVECODEBENCH_DIR / "livecodebench_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
