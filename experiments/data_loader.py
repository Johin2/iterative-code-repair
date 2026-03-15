"""Load HumanEval and MBPP benchmark datasets."""
import json
import gzip
import re
import urllib.request
from config import DATA_DIR

HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"


def load_humaneval():
    """Load HumanEval dataset, downloading if necessary."""
    jsonl_path = DATA_DIR / "HumanEval.jsonl"

    if not jsonl_path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Try datasets library first
        try:
            from datasets import load_dataset
            ds = load_dataset("openai_humaneval", split="test")
            problems = [dict(item) for item in ds]
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for p in problems:
                    f.write(json.dumps(p) + "\n")
            print(f"Loaded {len(problems)} problems via datasets library")
            return problems
        except Exception:
            pass

        # Fallback: direct download
        print("Downloading HumanEval dataset...")
        gz_path = DATA_DIR / "HumanEval.jsonl.gz"
        urllib.request.urlretrieve(HUMANEVAL_URL, str(gz_path))
        with gzip.open(str(gz_path), "rt", encoding="utf-8") as gz_file:
            with open(jsonl_path, "w", encoding="utf-8") as out_file:
                out_file.write(gz_file.read())
        gz_path.unlink()
        print("Download complete.")

    problems = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    return problems


def _extract_function_signature(code):
    """Extract function name and arguments from a def statement in reference code.

    Returns (func_name, args_string) or (None, None) if no def found.
    """
    match = re.search(r"^def\s+(\w+)\s*\(([^)]*)\)\s*:", code, re.MULTILINE)
    if match:
        return match.group(1), match.group(2).strip()
    return None, None


def load_mbpp():
    """Load MBPP sanitized dataset, downloading/caching as JSONL.

    Each problem is converted to a HumanEval-compatible dict with keys:
        task_id, prompt, entry_point, test
    """
    jsonl_path = DATA_DIR / "MBPP_sanitized.jsonl"

    if not jsonl_path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

        problems = []
        for item in ds:
            raw = dict(item)
            task_id = raw["task_id"]
            description = raw["prompt"]       # natural language description
            code = raw["code"]                # reference solution
            test_list = raw["test_list"]      # list of assert strings
            test_imports = raw.get("test_imports", [])  # optional imports for tests

            func_name, args_str = _extract_function_signature(code)
            if func_name is None:
                # Fallback: skip problems without a clear function definition
                continue

            # Build a HumanEval-style prompt: signature + docstring
            prompt = f'def {func_name}({args_str}):\n    """{description}"""\n'

            # Test code: prepend any required imports, then the assert statements
            test_parts = []
            if test_imports:
                test_parts.extend(test_imports)
            test_parts.extend(test_list)
            test_code = "\n".join(test_parts)

            problems.append({
                "task_id": f"Mbpp/{task_id}",
                "prompt": prompt,
                "entry_point": func_name,
                "test": test_code,
            })

        # Cache to disk
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for p in problems:
                f.write(json.dumps(p) + "\n")

        print(f"Loaded {len(problems)} MBPP sanitized problems via datasets library")
        return problems

    # Read from cache
    problems = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    return problems


if __name__ == "__main__":
    print("=== Testing MBPP loader ===\n")
    mbpp_problems = load_mbpp()
    print(f"Total MBPP problems loaded: {len(mbpp_problems)}\n")
    for p in mbpp_problems[:3]:
        print(f"--- {p['task_id']} (entry_point: {p['entry_point']}) ---")
        print(f"Prompt:\n{p['prompt']}")
        print(f"Test:\n{p['test']}")
        print()
