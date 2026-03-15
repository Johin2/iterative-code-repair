"""Safe sandboxed code execution for HumanEval and MBPP problems."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

COMMON_IMPORTS = """\
from typing import List, Tuple, Optional, Dict, Set, Any, Union
import math
import re as re_module
import sys
import collections
from collections import defaultdict, Counter, OrderedDict
import itertools
import functools
import operator
import string
import heapq
import bisect
import copy
"""


def execute_solution(
    code: str,
    test_code: str,
    entry_point: str,
    timeout: int = 15,
) -> dict[str, str | bool]:
    """Execute generated code against test cases in an isolated subprocess.

    Supports both HumanEval (``check(candidate)`` pattern) and MBPP
    (plain ``assert`` statements) test formats.

    Returns dict with keys: passed, error_message, error_type, stdout.
    """
    full_code = COMMON_IMPORTS + "\n" + code + "\n\n" + test_code

    # HumanEval tests define ``def check(candidate)`` which must be
    # invoked with the entry-point function; MBPP tests run as-is.
    if "def check(" in test_code:
        if f"check({entry_point})" not in test_code:
            full_code += f"\n\ncheck({entry_point})\n"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(full_code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        passed = result.returncode == 0
        stderr = result.stderr.strip()

        return {
            "passed": passed,
            "error_message": stderr if stderr else "",
            "error_type": _classify_error(stderr) if not passed else "none",
            "stdout": result.stdout.strip(),
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "error_message": f"Execution timed out after {timeout}s",
            "error_type": "timeout",
            "stdout": "",
        }
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def _classify_error(stderr: str) -> str:
    """Classify error type from stderr output."""
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
    if "AttributeError" in stderr:
        return "attribute_error"
    if "ValueError" in stderr:
        return "value_error"
    if "RecursionError" in stderr:
        return "recursion"
    if "ZeroDivisionError" in stderr:
        return "zero_division"
    if "ImportError" in stderr or "ModuleNotFoundError" in stderr:
        return "import_error"
    return "runtime_other"
