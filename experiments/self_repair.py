"""Prompt building and code extraction for self-repair experiments."""
from __future__ import annotations

import re


def build_initial_prompt(problem: dict) -> list[dict[str, str]]:
    """Build initial chat messages for code generation."""
    return [
        {
            "role": "system",
            "content": (
                "You are an expert Python programmer. Complete the given function. "
                "Return ONLY the Python code, no explanations, no markdown formatting."
            ),
        },
        {
            "role": "user",
            "content": f"Complete the following Python function:\n\n{problem['prompt']}",
        },
    ]


def build_repair_prompt(error_message: str) -> str:
    """Build minimal repair prompt given an error message."""
    if len(error_message) > 1500:
        error_message = error_message[:1500] + "\n... (truncated)"

    return (
        f"Your code produced an error when tested:\n\n"
        f"{error_message}\n\n"
        f"Please fix the code. Return ONLY the corrected Python function, "
        f"no explanations, no markdown formatting."
    )


def extract_code(response: str, entry_point: str, prompt: str) -> str:
    """Extract Python function code from model response."""
    if not response:
        return prompt + "    pass\n"

    code = response.strip()

    # Remove thinking traces (e.g. Qwen3 <think>...</think> tags)
    code = re.sub(r"<think>.*?</think>", "", code, flags=re.DOTALL).strip()

    # Extract from markdown code blocks (closed)
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", code, flags=re.DOTALL)
    if blocks:
        code = max(blocks, key=len)
    else:
        # Handle unclosed code blocks (e.g. truncated responses from reasoning
        # models like Gemini 2.5 Pro): strip the opening fence line
        code = re.sub(r"^```(?:python)?\s*\n", "", code)

    # Strip trailing whitespace but preserve leading indentation
    code = code.rstrip()

    # If response contains the full function definition, use it
    if f"def {entry_point}" in code:
        # Strip leading blank lines only
        return code.lstrip("\n")

    # Otherwise, the model returned just the body — prepend the prompt.
    # Ensure the body has consistent 4-space indentation.
    lines = code.split("\n")
    # Detect the indentation of the first non-empty line
    first_indent = 0
    for line in lines:
        if line.strip():
            first_indent = len(line) - len(line.lstrip())
            break

    if first_indent == 0:
        # Body has no indentation — add 4-space indent to each line
        indented = "\n".join(
            ("    " + line if line.strip() else line) for line in lines
        )
        return prompt + indented
    else:
        # Body already indented — use as-is
        return prompt + code
