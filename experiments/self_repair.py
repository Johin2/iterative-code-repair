"""Prompt building and code extraction for self-repair experiments."""
import re


def build_initial_prompt(problem):
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


def build_repair_prompt(error_message):
    """Build repair prompt given an error message."""
    if len(error_message) > 1500:
        error_message = error_message[:1500] + "\n... (truncated)"

    return (
        f"Your code produced an error when tested:\n\n"
        f"{error_message}\n\n"
        f"Please fix the code. Return ONLY the corrected Python function, "
        f"no explanations, no markdown formatting."
    )


def extract_code(response, entry_point, prompt):
    """Extract Python function code from model response."""
    if not response:
        return prompt + "    pass\n"

    code = response.strip()

    # Remove DeepSeek R1 thinking traces
    code = re.sub(r"<think>.*?</think>", "", code, flags=re.DOTALL).strip()

    # Extract from markdown code blocks
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", code, flags=re.DOTALL)
    if blocks:
        code = max(blocks, key=len).strip()

    # If response contains the full function definition, use it
    if f"def {entry_point}" in code:
        return code

    # Otherwise, the model returned just the body — prepend the prompt
    return prompt + code
