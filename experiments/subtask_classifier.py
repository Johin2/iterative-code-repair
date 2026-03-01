"""
Sub-task classifier for TaskRouter.
Classifies each agent action into one of 6 canonical types.
"""

from config import SUBTASK_TYPES


def classify_subtask(action_description: str, tool_used: str = "",
                     context: str = "") -> str:
    """
    Classify an agent action into a sub-task type.

    Args:
        action_description: What the agent is doing (from its reasoning)
        tool_used: The tool being invoked (e.g., "bash", "read_file", "edit_file")
        context: Additional context about the action

    Returns:
        One of: EXPL, COMP, LOC, PATCH, TEST, VER
    """
    desc_lower = action_description.lower()
    tool_lower = tool_used.lower()

    # Rule-based classification from tool + description signals
    # Priority order matters: more specific rules first

    # VERIFICATION: running tests, checking results
    if any(kw in desc_lower for kw in ["run test", "pytest", "verify", "check output",
                                        "check result", "validate", "assert"]):
        return "VER"
    if tool_lower == "bash" and any(kw in desc_lower for kw in ["test", "pytest", "unittest"]):
        return "VER"

    # TEST GENERATION: writing tests
    if any(kw in desc_lower for kw in ["write test", "create test", "add test",
                                        "test case", "test function"]):
        return "TEST"
    if tool_lower in ("edit_file", "write_file") and "test" in desc_lower:
        return "TEST"

    # PATCH GENERATION: writing/editing code (non-test)
    if any(kw in desc_lower for kw in ["fix", "patch", "modify", "change", "implement",
                                        "write code", "add function", "refactor",
                                        "update code", "edit"]):
        return "PATCH"
    if tool_lower in ("edit_file", "write_file") and "test" not in desc_lower:
        return "PATCH"

    # LOCALIZATION: finding specific code locations
    if any(kw in desc_lower for kw in ["localize", "locate", "find the bug",
                                        "identify", "which file", "where is",
                                        "trace", "narrow down"]):
        return "LOC"
    if tool_lower == "search_code":
        return "LOC"

    # COMPREHENSION: reading and understanding code
    if any(kw in desc_lower for kw in ["read", "understand", "analyze", "examine",
                                        "look at", "review code", "study"]):
        return "COMP"
    if tool_lower in ("read_file", "open_file"):
        return "COMP"

    # EXPLORATION: browsing repo structure
    if any(kw in desc_lower for kw in ["list", "explore", "browse", "directory",
                                        "find files", "search for", "grep", "ls"]):
        return "EXPL"
    if tool_lower == "bash" and any(kw in desc_lower for kw in ["ls", "find", "tree"]):
        return "EXPL"

    # Default: if we can't classify, assume it's the most expensive type
    return "PATCH"


# The 6 sub-task types used in our agentic workflow
SUBTASK_PROMPTS = {
    "EXPL": {
        "name": "Exploration",
        "description": "Explore the codebase structure and find relevant files",
        "prompt_template": (
            "You are exploring a codebase to understand its structure.\n\n"
            "Task: {task_description}\n\n"
            "Code:\n```python\n{code}\n```\n\n"
            "List the key components, functions, and their purposes. "
            "Identify which parts are relevant to the task."
        ),
    },
    "COMP": {
        "name": "Comprehension",
        "description": "Understand the existing code behavior and logic",
        "prompt_template": (
            "You are analyzing code to understand its behavior.\n\n"
            "Task: {task_description}\n\n"
            "Code:\n```python\n{code}\n```\n\n"
            "Explain what this code does, how it works, and identify any issues "
            "or bugs. Be specific about the problem."
        ),
    },
    "LOC": {
        "name": "Localization",
        "description": "Identify the exact location of the bug or area to change",
        "prompt_template": (
            "You are localizing a bug or identifying the exact code that needs to change.\n\n"
            "Task: {task_description}\n\n"
            "Code:\n```python\n{code}\n```\n\n"
            "Identify the EXACT line(s) that contain the bug or need to be changed. "
            "Explain why these specific lines are the problem. "
            "Output in format: LINE: <line_content> | REASON: <why>"
        ),
    },
    "PATCH": {
        "name": "Patch Generation",
        "description": "Generate the actual code fix or implementation",
        "prompt_template": (
            "You are fixing a bug or implementing a feature.\n\n"
            "Task: {task_description}\n\n"
            "Original code:\n```python\n{code}\n```\n\n"
            "Tests that must pass:\n```python\n{test_code}\n```\n\n"
            "Write the COMPLETE corrected/implemented code. "
            "Output ONLY the Python code inside ```python``` fences, no explanation."
        ),
    },
    "TEST": {
        "name": "Test Generation",
        "description": "Generate test cases for the code",
        "prompt_template": (
            "You are writing additional test cases for code.\n\n"
            "Task: {task_description}\n\n"
            "Code:\n```python\n{code}\n```\n\n"
            "Existing tests:\n```python\n{test_code}\n```\n\n"
            "Write 3 ADDITIONAL test functions that cover edge cases not tested above. "
            "Output ONLY the Python test code inside ```python``` fences."
        ),
    },
    "VER": {
        "name": "Verification",
        "description": "Verify whether the fix is correct",
        "prompt_template": (
            "You are verifying whether a code fix is correct.\n\n"
            "Task: {task_description}\n\n"
            "Original buggy code:\n```python\n{original_code}\n```\n\n"
            "Proposed fix:\n```python\n{fixed_code}\n```\n\n"
            "Tests:\n```python\n{test_code}\n```\n\n"
            "Analyze: Will the proposed fix make ALL tests pass? "
            "Answer with VERDICT: PASS or VERDICT: FAIL, followed by explanation."
        ),
    },
}
