from __future__ import annotations

from src.eval.benchmark_registry import CoTMode
from src.eval.prompt_builders import (
    CODE_COMPLETION_PLACEHOLDER,
    FINAL_ANSWER_PLACEHOLDER,
    LOGPROBS_PLACEHOLDER,
    build_human_eval_expected_context,
    build_instruction_following_prompt,
    build_livecodebench_expected_context,
    build_maths_expected_context,
    build_mbpp_expected_context,
    build_multiple_choice_expected_context,
    prompt_for_cot,
    prompt_for_marker,
)


def test_multiple_choice_prompt_builder_matches_rwkv_rs_shape() -> None:
    context = build_multiple_choice_expected_context(
        subject="computer science",
        question="2 + 2 = ?",
        choices=["1", "4", "9"],
        cot_mode=CoTMode.COT,
    )
    assert context == (
        "User: You are a very talented expert in computer science.\n"
        "Answer this question and finish with a single option letter.\n"
        "Question: 2 + 2 = ?\n"
        "Choices:\n"
        "A. 1\n"
        "B. 4\n"
        "C. 9\n\n"
        "Assistant: <think><|completions_of_cot|></think>\n"
        "Therefore, the answer is<|logprobs_of_choices|>"
    )
    assert prompt_for_cot(context).endswith("Assistant: <think>")
    assert prompt_for_marker(context, LOGPROBS_PLACEHOLDER, completions_of_cot="reasoning").endswith(
        "Assistant: <think>reasoning</think>\nTherefore, the answer is"
    )


def test_maths_prompt_builder_matches_rwkv_rs_shape() -> None:
    context = build_maths_expected_context(
        subject="maths",
        question="What is 6 * 7?",
        cot_mode=CoTMode.COT,
    )
    assert context == (
        "User: You are a very talented expert in maths.\n"
        "Solve the problem and output the final answer in \\boxed{}.\n"
        "Problem: What is 6 * 7?\n\n"
        "Assistant: <think><|completions_of_cot|></think>\n"
        "Therefore, the answer is \\(\\boxed{<|final_answer|>}\\)."
    )
    assert prompt_for_cot(context).endswith("Assistant: <think>")
    assert prompt_for_marker(context, FINAL_ANSWER_PLACEHOLDER, completions_of_cot="work").endswith(
        "Assistant: <think>work</think>\nTherefore, the answer is \\(\\boxed{"
    )


def test_instruction_following_prompt_builder_matches_rwkv_rs_shape() -> None:
    assert build_instruction_following_prompt("Hello") == (
        "User: Hello\n\nAssistant: <think></think>"
    )
    assert build_instruction_following_prompt("Hello", enable_think=True) == "User: Hello\n\nAssistant: <think"


def test_human_eval_prompt_builder_matches_rwkv_rs_shape() -> None:
    context = build_human_eval_expected_context(
        "def add(a, b):\n    return a + b",
        assistant_code_prefix="def add(a, b):\n    return a + b",
        cot_mode=CoTMode.NO_COT,
    )
    assert prompt_for_marker(context, CODE_COMPLETION_PLACEHOLDER) == (
        "User: You are a top-level code master. Complete the following code without any additional text or explanation:\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "Assistant: <think>\n"
        "</think>\n"
        "```python"
        "def add(a, b):\n"
        "    return a + b"
    )


def test_mbpp_prompt_builder_matches_rwkv_rs_shape() -> None:
    context = build_mbpp_expected_context(
        "Write a function that doubles a number.",
        signature="def double(x):",
        cot_mode=CoTMode.COT,
    )
    assert context == (
        "User: You are a top-level code master. Complete the following code without any additional text or explanation:\n"
        "Write a function that doubles a number.\n"
        "Function signature: def double(x):\n"
        "Write the full function definition.\n\n"
        "Assistant: <think><|completions_of_cot|></think>\n"
        "```python"
        "<|completions|>"
    )
    assert prompt_for_cot(context).endswith("Assistant: <think>")
    assert prompt_for_marker(context, CODE_COMPLETION_PLACEHOLDER, completions_of_cot="reasoning").endswith(
        "Assistant: <think>reasoning</think>\n```python"
    )


def test_livecodebench_prompt_builder_matches_rwkv_rs_shape() -> None:
    context = build_livecodebench_expected_context(
        "Read n and print n squared.",
        starter_code="def solve():\n    pass",
        cot_mode=CoTMode.COT,
    )
    assert prompt_for_cot(context) == (
        "User: You are an expert Python programmer. You will be given a question "
        "(problem specification) and will generate a correct Python program "
        "that matches the specification and passes all tests.\n"
        "### Question:\n"
        "Read n and print n squared.\n"
        "\n"
        "### Format: You will use the following starter code to write the solution to the problem "
        "and enclose your code within delimiters.\n"
        "```python\n"
        "def solve():\n"
        "    pass\n"
        "```\n\n"
        "### Answer: (use the provided format with backticks)\n\n"
        "Assistant: <think"
    )
    assert prompt_for_marker(context, CODE_COMPLETION_PLACEHOLDER, completions_of_cot=">reasoning") == (
        "User: You are an expert Python programmer. You will be given a question "
        "(problem specification) and will generate a correct Python program "
        "that matches the specification and passes all tests.\n"
        "### Question:\n"
        "Read n and print n squared.\n"
        "\n"
        "### Format: You will use the following starter code to write the solution to the problem "
        "and enclose your code within delimiters.\n"
        "```python\n"
        "def solve():\n"
        "    pass\n"
        "```\n\n"
        "### Answer: (use the provided format with backticks)\n\n"
        "Assistant: <think>reasoning\n"
        "</think>\n"
        "```python\n"
    )
