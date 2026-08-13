"""Shared rwkv-rs-style prompt builders for benchmark fields."""

from __future__ import annotations

from collections.abc import Sequence

from src.eval.benchmark_registry import CoTMode
from src.eval.naive_prompt_protocol import NAIVE_NOCOT_ASSISTANT_PREFIX

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
COT_PLACEHOLDER = "<|completions_of_cot|>"
LOGPROBS_PLACEHOLDER = "<|logprobs_of_choices|>"
FINAL_ANSWER_PLACEHOLDER = "<|final_answer|>"
CODE_COMPLETION_PLACEHOLDER = "<|completions|>"


def normalize_subject(subject: str | None, fallback: str) -> str:
    value = (subject or fallback).strip() or fallback
    return value.replace("_", " ")


def concat_choices(choices: Sequence[str]) -> str:
    return "\n".join(f"{ALPHABET[idx]}. {choice}" for idx, choice in enumerate(choices))


def apply_user_assistant_template(user_part: str, assistant_part: str) -> str:
    return f"User: {user_part}\n\nAssistant: {assistant_part}"


def _split_before_marker(text: str, marker: str) -> str:
    head, sep, _tail = text.partition(marker)
    if not sep:
        raise ValueError(f"expected marker missing from prompt context: {marker}")
    return head


def render_context(expected_context: str, replacements: Sequence[tuple[str, str]]) -> str:
    context = expected_context
    for placeholder, value in replacements:
        context = context.replace(placeholder, value)
    return context


def prompt_for_cot(expected_context: str) -> str:
    return _split_before_marker(expected_context, COT_PLACEHOLDER)


def prompt_for_marker(
    expected_context: str,
    marker: str,
    *,
    completions_of_cot: str | None = None,
) -> str:
    context = expected_context
    if completions_of_cot is not None:
        context = context.replace(COT_PLACEHOLDER, completions_of_cot)
    return _split_before_marker(context, marker)


def build_multiple_choice_expected_context(
    *,
    subject: str,
    question: str,
    choices: Sequence[str],
    cot_mode: CoTMode,
) -> str:
    user_part = (
        f"You are a very talented expert in {subject}.\n"
        "Answer this question and finish with a single option letter.\n"
        f"Question: {question}\n"
        f"Choices:\n{concat_choices(choices)}"
    )
    assistant_part = {
        CoTMode.NO_COT: f"Therefore, the answer is{LOGPROBS_PLACEHOLDER}",
        CoTMode.COT: (
            f"<think>{COT_PLACEHOLDER}</think>\n"
            f"Therefore, the answer is{LOGPROBS_PLACEHOLDER}"
        ),
    }[cot_mode]
    return apply_user_assistant_template(user_part, assistant_part)


def build_maths_expected_context(*, subject: str, question: str, cot_mode: CoTMode) -> str:
    if cot_mode is not CoTMode.COT:
        raise ValueError(f"maths only supports CoT mode, got {cot_mode!r}")
    user_part = (
        f"You are a very talented expert in {subject}.\n"
        "Solve the problem and output the final answer in \\boxed{}.\n"
        f"Problem: {question}"
    )
    assistant_part = (
        f"<think>{COT_PLACEHOLDER}</think>\n"
        f"Therefore, the answer is \\(\\boxed{{{FINAL_ANSWER_PLACEHOLDER}}}\\)."
    )
    return apply_user_assistant_template(user_part, assistant_part)


def build_instruction_following_prompt(prompt: str, *, enable_think: bool = False) -> str:
    clean_prompt = prompt.lstrip().rstrip(" ")
    suffix = "<think" if enable_think else NAIVE_NOCOT_ASSISTANT_PREFIX
    return f"User: {clean_prompt}\n\nAssistant: {suffix}"


def build_human_eval_expected_context(
    prompt: str,
    *,
    assistant_code_prefix: str | None,
    cot_mode: CoTMode,
) -> str:
    user_part = (
        "You are a top-level code master. Complete the following code without any additional text or explanation:\n"
        f"{prompt}"
    )
    prefix = assistant_code_prefix or ""
    assistant_part = {
        CoTMode.NO_COT: (
            "<think>\n"
            "</think>\n"
            f"```python{prefix}{CODE_COMPLETION_PLACEHOLDER}"
        ),
        CoTMode.COT: (
            f"<think>{COT_PLACEHOLDER}</think>\n"
            f"```python{prefix}{CODE_COMPLETION_PLACEHOLDER}"
        ),
    }[cot_mode]
    return apply_user_assistant_template(user_part, assistant_part)


def trim_empty_lines(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def extract_function_signature(code: str | None) -> str | None:
    if not code:
        return None
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") and stripped.endswith(":"):
            return stripped
    return None


def build_mbpp_expected_context(
    prompt: str,
    *,
    signature: str | None,
    cot_mode: CoTMode,
) -> str:
    prompt_body = (
        f"{prompt}\nFunction signature: {signature}\nWrite the full function definition."
        if signature
        else prompt
    )
    prompt_body = trim_empty_lines(prompt_body)
    user_part = (
        "You are a top-level code master. Complete the following code without any additional text or explanation:\n"
        f"{prompt_body}"
    )
    assistant_part = {
        CoTMode.NO_COT: f"<think></think>\n```python{CODE_COMPLETION_PLACEHOLDER}",
        CoTMode.COT: (
            f"<think>{COT_PLACEHOLDER}</think>\n"
            f"```python{CODE_COMPLETION_PLACEHOLDER}"
        ),
    }[cot_mode]
    return apply_user_assistant_template(user_part, assistant_part)


def build_livecodebench_expected_context(
    prompt: str,
    *,
    starter_code: str | None,
    cot_mode: CoTMode,
) -> str:
    if cot_mode is not CoTMode.COT:
        raise ValueError(f"livecodebench only supports CoT mode, got {cot_mode!r}")
    clean = (prompt or "").strip()
    body = f"### Question:\n{clean}\n\n"
    if starter_code and starter_code.strip():
        body += (
            "### Format: You will use the following starter code to write the solution to the problem "
            "and enclose your code within delimiters.\n"
        )
        body += f"```python\n{starter_code}\n```\n\n"
    else:
        body += (
            "### Format: Read the inputs from stdin solve the problem and write the answer to STDOUT "
            "(do not directly test on the sample inputs). Enclose your code within delimiters as follows. "
            "Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
        )
        body += "```python\n# YOUR CODE HERE\n```\n\n"
    body += "### Answer: (use the provided format with backticks)\n\n"
    return (
        "User: You are an expert Python programmer. You will be given a question "
        "(problem specification) and will generate a correct Python program "
        "that matches the specification and passes all tests.\n"
        f"{body}"
        f"Assistant: <think{COT_PLACEHOLDER}\n</think>\n```python\n{CODE_COMPLETION_PLACEHOLDER}"
    )


__all__ = [
    "ALPHABET",
    "CODE_COMPLETION_PLACEHOLDER",
    "COT_PLACEHOLDER",
    "FINAL_ANSWER_PLACEHOLDER",
    "LOGPROBS_PLACEHOLDER",
    "apply_user_assistant_template",
    "build_human_eval_expected_context",
    "build_instruction_following_prompt",
    "build_livecodebench_expected_context",
    "build_maths_expected_context",
    "build_mbpp_expected_context",
    "build_multiple_choice_expected_context",
    "concat_choices",
    "extract_function_signature",
    "normalize_subject",
    "prompt_for_cot",
    "prompt_for_marker",
    "render_context",
    "trim_empty_lines",
]
