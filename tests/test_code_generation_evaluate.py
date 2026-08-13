from __future__ import annotations

from src.eval.metrics.code_generation.evaluate import extract_code_completion


def test_extract_code_completion_keeps_bare_code() -> None:
    completion = "def add(a, b):\n    return a + b\n"

    assert extract_code_completion(completion) == "def add(a, b):\n    return a + b"


def test_extract_code_completion_removes_generated_empty_think_closer() -> None:
    completion = ">\n```python\ndef add(a, b):\n    return a + b\n```"
    assert extract_code_completion(completion) == "def add(a, b):\n    return a + b"


def test_extract_code_completion_removes_think_block_and_python_fence() -> None:
    completion = (
        "<think>Need a direct implementation.</think>\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n"
    )

    assert extract_code_completion(completion) == "def add(a, b):\n    return a + b"


def test_extract_code_completion_handles_leading_end_think_fenced_answer() -> None:
    completion = (
        "\n</think>\n"
        "```python\n"
        "from typing import List\n\n"
        "def mean_absolute_deviation(numbers: List[float]) -> float:\n"
        "    mean = sum(numbers) / len(numbers)\n"
        "    return sum(abs(x - mean) for x in numbers) / len(numbers)\n"
        "```\n"
    )

    assert extract_code_completion(completion).startswith("from typing import List")
    assert "```" not in extract_code_completion(completion)
    assert "</think>" not in extract_code_completion(completion)


def test_extract_code_completion_removes_dangling_trailing_fence() -> None:
    completion = (
        "from typing import List\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        "    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i + 1:])\n"
        "```"
    )

    extracted = extract_code_completion(completion)

    assert extracted.startswith("from typing import List")
    assert "```" not in extracted
