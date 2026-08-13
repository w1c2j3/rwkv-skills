from __future__ import annotations

"""Code-generation evaluation adapter for canonical `results/completions` JSONL.

The upstream HumanEval/MBPP evaluators expect sample JSONL lines containing
`task_id` and `completion`. Our canonical completions schema intentionally
does not store dataset labels/ids, so this module bridges by joining:
- canonical completions (sample_index/repeat_index + completion text)
- the original dataset file (to recover task_id)

It then writes canonical evaluator output (results/eval) with:
benchmark_name, dataset_split, sample_index, repeat_index, context,
answer, ref_answer, is_passed, fail_reason
"""

import json
import orjson
from pathlib import Path
import re
import tempfile
from typing import Iterable


from src.eval.datasets.data_loader.code_generation import JsonlCodeGenerationLoader
from src.eval.datasets.data_struct.code_generation import CodeGenerationRecord
from src.eval.results.io import iter_jsonl
from src.eval.results.schema import build_context_from_completions, strict_nonneg_int
from src.eval.metrics.code_generation.human_eval import evaluate_functional_correctness
from src.eval.metrics.code_generation.mbpp import evaluate_mbpp
from src.eval.naive_prompt_protocol import strip_generated_empty_think_closer

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_FENCED_CODE_RE = re.compile(
    r"```[ \t]*(?:python|py)?[^\S\r\n]*\r?\n(?P<code>.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_LEADING_END_THINK_RE = re.compile(r"^[\s\r\n]*</think>[ \t]*\r?\n?", re.IGNORECASE)
_STANDALONE_CODE_FENCE_RE = re.compile(r"^[ \t]*```[ \t]*(?:python|py)?[ \t]*$", re.IGNORECASE)


def _iter_completions(source: Iterable[dict] | str | Path) -> Iterable[dict]:
    if isinstance(source, (str, Path)):
        yield from iter_jsonl(source)
        return
    yield from source


def _max_stage_index(payload: dict) -> int:
    stage = 0
    for key in payload:
        if key.startswith("completion") and key.removeprefix("completion").isdigit():
            stage = max(stage, int(key.removeprefix("completion")))
    return stage


def extract_code_completion(text: str) -> str:
    """Recover executable code from chatty code-benchmark completions."""

    if not text:
        return ""
    body = strip_generated_empty_think_closer(text)
    body = _THINK_BLOCK_RE.sub("", body)
    body = _LEADING_END_THINK_RE.sub("", body, count=1)
    matches = list(_FENCED_CODE_RE.finditer(body))
    if matches:
        return matches[-1].group("code").strip("\r\n").rstrip()
    return _strip_standalone_code_fence_edges(body)


def _strip_standalone_code_fence_edges(text: str) -> str:
    lines = str(text or "").rstrip().splitlines()
    while lines and _STANDALONE_CODE_FENCE_RE.match(lines[0]):
        lines.pop(0)
    while lines and _STANDALONE_CODE_FENCE_RE.match(lines[-1]):
        lines.pop()
    return "\n".join(lines).rstrip()


def _write_temp_samples(
    completions: Iterable[dict] | str | Path,
    *,
    dataset_records: list,
    temp_path: Path,
) -> int:
    """Write evaluator-compatible samples JSONL into `temp_path`."""

    count = 0
    with temp_path.open("wb") as out_f:
        for payload in _iter_completions(completions):
            sample_index = strict_nonneg_int(payload.get("sample_index"), "sample_index")
            repeat_index = strict_nonneg_int(payload.get("repeat_index"), "repeat_index")
            task_id = ""
            if 0 <= sample_index < len(dataset_records):
                task_id = str(getattr(dataset_records[sample_index], "task_id", ""))
            last_stage = _max_stage_index(payload)
            completion = str(payload.get(f"completion{last_stage}", "") or "")
            context = build_context_from_completions(payload)
            executable_completion = extract_code_completion(completion)
            sample = {
                "benchmark_name": payload.get("benchmark_name", ""),
                "dataset_split": payload.get("dataset_split", ""),
                "sample_index": sample_index,
                "repeat_index": repeat_index,
                "context": context,
                "task_id": task_id,
                # Keep evaluator behaviour consistent with prior runs: strip trailing whitespace.
                "completion": executable_completion,
            }
            out_f.write(orjson.dumps(sample, option=orjson.OPT_APPEND_NEWLINE))
            count += 1
    return count


def _resolve_reference_answer(record: CodeGenerationRecord | None) -> str:
    if record is None:
        return ""
    canonical = (record.canonical_solution or "").strip() if record.canonical_solution is not None else ""
    if canonical:
        return canonical
    tests = record.test_cases
    if tests is None:
        return ""
    if isinstance(tests, str):
        return tests
    try:
        return json.dumps(tests, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(tests)


def _build_canonical_eval_from_results(
    results_path: Path,
    *,
    dataset_records: list[CodeGenerationRecord],
) -> list[dict]:
    eval_payloads: list[dict] = []
    with results_path.open("r", encoding="utf-8") as in_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            passed = bool(payload.get("passed", False))
            result = str(payload.get("result", "") or "")
            sample_index = strict_nonneg_int(payload.get("sample_index"), "sample_index")
            repeat_index = strict_nonneg_int(payload.get("repeat_index"), "repeat_index")
            record = dataset_records[sample_index] if 0 <= sample_index < len(dataset_records) else None
            eval_payloads.append(
                {
                    "benchmark_name": str(payload.get("benchmark_name", "")),
                    "dataset_split": str(payload.get("dataset_split", "")),
                    "sample_index": sample_index,
                    "repeat_index": repeat_index,
                    "context": str(payload.get("context", "")),
                    "answer": str(payload.get("completion", "") or ""),
                    "ref_answer": _resolve_reference_answer(record),
                    "is_passed": passed,
                    "fail_reason": "" if passed else result,
                }
            )
    return eval_payloads


def eval_rows_from_payloads(payloads: Iterable[dict]) -> list[tuple[int, int, bool]]:
    rows: list[tuple[int, int, bool]] = []
    for payload in payloads:
        sample_index = strict_nonneg_int(payload.get("sample_index"), "sample_index")
        repeat_index = strict_nonneg_int(payload.get("repeat_index"), "repeat_index")
        rows.append((sample_index, repeat_index, bool(payload.get("is_passed"))))
    return rows


def evaluate_human_eval(
    completions: Iterable[dict] | str | Path,
    *,
    dataset_path: str | Path,
    pass_k: tuple[int, ...] = (1,),
    n_workers: int = 4,
    timeout: float = 3.0,
) -> tuple[dict[str, float], list[dict]]:
    """Run HumanEval functional correctness eval and return canonical eval payloads."""

    dataset_records = list(JsonlCodeGenerationLoader(str(dataset_path)).load())
    with tempfile.TemporaryDirectory(prefix="rwkv_skills_humaneval_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        sample_file = tmp_dir_path / "samples.jsonl"
        _write_temp_samples(
            completions,
            dataset_records=dataset_records,
            temp_path=sample_file,
        )
        metrics, results_path_str = evaluate_functional_correctness(
            sample_file=str(sample_file),
            k=tuple(pass_k),
            n_workers=n_workers,
            timeout=timeout,
            problem_file=str(dataset_path),
        )
        eval_payloads = _build_canonical_eval_from_results(
            Path(results_path_str),
            dataset_records=dataset_records,
        )
        return metrics or {}, eval_payloads


def evaluate_mbpp_dataset(
    completions: Iterable[dict] | str | Path,
    *,
    dataset_path: str | Path,
    pass_k: tuple[int, ...] = (1,),
    n_workers: int = 4,
    timeout: float = 3.0,
) -> tuple[dict[str, float], list[dict]]:
    """Run MBPP (or MBPP+) eval and return canonical eval payloads."""

    dataset_records = list(JsonlCodeGenerationLoader(str(dataset_path)).load())
    with tempfile.TemporaryDirectory(prefix="rwkv_skills_mbpp_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        sample_file = tmp_dir_path / "samples.jsonl"
        _write_temp_samples(
            completions,
            dataset_records=dataset_records,
            temp_path=sample_file,
        )
        metrics, results_path_str = evaluate_mbpp(
            sample_file=str(sample_file),
            k=tuple(pass_k),
            n_workers=n_workers,
            timeout=timeout,
            problem_file=str(dataset_path),
        )
        eval_payloads = _build_canonical_eval_from_results(
            Path(results_path_str),
            dataset_records=dataset_records,
        )
        return metrics or {}, eval_payloads


__all__ = [
    "eval_rows_from_payloads",
    "evaluate_human_eval",
    "evaluate_mbpp_dataset",
    "extract_code_completion",
]
