from __future__ import annotations

"""Instruction-following evaluation over canonical `results/completions` JSONL."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


from src.eval.datasets.data_loader.instruction_following import JsonlInstructionFollowingLoader
from src.eval.k_values import NumericK
from src.eval.metrics.at_k import compute_avg_at_k
from src.eval.results.io import iter_jsonl
from src.eval.results.schema import make_eval_payload, strict_nonneg_int
from src.eval.naive_prompt_protocol import strip_generated_empty_think_closer

from . import instructions_registry

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_LEADING_THINK_RE = re.compile(r"^\s*<think\b[^>]*>", re.IGNORECASE | re.DOTALL)


@dataclass(slots=True)
class InstructionFollowingMetrics:
    prompt_accuracy: float
    instruction_accuracy: float
    tier0_accuracy: dict[str, float]
    tier1_accuracy: dict[str, float]
    avg_at_k: dict[str, float] | None = None
    samples: int = 0
    payloads: list[dict] | None = None


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


def _response_from_completion(payload: dict) -> str:
    last_stage = _max_stage_index(payload)
    completion = str(payload.get(f"completion{last_stage}", "") or "")
    return _strip_reasoning_block(completion)


def _strip_reasoning_block(text: str) -> str:
    body = strip_generated_empty_think_closer(text)
    body = _THINK_BLOCK_RE.sub("", body).strip()
    if _LEADING_THINK_RE.match(body):
        return ""
    return body


def evaluate_instruction_following(
    completions: Iterable[dict] | str | Path,
    *,
    dataset_path: str | Path,
    dataset_slug: str | None = None,
    strict: bool = True,
    avg_k: tuple[NumericK, ...] = (),
) -> InstructionFollowingMetrics:
    dataset = list(JsonlInstructionFollowingLoader(str(dataset_path)).load())
    registry = _registry_for_dataset(dataset_path, dataset_slug=dataset_slug)

    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0
    tier0_total: dict[str, int] = {}
    tier0_correct: dict[str, int] = {}
    tier1_total: dict[str, int] = {}
    tier1_correct: dict[str, int] = {}

    rows_for_avg: list[tuple[int, int, bool]] = []

    eval_payloads: list[dict] = []
    for payload in _iter_completions(completions):
        sample_index = strict_nonneg_int(payload.get("sample_index"), "sample_index")
        repeat_index = strict_nonneg_int(payload.get("repeat_index"), "repeat_index")

        if sample_index < 0 or sample_index >= len(dataset):
            response = _response_from_completion(payload)
            follow_list: list[bool] = []
            ref_answer = ""
        else:
            record = dataset[sample_index]
            response = _response_from_completion(payload)
            follow_list = []
            ref_parts: list[str] = []

            variants = None if strict else _build_loose_variants(response)
            for idx, instruction_id in enumerate(record.instruction_ids):
                instruction_cls = registry[instruction_id]
                instruction = instruction_cls(instruction_id)
                kwargs = {
                    key: value
                    for key, value in record.kwargs_list[idx].items()
                    if value is not None
                }
                description = instruction.build_description(**kwargs)
                args = instruction.get_instruction_args()
                if args and "prompt" in args:
                    description = instruction.build_description(prompt=record.prompt)

                if strict:
                    is_following = bool(response.strip() and instruction.check_following(response))
                else:
                    is_following = False
                    for variant in variants:
                        if variant.strip() and instruction.check_following(variant):
                            is_following = True
                            break
                follow_list.append(is_following)
                if description:
                    ref_parts.append(f"{instruction_id}: {description}")
                else:
                    ref_parts.append(str(instruction_id))

                tier0_key = instruction_id.split(":")[0]
                tier0_total[tier0_key] = tier0_total.get(tier0_key, 0) + 1
                tier1_total[instruction_id] = tier1_total.get(instruction_id, 0) + 1
                if is_following:
                    tier0_correct[tier0_key] = tier0_correct.get(tier0_key, 0) + 1
                    tier1_correct[instruction_id] = tier1_correct.get(instruction_id, 0) + 1
            ref_answer = "\n".join(ref_parts)

        follow_all = bool(follow_list) and all(follow_list)
        prompt_total += 1
        if follow_all:
            prompt_correct += 1
        instruction_total += len(follow_list)
        instruction_correct += sum(1 for flag in follow_list if flag)

        rows_for_avg.append((sample_index, repeat_index, follow_all))
        eval_payloads.append(
            make_eval_payload(
                payload,
                is_passed=follow_all,
                answer=response,
                ref_answer=ref_answer,
            )
        )

    prompt_accuracy = prompt_correct / prompt_total if prompt_total else 0.0
    instruction_accuracy = instruction_correct / instruction_total if instruction_total else 0.0
    tier0_accuracy = {
        key: tier0_correct.get(key, 0) / total if total else 0.0
        for key, total in tier0_total.items()
    }
    tier1_accuracy = {
        key: tier1_correct.get(key, 0) / total if total else 0.0
        for key, total in tier1_total.items()
    }

    metrics = InstructionFollowingMetrics(
        prompt_accuracy=prompt_accuracy,
        instruction_accuracy=instruction_accuracy,
        tier0_accuracy=tier0_accuracy,
        tier1_accuracy=tier1_accuracy,
        samples=prompt_total,
        payloads=eval_payloads,
    )
    if avg_k:
        metrics.avg_at_k = compute_avg_at_k(rows_for_avg, avg_k)
    return metrics


def _registry_for_dataset(dataset_path: str | Path, *, dataset_slug: str | None) -> dict:
    slug = str(dataset_slug or "").lower()
    path_parts = {part.lower() for part in Path(dataset_path).parts}
    if slug.startswith("ifbench") or "ifbench" in path_parts:
        from .ifbench_official import instructions_registry as ifbench_registry

        return ifbench_registry.INSTRUCTION_DICT
    return instructions_registry.INSTRUCTION_DICT


def _build_loose_variants(response: str) -> list[str]:
    lines = response.split("\n")
    variants = [response]
    if len(lines) > 1:
        variants.append("\n".join(lines[1:]).strip())
        variants.append("\n".join(lines[:-1]).strip())
        variants.append("\n".join(lines[1:-1]).strip())
    stripped = response.replace("*", "")
    variants.append(stripped)
    more = [text.replace("*", "") for text in variants if text]
    variants.extend(more)
    return [v for v in variants if v]


__all__ = ["InstructionFollowingMetrics", "evaluate_instruction_following"]
