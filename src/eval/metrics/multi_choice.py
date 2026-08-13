from __future__ import annotations

"""Multiple-choice evaluation over canonical `results/completions` JSONL."""

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Iterable, Sequence


from src.eval.datasets.data_loader.multiple_choice import JsonlMultipleChoiceLoader
from src.eval.results.io import iter_jsonl
from src.eval.results.schema import make_eval_payload, strict_nonneg_int
from src.eval.naive_prompt_protocol import strip_generated_empty_think_closer

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LETTER_RE = re.compile(r"[A-Z]")


@dataclass(slots=True)
class MultipleChoiceMetrics:
    accuracy: float
    accuracy_by_subject: dict[str | None, float]
    samples: int
    rows: list[tuple[int, int, bool]]
    payloads: list[dict]


@dataclass(slots=True)
class MultipleChoiceCascadeMetrics:
    rows_by_group: dict[str, list[tuple[int, int, bool]]]
    score_rows_by_group: dict[str, list[tuple[int, int, float]]]
    payloads_by_group: dict[str, list[dict]]
    metrics_by_group: dict[str, dict[str, float]]
    samples: int
    primary_group: str = "strategy_b"


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


def _extract_choice_letter(token_text: str) -> str | None:
    match = _LETTER_RE.search(token_text or "")
    return match.group(0) if match else None


def _extract_direct_choice(
    text: str,
    num_choices: int,
    choices: Sequence[str] | None = None,
) -> str:
    """Extract a direct-fill answer without trusting a stale derived field.

    Persisted completion payloads may contain both the model's raw text and a
    historical ``completionN`` value produced by the answer adapter that was
    active at generation time.  Re-evaluation must start from the immutable
    raw text; otherwise fixing the adapter cannot repair those rows.
    """

    text = strip_generated_empty_think_closer(text)
    valid_letters = ALPHABET[:num_choices]
    leading = re.match(
        rf"^\s*[\[(ï¼ˆ]?([{re.escape(valid_letters)}])"
        r"(?:[\])ï¼‰.,:ï¼šã€-]|\s|$)",
        text or "",
        re.IGNORECASE,
    )
    if leading:
        return leading.group(1).upper()
    return extract_answer_after_think(text, num_choices, choices)


def _normalize_choice_text(text: str) -> str:
    text = str(text or "").strip().casefold()
    text = re.sub(r"^[\s\-–—•]+", "", text)
    text = re.sub(r"^[\s\W]*[A-Z]\s*[\).:：、-]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[*_`~\"'“”‘’（）()\[\]{}<>]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n,.;:：。！？!?")


def _candidate_answer_texts(answer_text: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(
        r"<answer>\s*(.*?)\s*</answer>",
        answer_text or "",
        re.IGNORECASE | re.DOTALL,
    ):
        candidates.append(match.group(1).strip())
    cue_pattern = re.compile(
        r"(?:final\s+answer|correct\s+answer|answer)\s*"
        r"(?:(?:choice|option)\s*)?(?:is\s*|[:=]\s*)\s*(.+)",
        re.IGNORECASE,
    )
    for line in (answer_text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = cue_pattern.search(stripped)
        if match:
            candidates.append(match.group(1).strip())
    nonempty_lines = [line.strip() for line in (answer_text or "").splitlines() if line.strip()]
    if nonempty_lines:
        tail = nonempty_lines[-1]
        if len(tail) <= 120:
            candidates.append(tail)
    return candidates


def _extract_choice_by_option_text(
    answer_text: str,
    choices: Sequence[str] | None,
) -> str:
    if not choices:
        return ""
    normalized_choices = [_normalize_choice_text(choice) for choice in choices]
    # This is a deliberately narrow, last-resort fallback. If the model
    # revises its answer, the later explicit answer must win; walking the
    # candidates backwards keeps this fallback aligned with the formal parser
    # below instead of resurrecting an early discarded option.
    for candidate in reversed(_candidate_answer_texts(answer_text)):
        normalized_candidate = _normalize_choice_text(candidate)
        if not normalized_candidate:
            continue
        matches: list[int] = []
        for idx, normalized_choice in enumerate(normalized_choices):
            if not normalized_choice:
                continue
            if normalized_candidate == normalized_choice:
                matches.append(idx)
            elif (
                len(normalized_choice) >= 2
                and re.search(
                    rf"(?<!\w){re.escape(normalized_choice)}(?!\w)",
                    normalized_candidate,
                )
            ):
                matches.append(idx)
            elif (
                len(normalized_candidate) >= 3
                and re.search(
                    rf"(?<!\w){re.escape(normalized_candidate)}(?!\w)",
                    normalized_choice,
                )
            ):
                matches.append(idx)
        if len(set(matches)) == 1:
            return ALPHABET[matches[0]]
    return ""


def extract_answer_after_think(
    text: str,
    num_choices: int,
    choices: Sequence[str] | None = None,
) -> str:
    valid_letters = ALPHABET[:num_choices]
    reasoning, separator, answer_text = (text or "").rpartition("</think>")
    if not separator:
        malformed_close = re.search(r"</think\s*[)>]", text or "", re.IGNORECASE)
        if malformed_close:
            answer_text = (text or "")[malformed_close.end() :]
        else:
            # A truncated CoT may still contain one explicit final answer.
            # Parse that raw completion directly, but never infer a choice
            # from arbitrary capital letters or option text in its reasoning.
            answer_text = text or ""
    decoration = r"[*_`~]*"
    patterns = (
        rf"\\boxed\s*\{{[^}}]*?{decoration}([{re.escape(valid_letters)}])[^}}]*\}}",
        rf"(?:final\s+answer|correct\s+answer|answer)\s*(?:(?:choice|option)\s*)?"
        rf"(?:is\s*|[:=]\s*){decoration}\(?\s*(?:option\s*)?"
        r"([1-9][0-9]?)\b",
        r"(?:option|choice)\s*([1-9][0-9]?)\b",
        rf"(?:final\s+answer|correct\s+answer|answer)\s*(?:(?:choice|option)\s*)?"
        rf"(?:is\s*|[:=]\s*){decoration}\(?\s*([{re.escape(valid_letters)}])",
        rf"(?:choice|option)?\s*{decoration}\(?\s*([{re.escape(valid_letters)}])\s*\)?"
        rf"{decoration}\s+is\s+(?:the\s+)?(?:final\s+|correct\s+)?answer",
        rf"(?:最终答案|正确答案|答案|选择|选项)\s*(?:是|为|[:：=])?\s*"
        rf"{decoration}[（(]?([{re.escape(valid_letters)}])",
    )
    matches: list[tuple[int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, answer_text, re.IGNORECASE):
            raw = match.group(1)
            if raw.isdigit():
                index = int(raw) - 1
                if not 0 <= index < num_choices:
                    continue
                matches.append((match.start(), ALPHABET[index]))
                continue
            matches.append((match.start(), raw.upper()))
    if matches:
        return max(matches, key=lambda item: item[0])[1]
    for line in reversed(answer_text.splitlines()):
        match = re.fullmatch(
            rf"\s*(?:final\s+answer\s*[:=]?\s*)?{decoration}[\[(]?"
            rf"([{re.escape(valid_letters)}])[\])]?{decoration}[.!]?\s*",
            line,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()
    # When a normal completion closed its think block but omitted a clean
    # answer segment, accept only its last explicit answer cue from the
    # reasoning. This mirrors the reference Albatross extractor and avoids
    # treating a random uppercase character as an answer.
    if separator:
        reasoning_matches = list(
            re.finditer(
                rf"(?:final\s+answer|correct\s+answer|answer|最终答案|正确答案|答案)"
                rf"\s*(?:is\s*|[:=：]\s*|是\s*|为\s*)?{decoration}[（(]?"
                rf"([{re.escape(valid_letters)}])",
                reasoning,
                re.IGNORECASE,
            )
        )
        if reasoning_matches:
            return reasoning_matches[-1].group(1).upper()
    option_text_letter = _extract_choice_by_option_text(answer_text, choices)
    if option_text_letter in valid_letters:
        return option_text_letter
    return ""


def evaluate_multiple_choice_cascade(
    completions: Iterable[dict] | str | Path,
    *,
    dataset_path: str | Path,
    missing_prediction_score: float = 0.0,
) -> MultipleChoiceCascadeMetrics:
    dataset = list(JsonlMultipleChoiceLoader(str(dataset_path)).load())
    groups = ("strategy_a", "strategy_b")
    rows_by_group: dict[str, list[tuple[int, int, bool]]] = {group: [] for group in groups}
    score_rows_by_group: dict[str, list[tuple[int, int, float]]] = {
        group: [] for group in groups
    }
    payloads_by_group: dict[str, list[dict]] = {group: [] for group in groups}
    valid_by_group = {group: 0 for group in groups}
    correct_by_group = {group: 0 for group in groups}
    rescued = 0
    rerouted = 0
    total = 0

    for payload in _iter_completions(completions):
        sample_index = strict_nonneg_int(payload.get("sample_index"), "sample_index")
        repeat_index = strict_nonneg_int(payload.get("repeat_index"), "repeat_index")
        record = dataset[sample_index] if 0 <= sample_index < len(dataset) else None
        answer_letter = ALPHABET[record.answer_index] if record is not None else ""
        num_choices = len(record.choices) if record is not None else len(ALPHABET)
        strategy_a_text = str(payload.get("strategy_a_completion", ""))
        strategy_a_prediction = extract_answer_after_think(
            strategy_a_text,
            num_choices,
            record.choices if record is not None else None,
        )
        strategy_a_passed = bool(answer_letter) and strategy_a_prediction == answer_letter
        strategy_a_score = (
            1.0
            if strategy_a_passed
            else float(missing_prediction_score)
            if not strategy_a_prediction
            else 0.0
        )

        last_stage = _max_stage_index(payload)
        strategy_b_generated = (
            _extract_choice_letter(str(payload.get(f"completion{last_stage}", ""))) or ""
            if last_stage > 0
            else ""
        )
        if not strategy_b_generated and last_stage > 0:
            strategy_b_generated = extract_answer_after_think(
                str(payload.get("completion1", "")),
                num_choices,
                record.choices if record is not None else None,
            )
        if strategy_a_passed:
            strategy_b_prediction = strategy_a_prediction
            strategy_b_passed = True
        else:
            rerouted += 1
            strategy_b_prediction = strategy_b_generated or strategy_a_prediction
            strategy_b_passed = bool(answer_letter) and strategy_b_generated == answer_letter
            if strategy_b_passed:
                rescued += 1
        # Chance credit for an unparseable strategy-A completion must not
        # survive once strategy B produced a legal answer.  At that point the
        # final prediction is observed, so a wrong choice scores zero.  Keep
        # the missing-prediction fallback only when neither strategy yielded
        # any usable answer.
        strategy_b_score = (
            1.0
            if strategy_b_passed
            else 0.0
            if strategy_b_prediction
            else strategy_a_score
        )

        group_values = {
            "strategy_a": (strategy_a_prediction, strategy_a_passed, strategy_a_score),
            "strategy_b": (strategy_b_prediction, strategy_b_passed, strategy_b_score),
        }
        strategy_a_context = f"{payload.get('strategy_a_prompt', '')}{strategy_a_text}"
        for group, (prediction, passed, score) in group_values.items():
            rows_by_group[group].append((sample_index, repeat_index, passed))
            score_rows_by_group[group].append((sample_index, repeat_index, score))
            valid_by_group[group] += int(bool(prediction))
            correct_by_group[group] += int(passed)
            eval_payload = make_eval_payload(
                payload,
                is_passed=passed,
                fail_reason="missing_prediction" if not prediction else "incorrect",
                answer=prediction,
                ref_answer=answer_letter,
            )
            if group == "strategy_a" or not eval_payload["context"]:
                eval_payload["context"] = strategy_a_context
            payloads_by_group[group].append(eval_payload)
        total += 1

    denominator = total or 1
    metrics_by_group = {
        group: {
            "exact_accuracy": correct_by_group[group] / denominator,
            "score": sum(row[2] for row in score_rows_by_group[group]) / denominator,
            "valid": valid_by_group[group] / denominator,
        }
        for group in groups
    }
    metrics_by_group["strategy_b"].update(
        {
            "rerouted": float(rerouted),
            "rescued": float(rescued),
            "rescue_rate": rescued / rerouted if rerouted else 0.0,
        }
    )
    return MultipleChoiceCascadeMetrics(
        rows_by_group=rows_by_group,
        score_rows_by_group=score_rows_by_group,
        payloads_by_group=payloads_by_group,
        metrics_by_group=metrics_by_group,
        samples=total,
    )


def evaluate_multiple_choice(
    completions: Iterable[dict] | str | Path,
    *,
    dataset_path: str | Path,
) -> MultipleChoiceMetrics:
    """Evaluate multiple-choice completions and return canonical eval payloads."""

    dataset = list(JsonlMultipleChoiceLoader(str(dataset_path)).load())
    total = 0
    correct = 0
    subject_totals: dict[str | None, tuple[int, int]] = {}
    eval_payloads: list[dict] = []
    rows_for_at_k: list[tuple[int, int, bool]] = []

    for payload in _iter_completions(completions):
        sample_index = strict_nonneg_int(payload.get("sample_index"), "sample_index")
        repeat_index = strict_nonneg_int(payload.get("repeat_index"), "repeat_index")
        if sample_index < 0 or sample_index >= len(dataset):
            # Unknown sample index -> mark incorrect, but still emit an eval row.
            passed = False
            subject = None
            answer_letter = None
            predicted = ""
        else:
            record = dataset[sample_index]
            subject = record.subject
            answer_letter = ALPHABET[record.answer_index]
            raw_text = str(payload.get("direct_raw_completion", "") or "")
            if raw_text:
                predicted = _extract_direct_choice(
                    raw_text,
                    len(record.choices),
                    record.choices,
                )
            else:
                last_stage = _max_stage_index(payload)
                token_text = str(payload.get(f"completion{last_stage}", ""))
                predicted = _extract_direct_choice(
                    token_text,
                    len(record.choices),
                    record.choices,
                )
            passed = bool(predicted) and predicted == answer_letter

        total += 1
        if passed:
            correct += 1
        rows_for_at_k.append((sample_index, repeat_index, passed))

        sub_total, sub_hits = subject_totals.get(subject, (0, 0))
        sub_total += 1
        if passed:
            sub_hits += 1
        subject_totals[subject] = (sub_total, sub_hits)

        eval_payloads.append(
            make_eval_payload(
                payload,
                is_passed=passed,
                answer=predicted or "",
                ref_answer=answer_letter or "",
            )
        )

    accuracy_by_subject = {
        subj: (hits / count if count else 0.0) for subj, (count, hits) in subject_totals.items()
    }
    accuracy = correct / total if total else 0.0
    return MultipleChoiceMetrics(
        accuracy=accuracy,
        accuracy_by_subject=accuracy_by_subject,
        samples=total,
        rows=rows_for_at_k,
        payloads=eval_payloads,
    )


__all__ = [
    "MultipleChoiceCascadeMetrics",
    "MultipleChoiceMetrics",
    "evaluate_multiple_choice",
    "evaluate_multiple_choice_cascade",
    "extract_answer_after_think",
]
