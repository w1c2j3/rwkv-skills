"""Maths benchmark pipeline for full free-response generation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Sequence

from src.eval.datasets.data_loader.free_answer import JsonlFreeAnswerLoader
from src.eval.datasets.data_struct.free_answer import FreeAnswerRecord
from src.eval.execution_plan import AttemptKey
from src.eval.results.schema import (
    dataset_slug_parts,
    normalize_sampling_config_by_stage,
    prompt_delta,
    sampling_config_to_dict,
)
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.eval.context_budget import compose_context_question
from src.eval.long_doc_evidence import LongDocEvidenceConfig
from src.infer.backend import InferenceBackend, resolve_generation_prompt_batch_size
from src.infer.sampling import GenerationOutput, SamplingConfig
from src.eval.naive_prompt_protocol import NAIVE_NOCOT_ASSISTANT_PREFIX

USER_SENTINEL = "\nUser:"
LEGACY_GENERATION_STOP_SUFFIXES = (USER_SENTINEL,)
G1H_GENERATION_STOP_SUFFIXES = (
    USER_SENTINEL,
    "\nUser✿",
    "User✿",
    "\nBot✿",
    "Bot✿",
    "\nAssistant:",
    "Assistant:",
    "✿",
)
G1H_REMOTE_STOP_SUFFIXES = tuple(
    suffix for suffix in G1H_GENERATION_STOP_SUFFIXES if suffix != "✿"
)
FREE_RESPONSE_STOP_TOKENS = (0,)
# The final-stage prompt pre-fills ``\boxed{``.  Stop on the closing box plus
# math-delimiter text rather than relying only on tokenizer-specific token IDs.
# A literal text stop remains correct when BPE merges ``}}\`` (common for
# answers such as ``\sqrt{2}``) into a token not present in ``stop_tokens``.
# Stopping on ``}`` alone would be unsafe because braces also occur inside the
# answer (fractions, roots, tuples, aligned expressions, ...).
FINAL_BOXED_STOP_SUFFIXES = ("}\\)", "} \\)", "}\\]")

DEFAULT_COT_PROMPT = """User: <Q>

Assistant: <think"""

DEFAULT_DIRECT_PROMPT = f"""User: <Q>

Assistant: {NAIVE_NOCOT_ASSISTANT_PREFIX}"""

DEFAULT_FINAL_PROMPT = """<Q><COT>
Therefore, the answer is \\(\\boxed{"""


def _render_prompt(template: str, question: str, *, prompt_max_chars: int | None = None) -> str:
    if prompt_max_chars is not None and prompt_max_chars > 0:
        empty_prompt_len = len(template.replace("<Q>", "").rstrip(" "))
        question_budget = max(0, prompt_max_chars - empty_prompt_len)
        question = _truncate_prompt_text(question.lstrip(), question_budget)
    return template.replace("<Q>", question.lstrip()).rstrip(" ")


def _render_final_prompt(template: str, cot_prompt: str, cot_completion: str) -> str:
    return template.replace("<Q>", cot_prompt).replace("<COT>", cot_completion).rstrip(" ")


def _truncate_prompt_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = "\n[...truncated...]\n"
    if max_chars <= len(marker) + 2:
        return text[-max_chars:]
    head_budget = min(512, max_chars // 5)
    tail_budget = max_chars - head_budget - len(marker)
    return f"{text[:head_budget].rstrip()}{marker}{text[-tail_budget:].lstrip()}"


def _generation_stop_suffixes_for_prompt(prompt: str) -> tuple[str, ...]:
    if "User✿" in prompt or "Bot✿" in prompt:
        return G1H_GENERATION_STOP_SUFFIXES
    return LEGACY_GENERATION_STOP_SUFFIXES


def _remote_stop_suffixes_for_prompt(prompt: str) -> tuple[str, ...]:
    if "User✿" in prompt or "Bot✿" in prompt:
        return G1H_REMOTE_STOP_SUFFIXES
    return LEGACY_GENERATION_STOP_SUFFIXES


def _prompt_stop_suffixes(prompts: Sequence[str]) -> list[tuple[str, ...]]:
    result: list[tuple[str, ...]] = []
    for prompt in prompts:
        suffixes = _remote_stop_suffixes_for_prompt(prompt)
        if prompt.rstrip().endswith("\\boxed{"):
            suffixes = (*suffixes, *FINAL_BOXED_STOP_SUFFIXES)
        result.append(suffixes)
    return result


def _clip_user_sentinel(text: str, *, prompt: str = "") -> str:
    suffixes = _generation_stop_suffixes_for_prompt(prompt)
    cut = len(text)
    for sentinel in suffixes:
        index = text.find(sentinel)
        if index >= 0:
            cut = min(cut, index)
    return text[:cut].rstrip()


def _is_truncated(output: GenerationOutput) -> bool:
    return output.finish_reason in {"max_tokens", "max_length"}


def _output_stats(output: GenerationOutput) -> dict[str, object]:
    truncated = _is_truncated(output)
    # Contract: class 1 means a normal/explicit stop (token 0 or a role/text
    # stop); class 2 means the configured generation budget was exhausted.
    # Remote backends may use different detail strings, so max_tokens and
    # max_length are the only class-2 signals.
    # The remote OpenAI/vLLM adapters preserve text and finish_reason but do
    # not receive token ids in the response.  Do not serialize that unknown
    # value as a false zero; local backends with token ids still report the
    # exact count.
    token_count: int | None = len(output.token_ids) if output.token_ids else None
    if token_count is None and output.tokens:
        token_count = len(output.tokens)
    return {
        "truncated": truncated,
        "stop_detail": output.finish_reason or "",
        "stop_reason_class": 2 if truncated else 1,
        "termination_reason": "max_tokens" if truncated else "stop",
        "generated_token_count": token_count,
    }


def _sum_known_token_counts(*stats: dict[str, object]) -> int | None:
    counts = [item.get("generated_token_count") for item in stats]
    if not all(isinstance(value, int) for value in counts):
        return None
    return sum(value for value in counts if isinstance(value, int))


@dataclass(slots=True)
class FreeResponsePipelineResult:
    dataset: str
    sample_count: int
    problem_count: int
    payloads: list[dict]


class FreeResponsePipeline:
    def __init__(self, backend: InferenceBackend) -> None:
        self.backend = backend

    def run(
        self,
        dataset_path: str,
        *,
        prompt_template: str = DEFAULT_COT_PROMPT,
        generation_sampling: SamplingConfig | None = None,
        strategy_a_prompt_template: str | None = None,
        strategy_a_sampling: SamplingConfig | None = None,
        strategy_a_filter_correct: bool = False,
        cot_prompt_template: str | None = None,
        cot_sampling: SamplingConfig | None = None,
        final_answer_template: str | None = None,
        final_sampling: SamplingConfig | None = None,
        batch_size: int = 64,
        dataset_name: str | None = None,
        sample_limit: int | None = None,
        record_indices: Sequence[int] | None = None,
        pad_to_batch: bool = False,
        pass_k: Iterable[int] | None = None,
        samples_per_task: int | None = None,
        probe_only: bool = False,
        attempt_keys: Sequence[AttemptKey] | None = None,
        resume_start_index: int = 0,
        skip_keys: set[tuple[int, int, int]] | None = None,
        on_record: Callable[[dict], None] | None = None,
        prompt_max_chars: int | None = None,
        long_doc_config: LongDocEvidenceConfig | None = None,
    ) -> FreeResponsePipelineResult:
        if cot_prompt_template:
            prompt_template = cot_prompt_template
        if generation_sampling is None:
            generation_sampling = cot_sampling
        if generation_sampling is None:
            raise ValueError("generation_sampling is required")
        use_final_stage = final_answer_template is not None or final_sampling is not None
        if use_final_stage:
            if final_answer_template is None:
                raise ValueError("final_answer_template is required when final_sampling is provided")
            if final_sampling is None:
                raise ValueError("final_sampling is required when final_answer_template is provided")

        samples_per_task = (samples_per_task or max(1, max(pass_k) if pass_k else 1)) if not probe_only else 1
        raw_records, resolved_name = self._load_records(
            dataset_path,
            sample_limit,
            record_indices=record_indices,
        )
        problem_count = len(raw_records)
        dataset_name = dataset_name or resolved_name
        benchmark_name, dataset_split = dataset_slug_parts(dataset_name)
        record_map = {int(idx): record for idx, record in raw_records}
        expanded: list[tuple[AttemptKey, FreeAnswerRecord]] = []
        repeats = max(1, samples_per_task)
        skip_keys = skip_keys or set()
        if attempt_keys is not None:
            total_expected = len(attempt_keys)
            for key in attempt_keys:
                record = record_map.get(int(key.sample_index))
                if record is None or key.as_tuple() in skip_keys:
                    continue
                expanded.append((key, record))
        else:
            total_expected = len(raw_records) * repeats
            for idx, record in raw_records:
                for sample_id in range(repeats):
                    key = AttemptKey(sample_index=int(idx), repeat_index=int(sample_id), pass_index=0)
                    if key.as_tuple() in skip_keys:
                        continue
                    expanded.append((key, record))
        if pad_to_batch and expanded and len(expanded) < batch_size:
            original_len = len(expanded)
            repeat = (batch_size + original_len - 1) // original_len
            expanded = (expanded * repeat)[:batch_size]
        if not expanded:
            return FreeResponsePipelineResult(dataset_name, 0, problem_count, [])
        long_doc_cfg = long_doc_config or LongDocEvidenceConfig(enabled=False)
        question_cache: dict[int, tuple[str, dict[str, object] | None]] = {}

        def _effective_question(key: AttemptKey, record: FreeAnswerRecord) -> tuple[str, dict[str, object] | None]:
            cached = question_cache.get(int(key.sample_index))
            if cached is not None:
                return cached
            value = compose_context_question(
                getattr(record, "context", None),
                record.question,
                long_doc_config=long_doc_cfg,
                label=f"{dataset_name}:{key.sample_index}",
            )
            question_cache[int(key.sample_index)] = value
            return value

        def _trace_for_prompt(trace: dict[str, object] | None, prompt: str) -> dict[str, object] | None:
            if trace is None:
                return None
            payload = dict(trace)
            payload["prompt_chars"] = len(prompt)
            return payload

        skipped = total_expected - len(expanded)
        if skipped > 0:
            print(f"⏩ 自由问答恢复运行：已跳过 {skipped}/{total_expected} 个样本")

        if resume_start_index < 0:
            resume_start_index = 0
        if resume_start_index:
            if resume_start_index >= len(expanded):
                return FreeResponsePipelineResult(dataset_name, len(expanded), problem_count, [])
            if resume_start_index % repeats != 0:
                print(
                    f"⚠️  Resume index {resume_start_index} 未按 repeats={repeats} 对齐，可能存在漏样本。"
                )
            remaining_entries = expanded[resume_start_index:]
            print(
                f"⏩ 自由问答恢复运行：已完成 {resume_start_index}/{len(expanded)}，剩余 {len(remaining_entries)}"
            )
        else:
            remaining_entries = expanded

        generation_sampling = replace(generation_sampling, stop_tokens=FREE_RESPONSE_STOP_TOKENS)
        sampling_items = [(1, generation_sampling)]
        if use_final_stage:
            assert final_sampling is not None
            sampling_items.append((2, final_sampling))
        sampling_config = normalize_sampling_config_by_stage(sampling_items)
        if strategy_a_sampling is not None:
            sampling_config["strategy_a"] = sampling_config_to_dict(strategy_a_sampling)

        if probe_only:
            prompts = [
                _render_prompt(prompt_template, _effective_question(key, record)[0], prompt_max_chars=prompt_max_chars)
                for key, record in remaining_entries
            ]
            if strategy_a_sampling is not None:
                strategy_template = strategy_a_prompt_template or prompt_template
                strategy_a_prompts = [
                    _render_prompt(strategy_template, _effective_question(key, record)[0], prompt_max_chars=prompt_max_chars)
                    for key, record in remaining_entries
                ]
                _ = self.backend.generate(
                    strategy_a_prompts,
                    sampling=strategy_a_sampling,
                    batch_size=batch_size,
                    progress_desc="Probing strategy A full responses",
                    probe_only=probe_only,
                    prompt_stop_suffixes=_prompt_stop_suffixes(strategy_a_prompts),
                    prompt_seeds=[
                        sample_repeat_seed(
                            key.sample_index,
                            key.repeat_index,
                            pass_index=key.pass_index,
                            stage=1,
                        )
                        for key, _record in remaining_entries
                    ],
                )
            probe_seeds = [
                sample_repeat_seed(
                    key.sample_index,
                    key.repeat_index,
                    pass_index=key.pass_index,
                    stage=2 if strategy_a_sampling is not None else 1,
                )
                for key, _record in remaining_entries
            ]
            cot_outputs = self.backend.generate(
                prompts,
                sampling=generation_sampling,
                batch_size=batch_size,
                progress_desc="Probing CoT" if use_final_stage else "Generating full responses",
                probe_only=probe_only,
                prompt_stop_suffixes=_prompt_stop_suffixes(prompts),
                prompt_seeds=probe_seeds,
            )
            if use_final_stage:
                assert final_answer_template is not None
                assert final_sampling is not None
                final_prompts: list[str] = []
                final_source_indices: list[int] = []
                for output in cot_outputs:
                    if 0 <= output.prompt_index < len(prompts):
                        final_prompts.append(
                            _render_final_prompt(
                                final_answer_template,
                                prompts[output.prompt_index],
                                _clip_user_sentinel(output.text, prompt=prompts[output.prompt_index]),
                            )
                        )
                        final_source_indices.append(int(output.prompt_index))
                if final_prompts:
                    _ = self.backend.generate(
                        final_prompts,
                        sampling=final_sampling,
                        batch_size=min(batch_size, len(final_prompts)),
                        progress_desc="Probing final answers",
                        probe_only=probe_only,
                        prompt_stop_suffixes=_prompt_stop_suffixes(final_prompts),
                        prompt_seeds=[
                            sample_repeat_seed(
                                remaining_entries[idx][0].sample_index,
                                remaining_entries[idx][0].repeat_index,
                                pass_index=remaining_entries[idx][0].pass_index,
                                stage=3 if strategy_a_sampling is not None else 2,
                            )
                            for idx in final_source_indices
                        ],
                    )
            return FreeResponsePipelineResult(dataset_name, len(expanded), problem_count, [])

        payloads: list[dict] = []
        chunk_size = resolve_generation_prompt_batch_size(self.backend, batch_size)
        for start in range(0, len(remaining_entries), chunk_size):
            chunk = remaining_entries[start : start + chunk_size]
            effective_questions = [_effective_question(key, record) for key, record in chunk]
            prompts = [
                _render_prompt(prompt_template, question, prompt_max_chars=prompt_max_chars)
                for question, _trace in effective_questions
            ]
            long_doc_traces = [
                _trace_for_prompt(trace, prompt)
                for (_question, trace), prompt in zip(effective_questions, prompts)
            ]

            if use_final_stage:
                assert final_answer_template is not None
                assert final_sampling is not None
                strategy_a_by_idx: dict[int, GenerationOutput] = {}
                strategy_a_prompts: list[str] = []
                if strategy_a_sampling is not None:
                    strategy_template = strategy_a_prompt_template or prompt_template
                    strategy_a_prompts = [
                        _render_prompt(strategy_template, effective_question, prompt_max_chars=prompt_max_chars)
                        for effective_question, _trace in effective_questions
                    ]
                    strategy_a_outputs = self.backend.generate(
                        strategy_a_prompts,
                        sampling=strategy_a_sampling,
                        batch_size=min(batch_size, len(strategy_a_prompts)),
                        progress_desc="Generating strategy A full responses",
                        prompt_stop_suffixes=_prompt_stop_suffixes(strategy_a_prompts),
                        prompt_seeds=[
                            sample_repeat_seed(
                                key.sample_index,
                                key.repeat_index,
                                pass_index=key.pass_index,
                                stage=1,
                            )
                            for key, _record in chunk
                        ],
                    )
                    strategy_a_by_idx = {
                        int(output.prompt_index): output
                        for output in strategy_a_outputs
                        if 0 <= int(output.prompt_index) < len(chunk)
                    }

                def _build_strategy_a_payload(local_idx: int, output: GenerationOutput) -> dict:
                    key, _record = chunk[local_idx]
                    strategy_stats = _output_stats(output)
                    payload = SampleRecord(
                        benchmark_name=benchmark_name,
                        dataset_split=dataset_split,
                        sample_index=key.sample_index,
                        repeat_index=key.repeat_index,
                        pass_index=key.pass_index,
                        sampling_config=sampling_config,
                        stages=[],
                    ).as_payload()
                    payload["strategy_a_prompt"] = strategy_a_prompts[local_idx]
                    payload["strategy_a_completion"] = _clip_user_sentinel(
                        output.text,
                        prompt=strategy_a_prompts[local_idx],
                    )
                    payload["strategy_a_stop_reason"] = output.finish_reason
                    payload["stats"] = {**strategy_stats, "strategy_a": strategy_stats}
                    trace = _trace_for_prompt(long_doc_traces[local_idx], strategy_a_prompts[local_idx])
                    if trace is not None:
                        payload["long_doc"] = trace
                    payload["_stage"] = "answer"
                    return payload

                two_stage_indices = list(range(len(chunk)))
                if strategy_a_sampling is not None and strategy_a_filter_correct:
                    from src.eval.metrics.free_response import (
                        STRATEGY_A,
                        resolve_reference_answer,
                        score_free_response_strategy,
                    )

                    two_stage_indices = []
                    for local_idx, (key, record) in enumerate(chunk):
                        strategy_a_output = strategy_a_by_idx.get(local_idx)
                        if strategy_a_output is None:
                            two_stage_indices.append(local_idx)
                            continue
                        payload = _build_strategy_a_payload(local_idx, strategy_a_output)
                        scored = score_free_response_strategy(
                            STRATEGY_A,
                            payload,
                            sample_index=key.sample_index,
                            repeat_index=key.repeat_index,
                            question=record.question,
                            reference=resolve_reference_answer(record),
                        )
                        if scored.final_passed:
                            if on_record is not None:
                                on_record(payload)
                            payloads.append(payload)
                        else:
                            two_stage_indices.append(local_idx)

                if not two_stage_indices:
                    continue

                cot_prompts = [prompts[local_idx] for local_idx in two_stage_indices]
                cot_outputs = self.backend.generate(
                    cot_prompts,
                    sampling=generation_sampling,
                    batch_size=min(batch_size, len(cot_prompts)),
                    progress_desc="Generating CoT",
                    prompt_stop_suffixes=_prompt_stop_suffixes(cot_prompts),
                    prompt_seeds=[
                        sample_repeat_seed(
                            chunk[local_idx][0].sample_index,
                            chunk[local_idx][0].repeat_index,
                            pass_index=chunk[local_idx][0].pass_index,
                            stage=2 if strategy_a_sampling is not None else 1,
                        )
                        for local_idx in two_stage_indices
                    ],
                )
                cot_by_idx: dict[int, GenerationOutput] = {
                    two_stage_indices[int(output.prompt_index)]: output
                    for output in cot_outputs
                    if 0 <= int(output.prompt_index) < len(two_stage_indices)
                }
                final_prompts: list[str] = []
                final_prompt_indices: list[int] = []
                final_prompt_deltas: dict[int, str] = {}
                cot_texts: dict[int, str] = {}
                for local_idx, cot_output in sorted(cot_by_idx.items()):
                    cot_text = _clip_user_sentinel(cot_output.text, prompt=prompts[local_idx])
                    cot_texts[local_idx] = cot_text
                    full_final_prompt = _render_final_prompt(
                        final_answer_template,
                        prompts[local_idx],
                        cot_text,
                    )
                    prior_context = f"{prompts[local_idx]}{cot_text}"
                    try:
                        delta_prompt = prompt_delta(full_final_prompt, prior_context)
                    except ValueError as exc:
                        raise ValueError(
                            "math final_answer_template must preserve the <Q><COT> prefix "
                            "so stage payloads can be reconstructed"
                        ) from exc
                    final_prompt_deltas[local_idx] = delta_prompt
                    final_prompt_indices.append(local_idx)
                    final_prompts.append(full_final_prompt)

                if not final_prompts:
                    continue

                def _on_final_complete(output: GenerationOutput) -> None:
                    prompt_idx = output.prompt_index
                    if prompt_idx < 0 or prompt_idx >= len(final_prompt_indices):
                        return
                    local_idx = final_prompt_indices[prompt_idx]
                    cot_output = cot_by_idx.get(local_idx)
                    if cot_output is None:
                        return
                    key, _record = chunk[local_idx]
                    cot_text = cot_texts[local_idx]
                    final_text = _clip_user_sentinel(output.text, prompt=final_prompts[prompt_idx])
                    cot_stage = StageRecord(
                        prompt=prompts[local_idx],
                        completion=cot_text,
                        stop_reason=cot_output.finish_reason,
                    )
                    final_stage = StageRecord(
                        prompt=final_prompt_deltas[local_idx],
                        completion=final_text,
                        stop_reason=output.finish_reason,
                    )
                    payload = SampleRecord(
                        benchmark_name=benchmark_name,
                        dataset_split=dataset_split,
                        sample_index=key.sample_index,
                        repeat_index=key.repeat_index,
                        pass_index=key.pass_index,
                        sampling_config=sampling_config,
                        stages=[cot_stage, final_stage],
                    ).as_payload()
                    cot_stats = _output_stats(cot_output)
                    final_stats = _output_stats(output)
                    payload["stats"] = {
                        "truncated": bool(final_stats["truncated"]),
                        "stop_detail": final_stats["stop_detail"],
                        "generated_token_count": _sum_known_token_counts(cot_stats, final_stats),
                        "stage1": cot_stats,
                        "stage2": final_stats,
                    }
                    trace = long_doc_traces[local_idx]
                    if trace is not None:
                        payload["long_doc"] = trace
                    strategy_a_output = strategy_a_by_idx.get(local_idx)
                    if strategy_a_output is not None and strategy_a_prompts:
                        payload["strategy_a_prompt"] = strategy_a_prompts[local_idx]
                        payload["strategy_a_completion"] = _clip_user_sentinel(
                            strategy_a_output.text,
                            prompt=strategy_a_prompts[local_idx],
                        )
                        payload["strategy_a_stop_reason"] = strategy_a_output.finish_reason
                        payload["stats"]["strategy_a"] = _output_stats(strategy_a_output)
                    payload["_stage"] = "answer"
                    if on_record is not None:
                        on_record(payload)
                    payloads.append(payload)

                _ = self.backend.generate(
                    final_prompts,
                    sampling=final_sampling,
                    batch_size=min(batch_size, len(final_prompts)),
                    progress_desc="Generating final answers",
                    on_complete=_on_final_complete,
                    prompt_stop_suffixes=_prompt_stop_suffixes(final_prompts),
                    prompt_seeds=[
                        sample_repeat_seed(
                            chunk[local_idx][0].sample_index,
                            chunk[local_idx][0].repeat_index,
                            pass_index=chunk[local_idx][0].pass_index,
                            stage=3 if strategy_a_sampling is not None else 2,
                        )
                        for local_idx in final_prompt_indices
                    ],
                )
                continue

            def _on_complete(output: GenerationOutput) -> None:
                local_idx = output.prompt_index
                if local_idx < 0 or local_idx >= len(chunk):
                    return
                key, _record = chunk[local_idx]
                clipped_text = _clip_user_sentinel(output.text, prompt=prompts[local_idx])
                stats = _output_stats(output)
                stage = StageRecord(
                    prompt=prompts[local_idx],
                    completion=clipped_text,
                    stop_reason=output.finish_reason,
                )
                payload = SampleRecord(
                    benchmark_name=benchmark_name,
                    dataset_split=dataset_split,
                    sample_index=key.sample_index,
                    repeat_index=key.repeat_index,
                    pass_index=key.pass_index,
                    sampling_config=sampling_config,
                    stages=[stage],
                ).as_payload()
                payload["stats"] = stats
                trace = long_doc_traces[local_idx]
                if trace is not None:
                    payload["long_doc"] = trace
                payload["_stage"] = "answer"
                if on_record is not None:
                    on_record(payload)
                payloads.append(payload)

            _ = self.backend.generate(
                prompts,
                sampling=generation_sampling,
                batch_size=min(batch_size, len(prompts)),
                progress_desc="Generating full responses",
                on_complete=_on_complete,
                prompt_stop_suffixes=_prompt_stop_suffixes(prompts),
                prompt_seeds=[
                    sample_repeat_seed(
                        key.sample_index,
                        key.repeat_index,
                        pass_index=key.pass_index,
                        stage=1,
                    )
                    for key, _record in chunk
                ],
            )
        return FreeResponsePipelineResult(dataset_name, len(expanded), problem_count, payloads)

    def _load_records(
        self,
        dataset_path: str,
        sample_limit: int | None,
        *,
        record_indices: Sequence[int] | None = None,
    ) -> tuple[list[tuple[int, FreeAnswerRecord]], str]:
        loader = JsonlFreeAnswerLoader(dataset_path)
        records = list(loader)
        if record_indices is not None:
            indexed_records = [(int(index), records[int(index)]) for index in record_indices]
        else:
            indexed_records = list(enumerate(records))
            if sample_limit is not None and sample_limit > 0:
                indexed_records = indexed_records[: min(sample_limit, len(indexed_records))]
        return indexed_records, infer_dataset_slug_from_path(dataset_path)


__all__ = [
    "DEFAULT_COT_PROMPT",
    "DEFAULT_DIRECT_PROMPT",
    "DEFAULT_FINAL_PROMPT",
    "FREE_RESPONSE_STOP_TOKENS",
    "G1H_GENERATION_STOP_SUFFIXES",
    "LEGACY_GENERATION_STOP_SUFFIXES",
    "FreeResponsePipeline",
    "FreeResponsePipelineResult",
    "USER_SENTINEL",
]
