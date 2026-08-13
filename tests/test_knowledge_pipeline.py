from __future__ import annotations

import json

import pytest

from src.eval.tasks.knowledge.pipeline import MultipleChoicePipeline
from src.eval.metrics.multi_choice import (
    evaluate_multiple_choice,
    evaluate_multiple_choice_cascade,
    extract_answer_after_think,
)
from src.eval.long_doc_evidence import LongDocEvidenceConfig
from src.infer.sampling import GenerationOutput
from src.infer.sampling import SamplingConfig


class _FallbackOnlyBackend:
    def __init__(self, *, text: str = " B") -> None:
        self.model_name = "remote-openai"
        self.text = text
        self.generate_calls: list[list[str]] = []
        self.generate_batch_sizes: list[int] = []
        self.text_stop_detectors: list[object] = []
        self.min_tokens: list[int | None] = []
        self.samplings: list[SamplingConfig] = []
        self.resolved_token_texts: list[tuple[str, ...]] = []

    def resolve_single_token_ids(self, token_texts):
        texts = tuple(str(text) for text in token_texts)
        self.resolved_token_texts.append(texts)
        return {text: 300 + ord(text[-1]) - ord("A") for text in texts}

    def generate(
        self,
        prompts,
        *,
        sampling,
        batch_size,
        progress_desc="Generating",
        probe_only=False,
        on_complete=None,
        prompt_seeds=None,
        text_stop_detectors=None,
        prefill_chunk_size=16,
        show_progress=True,
        min_tokens=None,
    ):
        self.generate_calls.append(list(prompts))
        self.generate_batch_sizes.append(int(batch_size))
        self.text_stop_detectors.append(text_stop_detectors)
        self.min_tokens.append(min_tokens)
        self.samplings.append(sampling)
        outputs = [
            GenerationOutput(
                prompt_index=index,
                prompt=str(prompt),
                token_ids=[],
                text=self.text,
                finish_reason="stop_token",
            )
            for index, prompt in enumerate(prompts)
        ]
        if on_complete is not None and not probe_only:
            for output in outputs:
                on_complete(output)
        return outputs

class _ScriptedBackend(_FallbackOnlyBackend):
    def __init__(self, texts: list[str]) -> None:
        super().__init__()
        self.texts = list(texts)

    def generate(self, prompts, **kwargs):
        if not self.texts:
            raise AssertionError("unexpected generation call")
        self.text = self.texts.pop(0)
        return super().generate(prompts, **kwargs)


def test_multiple_choice_pipeline_generates_choice_by_default(tmp_path) -> None:
    dataset_path = tmp_path / "mmlu_demo_test.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "2+2=?",
                "A": "3",
                "B": "4",
                "C": "5",
                "D": "6",
                "answer": "B",
                "subject": "math",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    backend = _FallbackOnlyBackend()
    pipeline = MultipleChoicePipeline(backend)

    result = pipeline.run_direct(str(dataset_path))

    assert result.sample_count == 1
    assert result.payloads[0]["completion1"] == " B"
    assert result.payloads[0]["stop_reason1"] == "generated_choice"
    assert len(backend.generate_calls) == 1
    assert backend.min_tokens == [1]
    assert backend.samplings[0].max_generate_tokens == 1
    assert backend.samplings[0].allowed_token_ids == (300, 301, 302, 303)


def test_multiple_choice_pipeline_batches_generation(tmp_path) -> None:
    dataset_path = tmp_path / "cmmlu_demo_test.jsonl"
    rows = [
        {
            "question": f"{index}+1=?",
            "A": str(index),
            "B": str(index + 1),
            "C": str(index + 2),
            "D": str(index + 3),
            "answer": "B",
            "subject": "math",
        }
        for index in range(5)
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    backend = _FallbackOnlyBackend()
    pipeline = MultipleChoicePipeline(backend)

    result = pipeline.run_direct(str(dataset_path), batch_size=3)

    assert result.sample_count == 5
    assert len(result.payloads) == 5
    assert [len(call) for call in backend.generate_calls] == [5]
    assert backend.generate_batch_sizes == [3]


def test_multiple_choice_pipeline_compacts_context_with_question_and_choices_query(tmp_path) -> None:
    dataset_path = tmp_path / "gpqa_demo_test.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "Which color is tied to catalyst77?",
                "A": "red",
                "B": "blue",
                "C": "green",
                "D": "yellow",
                "answer": "B",
                "subject": "chemistry",
                "context": "\n".join(
                    [f"noise row {idx}" for idx in range(20)]
                    + ["catalyst77 blue pathway evidence"]
                    + [f"tail row {idx}" for idx in range(20)]
                ),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    backend = _FallbackOnlyBackend()
    pipeline = MultipleChoicePipeline(backend)

    result = pipeline.run_direct(
        str(dataset_path),
        long_doc_config=LongDocEvidenceConfig(
            enabled=True,
            max_chunk_chars=120,
            overlap_lines=0,
            min_long_text_chars=100,
            max_evidence_chunks=1,
            max_evidence_chars=180,
        ),
    )

    prompt = backend.generate_calls[0][0]
    assert "Context:" in prompt
    assert "Long document compacted" in prompt
    assert "catalyst77 blue pathway evidence" in prompt
    assert result.payloads[0]["long_doc"]["query_policy"] == "question_and_choices"
    assert result.payloads[0]["long_doc"]["compacted"] is True


def test_multiple_choice_pipeline_streams_generated_payloads_in_order(tmp_path) -> None:
    dataset_path = tmp_path / "mmlu_demo_test.jsonl"
    rows = [
        {
            "question": f"{index}+1=?",
            "A": str(index),
            "B": str(index + 1),
            "C": str(index + 2),
            "D": str(index + 3),
            "answer": "B",
            "subject": "math",
        }
        for index in range(4)
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    backend = _FallbackOnlyBackend()
    pipeline = MultipleChoicePipeline(backend)
    streamed_payloads: list[dict] = []

    result = pipeline.run_direct(str(dataset_path), batch_size=4, on_record=streamed_payloads.append)

    assert result.sample_count == 4
    assert [payload["sample_index"] for payload in result.payloads] == [0, 1, 2, 3]
    assert [payload["sample_index"] for payload in streamed_payloads] == [0, 1, 2, 3]
    assert [payload["completion1"] for payload in result.payloads] == [" B"] * 4
    assert [len(call) for call in backend.generate_calls] == [4]


def test_multiple_choice_pipeline_marks_invalid_generation_wrong(tmp_path) -> None:
    dataset_path = tmp_path / "cmmlu_demo_test.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "2+2=?",
                "A": "3",
                "B": "4",
                "C": "5",
                "D": "6",
                "answer": "B",
                "subject": "math",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    backend = _FallbackOnlyBackend(text=" (1)(2)(3).\n")
    pipeline = MultipleChoicePipeline(backend)

    result = pipeline.run_direct(str(dataset_path))
    metrics = evaluate_multiple_choice(result.payloads, dataset_path=dataset_path)

    assert result.sample_count == 1
    assert result.payloads[0]["completion1"] == " "
    assert metrics.accuracy == 0.0


def test_multiple_choice_replay_prefers_raw_completion_over_stale_extraction(tmp_path) -> None:
    dataset_path = tmp_path / "arc_easy_test.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "Which choice is correct?",
                "A": "first",
                "B": "second",
                "C": "third",
                "D": "fourth",
                "answer": "C",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payload = {
        "sample_index": 0,
        "repeat_index": 0,
        "pass_index": 0,
        # This is the historical adapter bug we are repairing.
        "completion1": " A",
        "direct_raw_completion": " (C).\nThe third option is correct.",
    }

    metrics = evaluate_multiple_choice([payload], dataset_path=dataset_path)

    assert metrics.accuracy == 1.0
    assert metrics.payloads[0]["answer"] == "C"


def test_multiple_choice_replay_accepts_leading_label_with_explanation(tmp_path) -> None:
    dataset_path = tmp_path / "mmlu_test.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "Pick four.",
                "A": "0",
                "B": "4",
                "C": "2",
                "D": "6",
                "answer": "B",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payload = {
        "sample_index": 0,
        "repeat_index": 0,
        "pass_index": 0,
        "completion1": " A",
        "direct_raw_completion": " B. 4.",
    }

    metrics = evaluate_multiple_choice([payload], dataset_path=dataset_path)

    assert metrics.accuracy == 1.0
    assert metrics.payloads[0]["answer"] == "B"


def test_multiple_choice_replay_strips_generated_empty_think_closer(tmp_path) -> None:
    dataset_path = tmp_path / "mmlu_test.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "Pick four.",
                "A": "0",
                "B": "4",
                "C": "2",
                "D": "6",
                "answer": "B",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payload = {
        "sample_index": 0,
        "repeat_index": 0,
        "pass_index": 0,
        "completion1": ">\nB",
        "direct_raw_completion": ">\nB",
    }

    metrics = evaluate_multiple_choice([payload], dataset_path=dataset_path)

    assert metrics.accuracy == 1.0
    assert metrics.payloads[0]["answer"] == "B"


def test_multiple_choice_cot_generates_final_answer_by_default(tmp_path) -> None:
    dataset_path = tmp_path / "mmlu_pro_demo_test.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "2+2=?",
                "A": "3",
                "B": "4",
                "C": "5",
                "D": "6",
                "answer": "B",
                "subject": "math",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    backend = _FallbackOnlyBackend()
    pipeline = MultipleChoicePipeline(backend)

    result = pipeline.run_chain_of_thought(
        str(dataset_path),
        cot_sampling=SamplingConfig(max_generate_tokens=32),
        batch_size=1,
    )

    assert result.sample_count == 1
    assert result.payloads[0]["completion2"] == " B"
    assert result.payloads[0]["stop_reason2"] == "generated_choice"
    assert len(backend.generate_calls) == 2


def test_multiple_choice_cot_streams_generated_final_answer(tmp_path) -> None:
    dataset_path = tmp_path / "mmlu_pro_demo_test.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "2+2=?",
                "A": "3",
                "B": "4",
                "C": "5",
                "D": "6",
                "answer": "B",
                "subject": "math",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    backend = _FallbackOnlyBackend()
    pipeline = MultipleChoicePipeline(backend)
    streamed_payloads: list[dict] = []

    result = pipeline.run_chain_of_thought(
        str(dataset_path),
        cot_sampling=SamplingConfig(max_generate_tokens=32),
        batch_size=1,
        on_record=streamed_payloads.append,
    )
    metrics = evaluate_multiple_choice(result.payloads, dataset_path=dataset_path)

    assert result.sample_count == 1
    assert len(streamed_payloads) == 2
    assert streamed_payloads[0]["_stage"] == "cot"
    assert streamed_payloads[1]["_stage"] == "answer"
    assert len(result.payloads) == 1
    assert result.payloads[0]["completion2"] == " B"
    assert len(backend.generate_calls) == 2
    assert metrics.accuracy == 1.0


def test_multiple_choice_cot_can_extract_answer_from_same_completion(tmp_path) -> None:
    dataset_path = tmp_path / "gpqa_diamond_test.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "2+2=?",
                "A": "3",
                "B": "4",
                "C": "5",
                "D": "6",
                "answer": "B",
                "subject": "math",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    backend = _FallbackOnlyBackend(text=">reasoning</think>\nFinal answer: B")
    pipeline = MultipleChoicePipeline(backend)

    result = pipeline.run_chain_of_thought(
        str(dataset_path),
        cot_sampling=SamplingConfig(max_generate_tokens=32),
        batch_size=1,
        answer_strategy="cascade_a_b",
    )
    metrics = evaluate_multiple_choice_cascade(result.payloads, dataset_path=dataset_path)

    assert result.payloads[0]["strategy_a_completion"] == ">reasoning</think>\nFinal answer: B"
    assert "completion1" not in result.payloads[0]
    assert len(backend.generate_calls) == 1
    assert metrics.metrics_by_group["strategy_a"]["exact_accuracy"] == 1.0
    assert metrics.metrics_by_group["strategy_b"]["exact_accuracy"] == 1.0


def test_multiple_choice_cot_extracts_chinese_final_answer(tmp_path) -> None:
    dataset_path = tmp_path / "ceval_demo_test.jsonl"
    dataset_path.write_text(
        json.dumps({"question": "2+2=?", "A": "3", "B": "4", "answer": "B"}) + "\n",
        encoding="utf-8",
    )
    backend = _FallbackOnlyBackend(text=">推理过程</think>\n最终答案是 B。")

    result = MultipleChoicePipeline(backend).run_chain_of_thought(
        str(dataset_path),
        cot_sampling=SamplingConfig(max_generate_tokens=32),
        answer_strategy="cascade_a_b",
    )

    assert result.payloads[0]["strategy_a_completion"] == ">推理过程</think>\n最终答案是 B。"
    assert len(backend.generate_calls) == 1


def test_answer_after_think_uses_last_explicit_answer_and_handles_truncation() -> None:
    text = ">reasoning</think>\nThe correct answer is **B. 4**\ncontinued text\nFinal answer: C"

    assert extract_answer_after_think(text, 4) == "C"
    assert extract_answer_after_think("Final answer: B", 4) == "B"
    assert extract_answer_after_think("reasoning mentions A, then concludes: final answer is D", 4) == "D"


def test_cascade_cot_does_not_stop_at_an_early_answer_marker(tmp_path) -> None:
    dataset_path = tmp_path / "gpqa_diamond_test.jsonl"
    dataset_path.write_text(
        json.dumps(
            {"question": "2+2=?", "A": "3", "B": "4", "C": "5", "D": "6", "answer": "B"}
        )
        + "\n",
        encoding="utf-8",
    )
    backend = _FallbackOnlyBackend(text=">reasoning</think>\nFinal answer: B")

    MultipleChoicePipeline(backend).run_chain_of_thought(
        str(dataset_path),
        cot_sampling=SamplingConfig(max_generate_tokens=32),
        answer_strategy="cascade_a_b",
    )

    assert backend.text_stop_detectors == [None]


def test_multiple_choice_cascade_routes_only_strategy_a_failure_to_b(tmp_path) -> None:
    dataset_path = tmp_path / "gpqa_diamond_test.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "2+2=?",
                "A": "3",
                "B": "4",
                "C": "5",
                "D": "6",
                "answer": "B",
                "subject": "math",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    backend = _ScriptedBackend(
        [
            ">first attempt</think>\nFinal answer: A",
            ">fresh reasoning without a final answer",
            " B",
        ]
    )

    result = MultipleChoicePipeline(backend).run_chain_of_thought(
        str(dataset_path),
        cot_sampling=SamplingConfig(max_generate_tokens=32),
        answer_strategy="cascade_a_b",
    )
    metrics = evaluate_multiple_choice_cascade(result.payloads, dataset_path=dataset_path)

    assert len(backend.generate_calls) == 3
    assert (
        ">fresh reasoning without a final answer\n</think>\nTherefore, the answer is"
        in backend.generate_calls[2][0]
    )
    assert metrics.metrics_by_group["strategy_a"]["exact_accuracy"] == 0.0
    assert metrics.metrics_by_group["strategy_b"]["exact_accuracy"] == 1.0
    assert metrics.metrics_by_group["strategy_b"]["rescued"] == 1.0
    assert result.payloads[0]["format_bridges"] == {
        "strategy_b_final_raw_completion": " B",
        "strategy_b_final_raw_stop_reason": "stop_token",
        "strategy_b_final_extracted_letter": "B",
    }


def test_cascade_wrong_strategy_b_does_not_inherit_missing_prediction_credit(
    tmp_path,
) -> None:
    dataset_path = tmp_path / "gpqa_diamond_test.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "2+2=?",
                "A": "3",
                "B": "4",
                "C": "5",
                "D": "6",
                "answer": "B",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payload = {
        "sample_index": 0,
        "repeat_index": 0,
        "pass_index": 0,
        "strategy_a_completion": ">reasoning without a final answer",
        "completion1": ">second attempt reasoning",
        "completion2": " A",
    }

    metrics = evaluate_multiple_choice_cascade(
        [payload],
        dataset_path=dataset_path,
        missing_prediction_score=0.25,
    )

    assert metrics.metrics_by_group["strategy_a"]["score"] == 0.25
    assert metrics.metrics_by_group["strategy_b"]["valid"] == 1.0
    assert metrics.metrics_by_group["strategy_b"]["exact_accuracy"] == 0.0
    assert metrics.metrics_by_group["strategy_b"]["score"] == 0.0


def test_generated_choice_accepts_exact_option_text() -> None:
    pipeline = MultipleChoicePipeline(_ScriptedBackend([]))

    assert pipeline._extract_generated_choice_letter(
        " Spironolactone",
        ["Furosemide", "Spironolactone", "Digoxin", "Aspirin"],
    ) == "B"


def test_generated_choice_accepts_unique_option_text_in_answer_sentence() -> None:
    pipeline = MultipleChoicePipeline(_ScriptedBackend([]))

    assert pipeline._extract_generated_choice_letter(
        "The correct answer is the Pre-Botzinger complex.",
        ["Pons", "Pre-Botzinger complex", "Medulla", "Cerebellum"],
    ) == "B"


def test_generated_choice_accepts_compact_chinese_answer_markers() -> None:
    pipeline = MultipleChoicePipeline(_ScriptedBackend([]))
    choices = ["甲", "乙", "丙", "丁"]

    assert pipeline._extract_generated_choice_letter("选项C。", choices) == "C"
    assert pipeline._extract_generated_choice_letter("答案为D", choices) == "D"


def test_generated_choice_keeps_letter_and_rejects_ambiguous_option_text() -> None:
    pipeline = MultipleChoicePipeline(_ScriptedBackend([]))
    choices = ["alpha", "alpha beta", "gamma", "delta"]

    assert pipeline._extract_generated_choice_letter(" D", choices) == "D"
    assert pipeline._extract_generated_choice_letter("The answer is alpha beta.", choices) == ""
    assert pipeline._extract_generated_choice_letter("unrelated prose", choices) == ""


def test_generated_choice_does_not_take_last_letter_from_prose() -> None:
    pipeline = MultipleChoicePipeline(_ScriptedBackend([]))
    choices = ["alpha", "beta", "gamma", "delta"]

    assert pipeline._extract_generated_choice_letter("(C) explanation, then Option A appears", choices) == ""


def test_direct_generation_groups_mixed_choice_counts_for_constraints(tmp_path) -> None:
    dataset_path = tmp_path / "mixed_choices_test.jsonl"
    dataset_path.write_text(
        "\n".join(
            (
                json.dumps({"question": "q1", "A": "a", "B": "b", "answer": "A"}),
                json.dumps(
                    {
                        "question": "q2",
                        "A": "a",
                        "B": "b",
                        "C": "c",
                        "D": "d",
                        "answer": "B",
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    backend = _FallbackOnlyBackend(text=" A")

    result = MultipleChoicePipeline(backend).run_direct(str(dataset_path), batch_size=8)

    assert result.sample_count == 2
    assert [sampling.allowed_token_ids for sampling in backend.samplings] == [
        (300, 301),
        (300, 301, 302, 303),
    ]


def test_choice_sampling_protocol_persists_exact_tokenizer_mapping() -> None:
    backend = _FallbackOnlyBackend()
    protocol = MultipleChoicePipeline(backend).resolve_choice_sampling_protocol(
        [4, 2, 4]
    )

    assert backend.resolved_token_texts == [(" A", " B"), (" A", " B", " C", " D")]
    assert protocol["schema_version"] == "rwkv.knowledge-direct-sampling.v1"
    assert protocol["tokenizer_identity"]["model"] == "remote-openai"
    assert protocol["tokenizer_identity"]["token_text_to_id"] == {
        " A": 300,
        " B": 301,
        " C": 302,
        " D": 303,
    }
    four_choice = protocol["by_choice_count"]["4"]
    assert four_choice["letter_to_token_id"] == {
        "A": 300,
        "B": 301,
        "C": 302,
        "D": 303,
    }
    assert four_choice["allowed_token_ids"] == [300, 301, 302, 303]
    assert four_choice["sampling"]["max_new_tokens"] == 1
    assert four_choice["sampling"]["temperature"] == 1.0
    assert four_choice["sampling"]["top_k"] == 1


def test_choice_sampling_protocol_rejects_duplicate_or_drifting_token_ids() -> None:
    class DuplicateBackend(_FallbackOnlyBackend):
        def resolve_single_token_ids(self, token_texts):
            return {str(text): 300 for text in token_texts}

    with pytest.raises(RuntimeError, match="distinct tokens"):
        MultipleChoicePipeline(DuplicateBackend()).resolve_choice_sampling_protocol([4])

    class DriftingBackend(_FallbackOnlyBackend):
        def resolve_single_token_ids(self, token_texts):
            texts = tuple(str(text) for text in token_texts)
            offset = 100 if len(texts) > 2 else 0
            return {
                text: 300 + offset + ord(text[-1]) - ord("A") for text in texts
            }

    with pytest.raises(RuntimeError, match="changed across choice counts"):
        MultipleChoicePipeline(DriftingBackend()).resolve_choice_sampling_protocol([2, 4])
