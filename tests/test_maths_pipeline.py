from __future__ import annotations

import json

from src.eval.tasks.maths.pipeline import (
    DEFAULT_DIRECT_PROMPT,
    FINAL_BOXED_STOP_SUFFIXES,
    FREE_RESPONSE_STOP_TOKENS,
    G1H_GENERATION_STOP_SUFFIXES,
    G1H_REMOTE_STOP_SUFFIXES,
    LEGACY_GENERATION_STOP_SUFFIXES,
    FreeResponsePipeline,
    _output_stats,
    _prompt_stop_suffixes,
)
from src.eval.metrics import free_response as fr
from src.eval.long_doc_evidence import LongDocEvidenceConfig
from src.infer.sampling import GenerationOutput, SamplingConfig


def test_naive_direct_math_prompt_prefills_empty_think() -> None:
    assert DEFAULT_DIRECT_PROMPT.endswith("Assistant: <think></think>")


def test_math_scoring_strips_generated_empty_think_closer() -> None:
    payload = {
        "prompt1": "User: 2+2?\n\nAssistant: <think></think",
        "completion1": ">\nTherefore, the answer is \\(\\boxed{4}\\).",
    }
    assert fr._stage_text(payload, 1).startswith("Therefore")


def test_output_stats_does_not_claim_remote_empty_token_ids_are_zero() -> None:
    stats = _output_stats(
        GenerationOutput(
            prompt_index=0,
            prompt="User✿Q✿\nBot✿<think>",
            token_ids=[],
            text="a long remote completion",
            finish_reason="max_tokens",
        )
    )

    assert stats["truncated"] is True
    assert stats["stop_reason_class"] == 2
    assert stats["generated_token_count"] is None


def test_final_boxed_prompt_uses_text_stops_that_survive_token_merges() -> None:
    legacy_final = "work\nTherefore, the answer is \\(\\boxed{"
    # Build the prompt from the production marker constants so this test also
    # remains stable in checkouts containing the historical mojibake marker.
    production_g1h_final = (
        f"{G1H_REMOTE_STOP_SUFFIXES[2]}Q\n{G1H_REMOTE_STOP_SUFFIXES[4]}"
        "<think>work</think>\nTherefore, the answer is \\(\\boxed{"
    )
    g1h_final = "Userâœ¿Qâœ¿\nBotâœ¿<think>work</think>\nTherefore, the answer is \\(\\boxed{"

    assert _prompt_stop_suffixes([legacy_final]) == [
        (*LEGACY_GENERATION_STOP_SUFFIXES, *FINAL_BOXED_STOP_SUFFIXES)
    ]
    assert _prompt_stop_suffixes([g1h_final]) == [
        (*LEGACY_GENERATION_STOP_SUFFIXES, *FINAL_BOXED_STOP_SUFFIXES)
    ]
    assert _prompt_stop_suffixes([production_g1h_final]) == [
        (*G1H_REMOTE_STOP_SUFFIXES, *FINAL_BOXED_STOP_SUFFIXES)
    ]
    assert "}\\)" in FINAL_BOXED_STOP_SUFFIXES
    assert "}" not in FINAL_BOXED_STOP_SUFFIXES


def _patch_math_verify(monkeypatch) -> None:
    def parse(text: str):
        import re

        boxes = re.findall(r"\\boxed\{([^{}]+)\}", text)
        if boxes:
            return [("boxed", boxes[-1])]
        return []

    def verify(gold, pred, *, strict: bool = False):
        _ = strict
        return bool(gold and pred and gold[-1][-1] == pred[-1][-1])

    monkeypatch.setattr(fr, "_load_math_verify", lambda: (parse, verify))


def test_free_response_pipeline_generates_single_full_response_stage(tmp_path) -> None:
    dataset = tmp_path / "math.jsonl"
    dataset.write_text('{"question":"2+5?","answer":"7"}\n', encoding="utf-8")
    backend = _FakeBackend()
    pipeline = FreeResponsePipeline(backend)

    result = pipeline.run(
        dataset_path=str(dataset),
        prompt_template="User: solve\n<Q>\n\nAssistant: <think",
        generation_sampling=SamplingConfig(max_generate_tokens=32, stop_tokens=(0, 261)),
        batch_size=4,
        dataset_name="math_test",
        pass_k=(1,),
        samples_per_task=1,
    )

    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert call["sampling"].stop_tokens == FREE_RESPONSE_STOP_TOKENS
    assert call["prompt_stop_suffixes"] == [LEGACY_GENERATION_STOP_SUFFIXES]
    assert call["prompts"] == ["User: solve\n2+5?\n\nAssistant: <think"]
    assert result.payloads == [
        {
            "benchmark_name": "math",
            "dataset_split": "test",
            "sample_index": 0,
            "repeat_index": 0,
            "pass_index": 0,
            "sampling_config": {
                "stage1": {
                    "max_new_tokens": 32,
                    "temperature": 0.3,
                    "top_k": 50,
                    "top_p": 0.3,
                    "presence_penalty": 0.5,
                    "repetition_penalty": 0.5,
                    "penalty_decay": 0.99,
                    "stop_tokens": [0],
                    "ban_tokens": None,
                    "pad_zero": True,
                    "no_penalty_token_ids": [33, 10, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58],
                }
            },
            "prompt1": "User: solve\n2+5?\n\nAssistant: <think",
            "completion1": "</think>\nTherefore, the final answer is \\(\\boxed{7}\\).",
            "stop_reason1": "stop_token",
            "stats": {
                "truncated": False,
                "stop_detail": "stop_token",
                "stop_reason_class": 1,
                "termination_reason": "stop",
                "generated_token_count": 1,
            },
            "_stage": "answer",
        }
    ]


def test_free_response_pipeline_uses_g1h_remote_stops_only_for_g1h_prompt(tmp_path) -> None:
    dataset = tmp_path / "math.jsonl"
    dataset.write_text('{"question":"2+5?","answer":"7"}\n', encoding="utf-8")
    backend = _FakeBackend()
    pipeline = FreeResponsePipeline(backend)

    result = pipeline.run(
        dataset_path=str(dataset),
        prompt_template="User✿<Q>✿\nBot✿<think></think>",
        generation_sampling=SamplingConfig(max_generate_tokens=32, stop_tokens=(0, 261)),
        batch_size=4,
        dataset_name="math_test",
        pass_k=(1,),
        samples_per_task=1,
    )

    assert backend.calls[0]["prompt_stop_suffixes"] == [G1H_REMOTE_STOP_SUFFIXES]
    assert "✿" in G1H_GENERATION_STOP_SUFFIXES
    assert "✿" not in G1H_REMOTE_STOP_SUFFIXES
    assert result.payloads[0]["prompt1"] == "User✿2+5?✿\nBot✿<think></think>"
    assert result.payloads[0]["completion1"] == "</think>\nTherefore, the final answer is \\(\\boxed{7}\\)."


def test_free_response_pipeline_clamps_rendered_prompt_chars(tmp_path) -> None:
    dataset = tmp_path / "math.jsonl"
    dataset.write_text(
        '{"question":"prefix ' + ("context " * 200) + ' final question?","answer":"7"}\n',
        encoding="utf-8",
    )
    backend = _FakeBackend()
    pipeline = FreeResponsePipeline(backend)

    pipeline.run(
        dataset_path=str(dataset),
        prompt_template="User: solve\n<Q>\n\nAssistant: <think",
        generation_sampling=SamplingConfig(max_generate_tokens=32),
        batch_size=4,
        dataset_name="math_test",
        pass_k=(1,),
        samples_per_task=1,
        prompt_max_chars=220,
    )

    prompt = backend.calls[0]["prompts"][0]
    assert len(prompt) <= 220
    assert "prefix context" in prompt
    assert "[...truncated...]" in prompt
    assert "final question?" in prompt
    assert prompt.endswith("Assistant: <think")


def test_free_response_pipeline_uses_context_only_in_generation_prompt(tmp_path) -> None:
    dataset = tmp_path / "math.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "question": "Which entry answers case77?",
                "answer": "blue",
                "context": "\n".join(
                    [f"noise row {idx}" for idx in range(20)]
                    + ["case77 answer blue supporting evidence"]
                    + [f"tail row {idx}" for idx in range(20)]
                ),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    backend = _FakeBackend()
    pipeline = FreeResponsePipeline(backend)

    result = pipeline.run(
        dataset_path=str(dataset),
        prompt_template="User: solve\n<Q>\n\nAssistant: <think",
        generation_sampling=SamplingConfig(max_generate_tokens=32),
        batch_size=4,
        dataset_name="math_test",
        pass_k=(1,),
        samples_per_task=1,
        long_doc_config=LongDocEvidenceConfig(
            enabled=True,
            max_chunk_chars=120,
            overlap_lines=0,
            min_long_text_chars=100,
            max_evidence_chunks=1,
            max_evidence_chars=180,
        ),
    )

    prompt = backend.calls[0]["prompts"][0]
    assert "Context:" in prompt
    assert "Long document compacted" in prompt
    assert "case77 answer blue supporting evidence" in prompt
    assert result.payloads[0]["long_doc"]["compacted"] is True
    assert result.payloads[0]["long_doc"]["selected_chunk_ids"]


def test_free_response_pipeline_generates_cot_then_final_answer(tmp_path) -> None:
    dataset = tmp_path / "math.jsonl"
    dataset.write_text('{"question":"2+5?","answer":"7"}\n', encoding="utf-8")
    backend = _TwoStageFakeBackend()
    pipeline = FreeResponsePipeline(backend)

    result = pipeline.run(
        dataset_path=str(dataset),
        prompt_template="User: solve\n<Q>\n\nAssistant: <think",
        generation_sampling=SamplingConfig(max_generate_tokens=32, stop_tokens=(0, 261)),
        final_answer_template="<Q><COT>\nTherefore, the answer is \\(\\boxed{",
        final_sampling=SamplingConfig(max_generate_tokens=8, temperature=1.0, top_p=0.3, stop_tokens=(0, 2402)),
        batch_size=4,
        dataset_name="math_test",
        pass_k=(1,),
        samples_per_task=1,
    )

    assert len(backend.calls) == 2
    assert backend.calls[0]["sampling"].stop_tokens == FREE_RESPONSE_STOP_TOKENS
    assert backend.calls[1]["sampling"].stop_tokens == (0, 2402)
    assert backend.calls[1]["prompt_stop_suffixes"] == [
        (*LEGACY_GENERATION_STOP_SUFFIXES, *FINAL_BOXED_STOP_SUFFIXES)
    ]
    assert backend.calls[1]["prompts"] == [
        "User: solve\n2+5?\n\nAssistant: <think</think>\nwork\nTherefore, the answer is \\(\\boxed{"
    ]
    assert result.payloads[0]["prompt1"] == "User: solve\n2+5?\n\nAssistant: <think"
    assert result.payloads[0]["completion1"] == "</think>\nwork"
    assert result.payloads[0]["prompt2"] == "\nTherefore, the answer is \\(\\boxed{"
    assert result.payloads[0]["completion2"] == "7}\\)."
    assert result.payloads[0]["sampling_config"]["stage2"]["max_new_tokens"] == 8
    assert result.payloads[0]["stats"]["stage1"]["generated_token_count"] == 1
    assert result.payloads[0]["stats"]["stage2"]["generated_token_count"] == 2
    assert result.payloads[0]["_stage"] == "answer"


def test_free_response_pipeline_emits_strategy_a_full_and_two_stage_bc(tmp_path) -> None:
    dataset = tmp_path / "math.jsonl"
    dataset.write_text('{"question":"2+5?","answer":"7"}\n', encoding="utf-8")
    backend = _StrategyAFakeBackend()
    pipeline = FreeResponsePipeline(backend)

    result = pipeline.run(
        dataset_path=str(dataset),
        prompt_template="User: solve\n<Q>\n\nAssistant: <think",
        generation_sampling=SamplingConfig(max_generate_tokens=32, stop_tokens=(0, 261)),
        strategy_a_prompt_template="User: solve fully\n<Q>\n\nAssistant: <think",
        strategy_a_sampling=SamplingConfig(max_generate_tokens=64, temperature=0.25),
        final_answer_template="<Q><COT>\nTherefore, the answer is \\(\\boxed{",
        final_sampling=SamplingConfig(max_generate_tokens=8, temperature=1.0, top_p=0.3, stop_tokens=(0, 2402)),
        batch_size=4,
        dataset_name="math_test",
        pass_k=(1,),
        samples_per_task=1,
    )

    assert len(backend.calls) == 3
    assert backend.calls[0]["prompts"] == ["User: solve fully\n2+5?\n\nAssistant: <think"]
    assert backend.calls[1]["prompts"] == ["User: solve\n2+5?\n\nAssistant: <think"]
    assert backend.calls[2]["prompts"] == [
        "User: solve\n2+5?\n\nAssistant: <think</think>\nwork\nTherefore, the answer is \\(\\boxed{"
    ]
    payload = result.payloads[0]
    assert payload["strategy_a_prompt"] == "User: solve fully\n2+5?\n\nAssistant: <think"
    assert payload["strategy_a_completion"] == "</think>\nfull answer \\(\\boxed{7}\\)."
    assert payload["completion1"] == "</think>\nwork"
    assert payload["completion2"] == "8}\\)."
    assert payload["sampling_config"]["strategy_a"]["max_new_tokens"] == 64
    assert payload["sampling_config"]["stage1"]["max_new_tokens"] == 32
    assert payload["sampling_config"]["stage2"]["max_new_tokens"] == 8
    assert payload["stats"]["strategy_a"]["generated_token_count"] == 4
    assert payload["stats"]["stage1"]["generated_token_count"] == 1
    assert payload["stats"]["stage2"]["generated_token_count"] == 2


def test_strategy_a_filter_only_runs_two_stage_for_a_failures(monkeypatch, tmp_path) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "math.jsonl"
    dataset.write_text(
        '{"question":"2+5?","answer":"7"}\n{"question":"4+5?","answer":"9"}\n',
        encoding="utf-8",
    )
    backend = _MixedStrategyAFakeBackend()
    pipeline = FreeResponsePipeline(backend)

    result = pipeline.run(
        dataset_path=str(dataset),
        prompt_template="User: solve\n<Q>\n\nAssistant: <think",
        generation_sampling=SamplingConfig(max_generate_tokens=32, stop_tokens=(0, 261)),
        strategy_a_prompt_template="User: solve fully\n<Q>\n\nAssistant: <think",
        strategy_a_sampling=SamplingConfig(max_generate_tokens=64, temperature=0.25),
        strategy_a_filter_correct=True,
        final_answer_template="<Q><COT>\nTherefore, the answer is \\(\\boxed{",
        final_sampling=SamplingConfig(max_generate_tokens=8, temperature=1.0, top_p=0.3, stop_tokens=(0, 2402)),
        batch_size=4,
        dataset_name="math_test",
        pass_k=(1,),
        samples_per_task=1,
    )

    assert len(backend.calls) == 3
    assert backend.calls[0]["prompts"] == [
        "User: solve fully\n2+5?\n\nAssistant: <think",
        "User: solve fully\n4+5?\n\nAssistant: <think",
    ]
    assert backend.calls[1]["prompts"] == ["User: solve\n4+5?\n\nAssistant: <think"]
    assert backend.calls[2]["prompts"] == [
        "User: solve\n4+5?\n\nAssistant: <think</think>\nwork\nTherefore, the answer is \\(\\boxed{"
    ]

    by_sample = {payload["sample_index"]: payload for payload in result.payloads}
    assert set(by_sample) == {0, 1}
    assert by_sample[0]["strategy_a_completion"] == "</think>\nfull answer \\(\\boxed{7}\\)."
    assert "completion1" not in by_sample[0]
    assert by_sample[1]["strategy_a_completion"] == "</think>\nfull answer \\(\\boxed{0}\\)."
    assert by_sample[1]["completion1"] == "</think>\nwork"
    assert by_sample[1]["completion2"] == "9}\\)."


class _FakeBackend:
    model_name = "fake"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, prompts, **kwargs):
        self.calls.append({"prompts": list(prompts), **kwargs})
        outputs = [
            GenerationOutput(
                prompt_index=idx,
                prompt=prompt,
                token_ids=[1],
                text=(
                    "</think>\nTherefore, the final answer is \\(\\boxed{7}\\).✿clean_length \\(\\boxed{0}\\)"
                    if "User✿" in prompt or "Bot✿" in prompt
                    else "</think>\nTherefore, the final answer is \\(\\boxed{7}\\).\nUser: next"
                ),
                finish_reason="stop_token",
            )
            for idx, prompt in enumerate(prompts)
        ]
        on_complete = kwargs.get("on_complete")
        if on_complete is not None:
            for output in outputs:
                on_complete(output)
        return outputs

    def score_choice_tokens(self, *, prompt: str, choice_token_texts):
        raise NotImplementedError


class _TwoStageFakeBackend:
    model_name = "fake"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, prompts, **kwargs):
        self.calls.append({"prompts": list(prompts), **kwargs})
        is_final = "Therefore, the answer is" in prompts[0]
        outputs = [
            GenerationOutput(
                prompt_index=idx,
                prompt=prompt,
                token_ids=[2, 3] if is_final else [1],
                text="7}\\).\nUser: next" if is_final else "</think>\nwork",
                finish_reason="stop_token",
            )
            for idx, prompt in enumerate(prompts)
        ]
        on_complete = kwargs.get("on_complete")
        if on_complete is not None:
            for output in outputs:
                on_complete(output)
        return outputs

    def score_choice_tokens(self, *, prompt: str, choice_token_texts):
        raise NotImplementedError


class _StrategyAFakeBackend:
    model_name = "fake"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, prompts, **kwargs):
        self.calls.append({"prompts": list(prompts), **kwargs})
        call_idx = len(self.calls)
        if call_idx == 1:
            text = "</think>\nfull answer \\(\\boxed{7}\\).\nUser: next"
            token_ids = [4, 5, 6, 7]
        elif call_idx == 2:
            text = "</think>\nwork"
            token_ids = [1]
        else:
            text = "8}\\).\nUser: next"
            token_ids = [2, 3]
        outputs = [
            GenerationOutput(
                prompt_index=idx,
                prompt=prompt,
                token_ids=token_ids,
                text=text,
                finish_reason="stop_token",
            )
            for idx, prompt in enumerate(prompts)
        ]
        on_complete = kwargs.get("on_complete")
        if on_complete is not None:
            for output in outputs:
                on_complete(output)
        return outputs

    def score_choice_tokens(self, *, prompt: str, choice_token_texts):
        raise NotImplementedError


class _MixedStrategyAFakeBackend:
    model_name = "fake"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, prompts, **kwargs):
        self.calls.append({"prompts": list(prompts), **kwargs})
        call_idx = len(self.calls)
        outputs = []
        for idx, prompt in enumerate(prompts):
            if call_idx == 1:
                answer = "7" if idx == 0 else "0"
                text = f"</think>\nfull answer \\(\\boxed{{{answer}}}\\).\nUser: next"
                token_ids = [4, 5, 6, 7]
            elif call_idx == 2:
                text = "</think>\nwork"
                token_ids = [1]
            else:
                text = "9}\\).\nUser: next"
                token_ids = [2, 3]
            outputs.append(
                GenerationOutput(
                    prompt_index=idx,
                    prompt=prompt,
                    token_ids=token_ids,
                    text=text,
                    finish_reason="stop_token",
                )
            )
        on_complete = kwargs.get("on_complete")
        if on_complete is not None:
            for output in outputs:
                on_complete(output)
        return outputs

    def score_choice_tokens(self, *, prompt: str, choice_token_texts):
        raise NotImplementedError
