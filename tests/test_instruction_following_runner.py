from __future__ import annotations

import zipfile

import pytest

from src.eval.tasks.instruction_following import runner as instruction_following_runner
from src.eval.metrics.instruction_following.metrics import evaluate_instruction_following
from src.eval.metrics.instruction_following.ifbench_official import instructions_util as ifbench_util


def test_instruction_following_runner_parser_accepts_core_flags() -> None:
    args = instruction_following_runner.parse_args(
        [
            "--dataset",
            "dataset.jsonl",
            "--enable-think",
        ]
    )
    assert args.enable_think is True


def test_instruction_following_runner_rejects_data_only_benchmarks() -> None:
    with pytest.raises(ValueError, match="does not have a rule-based instruction-following scorer"):
        instruction_following_runner._ensure_rule_based_dataset("flores200_devtest")


def test_ifeval_runner_skips_optional_checker() -> None:
    assert instruction_following_runner._should_run_checker("ifeval_test") is False
    assert instruction_following_runner._should_run_checker("ifbench_test") is False
    assert instruction_following_runner._should_run_checker("instruction_following_custom_test") is True


def test_ifbench_uses_official_rule_registry(tmp_path) -> None:
    dataset = tmp_path / "ifbench" / "test.jsonl"
    dataset.parent.mkdir()
    dataset.write_text(
        '{"key": 0, "prompt": "Answer without whitespace.", '
        '"instruction_id_list": ["format:no_whitespace"], '
        '"kwargs": [{"N": null}]}\n'
    )

    metrics = evaluate_instruction_following(
        [{"sample_index": 0, "repeat_index": 0, "completion1": "NoSpaces"}],
        dataset_path=dataset,
        dataset_slug="ifbench_test",
        strict=False,
    )

    assert metrics.samples == 1
    assert metrics.prompt_accuracy == 1.0
    assert metrics.instruction_accuracy == 1.0
    assert metrics.tier1_accuracy["format:no_whitespace"] == 1.0


def test_ifbench_start_verb_bad_nltk_tagger_does_not_abort(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "ifbench" / "test.jsonl"
    dataset.parent.mkdir()
    dataset.write_text(
        '{"key": 0, "prompt": "Start with a verb.", '
        '"instruction_id_list": ["words:start_verb"], '
        '"kwargs": [{"N": null}]}\n'
    )

    monkeypatch.setattr(ifbench_util, "_ensure_nltk_resource", lambda *_args, **_kwargs: True)

    def _bad_pos_tag(_words):
        raise zipfile.BadZipFile("bad tagger")

    monkeypatch.setattr(ifbench_util.nltk, "pos_tag", _bad_pos_tag)

    metrics = evaluate_instruction_following(
        [{"sample_index": 0, "repeat_index": 0, "completion1": "Run quickly."}],
        dataset_path=dataset,
        dataset_slug="ifbench_test",
        strict=False,
    )

    assert metrics.samples == 1
    assert metrics.prompt_accuracy == 0.0
    assert metrics.instruction_accuracy == 0.0


def test_ifbench_nltk_resource_check_does_not_download_by_default(monkeypatch) -> None:
    ifbench_util._ensure_nltk_resource.cache_clear()
    monkeypatch.delenv("RWKV_IFBENCH_ALLOW_NLTK_DOWNLOAD", raising=False)

    def _missing_resource(_resource_path):
        raise LookupError("missing")

    def _unexpected_download(*_args, **_kwargs):
        raise AssertionError("nltk.download should not be called by default")

    monkeypatch.setattr(ifbench_util.nltk.data, "find", _missing_resource)
    monkeypatch.setattr(ifbench_util.nltk, "download", _unexpected_download)

    assert ifbench_util._ensure_nltk_resource("tokenizers/punkt", "punkt") is False
    ifbench_util._ensure_nltk_resource.cache_clear()


def test_ifeval_strips_complete_think_block_before_scoring(tmp_path) -> None:
    dataset = tmp_path / "ifeval" / "test.jsonl"
    dataset.parent.mkdir()
    dataset.write_text(
        '{"key": 0, "prompt": "Mention alpha.", '
        '"instruction_id_list": ["keywords:existence"], '
        '"kwargs": [{"keywords": ["alpha"]}]}\n'
    )

    metrics = evaluate_instruction_following(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "User: Mention alpha.\n\nAssistant:",
                "completion1": "<think>planning only</think>\nalpha",
            }
        ],
        dataset_path=dataset,
        dataset_slug="ifeval_test",
    )

    assert metrics.prompt_accuracy == 1.0
    assert metrics.payloads[0]["answer"] == "alpha"


def test_ifeval_strips_generated_empty_think_closer_before_scoring(tmp_path) -> None:
    dataset = tmp_path / "ifeval" / "test.jsonl"
    dataset.parent.mkdir()
    dataset.write_text(
        '{"key": 0, "prompt": "Mention alpha.", '
        '"instruction_id_list": ["keywords:existence"], '
        '"kwargs": [{"keywords": ["alpha"]}]}\n'
    )

    metrics = evaluate_instruction_following(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "User: Mention alpha.\n\nAssistant: <think></think",
                "completion1": ">\nalpha",
            }
        ],
        dataset_path=dataset,
        dataset_slug="ifeval_test",
    )

    assert metrics.prompt_accuracy == 1.0
    assert metrics.payloads[0]["answer"] == "alpha"


def test_ifeval_unclosed_leading_think_is_not_scored_as_answer(tmp_path) -> None:
    dataset = tmp_path / "ifeval" / "test.jsonl"
    dataset.parent.mkdir()
    dataset.write_text(
        '{"key": 0, "prompt": "Mention alpha.", '
        '"instruction_id_list": ["keywords:existence"], '
        '"kwargs": [{"keywords": ["alpha"]}]}\n'
    )

    metrics = evaluate_instruction_following(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "User: Mention alpha.\n\nAssistant:",
                "completion1": "<think>alpha appears only in hidden reasoning",
            }
        ],
        dataset_path=dataset,
        dataset_slug="ifeval_test",
    )

    assert metrics.prompt_accuracy == 0.0
    assert metrics.payloads[0]["answer"] == ""
