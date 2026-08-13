from __future__ import annotations

import os
from pathlib import Path

from src.eval.benchmark_config import (
    config_path_for_benchmark,
    resolve_benchmark_model_config,
    resolve_sampling_config,
)
from src.eval.results.schema import sampling_config_to_dict
from src.infer.sampling import SamplingConfig
from src.eval import benchmark_config


def test_config_cache_is_content_addressed_not_mtime_addressed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "demo.toml"
    path.write_text("[default]\ntarget_samples = 10\n")
    benchmark_config._CONFIG_CACHE.clear()  # type: ignore[attr-defined]
    first = benchmark_config._load_toml(path)  # type: ignore[attr-defined]
    original = path.stat()

    # Keep the exact old mtime while changing the semantics. An mtime cache
    # would silently reuse the first table and bind the wrong protocol.
    path.write_text("[default]\ntarget_samples = 20\n")
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    second = benchmark_config._load_toml(path)  # type: ignore[attr-defined]

    assert first["default"]["target_samples"] == 10
    assert second["default"]["target_samples"] == 20


def test_config_path_for_benchmark_strips_split_suffix() -> None:
    path = config_path_for_benchmark("human_eval_plus_test")
    assert path.name == "human_eval_plus.toml"


def test_resolve_math_500_cot_config_merges_default_and_template() -> None:
    config = resolve_benchmark_model_config("math_500_test", "rwkv7-g1a-2.9b", stage="cot")

    assert config is not None
    assert config.pass_k == ()
    assert config.avg_k == (8,)
    assert config.report_pass_k == ()
    assert config.report_avg_k == (8,)
    assert config.sampling_overrides["max_generate_tokens"] == 4096
    assert config.sampling_overrides["top_k"] == 40
    assert config.sampling_overrides["temperature"] == 0.8
    assert config.sampling_overrides["stop_tokens"] == (0,)


def test_g1g_g1h_and_g1i_expand_legacy_4k_generation_budget() -> None:
    g1g = resolve_benchmark_model_config(
        "math_500_test",
        "rwkv7-g1g-7.2b-20260523-ctx8192",
        stage="cot",
    )
    g1h = resolve_benchmark_model_config(
        "math_500_test",
        "rwkv7-g1h-7.2b-20260710-ctx10240",
        stage="cot",
    )
    g1i = resolve_benchmark_model_config(
        "math_500_test",
        "rwkv7-g1i-7.2b-20260805-ctx16384",
        stage="cot",
    )

    assert g1g is not None
    assert g1h is not None
    assert g1i is not None
    assert g1g.sampling_overrides["max_generate_tokens"] == 6144
    assert g1h.sampling_overrides["max_generate_tokens"] == 8192
    assert g1i.sampling_overrides["max_generate_tokens"] == 12288


def test_g1g_and_g1h_mbpp_use_same_frontend_avg8_metric(monkeypatch) -> None:
    monkeypatch.delenv("RWKV_BENCHMARK_CONFIG_ROOT", raising=False)
    g1g = resolve_benchmark_model_config(
        "mbpp_test",
        "rwkv7-g1g-2.9b-20260526-ctx8192",
        stage="no_cot",
    )
    monkeypatch.setenv("RWKV_BENCHMARK_CONFIG_ROOT", "configs/g1h")
    g1h = resolve_benchmark_model_config(
        "mbpp_test",
        "rwkv7-g1h-2.9b-20260710-ctx10240",
        stage="no_cot",
    )

    assert g1g is not None
    assert g1h is not None
    assert g1g.avg_k == g1h.avg_k == (8,)
    assert g1g.report_avg_k == g1h.report_avg_k == (8,)


def test_g1h_olympiadbench_uses_larger_final_answer_budget(monkeypatch) -> None:
    monkeypatch.setenv("RWKV_BENCHMARK_CONFIG_ROOT", "configs/g1h")
    config = resolve_sampling_config(
        "olympiadbench_test",
        "rwkv7-g1h-13.3b-20260710-ctx10240",
        stage="final",
    )

    assert config is not None
    assert config.max_generate_tokens == 512


def test_g1h_answer_judge_cot_keeps_minimum_think_guard(monkeypatch) -> None:
    monkeypatch.setenv("RWKV_BENCHMARK_CONFIG_ROOT", "configs/g1h")
    config = resolve_sampling_config(
        "answer_judge_test",
        "rwkv7-g1h-1.5b-20260710-ctx10240",
        stage="cot",
    )

    assert config is not None
    assert config.max_generate_tokens == 8192
    assert config.min_think_tokens == 16
    assert config.bad_words == ("</think>",)
    assert config.stop_tokens == (0,)


def test_generation_budget_override_leaves_other_limits_unchanged() -> None:
    config = resolve_sampling_config(
        "livecodebench_test",
        "rwkv7-g1h-7.2b-20260710-ctx10240",
        stage="final",
    )

    assert config is not None
    assert config.max_generate_tokens == 8192


def test_g1g_caps_explicit_8k_long_generation_budget_at_6k() -> None:
    config = resolve_sampling_config(
        "livecodebench_test",
        "rwkv7-g1g-7.2b-20260523-ctx8192",
        stage="final",
    )

    assert config is not None
    assert config.max_generate_tokens == 6144


def test_g1g_keeps_short_generation_budget_unchanged() -> None:
    config = resolve_sampling_config(
        "human_eval_test",
        "rwkv7-g1g-7.2b-20260523-ctx8192",
        stage="code",
    )

    assert config is not None
    assert config.max_generate_tokens == 1024


def test_g1h_math_odyssey_final_answer_budget_is_512(monkeypatch) -> None:
    monkeypatch.setenv("RWKV_BENCHMARK_CONFIG_ROOT", "configs/g1h")
    config = resolve_sampling_config(
        "math_odyssey_test",
        "rwkv7-g1h-1.5b-20260710-ctx10240",
        stage="final",
    )

    assert config is not None
    assert config.max_generate_tokens == 512


def test_resolve_livecodebench_final_sampling_config_uses_code_template() -> None:
    config = resolve_sampling_config("livecodebench_test", "rwkv7-g1a-2.9b", stage="final")

    assert config is not None
    assert config.max_generate_tokens == 8192
    assert config.temperature == 0.8
    assert config.top_p == 0.6
    assert config.stop_tokens == (6884, 21214)
    assert config.pad_zero is True


def test_resolve_sampling_config_supports_fallback_templates() -> None:
    base = SamplingConfig(max_generate_tokens=128, temperature=1.0, top_p=1.0)

    config = resolve_sampling_config(
        "unknown_benchmark_test",
        "rwkv7-g1a-2.9b",
        stage="cot",
        base=base,
        fallback_templates="code_default",
    )

    assert config is not None
    assert config.max_generate_tokens == 1024
    assert config.temperature == 0.8
    assert config.top_p == 0.6
    assert config.stop_tokens == (0, 261, 6884, 21214, 24281)


def test_resolve_sampling_config_expands_nested_template_aliases() -> None:
    config = resolve_sampling_config(
        "unknown_function_benchmark_test",
        "rwkv7-g1a-2.9b",
        stage="tool",
        fallback_templates="function_call_default",
    )

    assert config is not None
    assert config.max_generate_tokens == 2048
    assert config.temperature == 0.8
    assert config.top_k == 200
    assert config.top_p == 1e-5


def test_parse_table_accepts_rwkv_rs_sampling_aliases() -> None:
    config = benchmark_config._parse_table(  # type: ignore[attr-defined]
        {
            "max_new_tokens": 512,
            "presence_penalty": 0.7,
            "repetition_penalty": 0.2,
            "penalty_decay": 0.95,
        }
    )

    assert config.sampling_overrides == {
        "max_generate_tokens": 512,
        "alpha_presence": 0.7,
        "alpha_frequency": 0.2,
        "alpha_decay": 0.95,
    }


def test_parse_table_accepts_target_samples() -> None:
    config = benchmark_config._parse_table(  # type: ignore[attr-defined]
        {
            "target_samples": 500,
            "max_samples": 20,
        }
    )

    assert config.target_samples == 500
    assert config.max_samples == 20


def test_sampling_config_to_dict_uses_rwkv_rs_field_names() -> None:
    payload = sampling_config_to_dict(
        SamplingConfig(
            max_generate_tokens=256,
            temperature=0.5,
            top_k=50,
            top_p=0.3,
            alpha_presence=1.0,
            alpha_frequency=0.1,
            alpha_decay=0.99,
            allowed_token_ids=(300, 301),
        )
    )

    assert payload["max_new_tokens"] == 256
    assert payload["presence_penalty"] == 1.0
    assert payload["repetition_penalty"] == 0.1
    assert payload["penalty_decay"] == 0.99
    assert payload["allowed_token_ids"] == [300, 301]
    assert "max_generate_tokens" not in payload
    assert "alpha_presence" not in payload
