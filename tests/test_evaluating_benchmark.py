from __future__ import annotations

from src.eval.benchmark_registry import BenchmarkField
from src.eval.evaluating import (
    collect_benchmark_dataset_slugs,
    collect_benchmarks,
    resolve_registered_benchmark_name,
)
from src.eval.scheduler.dataset_utils import canonical_slug


def test_collect_benchmarks_matches_field_plus_extra_union() -> None:
    selected = collect_benchmarks(
        fields=(BenchmarkField.KNOWLEDGE,),
        extra_benchmark_names=("gsm8k",),
    )

    names = [item.name for item in selected]
    dataset_slugs = [item.dataset_slug for item in selected]

    assert "mmlu" in names
    assert "gsm8k" in names
    assert "gpqa_main" in names
    assert "gpqa_extended" in names
    assert "gpqa_diamond" in names
    assert canonical_slug("mmlu_test") in dataset_slugs
    assert canonical_slug("gpqa_main") in dataset_slugs
    assert canonical_slug("gpqa_extended") in dataset_slugs
    assert canonical_slug("gpqa_diamond") in dataset_slugs
    assert canonical_slug("gsm8k_test") in dataset_slugs


def test_resolve_registered_benchmark_name_accepts_dataset_slug() -> None:
    assert resolve_registered_benchmark_name("simpleqa_verified") == "simpleqa"
    assert resolve_registered_benchmark_name("ifeval_test") == "ifeval"


def test_collect_benchmark_dataset_slugs_uses_default_splits() -> None:
    slugs = collect_benchmark_dataset_slugs(
        fields=(BenchmarkField.INSTRUCTION_FOLLOWING,),
        extra_benchmark_names=("browsecomp",),
    )

    assert canonical_slug("ifeval_test") in slugs
    assert canonical_slug("ifbench_test") in slugs
    assert canonical_slug("browsecomp_test") in slugs


def test_collect_benchmark_dataset_slugs_expands_group_aliases() -> None:
    slugs = collect_benchmark_dataset_slugs(extra_benchmark_names=("tau_bench", "gpqa"))

    assert canonical_slug("tau_bench_retail_test") in slugs
    assert canonical_slug("tau_bench_airline_test") in slugs
    assert canonical_slug("tau_bench_telecom_test") in slugs
    assert canonical_slug("gpqa_main") in slugs
    assert canonical_slug("gpqa_extended") in slugs
    assert canonical_slug("gpqa_diamond") in slugs


def test_default_collection_is_the_formal_strict46() -> None:
    selected = collect_benchmarks()

    assert len(selected) == 46
    assert all(item.field is not BenchmarkField.FUNCTION_CALLING for item in selected)


def test_explicit_function_calling_field_keeps_auxiliary_catalog_available() -> None:
    selected = collect_benchmarks(fields=(BenchmarkField.FUNCTION_CALLING,))

    assert len(selected) == 62
    assert all(item.field is BenchmarkField.FUNCTION_CALLING for item in selected)
