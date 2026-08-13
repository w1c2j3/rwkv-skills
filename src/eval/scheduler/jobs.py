"""Job catalogue & dataset bookkeeping for the scheduler."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

from src.eval.benchmark_registry import KNOWN_BENCHMARKS, KNOWN_BENCHMARKS_BY_FIELD, BenchmarkField
from src.eval.runner_registry import ALL_RUNNERS, RunnerGroup, RunnerSpec as RegisteredRunnerSpec

from .dataset_utils import (
    DATASET_SLUG_ALIASES,
    canonical_slug,
    infer_dataset_slug_from_path,
    make_dataset_slug,
    safe_slug,
)


@dataclass(frozen=True)
class DatasetPrepSpec:
    dataset: str
    split: str


@dataclass(frozen=True)
class JobSpec:
    name: str
    module: str
    dataset_slugs: tuple[str, ...]
    is_cot: bool
    domain: str
    runner_group: RunnerGroup | None = None
    extra_args: tuple[str, ...] = ()
    batch_flag: str | None = None
    probe_flag: str | None = None
    probe_max_generate_flag: str | None = None
    probe_dataset_required: bool = False
    probe_extra_args: tuple[str, ...] = ()
    # 用于探测 batch size 时估算实际样本量（例如 pass@k>1 时每题会展开多条样本）
    probe_samples_per_task: int = 1
    # probe 时可按较大规模挑选候选 batch，避免只用小样本探测出不代表完整评测的 batch。
    probe_question_floor: int = 0

    @property
    def id_prefix(self) -> str:
        return f"{self.name}__"


def _field_dataset_slugs(field: BenchmarkField) -> tuple[str, ...]:
    return tuple(
        sorted(
            canonical_slug(make_dataset_slug(metadata.dataset, metadata.default_split))
            for metadata in KNOWN_BENCHMARKS_BY_FIELD.get(field, ())
        )
    )


def _build_dataset_catalogues() -> tuple[
    dict[str, DatasetPrepSpec],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    dict[str, tuple[str, ...]],
]:
    specs: dict[str, DatasetPrepSpec] = {}
    job_dataset_slugs: dict[str, list[str]] = {}
    for metadata in KNOWN_BENCHMARKS:
        slug = canonical_slug(make_dataset_slug(metadata.dataset, metadata.default_split))
        specs.setdefault(slug, DatasetPrepSpec(metadata.dataset, metadata.default_split))
        for job_name in metadata.scheduler_jobs:
            job_dataset_slugs.setdefault(job_name, []).append(slug)

    for alias, target in DATASET_SLUG_ALIASES.items():
        canonical = canonical_slug(target)
        if canonical in specs:
            alias_slug = canonical_slug(alias)
            specs.setdefault(alias_slug, specs[canonical])

    multi_choice_tuple = _field_dataset_slugs(BenchmarkField.KNOWLEDGE)
    math_tuple = _field_dataset_slugs(BenchmarkField.MATHS)
    special_tuple = _field_dataset_slugs(BenchmarkField.INSTRUCTION_FOLLOWING)
    code_tuple = tuple(
        sorted(
            set(_field_dataset_slugs(BenchmarkField.CODING))
            | set(_field_dataset_slugs(BenchmarkField.FUNCTION_CALLING))
        )
    )
    job_tuples = {
        job_name: tuple(sorted({canonical_slug(slug) for slug in slugs}))
        for job_name, slugs in job_dataset_slugs.items()
    }
    return specs, multi_choice_tuple, math_tuple, special_tuple, code_tuple, job_tuples


(
    DATASET_PREP_SPECS,
    MULTICHOICE_DATASET_SLUGS,
    MATH_DATASET_SLUGS,
    SPECIAL_DATASET_SLUGS,
    CODE_DATASET_SLUGS,
    _JOB_DATASET_SLUGS,
) = _build_dataset_catalogues()


def _job_dataset_slugs(job_name: str) -> tuple[str, ...]:
    return _JOB_DATASET_SLUGS.get(job_name, ())


LLM_JUDGE_DATASET_SLUGS: Final[tuple[str, ...]] = _job_dataset_slugs("free_response_judge")
# judge-only 数据集不再调度到 free_response；parser-grade Math such as
# Math-500/GSM8K/OlympiadBench stays on free_response/math_verify.
MATH_DATASET_SLUGS_FOR_FREE_RESPONSE: Final[tuple[str, ...]] = tuple(
    _job_dataset_slugs("free_response") or (slug for slug in MATH_DATASET_SLUGS if slug not in LLM_JUDGE_DATASET_SLUGS)
)

INSTRUCTION_FOLLOWING_DATASET_SLUGS: Final[tuple[str, ...]] = tuple(
    sorted(set(_job_dataset_slugs("instruction_following") or SPECIAL_DATASET_SLUGS or (canonical_slug("ifeval_test"),)))
)

HUMAN_EVAL_CODE_SLUGS: Final[tuple[str, ...]] = _job_dataset_slugs("code_human_eval")
MBPP_CODE_SLUGS: Final[tuple[str, ...]] = _job_dataset_slugs("code_mbpp")
LCB_CODE_SLUGS: Final[tuple[str, ...]] = _job_dataset_slugs("code_livecodebench")
FUNCTION_AGENT_TOOL_CALL_SLUGS: Final[tuple[str, ...]] = _job_dataset_slugs("function_agent_tool_call")
FUNCTION_BROWSECOMP_SLUGS: Final[tuple[str, ...]] = _job_dataset_slugs("function_browsecomp")
FUNCTION_BROWSECOMP_PLUS_SLUGS: Final[tuple[str, ...]] = _job_dataset_slugs("function_browsecomp_plus")
FUNCTION_COMPLEXFUNCBENCH_SLUGS: Final[tuple[str, ...]] = _job_dataset_slugs("function_complexfuncbench")
FUNCTION_LONGBENCH_SLUGS: Final[tuple[str, ...]] = _job_dataset_slugs("function_longbench")
FUNCTION_LONGCODEBENCH_SLUGS: Final[tuple[str, ...]] = _job_dataset_slugs("function_longcodebench")
FUNCTION_MCP_BENCH_SLUGS: Final[tuple[str, ...]] = _job_dataset_slugs("function_mcp_bench")
FUNCTION_BFCL_AST_SLUGS: Final[tuple[str, ...]] = _job_dataset_slugs("function_bfcl_ast")
FUNCTION_TOOLALPACA_SLUGS: Final[tuple[str, ...]] = _job_dataset_slugs("function_toolalpaca")
FUNCTION_TAU_BENCH_SLUGS: Final[tuple[str, ...]] = _job_dataset_slugs("function_tau_bench")
FUNCTION_TAU2_BENCH_SLUGS: Final[tuple[str, ...]] = _job_dataset_slugs("function_tau2_bench")
FUNCTION_TAU3_BENCH_SLUGS: Final[tuple[str, ...]] = _job_dataset_slugs("function_tau3_bench")


def _dataset_slugs_for_runner(runner: RegisteredRunnerSpec) -> tuple[str, ...]:
    return _job_dataset_slugs(runner.name) or runner.fallback_dataset_slugs


def _build_job_catalogue() -> dict[str, JobSpec]:
    catalogue: dict[str, JobSpec] = {}
    for runner in ALL_RUNNERS:
        catalogue[runner.name] = JobSpec(
            name=runner.name,
            module=runner.module,
            dataset_slugs=_dataset_slugs_for_runner(runner),
            is_cot=runner.is_cot,
            domain=runner.scheduler_domain,
            runner_group=runner.group,
            extra_args=runner.extra_args,
            batch_flag=runner.batch_flag,
            probe_flag=runner.probe_flag,
            probe_max_generate_flag=runner.probe_max_generate_flag,
            probe_dataset_required=runner.probe_dataset_required,
            probe_extra_args=runner.probe_extra_args,
            probe_samples_per_task=runner.probe_samples_per_task,
            probe_question_floor=runner.probe_question_floor,
        )
    return catalogue


JOB_CATALOGUE: dict[str, JobSpec] = _build_job_catalogue()

JOB_ORDER: tuple[str, ...] = tuple(runner.name for runner in ALL_RUNNERS)


def detect_job_from_dataset(dataset_slug: str, is_cot: bool) -> str | None:
    slug = canonical_slug(dataset_slug)
    # 优先判定真正要求 LLM judge 的数据集；parser-grade Math uses free_response.
    if is_cot and slug in LLM_JUDGE_DATASET_SLUGS:
        return "free_response_judge"
    for job_name, spec in JOB_CATALOGUE.items():
        if spec.is_cot == is_cot and slug in spec.dataset_slugs:
            return job_name
    return None


def locate_dataset(slug: str, *, search: Sequence[Path], output_root: Path, record_stats: bool = True) -> Path:
    from .datasets import find_dataset_file, refresh_dataset_index
    from .dataset_stats import record_dataset_samples

    canonical = canonical_slug(slug)
    spec = DATASET_PREP_SPECS.get(canonical)
    found = find_dataset_file(canonical, search)
    if found:
        if spec is not None and _dataset_refresh_requested():
            expected_artifact = (output_root.expanduser().resolve() / spec.dataset / f"{spec.split}.jsonl").resolve()
            if found.resolve() == expected_artifact:
                from src.eval.datasets.data_prepper.data_manager import prepare_dataset

                prepared_paths = prepare_dataset(spec.dataset, output_root, spec.split)
                refresh_dataset_index(search)
                for path in prepared_paths:
                    detected = canonical_slug(infer_dataset_slug_from_path(str(path)))
                    if detected == canonical:
                        if record_stats:
                            record_dataset_samples(path, dataset_slug=canonical)
                        return path
                refreshed = find_dataset_file(canonical, search)
                if refreshed:
                    if record_stats:
                        record_dataset_samples(refreshed, dataset_slug=canonical)
                    return refreshed
        if record_stats:
            record_dataset_samples(found, dataset_slug=canonical)
        return found

    if spec is None:
        locations = "\n".join(f"  - {root}" for root in search)
        raise FileNotFoundError(
            f"未找到数据集 {slug!r}。请将 JSONL 文件放置于以下目录之一：\n{locations}"
        )

    from src.eval.datasets.data_prepper.data_manager import prepare_dataset

    prepared_paths = prepare_dataset(spec.dataset, output_root, spec.split)
    refresh_dataset_index(search)
    for path in prepared_paths:
        detected = canonical_slug(infer_dataset_slug_from_path(str(path)))
        if detected == canonical:
            if record_stats:
                record_dataset_samples(path, dataset_slug=canonical)
            return path
    refreshed = find_dataset_file(canonical, search)
    if refreshed:
        if record_stats:
            record_dataset_samples(refreshed, dataset_slug=canonical)
        return refreshed
    raise FileNotFoundError(f"数据集 {slug!r} 未生成 JSONL，prepare_dataset 返回 {prepared_paths}")


def _dataset_refresh_requested() -> bool:
    value = os.environ.get("RWKV_EVAL_REFRESH_DATASET") or os.environ.get("RWKV_DATASET_REFRESH")
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


__all__ = [
    "JobSpec",
    "JOB_CATALOGUE",
    "JOB_ORDER",
    "DATASET_PREP_SPECS",
    "detect_job_from_dataset",
    "locate_dataset",
    "safe_slug",
]
