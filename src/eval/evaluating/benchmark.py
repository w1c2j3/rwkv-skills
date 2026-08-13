"""Benchmark selection helpers aligned with rwkv-rs' evaluating layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.eval.benchmark_registry import (
    ALL_BENCHMARKS,
    BENCHMARKS_BY_FIELD,
    BenchmarkField,
    BenchmarkMetadata,
    KNOWN_BENCHMARKS,
    KNOWN_BENCHMARKS_BY_FIELD,
    expand_benchmark_alias,
    resolve_benchmark_metadata,
)
from src.eval.scheduler.dataset_utils import make_dataset_slug


_BENCHMARKS_BY_NAME: dict[str, BenchmarkMetadata] = {item.name: item for item in KNOWN_BENCHMARKS}


@dataclass(frozen=True, slots=True)
class SelectedBenchmark:
    metadata: BenchmarkMetadata
    dataset_slug: str

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def field(self) -> BenchmarkField:
        return self.metadata.field


def resolve_registered_benchmark_name(raw_name: str) -> str:
    resolved = expand_benchmark_alias(raw_name)
    if len(resolved) == 1:
        return resolved[0]

    raise ValueError(f"unknown benchmark name: {raw_name}")


def benchmark_dataset_slug(metadata: BenchmarkMetadata) -> str:
    return make_dataset_slug(metadata.dataset, metadata.default_split)


def collect_benchmarks(
    *,
    fields: Sequence[BenchmarkField] | None = None,
    extra_benchmark_names: Sequence[str] | None = None,
) -> tuple[SelectedBenchmark, ...]:
    selected: dict[str, SelectedBenchmark] = {}

    for field in fields or ():
        # Explicit field selection keeps auxiliary function-calling workflows
        # available.  The no-filter default below remains the formal Strict46.
        field_catalogue = (
            KNOWN_BENCHMARKS_BY_FIELD
            if field is BenchmarkField.FUNCTION_CALLING
            else BENCHMARKS_BY_FIELD
        )
        for metadata in field_catalogue.get(field, ()):
            selected.setdefault(
                metadata.name,
                SelectedBenchmark(metadata=metadata, dataset_slug=benchmark_dataset_slug(metadata)),
            )

    for raw_name in extra_benchmark_names or ():
        resolved_names = expand_benchmark_alias(raw_name)
        if not resolved_names:
            raise ValueError(f"unknown benchmark name: {raw_name}")
        for resolved_name in resolved_names:
            metadata = _BENCHMARKS_BY_NAME.get(resolved_name)
            if metadata is None:
                metadata = resolve_benchmark_metadata(resolved_name)
            selected[metadata.name] = SelectedBenchmark(
                metadata=metadata,
                dataset_slug=benchmark_dataset_slug(metadata),
            )

    if not selected and not fields and not extra_benchmark_names:
        for metadata in ALL_BENCHMARKS:
            selected.setdefault(
                metadata.name,
                SelectedBenchmark(metadata=metadata, dataset_slug=benchmark_dataset_slug(metadata)),
            )

    return tuple(sorted(selected.values(), key=lambda item: (item.field.value, item.name)))


def collect_benchmark_dataset_slugs(
    *,
    fields: Sequence[BenchmarkField] | None = None,
    extra_benchmark_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    return tuple(item.dataset_slug for item in collect_benchmarks(fields=fields, extra_benchmark_names=extra_benchmark_names))


__all__ = [
    "SelectedBenchmark",
    "benchmark_dataset_slug",
    "collect_benchmark_dataset_slugs",
    "collect_benchmarks",
    "resolve_registered_benchmark_name",
]
