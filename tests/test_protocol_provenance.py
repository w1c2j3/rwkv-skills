from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.db.eval_db_service import ResumeContext
from src.eval.benchmark_registry import CoTMode
from src.eval.datasets.snapshot import (
    RUNTIME_ATTESTATION_PROVENANCE_SCHEMA_VERSION,
    bind_resume_identity,
    build_dataset_snapshot,
    build_protocol_bundle,
    canonical_json_sha256,
)
from src.eval.evaluating import RunMode, prepare_task_execution
from src.eval.evaluating.task_persistence import STRICT_RUNTIME_PROVENANCE_ENV
from src.eval.field_common import build_task_sampling_config


class _SnapshotLoader:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[dict[str, object]]:
        return [json.loads(line) for line in self.path.read_text().splitlines()]


def _snapshot_resolver(slug: str) -> str:
    return slug


class _CaptureService:
    def __init__(self) -> None:
        self.get_calls: list[dict[str, object]] = []
        self.create_calls: list[dict[str, object]] = []

    def get_resume_context(self, **kwargs: object) -> ResumeContext:
        self.get_calls.append(dict(kwargs))
        return ResumeContext()

    def create_task_from_context(self, **kwargs: object) -> str:
        self.create_calls.append(dict(kwargs))
        return "101"


def _make_snapshot(tmp_path: Path) -> tuple[dict[str, object], Path]:
    dataset = tmp_path / "demo_test.jsonl"
    dataset.write_text('{"id":1,"text":"a"}\n{"id":2,"text":"b"}\n')
    Path(f"{dataset}.manifest.json").write_text(
        json.dumps({"source": {"revision": "dataset-commit-42"}})
    )
    snapshot = build_dataset_snapshot(
        dataset,
        dataset_slug="demo_test",
        loader=_SnapshotLoader,
        resolver=_snapshot_resolver,
        repo_root=tmp_path,
    )
    return snapshot, dataset


def _runtime_task_provenance(
    model: str = "rwkv7-g1i-1.5b-20260805-ctx16384",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": RUNTIME_ATTESTATION_PROVENANCE_SCHEMA_VERSION,
        "model": model,
        "route_kind": "local",
        "scheduler_endpoint": {
            "scheme": "http",
            "host": "127.0.0.1",
            "port": 19415,
            "api_prefix": "/v1",
        },
        "global_approval_sha256": "1" * 64,
        "global_approval_file_sha256": "2" * 64,
        "protocol_lock_sha256": "3" * 64,
        "protocol_lock_file_sha256": "4" * 64,
        "runtime_attestation_evidence_sha256": "5" * 64,
        "runtime_attestation_artifact_sha256": "6" * 64,
        "forward_attestation_artifact_sha256": None,
        "host_label": "157",
        "weight": {
            "path": f"/opt/rwkv-weights/{model}.pth",
            "bytes": 123,
            "sha256": "7" * 64,
        },
        "runtime_executable_sha256": "8" * 64,
        "runtime_tree_sha256": "9" * 64,
        "semantic_environment_sha256": "a" * 64,
        "launch_parameters_sha256": "b" * 64,
        "systemd_unit": "rwkv-g1i-15.service",
        "gpu_index": 3,
    }
    payload["provenance_sha256"] = canonical_json_sha256(payload)
    return payload


def test_dataset_snapshot_binds_bytes_records_order_code_and_revision(
    tmp_path: Path,
) -> None:
    snapshot, dataset = _make_snapshot(tmp_path)
    repeated = build_dataset_snapshot(
        dataset,
        dataset_slug="demo_test",
        loader=_SnapshotLoader,
        resolver=_snapshot_resolver,
        repo_root=tmp_path,
    )

    assert repeated == snapshot
    assert snapshot["row_count"] == 2
    assert snapshot["source_revision"] == "dataset-commit-42"
    assert snapshot["loader"]["fqcn"].endswith("._SnapshotLoader")
    assert snapshot["resolver"]["fqcn"].endswith("._snapshot_resolver")

    canonical_digest = snapshot["canonical_records_sha256"]
    raw_digest = snapshot["raw_file_sha256"]
    dataset.write_text('{ "id": 1, "text": "a" }\n{"id":2,"text":"b"}\n')
    whitespace_changed = build_dataset_snapshot(
        dataset,
        dataset_slug="demo_test",
        loader=_SnapshotLoader,
        resolver=_snapshot_resolver,
        repo_root=tmp_path,
    )
    assert whitespace_changed["canonical_records_sha256"] == canonical_digest
    assert whitespace_changed["raw_file_sha256"] != raw_digest

    dataset.write_text('{"id":2,"text":"b"}\n{"id":1,"text":"a"}\n')
    reordered = build_dataset_snapshot(
        dataset,
        dataset_slug="demo_test",
        loader=_SnapshotLoader,
        resolver=_snapshot_resolver,
        repo_root=tmp_path,
    )
    assert reordered["canonical_records_sha256"] != canonical_digest


def test_dataset_snapshot_canonicalizes_nonfinite_floats_deterministically(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "nonfinite_test.jsonl"
    dataset.write_text(
        '{"id":1,"value":NaN}\n'
        '{"id":2,"value":Infinity}\n'
        '{"id":3,"value":-Infinity}\n'
    )

    first = build_dataset_snapshot(
        dataset,
        dataset_slug="nonfinite_test",
        loader=_SnapshotLoader,
        resolver=_snapshot_resolver,
        repo_root=tmp_path,
    )
    repeated = build_dataset_snapshot(
        dataset,
        dataset_slug="nonfinite_test",
        loader=_SnapshotLoader,
        resolver=_snapshot_resolver,
        repo_root=tmp_path,
    )

    assert first == repeated
    assert first["row_count"] == 3
    assert canonical_json_sha256(float("nan")) != canonical_json_sha256(float("inf"))
    assert canonical_json_sha256(float("inf")) != canonical_json_sha256(float("-inf"))


def test_dataset_snapshot_rejects_caller_records_from_different_bytes(
    tmp_path: Path,
) -> None:
    _snapshot, dataset = _make_snapshot(tmp_path)

    with pytest.raises(RuntimeError, match="do not match the exact dataset bytes"):
        build_dataset_snapshot(
            dataset,
            dataset_slug="demo_test",
            loader=_SnapshotLoader,
            resolver=_snapshot_resolver,
            records=[{"id": 1, "text": "stale"}, {"id": 2, "text": "b"}],
            repo_root=tmp_path,
        )


def test_provenance_rejects_dataset_and_protocol_symlinks(tmp_path: Path) -> None:
    _snapshot, dataset = _make_snapshot(tmp_path)
    dataset_link = tmp_path / "linked.jsonl"
    dataset_link.symlink_to(dataset)

    with pytest.raises(ValueError, match="dataset file must not be a symlink"):
        build_dataset_snapshot(
            dataset_link,
            dataset_slug="demo_test",
            loader=_SnapshotLoader,
            resolver=_snapshot_resolver,
            repo_root=tmp_path,
        )

    source = tmp_path / "source.py"
    source.write_text("PROTOCOL = 1\n")
    source_link = tmp_path / "linked-source.py"
    source_link.symlink_to(source)
    with pytest.raises(ValueError, match="protocol source must not be a symlink"):
        build_protocol_bundle(
            protocol_name="demo",
            source_files=(source_link,),
            resolved_contract={"mode": "NoCoT"},
            repo_root=tmp_path,
        )


def test_protocol_bundle_and_resume_identity_fail_closed_on_drift(
    tmp_path: Path,
) -> None:
    snapshot, _dataset = _make_snapshot(tmp_path)
    source = tmp_path / "metric.py"
    config = tmp_path / "benchmark.toml"
    source.write_text("def score(x): return x\n")
    config.write_text("avg_k = [8]\n")
    bundle = build_protocol_bundle(
        protocol_name="demo",
        source_files=(source,),
        config_files=(config,),
        resolved_contract={"mode": "NoCoT", "sampling": {"temperature": 1.0}},
        repo_root=tmp_path,
    )
    task_config = build_task_sampling_config(
        cot_mode=CoTMode.NO_COT,
        avg_k=1,
        effective_sample_count=2,
        dataset_snapshot=snapshot,
        protocol_bundle=bundle,
    )

    assert bind_resume_identity(task_config) == task_config
    assert task_config["resume_identity_sha256"]

    stale = copy.deepcopy(task_config)
    stale["dataset_snapshot"]["raw_file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="stale resume_identity_sha256"):
        bind_resume_identity(stale)

    # Recomputing the outer digest must not authenticate a forged nested
    # snapshot.  Nested self-digests are independently mandatory.
    forged = copy.deepcopy(task_config)
    forged["dataset_snapshot"]["raw_file_sha256"] = "0" * 64
    forged.pop("resume_identity_sha256")
    forged["resume_identity_schema_version"] = "rwkv.task-resume-identity.v1"
    forged["resume_identity_sha256"] = canonical_json_sha256(forged)
    with pytest.raises(ValueError, match="dataset_snapshot self-digest mismatch"):
        bind_resume_identity(forged)

    source.write_text("def score(x): return not x\n")
    changed_bundle = build_protocol_bundle(
        protocol_name="demo",
        source_files=(source,),
        config_files=(config,),
        resolved_contract={"mode": "NoCoT", "sampling": {"temperature": 1.0}},
        repo_root=tmp_path,
    )
    assert changed_bundle["bundle_sha256"] != bundle["bundle_sha256"]


def test_prepare_task_execution_uses_one_bound_identity_for_lookup_and_create(
    tmp_path: Path,
) -> None:
    snapshot, _dataset = _make_snapshot(tmp_path)
    protocol_source = tmp_path / "runner.py"
    protocol_source.write_text("PROTOCOL = 1\n")
    bundle = build_protocol_bundle(
        protocol_name="demo",
        source_files=(protocol_source,),
        resolved_contract={"mode": "NoCoT"},
        repo_root=tmp_path,
    )
    unbound = {
        "dataset_snapshot": snapshot,
        "protocol_bundle": bundle,
        "sampling_config": {"temperature": 1.0},
    }
    service = _CaptureService()

    state = prepare_task_execution(
        service=service,
        dataset="demo_test",
        model="demo-model",
        is_param_search=False,
        job_name="multi_choice_plain_naive",
        sampling_config=unbound,
        run_mode=RunMode.FRESH,
    )

    assert state.task_id == "101"
    lookup_config = service.get_calls[0]["sampling_config"]
    create_config = service.create_calls[0]["sampling_config"]
    assert lookup_config == create_config
    assert lookup_config["resume_identity_sha256"]


def test_prepare_task_execution_binds_attested_runtime_before_database_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, _dataset = _make_snapshot(tmp_path)
    protocol_source = tmp_path / "runner.py"
    protocol_source.write_text("PROTOCOL = 1\n")
    bundle = build_protocol_bundle(
        protocol_name="demo",
        source_files=(protocol_source,),
        resolved_contract={"mode": "NoCoT"},
        repo_root=tmp_path,
    )
    model = "rwkv7-g1i-1.5b-20260805-ctx16384"
    provenance = _runtime_task_provenance(model)
    monkeypatch.setenv(
        STRICT_RUNTIME_PROVENANCE_ENV,
        json.dumps(provenance, sort_keys=True, separators=(",", ":")),
    )
    service = _CaptureService()

    prepare_task_execution(
        service=service,
        dataset="demo_test",
        model=model,
        is_param_search=False,
        job_name="multi_choice_plain_naive",
        sampling_config={
            "dataset_snapshot": snapshot,
            "protocol_bundle": bundle,
            "sampling_config": {"temperature": 1.0},
        },
        run_mode=RunMode.FRESH,
    )

    lookup_config = service.get_calls[0]["sampling_config"]
    assert lookup_config["runtime_attestation_provenance"] == provenance
    assert lookup_config["resume_identity_sha256"]

    wrong_model_service = _CaptureService()
    with pytest.raises(ValueError, match="does not match task model"):
        prepare_task_execution(
            service=wrong_model_service,
            dataset="demo_test",
            model="rwkv7-g1i-2.9b-20260805-ctx16384",
            is_param_search=False,
            job_name="multi_choice_plain_naive",
            sampling_config={
                "dataset_snapshot": snapshot,
                "protocol_bundle": bundle,
            },
            run_mode=RunMode.FRESH,
        )
    assert wrong_model_service.get_calls == []
    assert wrong_model_service.create_calls == []
