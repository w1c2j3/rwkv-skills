"""Content-addressed dataset and evaluation-protocol provenance.

The objects returned by this module are intentionally composed only of JSON
primitives.  They are persisted in ``task.sampling_config`` and therefore form
part of the database resume identity.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
import tempfile
from typing import Any, Callable


SNAPSHOT_SCHEMA_VERSION = "rwkv.dataset-snapshot.v1"
PROTOCOL_BUNDLE_SCHEMA_VERSION = "rwkv.protocol-bundle.v1"
RESUME_IDENTITY_SCHEMA_VERSION = "rwkv.task-resume-identity.v1"
RUNTIME_ATTESTATION_PROVENANCE_SCHEMA_VERSION = (
    "rwkv.g1i-runtime-task-provenance.v1"
)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REVISION_KEYS = (
    "dataset_revision",
    "source_revision",
    "revision",
    "commit",
    "commit_sha",
    "sha",
    "version",
)
_NONFINITE_FLOAT_TAG = "__rwkv_provenance_nonfinite_float__"


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {_NONFINITE_FLOAT_TAG: "nan"}
        if math.isinf(value):
            label = "positive_infinity" if value > 0 else "negative_infinity"
            return {_NONFINITE_FLOAT_TAG: label}
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _canonical_value(model_dump())
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if key in normalized:
                raise ValueError(f"duplicate canonical mapping key: {key!r}")
            normalized[key] = _canonical_value(raw_value)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (set, frozenset)):
        normalized_items = [_canonical_value(item) for item in value]
        return sorted(normalized_items, key=canonical_json_bytes)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_value(item) for item in value]
    raise TypeError(f"unsupported provenance value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    payload, _metadata = _read_stable_file(path)
    return hashlib.sha256(payload).hexdigest()


def read_stable_file_bytes(path: str | Path) -> bytes:
    """Return the exact bytes of one non-symlink regular file."""

    payload, _metadata = _read_stable_file(path)
    return payload


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_stable_file(path: str | Path) -> tuple[bytes, os.stat_result]:
    """Read one regular file once and reject path swaps or concurrent writes.

    Hashing a pathname and parsing it in a later operation permits a classic
    check/use split.  Provenance must describe the exact bytes that were
    parsed, so callers consume the returned bytes rather than reopening the
    pathname.
    """

    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise ValueError(f"provenance source must not be a symlink: {unresolved}")
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"provenance source must be a regular non-symlink file: {resolved}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_after = resolved.stat()
    if _stat_identity(before) != _stat_identity(after) or _stat_identity(after) != _stat_identity(path_after):
        raise RuntimeError(f"provenance source changed while it was read: {resolved}")
    payload = b"".join(chunks)
    if len(payload) != int(after.st_size):
        raise RuntimeError(f"short provenance read: {resolved}")
    return payload, after


def _portable_path(path: Path, *, repo_root: Path = _REPO_ROOT) -> tuple[str, str]:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix(), "repo_relative"
    except ValueError:
        return str(resolved), "absolute"


def _file_descriptor(
    path: str | Path, *, repo_root: Path = _REPO_ROOT
) -> dict[str, object]:
    expanded = Path(path).expanduser()
    if expanded.is_symlink():
        raise ValueError(f"provenance source must not be a symlink: {expanded}")
    resolved = expanded.resolve(strict=True)
    payload, metadata = _read_stable_file(resolved)
    rendered_path, path_kind = _portable_path(resolved, repo_root=repo_root)
    return {
        "path": rendered_path,
        "path_kind": path_kind,
        "size_bytes": int(metadata.st_size),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _callable_descriptor(
    callback: Callable[..., Any], *, repo_root: Path
) -> dict[str, object]:
    source_path = inspect.getsourcefile(callback) or inspect.getfile(callback)
    descriptor = _file_descriptor(source_path, repo_root=repo_root)
    descriptor["fqcn"] = f"{callback.__module__}.{callback.__qualname__}"
    return descriptor


def _manifest_candidates(dataset_path: Path) -> tuple[Path, ...]:
    candidates = (
        Path(f"{dataset_path}.manifest.json"),
        dataset_path.with_suffix(".manifest.json"),
    )
    return tuple(dict.fromkeys(path for path in candidates if path.is_file()))


def _find_source_revision(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    for key in _REVISION_KEYS:
        revision = value.get(key)
        if isinstance(revision, (str, int, float)) and str(revision).strip():
            return str(revision).strip()
    for child in value.values():
        revision = _find_source_revision(child)
        if revision:
            return revision
    return None


def _source_manifest_provenance(
    dataset_path: Path,
    *,
    repo_root: Path,
) -> tuple[list[dict[str, object]], str | None]:
    descriptors: list[dict[str, object]] = []
    source_revision: str | None = None
    for path in _manifest_candidates(dataset_path):
        payload_bytes, metadata = _read_stable_file(path)
        rendered_path, path_kind = _portable_path(path, repo_root=repo_root)
        descriptor = {
            "path": rendered_path,
            "path_kind": path_kind,
            "size_bytes": int(metadata.st_size),
            "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        }
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        if source_revision is None:
            source_revision = _find_source_revision(payload)
        descriptors.append(descriptor)
    return descriptors, source_revision


def build_dataset_snapshot(
    dataset_path: str | Path,
    *,
    dataset_slug: str,
    loader: type[Any],
    resolver: Callable[..., Any],
    records: Iterable[Any] | None = None,
    repo_root: str | Path = _REPO_ROOT,
) -> dict[str, object]:
    """Build deterministic evidence for the exact bytes and parsed records.

    ``canonical_records_sha256`` is order-sensitive and hashes one canonical
    JSON record plus ``\n`` at a time.  This distinguishes both record changes
    and record reordering without materialising a second JSON array in memory.
    """

    root = Path(repo_root).expanduser().resolve()
    expanded_path = Path(dataset_path).expanduser()
    if expanded_path.is_symlink():
        raise ValueError(f"dataset file must not be a symlink: {expanded_path}")
    path = expanded_path.resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(f"dataset file does not exist: {path}")
    raw_bytes, raw_metadata = _read_stable_file(path)

    # Parse a private copy of the exact bytes hashed above.  A caller may pass
    # records loaded earlier for efficiency, but those records are evidence
    # only after they match an independent parse of the content-addressed
    # bytes.  This prevents an A-records/B-file mixed snapshot.
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.snapshot.",
            suffix=path.suffix,
            delete=False,
        ) as temporary:
            temporary.write(raw_bytes)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        parsed_records = list(loader(Path(temporary_name)).load())
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    def _records_evidence(values: Iterable[Any]) -> tuple[str, int]:
        digest = hashlib.sha256()
        count = 0
        for record in values:
            digest.update(canonical_json_bytes(record))
            digest.update(b"\n")
            count += 1
        return digest.hexdigest(), count

    records_sha256, row_count = _records_evidence(parsed_records)
    if records is not None:
        supplied_sha256, supplied_count = _records_evidence(records)
        if (supplied_sha256, supplied_count) != (records_sha256, row_count):
            raise RuntimeError(
                "caller-provided dataset records do not match the exact dataset bytes"
            )

    rendered_path, path_kind = _portable_path(path, repo_root=root)
    manifests, source_revision = _source_manifest_provenance(path, repo_root=root)
    payload: dict[str, object] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "dataset_slug": str(dataset_slug),
        "resolved_path": rendered_path,
        "resolved_path_kind": path_kind,
        "raw_file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "raw_size_bytes": int(raw_metadata.st_size),
        "canonical_records_algorithm": "canonical-json-lines-v1",
        "canonical_records_sha256": records_sha256,
        "row_count": row_count,
        "loader": _callable_descriptor(loader, repo_root=root),
        "resolver": _callable_descriptor(resolver, repo_root=root),
        "source_manifests": manifests,
        "source_revision": source_revision,
    }
    payload["snapshot_sha256"] = canonical_json_sha256(payload)
    return payload


def build_protocol_bundle(
    *,
    protocol_name: str,
    source_files: Iterable[str | Path],
    config_files: Iterable[str | Path] = (),
    resolved_contract: Mapping[str, Any],
    repo_root: str | Path = _REPO_ROOT,
) -> dict[str, object]:
    """Bind the exact executable sources, configs, and resolved contract."""

    root = Path(repo_root).expanduser().resolve()

    def _descriptors(paths: Iterable[str | Path]) -> list[dict[str, object]]:
        unique: set[Path] = set()
        for raw_path in paths:
            expanded = Path(raw_path).expanduser()
            if expanded.is_symlink():
                raise ValueError(f"protocol source must not be a symlink: {expanded}")
            unique.add(expanded.resolve(strict=True))
        return [
            _file_descriptor(path, repo_root=root)
            for path in sorted(unique, key=lambda item: str(item))
        ]

    sources = _descriptors(source_files)
    configs = _descriptors(config_files)
    # Detect a source/config rewrite during the multi-file inventory.  The
    # enclosing launch gate repeats the same inventory at dispatch time; this
    # local second pass ensures the task itself never records a torn bundle.
    for descriptor in (*sources, *configs):
        raw_path = Path(str(descriptor["path"]))
        if descriptor.get("path_kind") == "repo_relative":
            raw_path = root / raw_path
        if sha256_file(raw_path) != descriptor.get("sha256"):
            raise RuntimeError(f"protocol source changed during bundle creation: {raw_path}")

    payload: dict[str, object] = {
        "schema_version": PROTOCOL_BUNDLE_SCHEMA_VERSION,
        "protocol_name": str(protocol_name),
        "sources": sources,
        "configs": configs,
        "resolved_contract": _canonical_value(resolved_contract),
    }
    payload["bundle_sha256"] = canonical_json_sha256(payload)
    return payload


def bind_resume_identity(payload: Mapping[str, Any]) -> dict[str, object]:
    """Return a task config whose provenance fields are hash-bound.

    A caller-supplied stale digest is rejected instead of silently repaired.
    Database resume matching compares the complete JSON object, so any bound
    dataset, source, config, prompt, or tokenizer change creates a new identity.
    """

    normalized = _canonical_value(payload)
    if not isinstance(normalized, dict):
        raise TypeError("task sampling config must be a mapping")
    existing = normalized.pop("resume_identity_sha256", None)
    has_runtime_provenance = "runtime_attestation_provenance" in normalized
    has_provenance = (
        "dataset_snapshot" in normalized
        or "protocol_bundle" in normalized
        or has_runtime_provenance
    )
    if has_provenance:
        if "dataset_snapshot" not in normalized or "protocol_bundle" not in normalized:
            raise ValueError(
                "provenance-bound task configs require both dataset_snapshot and protocol_bundle"
            )
        normalized["resume_identity_schema_version"] = RESUME_IDENTITY_SCHEMA_VERSION
        digest = canonical_json_sha256(normalized)
        if existing is not None and existing != digest:
            raise ValueError(
                "stale resume_identity_sha256: task protocol or dataset content changed"
            )
        # Validate nested self-digests even if a caller has recomputed the
        # outer digest.  Existing stale identities are reported first because
        # they are the normal corruption/drift case; a coherently re-signed
        # forged nested object still fails here.
        _validate_dataset_snapshot(normalized["dataset_snapshot"])
        _validate_protocol_bundle(normalized["protocol_bundle"])
        if has_runtime_provenance:
            validate_runtime_attestation_provenance(
                normalized["runtime_attestation_provenance"]
            )
        normalized["resume_identity_sha256"] = digest
    elif existing is not None:
        raise ValueError(
            "resume_identity_sha256 requires dataset_snapshot and protocol_bundle"
        )
    return normalized


def _validate_sha256(value: Any, *, label: str) -> str:
    rendered = str(value or "")
    if len(rendered) != 64 or any(character not in "0123456789abcdef" for character in rendered):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return rendered


def validate_runtime_attestation_provenance(value: Any) -> dict[str, object]:
    """Validate the compact, task-bound proof of the live G1i runtime.

    The full root-owned artifacts remain in the frozen runtime.  This object
    records their transitive digests plus the most useful runtime identity
    fields in ``task.sampling_config``.  Its own digest makes it part of the
    database resume identity, so a task cannot silently resume against a
    different endpoint, weight, engine tree, GPU, approval, or protocol lock.
    """

    if not isinstance(value, Mapping):
        raise ValueError("runtime_attestation_provenance must be a mapping")
    expected_keys = {
        "schema_version",
        "model",
        "route_kind",
        "scheduler_endpoint",
        "global_approval_sha256",
        "global_approval_file_sha256",
        "protocol_lock_sha256",
        "protocol_lock_file_sha256",
        "runtime_attestation_evidence_sha256",
        "runtime_attestation_artifact_sha256",
        "forward_attestation_artifact_sha256",
        "host_label",
        "weight",
        "runtime_executable_sha256",
        "runtime_tree_sha256",
        "semantic_environment_sha256",
        "launch_parameters_sha256",
        "systemd_unit",
        "gpu_index",
        "provenance_sha256",
    }
    if set(value) != expected_keys:
        raise ValueError("runtime_attestation_provenance schema keys mismatch")
    if value.get("schema_version") != RUNTIME_ATTESTATION_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("runtime_attestation_provenance schema mismatch")
    model = str(value.get("model") or "")
    if not model.startswith("rwkv7-g1i-"):
        raise ValueError("runtime_attestation_provenance model is not G1i")
    route_kind = value.get("route_kind")
    if route_kind not in {"local", "ssh_forward"}:
        raise ValueError("runtime_attestation_provenance route_kind is invalid")
    if value.get("host_label") not in {"157", "8222"}:
        raise ValueError("runtime_attestation_provenance host_label is invalid")

    endpoint = value.get("scheduler_endpoint")
    if not isinstance(endpoint, Mapping) or set(endpoint) != {
        "scheme",
        "host",
        "port",
        "api_prefix",
    }:
        raise ValueError("runtime_attestation_provenance endpoint is invalid")
    if endpoint.get("scheme") != "http" or not str(endpoint.get("host") or ""):
        raise ValueError("runtime_attestation_provenance endpoint is invalid")
    try:
        endpoint_port = int(endpoint.get("port"))
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime_attestation_provenance endpoint port is invalid") from exc
    if not 1 <= endpoint_port <= 65535:
        raise ValueError("runtime_attestation_provenance endpoint port is invalid")
    api_prefix = str(endpoint.get("api_prefix") or "")
    if not api_prefix.startswith("/") or api_prefix.endswith("/"):
        raise ValueError("runtime_attestation_provenance endpoint prefix is invalid")

    for key in (
        "global_approval_sha256",
        "global_approval_file_sha256",
        "protocol_lock_sha256",
        "protocol_lock_file_sha256",
        "runtime_attestation_evidence_sha256",
        "runtime_attestation_artifact_sha256",
        "runtime_executable_sha256",
        "runtime_tree_sha256",
        "semantic_environment_sha256",
        "launch_parameters_sha256",
    ):
        _validate_sha256(value.get(key), label=f"runtime_attestation_provenance.{key}")
    forward_sha = value.get("forward_attestation_artifact_sha256")
    if route_kind == "ssh_forward":
        _validate_sha256(
            forward_sha,
            label="runtime_attestation_provenance.forward_attestation_artifact_sha256",
        )
        if value.get("host_label") != "8222":
            raise ValueError("ssh_forward runtime provenance must attest host 8222")
    elif forward_sha is not None:
        raise ValueError("local runtime provenance cannot contain a forward digest")

    weight = value.get("weight")
    if not isinstance(weight, Mapping) or set(weight) != {"path", "bytes", "sha256"}:
        raise ValueError("runtime_attestation_provenance weight is invalid")
    if not Path(str(weight.get("path") or "")).is_absolute():
        raise ValueError("runtime_attestation_provenance weight path is invalid")
    try:
        weight_bytes = int(weight.get("bytes"))
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime_attestation_provenance weight size is invalid") from exc
    if weight_bytes <= 0:
        raise ValueError("runtime_attestation_provenance weight size is invalid")
    _validate_sha256(
        weight.get("sha256"),
        label="runtime_attestation_provenance.weight.sha256",
    )
    if not str(value.get("systemd_unit") or "").endswith(".service"):
        raise ValueError("runtime_attestation_provenance systemd_unit is invalid")
    try:
        gpu_index = int(value.get("gpu_index"))
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime_attestation_provenance gpu_index is invalid") from exc
    if gpu_index < 0 or (
        value.get("host_label") == "8222" and gpu_index == 3
    ):
        raise ValueError("runtime_attestation_provenance gpu_index is forbidden")

    expected_digest = _validate_sha256(
        value.get("provenance_sha256"),
        label="runtime_attestation_provenance.provenance_sha256",
    )
    unsigned = dict(value)
    unsigned.pop("provenance_sha256", None)
    if canonical_json_sha256(unsigned) != expected_digest:
        raise ValueError("runtime_attestation_provenance self-digest mismatch")
    return dict(_canonical_value(value))


def _validate_file_descriptor(value: Any, *, label: str, fqcn: bool = False) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a file descriptor mapping")
    if value.get("path_kind") not in {"repo_relative", "absolute"}:
        raise ValueError(f"{label} has an invalid path_kind")
    if not str(value.get("path") or ""):
        raise ValueError(f"{label} has no path")
    try:
        size_bytes = int(value.get("size_bytes"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} has an invalid size") from exc
    if size_bytes < 0:
        raise ValueError(f"{label} has an invalid size")
    _validate_sha256(value.get("sha256"), label=f"{label}.sha256")
    if fqcn and not str(value.get("fqcn") or ""):
        raise ValueError(f"{label} has no callable identity")


def _validate_dataset_snapshot(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("dataset_snapshot must be a mapping")
    if value.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("dataset_snapshot schema mismatch")
    expected = _validate_sha256(value.get("snapshot_sha256"), label="dataset_snapshot.snapshot_sha256")
    unsigned = dict(value)
    unsigned.pop("snapshot_sha256", None)
    if canonical_json_sha256(unsigned) != expected:
        raise ValueError("dataset_snapshot self-digest mismatch")
    if not str(value.get("dataset_slug") or ""):
        raise ValueError("dataset_snapshot has no dataset_slug")
    if int(value.get("row_count") or 0) <= 0:
        raise ValueError("dataset_snapshot has no records")
    try:
        raw_size_bytes = int(value.get("raw_size_bytes"))
    except (TypeError, ValueError) as exc:
        raise ValueError("dataset_snapshot has an invalid raw size") from exc
    if raw_size_bytes < 0:
        raise ValueError("dataset_snapshot has an invalid raw size")
    _validate_sha256(value.get("raw_file_sha256"), label="dataset_snapshot.raw_file_sha256")
    _validate_sha256(
        value.get("canonical_records_sha256"),
        label="dataset_snapshot.canonical_records_sha256",
    )
    _validate_file_descriptor(value.get("loader"), label="dataset_snapshot.loader", fqcn=True)
    _validate_file_descriptor(value.get("resolver"), label="dataset_snapshot.resolver", fqcn=True)
    manifests = value.get("source_manifests")
    if not isinstance(manifests, list):
        raise ValueError("dataset_snapshot source_manifests must be a list")
    for index, descriptor in enumerate(manifests):
        _validate_file_descriptor(descriptor, label=f"dataset_snapshot.source_manifests[{index}]")


def _validate_protocol_bundle(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("protocol_bundle must be a mapping")
    if value.get("schema_version") != PROTOCOL_BUNDLE_SCHEMA_VERSION:
        raise ValueError("protocol_bundle schema mismatch")
    expected = _validate_sha256(value.get("bundle_sha256"), label="protocol_bundle.bundle_sha256")
    unsigned = dict(value)
    unsigned.pop("bundle_sha256", None)
    if canonical_json_sha256(unsigned) != expected:
        raise ValueError("protocol_bundle self-digest mismatch")
    if not str(value.get("protocol_name") or ""):
        raise ValueError("protocol_bundle has no protocol_name")
    sources = value.get("sources")
    configs = value.get("configs")
    if not isinstance(sources, list) or not sources:
        raise ValueError("protocol_bundle contains no source descriptors")
    if not isinstance(configs, list):
        raise ValueError("protocol_bundle configs must be a list")
    for collection_name, descriptors in (("sources", sources), ("configs", configs)):
        for index, descriptor in enumerate(descriptors):
            _validate_file_descriptor(
                descriptor,
                label=f"protocol_bundle.{collection_name}[{index}]",
            )


__all__ = [
    "PROTOCOL_BUNDLE_SCHEMA_VERSION",
    "RESUME_IDENTITY_SCHEMA_VERSION",
    "RUNTIME_ATTESTATION_PROVENANCE_SCHEMA_VERSION",
    "SNAPSHOT_SCHEMA_VERSION",
    "bind_resume_identity",
    "build_dataset_snapshot",
    "build_protocol_bundle",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "read_stable_file_bytes",
    "sha256_file",
    "validate_runtime_attestation_provenance",
]
