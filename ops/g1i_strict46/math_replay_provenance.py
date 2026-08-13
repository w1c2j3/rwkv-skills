"""Frozen provenance contract for append-only strict-46 Math replays.

The accepted answer-extractor lineage and the final imported evaluator are
different facts: the comparator repair lives in the same production module as
the extractor, so the module-file SHA changes even though the accepted
extractor lineage does not.  This module records and validates both facts, plus
the function-level comparator SHA and a cross-``PYTHONHASHSEED`` attestation.

No helper in this module writes the evaluation database.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


PROVENANCE_VERSION = "g1i_math_replay_v1"
ATTESTATION_SCHEMA_VERSION = "g1i-math-determinism-attestation.v1"
ACCEPTED_EXTRACTOR_LINEAGE_SHA256 = (
    "92955682cd7b83842f5e1483aaaba2abc747a6f8c11a52cb2a26b1d3e2a2aca0"
)
# Frozen after the generic comparator repair landed.  CLI/env injection remains
# available for a later release, but any override must still match the runtime
# module, comparator source and task git hash exactly.
FINAL_IMPORTED_FREE_RESPONSE_SHA256 = (
    "0eeee0d6a2304f14bb82e47386ceb8b8ac0e3f346042c7e43500d9cfe219c1db"
)
FINAL_COMPARATOR_IMPLEMENTATION_SHA256 = (
    "86b49ee7f74a71b1cdf2b8670d74a55a531ec9968062ed5af8706fe91ee9d44b"
)
# This is the frozen replay deployment HEAD on 157.  A developer checkout at a
# different HEAD must fail the runtime gate instead of minting remotely
# indistinguishable provenance.
FINAL_REPLAY_GIT_HASH = "f2ac09bb51feabe22fc0397c0e12c7e742481d52"
FINAL_MATH_VERIFY_VERSION = "0.9.0"
FINAL_REPLAY_PYTHONHASHSEED = "42"

SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_HASH_RE = re.compile(r"[0-9a-f]{7,64}")
REASON_TAG_RE = re.compile(r"[A-Za-z0-9_.-]+")


def _normalized_sha256(value: object) -> str:
    text = str(value or "").strip().lower()
    return text if SHA256_RE.fullmatch(text) else ""


def _normalized_git_hash(value: object) -> str:
    text = str(value or "").strip().lower()
    return text if GIT_HASH_RE.fullmatch(text) else ""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


@dataclass(frozen=True)
class FinalMathReplayContract:
    extractor_lineage_sha256: str
    imported_free_response_sha256: str
    comparator_implementation_sha256: str
    math_verify_version: str
    replay_git_hash: str
    reason_tag: str

    @classmethod
    def from_values(
        cls,
        *,
        extractor_lineage_sha256: object = ACCEPTED_EXTRACTOR_LINEAGE_SHA256,
        imported_free_response_sha256: object = FINAL_IMPORTED_FREE_RESPONSE_SHA256,
        comparator_implementation_sha256: object = (
            FINAL_COMPARATOR_IMPLEMENTATION_SHA256
        ),
        math_verify_version: object = FINAL_MATH_VERIFY_VERSION,
        replay_git_hash: object = FINAL_REPLAY_GIT_HASH,
        reason_tag: object = "",
    ) -> "FinalMathReplayContract":
        extractor = _normalized_sha256(extractor_lineage_sha256)
        imported = _normalized_sha256(imported_free_response_sha256)
        comparator = _normalized_sha256(comparator_implementation_sha256)
        git_hash = _normalized_git_hash(replay_git_hash)
        version = str(math_verify_version or "").strip()
        requested_tag = str(reason_tag or "").strip()
        derived_tag = (
            f"global_answer_extractor_{extractor[:12]}_comparator_{comparator[:12]}"
            if extractor and comparator
            else ""
        )
        return cls(
            extractor_lineage_sha256=extractor,
            imported_free_response_sha256=imported,
            comparator_implementation_sha256=comparator,
            math_verify_version=version,
            replay_git_hash=git_hash,
            reason_tag=requested_tag or derived_tag,
        )

    def blockers(self) -> list[str]:
        blockers: list[str] = []
        if not self.extractor_lineage_sha256:
            blockers.append("final_extractor_lineage_sha256_missing_or_invalid")
        if not self.imported_free_response_sha256:
            blockers.append("final_imported_free_response_sha256_missing_or_invalid")
        if not self.comparator_implementation_sha256:
            blockers.append("final_comparator_implementation_sha256_missing_or_invalid")
        if not self.math_verify_version:
            blockers.append("final_math_verify_version_missing")
        if not self.replay_git_hash:
            blockers.append("final_replay_git_hash_missing_or_invalid")
        if not self.reason_tag or not REASON_TAG_RE.fullmatch(self.reason_tag):
            blockers.append("final_reason_tag_missing_or_invalid")
        expected_tag = (
            f"global_answer_extractor_{self.extractor_lineage_sha256[:12]}"
            f"_comparator_{self.comparator_implementation_sha256[:12]}"
            if self.extractor_lineage_sha256
            and self.comparator_implementation_sha256
            else ""
        )
        if expected_tag and self.reason_tag != expected_tag:
            blockers.append(
                f"final_reason_tag:{self.reason_tag or 'empty'}!=expected:{expected_tag}"
            )
        return blockers

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeMathProvenance:
    imported_module_path: str
    imported_free_response_sha256: str
    comparator_implementation: str
    comparator_implementation_sha256: str
    math_verify_version: str
    pythonhashseed: str
    replay_git_hash: str
    blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _git_head(repo: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return _normalized_git_hash(completed.stdout)


def collect_runtime_math_provenance(
    *, repo: Path | None = None
) -> RuntimeMathProvenance:
    """Hash the evaluator Python file and the actual imported comparator."""

    blockers: list[str] = []
    module = importlib.import_module("src.eval.metrics.free_response")
    source_file = inspect.getsourcefile(module) or getattr(module, "__file__", "")
    module_path = Path(source_file).resolve() if source_file else None
    module_sha = ""
    if module_path is None or not module_path.is_file():
        blockers.append("imported_free_response_source_file_missing")
    else:
        module_sha = sha256_file(module_path)

    comparator_name = "src.eval.metrics.free_response._deterministic_math_verify"
    comparator = getattr(module, "_deterministic_math_verify", None)
    comparator_sha = ""
    if comparator is None or not callable(comparator):
        blockers.append("deterministic_math_verify_comparator_missing")
    else:
        try:
            comparator_source = inspect.getsource(comparator).encode("utf-8")
        except (OSError, TypeError):
            blockers.append("deterministic_math_verify_comparator_source_unavailable")
        else:
            comparator_sha = sha256_bytes(comparator_source)

    try:
        math_verify_version = metadata.version("math-verify")
    except metadata.PackageNotFoundError:
        math_verify_version = ""
        blockers.append("math_verify_distribution_missing")

    seed = str(os.environ.get("PYTHONHASHSEED") or "").strip()
    if not seed or not seed.isdigit():
        blockers.append("pythonhashseed_missing_or_non_numeric")

    repo_root = (repo or Path(__file__).resolve().parents[2]).resolve()
    git_hash = _git_head(repo_root)
    if not git_hash:
        blockers.append("replay_git_hash_unavailable")

    return RuntimeMathProvenance(
        imported_module_path=str(module_path or ""),
        imported_free_response_sha256=module_sha,
        comparator_implementation=comparator_name,
        comparator_implementation_sha256=comparator_sha,
        math_verify_version=math_verify_version,
        pythonhashseed=seed,
        replay_git_hash=git_hash,
        blockers=tuple(blockers),
    )


def runtime_contract_reasons(
    runtime: RuntimeMathProvenance,
    contract: FinalMathReplayContract,
) -> list[str]:
    reasons = list(contract.blockers())
    reasons.extend(runtime.blockers)
    comparisons = (
        (
            "imported_free_response_sha256",
            runtime.imported_free_response_sha256,
            contract.imported_free_response_sha256,
        ),
        (
            "comparator_implementation_sha256",
            runtime.comparator_implementation_sha256,
            contract.comparator_implementation_sha256,
        ),
        (
            "math_verify_version",
            runtime.math_verify_version,
            contract.math_verify_version,
        ),
        ("replay_git_hash", runtime.replay_git_hash, contract.replay_git_hash),
    )
    for label, actual, expected in comparisons:
        if actual and expected and actual != expected:
            reasons.append(f"{label}:{actual}!=expected:{expected}")
    return list(dict.fromkeys(reasons))


@dataclass(frozen=True)
class DeterminismAttestation:
    path: str
    sha256: str
    seeds: tuple[str, ...]
    task_result_sha256: dict[str, str]
    source_evidence_sha256_by_task: dict[str, str]
    judge_transcript_sha256_by_task: dict[str, str]
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "seeds": list(self.seeds),
            "task_result_sha256": dict(self.task_result_sha256),
            "source_evidence_sha256_by_task": dict(
                self.source_evidence_sha256_by_task
            ),
            "judge_transcript_sha256_by_task": dict(
                self.judge_transcript_sha256_by_task
            ),
            "schema_version": self.payload.get("schema_version"),
            "passed": self.payload.get("passed"),
        }


def _attestation_runtime_reasons(
    payload: dict[str, Any],
    contract: FinalMathReplayContract,
    runtime: RuntimeMathProvenance,
) -> list[str]:
    reasons: list[str] = []
    expected = {
        "extractor_lineage_sha256": contract.extractor_lineage_sha256,
        "imported_free_response_sha256": contract.imported_free_response_sha256,
        "comparator_implementation_sha256": contract.comparator_implementation_sha256,
        "math_verify_version": contract.math_verify_version,
        "replay_git_hash": contract.replay_git_hash,
        "reason_tag": contract.reason_tag,
    }
    for key, value in expected.items():
        actual = str(payload.get(key) or "")
        if actual != value:
            reasons.append(f"attestation.{key}:{actual or 'empty'}!=expected:{value}")
    if runtime.imported_free_response_sha256 != str(
        payload.get("imported_free_response_sha256") or ""
    ):
        reasons.append("attestation_runtime_imported_file_sha_mismatch")
    if runtime.comparator_implementation_sha256 != str(
        payload.get("comparator_implementation_sha256") or ""
    ):
        reasons.append("attestation_runtime_comparator_sha_mismatch")
    if runtime.math_verify_version != str(payload.get("math_verify_version") or ""):
        reasons.append("attestation_runtime_math_verify_version_mismatch")
    if runtime.replay_git_hash != str(payload.get("replay_git_hash") or ""):
        reasons.append("attestation_runtime_git_hash_mismatch")
    return reasons


def load_determinism_attestation(
    path: Path | None,
    *,
    contract: FinalMathReplayContract,
    runtime: RuntimeMathProvenance,
    source_task_ids: list[int],
) -> tuple[DeterminismAttestation | None, list[str]]:
    """Load and fail-closed validate a multi-seed replay attestation."""

    if path is None:
        return None, ["determinism_attestation_missing"]
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, [f"determinism_attestation_unreadable:{type(exc).__name__}"]
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, ["determinism_attestation_invalid_json"]
    if not isinstance(payload, dict):
        return None, ["determinism_attestation_not_object"]

    reasons: list[str] = []
    if payload.get("schema_version") != ATTESTATION_SCHEMA_VERSION:
        reasons.append("determinism_attestation_schema_mismatch")
    if payload.get("passed") is not True:
        reasons.append("determinism_attestation_not_passed")
    reasons.extend(_attestation_runtime_reasons(payload, contract, runtime))

    expected_ids = sorted(set(int(task_id) for task_id in source_task_ids))
    try:
        raw_attested_ids = [int(task_id) for task_id in payload["source_task_ids"]]
        attested_ids = sorted(set(raw_attested_ids))
    except (KeyError, TypeError, ValueError):
        raw_attested_ids = []
        attested_ids = []
        reasons.append("determinism_attestation_source_task_ids_invalid")
    if len(raw_attested_ids) != len(attested_ids):
        reasons.append("determinism_attestation_source_task_ids_duplicate")
    if attested_ids != expected_ids:
        reasons.append(
            f"determinism_attestation_source_task_ids:{attested_ids}"
            f"!=expected:{expected_ids}"
        )

    raw_source_evidence = payload.get("source_evidence_sha256_by_task")
    if not isinstance(raw_source_evidence, dict):
        source_evidence_by_task: dict[str, str] = {}
        reasons.append("determinism_attestation_source_evidence_missing")
    else:
        source_evidence_by_task = {
            str(key): str(value).lower()
            for key, value in raw_source_evidence.items()
        }
        if any(not key.isdigit() for key in source_evidence_by_task):
            reasons.append("determinism_attestation_source_evidence_task_id_invalid")
        numeric_source_evidence_ids = sorted(
            int(key) for key in source_evidence_by_task if key.isdigit()
        )
        if (
            numeric_source_evidence_ids != expected_ids
            or len(source_evidence_by_task) != len(expected_ids)
        ):
            reasons.append("determinism_attestation_source_evidence_task_ids_mismatch")
        if any(
            not SHA256_RE.fullmatch(value)
            for value in source_evidence_by_task.values()
        ):
            reasons.append("determinism_attestation_source_evidence_sha_invalid")

    raw_judge_transcripts = payload.get("judge_transcript_sha256_by_task", {})
    if not isinstance(raw_judge_transcripts, dict):
        judge_transcripts_by_task: dict[str, str] = {}
        reasons.append("determinism_attestation_judge_transcripts_invalid")
    else:
        judge_transcripts_by_task = {
            str(key): str(value).lower()
            for key, value in raw_judge_transcripts.items()
        }
        if any(not key.isdigit() for key in judge_transcripts_by_task):
            reasons.append(
                "determinism_attestation_judge_transcript_task_id_invalid"
            )
        judge_task_ids = {
            int(key) for key in judge_transcripts_by_task if key.isdigit()
        }
        if not judge_task_ids.issubset(set(expected_ids)):
            reasons.append(
                "determinism_attestation_judge_transcript_task_ids_mismatch"
            )
        if any(
            not SHA256_RE.fullmatch(value)
            for value in judge_transcripts_by_task.values()
        ):
            reasons.append("determinism_attestation_judge_transcript_sha_invalid")

    runs = payload.get("pythonhashseed_runs")
    if not isinstance(runs, list):
        runs = []
        reasons.append("determinism_attestation_runs_missing")
    seeds: list[str] = []
    result_maps: list[dict[str, str]] = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            reasons.append(f"determinism_attestation_run_{index}_not_object")
            continue
        seed = str(run.get("seed") or "").strip()
        if not seed.isdigit():
            reasons.append(f"determinism_attestation_run_{index}_seed_invalid")
            continue
        seeds.append(seed)
        raw_results = run.get("task_result_sha256")
        if not isinstance(raw_results, dict):
            reasons.append(f"determinism_attestation_run_{index}_results_missing")
            continue
        results = {str(key): str(value).lower() for key, value in raw_results.items()}
        if any(not key.isdigit() for key in results):
            reasons.append(
                f"determinism_attestation_run_{index}_task_id_key_invalid"
            )
        numeric_result_ids = sorted(
            int(key) for key in results if key.isdigit()
        )
        if numeric_result_ids != expected_ids or len(results) != len(expected_ids):
            reasons.append(f"determinism_attestation_run_{index}_task_ids_mismatch")
        if any(not SHA256_RE.fullmatch(value) for value in results.values()):
            reasons.append(f"determinism_attestation_run_{index}_result_sha_invalid")
        result_maps.append(results)
        raw_run_source_evidence = run.get("source_evidence_sha256_by_task")
        if not isinstance(raw_run_source_evidence, dict):
            reasons.append(
                f"determinism_attestation_run_{index}_source_evidence_missing"
            )
            continue
        run_source_evidence = {
            str(key): str(value).lower()
            for key, value in raw_run_source_evidence.items()
        }
        if run_source_evidence != source_evidence_by_task:
            reasons.append(
                f"determinism_attestation_run_{index}_source_evidence_mismatch"
            )
        raw_run_judge_transcripts = run.get(
            "judge_transcript_sha256_by_task", {}
        )
        if not isinstance(raw_run_judge_transcripts, dict):
            reasons.append(
                f"determinism_attestation_run_{index}_judge_transcripts_invalid"
            )
        else:
            run_judge_transcripts = {
                str(key): str(value).lower()
                for key, value in raw_run_judge_transcripts.items()
            }
            if run_judge_transcripts != judge_transcripts_by_task:
                reasons.append(
                    f"determinism_attestation_run_{index}_judge_transcripts_mismatch"
                )

    distinct_seeds = tuple(sorted(set(seeds), key=int)) if seeds else ()
    if len(distinct_seeds) < 4:
        reasons.append("determinism_attestation_requires_four_distinct_seeds")
    for required_seed in ("0", "1", "42"):
        if required_seed not in distinct_seeds:
            reasons.append(f"determinism_attestation_missing_seed:{required_seed}")
    if runtime.pythonhashseed and runtime.pythonhashseed not in distinct_seeds:
        reasons.append(
            f"runtime_pythonhashseed:{runtime.pythonhashseed}"
            "_not_in_determinism_attestation"
        )
    canonical_results = result_maps[0] if result_maps else {}
    if any(results != canonical_results for results in result_maps[1:]):
        reasons.append("determinism_attestation_cross_seed_result_mismatch")

    attestation = DeterminismAttestation(
        path=str(path.resolve()),
        sha256=sha256_bytes(raw),
        seeds=distinct_seeds,
        task_result_sha256=canonical_results,
        source_evidence_sha256_by_task=source_evidence_by_task,
        judge_transcript_sha256_by_task=judge_transcripts_by_task,
        payload=payload,
    )
    return attestation, list(dict.fromkeys(reasons))


def parse_task_desc(desc: object) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for part in str(desc or "").split(";"):
        key, separator, value = part.strip().partition("=")
        if separator and key and value:
            parsed[key] = value
    return parsed


def build_replay_task_desc(
    *,
    source_task_id: int,
    source_git_hash: str,
    contract: FinalMathReplayContract,
    runtime: RuntimeMathProvenance,
    attestation: DeterminismAttestation,
) -> str:
    source_evidence_sha256 = attestation.source_evidence_sha256_by_task.get(
        str(int(source_task_id)), ""
    )
    fields = [
        ("provenance_version", PROVENANCE_VERSION),
        ("replay_source_task_id", str(int(source_task_id))),
        ("reason_tag", contract.reason_tag),
        ("extractor_lineage_sha256", contract.extractor_lineage_sha256),
        ("imported_free_response_sha256", runtime.imported_free_response_sha256),
        ("comparator_implementation_sha256", runtime.comparator_implementation_sha256),
        ("math_verify_version", runtime.math_verify_version),
        ("determinism_attestation_sha256", attestation.sha256),
        ("pythonhashseed", runtime.pythonhashseed),
        ("replay_git_hash", runtime.replay_git_hash),
        ("source_git_hash", _normalized_git_hash(source_git_hash)),
        ("source_evidence_sha256", _normalized_sha256(source_evidence_sha256)),
    ]
    judge_transcript_sha256 = attestation.judge_transcript_sha256_by_task.get(
        str(int(source_task_id)), ""
    )
    if judge_transcript_sha256:
        fields.append(
            (
                "judge_transcript_sha256",
                _normalized_sha256(judge_transcript_sha256),
            )
        )
    if any(not value for _key, value in fields):
        missing = [key for key, value in fields if not value]
        raise ValueError(f"cannot build replay task desc; missing {','.join(missing)}")
    return ";".join(f"{key}={value}" for key, value in fields)


def evaluation_result_sha256(
    *,
    rows_by_group: dict[str, list[tuple[int, int, bool]]],
    payloads_by_group: dict[str, list[dict[str, Any]]],
) -> str:
    """Hash only deterministic scorer outputs, excluding timing/counters."""

    groups: dict[str, object] = {}
    for group in sorted(rows_by_group):
        rows = [
            [int(sample), int(repeat), bool(passed)]
            for sample, repeat, passed in rows_by_group[group]
        ]
        payloads = []
        for payload in payloads_by_group.get(group, []):
            payloads.append(
                {
                    "sample_index": int(payload.get("sample_index") or 0),
                    "repeat_index": int(payload.get("repeat_index") or 0),
                    "pass_index": int(payload.get("pass_index") or 0),
                    "answer": str(payload.get("answer") or ""),
                    "ref_answer": str(payload.get("ref_answer") or ""),
                    "is_passed": bool(payload.get("is_passed", False)),
                    "fail_reason": str(payload.get("fail_reason") or ""),
                }
            )
        groups[group] = {"rows": rows, "payloads": payloads}
    return canonical_json_sha256(groups)


__all__ = [
    "ACCEPTED_EXTRACTOR_LINEAGE_SHA256",
    "ATTESTATION_SCHEMA_VERSION",
    "DeterminismAttestation",
    "FINAL_COMPARATOR_IMPLEMENTATION_SHA256",
    "FINAL_IMPORTED_FREE_RESPONSE_SHA256",
    "FINAL_MATH_VERIFY_VERSION",
    "FINAL_REPLAY_PYTHONHASHSEED",
    "FINAL_REPLAY_GIT_HASH",
    "FinalMathReplayContract",
    "PROVENANCE_VERSION",
    "RuntimeMathProvenance",
    "build_replay_task_desc",
    "canonical_json_sha256",
    "collect_runtime_math_provenance",
    "evaluation_result_sha256",
    "load_determinism_attestation",
    "parse_task_desc",
    "runtime_contract_reasons",
    "sha256_file",
]
