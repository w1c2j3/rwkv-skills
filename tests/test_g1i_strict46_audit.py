from __future__ import annotations

import fcntl
import json
import runpy
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest


AUDIT_PATH = (
    Path(__file__).resolve().parents[1] / "ops" / "g1i_strict46" / "audit_current.py"
)
AUDIT = runpy.run_path(str(AUDIT_PATH), run_name="g1i_strict46_audit_test")
MODEL = "rwkv7-g1i-7.2b-20260805-ctx16384"
POST_RAW_COMPLETIONS_FIX = datetime(2026, 8, 6, 6)


class _BulkAuditCursor:
    def __init__(self, rows_by_query: dict[str, list[dict[str, object]]]) -> None:
        self.rows_by_query = rows_by_query
        self.executions: list[tuple[str, tuple[object, ...]]] = []
        self._current_query = ""

    def execute(self, query: str, params: tuple[object, ...]) -> None:
        self._current_query = query
        self.executions.append((query, params))

    def fetchall(self) -> list[dict[str, object]]:
        return self.rows_by_query.get(self._current_query, [])


def _task(benchmark: str, split: str, domain: str) -> dict[str, str]:
    return {
        "benchmark_name": benchmark,
        "benchmark_split": split,
        "domain": domain,
        "model_name": MODEL,
    }


def _sampling_stages(task: dict[str, str]) -> dict[str, object]:
    return AUDIT["_expected_sampling_stages"](task)


def test_global_audit_lock_blocks_a_second_open_descriptor(tmp_path: Path) -> None:
    lock_path = tmp_path / "g1i-audit.lock"

    with AUDIT["_exclusive_audit_lock"](lock_path):
        with lock_path.open("a+", encoding="utf-8") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)

    # The kernel releases the lock when the context exits, so a later monitor
    # or waiter can acquire it instead of leaving a stale PID-file gate.
    with lock_path.open("a+", encoding="utf-8") as contender:
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(contender, fcntl.LOCK_UN)


@pytest.mark.parametrize("same_output", [True, False])
def test_global_audit_lock_serializes_processes_and_keeps_json_atomic(
    tmp_path: Path,
    same_output: bool,
) -> None:
    lock_path = tmp_path / "g1i-audit.lock"
    first_output = tmp_path / "shared.json"
    second_output = first_output if same_output else tmp_path / "second.json"
    worker = """
import json
from pathlib import Path
import runpy
import sys
import time

audit = runpy.run_path(sys.argv[1], run_name="g1i_audit_lock_worker")
lock_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
token = sys.argv[4]
delay = float(sys.argv[5])
with audit["_exclusive_audit_lock"](lock_path):
    print(json.dumps({"event": "start", "token": token, "time": time.monotonic()}), flush=True)
    time.sleep(delay)
    audit["_atomic_write_text"](
        output_path,
        json.dumps({"token": token}, sort_keys=True) + "\\n",
    )
    print(json.dumps({"event": "end", "token": token, "time": time.monotonic()}), flush=True)
"""

    first = subprocess.Popen(
        [
            sys.executable,
            "-c",
            worker,
            str(AUDIT_PATH),
            str(lock_path),
            str(first_output),
            "first",
            "0.25",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert first.stdout is not None
    first_start_line = first.stdout.readline()
    assert first_start_line
    second = subprocess.Popen(
        [
            sys.executable,
            "-c",
            worker,
            str(AUDIT_PATH),
            str(lock_path),
            str(second_output),
            "second",
            "0.05",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    first_rest, first_stderr = first.communicate(timeout=10)
    second_stdout, second_stderr = second.communicate(timeout=10)
    assert first.returncode == 0, first_stderr
    assert second.returncode == 0, second_stderr

    first_events = [
        json.loads(line) for line in (first_start_line + first_rest).splitlines()
    ]
    second_events = [json.loads(line) for line in second_stdout.splitlines()]
    assert [event["event"] for event in first_events] == ["start", "end"]
    assert [event["event"] for event in second_events] == ["start", "end"]
    assert second_events[0]["time"] >= first_events[1]["time"]

    assert json.loads(first_output.read_text(encoding="utf-8")) == {
        "token": "second" if same_output else "first"
    }
    if not same_output:
        assert json.loads(second_output.read_text(encoding="utf-8")) == {
            "token": "second"
        }


def test_content_audit_candidates_cover_only_rows_with_audit_semantics() -> None:
    rows = [
        {
            "task_id": 1,
            "benchmark_name": "mmlu",
            "benchmark_split": "test",
            "status": "Completed",
            "score_created_at": POST_RAW_COMPLETIONS_FIX,
            "completion_count": 16,
        },
        {
            "task_id": 2,
            "benchmark_name": "math_500",
            "benchmark_split": "test",
            "status": "Running",
            "score_created_at": None,
            "completion_count": 0,
        },
        {
            "task_id": 3,
            "benchmark_name": "mmlu",
            "benchmark_split": "test",
            "status": "Failed",
            "score_created_at": None,
            "completion_count": 0,
        },
        {
            "task_id": 4,
            "benchmark_name": "mmlu",
            "benchmark_split": "test",
            "status": "Failed",
            "score_created_at": None,
            "completion_count": 3,
        },
        {
            "task_id": 5,
            "benchmark_name": "gpqa",
            "benchmark_split": "main",
            "status": "Completed",
            "score_created_at": None,
            "completion_count": 0,
        },
        {
            "task_id": 6,
            "benchmark_name": "ifeval",
            "benchmark_split": "test",
            "status": "Completed",
            "score_created_at": None,
            "completion_count": 541,
        },
        {
            "task_id": 7,
            "benchmark_name": "not_strict46",
            "benchmark_split": "test",
            "status": "Completed",
            "score_created_at": POST_RAW_COMPLETIONS_FIX,
            "completion_count": 100,
        },
        {
            "task_id": 8,
            "benchmark_name": "simpleqa",
            "benchmark_split": "verified",
            "status": "Completed",
            "score_created_at": POST_RAW_COMPLETIONS_FIX,
            "completion_count": 4000,
        },
        # Duplicate metadata rows must not repeat a task in the ANY array.
        {
            "task_id": 1,
            "benchmark_name": "mmlu",
            "benchmark_split": "test",
            "status": "Completed",
            "score_created_at": POST_RAW_COMPLETIONS_FIX,
            "completion_count": 16,
        },
    ]

    candidates = AUDIT["_content_audit_candidate_task_ids"](
        rows,
        diagnostic_knowledge_replay_ids={5},
    )

    # Scored, Running, diagnostic replay, physical-split aliases, and partial
    # Failed tasks remain auditable.  Empty old failures, scoreless Completed
    # rows, and out-of-matrix tasks cannot affect a protocol gate or signal.
    assert candidates == [1, 2, 4, 5, 8]


def test_content_task_partitions_balance_completion_weight_deterministically() -> None:
    rows = [
        {"task_id": 1, "completion_count": 100},
        {"task_id": 2, "completion_count": 90},
        {"task_id": 3, "completion_count": 20},
        {"task_id": 4, "completion_count": 10},
    ]

    partitions = AUDIT["_partition_content_task_ids"](
        rows,
        [1, 2, 3, 4],
        max_workers=2,
    )

    assert partitions == [[1, 4], [2, 3]]


def test_task_loader_parallelizes_content_only_and_merges_exactly(monkeypatch) -> None:
    task_query = AUDIT["TASK_QUERY"]
    coordinate_query = AUDIT["COMPLETION_COORDINATE_STATS_QUERY"]
    eval_query = AUDIT["EVAL_STATS_QUERY"]
    task_rows = [
        {
            "task_id": task_id,
            "model_name": MODEL,
            "benchmark_name": "mmlu",
            "benchmark_split": "test",
            "status": "Completed",
            "score_created_at": POST_RAW_COMPLETIONS_FIX,
            "sampling_config": {"prompt_profile": "naive"},
            "metrics": {"avg@16": 0.5},
        }
        for task_id in range(1, 5)
    ]
    weights = {1: 100, 2: 90, 3: 20, 4: 10}
    coordinate_rows = [
        {
            "task_id": task_id,
            "completion_count": completion_count,
            "distinct_completion_coordinates": completion_count,
        }
        for task_id, completion_count in weights.items()
    ]
    cursor = _BulkAuditCursor(
        {
            task_query: task_rows,
            coordinate_query: coordinate_rows,
            eval_query: [],
        }
    )
    queried_partitions: list[list[int]] = []

    def fake_partition_query(
        _conninfo: str,
        task_ids: list[int],
    ) -> list[dict[str, int]]:
        queried_partitions.append(task_ids)
        return [
            {"task_id": task_id, "blank_raw_count": task_id} for task_id in task_ids
        ]

    monkeypatch.setitem(
        AUDIT["_load_task_rows"].__globals__,
        "_query_content_stats_partition",
        fake_partition_query,
    )

    rows = AUDIT["_load_task_rows"](
        cursor,
        content_conninfo="host=read-only-profile",
        content_stats_workers=2,
    )

    assert queried_partitions == [[1, 4], [2, 3]]
    assert [row["blank_raw_count"] for row in rows] == [1, 2, 3, 4]
    assert [query for query, _params in cursor.executions] == [
        task_query,
        coordinate_query,
        eval_query,
    ]


def test_task_loader_preserves_left_join_defaults_without_wide_grouping() -> None:
    task_query = AUDIT["TASK_QUERY"]
    coordinate_query = AUDIT["COMPLETION_COORDINATE_STATS_QUERY"]
    content_query = AUDIT["COMPLETION_CONTENT_STATS_QUERY"]
    eval_query = AUDIT["EVAL_STATS_QUERY"]
    cursor = _BulkAuditCursor(
        {
            task_query: [
                {
                    "task_id": 21,
                    "model_name": MODEL,
                    "benchmark_name": "mmlu",
                    "benchmark_split": "test",
                    "status": "Completed",
                    "score_created_at": POST_RAW_COMPLETIONS_FIX,
                    "sampling_config": {"prompt_profile": "naive"},
                    "metrics": {"avg@16": 0.5},
                },
                {
                    "task_id": 22,
                    "model_name": MODEL,
                    "benchmark_name": "ifeval",
                    "benchmark_split": "test",
                    "status": "Completed",
                    "score_created_at": None,
                    "sampling_config": {},
                    "metrics": None,
                },
            ],
            coordinate_query: [
                {
                    "task_id": 21,
                    "completion_count": 16,
                    "distinct_completion_coordinates": 16,
                    "distinct_sample_indices": 1,
                    "min_sample_index": 0,
                    "max_sample_index": 0,
                    "distinct_avg_repeat_indices": 16,
                    "min_avg_repeat_index": 0,
                    "max_avg_repeat_index": 15,
                }
            ],
            content_query: [
                {
                    "task_id": 21,
                    "blank_raw_count": 2,
                    "blank_recovery_stage_count": 3,
                }
            ],
            eval_query: [
                {
                    "task_id": 21,
                    "eval_count": 16,
                    "passed_eval_count": 8,
                    "legacy_missing_prediction_count": 1,
                    "missing_recovery_prediction_count": 2,
                }
            ],
        }
    )

    rows = AUDIT["_load_task_rows"](cursor)

    assert rows[0]["sampling_config"] == {"prompt_profile": "naive"}
    assert rows[0]["metrics"] == {"avg@16": 0.5}
    assert rows[0]["completion_count"] == 16
    assert rows[0]["passed_eval_count"] == 8
    assert rows[0]["blank_raw_count"] == 2
    assert rows[0]["blank_recovery_stage_count"] == 3
    assert rows[0]["legacy_missing_prediction_count"] == 1
    assert rows[0]["missing_recovery_prediction_count"] == 2
    assert rows[0]["missing_prediction_count"] == 1
    assert rows[1]["completion_count"] == 0
    assert rows[1]["eval_count"] == 0
    assert rows[1]["distinct_completion_coordinates"] == 1
    assert rows[1]["min_sample_index"] is None
    assert rows[1]["max_avg_repeat_index"] is None
    assert [query for query, _params in cursor.executions] == [
        task_query,
        coordinate_query,
        eval_query,
        content_query,
    ]
    assert cursor.executions[0][1] == (list(AUDIT["MODELS"]),)
    assert cursor.executions[1][1] == ([21, 22],)
    assert cursor.executions[2][1] == ([21, 22],)
    assert cursor.executions[3][1] == (
        list(AUDIT["STAGE0_FINAL_TRUNCATION_EVALUATORS"]),
        [21],
    )
    assert "JOIN completions" not in task_query
    assert "JOIN eval" not in task_query
    assert "GROUP BY" not in task_query
    assert "GROUP BY c.task_id" in coordinate_query
    assert "JOIN eval" not in content_query
    assert "c.context" not in coordinate_query
    assert "c.context" not in eval_query


def test_completion_evidence_bulk_loader_preserves_per_task_maps() -> None:
    raw_query = AUDIT["RAW_BATCH_QUERY"]
    prompt_query = AUDIT["PROMPT_BATCH_QUERY"]
    truncation_query = AUDIT["TRUNCATION_EXAMPLE_BATCH_QUERY"]
    cursor = _BulkAuditCursor(
        {
            raw_query: [
                {"task_id": 10, "raw_values": ["B", "A"]},
                {"task_id": 12, "raw_values": [None]},
                {"task_id": 13, "raw_values": ["C"]},
            ],
            prompt_query: [
                {"task_id": 10, "prompt": "prompt-10"},
                {"task_id": 11, "prompt": "prompt-11"},
                {"task_id": 12, "prompt": None},
            ],
            truncation_query: [
                {
                    "task_id": 10,
                    "sample_index": 1,
                    "avg_repeat_index": 0,
                    "pass_index": 0,
                    "completion_tail": "tail-1",
                },
                {
                    "task_id": 10,
                    "sample_index": 3,
                    "avg_repeat_index": 0,
                    "pass_index": 0,
                    "completion_tail": "tail-3",
                },
            ],
        }
    )
    rows = [
        {
            "task_id": 10,
            "benchmark_name": "mmlu",
            "benchmark_split": "test",
            "status": "Completed",
            "score_created_at": POST_RAW_COMPLETIONS_FIX,
            "overall_truncation_count": 2,
        },
        {
            "task_id": 11,
            "benchmark_name": "ifeval",
            "benchmark_split": "test",
            "status": "Completed",
            "score_created_at": POST_RAW_COMPLETIONS_FIX,
        },
        {
            "task_id": 12,
            "benchmark_name": "gpqa",
            "benchmark_split": "main",
            "status": "Running",
            "score_created_at": None,
        },
        # TASK_QUERY may return the same task more than once if historical
        # score rows exist.  The old maps overwrote identical reads; bulk mode
        # must query and retain the task only once.
        {
            "task_id": 10,
            "benchmark_name": "mmlu",
            "benchmark_split": "test",
            "status": "Completed",
            "score_created_at": POST_RAW_COMPLETIONS_FIX,
            "overall_truncation_count": 2,
        },
        {
            "task_id": 13,
            "benchmark_name": "mmlu_pro",
            "benchmark_split": "test",
            "status": "Failed",
            "score_created_at": None,
        },
    ]

    raw, prompts, truncations = AUDIT["_load_completion_audit_maps"](
        cursor,
        rows,
        diagnostic_knowledge_replay_ids={13},
    )

    assert raw == {10: ["B", "A"], 12: [None], 13: ["C"]}
    assert prompts == {
        10: "prompt-10",
        11: "prompt-11",
        12: "",
        13: "",
    }
    assert truncations == {
        10: [
            {
                "sample_index": 1,
                "avg_repeat_index": 0,
                "pass_index": 0,
                "completion_tail": "tail-1",
            },
            {
                "sample_index": 3,
                "avg_repeat_index": 0,
                "pass_index": 0,
                "completion_tail": "tail-3",
            },
        ]
    }
    assert [query for query, _params in cursor.executions] == [
        raw_query,
        prompt_query,
        truncation_query,
    ]
    assert cursor.executions[0][1] == ([10, 12, 13],)
    assert cursor.executions[1][1] == ([10, 11, 12, 13],)
    assert cursor.executions[2][1] == ([10],)


def test_completion_evidence_queries_are_constant_count_and_per_task_limited() -> None:
    cursor = _BulkAuditCursor({})
    rows = [
        {
            "task_id": task_id,
            "benchmark_name": "mmlu",
            "benchmark_split": "test",
            "status": "Completed",
            "score_created_at": POST_RAW_COMPLETIONS_FIX,
            "overall_truncation_count": 1,
        }
        for task_id in range(1, 501)
    ]

    AUDIT["_load_completion_audit_maps"](cursor, rows)

    assert len(cursor.executions) == 3
    assert "ARRAY_AGG(" in AUDIT["RAW_BATCH_QUERY"]
    assert "ORDER BY c.sample_index" in AUDIT["RAW_BATCH_QUERY"]
    assert "LEFT JOIN LATERAL" in AUDIT["PROMPT_BATCH_QUERY"]
    assert "LIMIT 1" in AUDIT["PROMPT_BATCH_QUERY"]
    truncation_query = AUDIT["TRUNCATION_EXAMPLE_BATCH_QUERY"]
    assert "CROSS JOIN LATERAL" in truncation_query
    assert "LIMIT 5" in truncation_query


def test_simpleqa_physical_verified_split_maps_to_strict46_logical_test() -> None:
    row = {
        "benchmark_name": "simpleqa",
        "benchmark_split": "verified",
    }

    benchmark = AUDIT["canonicalize_task_benchmark"](row)

    assert benchmark == ("simpleqa", "test")
    assert row["benchmark_name"] == "simpleqa"
    assert row["benchmark_split"] == "test"
    assert row["source_benchmark_name"] == "simpleqa"
    assert row["source_benchmark_split"] == "verified"


def test_simpleqa_alias_protocol_lookups_use_physical_source_split() -> None:
    row = {
        "benchmark_name": "simpleqa",
        "benchmark_split": "verified",
        "domain": "math",
        "model_name": MODEL,
        "benchmark_num_samples": 1000,
    }
    AUDIT["canonicalize_task_benchmark"](row)

    assert AUDIT["_task_source_slug"](row) == "simpleqa_verified"
    assert AUDIT["_expected_evaluator"](row) == "free_response_naive"
    assert AUDIT["_expected_avg_k"](row) == 8.0
    assert AUDIT["_expected_effective_sample_count"](row) == 8000
    stages = AUDIT["_expected_sampling_stages"](row)
    assert stages["stage1"]["max_new_tokens"] == 12288
    assert stages["stage2"]["max_new_tokens"] == 128
    assert stages["strategy_a"]["max_new_tokens"] == 12288


def test_protocol_lookups_fall_back_to_logical_fields_without_provenance() -> None:
    row = _task("amc23", "test", "math")

    assert AUDIT["_task_source_slug"](row) == "amc23_test"
    assert AUDIT["_expected_evaluator"](row) == "free_response_judge_naive"


def test_valid_candidate_selection_retains_every_superseded_task_id() -> None:
    cell = ("model", "mmlu", "test")
    latest: dict[tuple[str, str, str], dict[str, object]] = {}
    superseded: list[dict[str, object]] = []

    for task_id in (20, 10, 30):
        AUDIT["_record_valid_candidate"](
            latest,
            superseded,
            cell,
            {"task_id": task_id},
        )

    assert latest[cell]["task_id"] == 30
    assert sorted(row["task_id"] for row in superseded) == [10, 20]


def test_all_replay_marker_versions_are_diagnostic_provenance_only() -> None:
    marker = AUDIT["_is_diagnostic_knowledge_replay_row"]

    assert marker({"diagnostic_only": True})
    assert marker({"knowledge_replay_diagnostic_evidence": True})
    assert marker({"replay_eligible_except_cutoff": True})
    assert not marker({})
    assert not marker({"diagnostic_only": False})


def test_strict46_expected_runner_families() -> None:
    expected_evaluator = AUDIT["_expected_evaluator"]

    assert (
        expected_evaluator(_task("gpqa", "main", "knowledge"))
        == "multi_choice_plain_naive"
    )
    assert (
        expected_evaluator(_task("amc23", "test", "math"))
        == "free_response_judge_naive"
    )
    assert (
        expected_evaluator(_task("math_500", "test", "math")) == "free_response_naive"
    )
    assert (
        expected_evaluator(_task("livecodebench", "test", "coding"))
        == "code_livecodebench_plain_naive"
    )
    assert (
        expected_evaluator(_task("ifbench", "test", "instruction_following"))
        == "instruction_following_naive"
    )


def test_strict46_stage0_final_truncation_families_exclude_math_and_choice() -> None:
    evaluators = set(AUDIT["STAGE0_FINAL_TRUNCATION_EVALUATORS"])

    assert evaluators == {
        "code_human_eval_naive",
        "code_mbpp_naive",
        "code_livecodebench_plain_naive",
        "instruction_following_naive",
    }
    assert "free_response_naive" not in evaluators
    assert "free_response_judge_naive" not in evaluators
    assert "multi_choice_plain_naive" not in evaluators
    assert "t.evaluator = ANY(%s)" in AUDIT["COMPLETION_CONTENT_STATS_QUERY"]


def test_strict46_audits_blank_recovery_as_explicit_failed_prediction() -> None:
    content_query = AUDIT["COMPLETION_CONTENT_STATS_QUERY"]
    eval_query = AUDIT["EVAL_STATS_QUERY"]
    predicate = AUDIT["BLANK_RECOVERY_STAGE_SQL_PREDICATE"]

    assert "AS blank_recovery_stage_count" in content_query
    assert predicate in content_query
    assert "jsonb_array_length(c.context->'stages') > 1" in predicate
    assert "BTRIM(COALESCE(c.context #>> '{stages,1,completion}', '')) = ''" in predicate
    assert "missing_recovery_prediction" in eval_query
    assert "c.context" not in eval_query
    assert AUDIT["COMPLETION_STAT_DEFAULTS"]["blank_recovery_stage_count"] == 0
    assert (
        AUDIT["COMPLETION_STAT_DEFAULTS"]["missing_recovery_prediction_count"]
        == 0
    )
    assert (
        AUDIT["COMPLETION_STAT_DEFAULTS"][
            "blank_recovery_strategy_a_inheritance_count"
        ]
        == 0
    )


def test_strict46_audits_blank_strategy_a_as_explicit_failed_prediction() -> None:
    content_query = AUDIT["COMPLETION_CONTENT_STATS_QUERY"]
    eval_query = AUDIT["STRATEGY_A_EVAL_STATS_QUERY"]

    assert "AS blank_strategy_a_generation_count" in content_query
    assert "jsonb_typeof(c.context->'strategy_a') = 'object'" in content_query
    assert "BTRIM(COALESCE(c.context #>> '{strategy_a,completion}', '')) = ''" in content_query
    assert "User✿|Bot✿" in content_query
    assert "missing_strategy_a_prediction" in eval_query
    assert AUDIT["COMPLETION_STAT_DEFAULTS"]["blank_strategy_a_generation_count"] == 0
    assert AUDIT["COMPLETION_STAT_DEFAULTS"]["missing_strategy_a_prediction_count"] == 0


def test_blank_strategy_a_eval_loader_uses_companion_task() -> None:
    query = AUDIT["STRATEGY_A_EVAL_STATS_QUERY"]
    cursor = _BulkAuditCursor(
        {
            query: [
                {
                    "task_id": 21,
                    "missing_strategy_a_prediction_count": 2,
                }
            ]
        }
    )
    rows = [
        {
            "task_id": 21,
            "blank_strategy_a_generation_count": 2,
            "missing_strategy_a_prediction_count": 0,
            "metrics": {"strategy_task_ids": {"strategy_a": 121}},
        },
        {
            "task_id": 22,
            "blank_strategy_a_generation_count": 0,
            "missing_strategy_a_prediction_count": 0,
            "metrics": {"strategy_task_ids": {"strategy_a": 122}},
        },
        {
            "task_id": 23,
            "blank_strategy_a_generation_count": 1,
            "missing_strategy_a_prediction_count": 0,
            "metrics": {},
        },
        # Historical duplicate score rows for one parent must not multiply
        # companion eval counts; the newest (largest) companion id wins.
        {
            "task_id": 21,
            "blank_strategy_a_generation_count": 2,
            "missing_strategy_a_prediction_count": 0,
            "metrics": {"strategy_task_ids": {"strategy_a": 120}},
        },
    ]

    AUDIT["_load_strategy_a_eval_stats"](cursor, rows)

    assert cursor.executions == [(query, ([21], [121]))]
    assert rows[0]["missing_strategy_a_prediction_count"] == 2
    assert rows[1]["missing_strategy_a_prediction_count"] == 0
    assert rows[2]["missing_strategy_a_prediction_count"] == 0


def test_blank_recovery_inheritance_loader_requires_linked_companion_task() -> None:
    query = AUDIT["BLANK_RECOVERY_INHERITANCE_STATS_QUERY"]
    cursor = _BulkAuditCursor(
        {
            query: [
                {
                    "task_id": 21,
                    "blank_recovery_strategy_a_inheritance_count": 1,
                }
            ]
        }
    )
    rows = [
        {
            "task_id": 21,
            "blank_recovery_stage_count": 1,
            "blank_recovery_strategy_a_inheritance_count": 0,
            "metrics": {"strategy_task_ids": {"strategy_a": 121}},
        },
        {
            "task_id": 22,
            "blank_recovery_stage_count": 0,
            "blank_recovery_strategy_a_inheritance_count": 0,
            "metrics": {"strategy_task_ids": {"strategy_a": 122}},
        },
        {
            "task_id": 23,
            "blank_recovery_stage_count": 1,
            "blank_recovery_strategy_a_inheritance_count": 0,
            "metrics": {},
        },
        # Duplicate score rows cannot multiply the coordinate-level join; use
        # the newest companion task id just like the blank-A evidence loader.
        {
            "task_id": 21,
            "blank_recovery_stage_count": 1,
            "blank_recovery_strategy_a_inheritance_count": 0,
            "metrics": {"strategy_task_ids": {"strategy_a": 120}},
        },
    ]

    AUDIT["_load_blank_recovery_inheritance_stats"](cursor, rows)

    assert cursor.executions == [(query, ([21], [121]))]
    assert rows[0]["blank_recovery_strategy_a_inheritance_count"] == 1
    assert rows[1]["blank_recovery_strategy_a_inheritance_count"] == 0
    assert rows[2]["blank_recovery_strategy_a_inheritance_count"] == 0


def test_blank_recovery_inheritance_query_is_coordinate_and_eval_exact() -> None:
    query = AUDIT["BLANK_RECOVERY_INHERITANCE_STATS_QUERY"]

    assert "parent_completion.context" in query
    assert "parent_completion.sample_index" in query
    assert "parent_completion.avg_repeat_index" in query
    assert "parent_completion.pass_index" in query
    assert "parent_eval.is_passed IS TRUE" in query
    assert "strategy_a_eval.is_passed IS TRUE" in query
    assert "BTRIM(COALESCE(strategy_a_eval.answer, '')) <> ''" in query
    assert "parent_eval.answer IS NOT DISTINCT FROM strategy_a_eval.answer" in query
    assert (
        "parent_eval.ref_answer IS NOT DISTINCT FROM strategy_a_eval.ref_answer"
        in query
    )
    assert "COALESCE(parent_eval.fail_reason, '') = ''" in query
    assert "COALESCE(strategy_a_eval.fail_reason, '') = ''" in query


def test_blank_recovery_is_explained_by_missing_or_verified_a_inheritance() -> None:
    reasons = AUDIT["_blank_recovery_protocol_reasons"]
    post_fix = {
        "task_created_at": datetime(2026, 8, 7, 18),
        "blank_recovery_stage_count": 1,
    }

    assert reasons(
        {
            **post_fix,
            "missing_recovery_prediction_count": 1,
            "blank_recovery_strategy_a_inheritance_count": 0,
        }
    ) == []
    assert reasons(
        {
            **post_fix,
            "missing_recovery_prediction_count": 0,
            "blank_recovery_strategy_a_inheritance_count": 1,
        }
    ) == []
    assert reasons(
        {
            **post_fix,
            "blank_recovery_stage_count": 2,
            "missing_recovery_prediction_count": 1,
            "blank_recovery_strategy_a_inheritance_count": 1,
        }
    ) == []
    assert reasons(
        {
            **post_fix,
            "missing_recovery_prediction_count": 0,
            "blank_recovery_strategy_a_inheritance_count": 0,
        }
    ) == ["blank_recovery_eval_mismatch:raw=1,missing=0,inherited_a=0"]
    # Over-counting is also rejected: evidence cannot explain more raw blank
    # coordinates than actually exist.
    assert reasons(
        {
            **post_fix,
            "missing_recovery_prediction_count": 1,
            "blank_recovery_strategy_a_inheritance_count": 1,
        }
    ) == ["blank_recovery_eval_mismatch:raw=1,missing=1,inherited_a=1"]


def test_strict46_blank_strategy_a_requires_raw_eval_agreement() -> None:
    reasons = AUDIT["_blank_strategy_a_protocol_reasons"]

    assert reasons({}) == []
    assert reasons(
        {
            "blank_strategy_a_generation_count": 2,
            "missing_strategy_a_prediction_count": 2,
        }
    ) == []
    assert reasons(
        {
            "blank_strategy_a_generation_count": 2,
            "missing_strategy_a_prediction_count": 1,
        }
    ) == ["blank_strategy_a_eval_mismatch:raw=2,eval=1"]
    assert reasons(
        {
            "blank_strategy_a_generation_count": 0,
            "missing_strategy_a_prediction_count": 1,
        }
    ) == ["blank_strategy_a_eval_mismatch:raw=0,eval=1"]
    assert reasons(
        {
            "blank_strategy_a_generation_count": 2,
            "missing_strategy_a_prediction_count": 0,
        },
        require_eval_match=False,
    ) == []


def test_strict46_blank_recovery_cutoff_forces_old_tasks_to_replay(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "RWKV_BENCHMARK_CONFIG_ROOT",
        str(AUDIT["STRICT_CONFIG_ROOT"]),
    )
    cutoff = datetime(2026, 8, 7, 12)
    monkeypatch.setitem(
        AUDIT["_general_protocol_reasons"].__globals__,
        "BLANK_RECOVERY_PROTOCOL_DEPLOYED_AT",
        cutoff,
    )
    task = {
        **_task("math_500", "test", "math"),
        "status": "Completed",
        "evaluator": "free_response_naive",
        "benchmark_num_samples": 500,
        "sampling_config": {
            "avg_k": 8,
            "effective_sample_count": 4000,
            "prompt_profile": "naive",
            "sample_limit": None,
            "sampling_config": _sampling_stages(
                _task("math_500", "test", "math")
            ),
        },
        "completion_count": 4000,
        "eval_count": 4000,
        "distinct_completion_coordinates": 4000,
        "distinct_sample_indices": 500,
        "min_sample_index": 0,
        "max_sample_index": 499,
        "distinct_avg_repeat_indices": 8,
        "min_avg_repeat_index": 0,
        "max_avg_repeat_index": 7,
        "passed_eval_count": 2000,
        "missing_prediction_count": 0,
        "blank_recovery_stage_count": 0,
        "metrics": {"avg@8": 0.5},
        "task_created_at": POST_RAW_COMPLETIONS_FIX,
    }

    # The cutoff never invalidates an otherwise valid task with zero blank
    # recovery rows.
    assert AUDIT["_general_protocol_reasons"](task) == []

    historical_blank = {
        **task,
        "blank_recovery_stage_count": 7,
    }
    reasons = AUDIT["_general_protocol_reasons"](historical_blank)
    assert "blank_recovery_stage_predates_missing_protocol_fix:7" in reasons
    active_reasons = AUDIT["_active_protocol_reasons"](
        historical_blank,
        "User: solve\n\nAssistant: <think",
        [],
    )
    assert "blank_recovery_stage_predates_missing_protocol_fix:7" in active_reasons

    post_deploy_blank = {
        **historical_blank,
        "task_created_at": datetime(2026, 8, 7, 12, 0, 1),
        "missing_recovery_prediction_count": 7,
    }
    reasons = AUDIT["_general_protocol_reasons"](post_deploy_blank)
    assert not any(
        reason.startswith("blank_recovery_stage_predates_missing_protocol_fix:")
        for reason in reasons
    )
    assert "missing_prediction:7" not in reasons
    assert not any(
        reason.startswith("blank_recovery_eval_mismatch:")
        for reason in reasons
    )

    # The raw completion and evaluator evidence must agree before a completed
    # task is accepted.  This catches both an old scorer running after the
    # deployment boundary and any future persistence drift.
    post_deploy_mismatch = {
        **post_deploy_blank,
        "missing_recovery_prediction_count": 6,
    }
    reasons = AUDIT["_general_protocol_reasons"](post_deploy_mismatch)
    assert (
        "blank_recovery_eval_mismatch:raw=7,missing=6,inherited_a=0"
        in reasons
    )

    eval_without_raw = {
        **post_deploy_blank,
        "blank_recovery_stage_count": 0,
    }
    reasons = AUDIT["_general_protocol_reasons"](eval_without_raw)
    assert (
        "blank_recovery_eval_mismatch:raw=0,missing=7,inherited_a=0"
        in reasons
    )

    # Eval rows are persisted only once generation completes, so a post-fix
    # active task must not be reported as mismatched merely because its eval
    # phase has not started yet.
    active_post_deploy = {
        **post_deploy_blank,
        "status": "Running",
        "missing_recovery_prediction_count": 0,
    }
    active_reasons = AUDIT["_active_protocol_reasons"](
        active_post_deploy,
        "User: solve\n\nAssistant: <think",
        [],
    )
    assert not any(
        reason.startswith("blank_recovery_eval_mismatch:")
        for reason in active_reasons
    )


def test_strict46_root_judge_tasks_before_deterministic_cutoff_require_replay() -> None:
    cutoff = AUDIT["JUDGE_DETERMINISM_DEPLOYED_AT"]
    historical = {
        **_task("amc23", "test", "math"),
        "evaluator": "free_response_judge_naive",
        "task_created_at": cutoff.replace(microsecond=0),
        "sampling_config": {"cot_mode": "CoT", "prompt_profile": "naive"},
    }
    historical["task_created_at"] = historical["task_created_at"].replace(second=37)

    assert "judge_sampling_predates_deterministic_fix" in AUDIT[
        "_general_protocol_reasons"
    ](historical)
    assert "judge_sampling_predates_deterministic_fix" in AUDIT[
        "_active_protocol_reasons"
    ](historical, "User: solve\n\nAssistant: <think", [])


def test_strict46_post_cutoff_root_judge_and_auxiliary_rows_pass_determinism_gate() -> None:
    cutoff = AUDIT["JUDGE_DETERMINISM_DEPLOYED_AT"]
    post_cutoff = {
        **_task("amc23", "test", "math"),
        "evaluator": "free_response_judge_naive",
        "task_created_at": cutoff,
        "sampling_config": {"cot_mode": "CoT", "prompt_profile": "naive"},
    }
    strategy_a = {
        **post_cutoff,
        "task_created_at": cutoff.replace(second=37),
        "evaluator": "free_response_judge_naive:strategy_a",
    }
    exact_match = {
        **strategy_a,
        "evaluator": "free_response_naive",
    }

    for task in (post_cutoff, strategy_a, exact_match):
        assert AUDIT["_judge_determinism_protocol_reasons"](task) == []
        assert "judge_sampling_predates_deterministic_fix" not in AUDIT[
            "_general_protocol_reasons"
        ](task)
        assert "judge_sampling_predates_deterministic_fix" not in AUDIT[
            "_active_protocol_reasons"
        ](task, "User: solve\n\nAssistant: <think", [])


def test_strict46_post_cutoff_root_judge_without_protocol_fingerprint_is_invalid() -> None:
    cutoff = AUDIT["JUDGE_DETERMINISM_DEPLOYED_AT"]
    task = {
        **_task("amc23", "test", "math"),
        "evaluator": "free_response_judge_naive",
        "task_created_at": cutoff,
        "sampling_config": {
            "cot_mode": "CoT",
            "prompt_profile": "naive",
            "judger_model_name": "judge-model",
        },
        "metrics": {
            "judge_stats": {
                "total": 1,
                "parsed_count": 1,
                "invalid_output_count": 0,
                "request_error_count": 0,
                "error_count": 0,
            }
        },
    }

    reasons = AUDIT["_general_protocol_reasons"](task)

    assert any(
        reason.startswith("judge_protocol_missing_fields:") for reason in reasons
    )


def test_judgement_output_audit_uses_generated_final_stage_not_prompt() -> None:
    mismatch = AUDIT["_judgement_output_source_mismatch"]
    context = {
        "stages": [
            {
                "prompt": "Allowed outputs: Judgement: Yes or Judgement: No",
                "completion": "reasoning",
            },
            {"prompt": "final", "completion": "Yes"},
        ]
    }

    assert not mismatch(
        {
            "context": context,
            "answer": "Judgement: Yes",
            "ref_answer": "Judgement: Yes",
        }
    )
    assert mismatch(
        {
            "context": context,
            "answer": "Judgement: No",
            "ref_answer": "Judgement: Yes",
        }
    )


def test_judgement_output_audit_mirrors_strategy_a_inheritance() -> None:
    mismatch = AUDIT["_judgement_output_source_mismatch"]
    context = {
        "strategy_a": {
            "prompt": "Allowed outputs: Judgement: Yes or Judgement: No",
            "completion": "No",
        },
        "stages": [
            {"prompt": "reasoning", "completion": "reasoning"},
            {"prompt": "final", "completion": "Yes"},
        ],
    }

    # A already matches the reference, so evaluate_free_response inherits A
    # into the primary C lane instead of replacing it with C's different label.
    assert not mismatch(
        {
            "context": context,
            "answer": "Judgement: No",
            "ref_answer": "Judgement: No",
        }
    )
    assert mismatch(
        {
            "context": context,
            "answer": "Judgement: Yes",
            "ref_answer": "Judgement: No",
        }
    )


def test_judgement_output_audit_allows_consistent_missing_label() -> None:
    mismatch = AUDIT["_judgement_output_source_mismatch"]
    context = {
        "strategy_a": {"prompt": "judge", "completion": "unclear"},
        "stages": [
            {"prompt": "reasoning", "completion": "unclear"},
            {"prompt": "final", "completion": "unknown"},
        ],
    }

    assert not mismatch(
        {
            "context": context,
            "answer": "unknown",
            "ref_answer": "Judgement: Yes",
        }
    )


def test_strict46_rejects_judgement_output_source_mismatch(monkeypatch) -> None:
    monkeypatch.setenv(
        "RWKV_BENCHMARK_CONFIG_ROOT",
        str(AUDIT["STRICT_CONFIG_ROOT"]),
    )
    task = {
        **_task("answer_judge", "test", "math"),
        "status": "Completed",
        "evaluator": "free_response_naive",
        "benchmark_num_samples": 200,
        "sampling_config": {
            "avg_k": 8,
            "effective_sample_count": 1600,
            "prompt_profile": "naive",
            "sample_limit": None,
            "sampling_config": _sampling_stages(_task("answer_judge", "test", "math")),
        },
        "completion_count": 1600,
        "eval_count": 1600,
        "distinct_completion_coordinates": 1600,
        "distinct_sample_indices": 200,
        "min_sample_index": 0,
        "max_sample_index": 199,
        "distinct_avg_repeat_indices": 8,
        "min_avg_repeat_index": 0,
        "max_avg_repeat_index": 7,
        "passed_eval_count": 800,
        "metrics": {"avg@8": 0.5},
        "task_created_at": POST_RAW_COMPLETIONS_FIX,
        "judgement_output_source_mismatch_count": 1,
    }

    assert "judgement_output_source_mismatch:1" in AUDIT["_general_protocol_reasons"](
        task
    )


def test_strict46_expected_avg_k_comes_from_approved_config(monkeypatch) -> None:
    monkeypatch.setenv(
        "RWKV_BENCHMARK_CONFIG_ROOT",
        str(AUDIT["STRICT_CONFIG_ROOT"]),
    )
    expected_avg_k = AUDIT["_expected_avg_k"]

    assert expected_avg_k(_task("gpqa", "main", "knowledge")) == 16.0
    assert expected_avg_k(_task("amc23", "test", "math")) == 64.0
    assert expected_avg_k(_task("math_500", "test", "math")) == 8.0
    assert expected_avg_k(_task("livecodebench", "test", "coding")) == 4.0
    assert expected_avg_k(_task("ifbench", "test", "instruction_following")) == 16.0


def test_strict46_rejects_limited_or_self_reported_partial_coverage(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "RWKV_BENCHMARK_CONFIG_ROOT",
        str(AUDIT["STRICT_CONFIG_ROOT"]),
    )
    protocol_reasons = AUDIT["_general_protocol_reasons"]
    task = {
        **_task("math_500", "test", "math"),
        "status": "Completed",
        "evaluator": "free_response_naive",
        "benchmark_num_samples": 500,
        "sampling_config": {
            "avg_k": 8,
            "effective_sample_count": 4000,
            "prompt_profile": "naive",
            "sample_limit": None,
            "sampling_config": _sampling_stages(_task("math_500", "test", "math")),
        },
        "completion_count": 4000,
        "eval_count": 4000,
        "distinct_completion_coordinates": 4000,
        "distinct_sample_indices": 500,
        "min_sample_index": 0,
        "max_sample_index": 499,
        "distinct_avg_repeat_indices": 8,
        "min_avg_repeat_index": 0,
        "max_avg_repeat_index": 7,
        "passed_eval_count": 2000,
        "metrics": {"avg@8": 0.5},
        "task_created_at": POST_RAW_COMPLETIONS_FIX,
    }
    assert protocol_reasons(task) == []

    task["sampling_config"] = {
        **task["sampling_config"],
        "effective_sample_count": 800,
        "sample_limit": 100,
    }
    task["completion_count"] = 800
    task["eval_count"] = 800
    task["distinct_completion_coordinates"] = 800
    task["distinct_sample_indices"] = 100
    task["max_sample_index"] = 99
    task["passed_eval_count"] = 400
    reasons = protocol_reasons(task)
    assert "sample_limit:100" in reasons
    assert "effective_sample_count:800!=expected:4000" in reasons
    assert "completion_count:800!=expected:4000" in reasons


def test_strict46_rejects_duplicate_or_missing_completion_coordinates(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "RWKV_BENCHMARK_CONFIG_ROOT",
        str(AUDIT["STRICT_CONFIG_ROOT"]),
    )
    task = {
        **_task("math_500", "test", "math"),
        "status": "Completed",
        "evaluator": "free_response_naive",
        "benchmark_num_samples": 500,
        "sampling_config": {
            "avg_k": 8,
            "effective_sample_count": 4000,
            "prompt_profile": "naive",
            "sample_limit": None,
            "sampling_config": _sampling_stages(_task("math_500", "test", "math")),
        },
        "completion_count": 4000,
        "eval_count": 4000,
        "distinct_completion_coordinates": 3999,
        "distinct_sample_indices": 500,
        "min_sample_index": 0,
        "max_sample_index": 499,
        "distinct_avg_repeat_indices": 8,
        "min_avg_repeat_index": 0,
        "max_avg_repeat_index": 7,
        "passed_eval_count": 2000,
        "metrics": {"avg@8": 0.5},
        "task_created_at": POST_RAW_COMPLETIONS_FIX,
    }
    assert "distinct_completion_coordinates:3999!=completions:4000" in AUDIT[
        "_general_protocol_reasons"
    ](task)


def test_strict46_domain_prompt_protocols() -> None:
    scored = {"score_created_at": datetime(2026, 8, 6)}
    coding_protocol_ok = AUDIT["_coding_protocol_ok"]
    math_protocol_ok = AUDIT["_math_protocol_ok"]

    assert coding_protocol_ok(
        scored,
        "User: implement\n\nAssistant: <think></think>\n```python\n",
    )
    assert coding_protocol_ok(
        scored,
        "User: implement\n\nAssistant: <think>\n</think>\n```python",
    )
    assert not coding_protocol_ok(
        scored,
        "User: implement\n\nAssistant: <think",
    )
    assert math_protocol_ok(
        scored,
        "User: solve\n\nAssistant: <think",
    )
    assert not math_protocol_ok(
        scored,
        "User: solve\n\nAssistant: <think></think>",
    )


def test_strict46_rejects_sampling_drift_across_runner_families(monkeypatch) -> None:
    monkeypatch.setenv(
        "RWKV_BENCHMARK_CONFIG_ROOT",
        str(AUDIT["STRICT_CONFIG_ROOT"]),
    )
    sampling_reasons = AUDIT["_sampling_protocol_reasons"]
    cases = (
        _task("math_500", "test", "math"),
        _task("human_eval", "test", "coding"),
        _task("livecodebench", "test", "coding"),
        _task("ifeval", "test", "instruction_following"),
    )
    for task in cases:
        stages = _sampling_stages(task)
        complete = {**task, "sampling_config": {"sampling_config": stages}}
        assert sampling_reasons(complete) == []

        first_stage = next(iter(stages))
        drifted_stages = {key: dict(value) for key, value in stages.items()}
        drifted_stages[first_stage]["max_new_tokens"] = 17
        drifted = {
            **task,
            "sampling_config": {"sampling_config": drifted_stages},
        }
        assert any(
            reason.startswith(f"sampling:{first_stage}.max_new_tokens:")
            for reason in sampling_reasons(drifted)
        )


def test_strict46_math_strategy_a_allows_think_close_but_staged_cot_does_not(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "RWKV_BENCHMARK_CONFIG_ROOT",
        str(AUDIT["STRICT_CONFIG_ROOT"]),
    )
    stages = _sampling_stages(_task("aime24", "test", "math"))

    assert stages["stage1"]["bad_words"] == ["</think>"]
    assert stages["stage1"]["min_think_tokens"] == 16
    assert "bad_words" not in stages["strategy_a"]
    assert "min_think_tokens" not in stages["strategy_a"]


def test_strict46_flags_active_prompt_and_sampling_drift(monkeypatch) -> None:
    monkeypatch.setenv(
        "RWKV_BENCHMARK_CONFIG_ROOT",
        str(AUDIT["STRICT_CONFIG_ROOT"]),
    )
    active_reasons = AUDIT["_active_protocol_reasons"]
    task = _task("ifeval", "test", "instruction_following")
    stages = _sampling_stages(task)
    running = {
        **task,
        "sampling_config": {
            "cot_mode": "NoCoT",
            "sampling_config": stages,
        },
        "task_created_at": POST_RAW_COMPLETIONS_FIX,
    }
    assert (
        active_reasons(
            running,
            "User: comply\n\nAssistant: <think></think>\n",
            [],
        )
        == []
    )
    reasons = active_reasons(running, "User: comply\n\nAssistant:", [])
    assert "instruction_nocot_empty_think_protocol" in reasons

    drifted_stages = {key: dict(value) for key, value in stages.items()}
    drifted_stages["stage1"]["max_new_tokens"] = 16
    drifted = {
        **running,
        "sampling_config": {
            "cot_mode": "NoCoT",
            "sampling_config": drifted_stages,
        },
    }
    assert any(
        reason.startswith("sampling:stage1.max_new_tokens:")
        for reason in active_reasons(
            drifted,
            "User: comply\n\nAssistant: <think></think>\n",
            [],
        )
    )


def test_strict46_rejects_generation_before_raw_completions_fix(monkeypatch) -> None:
    monkeypatch.setenv(
        "RWKV_BENCHMARK_CONFIG_ROOT",
        str(AUDIT["STRICT_CONFIG_ROOT"]),
    )
    task = _task("ifeval", "test", "instruction_following")
    stages = _sampling_stages(task)
    running = {
        **task,
        "sampling_config": {
            "cot_mode": "NoCoT",
            "sampling_config": stages,
        },
        "task_created_at": datetime(2026, 8, 6, 5, 9, 59),
    }
    reasons = AUDIT["_active_protocol_reasons"](
        running,
        "User: comply\n\nAssistant: <think></think>\n",
        [],
    )
    assert "generation_predates_raw_completions_protocol_fix" in reasons


def test_historical_knowledge_replay_is_diagnostic_only_for_final_coverage(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "RWKV_BENCHMARK_CONFIG_ROOT",
        str(AUDIT["STRICT_CONFIG_ROOT"]),
    )
    base = _task("mmlu", "test", "knowledge")
    avg_k = int(AUDIT["_expected_avg_k"](base))
    task_id = 28460
    historical = {
        **base,
        "task_id": task_id,
        "status": "Completed",
        "evaluator": "multi_choice_plain_naive",
        "benchmark_num_samples": 1,
        "sampling_config": {
            "avg_k": avg_k,
            "effective_sample_count": avg_k,
            "prompt_profile": "naive",
            "sample_limit": None,
        },
        "completion_count": avg_k,
        "eval_count": avg_k,
        "distinct_completion_coordinates": avg_k,
        "distinct_sample_indices": 1,
        "min_sample_index": 0,
        "max_sample_index": 0,
        "distinct_avg_repeat_indices": avg_k,
        "min_avg_repeat_index": 0,
        "max_avg_repeat_index": avg_k - 1,
        "passed_eval_count": avg_k,
        "metrics": {f"avg@{avg_k}": 1.0},
        "task_created_at": datetime(2026, 8, 6, 5, 9, 59),
        "score_created_at": datetime(2026, 8, 6, 5, 10),
    }

    assert AUDIT["_knowledge_protocol_ok"](
        historical,
        ["A"] * avg_k,
        "User: choose one\n\nAssistant: <think></think>\nThe answer is",
    )

    reasons, has_diagnostic_replay = AUDIT["_final_protocol_assessment"](
        historical,
        {task_id},
    )

    assert has_diagnostic_replay is True
    assert reasons == ["generation_predates_raw_completions_protocol_fix"]

    current = {
        **historical,
        "task_created_at": POST_RAW_COMPLETIONS_FIX,
    }
    current_reasons, current_has_diagnostic_replay = AUDIT[
        "_final_protocol_assessment"
    ](current, {task_id})
    assert current_has_diagnostic_replay is True
    assert current_reasons == []


def test_strict46_rejects_leading_orphan_close_for_any_domain(monkeypatch) -> None:
    monkeypatch.setenv(
        "RWKV_BENCHMARK_CONFIG_ROOT",
        str(AUDIT["STRICT_CONFIG_ROOT"]),
    )
    task = _task("ifeval", "test", "instruction_following")
    stages = _sampling_stages(task)
    running = {
        **task,
        "sampling_config": {
            "cot_mode": "NoCoT",
            "sampling_config": stages,
        },
        "task_created_at": POST_RAW_COMPLETIONS_FIX,
        "leading_orphan_close_count": 3,
    }
    reasons = AUDIT["_active_protocol_reasons"](
        running,
        "User: comply\n\nAssistant: <think></think>\n",
        [],
    )
    assert "leading_orphan_close:3" in reasons
