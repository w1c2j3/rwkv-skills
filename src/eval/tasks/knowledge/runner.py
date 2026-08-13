from __future__ import annotations

"""Field-oriented knowledge runner aligned with rwkv-rs knowledge datasets."""

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Sequence

from src.eval.benchmark_registry import CoTMode
from src.eval.common.field_runner import (
    add_common_runner_args,
    attempt_stage,
    resolve_prompt_profile,
    scoring_stage,
)
from src.eval.context_budget import add_long_doc_cli_args, long_doc_config_payload, resolve_long_doc_config
from src.eval.field_common import (
    build_plan_task_details,
    build_task_sampling_config,
    resolve_configured_k_plan,
    set_task_env,
)
from src.eval.k_values import filter_metrics_by_k
from src.eval.naive_prompt_protocol import (
    NAIVE_COT_ASSISTANT_PREFIX,
    NAIVE_NOCOT_ASSISTANT_PREFIX,
)
from src.infer.backend import (
    add_inference_backend_arguments,
    build_inference_backend_from_args,
    require_completion_style_remote_protocol,
    resolve_backend_model_name,
    validate_inference_backend_args,
)

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext, TaskSpec
    from src.infer.sampling import SamplingConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV knowledge benchmark runner")
    add_common_runner_args(parser, batch_size_default=64, db_write_queue_default=4096)
    add_inference_backend_arguments(parser)
    parser.add_argument("--target-token-format", default=" <LETTER>", help="Token format for answer tokens")
    parser.add_argument(
        "--cot-mode",
        choices=[mode.value for mode in CoTMode],
        default=CoTMode.NO_COT.value,
        help="Prompt mode for knowledge benchmarks",
    )
    parser.add_argument(
        "--prompt-profile",
        choices=("normal", "naive"),
        help="Prompt profile: normal uses benchmark configs/defaults; naive uses only question/options plus answer prefill.",
    )
    parser.add_argument("--prompt-max-chars", type=int, help="Clamp rendered prompt length")
    add_long_doc_cli_args(parser)
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Run a single-batch CoT probe and skip scoring",
    )
    parser.add_argument(
        "--allow-generative-mc-fallback",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pass-k",
        type=int,
        action="append",
        help="pass@k values to compute (default: none; can be set in configs/<benchmark>.toml)",
    )
    parser.add_argument(
        "--run-checker",
        action="store_true",
        help="Run the optional wrong-answer checker before writing scores.",
    )
    return parser.parse_args(argv)


def _env_enabled(name: str) -> bool:
    raw = os.getenv(name, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _should_run_checker(args: argparse.Namespace) -> bool:
    if _env_enabled("RWKV_SKILLS_DISABLE_CHECKER") or _env_enabled("DISABLE_CHECKER"):
        return False
    if bool(getattr(args, "run_checker", False)):
        return True
    return _env_enabled("RWKV_KNOWLEDGE_RUN_CHECKER")


def _default_job_name(cot_mode: CoTMode) -> str:
    if cot_mode is CoTMode.NO_COT:
        return "multi_choice_plain"
    return "multi_choice_cot"


def _resolve_knowledge_cot_strategy(root_config: object | None) -> str:
    override = os.environ.get("RWKV_KNOWLEDGE_COT_STRATEGY", "").strip()
    strategy = override or getattr(root_config, "knowledge_cot_strategy", None) or "cascade_a_b"
    if strategy not in {"two_stage", "cascade_a_b"}:
        raise ValueError(f"unsupported knowledge CoT strategy: {strategy}")
    return strategy


def _naive_direct_prompt_template() -> str:
    return f"User: <Q>\n<CHOICES>\n\nAssistant: {NAIVE_NOCOT_ASSISTANT_PREFIX}"


def _naive_cot_prompt_template() -> str:
    return f"User: <Q>\n<CHOICES>\n\nAssistant: {NAIVE_COT_ASSISTANT_PREFIX}"


def _naive_final_answer_template() -> str:
    return "<Q><COT>\nFinal answer (option letter only):"


def _print_done_message(cot_mode: CoTMode, sample_count: int) -> None:
    if cot_mode is CoTMode.NO_COT:
        print(f"✅ direct multiple-choice done: {sample_count} samples")
        return
    print(f"✅ CoT multiple-choice done: {sample_count} samples")


def _resolve_cot_sampling_config(slug: str, model_name: str) -> "SamplingConfig | None":
    from src.eval.benchmark_config import resolve_sampling_config

    return resolve_sampling_config(
        slug,
        model_name,
        stage="cot",
        fallback_templates="multi_choice_cot_default",
    )


def _task_sampling_config(
    cot_mode: CoTMode,
    *,
    avg_k: float,
    effective_sample_count: int,
    cot_sampling: object | None = None,
    pass_ks: Sequence[int] | None = None,
    prompt_profile: str = "normal",
    long_doc_config: object | None = None,
    prompt_max_chars: int | None = None,
    choice_sampling_protocol: dict[str, object] | None = None,
    dataset_snapshot: dict[str, object] | None = None,
    protocol_bundle: dict[str, object] | None = None,
) -> dict[str, object]:
    from src.eval.results.schema import sampling_config_to_dict

    sampling_payload: dict[str, object] = {}
    if cot_mode is CoTMode.COT:
        sampling_payload["stage1"] = sampling_config_to_dict(cot_sampling)
    if long_doc_config is not None and getattr(long_doc_config, "enabled", False):
        sampling_payload["long_doc"] = long_doc_config_payload(long_doc_config)  # type: ignore[arg-type]
    if prompt_max_chars is not None:
        sampling_payload["prompt_max_chars"] = int(prompt_max_chars)
    if choice_sampling_protocol is not None:
        sampling_payload["choice_answer"] = choice_sampling_protocol
    return build_task_sampling_config(
        cot_mode=cot_mode,
        avg_k=avg_k,
        sampling_config=sampling_payload,
        pass_ks=pass_ks,
        effective_sample_count=effective_sample_count,
        prompt_profile=prompt_profile,
        dataset_snapshot=dataset_snapshot,
        protocol_bundle=protocol_bundle,
    )


def _build_cascade_metrics_payload(
    evaluation,
    *,
    pass_k,
    avg_k,
    report_pass_k,
    report_avg_k,
) -> tuple[dict[str, object], dict[str, object]]:
    from src.eval.metrics.at_k import compute_avg_score_at_k, compute_pass_at_k

    strategy_metrics: dict[str, dict[str, float]] = {}
    primary_pass_all: dict[str, float] = {}
    primary_avg_all: dict[str, float] = {}
    primary_pass_payload: dict[str, float] = {}
    primary_avg_payload: dict[str, float] = {}
    for group in ("strategy_a", "strategy_b"):
        group_metrics = dict(evaluation.metrics_by_group[group])
        pass_all = compute_pass_at_k(evaluation.rows_by_group[group], pass_k)
        avg_all = compute_avg_score_at_k(evaluation.score_rows_by_group[group], avg_k)
        pass_payload = filter_metrics_by_k(pass_all, report_pass_k, "pass@")
        avg_payload = filter_metrics_by_k(avg_all, report_avg_k, "avg@")
        if report_pass_k and not pass_payload:
            pass_payload = pass_all
        if report_avg_k and not avg_payload:
            avg_payload = avg_all
        group_metrics.update(pass_payload)
        group_metrics.update(avg_payload)
        strategy_metrics[group] = group_metrics
        if group == evaluation.primary_group:
            primary_pass_all = pass_all
            primary_avg_all = avg_all
            primary_pass_payload = pass_payload
            primary_avg_payload = avg_payload

    metrics_payload: dict[str, object] = dict(strategy_metrics[evaluation.primary_group])
    metrics_payload["strategy_metrics"] = strategy_metrics
    metrics_payload["strategy_diagnostics"] = {
        "rerouted": strategy_metrics["strategy_b"].get("rerouted", 0.0),
        "rescued": strategy_metrics["strategy_b"].get("rescued", 0.0),
        "rescue_rate": strategy_metrics["strategy_b"].get("rescue_rate", 0.0),
    }
    details: dict[str, object] = {}
    if primary_pass_all and primary_pass_payload != primary_pass_all:
        details["pass_curve"] = primary_pass_all
    if primary_avg_all and primary_avg_payload != primary_avg_all:
        details["avg_curve"] = primary_avg_all
    return metrics_payload, details


def main(
    argv: Sequence[str] | None = None,
    *,
    run_context: "RunContext | None" = None,
    task_spec: "TaskSpec | None" = None,
) -> int:
    del task_spec
    args = parse_args(argv)
    validate_inference_backend_args(args)

    from src.eval.benchmark_config import (
        benchmark_config_source_paths,
        resolve_benchmark_model_config,
    )
    from src.eval.datasets.data_loader.multiple_choice import JsonlMultipleChoiceLoader
    from src.eval.datasets.snapshot import build_dataset_snapshot, build_protocol_bundle
    from src.eval.evaluating import TaskRunController, TaskRunState, prepare_task_execution
    from src.eval.execution_plan import build_attempt_keys, plan_attempt_count
    from src.eval.tasks.knowledge.pipeline import MultipleChoicePipeline
    from src.eval.metrics.at_k import compute_avg_at_k, compute_avg_score_at_k, compute_pass_at_k
    from src.eval.metrics.multi_choice import (
        evaluate_multiple_choice,
        evaluate_multiple_choice_cascade,
    )
    from src.eval.results.payloads import make_score_payload
    from src.eval.scheduler.config import DEFAULT_DB_CONFIG
    from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
    from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
    from src.db.database import init_db
    from src.db.eval_db_service import EvalDbService

    cot_mode = CoTMode(args.cot_mode)
    job_name = run_context.job_name if run_context is not None else os.environ.get("RWKV_SKILLS_JOB_NAME", _default_job_name(cot_mode))
    prompt_profile = resolve_prompt_profile(args.prompt_profile, job_name)
    if args.probe_only and cot_mode is not CoTMode.COT:
        raise ValueError("--probe-only is only supported with --cot-mode cot")
    completion_style_remote = require_completion_style_remote_protocol(
        args,
        benchmark_name="multiple-choice knowledge benchmarks",
    )

    dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    slug = infer_dataset_slug_from_path(str(dataset_path))
    model_name = resolve_backend_model_name(args)
    dataset_records = JsonlMultipleChoiceLoader(str(dataset_path)).load()
    k_plan = resolve_configured_k_plan(
        slug=slug,
        model_name=model_name,
        dataset_len=len(dataset_records),
        args=args,
    )
    plan = k_plan.plan
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    backend = build_inference_backend_from_args(args)
    pipeline = MultipleChoicePipeline(
        backend,
        target_token_format=args.target_token_format,
    )
    choice_sampling_protocol = pipeline.resolve_choice_sampling_protocol(
        [len(record.choices) for record in dataset_records]
    )
    direct_config = resolve_benchmark_model_config(slug, model_name, stage="direct")
    cot_config = resolve_benchmark_model_config(slug, model_name, stage="cot")
    final_config = resolve_benchmark_model_config(slug, model_name, stage="final")
    root_config = resolve_benchmark_model_config(slug, model_name, stage=None)
    long_doc_config = resolve_long_doc_config(args, root_config)
    prompt_max_chars = args.prompt_max_chars if args.prompt_max_chars is not None else getattr(root_config, "prompt_max_chars", None)
    knowledge_cot_strategy = _resolve_knowledge_cot_strategy(root_config)
    missing_prediction_score = getattr(root_config, "missing_prediction_score", None)
    if missing_prediction_score is not None and not 0.0 <= missing_prediction_score <= 1.0:
        raise ValueError("missing_prediction_score must be within [0, 1]")
    if prompt_profile == "naive":
        direct_prompt_template = _naive_direct_prompt_template()
        cot_prompt_template = _naive_cot_prompt_template()
        final_answer_template = _naive_final_answer_template()
    else:
        direct_prompt_template = (
            direct_config.direct_prompt_template
            if direct_config is not None and direct_config.direct_prompt_template
            else None
        )
        cot_prompt_template = (
            cot_config.cot_prompt_template
            if cot_config is not None and cot_config.cot_prompt_template
            else None
        )
        final_answer_template = (
            final_config.final_prompt_template
            if final_config is not None and final_config.final_prompt_template
            else None
        )

    cot_sampling = None
    if cot_mode is CoTMode.COT:
        cot_sampling = _resolve_cot_sampling_config(slug, model_name)
        if cot_sampling is None:
            raise ValueError(f"缺少采样配置: {slug} ({model_name})")

        if args.probe_only:
            batch_size = max(1, args.batch_size)
            pipeline.run_chain_of_thought(
                dataset_path=str(dataset_path),
                **({"cot_prompt_template": cot_prompt_template} if cot_prompt_template else {}),
                **({"final_answer_template": final_answer_template} if final_answer_template else {}),
                cot_sampling=cot_sampling,
                batch_size=batch_size,
                sample_limit=batch_size,
                min_prompt_count=batch_size,
                samples_per_task=1,
                probe_only=True,
                long_doc_config=long_doc_config,
                prompt_max_chars=prompt_max_chars,
                answer_strategy=knowledge_cot_strategy,
            )
            print(
                f"🧪 probe-only run completed: {batch_size} sample(s) evaluated with batch {args.batch_size}."
            )
            return 0

    dataset_snapshot = build_dataset_snapshot(
        dataset_path,
        dataset_slug=slug,
        loader=JsonlMultipleChoiceLoader,
        resolver=resolve_or_prepare_dataset,
        records=dataset_records,
    )
    protocol_bundle = build_protocol_bundle(
        protocol_name="knowledge",
        source_files=(
            Path(__file__),
            Path(__file__).with_name("pipeline.py"),
            Path(__file__).parents[2] / "field_common.py",
            Path(__file__).parents[2] / "benchmark_config.py",
            Path(__file__).parents[2] / "prompt_builders.py",
            Path(__file__).parents[2] / "metrics" / "multi_choice.py",
            Path(__file__).parents[2]
            / "datasets"
            / "data_loader"
            / "multiple_choice.py",
        ),
        config_files=benchmark_config_source_paths(slug, model_name),
        resolved_contract={
            "cot_mode": cot_mode.value,
            "prompt_profile": prompt_profile,
            "direct_prompt_template": direct_prompt_template,
            "cot_prompt_template": cot_prompt_template,
            "final_answer_template": final_answer_template,
            "knowledge_cot_strategy": knowledge_cot_strategy,
            "missing_prediction_score": missing_prediction_score,
            "prompt_max_chars": prompt_max_chars,
            "choice_sampling_protocol": choice_sampling_protocol,
        },
    )

    init_db(DEFAULT_DB_CONFIG)
    service = EvalDbService()
    expected_count = plan_attempt_count(plan, max_pass_k=1)
    task_state = prepare_task_execution(
        service=service,
        dataset=str(slug),
        model=model_name,
        is_param_search=False,
        job_name=job_name,
        run_mode=(run_context.run_mode if run_context is not None else None),
        sampling_config=_task_sampling_config(
            cot_mode,
            avg_k=plan.avg_k,
            effective_sample_count=plan.effective_sample_count,
            cot_sampling=cot_sampling,
            pass_ks=k_plan.pass_k,
            prompt_profile=prompt_profile,
            long_doc_config=long_doc_config,
            prompt_max_chars=prompt_max_chars,
            choice_sampling_protocol=choice_sampling_protocol,
            dataset_snapshot=dataset_snapshot,
            protocol_bundle=protocol_bundle,
        ),
    )
    task_run = TaskRunState.from_task_execution(
        execution_state=task_state,
        attempt_keys=attempt_keys,
        expected_attempt_count=expected_count,
    )
    runtime = TaskRunController(service=service, state=task_run)
    task_id = task_run.task_id
    skip_keys = task_state.skip_keys

    set_task_env(task_id)
    writer = runtime.create_writer(max_queue=args.db_write_queue)
    with attempt_stage(runtime, writer):
        if cot_mode is CoTMode.COT:
            result = pipeline.run_chain_of_thought(
                dataset_path=str(dataset_path),
                **({"cot_prompt_template": cot_prompt_template} if cot_prompt_template else {}),
                **({"final_answer_template": final_answer_template} if final_answer_template else {}),
                cot_sampling=cot_sampling,
                batch_size=max(1, args.batch_size),
                record_indices=plan.sample_indices,
                samples_per_task=max(plan.repeat_count, 1),
                attempt_keys=attempt_keys,
                skip_keys=skip_keys,
                on_record=writer.enqueue,
                long_doc_config=long_doc_config,
                prompt_max_chars=prompt_max_chars,
                answer_strategy=knowledge_cot_strategy,
            )
        else:
            result = pipeline.run_direct(
                dataset_path=str(dataset_path),
                prompt_template=direct_prompt_template,
                cot_mode=cot_mode,
                batch_size=max(1, args.batch_size),
                record_indices=plan.sample_indices,
                samples_per_task=plan.repeat_count,
                attempt_keys=attempt_keys,
                skip_keys=skip_keys,
                on_record=writer.enqueue,
                long_doc_config=long_doc_config,
                prompt_max_chars=prompt_max_chars,
            )

    completions_payloads = runtime.complete_attempt_stage(writer)
    with scoring_stage(runtime):
        if cot_mode is CoTMode.COT and knowledge_cot_strategy == "cascade_a_b":
            cascade = evaluate_multiple_choice_cascade(
                completions_payloads,
                dataset_path=dataset_path,
                missing_prediction_score=missing_prediction_score or 0.0,
            )
            strategy_task_ids = service.ingest_eval_payload_groups(
                task_id=task_id,
                completion_payloads=completions_payloads,
                payloads_by_group=cascade.payloads_by_group,
                primary_group=cascade.primary_group,
            )
            metrics_payload, metric_details = _build_cascade_metrics_payload(
                cascade,
                pass_k=k_plan.pass_k,
                avg_k=k_plan.avg_k,
                report_pass_k=k_plan.report_pass_k,
                report_avg_k=k_plan.report_avg_k,
            )
            metrics_payload["strategy_task_ids"] = {
                group: int(strategy_task_id)
                for group, strategy_task_id in strategy_task_ids.items()
            }
            task_details = {
                "knowledge_cot_strategy": knowledge_cot_strategy,
                "missing_prediction_score": missing_prediction_score,
                **metric_details,
                **build_plan_task_details(plan, cot_mode=cot_mode.value, prompt_profile=prompt_profile),
            }
            runtime.state.task_results = {
                (
                    int(payload["sample_index"]),
                    int(payload["repeat_index"]),
                    int(payload.get("pass_index", 0)),
                ): bool(payload.get("is_passed", False))
                for payload in cascade.payloads_by_group[cascade.primary_group]
            }
            runtime.state.eval_count = len(runtime.state.task_results)
            score_samples = cascade.samples
        else:
            metrics = evaluate_multiple_choice(completions_payloads, dataset_path=dataset_path)
            pass_metrics_all = compute_pass_at_k(metrics.rows, k_plan.pass_k)
            if missing_prediction_score is None or cot_mode is CoTMode.NO_COT:
                avg_metrics_all = compute_avg_at_k(metrics.rows, k_plan.avg_k)
            else:
                score_rows = [
                    (
                        sample_index,
                        repeat_index,
                        1.0
                        if passed
                        else missing_prediction_score
                        if not eval_payload["answer"]
                        else 0.0,
                    )
                    for (sample_index, repeat_index, passed), eval_payload in zip(
                        metrics.rows,
                        metrics.payloads,
                        strict=True,
                    )
                ]
                avg_metrics_all = compute_avg_score_at_k(score_rows, k_plan.avg_k)
            has_explicit_k_metrics = bool(k_plan.report_pass_k) or bool(k_plan.report_avg_k)
            metrics_payload = {}
            if not has_explicit_k_metrics:
                metrics_payload["accuracy"] = metrics.accuracy
            pass_payload = filter_metrics_by_k(pass_metrics_all, k_plan.report_pass_k, "pass@")
            if k_plan.report_pass_k and not pass_payload:
                pass_payload = pass_metrics_all or {}
            if pass_payload:
                metrics_payload.update(pass_payload)
            avg_payload = filter_metrics_by_k(avg_metrics_all, k_plan.report_avg_k, "avg@")
            if k_plan.report_avg_k and not avg_payload:
                avg_payload = avg_metrics_all or {}
            if avg_payload:
                metrics_payload.update(avg_payload)
            task_details = {
                "accuracy_by_subject": metrics.accuracy_by_subject,
                "knowledge_cot_strategy": knowledge_cot_strategy,
                "missing_prediction_score": missing_prediction_score,
                **build_plan_task_details(plan, cot_mode=cot_mode.value, prompt_profile=prompt_profile),
            }
            if pass_metrics_all and pass_payload != pass_metrics_all:
                task_details["pass_curve"] = pass_metrics_all
            if avg_metrics_all and avg_payload != avg_metrics_all:
                task_details["avg_curve"] = avg_metrics_all
            runtime.ingest_eval_payloads(metrics.payloads)
            score_samples = metrics.samples
        if _should_run_checker(args):
            runtime.run_checker(model_name=model_name)
        score_payload = make_score_payload(
            slug,
            is_cot=cot_mode is not CoTMode.NO_COT,
            model_name=model_name,
            metrics=metrics_payload,
            samples=score_samples,
            task=job_name,
            task_details=task_details,
            extra={
                "cot_mode": cot_mode.value,
                "prompt_profile": prompt_profile,
                "infer_protocol": getattr(args, "infer_protocol", "local"),
                "completion_style_remote": completion_style_remote,
                "choice_scoring": "generative_fill_in",
            },
        )
        runtime.record_score(score_payload)
    _print_done_message(cot_mode, result.sample_count)
    return 0


__all__ = ["main", "parse_args"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
