# ruff: noqa: E402
from __future__ import annotations

"""Argparse-based CLI that exposes the scheduler actions."""

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

from src.eval.env_config import load_env_file

load_env_file(Path(".env"))

from src.eval.benchmark_registry import BENCHMARK_ALIASES, KNOWN_BENCHMARKS, BenchmarkField
from src.eval.evaluating import RunMode, collect_benchmark_dataset_slugs
from src.eval.tasks.function_calling.rwkv_prompt import (
    FUNCTION_PROMPT_STYLE_CHOICES,
    FUNCTION_TOOL_CATALOG_FORMAT_CHOICES,
)
from src.eval.tasks.function_calling.tool_router import TOOL_ROUTER_MODE_CHOICES
from src.eval.performance.workload import parse_int_csv
from src.infer.backend import REMOTE_INFERENCE_PROTOCOL_CHOICES

from .actions import (
    CodingConfig,
    DispatchOptions,
    FunctionCallingConfig,
    InferenceConfig,
    KnowledgeConfig,
    LogsOptions,
    MathConfig,
    StatusOptions,
    StopOptions,
    action_dispatch,
    action_logs,
    action_queue,
    action_status,
    action_stop,
)
from .action_dispatch import require_strict_g1i_runtime_attestation
from .admin import SchedulerAdminController, serve_scheduler_admin
from .config import (
    DEFAULT_ADMIN_API_KEY,
    DEFAULT_ADMIN_HOST,
    DEFAULT_ADMIN_PORT,
    DEFAULT_ADMIN_STATE_DIR,
    DEFAULT_INFER_MAX_WORKERS,
    DEFAULT_INFER_SLOTS_PER_MODEL,
    DEFAULT_LOG_DIR,
    DEFAULT_MODEL_GLOBS,
    DEFAULT_PID_DIR,
    DEFAULT_RUN_LOG_DIR,
)
from .db_bootstrap import bootstrap_db_schema, check_db_schema, render_db_schema_report
from .dataset_utils import canonical_slug, canonicalize_benchmark_list
from .jobs import JOB_CATALOGUE, JOB_ORDER
from .launch_config import (
    DEFAULT_PROFILE_NAME,
    SchedulerLaunchRequest,
    launch_request_to_json,
    load_launch_profile,
)
from .models import MODEL_SELECT_CHOICES
from .remote_slots import INFER_WORKER_PROFILE_CHOICES
from .remote_profiler import DEFAULT_REMOTE_PROBE_PROMPT, probe_remote_inference, write_remote_probe_result


_KNOWN_DATASET_SLUGS: tuple[str, ...] = tuple(
    sorted({canonical_slug(slug) for spec in JOB_CATALOGUE.values() for slug in spec.dataset_slugs})
)
_KNOWN_BENCHMARK_NAMES: tuple[str, ...] = tuple(
    sorted({item.name for item in KNOWN_BENCHMARKS} | set(BENCHMARK_ALIASES))
)
_BENCHMARK_FIELD_CHOICES: tuple[str, ...] = tuple(field.value for field in BenchmarkField)
_RUN_MODE_CHOICES: tuple[str, ...] = tuple(mode.value for mode in RunMode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RWKV 调度器 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    queue_parser = sub.add_parser("queue", help="查看待调度队列")
    _add_job_filters(queue_parser)
    _add_dispatch_options(queue_parser)

    dispatch_parser = sub.add_parser("dispatch", help="根据 GPU 空闲情况调度任务")
    _add_job_filters(dispatch_parser)
    _add_dispatch_options(dispatch_parser)

    run_parser = sub.add_parser("run", help="按 configs/scheduler/*.toml profile 预检并启动调度器")
    run_parser.add_argument("--profile", default=DEFAULT_PROFILE_NAME, help="scheduler profile 名称或 TOML 路径")
    run_parser.add_argument("--run-mode", choices=_RUN_MODE_CHOICES, help="覆盖 profile 中的 run_mode")
    run_parser.add_argument("--dry-run", action="store_true", help="只解析 profile、预检 DB 并输出队列，不启动任务")
    run_parser.add_argument("--print-config", action="store_true", help="打印解析后的 profile 配置")
    run_parser.add_argument("--no-bootstrap-db", action="store_true", help="DB schema 缺失时只报错，不自动应用 scripts/schema.sql")

    doctor_parser = sub.add_parser("doctor", help="检查 profile 和评测数据库 schema")
    doctor_parser.add_argument("--profile", default=DEFAULT_PROFILE_NAME, help="scheduler profile 名称或 TOML 路径")
    doctor_parser.add_argument("--json", action="store_true", help="以 JSON 输出诊断结果")

    bootstrap_parser = sub.add_parser("bootstrap-db", help="幂等应用 scripts/schema.sql 初始化评测数据库")
    bootstrap_parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")

    status_parser = sub.add_parser("status", help="查看正在运行的任务")
    status_parser.add_argument("--pid-dir", default=str(DEFAULT_PID_DIR), help="PID 文件目录")

    stop_parser = sub.add_parser("stop", help="停止任务")
    stop_parser.add_argument("--pid-dir", default=str(DEFAULT_PID_DIR), help="PID 文件目录")
    stop_parser.add_argument("--all", action="store_true", help="停止全部任务")
    stop_parser.add_argument("job_ids", nargs="*", help="待停止的 job id")

    logs_parser = sub.add_parser("logs", help="轮询输出运行日志")
    logs_parser.add_argument("--pid-dir", default=str(DEFAULT_PID_DIR), help="PID 文件目录")
    logs_parser.add_argument("--run-log-dir", default=str(DEFAULT_RUN_LOG_DIR), help="运行日志目录")
    logs_parser.add_argument("--tail-lines", type=int, default=60, help="每次展示的尾行数")
    logs_parser.add_argument("--rotate-seconds", type=int, default=15, help="轮播间隔秒数")

    serve_parser = sub.add_parser("serve", help="启动 HTTP / admin 控制服务")
    serve_parser.add_argument("--host", default=DEFAULT_ADMIN_HOST, help="HTTP 监听地址")
    serve_parser.add_argument("--port", type=int, default=DEFAULT_ADMIN_PORT, help="HTTP 监听端口")
    serve_parser.add_argument("--state-dir", default=str(DEFAULT_ADMIN_STATE_DIR), help="scheduler admin 状态目录")
    serve_parser.add_argument(
        "--admin-api-key",
        default=DEFAULT_ADMIN_API_KEY,
        help="Bearer token；为空时不鉴权",
    )

    probe_parser = sub.add_parser("probe-infer", help="探测远端推理服务最大健康并发")
    probe_parser.add_argument("--infer-base-url", required=True, help="远端推理服务地址")
    probe_parser.add_argument("--infer-model", required=True, help="远端推理服务上的模型名")
    probe_parser.add_argument("--infer-api-key", default="", help="远端推理服务 API key")
    probe_parser.add_argument("--infer-timeout-s", type=float, default=600.0, help="远端推理请求超时")
    probe_parser.add_argument(
        "--infer-protocol",
        choices=REMOTE_INFERENCE_PROTOCOL_CHOICES,
        default="vllm",
        help="远端推理协议",
    )
    probe_parser.add_argument(
        "--candidates",
        default="1,2,4,8,16,32,64",
        help="逗号分隔并发候选值，按升序探测到首次失败为止",
    )
    probe_parser.add_argument("--prompt", default=DEFAULT_REMOTE_PROBE_PROMPT, help="探测 prompt")
    probe_parser.add_argument("--max-tokens", type=int, default=16, help="每个请求生成 token 数")
    probe_parser.add_argument("--temperature", type=float, default=1e-5, help="探测 temperature")
    probe_parser.add_argument("--top-p", type=float, default=0.8, help="探测 top-p")
    probe_parser.add_argument("--top-k", type=int, default=50, help="探测 top-k")
    probe_parser.add_argument("--stop-suffix", help="可选文本停止后缀，nano contents 会映射为 stop_tokens")
    probe_parser.add_argument("--gpu-index", type=int, help="可选本机 GPU index；传入后采样利用率/显存")
    probe_parser.add_argument("--target-gpu-utilization", type=float, default=90.0, help="认为 GPU 已吃满的峰值利用率阈值")
    probe_parser.add_argument("--warmup-requests", type=int, default=1, help="正式并发曲线前的预热请求数")
    probe_parser.add_argument("--max-p95-latency-s", type=float, help="候选并发 p95 延迟上限")
    probe_parser.add_argument("--min-throughput-gain", type=float, default=0.03, help="选择吞吐拐点的最小相对增益")
    probe_parser.add_argument("--output-json", help="写出探测结果 JSON")

    return parser


def _add_job_filters(parser: argparse.ArgumentParser) -> None:
    domain_choices = sorted({spec.domain for spec in JOB_CATALOGUE.values() if spec.domain})
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="调度器日志/缓存目录")
    parser.add_argument("--pid-dir", default=str(DEFAULT_PID_DIR), help="PID 文件目录")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODEL_GLOBS),
        help="本地模型文件 glob（用于定位权重；远端推理模式下忽略）",
    )
    parser.add_argument(
        "--model-regex",
        nargs="+",
        help="仅保留文件名（不含路径）匹配任一正则的模型，例如 --model-regex '^rwkv7-.*7\\.2b'",
    )
    parser.add_argument(
        "--model-select",
        choices=MODEL_SELECT_CHOICES,
        default="latest-data",
        help="模型筛选策略（默认 latest-data：每档参数取 data_version 最新，忽略 0.1b/0.4b）",
    )
    parser.add_argument("--min-param-b", type=float, help="仅保留参数量 >= 阈值 (B)")
    parser.add_argument("--max-param-b", type=float, help="仅保留参数量 <= 阈值 (B)")
    parser.add_argument(
        "--only-jobs",
        nargs="+",
        choices=sorted(JOB_CATALOGUE.keys()),
        help="仅运行指定 job",
    )
    parser.add_argument(
        "--skip-jobs",
        nargs="+",
        choices=sorted(JOB_CATALOGUE.keys()),
        help="跳过指定 job",
    )
    parser.add_argument(
        "--job-order",
        nargs="+",
        choices=sorted(JOB_CATALOGUE.keys()),
        help="自定义 job 优先级（按给定顺序优先），未指定时按题量/CoT 自动排序",
    )
    if domain_choices:
        parser.add_argument(
            "--domains",
            nargs="+",
            choices=domain_choices,
            help=f"按任务域筛选 job，例如 --domains {'/'.join(domain_choices)}",
        )
    parser.add_argument(
        "--benchmark-fields",
        nargs="+",
        choices=_BENCHMARK_FIELD_CHOICES,
        help="按 benchmark 领域筛选，语义对齐 rwkv-rs 的 benchmark_field",
    )
    parser.add_argument(
        "--extra-benchmarks",
        nargs="+",
        choices=_KNOWN_BENCHMARK_NAMES,
        help="额外包含的 benchmark，语义对齐 rwkv-rs 的 extra_benchmark_name",
    )
    parser.add_argument(
        "--only-datasets",
        nargs="+",
        help="仅运行指定 benchmark（使用数据集名称即可，如 aime24 或 gpqa）",
    )
    parser.add_argument(
        "--skip-datasets",
        nargs="+",
        help="跳过指定 benchmark（名称即可，无需 *_test 后缀）",
    )
    parser.add_argument(
        "--enable-param-search",
        action="store_true",
        help="启用 param-search（默认关闭，仅对最新 2.9b 生效）",
    )


def _add_dispatch_options(parser: argparse.ArgumentParser) -> None:
    """Add dispatch-related options (also used by `queue` for dry-run parity)."""

    parser.add_argument("--run-log-dir", default=str(DEFAULT_RUN_LOG_DIR), help="运行日志目录")
    parser.add_argument("--infer-base-url", help="远端推理服务地址；设置后 scheduler 进入评测/推理分离模式")
    parser.add_argument("--infer-models", nargs="+", help="远端推理服务上的模型名列表")
    parser.add_argument("--infer-api-key", default="", help="远端推理服务 API key")
    parser.add_argument("--infer-timeout-s", type=float, default=600.0, help="远端推理请求超时")
    parser.add_argument(
        "--infer-max-workers",
        type=int,
        default=DEFAULT_INFER_MAX_WORKERS,
        help="每个评测 worker 的远端请求并发上限",
    )
    parser.add_argument(
        "--infer-slots-per-model",
        type=int,
        default=DEFAULT_INFER_SLOTS_PER_MODEL,
        help="每个远端模型展开为多少个并发 slot，使多个评测任务同时喂一个批处理服务（单 GPU 单服务建议 2-4）",
    )
    parser.add_argument(
        "--infer-worker-profile",
        choices=INFER_WORKER_PROFILE_CHOICES,
        default="fixed",
        help="远端 worker 档位：fixed 使用 --infer-max-workers；param-size 按模型大小分配",
    )
    parser.add_argument(
        "--infer-protocol",
        choices=REMOTE_INFERENCE_PROTOCOL_CHOICES,
        default="openai",
        help="远端推理协议：标准 OpenAI 兼容，或按 vLLM continuous batching 调度",
    )
    parser.add_argument(
        "--infer-seed-policy",
        choices=("preserve", "omit"),
        default="preserve",
        help="远端 seed 策略：默认保留 seed；omit 在 vLLM/completions 请求中丢弃逐 prompt seed",
    )
    parser.add_argument("--remote-batch-size", type=int, help="远端推理模式下传给支持 batch 的 runner 的 --batch-size")
    parser.add_argument(
        "--plain-choice-batch-size",
        type=int,
        help="远端 multi_choice_plain/multi_choice_plain_naive 的专用 --batch-size；未设置则使用 --remote-batch-size",
    )
    parser.add_argument(
        "--plain-choice-timeout-s",
        type=float,
        help="远端 multi_choice_plain/multi_choice_plain_naive 的专用请求超时；未设置则使用 --infer-timeout-s",
    )
    parser.add_argument("--sample-workers", type=int, help="runner 侧 episode 并发数；当前透传给 function-calling runner")
    parser.add_argument("--coding-eval-workers", type=int, help="coding runner 本地评测并发数；透传为 --eval-workers")
    parser.add_argument("--max-active-coding-runners", type=int, help="最多同时运行的 coding runner 数；空出的远端槽可调度非 coding 任务")
    parser.add_argument("--math-judge-max-workers", type=int, help="maths free_response_judge 的 --judge-max-workers")
    parser.add_argument("--math-prompt-max-chars", type=int, help="maths runner 的 --prompt-max-chars")
    parser.add_argument("--math-long-doc-mode", choices=("off", "lexical"), help="maths runner 的 --long-doc-mode")
    parser.add_argument("--knowledge-prompt-max-chars", type=int, help="knowledge runner 的 --prompt-max-chars")
    parser.add_argument("--knowledge-long-doc-mode", choices=("off", "lexical"), help="knowledge runner 的 --long-doc-mode")
    parser.add_argument(
        "--disable-infer-backpressure",
        action="store_true",
        help="关闭远端推理背压读取，使用静态 --infer-max-workers/--remote-batch-size",
    )
    parser.add_argument(
        "--infer-backpressure-timeout-s",
        type=float,
        default=2.0,
        help="读取远端 /v1/backpressure 的超时秒数",
    )
    parser.add_argument(
        "--infer-backpressure-pending-high-watermark",
        type=int,
        default=0,
        help="远端 pending queue 超过该值时暂缓启动该模型的下一个 benchmark",
    )
    parser.add_argument(
        "--infer-budget-min-workers",
        type=int,
        default=1,
        help="背压预算下调时保留的最小远端请求并发",
    )
    parser.add_argument(
        "--function-prompt-style",
        choices=FUNCTION_PROMPT_STYLE_CHOICES,
        help="function-calling runner 的 --prompt-style",
    )
    parser.add_argument(
        "--function-tool-catalog-format",
        choices=FUNCTION_TOOL_CATALOG_FORMAT_CHOICES,
        help="function-calling runner 的 --tool-catalog-format",
    )
    parser.add_argument("--function-cot-max-tokens", type=int, help="function-calling runner 的 --cot-max-tokens")
    parser.add_argument("--function-decision-max-tokens", type=int, help="function-calling runner 的 --decision-max-tokens")
    parser.add_argument("--function-planning-max-tokens", type=int, help="function-calling runner 的 --planning-max-tokens")
    parser.add_argument("--function-final-max-tokens", type=int, help="function-calling runner 的 --final-max-tokens")
    parser.add_argument("--function-answer-max-tokens", type=int, help="function-calling runner 的 --answer-max-tokens")
    parser.add_argument("--function-judge-max-workers", type=int, help="BrowseComp judge 的 --judge-max-workers")
    parser.add_argument("--function-history-max-chars", type=int, help="function-calling runner 的 --history-max-chars")
    parser.add_argument("--function-prompt-max-chars", type=int, help="function-calling runner 的 --prompt-max-chars")
    parser.add_argument(
        "--function-long-doc-mode",
        choices=("off", "lexical"),
        help="function-calling runner 的 --long-doc-mode",
    )
    parser.add_argument(
        "--function-tool-router-mode",
        choices=TOOL_ROUTER_MODE_CHOICES,
        help="function-calling runner 的 --tool-router-mode",
    )
    parser.add_argument(
        "--function-tool-router-max-tools",
        type=int,
        help="function-calling runner 的 --tool-router-max-tools",
    )
    parser.add_argument(
        "--function-tool-router-trigger-tool-count",
        type=int,
        help="function-calling runner 的 --tool-router-trigger-tool-count",
    )
    parser.add_argument(
        "--function-tool-router-trigger-catalog-chars",
        type=int,
        help="function-calling runner 的 --tool-router-trigger-catalog-chars",
    )
    parser.add_argument(
        "--function-candidate-router-mode",
        choices=("off", "auto", "parallel"),
        help="function-calling runner 的 --candidate-router-mode",
    )
    parser.add_argument(
        "--function-candidate-router-chunk-tools",
        type=int,
        help="function-calling runner 的 --candidate-router-chunk-tools",
    )
    parser.add_argument(
        "--function-candidate-router-batch-size",
        type=int,
        help="function-calling runner 的 --candidate-router-batch-size",
    )
    parser.add_argument(
        "--function-candidate-router-prompt-max-chars",
        type=int,
        help="function-calling runner 的 --candidate-router-prompt-max-chars",
    )
    parser.add_argument(
        "--function-candidate-router-context-chars",
        type=int,
        help="function-calling runner 的 --candidate-router-context-chars",
    )
    parser.add_argument(
        "--function-candidate-router-candidate-max-tokens",
        type=int,
        help="function-calling runner 的 --candidate-router-candidate-max-tokens",
    )
    parser.add_argument(
        "--function-candidate-router-aggregate-max-tokens",
        type=int,
        help="function-calling runner 的 --candidate-router-aggregate-max-tokens",
    )
    parser.add_argument(
        "--function-candidate-router-max-candidates",
        type=int,
        help="function-calling runner 的 --candidate-router-max-candidates",
    )
    parser.add_argument(
        "--function-candidate-router-tool-schema-mode",
        choices=("minimal", "compact", "full"),
        help="function-calling runner 的 --candidate-router-tool-schema-mode",
    )
    parser.add_argument(
        "--function-candidate-router-evidence-chars",
        type=int,
        help="function-calling runner 的 --candidate-router-evidence-chars",
    )
    parser.add_argument(
        "--function-candidate-router-policy-chars",
        type=int,
        help="function-calling runner 的 --candidate-router-policy-chars",
    )
    parser.add_argument("--function-max-rounds", type=int, help="function-calling runner 的 --max-rounds")
    parser.add_argument("--function-max-steps", type=int, help="function-calling runner 的 --max-steps")
    parser.add_argument("--function-max-tool-errors", type=int, help="function-calling runner 的 --max-tool-errors")
    parser.add_argument(
        "--function-complexfuncbench-disable-response-eval",
        action="store_true",
        help="透传给 ComplexFuncBench runner，关闭官方 GPT response eval",
    )
    parser.add_argument(
        "--function-complexfuncbench-offline-compare",
        action="store_true",
        help="透传给 ComplexFuncBench runner，使用离线比较避免 RapidAPI/GPT 等外部调用",
    )
    parser.add_argument("--distributed-claims", action="store_true", help="启用 PostgreSQL claim/lease，允许多个 scheduler 节点协同")
    parser.add_argument("--scheduler-node-id", help="当前 scheduler 节点标识；默认取主机名")
    parser.add_argument("--lease-duration-s", type=int, default=900, help="claim/lease 有效期秒数")
    parser.add_argument(
        "--run-mode",
        choices=_RUN_MODE_CHOICES,
        default=RunMode.AUTO.value,
        help=(
            "任务执行语义：auto/new/resume/rerun/fresh；fresh 在队列层跳过已完成分数，"
            "但对缺失项强制新建 task，避免续跑旧 Running/Failed"
        ),
    )
    parser.add_argument(
        "--dispatch-poll-seconds",
        type=int,
        default=30,
        help="空闲 GPU 轮询间隔",
    )
    parser.add_argument(
        "--gpu-idle-max-mem",
        type=int,
        default=1000,
        help="将 GPU 视为空闲的显存占用阈值 (MB)",
    )
    parser.add_argument(
        "--skip-missing-dataset",
        action="store_true",
        help="缺少数据集时跳过该任务",
    )
    parser.add_argument(
        "--clean-param-swap",
        action="store_true",
        help="启动前清理 log_dir/param_swap",
    )
    parser.add_argument(
        "--batch-cache",
        help="自定义 batch profiler 缓存路径 (默认为 log_dir/batch_cache.json)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="兼容旧接口，相当于 --run-mode rerun",
    )
    parser.add_argument(
        "--disable-checker",
        action="store_true",
        help="关闭 LLM wrong-answer checker（不写 checker 表，也不跑离线 checker）",
    )


def _dispatch_options_from_args(
    args: argparse.Namespace,
    *,
    job_list: tuple[str, ...],
    job_priority: tuple[str, ...] | None,
    model_globs: tuple[str, ...],
    skip_dataset_slugs: tuple[str, ...],
    only_dataset_slugs: tuple[str, ...],
    model_name_patterns: tuple[re.Pattern[str], ...],
    min_param_b: float | None,
    max_param_b: float | None,
    model_select: str,
    run_mode: RunMode,
    infer_base_url: str | None,
    infer_models: tuple[str, ...],
) -> DispatchOptions:
    batch_cache = Path(args.batch_cache) if getattr(args, "batch_cache", None) else None
    return DispatchOptions(
        log_dir=Path(args.log_dir),
        pid_dir=Path(args.pid_dir),
        run_log_dir=Path(args.run_log_dir),
        job_order=job_list,
        job_priority=job_priority,
        model_select=model_select,
        min_param_b=min_param_b,
        max_param_b=max_param_b,
        skip_dataset_slugs=skip_dataset_slugs,
        model_globs=model_globs,
        only_dataset_slugs=only_dataset_slugs,
        model_name_patterns=model_name_patterns,
        enable_param_search=bool(args.enable_param_search),
        run_mode=run_mode,
        inference=InferenceConfig(
            base_url=infer_base_url,
            models=infer_models,
            api_key=str(getattr(args, "infer_api_key", "") or ""),
            timeout_s=float(getattr(args, "infer_timeout_s", 600.0)),
            max_workers=int(getattr(args, "infer_max_workers", DEFAULT_INFER_MAX_WORKERS)),
            worker_profile=str(getattr(args, "infer_worker_profile", "fixed") or "fixed"),
            protocol=str(getattr(args, "infer_protocol", "openai") or "openai"),
            seed_policy=str(getattr(args, "infer_seed_policy", "preserve") or "preserve"),
            remote_batch_size=(
                int(getattr(args, "remote_batch_size"))
                if getattr(args, "remote_batch_size", None) is not None
                else None
            ),
            plain_choice_batch_size=(
                int(getattr(args, "plain_choice_batch_size"))
                if getattr(args, "plain_choice_batch_size", None) is not None
                else None
            ),
            plain_choice_timeout_s=(
                float(getattr(args, "plain_choice_timeout_s"))
                if getattr(args, "plain_choice_timeout_s", None) is not None
                else None
            ),
            sample_workers=(
                int(getattr(args, "sample_workers"))
                if getattr(args, "sample_workers", None) is not None
                else None
            ),
            backpressure=not bool(getattr(args, "disable_infer_backpressure", False)),
            backpressure_timeout_s=float(getattr(args, "infer_backpressure_timeout_s", 2.0)),
            backpressure_pending_high_watermark=int(
                getattr(args, "infer_backpressure_pending_high_watermark", 0)
            ),
            budget_min_workers=int(getattr(args, "infer_budget_min_workers", 1)),
        ),
        functions=FunctionCallingConfig(
            prompt_style=getattr(args, "function_prompt_style", None),
            tool_catalog_format=getattr(args, "function_tool_catalog_format", None),
            cot_max_tokens=getattr(args, "function_cot_max_tokens", None),
            decision_max_tokens=getattr(args, "function_decision_max_tokens", None),
            planning_max_tokens=getattr(args, "function_planning_max_tokens", None),
            final_max_tokens=getattr(args, "function_final_max_tokens", None),
            answer_max_tokens=getattr(args, "function_answer_max_tokens", None),
            judge_max_workers=getattr(args, "function_judge_max_workers", None),
            history_max_chars=getattr(args, "function_history_max_chars", None),
            prompt_max_chars=getattr(args, "function_prompt_max_chars", None),
            long_doc_mode=getattr(args, "function_long_doc_mode", None),
            tool_router_mode=getattr(args, "function_tool_router_mode", None),
            tool_router_max_tools=getattr(args, "function_tool_router_max_tools", None),
            tool_router_trigger_tool_count=getattr(args, "function_tool_router_trigger_tool_count", None),
            tool_router_trigger_catalog_chars=getattr(args, "function_tool_router_trigger_catalog_chars", None),
            candidate_router_mode=getattr(args, "function_candidate_router_mode", None),
            candidate_router_chunk_tools=getattr(args, "function_candidate_router_chunk_tools", None),
            candidate_router_batch_size=getattr(args, "function_candidate_router_batch_size", None),
            candidate_router_prompt_max_chars=getattr(args, "function_candidate_router_prompt_max_chars", None),
            candidate_router_context_chars=getattr(args, "function_candidate_router_context_chars", None),
            candidate_router_candidate_max_tokens=getattr(args, "function_candidate_router_candidate_max_tokens", None),
            candidate_router_aggregate_max_tokens=getattr(args, "function_candidate_router_aggregate_max_tokens", None),
            candidate_router_max_candidates=getattr(args, "function_candidate_router_max_candidates", None),
            candidate_router_tool_schema_mode=getattr(args, "function_candidate_router_tool_schema_mode", None),
            candidate_router_evidence_chars=getattr(args, "function_candidate_router_evidence_chars", None),
            candidate_router_policy_chars=getattr(args, "function_candidate_router_policy_chars", None),
            max_rounds=getattr(args, "function_max_rounds", None),
            max_steps=getattr(args, "function_max_steps", None),
            max_tool_errors=getattr(args, "function_max_tool_errors", None),
            complexfuncbench_disable_response_eval=bool(
                getattr(args, "function_complexfuncbench_disable_response_eval", False)
            ),
            complexfuncbench_offline_compare=bool(
                getattr(args, "function_complexfuncbench_offline_compare", False)
            ),
        ),
        coding=CodingConfig(
            eval_workers=(
                int(getattr(args, "coding_eval_workers"))
                if getattr(args, "coding_eval_workers", None) is not None
                else None
            ),
            max_active_runners=(
                int(getattr(args, "max_active_coding_runners"))
                if getattr(args, "max_active_coding_runners", None) is not None
                else None
            ),
        ),
        math=MathConfig(
            judge_max_workers=(
                int(getattr(args, "math_judge_max_workers"))
                if getattr(args, "math_judge_max_workers", None) is not None
                else None
            ),
            prompt_max_chars=(
                int(getattr(args, "math_prompt_max_chars"))
                if getattr(args, "math_prompt_max_chars", None) is not None
                else None
            ),
            long_doc_mode=getattr(args, "math_long_doc_mode", None),
        ),
        knowledge=KnowledgeConfig(
            prompt_max_chars=(
                int(getattr(args, "knowledge_prompt_max_chars"))
                if getattr(args, "knowledge_prompt_max_chars", None) is not None
                else None
            ),
            long_doc_mode=getattr(args, "knowledge_long_doc_mode", None),
        ),
        distributed_claims=bool(getattr(args, "distributed_claims", False)),
        scheduler_node_id=(str(getattr(args, "scheduler_node_id", "") or "").strip() or None),
        lease_duration_s=int(getattr(args, "lease_duration_s", 900)),
        dispatch_poll_seconds=int(args.dispatch_poll_seconds),
        gpu_idle_max_mem=int(args.gpu_idle_max_mem),
        skip_missing_dataset=bool(args.skip_missing_dataset),
        clean_param_swap=bool(args.clean_param_swap),
        batch_cache_path=batch_cache,
        disable_checker=bool(args.disable_checker),
    )


def _resolve_job_list(
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
    domains: Sequence[str] | None,
) -> tuple[str, ...]:
    order = list(JOB_ORDER)

    if domains:
        allowed_domains = set(domains)
        order = [job for job in order if JOB_CATALOGUE[job].domain in allowed_domains]

    if include:
        allowed = {job for job in include}
        order = [job for job in order if job in allowed]
    if exclude:
        blocked = {job for job in exclude}
        order = [job for job in order if job not in blocked]
    return tuple(order)


def _canonicalize_slugs(
    parser: argparse.ArgumentParser,
    slugs: Sequence[str] | None,
) -> tuple[str, ...]:
    if not slugs:
        return tuple()
    try:
        return canonicalize_benchmark_list(slugs, known_slugs=_KNOWN_DATASET_SLUGS)
    except ValueError as exc:  # pragma: no cover - argparse already prints
        parser.error(str(exc))


def _compile_model_patterns(
    parser: argparse.ArgumentParser,
    patterns: Sequence[str] | None,
) -> tuple[re.Pattern[str], ...]:
    if not patterns:
        return tuple()
    compiled: list[re.Pattern[str]] = []
    for raw in patterns:
        try:
            compiled.append(re.compile(raw))
        except re.error as exc:  # pragma: no cover - argparse already prints
            parser.error(f"无效的模型正则 {raw!r}: {exc}")
    return tuple(compiled)


def _resolve_job_priority(priority: Sequence[str] | None, available: Sequence[str]) -> tuple[str, ...] | None:
    if not priority:
        return None
    allowed = {job for job in available}
    ordered: list[str] = []
    for job in priority:
        if job in allowed and job not in ordered:
            ordered.append(job)
    return tuple(ordered) if ordered else None


def _parse_benchmark_fields(values: Sequence[str] | None) -> tuple[BenchmarkField, ...]:
    if not values:
        return tuple()
    return tuple(BenchmarkField(value) for value in values)


def _collect_selected_dataset_slugs(
    parser: argparse.ArgumentParser,
    *,
    benchmark_fields: Sequence[BenchmarkField],
    extra_benchmarks: Sequence[str] | None,
    only_datasets: Sequence[str] | None,
) -> tuple[str, ...]:
    selected: set[str] = set()
    if benchmark_fields or extra_benchmarks:
        try:
            selected.update(
                collect_benchmark_dataset_slugs(
                    fields=benchmark_fields,
                    extra_benchmark_names=tuple(extra_benchmarks or ()),
                )
            )
        except ValueError as exc:
            parser.error(str(exc))

    selected.update(_canonicalize_slugs(parser, only_datasets))
    return tuple(sorted(selected))


def _resolve_run_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> RunMode:
    explicit = getattr(args, "run_mode", RunMode.AUTO.value)
    overwrite = bool(getattr(args, "overwrite", False))
    if overwrite and explicit not in (RunMode.AUTO.value, RunMode.RERUN.value):
        parser.error("--overwrite 只能与 --run-mode auto/rerun 搭配使用")
    if overwrite:
        return RunMode.RERUN
    try:
        return RunMode.parse(explicit)
    except ValueError as exc:
        parser.error(str(exc))


def _resolve_scheduler_inference_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    model_globs: tuple[str, ...],
) -> tuple[tuple[str, ...], str | None, tuple[str, ...]]:
    infer_base_url = str(getattr(args, "infer_base_url", "") or "").strip() or None
    infer_models = tuple(str(item).strip() for item in (getattr(args, "infer_models", None) or []) if str(item).strip())
    remote_mode = bool(infer_base_url or infer_models)
    if remote_mode:
        if not infer_base_url:
            parser.error("远端推理模式缺少 --infer-base-url")
        if not infer_models:
            parser.error("远端推理模式缺少 --infer-models")
        slots_per_model = int(getattr(args, "infer_slots_per_model", DEFAULT_INFER_SLOTS_PER_MODEL) or 1)
        infer_models = _expand_infer_model_slots(infer_models, slots_per_model)
        return tuple(), infer_base_url, infer_models
    return model_globs, None, tuple()


def _expand_infer_model_slots(
    infer_models: Sequence[str],
    slots_per_model: int,
) -> tuple[str, ...]:
    """把每个模型展开为 N 个并发 slot（slot 唯一、model 保持真实名）。

    机制见 scheduler/remote_slots.py：多个 slot 可指向同一后端模型，让多个
    benchmark job 并行喂同一个批处理服务，而 DB 身份仍用真实模型名。
    """
    count = max(1, int(slots_per_model))
    if count == 1:
        return tuple(infer_models)
    expanded: list[str] = []
    seen: set[str] = set()
    for raw in infer_models:
        text = str(raw).strip()
        if not text:
            continue
        if "=" in text:
            slot, model = (part.strip() for part in text.split("=", 1))
            spec = f"{slot}={model}"
            if spec not in seen:
                seen.add(spec)
                expanded.append(spec)
            continue
        else:
            slot = model = text
        for index in range(count):
            spec = f"{slot}-s{index}={model}"
            if spec in seen:
                continue
            seen.add(spec)
            expanded.append(spec)
    return tuple(expanded)


def _run_probe_infer(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    try:
        candidates = parse_int_csv(str(args.candidates))
    except ValueError as exc:
        parser.error(str(exc))
    result = probe_remote_inference(
        base_url=str(args.infer_base_url),
        model=str(args.infer_model),
        api_key=str(getattr(args, "infer_api_key", "") or ""),
        timeout_s=float(getattr(args, "infer_timeout_s", 600.0)),
        protocol=str(getattr(args, "infer_protocol", "openai") or "openai"),  # type: ignore[arg-type]
        candidates=candidates,
        prompt=str(getattr(args, "prompt", DEFAULT_REMOTE_PROBE_PROMPT)),
        max_tokens=int(getattr(args, "max_tokens", 16)),
        temperature=float(getattr(args, "temperature", 1e-5)),
        top_p=float(getattr(args, "top_p", 0.8)),
        top_k=int(getattr(args, "top_k", 50)),
        stop_suffix=getattr(args, "stop_suffix", None),
        gpu_index=getattr(args, "gpu_index", None),
        target_gpu_utilization=float(getattr(args, "target_gpu_utilization", 90.0)),
        warmup_requests=int(getattr(args, "warmup_requests", 1)),
        max_p95_latency_s=getattr(args, "max_p95_latency_s", None),
        min_throughput_gain=float(getattr(args, "min_throughput_gain", 0.03)),
    )
    if getattr(args, "output_json", None):
        write_remote_probe_result(Path(args.output_json), result)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _load_request_for_profile_command(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> SchedulerLaunchRequest:
    try:
        request = load_launch_profile(getattr(args, "profile", DEFAULT_PROFILE_NAME))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    request = request.copy()
    if getattr(args, "run_mode", None):
        request.run_mode = str(args.run_mode)
        request.overwrite = False
    return request


def _run_doctor(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    request = _load_request_for_profile_command(parser, args)
    config_error: str | None = None
    try:
        opts = request.to_dispatch_options()
    except Exception as exc:  # noqa: BLE001 - compact CLI diagnostic
        config_error = f"{type(exc).__name__}: {exc}"
        opts = None
    db_report = check_db_schema()
    payload = {
        "profile": request.profile,
        "config_ok": config_error is None,
        "config_error": config_error,
        "db": db_report.to_dict(),
    }
    if opts is not None:
        payload["dispatch"] = _dispatch_options_summary(opts)
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"profile={request.profile}")
        if config_error:
            print(f"config_ok=false\nconfig_error={config_error}")
        else:
            print("config_ok=true")
            assert opts is not None
            print(
                "dispatch="
                f"jobs:{len(opts.job_order)} datasets:{len(opts.only_dataset_slugs) or 'all'} "
                f"remote:{bool(opts.inference.base_url)}"
            )
        print(render_db_schema_report(db_report))
    return 0 if config_error is None and db_report.ok else 1


def _run_bootstrap_db(args: argparse.Namespace) -> int:
    report = bootstrap_db_schema()
    if getattr(args, "json", False):
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_db_schema_report(report))
    return 0 if report.ok else 1


def _run_profile_dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    request = _load_request_for_profile_command(parser, args)
    try:
        opts = request.to_dispatch_options()
    except Exception as exc:  # noqa: BLE001
        parser.error(str(exc))

    # Profile-based launches may bootstrap the DB before reaching
    # action_dispatch.  Strict G1i therefore needs the same innermost runtime
    # proof here, before that first possible write.
    require_strict_g1i_runtime_attestation(opts)

    if getattr(args, "print_config", False):
        print(launch_request_to_json(request))

    schema_ok = _ensure_db_schema_for_run(bootstrap=not bool(getattr(args, "no_bootstrap_db", False)))
    if not schema_ok:
        return 1

    opts.log_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = opts.log_dir / "resolved_config.json"
    resolved_path.write_text(launch_request_to_json(request) + "\n", encoding="utf-8")
    print(f"resolved_config={resolved_path}")

    if getattr(args, "dry_run", False):
        action_queue(opts)
        return 0
    action_dispatch(opts)
    return 0


def _ensure_db_schema_for_run(*, bootstrap: bool) -> bool:
    report = check_db_schema()
    if report.ok:
        print(render_db_schema_report(report))
        return True
    print(render_db_schema_report(report))
    if not bootstrap:
        return False
    print("bootstrap_db=applying scripts/schema.sql")
    report = bootstrap_db_schema()
    print(render_db_schema_report(report))
    return report.ok


def _dispatch_options_summary(opts: DispatchOptions) -> dict[str, object]:
    return {
        "log_dir": str(opts.log_dir),
        "pid_dir": str(opts.pid_dir),
        "run_log_dir": str(opts.run_log_dir),
        "run_mode": opts.run_mode.value,
        "jobs": list(opts.job_order),
        "job_priority": list(opts.job_priority or ()),
        "only_datasets": list(opts.only_dataset_slugs),
        "skip_datasets": list(opts.skip_dataset_slugs),
        "model_globs": list(opts.model_globs),
        "infer_base_url": opts.inference.base_url,
        "infer_models": list(opts.inference.models),
        "infer_protocol": opts.inference.protocol,
        "infer_max_workers": opts.inference.max_workers,
        "remote_batch_size": opts.inference.remote_batch_size,
        "disable_checker": opts.disable_checker,
    }


__all__ = ["build_parser", "main"]

def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command
    if command == "probe-infer":
        return _run_probe_infer(parser, args)
    if command == "doctor":
        return _run_doctor(parser, args)
    if command == "bootstrap-db":
        return _run_bootstrap_db(args)
    if command == "run":
        return _run_profile_dispatch(parser, args)

    job_list = _resolve_job_list(
        getattr(args, "only_jobs", None),
        getattr(args, "skip_jobs", None),
        getattr(args, "domains", None),
    )
    if not job_list:
        print("⚠️ 未剩余可调度的 job，请检查 --domains / --only-jobs / --skip-jobs 参数设置")
        return 1
    job_priority = _resolve_job_priority(getattr(args, "job_order", None), job_list)

    model_globs = tuple(getattr(args, "models", list(DEFAULT_MODEL_GLOBS)))
    model_globs, infer_base_url, infer_models = _resolve_scheduler_inference_args(
        parser,
        args,
        model_globs=model_globs,
    )
    skip_dataset_slugs = _canonicalize_slugs(parser, getattr(args, "skip_datasets", None))
    benchmark_fields = _parse_benchmark_fields(getattr(args, "benchmark_fields", None))
    only_dataset_slugs = _collect_selected_dataset_slugs(
        parser,
        benchmark_fields=benchmark_fields,
        extra_benchmarks=getattr(args, "extra_benchmarks", None),
        only_datasets=getattr(args, "only_datasets", None),
    )
    model_name_patterns = _compile_model_patterns(parser, getattr(args, "model_regex", None))
    min_param_b = getattr(args, "min_param_b", None)
    max_param_b = getattr(args, "max_param_b", None)
    model_select = getattr(args, "model_select", "all")
    run_mode = _resolve_run_mode(parser, args)

    if command == "queue":
        opts = _dispatch_options_from_args(
            args,
            job_list=job_list,
            job_priority=job_priority,
            model_globs=model_globs,
            skip_dataset_slugs=skip_dataset_slugs,
            only_dataset_slugs=only_dataset_slugs,
            model_name_patterns=model_name_patterns,
            min_param_b=min_param_b,
            max_param_b=max_param_b,
            model_select=model_select,
            run_mode=run_mode,
            infer_base_url=infer_base_url,
            infer_models=infer_models,
        )
        action_queue(opts)
    elif command == "dispatch":
        opts = _dispatch_options_from_args(
            args,
            job_list=job_list,
            job_priority=job_priority,
            model_globs=model_globs,
            skip_dataset_slugs=skip_dataset_slugs,
            only_dataset_slugs=only_dataset_slugs,
            model_name_patterns=model_name_patterns,
            min_param_b=min_param_b,
            max_param_b=max_param_b,
            model_select=model_select,
            run_mode=run_mode,
            infer_base_url=infer_base_url,
            infer_models=infer_models,
        )
        action_dispatch(opts)
    elif command == "status":
        action_status(StatusOptions(pid_dir=Path(args.pid_dir)))
    elif command == "stop":
        job_ids = tuple(str(job) for job in args.job_ids)
        action_stop(StopOptions(pid_dir=Path(args.pid_dir), job_ids=job_ids, stop_all=bool(args.all)))
    elif command == "logs":
        action_logs(
            LogsOptions(
                pid_dir=Path(args.pid_dir),
                run_log_dir=Path(args.run_log_dir),
                tail_lines=int(args.tail_lines),
                rotate_seconds=int(args.rotate_seconds),
            )
        )
    elif command == "serve":
        controller = SchedulerAdminController(state_dir=Path(args.state_dir))
        serve_scheduler_admin(
            host=str(args.host),
            port=int(args.port),
            controller=controller,
            api_key=str(args.admin_api_key) if args.admin_api_key else None,
        )
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
