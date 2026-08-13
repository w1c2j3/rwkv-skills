"""Benchmark metadata aligned with rwkv-rs' evaluator matrix."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from src.eval.execution_plan import TARGET_EVAL_ATTEMPTS
from src.eval.scheduler.dataset_utils import canonical_slug, make_dataset_slug, safe_slug, split_benchmark_and_split


AUTO_TARGET_ATTEMPTS = TARGET_EVAL_ATTEMPTS


class BenchmarkField(str, Enum):
    KNOWLEDGE = "knowledge"
    MATHS = "maths"
    CODING = "coding"
    INSTRUCTION_FOLLOWING = "instruction_following"
    FUNCTION_CALLING = "function_calling"


class CoTMode(str, Enum):
    NO_COT = "no_cot"
    COT = "cot"

    @property
    def is_cot(self) -> bool:
        return self is not CoTMode.NO_COT


@dataclass(frozen=True, slots=True)
class BenchmarkMetadata:
    name: str
    field: BenchmarkField
    cot_modes: tuple[CoTMode, ...]
    default_split: str = "test"
    dataset_name: str | None = None
    scheduler_jobs: tuple[str, ...] = ()
    n_shots: tuple[int, ...] = (0,)
    # Empty avg_ks means "derive avg@k automatically from dataset size":
    # run the benchmark once, unless it is larger than target_eval_attempts.
    avg_ks: tuple[float, ...] = ()
    pass_ks: tuple[int, ...] = ()
    target_eval_attempts: int = AUTO_TARGET_ATTEMPTS

    @property
    def dataset(self) -> str:
        return self.dataset_name or self.name


_TWO_MODE_KNOWLEDGE = (CoTMode.NO_COT, CoTMode.COT)
_COT_ONLY = (CoTMode.COT,)
_NO_COT_ONLY = (CoTMode.NO_COT,)

_MULTI_CHOICE_JOBS = (
    "multi_choice_plain",
    "multi_choice_cot",
    "multi_choice_plain_naive",
    "multi_choice_cot_naive",
)
_FREE_RESPONSE_JOBS = ("free_response",)
_FREE_RESPONSE_NAIVE_JOBS = (
    "free_response",
    "free_response_naive",
    "free_response_plain",
    "free_response_plain_naive",
)
_FREE_RESPONSE_JUDGE_JOBS = ("free_response_judge",)
_FREE_RESPONSE_JUDGE_NAIVE_JOBS = (
    "free_response_judge",
    "free_response_judge_naive",
    "free_response_judge_plain",
    "free_response_judge_plain_naive",
)
_HUMAN_EVAL_JOBS = ("code_human_eval",)
_HUMAN_EVAL_NAIVE_JOBS = ("code_human_eval", "code_human_eval_naive")
_MBPP_JOBS = ("code_mbpp",)
_MBPP_NAIVE_JOBS = ("code_mbpp", "code_mbpp_naive")
_LIVECODEBENCH_JOBS = ("code_livecodebench",)
_LIVECODEBENCH_NAIVE_JOBS = (
    "code_livecodebench",
    "code_livecodebench_naive",
    "code_livecodebench_plain",
    "code_livecodebench_plain_naive",
)
_INSTRUCTION_FOLLOWING_JOBS = ("instruction_following", "instruction_following_naive")
_AGENT_TOOL_CALL_JOBS = ("function_agent_tool_call",)
_AGENT_LOOP_JOBS = ("function_agent_loop",)
_BROWSECOMP_JOBS = ("function_browsecomp",)
_COMPLEXFUNCBENCH_JOBS = ("function_complexfuncbench",)
_LONGBENCH_JOBS = ("function_longbench",)
_LONGCODEBENCH_JOBS = ("function_longcodebench",)
_MCP_BENCH_JOBS = ("function_mcp_bench",)
_API_BANK_JOBS = ("function_api_bank",)
_AGENTBENCH_JOBS = ("function_agentbench",)
_BFCL_AST_JOBS = ("function_bfcl_ast",)
_BFCL_EXEC_JOBS = ("function_bfcl_exec",)
_BFCL_V3_JOBS = ("function_bfcl_v3",)
_TOOLALPACA_JOBS = ("function_toolalpaca",)
_BROWSECOMP_PLUS_JOBS = ("function_browsecomp_plus",)
_TAU_BENCH_JOBS = ("function_tau_bench",)
_TAU2_BENCH_JOBS = ("function_tau2_bench",)
_TAU3_BENCH_JOBS = ("function_tau3_bench",)


def _metadata(
    name: str,
    *,
    field: BenchmarkField,
    cot_modes: tuple[CoTMode, ...],
    default_split: str = "test",
    dataset_name: str | None = None,
    scheduler_jobs: tuple[str, ...] = (),
    n_shots: tuple[int, ...] = (0,),
    avg_ks: tuple[float, ...] = (),
    pass_ks: tuple[int, ...] = (),
) -> BenchmarkMetadata:
    return BenchmarkMetadata(
        name=safe_slug(name).lower(),
        field=field,
        cot_modes=cot_modes,
        default_split=default_split,
        dataset_name=safe_slug(dataset_name).lower() if dataset_name else None,
        scheduler_jobs=scheduler_jobs,
        n_shots=n_shots,
        avg_ks=avg_ks,
        pass_ks=pass_ks,
    )


def _knowledge(
    name: str,
    *,
    default_split: str = "test",
    dataset_name: str | None = None,
) -> BenchmarkMetadata:
    return _metadata(
        name,
        field=BenchmarkField.KNOWLEDGE,
        cot_modes=_TWO_MODE_KNOWLEDGE,
        default_split=default_split,
        dataset_name=dataset_name,
        scheduler_jobs=_MULTI_CHOICE_JOBS,
    )


def _math(
    name: str,
    *,
    default_split: str = "test",
    dataset_name: str | None = None,
    scheduler_jobs: tuple[str, ...] = _FREE_RESPONSE_NAIVE_JOBS,
) -> BenchmarkMetadata:
    return _metadata(
        name,
        field=BenchmarkField.MATHS,
        cot_modes=_TWO_MODE_KNOWLEDGE,
        default_split=default_split,
        dataset_name=dataset_name,
        scheduler_jobs=scheduler_jobs,
    )


def _coding_human_eval(name: str, *, dataset_name: str | None = None) -> BenchmarkMetadata:
    return _metadata(
        name,
        field=BenchmarkField.CODING,
        cot_modes=_NO_COT_ONLY,
        dataset_name=dataset_name,
        scheduler_jobs=_HUMAN_EVAL_NAIVE_JOBS,
    )


def _coding_mbpp(name: str, *, dataset_name: str | None = None) -> BenchmarkMetadata:
    return _metadata(
        name,
        field=BenchmarkField.CODING,
        cot_modes=_NO_COT_ONLY,
        dataset_name=dataset_name,
        scheduler_jobs=_MBPP_NAIVE_JOBS,
    )


def _coding_livecodebench(name: str, *, dataset_name: str | None = None) -> BenchmarkMetadata:
    return _metadata(
        name,
        field=BenchmarkField.CODING,
        cot_modes=_TWO_MODE_KNOWLEDGE,
        dataset_name=dataset_name,
        scheduler_jobs=_LIVECODEBENCH_NAIVE_JOBS,
    )


def _instruction_following(
    name: str,
    *,
    default_split: str = "test",
    dataset_name: str | None = None,
    scheduler_jobs: tuple[str, ...] = _INSTRUCTION_FOLLOWING_JOBS,
) -> BenchmarkMetadata:
    return _metadata(
        name,
        field=BenchmarkField.INSTRUCTION_FOLLOWING,
        cot_modes=_NO_COT_ONLY,
        default_split=default_split,
        dataset_name=dataset_name,
        scheduler_jobs=scheduler_jobs,
    )


def _function_calling(
    name: str,
    *,
    default_split: str = "test",
    dataset_name: str | None = None,
    scheduler_jobs: tuple[str, ...],
) -> BenchmarkMetadata:
    return _metadata(
        name,
        field=BenchmarkField.FUNCTION_CALLING,
        cot_modes=_COT_ONLY,
        default_split=default_split,
        dataset_name=dataset_name,
        scheduler_jobs=scheduler_jobs,
    )


_EXPLICIT_METADATA: dict[str, BenchmarkMetadata] = {
    # Knowledge
    canonical_slug("arc_easy"): _knowledge("arc_easy"),
    canonical_slug("mmlu"): _knowledge("mmlu"),
    canonical_slug("openbookqa"): _knowledge("openbookqa"),
    canonical_slug("cmmlu"): _knowledge("cmmlu"),
    canonical_slug("commonsense_qa"): _knowledge("commonsense_qa", default_split="validation"),
    canonical_slug("ceval"): _knowledge("ceval"),
    canonical_slug("truthfulqa_mc1"): _knowledge("truthfulqa_mc1", default_split="validation"),
    canonical_slug("mmlu_pro"): _knowledge("mmlu_pro"),
    canonical_slug("hellaswag"): _knowledge("hellaswag", default_split="validation"),
    canonical_slug("mmlu_redux"): _knowledge("mmlu_redux"),
    canonical_slug("winogrande"): _knowledge("winogrande", default_split="validation"),
    canonical_slug("agieval_mcq"): _knowledge("agieval_mcq"),
    canonical_slug("mmlu_sr_question_and_answer"): _knowledge("mmlu_sr_question_and_answer"),
    canonical_slug("bbh_mcq"): _knowledge("bbh_mcq"),
    canonical_slug("kmmlu"): _knowledge("kmmlu"),
    canonical_slug("gpqa_main"): _knowledge("gpqa_main", dataset_name="gpqa", default_split="main"),
    canonical_slug("gpqa_extended"): _knowledge("gpqa_extended", dataset_name="gpqa", default_split="extended"),
    canonical_slug("medqa"): _knowledge("medqa"),
    canonical_slug("gpqa_diamond"): _knowledge("gpqa_diamond", dataset_name="gpqa", default_split="diamond"),
    canonical_slug("medmcqa"): _knowledge("medmcqa", default_split="validation"),
    canonical_slug("arc_challenge"): _knowledge("arc_challenge"),
    # Maths / free response
    canonical_slug("aime24"): _math("aime24"),
    canonical_slug("aime25"): _math("aime25"),
    canonical_slug("amc23"): _math("amc23", scheduler_jobs=_FREE_RESPONSE_JUDGE_NAIVE_JOBS),
    canonical_slug("answer_judge"): _math("answer_judge"),
    canonical_slug("beyond_aime"): _math("beyond_aime"),
    canonical_slug("brumo25"): _math("brumo25"),
    canonical_slug("comp_math_24_25"): _math("comp_math_24_25", scheduler_jobs=_FREE_RESPONSE_JUDGE_NAIVE_JOBS),
    canonical_slug("gaokao2023en"): _math("gaokao2023en", scheduler_jobs=_FREE_RESPONSE_JUDGE_NAIVE_JOBS),
    canonical_slug("gsm8k"): _math("gsm8k"),
    canonical_slug("hmmt_feb25"): _math("hmmt_feb25"),
    canonical_slug("math_500"): _math("math_500"),
    canonical_slug("math_odyssey"): _math("math_odyssey"),
    canonical_slug("minerva_math"): _math("minerva_math", scheduler_jobs=_FREE_RESPONSE_JUDGE_NAIVE_JOBS),
    canonical_slug("olympiadbench"): _math("olympiadbench"),
    canonical_slug("simpleqa"): _math("simpleqa", default_split="verified"),
    canonical_slug("svamp"): _math("svamp"),
    # Coding
    canonical_slug("human_eval"): _coding_human_eval("human_eval"),
    canonical_slug("human_eval_cn"): _coding_human_eval("human_eval_cn"),
    canonical_slug("human_eval_fix"): _coding_human_eval("human_eval_fix"),
    canonical_slug("human_eval_plus"): _coding_human_eval("human_eval_plus"),
    canonical_slug("mbpp"): _coding_mbpp("mbpp"),
    canonical_slug("mbpp_plus"): _coding_mbpp("mbpp_plus"),
    canonical_slug("livecodebench"): _coding_livecodebench("livecodebench"),
    # Instruction following
    canonical_slug("ifeval"): _instruction_following("ifeval"),
    canonical_slug("ifbench"): _instruction_following("ifbench"),
    # Function calling
    canonical_slug("terminal_bench_2_1"): _function_calling(
        "terminal_bench_2_1",
        scheduler_jobs=_AGENT_LOOP_JOBS,
    ),
    canonical_slug("nl2repo"): _function_calling("nl2repo", scheduler_jobs=_AGENT_LOOP_JOBS),
    canonical_slug("deepswe"): _function_calling("deepswe", scheduler_jobs=_AGENT_LOOP_JOBS),
    canonical_slug("hy_backend_2_0"): _function_calling(
        "hy_backend_2_0",
        scheduler_jobs=_AGENT_LOOP_JOBS,
    ),
    canonical_slug("hy_swe_max"): _function_calling("hy_swe_max", scheduler_jobs=_AGENT_LOOP_JOBS),
    canonical_slug("hy_companybench"): _function_calling(
        "hy_companybench",
        scheduler_jobs=_AGENT_LOOP_JOBS,
    ),
    canonical_slug("browsecomp"): _function_calling("browsecomp", scheduler_jobs=_BROWSECOMP_JOBS),
    canonical_slug("browsecomp_zh"): _function_calling("browsecomp_zh", scheduler_jobs=_BROWSECOMP_JOBS),
    canonical_slug("widesearch"): _function_calling("widesearch", scheduler_jobs=_AGENT_LOOP_JOBS),
    canonical_slug("deepsearchqa"): _function_calling("deepsearchqa", scheduler_jobs=_AGENT_LOOP_JOBS),
    canonical_slug("mcp_atlas"): _function_calling("mcp_atlas", scheduler_jobs=_AGENT_LOOP_JOBS),
    canonical_slug("toolathlon"): _function_calling("toolathlon", scheduler_jobs=_AGENT_LOOP_JOBS),
    canonical_slug("apex_agents"): _function_calling("apex_agents", scheduler_jobs=_AGENT_LOOP_JOBS),
    canonical_slug("claweval"): _function_calling("claweval", scheduler_jobs=_AGENT_LOOP_JOBS),
    canonical_slug("wildclawbench"): _function_calling("wildclawbench", scheduler_jobs=_AGENT_LOOP_JOBS),
    canonical_slug("skillsbench"): _function_calling("skillsbench", scheduler_jobs=_AGENT_LOOP_JOBS),
    canonical_slug("e_bench"): _function_calling("e_bench", scheduler_jobs=_AGENT_LOOP_JOBS),
    canonical_slug("hy_finmodelbench"): _function_calling(
        "hy_finmodelbench",
        scheduler_jobs=_AGENT_LOOP_JOBS,
    ),
    canonical_slug("prodbench"): _function_calling("prodbench", scheduler_jobs=_AGENT_LOOP_JOBS),
    canonical_slug("hy_skillsworld"): _function_calling(
        "hy_skillsworld",
        scheduler_jobs=_AGENT_LOOP_JOBS,
    ),
    canonical_slug("hle_with_tools"): _function_calling(
        "hle_with_tools",
        scheduler_jobs=_AGENT_LOOP_JOBS,
    ),
    canonical_slug("hy_euler_pro"): _function_calling("hy_euler_pro", scheduler_jobs=_AGENT_LOOP_JOBS),
    canonical_slug("browsecomp_plus"): _function_calling(
        "browsecomp_plus",
        scheduler_jobs=_BROWSECOMP_PLUS_JOBS,
    ),
    canonical_slug("complexfuncbench_official"): _function_calling(
        "complexfuncbench_official",
        scheduler_jobs=_COMPLEXFUNCBENCH_JOBS,
    ),
    canonical_slug("complexfuncbench_subset"): _function_calling(
        "complexfuncbench_subset",
        scheduler_jobs=_COMPLEXFUNCBENCH_JOBS,
    ),
    canonical_slug("longbench"): _function_calling("longbench", scheduler_jobs=_LONGBENCH_JOBS),
    canonical_slug("longbench_qa"): _function_calling("longbench_qa", scheduler_jobs=_LONGBENCH_JOBS),
    canonical_slug("longbench_qa_balanced"): _function_calling(
        "longbench_qa_balanced",
        scheduler_jobs=_LONGBENCH_JOBS,
    ),
    canonical_slug("longcodeqa"): _function_calling("longcodeqa", scheduler_jobs=_LONGCODEBENCH_JOBS),
    canonical_slug("mcp_bench"): _function_calling("mcp_bench", scheduler_jobs=_MCP_BENCH_JOBS),
    canonical_slug("mcp_bench_single"): _function_calling("mcp_bench_single", scheduler_jobs=_MCP_BENCH_JOBS),
    canonical_slug("mcp_bench_multi_2server"): _function_calling(
        "mcp_bench_multi_2server",
        scheduler_jobs=_MCP_BENCH_JOBS,
    ),
    canonical_slug("mcp_bench_multi_3server"): _function_calling(
        "mcp_bench_multi_3server",
        scheduler_jobs=_MCP_BENCH_JOBS,
    ),
    canonical_slug("apibank_l1"): _function_calling("apibank_l1", scheduler_jobs=_API_BANK_JOBS),
    canonical_slug("apibank_l2"): _function_calling("apibank_l2", scheduler_jobs=_API_BANK_JOBS),
    canonical_slug("apibank_level1"): _function_calling("apibank_level1", scheduler_jobs=_API_BANK_JOBS),
    canonical_slug("apibank_level2"): _function_calling("apibank_level2", scheduler_jobs=_API_BANK_JOBS),
    canonical_slug("agentbench_db"): _function_calling("agentbench_db", scheduler_jobs=_AGENTBENCH_JOBS),
    canonical_slug("agentbench_kg"): _function_calling("agentbench_kg", scheduler_jobs=_AGENTBENCH_JOBS),
    canonical_slug("bfcl_simple_python"): _function_calling("bfcl_simple_python", scheduler_jobs=_BFCL_AST_JOBS),
    canonical_slug("bfcl_exec_simple_ast"): _function_calling("bfcl_exec_simple_ast", scheduler_jobs=_BFCL_AST_JOBS),
    canonical_slug("bfcl_multiple"): _function_calling("bfcl_multiple", scheduler_jobs=_BFCL_AST_JOBS),
    canonical_slug("bfcl_exec_multiple_ast"): _function_calling("bfcl_exec_multiple_ast", scheduler_jobs=_BFCL_AST_JOBS),
    canonical_slug("bfcl_exec_simple"): _function_calling("bfcl_exec_simple", scheduler_jobs=_BFCL_EXEC_JOBS),
    canonical_slug("bfcl_exec_multiple"): _function_calling("bfcl_exec_multiple", scheduler_jobs=_BFCL_EXEC_JOBS),
    canonical_slug("bfcl_exec_parallel"): _function_calling("bfcl_exec_parallel", scheduler_jobs=_BFCL_EXEC_JOBS),
    canonical_slug("bfcl_exec_parallel_multiple"): _function_calling(
        "bfcl_exec_parallel_multiple",
        scheduler_jobs=_BFCL_EXEC_JOBS,
    ),
    canonical_slug("bfcl_v3"): _function_calling("bfcl_v3", scheduler_jobs=_BFCL_V3_JOBS),
    canonical_slug("toolalpaca_eval_simulated"): _function_calling(
        "toolalpaca_eval_simulated",
        scheduler_jobs=_TOOLALPACA_JOBS,
    ),
    canonical_slug("toolalpaca_eval_real"): _function_calling(
        "toolalpaca_eval_real",
        scheduler_jobs=_TOOLALPACA_JOBS,
    ),
    canonical_slug("tau_bench_retail"): _function_calling("tau_bench_retail", scheduler_jobs=_TAU_BENCH_JOBS),
    canonical_slug("tau_bench_airline"): _function_calling("tau_bench_airline", scheduler_jobs=_TAU_BENCH_JOBS),
    canonical_slug("tau_bench_telecom"): _function_calling("tau_bench_telecom", scheduler_jobs=_TAU_BENCH_JOBS),
    canonical_slug("tau2_bench_retail"): _function_calling(
        "tau2_bench_retail",
        default_split="base",
        scheduler_jobs=_TAU2_BENCH_JOBS,
    ),
    canonical_slug("tau2_bench_airline"): _function_calling(
        "tau2_bench_airline",
        default_split="base",
        scheduler_jobs=_TAU2_BENCH_JOBS,
    ),
    canonical_slug("tau2_bench_telecom"): _function_calling(
        "tau2_bench_telecom",
        default_split="base",
        scheduler_jobs=_TAU2_BENCH_JOBS,
    ),
    canonical_slug("tau3_bench_airline"): _function_calling(
        "tau3_bench_airline",
        default_split="base",
        scheduler_jobs=_TAU3_BENCH_JOBS,
    ),
    canonical_slug("tau3_bench_retail"): _function_calling(
        "tau3_bench_retail",
        default_split="base",
        scheduler_jobs=_TAU3_BENCH_JOBS,
    ),
    canonical_slug("tau3_bench_telecom"): _function_calling(
        "tau3_bench_telecom",
        default_split="base",
        scheduler_jobs=_TAU3_BENCH_JOBS,
    ),
    canonical_slug("tau3_bench_banking_knowledge"): _function_calling(
        "tau3_bench_banking_knowledge",
        default_split="base",
        scheduler_jobs=_TAU3_BENCH_JOBS,
    ),
    canonical_slug("tau3_bench_mock"): _function_calling(
        "tau3_bench_mock",
        default_split="base",
        scheduler_jobs=_TAU3_BENCH_JOBS,
    ),
    canonical_slug("tau3_bench_mock_long_context"): _function_calling(
        "tau3_bench_mock_long_context",
        default_split="base",
        scheduler_jobs=_TAU3_BENCH_JOBS,
    ),
}

BENCHMARK_ALIASES: dict[str, tuple[str, ...]] = {
    canonical_slug("gpqa"): (
        canonical_slug("gpqa_main"),
        canonical_slug("gpqa_extended"),
        canonical_slug("gpqa_diamond"),
    ),
    canonical_slug("mmlu_sr"): (
        canonical_slug("mmlu_sr_question_and_answer"),
    ),
    canonical_slug("longcodebench"): (canonical_slug("longcodeqa"),),
    canonical_slug("terminal_bench"): (canonical_slug("terminal_bench_2_1"),),
    canonical_slug("terminal_bench_2.1"): (canonical_slug("terminal_bench_2_1"),),
    canonical_slug("deep_swe"): (canonical_slug("deepswe"),),
    canonical_slug("apex_agent"): (canonical_slug("apex_agents"),),
    canonical_slug("apex_agents"): (canonical_slug("apex_agents"),),
    canonical_slug("apex-agents"): (canonical_slug("apex_agents"),),
    canonical_slug("apex-agent"): (canonical_slug("apex_agents"),),
    canonical_slug("hle_tools"): (canonical_slug("hle_with_tools"),),
    canonical_slug("claw_eval"): (canonical_slug("claweval"),),
    canonical_slug("wide_search"): (canonical_slug("widesearch"),),
    canonical_slug("apibank"): (
        canonical_slug("apibank_l1"),
        canonical_slug("apibank_l2"),
        canonical_slug("apibank_level1"),
        canonical_slug("apibank_level2"),
    ),
    canonical_slug("tau_bench"): (
        canonical_slug("tau_bench_retail"),
        canonical_slug("tau_bench_airline"),
        canonical_slug("tau_bench_telecom"),
    ),
    canonical_slug("tau2_bench"): (
        canonical_slug("tau2_bench_retail"),
        canonical_slug("tau2_bench_airline"),
        canonical_slug("tau2_bench_telecom"),
    ),
    canonical_slug("tau3_bench"): (
        canonical_slug("tau3_bench_retail"),
        canonical_slug("tau3_bench_airline"),
        canonical_slug("tau3_bench_telecom"),
        canonical_slug("tau3_bench_banking_knowledge"),
        canonical_slug("tau3_bench_mock"),
        canonical_slug("tau3_bench_mock_long_context"),
    ),
}

_PREFIX_FALLBACKS: tuple[tuple[tuple[str, ...], BenchmarkMetadata], ...] = (
    (
        ("mmlu", "cmmlu", "ceval"),
        _knowledge("knowledge"),
    ),
    (
        ("human_eval",),
        _coding_human_eval("human_eval"),
    ),
    (
        ("mbpp",),
        _coding_mbpp("mbpp"),
    ),
    (
        ("livecodebench",),
        _coding_livecodebench("livecodebench"),
    ),
    (("ifeval", "ifbench"), _instruction_following("instruction_following")),
    (
        ("browsecomp",),
        _function_calling("browsecomp", scheduler_jobs=_BROWSECOMP_JOBS),
    ),
    (
        ("mcp_bench",),
        _function_calling("mcp_bench", scheduler_jobs=_MCP_BENCH_JOBS),
    ),
    (
        ("complexfuncbench",),
        _function_calling("complexfuncbench_official", scheduler_jobs=_COMPLEXFUNCBENCH_JOBS),
    ),
)

KNOWN_BENCHMARKS: tuple[BenchmarkMetadata, ...] = tuple(
    sorted(_EXPLICIT_METADATA.values(), key=lambda item: (item.field.value, item.name))
)

# The formal score matrix is the Strict46 suite.  Function-calling integrations
# remain explicitly resolvable and schedulable through the auxiliary catalogue,
# but they are not part of the default/formal benchmark registry.
ALL_BENCHMARKS: tuple[BenchmarkMetadata, ...] = tuple(
    item for item in KNOWN_BENCHMARKS if item.field is not BenchmarkField.FUNCTION_CALLING
)

AUXILIARY_BENCHMARKS: tuple[BenchmarkMetadata, ...] = tuple(
    item for item in KNOWN_BENCHMARKS if item.field is BenchmarkField.FUNCTION_CALLING
)

BENCHMARKS_BY_FIELD: dict[BenchmarkField, tuple[BenchmarkMetadata, ...]] = {
    field: tuple(item for item in ALL_BENCHMARKS if item.field is field)
    for field in BenchmarkField
}

KNOWN_BENCHMARKS_BY_FIELD: dict[BenchmarkField, tuple[BenchmarkMetadata, ...]] = {
    field: tuple(item for item in KNOWN_BENCHMARKS if item.field is field)
    for field in BenchmarkField
}

_EXPLICIT_BY_DATASET_SLUG: dict[str, BenchmarkMetadata] = {
    canonical_slug(make_dataset_slug(item.dataset, item.default_split)): item
    for item in KNOWN_BENCHMARKS
}


def _resolve_single_alias(name: str) -> BenchmarkMetadata | None:
    target_names = BENCHMARK_ALIASES.get(name)
    if not target_names or len(target_names) != 1:
        return None
    return _EXPLICIT_METADATA[target_names[0]]


def expand_benchmark_alias(raw_name: str) -> tuple[str, ...]:
    slug = canonical_slug(raw_name)
    if slug in _EXPLICIT_METADATA:
        return (slug,)
    if slug in _EXPLICIT_BY_DATASET_SLUG:
        return (_EXPLICIT_BY_DATASET_SLUG[slug].name,)
    targets = BENCHMARK_ALIASES.get(slug)
    if targets:
        return targets

    benchmark_name, _ = split_benchmark_and_split(slug)
    if benchmark_name in _EXPLICIT_METADATA:
        return (benchmark_name,)
    if benchmark_name in _EXPLICIT_BY_DATASET_SLUG:
        return (_EXPLICIT_BY_DATASET_SLUG[benchmark_name].name,)
    targets = BENCHMARK_ALIASES.get(benchmark_name)
    if targets:
        return targets
    return tuple()


def resolve_benchmark_metadata(dataset_slug: str) -> BenchmarkMetadata:
    slug = canonical_slug(dataset_slug)
    explicit = _EXPLICIT_METADATA.get(slug)
    if explicit is not None:
        return explicit

    alias_target = _resolve_single_alias(slug)
    if alias_target is not None:
        return alias_target

    explicit = _EXPLICIT_BY_DATASET_SLUG.get(slug)
    if explicit is not None:
        return explicit

    benchmark_name, _ = split_benchmark_and_split(slug)
    explicit = _EXPLICIT_METADATA.get(benchmark_name)
    if explicit is not None:
        return explicit

    alias_target = _resolve_single_alias(benchmark_name)
    if alias_target is not None:
        return alias_target

    for prefixes, template in _PREFIX_FALLBACKS:
        if benchmark_name.startswith(prefixes):
            return replace(template, name=benchmark_name)

    return _math(benchmark_name)


def default_split_for_benchmark(dataset_slug: str) -> str:
    return resolve_benchmark_metadata(dataset_slug).default_split


def get_benchmarks_with_field(field: BenchmarkField) -> tuple[BenchmarkMetadata, ...]:
    return BENCHMARKS_BY_FIELD.get(field, ())


def scheduler_jobs_for_benchmark(dataset_slug: str) -> tuple[str, ...]:
    return resolve_benchmark_metadata(dataset_slug).scheduler_jobs


def supports_cot_mode(dataset_slug: str, cot_mode: CoTMode) -> bool:
    return cot_mode in resolve_benchmark_metadata(dataset_slug).cot_modes


__all__ = [
    "AUTO_TARGET_ATTEMPTS",
    "ALL_BENCHMARKS",
    "AUXILIARY_BENCHMARKS",
    "BENCHMARK_ALIASES",
    "BenchmarkField",
    "BenchmarkMetadata",
    "BENCHMARKS_BY_FIELD",
    "CoTMode",
    "default_split_for_benchmark",
    "expand_benchmark_alias",
    "get_benchmarks_with_field",
    "KNOWN_BENCHMARKS",
    "KNOWN_BENCHMARKS_BY_FIELD",
    "resolve_benchmark_metadata",
    "scheduler_jobs_for_benchmark",
    "supports_cot_mode",
]
