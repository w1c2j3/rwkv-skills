# RWKV Skills

English | [中文](README.zh-CN.md)

An evaluation scaffold for RWKV7 that targets an external vLLM-RWKV OpenAI-compatible inference service, with dataset preppers for common benchmarks and a GPU scheduler skeleton.

The formal benchmark registry is the Strict46 score matrix: 21 knowledge, 16
maths, 7 coding, and 2 instruction-following benchmarks. Function-calling
integrations are kept in a separate auxiliary catalogue: they remain available
for explicit resolution and scheduler workflows, but are not included in the
default formal benchmark set.

## Project structure
Third-party benchmark data and evaluation artifacts stay out of the source package. The public scoreboard is maintained separately in the Helicopter repository.

- `src/`: backend Python package
  - `src/eval/tasks/`: per-domain evaluation runners/pipelines — `knowledge`, `maths`, `coding`, `instruction_following`, `function_calling`, `agent_bench`.
  - `src/eval/scheduler`: CLI for queueing eval jobs, GPU/remote-worker detection, and dispatch.
  - `src/eval/datasets`: data structures, JSONL loaders, and per-dataset preppers.
  - `src/eval/{evaluating,evaluators,metrics,results,checkers}`: evaluation engine and metric/result handling.
  - `src/infer`: remote OpenAI/vLLM inference client, sampling configuration, and constraints.
  - `src/db`: PostgreSQL data layer.
  - `src/plugins/lexical_chunk_router`: lexical chunking / tool-routing plugin.
  - `src/bin`: console-script entry points registered in pyproject (infer server/fleet/router, perf, download-weights).
- `assets/agent_bench/`: agent-bench (tau_v1 / tau_v2) third-party benchmark data (loaded via sys.path, kept out of the `src` package, force-included at build time).
- `configs/`: per-benchmark `.toml` sampling / evaluation configs.
- `scripts/oneoff/`: one-off / operational scripts.
- `weights`, `data`, `results` (local, gitignored): default locations for model weights, datasets, and evaluation artifacts.

## Requirements
- Python 3.12+. `uv` is recommended for dependency management.
- NVIDIA/AMD GPU for the external vLLM-RWKV server, with CUDA/ROCm matching that server environment.

## Installation
```bash
# Install dependencies (example: CUDA 12.9, matching the torch-cu129 extra in pyproject)
uv sync --extra torch-cu129

# Editable install to expose CLI entry points
uv pip install -e .
```
For other CUDA/CPU builds, use `--extra torch-cu126` / `--extra torch-cpu`, etc.

## Download model weights
`rwkv-download-weights` enumerates and downloads `.pth` weights concurrently from a Hugging Face mirror:
```bash
rwkv-download-weights /path/to/weights
# or add extra repos:
rwkv-download-weights --repo BlinkDL/rwkv7-g1 --repo your/repo
```
You can override the default endpoint/token via environment variables (`HF_ENDPOINT`, `HF_TOKEN`).

## Dataset preparation
Datasets are stored under `data/` by default. You can call the prepper to generate JSONL files:
```bash
uv run python - <<'PY'
from pathlib import Path
from src.eval.datasets.data_prepper.data_manager import prepare_dataset

prepare_dataset("mmlu", Path("data"))  # writes data/mmlu/<split>.jsonl
PY
```
To see supported dataset aliases, check the `available_*_datasets()` family of functions.

## Evaluation & inference example
Start the external vLLM-RWKV server through the wrapper:
```bash
rwkv-skills-infer \
  --model-path /path/to/rwkv7.pth \
  --model-name rwkv7-demo \
  --vllm-rwkv-path ~/GitHub/vllm-rwkv \
  --port 19082
```
Then point eval commands at `--infer-base-url http://127.0.0.1:19082 --infer-model rwkv7-demo --infer-protocol vllm`.

## Scheduler CLI
`rwkv-skills-scheduler` provides commands for queue preview, dispatch, status, stop, and log rotation:
```bash
rwkv-skills-scheduler queue
rwkv-skills-scheduler dispatch --run-log-dir results/logs
```
`queue` is a dry-run of `dispatch` and accepts the same filtering/dispatch flags (including `--overwrite`) to preview what would be scheduled.
By default, jobs that already have scores are skipped.
To force a rerun, pass `--overwrite` on dispatch. This creates a new run/version without deleting historical completion / score / eval records.
By default the evaluator scripts run the LLM wrong-answer checker when configured; to skip it, pass `--disable-checker` on dispatch.

You can re-run only specific benchmarks with `--only-datasets aime24 aime25` (names only; no `_test` suffix), or exclude sets with `--skip-datasets mmlu`. To run only a subset of models, you can filter filenames via `--model-regex '^rwkv7-.*7\\.2b$'` while keeping the default weight glob.

The default model glob is configured in `src/eval/scheduler/config.py` (it only points to `weights/rwkv7-*.pth` within the repo; override as needed). Scheduler jobs now dispatch directly to the field runners:
`src.eval.tasks.knowledge.runner`, `src.eval.tasks.maths.runner`, `src.eval.tasks.coding.runner`, `src.eval.tasks.instruction_following.runner`, `src.eval.tasks.function_calling.runner`.

Formal maths free-response sets whose answers may be mathematically equivalent without textually matching are automatically dispatched to `src.eval.tasks.maths.runner --judge-mode llm`; other free-response tasks use `src.eval.tasks.maths.runner --judge-mode exact`.

Sampling-parameter grid search is handled via the param-search workflow:
- Runner jobs write *all* trial artifacts under `results/param_search/{completions,eval,scores}/{model}/{benchmark}/trial_*.{jsonl,json}`.
- The selector job aggregates `results/param_search/scores/...` across `gsm8k_test` + `hendrycks_math_test` (alias: `math`) and promotes one best shared grid point into the unsuffixed `{benchmark}` paths.

When evaluating the latest 2.9B model, the scheduler automatically runs param-search on `gsm8k_test` + `hendrycks_math_test`.

## HumanEval code generation evaluation
- Dataset prep: `prepare_dataset("human_eval", Path("data"))` downloads the official `HumanEval.jsonl.gz` and writes `data/human_eval/test.jsonl`.
- Run via CLI:
  ```bash
  uv run python -m src.eval.tasks.coding.runner \
    --model-path weights/rwkv7-*.pth \
    --dataset data/human_eval/test.jsonl \
    --benchmark-kind human_eval \
    --cot-mode no_cot \
    --batch-size 128 \
    --eval-timeout 3
  ```
  Samples are written to the evaluation database, and the official unit tests are executed automatically to produce `pass@1` plus the scheduler's derived `avg@k`.

## MBPP code generation evaluation
- Dataset prep: `prepare_dataset("mbpp", Path("data"))` uses the EvalPlus variant MBPP+ and converts 4-space indentation in prompts into tabs.
- Run via CLI:
  ```bash
  uv run python -m src.eval.tasks.coding.runner \
    --model-path weights/rwkv7-*.pth \
    --dataset data/mbpp/test.jsonl \
    --benchmark-kind mbpp \
    --cot-mode no_cot \
    --batch-size 128 \
    --eval-timeout 3
  ```
  Multiple samples are generated and executed against EvalPlus test cases to report `pass@1` plus the scheduler's derived `avg@k`.

## LiveCodeBench code generation evaluation
- Dataset prep: `prepare_dataset("livecodebench", Path("data"))` downloads the LiveCodeBench release_v6 (lite) split and writes `data/livecodebench/test.jsonl` (override with `RWKV_SKILLS_LIVECODEBENCH_VERSION_TAG`).
- Run via CLI:
  ```bash
  uv run python -m src.eval.tasks.coding.runner \
    --model-path weights/rwkv7-*.pth \
    --dataset data/livecodebench/test.jsonl \
    --benchmark-kind livecodebench \
    --cot-mode cot \
    --batch-size 64 \
    --eval-timeout 6 \
    --eval-workers 12
  ```
  LiveCodeBench uses the extracted code blocks for execution and reports `pass@1` plus the scheduler's derived `avg@k`.

## Known gaps / TODO
- Other code benchmarks (BigCodeBench, etc.) are not supported yet.

Contributions are welcome—please implement missing pieces and update the docs accordingly.

## Scheduler-only workflow (DB)
### C.1 One-time setup
1. Prepare PostgreSQL and ensure connectivity (set `.env` / env vars).
   `.env` reference: `.env.example`
2. Prepare model weights under `weights/`, or pass an explicit path through `--models`.
3. Prepare the dataset directory under `data/` (contains task JSONL files).

### C.2 Queue preview
Purpose: verify all 8 entries are schedulable and dataset paths resolve.
```bash
uv run rwkv-skills-scheduler queue \
  --model-select all \
  --models "<MODEL_PATH>" \
  --only-jobs code_human_eval code_livecodebench code_mbpp free_response free_response_judge instruction_following multi_choice_plain multi_choice_cot
```

### C.3 Dispatch
Purpose: run all 8 entries and write into the database.
```bash
uv run rwkv-skills-scheduler dispatch \
  --model-select all \
  --models "<MODEL_PATH>" \
  --only-jobs code_human_eval code_livecodebench code_mbpp free_response free_response_judge instruction_following multi_choice_plain multi_choice_cot \
  --skip-missing-dataset
```

### C.4 Monitor/stop
```bash
rwkv-skills-scheduler status
rwkv-skills-scheduler logs
rwkv-skills-scheduler stop --all
```

### Multi-model resume logic
For each model+dataset(+cot) combination, existing scores are skipped by default. When no score exists, the scheduler resumes the latest unfinished task. If you pass `--overwrite`, the scheduler forces a fresh rerun and writes a new task/version without deleting prior records.
