# RWKV Skills

正式 benchmark 注册表采用 Strict46：21 个知识、16 个数学、7 个代码和 2 个指令遵循 benchmark。Function Calling 集成保留在独立的辅助目录中，仍可显式解析和调度，但不再计入默认正式 benchmark 集合。

[English](README.md) | 中文

面向 RWKV7 的评测脚手架，推理侧接入外部 vLLM-RWKV OpenAI 兼容服务，包含常见评测数据集准备器以及一个 GPU 调度器骨架。

## 项目结构
第三方基准数据与评测产物不入源码包；公开分数前端由 Helicopter 仓库独立维护。

- `src/`：后端 Python 包
  - `src/eval/tasks/`：按域组织的评测 runner/pipeline —— `knowledge`、`maths`、`coding`、`instruction_following`、`function_calling`、`agent_bench`。
  - `src/eval/scheduler`：评测任务排队、GPU/远端 worker 侦测与调度的 CLI。
  - `src/eval/datasets`：数据结构、JSONL 加载器与各数据集准备器。
  - `src/eval/{evaluating,evaluators,metrics,results,checkers}`：评测引擎与指标/结果处理。
  - `src/infer`：远端 OpenAI/vLLM 推理客户端、采样配置与约束。
  - `src/db`：PostgreSQL 数据层。
  - `src/plugins/lexical_chunk_router`：词法分块 / 工具路由插件。
  - `src/bin`：pyproject 注册的 console-script 入口（infer server/fleet/router、perf、download-weights）。
- `assets/agent_bench/`：agent-bench（tau_v1 / tau_v2）第三方基准数据（经 sys.path 加载，不在 `src` 包内，构建时经 force-include 打包）。
- `configs/`：各 benchmark 的 `.toml` 采样 / 评测配置。
- `scripts/oneoff/`：一次性 / 运维脚本。
- `weights`、`data`、`results`（本地、gitignore）：模型权重、数据集与评测产物的默认存放位置。

## 环境要求
- Python 3.12+，推荐安装 `uv` 以管理依赖。
- 外部 vLLM-RWKV 服务需要 NVIDIA/AMD GPU，并与其服务环境中的 CUDA/ROCm 匹配。

## 安装
```bash
# 安装依赖（示例：CUDA 12.9，对应 pyproject 中 torch-cu129 可选项）
uv sync --extra torch-cu129

# 开发模式安装，暴露 CLI 入口
uv pip install -e .
```
如需其他 CUDA/CPU 发行版，请改用 `--extra torch-cu126` / `--extra torch-cpu` 等。

## 下载模型权重
`rwkv-download-weights` 会从 Hugging Face 镜像枚举并并发下载 `.pth` 权重：
```bash
rwkv-download-weights /path/to/weights
# 或指定额外仓库：
rwkv-download-weights --repo BlinkDL/rwkv7-g1 --repo your/repo
```
可通过环境变量覆盖默认镜像与 Token（`HF_ENDPOINT`、`HF_TOKEN`）。

## 数据集准备
数据集默认存放在 `data/`。可以直接调用准备器生成 JSONL：
```bash
uv run python - <<'PY'
from pathlib import Path
from src.eval.datasets.data_prepper.data_manager import prepare_dataset

prepare_dataset("mmlu", Path("data"))  # 会生成 data/mmlu/<split>.jsonl
PY
```
支持的数据集别名可通过 `available_*_datasets()` 系列函数查看。

## 评测与推理示例
通过包装器启动外部 vLLM-RWKV 服务：
```bash
rwkv-skills-infer \
  --model-path /path/to/rwkv7.pth \
  --model-name rwkv7-demo \
  --vllm-rwkv-path ~/GitHub/vllm-rwkv \
  --port 19082
```
评测命令随后使用 `--infer-base-url http://127.0.0.1:19082 --infer-model rwkv7-demo --infer-protocol vllm` 连接该服务。

## 调度器 CLI
`rwkv-skills-scheduler` 暴露了一组命令（队列预览、调度、状态、停止、日志轮播）：
```bash
rwkv-skills-scheduler queue
rwkv-skills-scheduler dispatch --run-log-dir results/logs
```
其中 `queue` 是 `dispatch` 的 dry-run，会接受与 `dispatch` 一致的参数（包含 `--overwrite`）并输出将被调度的任务列表。
默认会跳过已有分数的任务。
若需强制重跑，可在 dispatch 时附上 `--overwrite`；调度器会创建新一轮/新版本结果，不会删除历史 completion / score / eval 记录。
评测脚本在配置好 API_KEY/JUDGE_MODEL 时默认会运行 LLM wrong-answer checker；如需关闭，可在 dispatch 时附上 `--disable-checker`。
可以用 `--only-datasets aime24 aime25` 这类参数仅重测指定 benchmark（名称即可，不需要 `_test` 后缀），也可以用 `--skip-datasets mmlu` 排除特定集合。若想只跑部分模型，无需填写完整路径，可使用 `--model-regex '^rwkv7-.*7\\.2b$'` 等正则过滤模型文件名，配合默认的权重 glob 即可。
默认模型 glob 在 `src/eval/scheduler/config.py` 中配置（仅指向仓库内 `weights/rwkv7-*.pth`，请按需覆盖）。调度器现在直接派发到 field runner：
`src.eval.tasks.knowledge.runner`、`src.eval.tasks.maths.runner`、`src.eval.tasks.coding.runner`、`src.eval.tasks.instruction_following.runner`、`src.eval.tasks.function_calling.runner`。
数学答案可能等价但文本不完全一致的正式 math free-response benchmark 会自动走 `src.eval.tasks.maths.runner --judge-mode llm`，其余 free-response 走 `src.eval.tasks.maths.runner --judge-mode exact`。
采样参数的网格搜索通过 param-search 流程完成：
- runner job 会把完整网格每个 trial 的 completions/eval/scores 写到 `results/param_search/{completions,eval,scores}/{model}/{benchmark}/trial_*.{jsonl,json}`。
- selector job 会统计 `results/param_search/scores/...`（默认综合 `gsm8k_test` + `hendrycks_math_test`，其中 `math` 会自动映射到 `hendrycks_math_test`），并把唯一最佳格点复制/写入到不带后缀的 `{benchmark}` 产物路径。

调度器在评测最新 2.9B 模型时，会自动对 `gsm8k_test` + `hendrycks_math_test` 启用 param-search。

## HumanEval 代码生成评测
- 数据集准备：`prepare_dataset("human_eval", Path("data"))` 会下载官方 `HumanEval.jsonl.gz` 并写出 `data/human_eval/test.jsonl`。
- 直接运行 CLI：
  ```bash
  uv run python -m src.eval.tasks.coding.runner \
    --model-path weights/rwkv7-*.pth \
    --dataset data/human_eval/test.jsonl \
    --benchmark-kind human_eval \
    --cot-mode no_cot \
    --batch-size 128 \
    --eval-timeout 3
  ```
  结果会写入评估数据库，并自动执行官方测试用例输出 `pass@1` 与调度器派生的 `avg@k`。

## MBPP 代码生成评测
- 数据集准备：`prepare_dataset("mbpp", Path("data"))` 会使用 EvalPlus 版本的 MBPP+，并将 prompt 中的 4 空格转换为制表符。
- 运行 CLI：
  ```bash
  uv run python -m src.eval.tasks.coding.runner \
    --model-path weights/rwkv7-*.pth \
    --dataset data/mbpp/test.jsonl \
    --benchmark-kind mbpp \
    --cot-mode no_cot \
    --batch-size 128 \
    --eval-timeout 3
  ```
  会生成多样本并用 EvalPlus 测试用例执行，输出 `pass@1` 与调度器派生的 `avg@k`。

## LiveCodeBench 代码生成评测
- 数据集准备：`prepare_dataset("livecodebench", Path("data"))` 会下载 LiveCodeBench release_v6（lite）并写出 `data/livecodebench/test.jsonl`（可用 `RWKV_SKILLS_LIVECODEBENCH_VERSION_TAG` 覆盖版本）。
- 运行 CLI：
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
  会抽取代码块并执行 LiveCodeBench 测试，输出 `pass@1` 与调度器派生的 `avg@k`。

## 已知缺口 / TODO
- 尚未支持其他代码基准（BigCodeBench 等）。

欢迎根据上述缺口补全实现并更新文档。

## C. 仅使用调度器（DB）
### C.1 一次性准备
1. 准备 PostgreSQL 并可连接（写好 .env / 环境变量）  
   .env参考 `.env.example`
2. 准备模型权重到 `weights/`，或通过 `--models` 传入明确路径。
3. 准备 `data/` 数据集目录（包含各任务 JSONL）。

### C.2 调度器队列预览
用途：确认 8 个入口都会被调度、数据集路径可解析。
```bash
uv run rwkv-skills-scheduler queue \
  --model-select all \
  --models "<MODEL_PATH>" \
  --only-jobs code_human_eval code_livecodebench code_mbpp free_response free_response_judge instruction_following multi_choice_plain multi_choice_cot
```

### C.3 执行调度
用途：实际运行 8 个入口并写入数据库。
```bash
uv run rwkv-skills-scheduler dispatch \
  --model-select all \
  --models "<MODEL_PATH>" \
  --only-jobs code_human_eval code_livecodebench code_mbpp free_response free_response_judge instruction_following multi_choice_plain multi_choice_cot \
  --skip-missing-dataset
```

### C.4 监控/停止
```bash
rwkv-skills-scheduler status
rwkv-skills-scheduler logs
rwkv-skills-scheduler stop --all
```

### 多模型续跑逻辑
以 `model + dataset(+cot)` 为单位判断：默认会跳过已有分数；若无分数则续跑最近的未完成 task。若传 `--overwrite`，则会强制新建 task 重跑并写入新版本，不删除历史记录。任务失败会标记为 `failed`，下次调度在未产出分数前仍会续跑该 task。
