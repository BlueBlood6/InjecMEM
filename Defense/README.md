# <div align="center"> InjecMEM Defense Evaluation </div>

> This directory contains the optional defense evaluation code for InjecMEM.

# 👉 Setup

Copy the defense files into `MemorySystem/` from the repository root:

```bash
cp Defense/defense.py MemorySystem/defense.py
cp Defense/inject_generate.py MemorySystem/inject_generate.py
cp Defense/memoryos.py MemorySystem/memoryos.py
cp Defense/simple_run.sh MemorySystem/simple_run.sh
```

# 🔥 Quick Start

Start the backbone and embedding services described in the repository README and keep both services running.
Run the defense example from the repository root:

```bash
bash MemorySystem/simple_run.sh
```

Use a fresh output directory for each run:

```bash
MEMPATH=./MemorySystem/memory_base/defense_run \
bash MemorySystem/simple_run.sh
```

# ⚙️ Configuration

`MemorySystem/simple_run.sh` accepts the following environment variables:

| Variable | Default | Description |
|---|---:|---|
| `MEMPATH` | `./MemorySystem/memory_base/run_1` | Memory state, logs, and report directory |
| `BACKEND` | `llm_judge` | `llm_judge`, `perplexity`, `promptguard`, or `protectai` |
| `MINJA_TH` | `0.5` | Global detector threshold |
| `LLMJUDGE_TH` | `0.5` | LLM-judge threshold |
| `PROMPTGUARD_TH` | `0.5` | Prompt Guard threshold |
| `PROTECTAI_TH` | `0.5` | ProtectAI threshold |
| `PPL_TH` | `40` | Perplexity threshold |
| `DEF_STEP0` | `1` | Apply defense during benign initialization |
| `DEF_WRITE` | `0` | Apply defense before subsequent memory writes |
| `DEF_RETRIEVE` | `1` | Apply defense to retrieved memories |
| `PPL_MAX` | `512` | Detector token window or truncation length |
| `PPL_STRIDE` | `256` | Perplexity window stride |
| `PPL_MAXWIN` | `0` | Use maximum-window (`1`) or mean-window (`0`) perplexity |

For `llm_judge`, `promptguard`, and `protectai`, an input is detected when its risk score is greater than or equal to the selected threshold.
For `perplexity`, an input is detected when its perplexity is greater than or equal to `PPL_TH`.
Lower thresholds are more sensitive, while higher thresholds are more permissive.

## Model Checkpoints and Device

| Variable | Default | Description |
|---|---|---|
| `MINJA_JUDGE_MODEL` | `./models/Qwen2.5-0.5B-Instruct` | Local path or model identifier for `llm_judge` |
| `MINJA_PPL_MODEL` | `distilgpt2` | Model identifier or local path for `perplexity` |
| `PROMPTGUARD_MODEL` | `meta-llama/Llama-Prompt-Guard-2-86M` | Model identifier or local path for `promptguard` |
| `PROTECTAI_MODEL` | `protectai/deberta-v3-base-prompt-injection` | Model identifier or local path for `protectai` |
| `HF_DEFENSE_DEVICE` | automatic | Device for local defense models, such as `cpu`, `cuda`, or `auto` |

`MINJA_DEFENSE_DEVICE` is also accepted as an alias for `HF_DEFENSE_DEVICE`.

Examples:

```bash
MINJA_JUDGE_MODEL=/path/to/Qwen2.5-0.5B-Instruct \
BACKEND=llm_judge \
bash MemorySystem/simple_run.sh

BACKEND=promptguard PROMPTGUARD_TH=0.3 \
bash MemorySystem/simple_run.sh

BACKEND=perplexity PPL_TH=40 PPL_MAXWIN=1 \
bash MemorySystem/simple_run.sh

HF_DEFENSE_DEVICE=cpu BACKEND=protectai \
bash MemorySystem/simple_run.sh
```

# 🧪 Customize the Example

To change the target domain, number of benign conversations, number of evaluation queries, noise level, or random seeds, edit the corresponding arguments in `MemorySystem/simple_run.sh` after setup.

# 📦 Outputs

The selected `MEMPATH` contains:

- Memory state under `users/` and `assistants/`.
- Evaluation records in `eval/eval_log.jsonl`.
- Benign-initialization statistics in `eval/step0_benign_defense_stats.json`.
- The run configuration and summary in `run_report.txt`.

Always use a fresh `MEMPATH` to keep results from different runs separate.
