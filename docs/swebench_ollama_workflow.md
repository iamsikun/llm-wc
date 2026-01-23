# SWE-bench + mini-swe-agent + Ollama (gemma3:4b vs qwen3:4b)

This workflow compares two local Ollama models on SWE-bench using
mini-swe-agent as the agent scaffold and the SWE-bench harness for
evaluation.

## 0) Scope and track

Pick the track you want to replicate (Lite, Verified, or Full) and fix a
single run configuration for both models so results are comparable. The
SWE-bench leaderboard uses different tracks with different dataset sizes,
so track choice affects cost and runtime.

This repo is set up for SWE-bench Verified runs by default (the
leaderboard-style track).

## 1) Prerequisites

- Docker installed and configured.
- Enough disk/CPU for the chosen track. The SWE-bench evaluation harness
  recommends x86_64 with ~120GB free storage, 16GB RAM, and 8 CPU cores,
  and notes Docker Desktop disk sizing and worker limits.
- The mini-swe-agent SWE-bench Docker containers assume x86 Linux; other
  architectures may not be supported.
- Ollama running locally with the models pulled:
  - `gemma3:4b`
  - `qwen3:4b`

## 2) Add SWE-bench as a submodule

From the repo root:

```bash
git submodule add https://github.com/swe-bench/SWE-bench external/swe-bench
git submodule update --init --recursive
```

Pin the submodule to a specific commit for reproducibility.

## 3) Install mini-swe-agent and SWE-bench (editable)

Create a clean Python environment and install both projects:

```bash
# Example with venv + pip (adjust to uv/poetry if preferred)
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# mini-swe-agent
pip install git+https://github.com/SWE-agent/mini-swe-agent.git

# SWE-bench (from the submodule)
pip install -e external/swe-bench
```

## 4) Configure mini-swe-agent for local Ollama

mini-swe-agent uses LiteLLM. For local models, set the model name and
pass provider settings via `model_kwargs`. Example config (YAML):

```yaml
model:
  model_name: "ollama/gemma3:4b"
  model_kwargs:
    custom_llm_provider: "ollama"
    api_base: "http://localhost:11434"
  cost_tracking: "ignore_errors"
```

Notes:
- LiteLLM’s Ollama examples use `model="ollama/<model>"` with an
  `api_base` pointing at the local Ollama server. For chat-style
  requests, LiteLLM also supports an `ollama_chat/<model>` prefix.
- If you prefer proper cost tracking, add a LiteLLM model registry entry
  for each local model.

This repo includes `config/swebench_ollama.yaml` (copied from the
mini-swe-agent SWE-bench config, with Ollama defaults). Use `--model` to
override per run.

## 5) Run inference with mini-swe-agent

Use `mini-extra swebench` for batch inference. For reproducibility, fix:
subset, split, number of workers, and any run metadata.

Example (Verified, test split):

```bash
mini-extra swebench \
  --config config/swebench_ollama.yaml \
  --model "ollama/gemma3:4b" \
  --subset verified \
  --split test \
  --workers 4 \
  --output runs/swebench_verified/gemma3_4b
```

Repeat for `qwen3:4b` with a separate output directory.

For debugging a single instance:

```bash
mini-extra swebench-single \
  --subset verified \
  --split test \
  --model "ollama/gemma3:4b" \
  -i sympy__sympy-15599
```

Repo helper scripts:

```bash
# Run a Verified batch for a given model
scripts/run_swebench_verified.sh ollama/gemma3:4b
scripts/run_swebench_verified.sh ollama/qwen3:4b

# Evaluate a run (preds.json is created under the run output directory)
scripts/eval_swebench_verified.sh runs/swebench_verified/gemma3_4b/preds.json
scripts/eval_swebench_verified.sh runs/swebench_verified/qwen3_4b/preds.json
```

## 6) Collect predictions

mini-swe-agent writes predictions to its run output directory in
`preds.json`. This file is a JSON dict keyed by instance id, compatible
with the SWE-bench harness.

Example entry (values abbreviated):

```json
{
  "owner__repo-issue": {
    "model_name_or_path": "ollama/gemma3:4b",
    "instance_id": "owner__repo-issue",
    "model_patch": "diff --git ..."
  }
}
```

## 7) Evaluate with SWE-bench harness

Run evaluation from the SWE-bench module (Verified track):

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-Bench_Verified \
  --split test \
  --predictions_path /path/to/preds.json \
  --max_workers 8 \
  --run_id gemma3_4b_verified_test
```

The harness stores build logs under `logs/build_images`, evaluation logs
under `logs/run_evaluation`, and final results in `evaluation_results`.

## 8) Compare results

Compare `results.json` files:
- Resolved count
- Resolution rate
- Instance-level diffs (optional)

Keep the run configs, dataset splits, and version pins identical for a
fair comparison.

## 9) Record and report

Store:
- Submodule commit for SWE-bench
- mini-swe-agent version (git SHA if from source)
- Ollama model tags
- Docker version
- Full run config (subset, split, workers, seeds)

This makes the run reproducible and aligned with leaderboard rules.
