# Data sources

Record of the benchmark data in `resources/` behind the 14 reference settings.
Every number in `results/` is computed from the local snapshots documented
below; no live web fetch is performed at analysis time.

All 14 settings are loaded by `llm_wc.benchmarks.load_settings()`, which pins
each snapshot explicitly: the four SWE-bench classic variants to
`SWE_SNAPSHOT = 20260227T165949Z_7c4289f`, SWE-bench Pro to
`resources/swe_bench_pro/20260305/`, Tau2-Bench to `resources/tau2_bench/`,
and AlpacaEval 2 / WildBench v2 / LiveBench to `resources/external/`.

## SWE-bench classic variants (verified, bash-only, multimodal, multilingual)

**Upstream sources** (same for all four variants, per
`resources/swe_bench_history/<variant>/history_index.json`):

- Leaderboard JSON: `SWE-bench/swe-bench.github.io`, file
  `data/leaderboards.json`
  (<https://github.com/SWE-bench/swe-bench.github.io>).
- Per-instance trajectories: `SWE-bench/experiments`, directory
  `evaluation/<variant>/` (<https://github.com/SWE-bench/experiments>).
- Both are collected by `scripts/parsers/parse_swebench_history_trajs.py`, which
  walks the git history of the two repositories and materializes one dated
  snapshot per leaderboard commit.

**Pinned snapshot** (latest in each history index, produced by
the incremental parser as of `run_at = 2026-03-23T06:25:58Z`):

| Variant      | Snapshot ID              | Leaderboard commit (SWE-bench.github.io) | Leaderboard commit date | Experiments commit (SWE-bench/experiments) | Experiments commit date | Entries | Parsed models |
|--------------|--------------------------|------------------------------------------|--------------------------|---------------------------------------------|--------------------------|---------|----------------|
| verified     | `20260227T165949Z_7c4289f` | `7c4289f30aa1a1c63c2e2a25aae30c16d92b5114` | 2026-02-27 16:59:49 UTC | `82be2b5ac4967af9559484e7bffef227e97348af`  | 2026-01-26 20:05:37 UTC | 180     | 173            |
| bash-only    | `20260227T165949Z_7c4289f` | `7c4289f30aa1a1c63c2e2a25aae30c16d92b5114` | 2026-02-27 16:59:49 UTC | `2f95aad` (`evaluation/bash-only`)          | 2026-02-27 16:59:46 UTC | 48      | 41             |
| multimodal   | `20260227T165949Z_7c4289f` | `7c4289f30aa1a1c63c2e2a25aae30c16d92b5114` | 2026-02-27 16:59:49 UTC | `cff22d0` (`evaluation/multimodal`)         | 2025-11-21 17:46:50 UTC | 22      | 12             |
| multilingual | `20260227T165949Z_7c4289f` | `7c4289f30aa1a1c63c2e2a25aae30c16d92b5114` | 2026-02-27 16:59:49 UTC | `6083d31` (`evaluation/multilingual`)       | 2026-02-24 14:41:20 UTC | 14      | 14             |

The leaderboard commit is shared (`7c4289f`, "Add Gemini 3 Pro to bash-only
leaderboard"); the experiments commit differs per variant because the parser
resolves the most recent `evaluation/<variant>` commit that predates the
leaderboard commit. `Parsed models` is the number of rows retained after
the parser drops ineligible submissions (missing trajectories, mismatched
instance ids, etc.) and gives the number of models $J$ in each setting:

- Verified: $J = 173$, $N = 500$, $G = 12$ repositories.
- Bash-only: $J = 41$, $N = 500$, $G = 12$ repositories.
- Multimodal: $J = 12$, $N = 510$, $G = 12$ repositories.
- Multilingual: $J = 14$, $N = 300$, $G = 41$ repositories.

The $N$ and $G$ values are computed from the trajectory matrix at load time
(`llm_wc.benchmarks`) and are not stored in the history index.

Multimodal result files list only the issues each system resolved, so the
union of listed IDs (248) omits every issue no system resolved. The loader
therefore uses the official test split, pinned in
`llm_wc.parsers.swebench_tasks.SWEBENCH_TASK_SOURCES`
(`princeton-nlp/SWE-bench_Multimodal@aa2db68`, 510 issues) and frozen in
`resources/swe_bench_history/multimodal/task_ids.json`, scoring unlisted
issues 0. Regenerate that file with
`fetch_swebench_task_ids("multimodal")`. The leaderboard's own percentages
imply 517 issues. The published split has 510.

Multilingual drops the malformed upstream ID `mrdoob__three`
(`llm_wc.benchmarks.SWE_EXCLUDED_TASKS`), leaving the official 300 issues.

## SWE-bench Pro

- Upstream: Docent API at <https://docent.transluce.org>, collection id
  `032fb63d-4992-4bfc-911d-3b7dafcb931f`
  (`scripts/parsers/parse_swebench_pro_trajs.py`, constant
  `_DOCENT_COLLECTION_ID`). The Docent project re-evaluates the ScaleAI
  SWE-bench Pro task suite and exposes per-instance resolved/unresolved
  labels plus turn counts.
- Local snapshot: `resources/swe_bench_pro/20260305/` (directory name
  encodes the parse date 2026-03-05), pinned in `llm_wc.benchmarks.SWE_PRO_DIR`.
- Models: 14 (`leaderboard = "swe-bench-pro"`). Twelve systems lack records
  for some of the 730 tasks (Claude 4 Sonnet has 562). `trajectory.csv` fills
  those with 0, which understates their scores. The loader instead reads
  `trajectories/*.json` and keeps the tasks every system has a record for.
- Task and repository counts: $N = 395$, $G = 11$.

## Tau2-bench

`resources/tau2_bench/` contains a parsed snapshot of the
`sierra-research/tau2-bench` leaderboard (see
`scripts/parsers/parse_tau2_trajs.py`), last updated
`2026-04-10T02:42:36Z`, with three domains (airline: 50 tasks, retail: 114,
telecom: 114 in three labeled subcategories). Each domain keeps the models
with complete task vectors (7 per domain).

## AlpacaEval 2, WildBench v2, LiveBench

Task-level judge outputs pinned to upstream commits, with SHA-256 checksums,
in `resources/external/` (see its README for sources, filtering rules, and
`scripts/parsers/prepare_external_benchmark_data.py` to rebuild them).

## Reproducing the snapshots

```bash
# Rebuild the SWE-bench classic snapshots from the upstream git histories
uv run python scripts/parsers/parse_swebench_history_trajs.py --leaderboard-name verified
uv run python scripts/parsers/parse_swebench_history_trajs.py --leaderboard-name bash-only
uv run python scripts/parsers/parse_swebench_history_trajs.py --leaderboard-name multimodal
uv run python scripts/parsers/parse_swebench_history_trajs.py --leaderboard-name multilingual

# Rebuild the SWE-bench Pro snapshot from the Docent API
DOCENT_API_KEY=... uv run python scripts/parsers/parse_swebench_pro_trajs.py

# Run the RULER audit on the pinned snapshots
uv run python scripts/run_ruler_audit.py
```

An incremental parser run updates the `history_index.json` and appends a new
snapshot when a new leaderboard commit appears upstream. To reproduce the
committed results exactly, keep the snapshots listed in the table above;
passing a later snapshot will change $J$ (and therefore the bias estimates)
because new models are continually being added to the SWE-bench
leaderboards.

## Licenses

`resources/` redistributes task-level scores derived from these public
leaderboards. Each keeps its upstream license; this repository's MIT license
covers the code only.

| Data | Upstream | License |
|---|---|---|
| SWE-bench (Verified, Bash-only, Multimodal, Multilingual, Lite, Full) | [SWE-bench/swe-bench.github.io](https://github.com/SWE-bench/swe-bench.github.io), [SWE-bench/experiments](https://github.com/SWE-bench/experiments) | MIT |
| SWE-bench Pro | [Docent](https://docent.transluce.org) re-evaluation of the [SWE-bench Pro](https://scale.com/leaderboard/swe_bench_pro_public) public set | See upstream |
| Tau2-Bench | [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) | MIT |
| AlpacaEval 2 | [tatsu-lab/alpaca_eval](https://github.com/tatsu-lab/alpaca_eval) | Apache-2.0 |
| WildBench v2 | [allenai/WildBench](https://github.com/allenai/WildBench) | Apache-2.0 |
| LiveBench | [livebench/model_judgment](https://huggingface.co/datasets/livebench/model_judgment) | See upstream |

Only task IDs, model IDs, and scores are stored. Prompts, model responses, and
judge explanations are omitted.
