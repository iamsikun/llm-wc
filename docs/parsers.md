# Benchmark Parser Scripts

All parser scripts normalize benchmark results into a common output layout.
Shared utilities live in `src/llm_wc/parsers/common.py`.

---

## SWE-bench (all variants except Pro)

**Single source of truth:** `resources/swe_bench_history/<variant>/`

The history parser fetches GitHub commit history for the SWE-bench leaderboard
and produces per-snapshot trajectory matrices.  Each variant lives under one
directory with one consolidated run containing all historical snapshots.

**Supported variants:** `verified`, `bash-only`, `multimodal`, `multilingual`,
`lite`, `test`

### Output layout

```
resources/swe_bench_history/<variant>/<run_date>/
    history_index.json              # metadata for all snapshots
    leaderboard_history.csv         # per-snapshot leaderboard entries (not committed)
    model_history.csv               # per-snapshot per-model summary (not committed)
    trajectory_history.csv          # per-snapshot dense matrix (not committed)
    failures.csv                    # failures across all snapshots
    snapshots/
        <timestamp>_<short_sha>/
            index.json              # model list and counts for this commit
            trajectory.csv          # dense (Model x Task) binary matrix
            leaderboard.csv         # leaderboard at this commit
```

The three `*_history.csv` roll-ups are rebuilt from `snapshots/` on every
incremental run, so they are not committed.

### Usage

```bash
# Incremental update (default) — only fetches new commits
uv run python scripts/parse_swebench_history_trajs.py --leaderboard-name verified

# Update all variants
for lb in verified bash-only multimodal multilingual lite test; do
    uv run python scripts/parse_swebench_history_trajs.py --leaderboard-name "$lb"
done

# Full re-parse (ignore existing snapshots)
uv run python scripts/parse_swebench_history_trajs.py \
    --leaderboard-name verified --no-incremental

# Date-filtered parse
uv run python scripts/parse_swebench_history_trajs.py \
    --leaderboard-name verified \
    --since-date 2024-06-01 \
    --max-commits 10
```

Requires a GitHub token for API access (`--github-token` or `GITHUB_TOKEN`).

### Accessing the latest snapshot programmatically

```python
from llm_wc.parsers.common import (
    get_latest_trajectory_csv,
    get_trajectory_csv_at_date,
    get_latest_snapshot_dir,
    get_snapshot_at_date,
)

# Latest snapshot
csv_path = get_latest_trajectory_csv("resources/swe_bench_history", "verified")

# Snapshot as of a specific date
csv_path = get_trajectory_csv_at_date("resources/swe_bench_history", "verified", "2025-06-15")
```

## SWE-bench Pro

Fetches per-instance results from the [Docent API](https://docent.transluce.org)
(primary, 14 models with turn counts) or falls back to
[SWE-bench Pro GitHub repo](https://github.com/scaleapi/SWE-bench_Pro-os)
`eval_results.json` files (9 models).

Requires `DOCENT_API_KEY` in `.env` or environment for the Docent source.

```bash
# Default: auto (tries Docent first, falls back to GitHub)
uv run python scripts/parse_swebench_pro_trajs.py

# Force a specific source
uv run python scripts/parse_swebench_pro_trajs.py --source docent
uv run python scripts/parse_swebench_pro_trajs.py --source github \
    --cache-dir resources/.cache/swebench_pro
```

## LiveBench

Parses LiveBench model-judgment files (JSONL/JSON/CSV).

```bash
# From a directory of result files
uv run python scripts/parse_livebench.py --runs-dir /path/to/livebench/results

# From explicit files
uv run python scripts/parse_livebench.py --files result1.jsonl result2.jsonl
```

## LiveCodeBench Pro

Parses `benchmark_result.json` files from LiveCodeBench Pro.

```bash
uv run python scripts/parse_livecodebenchpro.py --runs-dir /path/to/results
uv run python scripts/parse_livecodebenchpro.py --files model1/benchmark_result.json
```

## Terminal-Bench 2.0

Parses Harbor-style `result.json` files with multiple attempts per task.

```bash
uv run python scripts/parse_termbench2.py --runs-dir /path/to/runs

# Use pass@k instead of pass@1
uv run python scripts/parse_termbench2.py --runs-dir /path/to/runs --reduce pass_at_k
```

## Tau2-Bench

Fetches submission data from the `sierra-research/tau2-bench` GitHub repository
(GitHub-first with local clone fallback). Supports incremental updates.

**Output:** `resources/tau2_bench/`

```
resources/tau2_bench/
    index.json              # metadata index with model list, counts, failures
    leaderboard.csv         # leaderboard summary from submission metadata
    trajectory.csv          # dense (Model x Task) matrix with pass rates
    trajectories/
        <model_id>.json     # detailed per-model results and conversation logs (not committed)
```

The analysis reads only `trajectory.csv`. `trajectories/` (~400 MB) is not
committed. Without it, an incremental run cannot reload earlier submissions,
so rebuild it with `--no-incremental --repo-path /path/to/tau2-bench`.

### Usage

```bash
# Fetch from GitHub (default -- gets leaderboard metadata; trajectory files
# for legacy submissions require a local clone)
uv run python scripts/parsers/parse_tau2_trajs.py

# Incremental update (skip already-processed submissions, default)
uv run python scripts/parsers/parse_tau2_trajs.py --verbose

# Full re-parse
uv run python scripts/parsers/parse_tau2_trajs.py --no-incremental

# With local clone (required for legacy submission trajectory files)
uv run python scripts/parsers/parse_tau2_trajs.py --repo-path /path/to/tau2-bench

# With GitHub token (higher API rate limits for trajectory file discovery)
uv run python scripts/parsers/parse_tau2_trajs.py --github-token $GITHUB_TOKEN
```

**Notes:**
- Legacy submissions have trajectory files only in the cloned repo
  (not on GitHub). Use `--repo-path` for initial setup.
- Active submissions reference trajectory files on S3 that may not be
  publicly accessible.
- Pass rates are stored as floats (0.0-1.0) in `trajectory.csv`,
  not binary 0/1.
