# Reproducible external-benchmark snapshots

This directory contains the task-level leaderboard data consumed by the
AlpacaEval 2, WildBench, and LiveBench settings. The snapshots are
checked into the repository so reproducing the statistics does not require a
live network fetch or mutable leaderboard state.

## Files

- `alpaca_eval_2_annotations.csv.gz`: all 177,892 rows from the 221 canonical
  `weighted_alpaca_eval_gpt4_turbo` annotation files at the pinned commit.
  It includes incomplete and invalid model files so the analysis can reproduce
  the 208-model eligibility filter rather than trusting a prefiltered matrix.
- `wildbench_v2_scores.csv.gz`: all 64,449 task scores in the 63 model score
  files from release `v2.0625` under the pinned GPT-4o evaluator.
- `livebench_model_judgment.csv.gz`: all 60,372 rows in the pinned Hugging Face
  leaderboard split, before timestamp deduplication or common-support
  selection.
- `source_manifest.json`: exact upstream revisions, row counts, column
  semantics, compressed-file sizes, and SHA-256 checksums.

These are analysis-complete raw snapshots, not merely the final matrices. They
retain every upstream row and field used to determine model eligibility,
deduplicate records, align tasks, choose common support, and calculate scores.
Prompt text, model responses, and judge explanations are omitted because the
statistical analysis never reads them. AlpacaEval instructions are represented
by SHA-256 task IDs, which preserves exact cross-model alignment without
duplicating prompt text 221 times.

## Exact upstream versions

- AlpacaEval:
  `cd543a149df89434d8a54582c0151c0b945c3d20`
- WildBench:
  `d6b8dcaf377d173d031980f97c16e1a82618c03d`
- LiveBench `model_judgment`:
  `9704e5da7bfbefe75ac1482a13de827127295993`

The manifest is authoritative and the analysis runner verifies every checksum
before loading data.

## Use in the analysis

`llm_wc.benchmarks.load_settings()` verifies the checksums in
`source_manifest.json` and reconstructs the complete task-by-model matrices
(AlpacaEval 2, WildBench v2, LiveBench pooled and its three categories). The
experiment scripts (`scripts/run_*.py`) and the tracker load
them from there; no network access is needed.

Expected complete matrices:

| Matrix | Tasks | Models |
|---|---:|---:|
| AlpacaEval 2 | 805 | 208 |
| WildBench v2 | 988 | 63 |
| LiveBench pooled | 318 | 81 |
| LiveBench coding | 128 | 81 |
| LiveBench language | 140 | 81 |
| LiveBench instruction following | 50 | 81 |

## Rebuild the snapshots from upstream

The following command clones the two repositories at immutable commits and
downloads the pinned Hugging Face dataset revision:

```bash
uv run python scripts/parsers/prepare_external_benchmark_data.py \
  --output-dir resources/external
```

Existing clones can be supplied to avoid another large download:

```bash
uv run python scripts/parsers/prepare_external_benchmark_data.py \
  --alpaca-eval-repo /path/to/alpaca_eval \
  --wildbench-repo /path/to/WildBench \
  --output-dir resources/external
```

The preparation script writes deterministic gzip streams (`mtime=0`) and
17-digit floating-point text, so identical upstream data produce identical
SHA-256 checksums.
