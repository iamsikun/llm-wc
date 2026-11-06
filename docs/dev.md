# Developer notes

This repository corrects the **winner's curse** in LLM leaderboards: when models
are ranked by benchmark scores and the top performer is selected, its score is
systematically inflated by sampling noise. The codebase has two
main parts:

1. **Benchmark data collection** -- gather per-task outcome matrices (tasks x
   models) from public leaderboards.
2. **Winner's curse evaluation** -- estimate and correct rank-induced bias with
   the bootstrap, report the remaining rank uncertainty (RULER), and validate
   both by sample splitting and a calibrated simulation.

---

## Repository structure

```
llm-wc/
├── src/llm_wc/           # Main Python package
│   ├── benchmarks.py     # The 14 reference leaderboard settings (pinned snapshots)
│   ├── wc/               # Winner's Curse bootstrap correction library + RULER
│   ├── simulation/       # Calibrated simulation (logistic DGP calibration, EB baseline)
│   ├── dgp/              # Long-format data helpers used by the calibration
│   ├── parsers/          # Shared parser utilities
│   └── tracker/          # Leaderboard tracker and website builder (docs/tracker.md)
├── scripts/
│   ├── parsers/          # Trajectory parsers (SWE-bench, LiveBench, TAU2, ...)
│   ├── run_ruler_audit.py, run_dependence.py, run_sample_splitting.py,
│   │   run_calibrated_simulation.py, run_gamma_sensitivity.py,
│   │   run_two_model_toy.py, run_gamma_table.py  # experiments -> results/<name>/
├── tests/                # Pytest tests
├── notebooks/            # paper.ipynb: runs the experiments, builds tables and figures
├── resources/            # Frozen benchmark snapshots (external/ = judge-scored benchmarks)
├── tracker/              # Tracker registry, live snapshots and reports, site templates
├── docs/                 # Documentation
│   └── methods/          # Method write-ups (bootstrap correction, etc.)
└── results/              # Experiment outputs (committed)
```

---

## Part 1: Benchmark data collection

The goal is to produce a **per-task outcome matrix** (tasks x models) for each
benchmark.

### Parsing public leaderboard trajectories

For benchmarks like SWE-bench where per-task results are publicly available in
GitHub repos, we parse trajectory data directly.

**Parsers** live in `scripts/parsers/`. Each parser:
- Fetches per-task pass/fail data from a public source (GitHub commits, APIs).
- Outputs a CSV outcome matrix to `resources/<benchmark>/`.
- Supports incremental updates (only fetch new data since the last run).

Available parsers:
- `parse_swebench_history_trajs.py` -- SWE-bench (verified, bash-only,
  multimodal, multilingual, lite, test). Single source of truth for all
  SWE-bench variants. Incremental by default.
- `parse_swebench_pro_trajs.py` -- SWE-bench Pro
- `parse_tau2_trajs.py` -- TAU2
- `parse_livebench.py` -- LiveBench
- `parse_livecodebenchpro.py` -- LiveCodeBench Pro
- `parse_termbench2.py` -- TermBench2

Shared parser utilities (slug generation, CSV I/O, snapshot resolution) are in
`src/llm_wc/parsers/common.py`. See `docs/parsers.md` for full usage.

---

## Part 2: Winner's Curse evaluation

Given per-task outcome matrices, we estimate and correct for rank-induced bias
using bootstrap methods and validate the correction by sample splitting and a
calibrated simulation.

### Bootstrap correction (`src/llm_wc/wc/`)

The core library for correcting rank-induced bias. Entry point:
`bootstrap_correction(obs, ...)` in `wc/core.py`.

**Input:** `obs` matrix (N tasks x J models) of per-task scores.

**Bootstrap methods:**
- `iid` -- standard i.i.d. task resampling.
- `clustered` -- resample entire clusters (e.g., repos in SWE-bench).
- `stratified` -- resample tasks within clusters, preserving cluster
  composition.
- `wild` -- wild cluster bootstrap with Rademacher/Mammen/Webb weight
  distributions.
- `moon` -- m-out-of-n bootstrap (subsample ratio < 1).

**Output:** `BootstrapResult` dataclass containing:
- `emp_rank`, `emp_rank_est` -- empirical ranks and estimates.
- `boot_wc` -- estimated Winner's Curse bias per model.
- `corr_rank_est` -- bias-corrected rank estimates.
- `rank_prob_matrix` -- probability each model achieves each rank.
- Simultaneous inference: `max_t_crit`, pairwise `gap_hat`/`gap_se`/`gap_ci_*`.
- Partial order: `dominance` matrix (directed edges), `rank_lower`/`rank_upper`
  bounds.
- `tiers` -- groups of statistically indistinguishable models (via Tarjan SCC
  on dominance graph).

**RULER** (`wc/ruler.py`): `ruler(obs)` runs the
m-out-of-N bootstrap and returns position adjustments, tie-averaged displayed
adjustments, the fractional-tie rank probability matrix, central rank-mass
intervals, and common-intersection tiers. Note that `BootstrapResult` tiers
(max-t dominance) are a different construction. `wc/ruler_figures.py` and
`wc/ruler_report.py` draw the figures and the interactive report.

### Calibrated simulation (`src/llm_wc/simulation/`)

`calibration.py` fits a logistic model with model fixed effects and a shared
task-difficulty effect to a real leaderboard (`calibrate_logistic()`, using
`dgp._data.prepare_long_format()`), and provides the beta-binomial empirical
Bayes baseline (`beta_binomial_eb()`). `dgp.py` simulates from the fitted model
(`simulate_bernoulli_logit_fe()`). `scripts/run_calibrated_simulation.py` uses
both.

---

## Development guidance

### General conventions

- Preserve behavior: refactors must not change outputs or semantics.
- Favor explicit, readable code over dense one-liners. Avoid nested ternaries.
- Keep imports sorted and remove unused imports.
- Add short docstrings for public functions/classes and non-obvious logic.
- Keep edits scoped; do not refactor unrelated files.
- Prefer small, incremental changes.
- Avoid introducing new dependencies unless required.
- If you change data formats, update docs and provide a migration note.

### Adding a new benchmark parser

1. Create a script in `scripts/parsers/parse_<benchmark>.py`.
2. Use utilities from `src/llm_wc/parsers/common.py` (slugify, CSV I/O).
3. Output a CSV outcome matrix to `resources/<benchmark>/`.
4. Support incremental updates where possible (only fetch new data).
5. Update `docs/parsers.md` with usage instructions.

### Adding a new bootstrap method

1. Implement the resampling logic in `src/llm_wc/wc/core.py` as a
   `_<method>_bootstrap_estimates()` function.
2. Add a dispatch branch in `bootstrap_correction()`.
3. Add tests in `tests/`.

### Testing

```bash
uv run pytest                          # all tests
uv run pytest tests/test_<file>.py     # single file
uv run pytest -k "test_name"           # by name
```

### Environment and tooling

- Python 3.13+ required.
- Use `uv` for all dependency management: `uv sync --all-groups`, `uv run
  python`, `uv add <pkg>`.
- No linter or formatter is configured.
