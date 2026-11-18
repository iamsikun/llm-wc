# Leaderboard tracker and website

The tracker keeps RULER reports for public LLM leaderboards up to date and builds the project
website ("SOTA or Luck?"). It is separate from the experiment scripts. It reads their
pinned inputs and committed results but never writes to `resources/` or `results/` (see
"Reference fidelity" below).

## Layout

```
src/llm_wc/tracker/        code (python -m llm_wc.tracker ...)
  snapshot.py              canonical snapshot format
  matrix.py                setting rules: snapshot -> task-by-model matrix
  sources/                 one adapter per upstream leaderboard
  analyze.py               RULER tables and JSON reports (uses llm_wc.wc.ruler)
  update.py, history.py    monthly update; per-snapshot history and top-tier membership
  pairwise.py              "A ranks above B" shares for each edition's top 30 systems
  fidelity.py              checks that the adapters reproduce the reference results
  discovery/               feasibility criteria, probe, scan
  site/build.py            static site builder and its checks
  site/share.py            link-preview images (og/*.png) and embeddable badges (badges/*.svg)
  site/texmath.py          LaTeX formulas in templates ({% filter tex %}...{% endfilter %}) to MathML
tracker/
  registry.yaml            site metadata, analysis parameters, sources, settings
  candidates.yaml          leaderboards under review (curated)
  data/<source>/<id>/      live snapshots (committed)
  results/<setting>/<id>.json   live RULER reports (committed)
  results/history/<setting>.csv per-snapshot summaries (committed)
  results/history/<setting>_tiers.json  rank-1 systems and top tier per snapshot (committed)
  results/pairwise/<setting>/<id>.json  pairwise comparisons (committed)
  status.json              last update result per source (committed)
  discovery_inbox.json     raw scan hits awaiting review (committed)
  site/templates, static   the website's templates and assets
_site/                     built site (not committed)
.cache/tracker/            downloads and parser scratch (not committed)
```

## Commands

```bash
uv sync --group tracker --group parsers      # parsers: only for SWE-bench Pro (Docent)

uv run python -m llm_wc.tracker update       # check every source; store and analyze new snapshots
uv run python -m llm_wc.tracker update --sources swebench_verified tau2
uv run python -m llm_wc.tracker status       # last result per source
uv run python -m llm_wc.tracker validate     # checksums, fingerprints, report schema
uv run python -m llm_wc.tracker history      # history CSVs and top-tier membership per snapshot
uv run python -m llm_wc.tracker pairwise     # pairwise files for any edition that lacks one
uv run python -m llm_wc.tracker check-reference  # adapters reproduce the reference matrices and numbers

uv run python -m llm_wc.tracker build        # write _site/ and run the site checks
uv run python -m llm_wc.tracker serve        # http://127.0.0.1:8000/

uv run python -m llm_wc.tracker probe scores.csv        # feasibility report for a table
uv run python -m llm_wc.tracker probe hf:org/dataset --split train
uv run python -m llm_wc.tracker scan                    # search HF and GitHub; writes the inbox
```

`update` needs network access. Set `GITHUB_TOKEN` (or log in with `gh`) for the GitHub API, and
`DOCENT_API_KEY` (environment or `.env`) for SWE-bench Pro. A source that cannot run is recorded
as `unavailable` or `failed` with the reason and keeps its last good data. The other sources
still run.

## Data format

Each snapshot directory `tracker/data/<source>/<snapshot_id>/` holds:

| File | Content |
|---|---|
| `scores.csv.gz` | Long format: `task_id, model_id, score` (score in [0, 1]); missing pairs are absent |
| `tasks.csv.gz` | One row per task, in task order; extra columns are task metadata (`repository`, `domain`, `subcategory`, `category`, `dataset`) |
| `models.csv` | One row per model, in model order (this order breaks observed ties); `model_id`, `model` (display name), optional metadata |
| `manifest.json` | `source`, `snapshot_id`, `created_at`, `upstream` (revision, date, URL), counts, per-file SHA-256, content `fingerprint`, `notes` (e.g. excluded models) |

Files are written deterministically (gzip mtime 0, shortest round-trip floats). The fingerprint
covers scores, tasks, and model IDs and names. A re-fetch with the same content therefore
produces no new snapshot.

A **setting** (in `registry.yaml`) turns a snapshot into a matrix: `pool` (task filter used to
choose models), `models` (`complete`, `intersection`, or `max_cells`), `tasks` (task filter applied
afterwards), and the task columns used as `strata` and `clusters`. Live reports use the reference
parameters (`analysis:` in the registry) and the same `llm_wc.wc.ruler.ruler` function.

**Migration note.** No existing data format changed. The tracker reads the existing formats as
they are: the SWE-bench `trajectory.csv` matrices, the Tau2 matrix, and the external `csv.gz` files
with their loaders in `llm_wc.benchmarks`. It writes only the new format above, under `tracker/`.
The SWE-bench parsers keep writing `resources/` when run directly. The tracker runs them with
`--output-dir` pointed at `.cache/tracker/raw/`.

## Sources

| Source | Upstream | Notes |
|---|---|---|
| `swebench_*` | SWE-bench leaderboard and experiments repos | Runs `scripts/parsers/parse_swebench_history_trajs.py --max-commits 1` into the cache. Bash-only was removed upstream on 2026-09-01 ("Drop bash-only board"), so it now reports `unavailable` and keeps the reference edition. |
| `swebench_pro` | Docent API | Runs `scripts/parsers/parse_swebench_pro_trajs.py`; needs `DOCENT_API_KEY`. Changes are detected by content fingerprint. |
| `tau2` | tau2-bench submissions plus the public `sierra-tau-bench-public` S3 bucket | Trajectories moved from GitHub to S3. Each file is reduced to per-task mean reward (as in the Tau2 parser), and reductions are cached by ETag. |
| `alpaca_eval_2`, `wildbench_v2` | GitHub repos | `latest_revision` asks whether the pinned commit already contains the newest change to the data path. If it does, the source is `same_as_reference` and nothing is fetched. |
| `livebench` | Hugging Face `livebench/model_judgment` | Compares the dataset revision with the pinned reference revision. |

Normalization for AlpacaEval, WildBench, and LiveBench calls the reference loaders
(`load_alpaca_eval`, `load_wildbench`, `load_livebench`), so live editions follow the same
eligibility and common-support rules.

## Site features that depend on derived data

- **Home page story** ("N new #1s since ..."): from `results/history/<featured>_tiers.json`, built by
  `history`. A handover counts when no system is #1 in both consecutive snapshots. It is
  "unresolved" when the previous #1 is still in the new snapshot's top tier. When the previous
  #1 left the board (renamed or withdrawn), the handover is shown but not counted.
  `site.featured` in the registry picks the leaderboard.
- **Compare two systems**: from `results/pairwise/`. These are shares of the same rebuilt
  leaderboards that produce each report's rank probability matrix (same seed and m).
- **Link previews and badges**: generated on every build from the latest edition of each leaderboard.

## Adding a leaderboard

1. Probe its data: `python -m llm_wc.tracker probe <table>`. It must pass the blocking criteria
   (per-task scores, a complete block with at least 30 tasks and 3 models, bounded scores).
2. Write an adapter in `src/llm_wc/tracker/sources/`: a `Source` subclass with `key`,
   `latest_revision(ctx)` (cheap, or `None` to compare content fingerprints), and `fetch(ctx, rev)`
   returning `make_snapshot(...)`. Register it in `sources/__init__.py`.
3. Add the source and its settings to `tracker/registry.yaml` with `in_reference: false`. Each setting
   needs a label, family, summary, description, metric, task population, resampling unit, and
   limitations. The site shows all of these.
4. Move its entry in `tracker/candidates.yaml` out of the candidates list, then run
   `update --sources <key>`, `build`, and the tests.

## Monthly automation and publishing

`.github/workflows/`:

- `ci.yml`: tests (including the reference-fidelity tests) and a site build on every push and PR.
- `update.yml`: on the 3rd of each month (and on manual dispatch), it runs `update`, `validate`,
  `scan`, and `build`. It then commits `tracker/` and redeploys the site, and opens an issue for
  sources that need attention.
- `pages.yml`: builds `_site/` and deploys it to GitHub Pages.

Nothing runs or publishes until it is switched on. The update job runs only if the repository
variable `TRACKER_UPDATES` is `true`. The deploy job runs only if `PUBLISH_SITE` is `true`.

**To publish the site:**

1. Make the repository public. GitHub Pages on the free plan needs a public repository.
2. Settings > Pages > Source: **GitHub Actions**.
3. Settings > Secrets and variables > Actions: add the variables `PUBLISH_SITE=true` and
   `TRACKER_UPDATES=true`, and the secret `DOCENT_API_KEY`.
4. Run "Deploy site" once from the Actions tab. The site is served at
   `https://sikunxu.com/llm-wc/` (the account's custom domain;
   `iamsikun.github.io/llm-wc/` redirects there).

## Reference fidelity

Each leaderboard on the site has a frozen *reference edition*: the pinned snapshot in `resources/`
analyzed by `scripts/run_ruler_audit.py`, with results in `results/ruler_audit/`. The tracker
never modifies those. `tests/tracker/test_reference_fidelity.py` checks two things. The adapters,
run on the pinned reference inputs, rebuild all 14 matrices exactly as `load_settings()` does. The
tracker's analysis then reproduces `results/ruler_audit/` exactly. The site's reference edition is
read from `results/ruler_audit/`, and every build compares it with those files.
