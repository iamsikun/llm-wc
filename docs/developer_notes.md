# Developer Notes (Humans + Coding Agents)

This repo provides a lightweight framework to evaluate LLMs across multiple
benchmarks. These notes capture the minimal conventions and workflows needed to
extend or maintain it without adding clutter.

## Quick orientation

- `src/llm_wc/`: core library code
- `scripts/`: CLI runners for each benchmark
- `config/`: YAML configs consumed by the runners
- `docs/`: documentation for data layout and benchmark notes
- `results/`: run outputs (created at runtime)
- `resources/`: auxiliary files (prompt examples, etc.)

Key files:
- `src/llm_wc/core/`: dataset + evaluation primitives
- `src/llm_wc/client/`: provider-agnostic LLM client + adapters
- `src/llm_wc/bench_utils.py`: config parsing, overrides, logger, progress bar
- `src/llm_wc/bench_cli.py`: shared CLI helpers (output dirs, metadata)

## Conventions (keep the repo clean)

- Preserve behavior: refactors must not change outputs or semantics.
- Favor explicit, readable code over dense one-liners.
- Avoid nested ternaries; prefer clear branching.
- Keep imports sorted and remove unused imports.
- Add short docstrings for public functions/classes and non-obvious logic.
- If a dataset is gated, include actionable error guidance (see `gpqa.py`).

## How evaluation works

1. A benchmark loader returns a `BenchmarkDataset` with:
   - `name`, `questions`, `prompts`, and optional metadata.
2. `evaluate_model_on_benchmark`:
   - builds prompts,
   - sends messages via `LLMClient`,
   - parses the answer,
   - records a normalized `EvalResult`.
3. `compute_accuracy` summarizes results.

Common hooks:
- `prompt_builder(bench, question, prompt_type)`
- `answer_parser(text, bench, question)` returning:
  - `str | None` (prediction), or
  - `(str | None, extra_meta)` for additional parsing metadata.

## Adding a new benchmark (minimal checklist)

1. Create a module in `src/llm_wc/<benchmark>.py` with:
   - `load_<benchmark>_benchmark() -> BenchmarkDataset`
   - `prompt_<benchmark>(...) -> str`
   - `evaluate_model_on_<benchmark>(...) -> list[EvalResult]`
2. Add a YAML config under `config/`.
3. Add a runner in `scripts/` using the shared CLI helpers.
4. Update `docs/benchmarks.md` and `docs/structure.md` if new data layout is needed.

Tip: reuse `core.mcqa.extract_choice_answer` for multiple-choice tasks, and
follow the GPQA and MMLU-Pro modules for parsing patterns.

## Outputs and metadata

Each run writes:
- `results/<run_id>/run.json` and `config.yaml`
- `results/<run_id>/models/<model>/results.json` + `summary.json`

`run.json` captures the benchmark settings, model configs, and timestamps. The
model summaries include `accuracy`, token usage (if provided by the API), and
the request parameters used.

## Provider adapters

Adapters normalize API differences:
- `OpenAIProvider` supports OpenAI-compatible endpoints (including Ollama).
- `AnthropicProvider` adapts the Messages API.

If you add a provider:
1. Implement `ProviderBase`.
2. Register it in `client/manager.py`.
3. Keep the response normalized to `ModelResponse`.

## For coding agents

- Keep edits scoped; do not refactor unrelated files.
- Prefer small, incremental changes that preserve behavior.
- Avoid introducing new dependencies unless required for a benchmark.
- If you change data formats, update docs and provide a migration note.

## If something fails

- Check the config YAML first.
- Verify dataset availability and gating requirements.
- Validate that model names exist at the target API base.
- Inspect `run.json` and `summary.json` for overrides applied.
