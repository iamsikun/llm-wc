#!/usr/bin/env python3
"""
Normalize Terminal-Bench 2.0 run outcomes into SWE-Bench-style artifacts.

This script parses Harbor-style ``result.json`` files and writes:
- ``trajectories/<model_id>.json``: normalized outcomes for one run/model.
- ``index.json``: summary of all processed runs and failures.
- ``trajectory.csv``: matrix of (Model x Task) outcomes.
- ``leaderboard.csv``: flattened per-model summary rows.

Output layout follows the SWE-Bench parser convention:
``<output-dir>/<YYYYMMDD>/...`` (or ``<YYYYMMDD_HHMMSS>`` if the date folder exists).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# Ensure the src directory is importable (workaround for macOS UF_HIDDEN .pth bug)
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from llm_wc.parsers.common import (
    TrajectoryResults,
    TrajectorySource,
    make_output_dir,
    make_unique_slug,
    slugify,
    write_json,
    write_leaderboard_csv,
    write_matrix_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_PATTERN = "**/result.json"
DEFAULT_REDUCE_MODE = "pass_at_1"
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_BENCHMARK_NAME = "termbench2"


@dataclass(frozen=True)
class AttemptRecord:
    """One attempted task execution parsed from a result.json."""

    run: str
    job: str
    trial: str
    task_id: str
    success: int | None
    result_path: str
    rule: str


# ---------------------------------------------------------------------------
# TermBench-specific: task/outcome inference from Harbor result schemas
# ---------------------------------------------------------------------------

def _get(dct: dict[str, Any], path: str) -> Any:
    """Safe nested dict get: path like ``config.job_id``."""
    cur: Any = dct
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _is_valid_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def infer_task_id(result: dict[str, Any], trial_dir: Path) -> str:
    for key in ("task_id", "task", "task_name", "name"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ("task", "task_info", "dataset_task"):
        value = result.get(key)
        if isinstance(value, dict):
            for sub_key in ("id", "task_id", "name", "slug"):
                sub_value = value.get(sub_key)
                if isinstance(sub_value, str) and sub_value.strip():
                    return sub_value.strip()

    name = trial_dir.name
    if "__" in name:
        return name.split("__", 1)[0]
    return name


def infer_outcome(
    result: dict[str, Any],
    *,
    score_threshold: float,
) -> tuple[int | None, dict[str, Any]]:
    diagnostics: dict[str, Any] = {}

    for key in ("success", "passed", "pass"):
        value = result.get(key)
        if isinstance(value, bool):
            diagnostics["rule"] = f"bool:{key}"
            return (1 if value else 0), diagnostics

    for key in ("score", "accuracy"):
        value = result.get(key)
        if _is_valid_number(value):
            diagnostics["rule"] = f"num:{key}"
            return (1 if float(value) >= score_threshold else 0), diagnostics

    for base in ("metrics", "result", "evaluation", "summary"):
        value = result.get(base)
        if isinstance(value, dict):
            for key in ("success", "passed"):
                nested = value.get(key)
                if isinstance(nested, bool):
                    diagnostics["rule"] = f"dict:{base}.{key}"
                    return (1 if nested else 0), diagnostics
            for key in ("score", "accuracy"):
                nested = value.get(key)
                if _is_valid_number(nested):
                    diagnostics["rule"] = f"dict:{base}.{key}"
                    return (1 if float(nested) >= score_threshold else 0), diagnostics

    verifier = result.get("verifier")
    if isinstance(verifier, dict):
        for key in ("success", "passed"):
            value = verifier.get(key)
            if isinstance(value, bool):
                diagnostics["rule"] = f"dict:verifier.{key}"
                return (1 if value else 0), diagnostics

        status = verifier.get("status")
        if isinstance(status, str):
            normalized = status.strip().lower()
            if normalized in {"passed", "pass", "success", "succeeded", "ok"}:
                diagnostics["rule"] = "dict:verifier.status"
                return 1, diagnostics
            if normalized in {"failed", "fail", "error"}:
                diagnostics["rule"] = "dict:verifier.status"
                return 0, diagnostics

    job_id = _get(result, "config.job_id")
    if isinstance(job_id, str):
        diagnostics["saw_config_job_id"] = True

    diagnostics["rule"] = "unknown"
    return None, diagnostics


def _load_json(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid JSON at {path}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"{path}: expected dict JSON object, got {type(parsed)}")
    return parsed


def _parse_run(
    run_dir: Path,
    *,
    pattern: str,
    score_threshold: float,
) -> tuple[list[AttemptRecord], list[dict[str, Any]]]:
    attempts: list[AttemptRecord] = []
    failures: list[dict[str, Any]] = []

    paths = sorted(run_dir.glob(pattern))
    if not paths:
        failures.append(
            {
                "path": str(run_dir),
                "error": f"No files found matching pattern {pattern!r}",
            }
        )
        return attempts, failures

    for path in paths:
        trial_dir = path.parent
        job_dir = trial_dir.parent

        try:
            result = _load_json(path)
            task_id = infer_task_id(result, trial_dir)
            success, diagnostics = infer_outcome(result, score_threshold=score_threshold)

            attempts.append(
                AttemptRecord(
                    run=run_dir.name,
                    job=job_dir.name if job_dir != run_dir else "",
                    trial=trial_dir.name,
                    task_id=task_id,
                    success=success,
                    result_path=str(path),
                    rule=str(diagnostics.get("rule", "unknown")),
                )
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"path": str(path), "error": str(exc)})
            logger.warning(f"Failed to parse {path}: {exc}")

    return attempts, failures


def _reduce_attempts(
    attempts: Sequence[AttemptRecord],
    *,
    mode: str,
) -> tuple[dict[str, int | None], dict[str, dict[str, Any]]]:
    grouped: dict[str, list[AttemptRecord]] = defaultdict(list)
    for attempt in attempts:
        grouped[attempt.task_id].append(attempt)

    reduced: dict[str, int | None] = {}
    diagnostics: dict[str, dict[str, Any]] = {}

    for task_id in sorted(grouped.keys()):
        task_attempts = sorted(
            grouped[task_id],
            key=lambda item: (item.job, item.trial, item.result_path),
        )
        known_values = [item.success for item in task_attempts if item.success is not None]

        if mode == "pass_at_k":
            outcome = None if not known_values else int(max(known_values))
            diagnostics[task_id] = {
                "mode": mode,
                "attempts": len(task_attempts),
                "known_attempts": len(known_values),
                "rules": sorted({item.rule for item in task_attempts}),
            }
        elif mode == "pass_at_1":
            first = task_attempts[0]
            outcome = first.success
            diagnostics[task_id] = {
                "mode": mode,
                "attempts": len(task_attempts),
                "known_attempts": len(known_values),
                "selected_trial": first.trial,
                "selected_job": first.job,
                "selected_rule": first.rule,
                "selected_result_path": first.result_path,
            }
        else:
            raise ValueError(f"Unknown reduce mode: {mode}")

        reduced[task_id] = outcome

    return reduced, diagnostics


def _coerce_instances(
    reduced_instances: dict[str, int | None],
) -> tuple[dict[str, int], list[str]]:
    instances: dict[str, int] = {}
    unknown_tasks: list[str] = []

    for task_id, value in reduced_instances.items():
        if value is None:
            instances[task_id] = 0
            unknown_tasks.append(task_id)
        else:
            instances[task_id] = int(value)

    return instances, unknown_tasks


def _collect_run_dirs(
    *,
    runs_dir: Path | None,
    run_dirs: Sequence[Path] | None,
) -> list[Path]:
    if runs_dir is not None:
        return sorted(path for path in runs_dir.iterdir() if path.is_dir())
    if run_dirs is not None:
        return sorted(path for path in run_dirs if path.is_dir())
    return []


def _count_resolved(reduced_instances: dict[str, int | None]) -> int:
    return sum(1 for value in reduced_instances.values() if value == 1)


# ---------------------------------------------------------------------------
# Custom TrajectoryResults subclass to carry termbench-specific fields
# ---------------------------------------------------------------------------

@dataclass
class TermBenchTrajectoryResults:
    """Normalized per-task outcomes for a single run/model."""

    source: TrajectorySource
    reduced_instances: dict[str, int | None]
    instances: dict[str, int]
    instances_pass_at_1: dict[str, int | None]
    instances_pass_at_k: dict[str, int | None]
    attempts_count: int
    unknown_attempts_count: int
    unknown_tasks: list[str]

    @property
    def resolved(self) -> int:
        return sum(1 for value in self.reduced_instances.values() if value == 1)

    @property
    def total(self) -> int:
        return len(self.reduced_instances)

    @property
    def unknown(self) -> int:
        return sum(1 for value in self.reduced_instances.values() if value is None)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Parse Terminal-Bench 2.0 Harbor-style result files and store "
            "normalized SWE-Bench-style outputs."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--runs-dir",
        type=Path,
        help="Directory containing run/model subdirectories",
    )
    source.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        help="Explicit run/model directories",
    )

    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Glob pattern inside each run directory (default: {DEFAULT_PATTERN})",
    )
    parser.add_argument(
        "--reduce",
        choices=["pass_at_1", "pass_at_k"],
        default=DEFAULT_REDUCE_MODE,
        help=f"Attempt reduction mode for primary outputs (default: {DEFAULT_REDUCE_MODE})",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help=f"Threshold for score/accuracy success inference (default: {DEFAULT_SCORE_THRESHOLD})",
    )
    parser.add_argument(
        "--benchmark-name",
        default=DEFAULT_BENCHMARK_NAME,
        help=f'Benchmark name written in outputs (default: "{DEFAULT_BENCHMARK_NAME}")',
    )
    parser.add_argument(
        "--output-dir",
        default="resources/termbench2",
        help=(
            "Parent output directory. Results are written to <output-dir>/<YYYYMMDD> "
            "(or <YYYYMMDD_HHMMSS> if that date already exists)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    run_dirs = _collect_run_dirs(
        runs_dir=args.runs_dir,
        run_dirs=args.run_dirs,
    )
    if not run_dirs:
        logger.error("No run directories found")
        return 1

    output_dir = make_output_dir(args.output_dir)
    trajectories_dir = output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    used_model_ids: set[str] = set()
    tb_results: list[TermBenchTrajectoryResults] = []
    leaderboard_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        model_name = run_dir.name
        model_id = make_unique_slug(slugify(model_name), used_model_ids)
        logger.info(f"Processing run: {model_name} (ID: {model_id})")

        attempts, run_failures = _parse_run(
            run_dir,
            pattern=args.pattern,
            score_threshold=args.score_threshold,
        )
        for failure in run_failures:
            failure["model"] = model_name
            failure["model_id"] = model_id
        failures.extend(run_failures)

        if not attempts:
            leaderboard_rows.append(
                {
                    "model": model_name,
                    "model_id": model_id,
                    "parsed": 0,
                    "error": "No valid attempts parsed",
                    "attempts": 0,
                    "unknown_attempts": 0,
                    "total": 0,
                    "resolved": 0,
                    "unknown_tasks": 0,
                    "resolved_rate": 0.0,
                    "reduce_mode": args.reduce,
                }
            )
            continue

        unknown_attempts = sum(1 for attempt in attempts if attempt.success is None)
        result_files = sorted({attempt.result_path for attempt in attempts})

        pass_at_1_instances, pass_at_1_diag = _reduce_attempts(
            attempts, mode="pass_at_1",
        )
        pass_at_k_instances, pass_at_k_diag = _reduce_attempts(
            attempts, mode="pass_at_k",
        )

        selected_reduced = (
            pass_at_1_instances if args.reduce == "pass_at_1" else pass_at_k_instances
        )
        selected_diag = pass_at_1_diag if args.reduce == "pass_at_1" else pass_at_k_diag
        dense_instances, unknown_tasks = _coerce_instances(selected_reduced)

        source = TrajectorySource(
            model_name=model_name,
            model_id=model_id,
            leaderboard_name=args.benchmark_name,
            result_files=result_files,
        )
        result = TermBenchTrajectoryResults(
            source=source,
            reduced_instances=selected_reduced,
            instances=dense_instances,
            instances_pass_at_1=pass_at_1_instances,
            instances_pass_at_k=pass_at_k_instances,
            attempts_count=len(attempts),
            unknown_attempts_count=unknown_attempts,
            unknown_tasks=unknown_tasks,
        )
        tb_results.append(result)

        rule_counts = Counter(attempt.rule for attempt in attempts)
        write_json(
            trajectories_dir / f"{model_id}.json",
            {
                "model": model_name,
                "model_id": model_id,
                "leaderboard": args.benchmark_name,
                "trajectory_url": result_files[0] if result_files else None,
                "result_files": result_files,
                "reduce_mode": args.reduce,
                "counts": {
                    "total": result.total,
                    "resolved": result.resolved,
                    "unknown": result.unknown,
                    "attempts": result.attempts_count,
                    "unknown_attempts": result.unknown_attempts_count,
                },
                "instances": result.instances,
                "instances_pass_at_1": pass_at_1_instances,
                "instances_pass_at_k": pass_at_k_instances,
                "unknown_tasks": result.unknown_tasks,
                "task_details": selected_diag,
                "attempt_rule_counts": dict(sorted(rule_counts.items())),
            },
        )

        resolved_rate = (result.resolved / result.total) if result.total else 0.0
        leaderboard_rows.append(
            {
                "model": model_name,
                "model_id": model_id,
                "parsed": 1,
                "error": None,
                "attempts": result.attempts_count,
                "unknown_attempts": result.unknown_attempts_count,
                "total": result.total,
                "resolved": result.resolved,
                "unknown_tasks": result.unknown,
                "resolved_rate": resolved_rate,
                "reduce_mode": args.reduce,
                "pass_at_1_resolved": _count_resolved(pass_at_1_instances),
                "pass_at_k_resolved": _count_resolved(pass_at_k_instances),
            }
        )

    # Build TrajectoryResults wrappers for shared write_matrix_csv
    common_results: list[TrajectoryResults] = [
        TrajectoryResults(source=r.source, instances=r.instances)
        for r in tb_results
    ]

    index_payload = {
        "leaderboard": args.benchmark_name,
        "source": {
            "runs_dir": str(args.runs_dir) if args.runs_dir else None,
            "run_dirs": [str(path) for path in args.run_dirs] if args.run_dirs else None,
            "pattern": args.pattern,
            "reduce_mode": args.reduce,
            "score_threshold": args.score_threshold,
        },
        "models": [
            {
                "model": result.source.model_name,
                "model_id": result.source.model_id,
                "trajectory_url": result.source.result_files[0]
                if result.source.result_files
                else None,
                "counts": {
                    "total": result.total,
                    "resolved": result.resolved,
                    "unknown": result.unknown,
                },
            }
            for result in tb_results
        ],
        "failures": failures,
    }

    write_json(output_dir / "index.json", index_payload)
    write_matrix_csv(output_dir, common_results)
    write_leaderboard_csv(output_dir, leaderboard_rows)

    logger.info(f"Wrote {len(tb_results)} model trajectories to {trajectories_dir}")
    if failures:
        logger.warning(
            f"{len(failures)} files/runs failed to parse; see {output_dir / 'index.json'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
