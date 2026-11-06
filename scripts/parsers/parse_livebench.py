#!/usr/bin/env python3
"""
Normalize LiveBench model-judgment files into SWE-Bench-style artifacts.

This script parses LiveBench judgment files (typically
``ground_truth_judgment.jsonl``) and writes:
- ``trajectories/<model_id>.json``: normalized outcomes for one model.
- ``index.json``: summary of all processed models and failures.
- ``trajectory.csv``: matrix of (Model x Question) outcomes.
- ``leaderboard.csv``: flattened per-model summary rows.

Output layout follows the SWE-Bench parser convention:
``<output-dir>/<YYYYMMDD>/...`` (or ``<YYYYMMDD_HHMMSS>`` if the date folder exists).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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
    load_json_records,
    make_output_dir,
    make_unique_slug,
    slugify,
    write_csv,
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

DEFAULT_PATTERN = "**/model_judgment/ground_truth_judgment.jsonl"
DEFAULT_BENCHMARK_NAME = "livebench"
DEFAULT_ACCEPTED_STATUSES = ("accepted", "correct", "passed", "pass", "success")


@dataclass
class ModelAccumulator:
    """Mutable aggregation state before finalizing a model result."""

    model_name: str
    instances: dict[str, int]
    scores: dict[str, float]
    metadata: dict[str, dict[str, Any]]
    files: set[str]


def _infer_model_name(path: Path, mode: str) -> str:
    if mode == "record":
        return path.parent.name
    if mode == "parent":
        return path.parent.name
    if mode == "stem":
        return path.stem
    if mode == "filename":
        return path.name
    raise ValueError(f"Unknown model-name mode: {mode}")


def _collect_result_paths(
    *,
    runs_dir: Path | None,
    files: Sequence[Path] | None,
    pattern: str,
) -> list[Path]:
    if runs_dir is not None:
        return sorted(runs_dir.glob(pattern))
    if files is not None:
        return sorted(files)
    return []


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("result", "verdict", "status", "label"):
            nested = value.get(key)
            if isinstance(nested, str):
                return nested
        return json.dumps(value, sort_keys=True)
    return str(value)


def _extract_model_name(record: dict[str, Any]) -> str | None:
    for key in ("model", "model_id", "name"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_question_id(record: dict[str, Any]) -> str | None:
    for key in ("question_id", "instance_id", "problem_id", "task_id", "id"):
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _extract_score(record: dict[str, Any]) -> float | None:
    for key in ("score", "normalized_score", "grade"):
        if key not in record:
            continue
        parsed = _as_float(record.get(key))
        if parsed is not None:
            return parsed
    return None


def _extract_status(record: dict[str, Any]) -> str | None:
    for key in ("status", "result", "judgment", "verdict", "outcome"):
        if key not in record:
            continue
        parsed = _as_str(record.get(key))
        if parsed is not None and parsed.strip():
            return parsed.strip()
    return None


def _extract_binary_outcome(
    record: dict[str, Any],
    *,
    score_threshold: float,
    accepted_statuses: set[str],
) -> tuple[int | None, float | None]:
    score = _extract_score(record)
    if score is not None:
        if score == -1:
            return None, score
        return (1 if score > score_threshold else 0), score

    status = _extract_status(record)
    if status is None:
        return None, None

    normalized = status.strip().lower()
    if normalized in accepted_statuses:
        return 1, None
    if normalized in {"incorrect", "wrong", "failed", "fail", "rejected", "error"}:
        return 0, None

    return None, None


def _record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": record.get("task"),
        "category": record.get("category"),
        "subtask": record.get("subtask"),
        "livebench_release_date": record.get("livebench_release_date"),
    }


def _aggregate_livebench_records(
    paths: Sequence[Path],
    *,
    benchmark_name: str,
    model_name_mode: str,
    score_threshold: float,
    accepted_statuses: set[str],
    keep_last_duplicate: bool = True,
) -> tuple[list[TrajectoryResults], list[dict[str, Any]]]:
    grouped: dict[str, ModelAccumulator] = {}
    failures: list[dict[str, Any]] = []

    for path in paths:
        try:
            records = load_json_records(path)
        except Exception as exc:  # noqa: BLE001
            failures.append({"path": str(path), "error": str(exc)})
            logger.warning(f"Failed to parse {path}: {exc}")
            continue

        if not records:
            failures.append({"path": str(path), "error": "No records found"})
            logger.warning(f"No records found in {path}")
            continue

        file_fallback_model = _infer_model_name(path, model_name_mode)

        for record in records:
            if not isinstance(record, dict):
                continue

            question_id = _extract_question_id(record)
            if question_id is None:
                continue

            if model_name_mode == "record":
                model_name = _extract_model_name(record) or file_fallback_model
            else:
                model_name = file_fallback_model

            outcome, score = _extract_binary_outcome(
                record,
                score_threshold=score_threshold,
                accepted_statuses=accepted_statuses,
            )
            if outcome is None:
                continue

            model_key = model_name.strip().lower()
            if model_key not in grouped:
                grouped[model_key] = ModelAccumulator(
                    model_name=model_name,
                    instances={},
                    scores={},
                    metadata={},
                    files=set(),
                )
            acc = grouped[model_key]

            if (not keep_last_duplicate) and (question_id in acc.instances):
                continue

            acc.instances[question_id] = outcome
            if score is not None:
                acc.scores[question_id] = score
            acc.metadata[question_id] = _record_metadata(record)
            acc.files.add(str(path))

    used_ids: set[str] = set()
    results: list[TrajectoryResults] = []
    for acc in sorted(grouped.values(), key=lambda item: item.model_name.lower()):
        if not acc.instances:
            continue
        model_id = make_unique_slug(slugify(acc.model_name), used_ids)
        source = TrajectorySource(
            model_name=acc.model_name,
            model_id=model_id,
            leaderboard_name=benchmark_name,
            result_files=sorted(acc.files),
        )
        results.append(
            TrajectoryResults(
                source=source,
                instances=acc.instances,
                scores=acc.scores,
                metadata=acc.metadata,
            )
        )

    if not results and not failures:
        failures.append({"error": "No valid per-question outcomes were extracted"})
    return results, failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Parse LiveBench model-judgment outcomes and store normalized "
            "SWE-Bench-style outputs."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--runs-dir",
        type=Path,
        help="Directory containing LiveBench result files",
    )
    source.add_argument(
        "--files",
        nargs="+",
        type=Path,
        help="Explicit LiveBench result file paths",
    )

    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Glob pattern under --runs-dir (default: {DEFAULT_PATTERN})",
    )
    parser.add_argument(
        "--model-name",
        choices=["record", "parent", "stem", "filename"],
        default="record",
        help=(
            "How to infer model name: from record field (default), or from file path "
            "(parent/stem/filename)"
        ),
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Mark solved when score > threshold (default: 0.0)",
    )
    parser.add_argument(
        "--accepted-status",
        nargs="+",
        default=list(DEFAULT_ACCEPTED_STATUSES),
        help="Status strings treated as solved when numeric score is missing",
    )
    parser.add_argument(
        "--benchmark-name",
        default=DEFAULT_BENCHMARK_NAME,
        help=f'Benchmark name written in outputs (default: "{DEFAULT_BENCHMARK_NAME}")',
    )
    parser.add_argument(
        "--output-dir",
        default="resources/livebench",
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

    paths = _collect_result_paths(
        runs_dir=args.runs_dir,
        files=args.files,
        pattern=args.pattern,
    )
    if not paths:
        logger.error("No LiveBench result files found")
        return 1

    output_dir = make_output_dir(args.output_dir)
    trajectories_dir = output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    accepted_statuses = {value.strip().lower() for value in args.accepted_status}
    results, failures = _aggregate_livebench_records(
        paths,
        benchmark_name=args.benchmark_name,
        model_name_mode=args.model_name,
        score_threshold=args.score_threshold,
        accepted_statuses=accepted_statuses,
    )

    leaderboard_rows: list[dict[str, Any]] = []
    for result in results:
        resolved_rate = (result.resolved / result.total) if result.total else 0.0
        average_score = (
            sum(result.scores.values()) / len(result.scores) if result.scores else None
        )

        write_json(
            trajectories_dir / f"{result.source.model_id}.json",
            {
                "model": result.source.model_name,
                "model_id": result.source.model_id,
                "leaderboard": args.benchmark_name,
                "trajectory_url": result.source.result_files[0]
                if result.source.result_files
                else None,
                "result_files": result.source.result_files,
                "counts": {
                    "total": result.total,
                    "resolved": result.resolved,
                },
                "instances": result.instances,
                "scores": result.scores,
                "task_metadata": result.metadata,
            },
        )

        leaderboard_rows.append(
            {
                "model": result.source.model_name,
                "model_id": result.source.model_id,
                "result_files": result.source.result_files,
                "parsed": 1,
                "error": None,
                "total": result.total,
                "resolved": result.resolved,
                "resolved_rate": resolved_rate,
                "avg_score": average_score,
            }
        )

    index_payload = {
        "leaderboard": args.benchmark_name,
        "source": {
            "runs_dir": str(args.runs_dir) if args.runs_dir else None,
            "files": [str(path) for path in args.files] if args.files else None,
            "pattern": args.pattern if args.runs_dir else None,
            "score_threshold": args.score_threshold,
            "accepted_status": sorted(accepted_statuses),
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
                },
            }
            for result in results
        ],
        "failures": failures,
    }

    write_json(output_dir / "index.json", index_payload)
    write_matrix_csv(output_dir, results)
    write_leaderboard_csv(output_dir, leaderboard_rows)

    logger.info(f"Wrote {len(results)} model trajectories to {trajectories_dir}")
    if failures:
        logger.warning(
            f"{len(failures)} files/records failed to parse; see {output_dir / 'index.json'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
