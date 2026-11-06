#!/usr/bin/env python3
"""
Normalize LiveCodeBench Pro per-task outcomes into SWE-Bench-style artifacts.

This script parses one or more ``benchmark_result.json`` files and writes:
- ``trajectories/<model_id>.json``: normalized outcomes for one model.
- ``index.json``: summary of all processed models and failures.
- ``trajectory.csv``: matrix of (Model x Problem) outcomes.
- ``leaderboard.csv``: flattened per-model summary rows.

Output layout follows the SWE-Bench parser convention:
``<output-dir>/<YYYYMMDD>/...`` (or ``<YYYYMMDD_HHMMSS>`` if the date folder exists).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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

DEFAULT_ACCEPTED_VALUES = ("Accepted",)
DEFAULT_BENCHMARK_NAME = "livecodebench_pro"


def _read_json(path: Path) -> Any:
    """Read and parse JSON from disk."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _normalize_records(obj: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        if all(isinstance(item, dict) for item in obj):
            return obj
        raise ValueError(f"{path}: JSON is a list but not list[dict].")

    if isinstance(obj, dict):
        for key in ("results", "data", "items"):
            value = obj.get(key)
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return value
        raise ValueError(
            f"{path}: JSON is a dict; expected list[dict] or dict key in "
            f"{['results', 'data', 'items']} containing list[dict]."
        )

    raise ValueError(f"{path}: JSON root must be list or dict.")


def _infer_model_name(path: Path, mode: str) -> str:
    if mode == "parent":
        return path.parent.name
    if mode == "stem":
        return path.stem
    if mode == "filename":
        return path.name
    raise ValueError(f"Unknown model-name mode: {mode}")


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("result", "verdict", "status"):
            nested = value.get(key)
            if isinstance(nested, str):
                return nested
        return json.dumps(value, sort_keys=True)
    return str(value)


def _parse_run(
    result_path: Path,
    *,
    accepted_values: set[str],
    keep_last_duplicate: bool = True,
) -> tuple[dict[str, int], dict[str, str | None], dict[str, dict[str, Any]]]:
    payload = _read_json(result_path)
    records = _normalize_records(payload, result_path)

    verdicts: dict[str, str | None] = {}
    metadata: dict[str, dict[str, Any]] = {}

    for record in records:
        if "problem_id" not in record:
            continue

        problem_id = str(record["problem_id"])
        verdict = _as_str(record.get("judge_result"))

        if (not keep_last_duplicate) and problem_id in verdicts:
            continue

        verdicts[problem_id] = verdict
        metadata[problem_id] = {
            "problem_title": record.get("problem_title"),
            "difficulty": record.get("difficulty"),
            "platform": record.get("platform"),
        }

    if not verdicts:
        raise ValueError(f"{result_path}: no records with problem_id found.")

    instances = {
        problem_id: 1 if verdict in accepted_values else 0
        for problem_id, verdict in verdicts.items()
    }
    return instances, verdicts, metadata


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Parse LiveCodeBench Pro benchmark_result.json outcomes and "
            "store normalized SWE-Bench-style outputs."
        )
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--runs-dir",
        type=Path,
        help="Directory containing per-model subdirectories with benchmark_result.json",
    )
    source.add_argument(
        "--files",
        nargs="+",
        type=Path,
        help="Explicit benchmark_result.json file paths",
    )

    parser.add_argument(
        "--pattern",
        default="*/benchmark_result.json",
        help="Glob pattern under --runs-dir (default: */benchmark_result.json)",
    )
    parser.add_argument(
        "--model-name",
        choices=["parent", "stem", "filename"],
        default="parent",
        help="How to infer model name from result path (default: parent)",
    )
    parser.add_argument(
        "--accepted",
        nargs="+",
        default=list(DEFAULT_ACCEPTED_VALUES),
        help='Judge verdicts treated as solved (default: ["Accepted"])',
    )
    parser.add_argument(
        "--output-dir",
        default="resources/livecodebench_pro",
        help=(
            "Parent output directory. Results are written to <output-dir>/<YYYYMMDD> "
            "(or <YYYYMMDD_HHMMSS> if that date already exists)."
        ),
    )
    parser.add_argument(
        "--benchmark-name",
        default=DEFAULT_BENCHMARK_NAME,
        help=f'Benchmark name written in outputs (default: "{DEFAULT_BENCHMARK_NAME}")',
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
        logger.error("No benchmark_result.json files found")
        return 1

    output_dir = make_output_dir(args.output_dir)
    trajectories_dir = output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    accepted_values = set(args.accepted)
    used_model_ids: set[str] = set()

    results: list[TrajectoryResults] = []
    leaderboard_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for path in paths:
        model_name = _infer_model_name(path, args.model_name)
        model_id = make_unique_slug(slugify(model_name), used_model_ids)

        logger.info(f"Processing model: {model_name} (ID: {model_id})")
        try:
            instances, verdicts, metadata = _parse_run(
                path,
                accepted_values=accepted_values,
            )
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            logger.warning(f"Failed to parse {path}: {error}")
            failures.append(
                {
                    "model": model_name,
                    "model_id": model_id,
                    "result_path": str(path),
                    "error": error,
                }
            )
            leaderboard_rows.append(
                {
                    "model": model_name,
                    "model_id": model_id,
                    "result_path": str(path),
                    "parsed": 0,
                    "error": error,
                    "total": 0,
                    "resolved": 0,
                    "resolved_rate": 0.0,
                }
            )
            continue

        source = TrajectorySource(
            model_name=model_name,
            model_id=model_id,
            leaderboard_name=args.benchmark_name,
            result_path=str(path),
            raw_entry={
                "model": model_name,
                "model_id": model_id,
                "result_path": str(path),
            },
        )
        result = TrajectoryResults(
            source=source,
            instances=instances,
            verdicts=verdicts,
            metadata=metadata,
        )
        results.append(result)

        write_json(
            trajectories_dir / f"{model_id}.json",
            {
                "model": model_name,
                "model_id": model_id,
                "leaderboard": args.benchmark_name,
                "trajectory_url": str(path),
                "counts": {
                    "total": result.total,
                    "resolved": result.resolved,
                },
                "instances": instances,
                "verdicts": verdicts,
                "task_metadata": metadata,
            },
        )

        resolved_rate = (result.resolved / result.total) if result.total else 0.0
        leaderboard_rows.append(
            {
                "model": model_name,
                "model_id": model_id,
                "result_path": str(path),
                "parsed": 1,
                "error": None,
                "total": result.total,
                "resolved": result.resolved,
                "resolved_rate": resolved_rate,
            }
        )

    index_payload = {
        "leaderboard": args.benchmark_name,
        "source": {
            "runs_dir": str(args.runs_dir) if args.runs_dir else None,
            "files": [str(path) for path in args.files] if args.files else None,
            "pattern": args.pattern if args.runs_dir else None,
            "accepted": list(args.accepted),
        },
        "models": [
            {
                "model": result.source.model_name,
                "model_id": result.source.model_id,
                "trajectory_url": result.source.result_path,
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
            f"{len(failures)} models failed to parse; see {output_dir / 'index.json'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
