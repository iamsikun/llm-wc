#!/usr/bin/env python3
"""
Parse SWE-Bench Pro trajectory results and store normalized outputs.

Data source: Docent API (https://docent.transluce.org), which has
per-instance results with metadata (model_name, instance_id, resolved, turns)
for all models evaluated on SWE-bench Pro.

Requires DOCENT_API_KEY environment variable (or in .env file).

Outputs:
    trajectories/<model_id>.json
    index.json
    trajectory.csv
    leaderboard.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from llm_wc.parsers.common import (
    TrajectoryResults,
    TrajectorySource,
    make_output_dir,
    slugify,
    write_json,
    write_matrix_csv,
    write_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DOCENT_COLLECTION_ID = "032fb63d-4992-4bfc-911d-3b7dafcb931f"
_DEFAULT_OUTPUT_DIR = "resources/swe_bench_pro"


def _load_env_file() -> None:
    """Load .env file from project root if it exists."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


# ---------------------------------------------------------------------------
# Docent data source
# ---------------------------------------------------------------------------

_ModelData = tuple[str, str, dict[str, int], dict[str, int]]
"""(model_name, created_at_date, instances, turns)"""


def _fetch_from_docent(
    max_models: int | None = None,
) -> list[_ModelData] | None:
    """
    Fetch per-instance results from Docent.

    Returns list of (model_name, created_at_date, instances, turns) tuples,
    or None on failure.
    ``instances`` maps instance_id -> 0/1.
    ``turns`` maps instance_id -> turn count.
    ``created_at_date`` is the earliest created_at date for the model (YYYY-MM-DD).
    """
    try:
        from docent import Docent
    except ImportError:
        logger.error("docent-python not installed; install with: uv add docent-python")
        return None

    api_key = os.environ.get("DOCENT_API_KEY")
    if not api_key:
        logger.error("DOCENT_API_KEY not set (set in .env or environment)")
        return None

    try:
        client = Docent(api_key=api_key)
        result = client.execute_dql(
            _DOCENT_COLLECTION_ID,
            """
            SELECT metadata_json->>'instance_id' AS instance_id,
                   metadata_json->>'model_name' AS model_name,
                   metadata_json->>'resolved' AS resolved,
                   metadata_json->>'turns' AS turns,
                   created_at
            FROM agent_runs
            """,
        )
        rows = client.dql_result_to_dicts(result)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Docent query failed: {exc}")
        return None

    logger.info(
        f"Fetched {len(rows)} agent runs from Docent "
        f"(truncated={result.get('truncated', False)})"
    )

    # Group by model
    by_model: dict[str, dict[str, int]] = defaultdict(dict)
    turns_by_model: dict[str, dict[str, int]] = defaultdict(dict)
    dates_by_model: dict[str, str] = {}

    for row in rows:
        model = row["model_name"]
        iid = row["instance_id"]
        by_model[model][iid] = 1 if row["resolved"] == "true" else 0
        try:
            turns_by_model[model][iid] = int(row["turns"])
        except (ValueError, TypeError):
            pass
        # Track earliest created_at per model (YYYY-MM-DD)
        created_at = str(row.get("created_at", ""))[:10]
        if created_at:
            prev = dates_by_model.get(model, "")
            if not prev or created_at < prev:
                dates_by_model[model] = created_at

    model_names = sorted(by_model.keys())
    if max_models is not None:
        model_names = model_names[:max_models]

    return [
        (m, dates_by_model.get(m, ""), by_model[m], turns_by_model.get(m, {}))
        for m in model_names
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse SWE-Bench Pro trajectory results from Docent API."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Parent output directory (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        help="Limit number of models to process (for testing)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    _load_env_file()

    output_parent = args.output_dir or _DEFAULT_OUTPUT_DIR

    # Fetch data
    logger.info("Fetching data from Docent API")
    model_data = _fetch_from_docent(max_models=args.max_models)
    if not model_data:
        logger.error("No model data found")
        return 1
    logger.info(f"Got {len(model_data)} models from Docent")

    # Write outputs
    output_dir = make_output_dir(output_parent)
    trajectories_dir = output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    results: list[TrajectoryResults] = []
    leaderboard_rows: list[dict[str, Any]] = []

    for model_name, date, instances, turns in model_data:
        model_id = slugify(model_name)
        total = len(instances)
        resolved = sum(instances.values())

        source = TrajectorySource(
            model_name=model_name,
            model_id=model_id,
            leaderboard_name="swe-bench-pro",
        )
        results.append(TrajectoryResults(source=source, instances=instances))

        traj_payload: dict[str, Any] = {
            "model": model_name,
            "model_id": model_id,
            "leaderboard": "swe-bench-pro",
            "date": date,
            "counts": {"total": total, "resolved": resolved},
            "instances": instances,
        }
        if turns:
            traj_payload["turns"] = turns

        write_json(trajectories_dir / f"{model_id}.json", traj_payload)

        leaderboard_rows.append({
            "model": model_name,
            "model_id": model_id,
            "date": date,
            "total": total,
            "resolved": resolved,
            "pct_resolved": round(100 * resolved / total, 2) if total else 0,
        })

    # Index
    write_json(output_dir / "index.json", {
        "leaderboard": "swe-bench-pro",
        "source": {"docent_collection": _DOCENT_COLLECTION_ID},
        "models": [
            {
                "model": item.source.model_name,
                "model_id": item.source.model_id,
                "counts": {"total": item.total, "resolved": item.resolved},
            }
            for item in results
        ],
    })

    write_matrix_csv(output_dir, results)

    if leaderboard_rows:
        write_csv(
            output_dir / "leaderboard.csv",
            leaderboard_rows,
            ["model", "model_id", "date", "total", "resolved", "pct_resolved"],
        )

    logger.info(f"Wrote {len(results)} model trajectories to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
