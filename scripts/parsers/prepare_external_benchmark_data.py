#!/usr/bin/env python3
"""Create deterministic, analysis-complete snapshots of external leaderboards.

The upstream repositories contain model outputs and judge explanations that are
hundreds of megabytes.  The statistical analysis uses only task ids,
model ids, task metadata, and task-level scores.  This script extracts every
upstream row needed to reproduce filtering, common-support selection, and the
bootstrap matrices without retaining prompt/response text.
"""

from __future__ import annotations

from llm_wc.benchmarks import EXTERNAL_DIR as EXTERNAL_DATA

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset


ALPACA_URL = "https://github.com/tatsu-lab/alpaca_eval.git"
ALPACA_REVISION = "cd543a149df89434d8a54582c0151c0b945c3d20"
WILDBENCH_URL = "https://github.com/allenai/WildBench.git"
WILDBENCH_REVISION = "d6b8dcaf377d173d031980f97c16e1a82618c03d"
WILDBENCH_PREFIX = (
    "eval_results/v2.0625/score.v2/eval=gpt-4o-2024-05-13/"
)
LIVEBENCH_DATASET = "livebench/model_judgment"
LIVEBENCH_REVISION = "9704e5da7bfbefe75ac1482a13de827127295993"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    )


def _ensure_git_revision(
    *,
    supplied_repo: Path | None,
    url: str,
    revision: str,
    scratch: Path,
    name: str,
) -> Path:
    if supplied_repo is not None:
        repo = supplied_repo.resolve()
        if not repo.exists():
            raise FileNotFoundError(repo)
    else:
        repo = scratch / name
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", url, str(repo)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "fetch", "--depth=1", "origin", revision],
            check=True,
        )

    resolved = _git(repo, "rev-parse", f"{revision}^{{commit}}").strip()
    if resolved != revision:
        raise ValueError(f"{name}: expected {revision}, resolved {resolved}")
    return repo


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_deterministic_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            compresslevel=9,
            mtime=0,
        ) as compressed:
            with io.TextIOWrapper(
                compressed,
                encoding="utf-8",
                newline="",
            ) as text:
                frame.to_csv(
                    text,
                    index=False,
                    lineterminator="\n",
                    float_format="%.17g",
                )


def _extract_alpaca_eval(repo: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    listing = _git(repo, "ls-tree", "-r", "--name-only", ALPACA_REVISION, "results")
    suffix = "/weighted_alpaca_eval_gpt4_turbo/annotations.json"
    # Match the official one-model/one-config layout used by the leaderboard.
    # Two legacy Vicuna paths contain the config directory twice and are not
    # discovered by the canonical results/*/<config>/annotations.json pattern.
    paths = sorted(
        path
        for path in listing.splitlines()
        if path.endswith(suffix) and len(Path(path).parts) == 4
    )
    rows: list[dict[str, Any]] = []

    for path in paths:
        model = Path(path).parts[1]
        payload = _git(repo, "show", f"{ALPACA_REVISION}:{path}")
        records = json.loads(payload)
        for index, record in enumerate(records):
            instruction = str(record["instruction"])
            rows.append(
                {
                    "model": model,
                    "annotation_index": index,
                    "task_id": _sha256_text(instruction),
                    "dataset": str(record.get("dataset", "unknown")),
                    "preference": record.get("preference"),
                }
            )

    frame = pd.DataFrame(rows).sort_values(
        ["model", "annotation_index"],
        kind="stable",
    )
    metadata = {
        "upstream_annotation_files": len(paths),
        "upstream_rows": len(frame),
        "upstream_models": int(frame["model"].nunique()),
        "columns": {
            "model": "Upstream result directory/model identifier.",
            "annotation_index": "Row position in the upstream annotations JSON.",
            "task_id": "SHA-256 of the exact upstream instruction text.",
            "dataset": "AlpacaEval source-dataset label.",
            "preference": "Raw weighted preference; analysis score is preference - 1.",
        },
    }
    return frame, metadata


def _extract_wildbench(repo: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    listing = _git(
        repo,
        "ls-tree",
        "-r",
        "--name-only",
        WILDBENCH_REVISION,
        WILDBENCH_PREFIX,
    )
    paths = sorted(path for path in listing.splitlines() if path.endswith(".json"))
    rows: list[dict[str, Any]] = []

    for path in paths:
        model = Path(path).stem
        payload = _git(repo, "show", f"{WILDBENCH_REVISION}:{path}")
        records = json.loads(payload)
        for index, record in enumerate(records):
            rows.append(
                {
                    "model": model,
                    "record_index": index,
                    "session_id": str(record["session_id"]),
                    "score": record.get("score"),
                }
            )

    frame = pd.DataFrame(rows).sort_values(
        ["model", "record_index"],
        kind="stable",
    )
    metadata = {
        "upstream_score_files": len(paths),
        "upstream_rows": len(frame),
        "upstream_models": int(frame["model"].nunique()),
        "columns": {
            "model": "Upstream score filename/model identifier.",
            "record_index": "Row position in the upstream score JSON.",
            "session_id": "WildBench task/session identifier.",
            "score": "Raw checklist-guided score on the 1--10 scale.",
        },
    }
    return frame, metadata


def _extract_livebench() -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = load_dataset(
        LIVEBENCH_DATASET,
        revision=LIVEBENCH_REVISION,
        split="leaderboard",
    ).to_pandas()
    expected = [
        "question_id",
        "task",
        "model",
        "score",
        "turn",
        "tstamp",
        "category",
    ]
    if list(frame.columns) != expected:
        raise ValueError(f"Unexpected LiveBench columns: {list(frame.columns)}")
    frame = frame.sort_values(
        ["question_id", "model", "tstamp", "turn"],
        kind="stable",
    )
    metadata = {
        "upstream_rows": len(frame),
        "upstream_models": int(frame["model"].nunique()),
        "upstream_tasks": int(frame["question_id"].nunique()),
        "columns": {
            column: str(dtype) for column, dtype in frame.dtypes.items()
        },
    }
    return frame, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXTERNAL_DATA,
    )
    parser.add_argument(
        "--alpaca-eval-repo",
        type=Path,
        help="Optional existing clone containing the pinned commit.",
    )
    parser.add_argument(
        "--wildbench-repo",
        type=Path,
        help="Optional existing clone containing the pinned commit.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="llmwc-external-data-") as temp_dir:
        scratch = Path(temp_dir)
        alpaca_repo = _ensure_git_revision(
            supplied_repo=args.alpaca_eval_repo,
            url=ALPACA_URL,
            revision=ALPACA_REVISION,
            scratch=scratch,
            name="alpaca_eval",
        )
        wildbench_repo = _ensure_git_revision(
            supplied_repo=args.wildbench_repo,
            url=WILDBENCH_URL,
            revision=WILDBENCH_REVISION,
            scratch=scratch,
            name="wildbench",
        )

        alpaca, alpaca_meta = _extract_alpaca_eval(alpaca_repo)
        wildbench, wildbench_meta = _extract_wildbench(wildbench_repo)
        livebench, livebench_meta = _extract_livebench()

    outputs = {
        "alpaca_eval_2_annotations.csv.gz": alpaca,
        "wildbench_v2_scores.csv.gz": wildbench,
        "livebench_model_judgment.csv.gz": livebench,
    }
    for filename, frame in outputs.items():
        _write_deterministic_csv_gz(frame, output_dir / filename)

    manifest = {
        "format_version": 1,
        "scope": (
            "Analysis-complete task-level fields. Prompt text, model responses, "
            "and judge explanations are intentionally omitted because the "
            "statistical analysis never consumes them."
        ),
        "sources": {
            "alpaca_eval_2": {
                "url": ALPACA_URL,
                "revision": ALPACA_REVISION,
                "annotation_config": "weighted_alpaca_eval_gpt4_turbo",
                "license": "Apache-2.0",
                **alpaca_meta,
            },
            "wildbench_v2": {
                "url": WILDBENCH_URL,
                "revision": WILDBENCH_REVISION,
                "release": "v2.0625",
                "judge": "gpt-4o-2024-05-13",
                "license": "Apache-2.0",
                **wildbench_meta,
            },
            "livebench": {
                "url": f"https://huggingface.co/datasets/{LIVEBENCH_DATASET}",
                "revision": LIVEBENCH_REVISION,
                **livebench_meta,
            },
        },
        "files": {
            filename: {
                "rows": len(frame),
                "bytes": (output_dir / filename).stat().st_size,
                "sha256": _sha256_file(output_dir / filename),
            }
            for filename, frame in outputs.items()
        },
    }
    manifest_path = output_dir / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
