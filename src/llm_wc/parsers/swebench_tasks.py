"""Official SWE-bench task lists, fetched from Hugging Face at pinned revisions.

Leaderboard result files for some SWE-bench variants list only the issues a
system resolved, so the union over systems misses every issue that no system
resolved. The official score divides by the full test set, so the
matrices are built on the official task list instead.

Usage::

    from llm_wc.parsers.swebench_tasks import fetch_swebench_task_ids, load_swebench_task_ids
    fetch_swebench_task_ids("multimodal")      # writes resources/swe_bench_history/multimodal/task_ids.json
    ids = load_swebench_task_ids("multimodal") # verifies the checksum
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

RESOURCES = Path(__file__).resolve().parents[3] / "resources"

# variant -> (Hugging Face dataset, pinned revision, split)
SWEBENCH_TASK_SOURCES = {
    "multimodal": ("princeton-nlp/SWE-bench_Multimodal", "aa2db68940196b6b59ae3f577faa0c25157bdd50", "test"),
}


def task_ids_path(variant: str) -> Path:
    return RESOURCES / "swe_bench_history" / variant / "task_ids.json"


def _checksum(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def fetch_swebench_task_ids(variant: str, out: Path | None = None) -> list[str]:
    """Download the official task IDs for ``variant`` and write them, sorted, with provenance."""
    from datasets import load_dataset

    dataset, revision, split = SWEBENCH_TASK_SOURCES[variant]
    ids = sorted(load_dataset(dataset, revision=revision, split=split)["instance_id"])
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate instance IDs in {dataset}@{revision}:{split}")
    out = out or task_ids_path(variant)
    out.write_text(json.dumps({"dataset": dataset, "revision": revision, "split": split,
                               "n_tasks": len(ids), "sha256": _checksum(ids), "instance_ids": ids}, indent=1) + "\n")
    return ids


def load_swebench_task_ids(variant: str) -> list[str] | None:
    """The frozen official task IDs for ``variant``, or None if the variant has no pinned list."""
    if variant not in SWEBENCH_TASK_SOURCES:
        return None
    path = task_ids_path(variant)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run fetch_swebench_task_ids({variant!r})")
    record = json.loads(path.read_text())
    ids = record["instance_ids"]
    if (record["dataset"], record["revision"], record["split"]) != SWEBENCH_TASK_SOURCES[variant] or _checksum(ids) != record["sha256"]:
        raise ValueError(f"{path} does not match its pinned source or checksum")
    return ids
