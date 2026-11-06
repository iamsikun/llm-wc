"""Canonical snapshot format.

A snapshot is one fetch of one upstream source, stored under
``tracker/data/<source>/<snapshot_id>/``:

- ``scores.csv.gz``: long format, one row per (task_id, model_id) with a
  score in [0, 1]. Missing pairs are simply absent.
- ``tasks.csv.gz``: one row per task; row order is the task order. Extra
  columns hold task metadata (repository, domain, category, ...).
- ``models.csv``: one row per model; row order is the model order (it
  breaks observed ties, as in the reference audit). Columns ``model_id``, ``model``
  (display name), and optional metadata.
- ``manifest.json``: source, upstream revision, fetch time, counts,
  per-file SHA-256, a content fingerprint, and notes (e.g. excluded models).

Files are written deterministically (gzip mtime 0, repr floats), so the same
content always produces the same bytes.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from llm_wc.tracker.paths import DATA

FORMAT_VERSION = 1
FILES = ("scores.csv.gz", "tasks.csv.gz", "models.csv")


@dataclass
class Snapshot:
    source: str
    snapshot_id: str
    scores: pd.DataFrame
    tasks: pd.DataFrame
    models: pd.DataFrame
    upstream: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None

    def fingerprint(self) -> str:
        """SHA-256 of the content: scores, tasks, and model IDs and names.

        Fetch time and optional model metadata (dates, organizations) are left
        out, so a re-fetch of unchanged results has the same fingerprint.
        """
        digest = hashlib.sha256()
        for frame in (self.scores, self.tasks, self.models[["model_id", "model"]]):
            digest.update(_csv_bytes(frame))
        return digest.hexdigest()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _write_gz(frame: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as gz:
        gz.write(_csv_bytes(frame))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_snapshot(source: str, snapshot_id: str, scores: pd.DataFrame, tasks: pd.DataFrame,
                  models: pd.DataFrame, upstream: dict | None = None, notes: dict | None = None) -> Snapshot:
    """Normalize column types and order, then return a Snapshot."""
    scores = scores[["task_id", "model_id", "score"]].copy()
    scores["task_id"] = scores["task_id"].astype(str)
    scores["model_id"] = scores["model_id"].astype(str)
    scores["score"] = scores["score"].astype(float)
    scores = scores.dropna(subset=["score"])
    tasks = tasks.copy()
    tasks["task_id"] = tasks["task_id"].astype(str)
    tasks = tasks[["task_id", *[c for c in tasks if c != "task_id"]]].reset_index(drop=True)
    models = models.copy()
    models["model_id"] = models["model_id"].astype(str)
    models["model"] = models["model"].astype(str)
    models = models[["model_id", "model", *[c for c in models if c not in ("model_id", "model")]]].reset_index(drop=True)
    task_pos = pd.Series(range(len(tasks)), index=tasks["task_id"])
    model_pos = pd.Series(range(len(models)), index=models["model_id"])
    scores = (scores.assign(_t=scores["task_id"].map(task_pos), _m=scores["model_id"].map(model_pos))
              .sort_values(["_t", "_m"], kind="stable").drop(columns=["_t", "_m"]).reset_index(drop=True))
    return Snapshot(source, snapshot_id, scores, tasks, models, upstream or {}, notes or {})


def validate(snap: Snapshot) -> list[str]:
    """Return a list of problems (empty when the snapshot is valid)."""
    problems = []
    if snap.tasks["task_id"].duplicated().any():
        problems.append("duplicate task_id in tasks")
    if snap.models["model_id"].duplicated().any():
        problems.append("duplicate model_id in models")
    if snap.scores.duplicated(["task_id", "model_id"]).any():
        problems.append("duplicate (task_id, model_id) in scores")
    if not snap.scores["task_id"].isin(snap.tasks["task_id"]).all():
        problems.append("scores reference unknown task_id")
    if not snap.scores["model_id"].isin(snap.models["model_id"]).all():
        problems.append("scores reference unknown model_id")
    s = snap.scores["score"]
    if s.isna().any() or (s < 0).any() or (s > 1).any():
        problems.append("scores outside [0, 1]")
    if len(snap.scores) == 0:
        problems.append("no scores")
    return problems


def snapshot_dir(source: str, snapshot_id: str, root: Path = DATA) -> Path:
    return Path(root) / source / snapshot_id


def write_snapshot(snap: Snapshot, root: Path = DATA) -> Path:
    problems = validate(snap)
    if problems:
        raise ValueError(f"{snap.source}/{snap.snapshot_id}: " + "; ".join(problems))
    out = snapshot_dir(snap.source, snap.snapshot_id, root)
    out.mkdir(parents=True, exist_ok=True)
    _write_gz(snap.scores, out / "scores.csv.gz")
    _write_gz(snap.tasks, out / "tasks.csv.gz")
    (out / "models.csv").write_bytes(_csv_bytes(snap.models))
    created = snap.created_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest = {
        "format_version": FORMAT_VERSION,
        "source": snap.source,
        "snapshot_id": snap.snapshot_id,
        "created_at": created,
        "upstream": snap.upstream,
        "counts": {"scores": len(snap.scores), "tasks": len(snap.tasks), "models": len(snap.models)},
        "fingerprint": snap.fingerprint(),
        "files": {name: {"bytes": (out / name).stat().st_size, "sha256": _sha256(out / name)} for name in FILES},
        "notes": snap.notes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    return out


def read_manifest(path: Path) -> dict:
    return json.loads((Path(path) / "manifest.json").read_text())


def read_snapshot(path: Path, verify: bool = True) -> Snapshot:
    path = Path(path)
    manifest = read_manifest(path)
    if verify:
        for name, meta in manifest["files"].items():
            if _sha256(path / name) != meta["sha256"]:
                raise ValueError(f"Checksum mismatch: {path / name}")
    read = dict(float_precision="round_trip", keep_default_na=False, na_values=[""])
    scores = pd.read_csv(path / "scores.csv.gz", dtype={"task_id": str, "model_id": str}, **read)
    tasks = pd.read_csv(path / "tasks.csv.gz", dtype=str, keep_default_na=False)
    models = pd.read_csv(path / "models.csv", dtype=str, keep_default_na=False)
    return Snapshot(manifest["source"], manifest["snapshot_id"], scores, tasks, models,
                    manifest.get("upstream", {}), manifest.get("notes", {}), manifest.get("created_at"))


def list_snapshots(source: str, root: Path = DATA) -> list[Path]:
    base = Path(root) / source
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if (p / "manifest.json").is_file())


def latest_snapshot_dir(source: str, root: Path = DATA) -> Path | None:
    snaps = list_snapshots(source, root)
    return snaps[-1] if snaps else None


def validate_dir(path: Path) -> list[str]:
    """Validate a stored snapshot: required files, checksums, fingerprint, content."""
    path = Path(path)
    missing = [n for n in (*FILES, "manifest.json") if not (path / n).is_file()]
    if missing:
        return [f"missing {', '.join(missing)}"]
    try:
        snap = read_snapshot(path)
    except ValueError as exc:
        return [str(exc)]
    problems = validate(snap)
    manifest = read_manifest(path)
    if snap.fingerprint() != manifest["fingerprint"]:
        problems.append("fingerprint mismatch")
    for key in ("source", "snapshot_id", "created_at", "upstream", "counts"):
        if key not in manifest:
            problems.append(f"manifest missing {key}")
    return problems


def frame_from_wide(frame: pd.DataFrame, id_cols: tuple[str, ...] = ("model", "model_id")) -> pd.DataFrame:
    """Long scores from a wide (model rows x task columns) frame; blank cells are dropped."""
    task_ids = [c for c in frame if c not in id_cols]
    values = frame[task_ids].apply(pd.to_numeric, errors="coerce")
    values.index = frame["model_id"].astype(str)
    long = values.T.rename_axis("task_id").reset_index().melt(id_vars="task_id", var_name="model_id", value_name="score")
    return long.dropna(subset=["score"])

