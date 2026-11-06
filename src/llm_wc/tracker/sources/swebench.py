"""SWE-bench sources: the classic leaderboards and SWE-bench Pro.

Fetching reuses the repository's parsers (``scripts/parsers``) unchanged,
writing into the tracker cache instead of ``resources/``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from llm_wc.benchmarks import RESOURCES, SWE_SNAPSHOT, extract_repo, load_settings
from llm_wc.tracker.snapshot import Snapshot, frame_from_wide, make_snapshot
from llm_wc.tracker.sources.base import (
    Context, Revision, Source, SourceUnavailable, github_latest_commit, run_script,
)

LEADERBOARD_REPO = "SWE-bench/swe-bench.github.io"
LEADERBOARD_PATH = "data/leaderboards.json"


def _wide_snapshot(source: str, snapshot_id: str, frame: pd.DataFrame, upstream: dict,
                   leaderboard: pd.DataFrame | None = None, notes: dict | None = None) -> Snapshot:
    task_ids = [c for c in frame if c not in ("model", "model_id")]
    repos = [extract_repo(t) for t in task_ids]
    tasks = pd.DataFrame({"task_id": task_ids, "repository": repos})
    models = frame[["model_id", "model"]].astype(str)
    if leaderboard is not None and "date" in leaderboard:
        dates = leaderboard.drop_duplicates("model").set_index("model")["date"]
        models = models.assign(date=models["model"].map(dates).fillna("").astype(str))
    return make_snapshot(source, snapshot_id, frame_from_wide(frame), tasks, models, upstream, notes)


def _read_wide(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"model": str, "model_id": str})


def _reference_wide(key: str) -> pd.DataFrame:
    """The reference matrix for ``key`` (official task set, see ``llm_wc.benchmarks``) as a wide frame."""
    setting = load_settings([key])[key]
    frame = pd.DataFrame(setting.obs.T, columns=setting.task_ids)
    frame.insert(0, "model", setting.model_names)
    frame.insert(0, "model_id", setting.model_ids)
    return frame


class SweBenchClassic(Source):
    """One SWE-bench leaderboard (verified, bash-only, ...) from the GitHub history."""

    def __init__(self, variant: str, in_reference: bool = True):
        self.variant = variant
        self.in_reference = in_reference
        self.key = "swebench_" + variant.replace("-", "_")
        self.label = f"SWE-bench {variant}"
        self.reference_snapshot_id = SWE_SNAPSHOT if in_reference else None

    def latest_revision(self, ctx: Context) -> Revision:
        return github_latest_commit(LEADERBOARD_REPO, LEADERBOARD_PATH, ctx)

    def fetch(self, ctx: Context, revision: Revision | None) -> Snapshot:
        out = ctx.cache / "raw" / self.key
        run_script([
            "scripts/parsers/parse_swebench_history_trajs.py",
            "--leaderboard-name", self.variant,
            "--output-dir", str(out),
            "--cache-dir", str(ctx.cache / "http" / "swebench"),
            "--max-commits", "1", "--no-incremental",
        ], ctx)
        snaps = sorted((out / "snapshots").iterdir())
        if revision is not None:
            snaps = [s for s in snaps if s.name.endswith(revision.revision[:7])] or snaps
        snap_dir = snaps[-1]
        index = json.loads((snap_dir / "index.json").read_text())
        if index.get("status") == "missing_leaderboard":
            commit = index.get("source", {}).get("leaderboard_commit", {})
            raise SourceUnavailable(f"upstream no longer lists this board (commit {commit.get('sha', '')[:7]}: "
                                    f"{commit.get('message', '').splitlines()[0] if commit.get('message') else ''})")
        if not index.get("models") or not (snap_dir / "trajectory.csv").is_file():
            raise SourceUnavailable(f"parser produced no per-instance results for {self.variant}")
        lb = pd.read_csv(snap_dir / "leaderboard.csv", dtype=str) if (snap_dir / "leaderboard.csv").is_file() else None
        upstream = {**(revision.as_dict() if revision else {}),
                    "leaderboard": f"{LEADERBOARD_REPO}/{LEADERBOARD_PATH}",
                    "experiments": "SWE-bench/experiments",
                    "index": {k: v for k, v in index.items() if k != "failures"}}
        notes = {"parser_failures": len(index.get("failures", []))}
        return _wide_snapshot(self.key, snap_dir.name, _read_wide(snap_dir / "trajectory.csv"), upstream, lb, notes)

    def reference_snapshot(self) -> Snapshot | None:
        if not self.in_reference:
            return None
        upstream = {"revision": "7c4289f30aa1a1c63c2e2a25aae30c16d92b5114", "date": "2026-02-27T16:59:49Z",
                    "url": f"https://github.com/{LEADERBOARD_REPO}/commit/7c4289f30aa1a1c63c2e2a25aae30c16d92b5114"}
        return _wide_snapshot(self.key, SWE_SNAPSHOT, _reference_wide(self.key), upstream)

    def history_snapshots(self) -> list[tuple[str, Path]]:
        """(snapshot_id, trajectory.csv) for every snapshot in the committed parser history."""
        base = RESOURCES / "swe_bench_history" / self.variant / "snapshots"
        if not base.is_dir():
            return []
        return [(d.name, d / "trajectory.csv") for d in sorted(base.iterdir()) if (d / "trajectory.csv").is_file()]


class SweBenchPro(Source):
    """SWE-bench Pro per-instance results from the Docent API."""

    key = "swebench_pro"
    label = "SWE-bench Pro"
    requires_env = ("DOCENT_API_KEY",)
    reference_snapshot_id = "20260305"

    def latest_revision(self, ctx: Context) -> Revision | None:
        return None  # no upstream revision; changes are detected by content fingerprint

    def fetch(self, ctx: Context, revision: Revision | None) -> Snapshot:
        self.check_env()
        try:
            import docent  # noqa: F401
        except ImportError as exc:
            raise SourceUnavailable("needs the 'parsers' dependency group (docent-python)") from exc
        out = ctx.cache / "raw" / self.key
        before = set(out.iterdir()) if out.is_dir() else set()
        run_script(["scripts/parsers/parse_swebench_pro_trajs.py", "--output-dir", str(out)], ctx)
        new = sorted(set(out.iterdir()) - before) or sorted(out.iterdir())
        run_dir = new[-1]
        now = datetime.now(timezone.utc)
        frame = _read_wide(run_dir / "trajectory.csv")
        lb = pd.read_csv(run_dir / "leaderboard.csv", dtype=str)
        snap = _wide_snapshot(self.key, "pending", frame, {"date": now.isoformat(timespec="seconds"),
                              "url": "https://docent.transluce.org",
                              "docent_collection": "032fb63d-4992-4bfc-911d-3b7dafcb931f"}, lb)
        fp = snap.fingerprint()
        snap.snapshot_id = f"{now:%Y%m%d}_{fp[:7]}"
        snap.upstream["revision"] = fp
        return snap

    def reference_snapshot(self) -> Snapshot:
        upstream = {"revision": "20260305", "date": "2026-03-05", "url": "https://docent.transluce.org"}
        return _wide_snapshot(self.key, self.reference_snapshot_id, _reference_wide(self.key), upstream)
