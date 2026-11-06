"""Judge-scored leaderboards: AlpacaEval 2, WildBench v2, LiveBench.

Normalization calls the reference loaders in ``llm_wc.benchmarks``
(eligibility filters, score transforms, common support), so a live snapshot
follows exactly the rules the reference settings apply. Fetching writes the same raw
files kept in ``resources/external`` into the tracker cache.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm_wc.benchmarks import (
    EXTERNAL_DIR, alpaca_source_strata, load_alpaca_eval, load_external_manifest, load_livebench, load_wildbench,
)
from llm_wc.tracker.snapshot import Snapshot, make_snapshot
from llm_wc.tracker.sources.base import (
    Context, Revision, Source, SourceUnavailable, ensure_partial_clone, git, github_contains,
    github_latest_commit, short_date,
)


def _matrix_snapshot(key: str, sid: str, obs: np.ndarray, model_ids, task_ids, task_meta: dict,
                     upstream: dict, notes: dict) -> Snapshot:
    n, j = obs.shape
    scores = pd.DataFrame({"task_id": np.repeat(task_ids, j), "model_id": np.tile(model_ids, n),
                           "score": obs.reshape(-1)})
    tasks = pd.DataFrame({"task_id": task_ids, **task_meta})
    models = pd.DataFrame({"model_id": model_ids, "model": model_ids})
    return make_snapshot(key, sid, scores, tasks, models, upstream, notes)


def _write_gz(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as gz:
        with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
            frame.to_csv(text, index=False, lineterminator="\n", float_format="%.17g")


def _reference_source(name: str) -> dict:
    return load_external_manifest()["sources"][name]


class _GitExternal(Source):
    repo = ""
    path = ""
    manifest_name = ""

    def latest_revision(self, ctx: Context) -> Revision:
        rev = github_latest_commit(self.repo, self.path, ctx)
        pinned = _reference_source(self.manifest_name)["revision"]
        if github_contains(self.repo, pinned, rev.revision, ctx):
            # The pinned reference commit already contains the newest change to the data path.
            return Revision(pinned, rev.date, f"https://github.com/{self.repo}/commit/{pinned}",
                            {"data_path_commit": rev.revision})
        return rev

    def _clone(self, ctx: Context, revision: str) -> Path:
        return ensure_partial_clone(f"https://github.com/{self.repo}.git",
                                    ctx.cache / "git" / self.repo.split("/")[1], revision)

    def _snapshot_id(self, rev: Revision) -> str:
        return f"{short_date(rev.date)}_{rev.revision[:7]}"


class AlpacaEval2(_GitExternal):
    key = "alpaca_eval_2"
    label = "AlpacaEval 2"
    repo = "tatsu-lab/alpaca_eval"
    path = "results"
    manifest_name = "alpaca_eval_2"
    reference_snapshot_id = "20241227_cd543a1"  # newest results/ commit contained in the pinned revision

    def fetch(self, ctx: Context, revision: Revision | None) -> Snapshot:
        rev = revision or self.latest_revision(ctx)
        repo = self._clone(ctx, rev.revision)
        listing = git("ls-tree", "-r", "--name-only", rev.revision, "results", cwd=repo)
        suffix = "/weighted_alpaca_eval_gpt4_turbo/annotations.json"
        paths = sorted(p for p in listing.splitlines() if p.endswith(suffix) and len(Path(p).parts) == 4)
        rows = []
        for p in paths:
            for i, rec in enumerate(json.loads(git("show", f"{rev.revision}:{p}", cwd=repo))):
                rows.append({"model": Path(p).parts[1], "annotation_index": i,
                             "task_id": hashlib.sha256(str(rec["instruction"]).encode()).hexdigest(),
                             "dataset": str(rec.get("dataset", "unknown")), "preference": rec.get("preference")})
        raw_dir = ctx.cache / "raw" / self.key
        _write_gz(pd.DataFrame(rows).sort_values(["model", "annotation_index"], kind="stable"),
                  raw_dir / "alpaca_eval_2_annotations.csv.gz")
        return self._from_dir(raw_dir, self._snapshot_id(rev), rev.as_dict())

    def _from_dir(self, data_dir: Path, sid: str, upstream: dict) -> Snapshot:
        obs, ids, tasks, meta = load_alpaca_eval(data_dir)
        strata = alpaca_source_strata(tasks, data_dir)
        notes = {"excluded": meta["excluded_incomplete_models"]}
        return _matrix_snapshot(self.key, sid, obs, ids, tasks, {"dataset": strata}, upstream, notes)

    def reference_snapshot(self) -> Snapshot:
        src = _reference_source(self.manifest_name)
        return self._from_dir(EXTERNAL_DIR, self.reference_snapshot_id, {"revision": src["revision"], "url": src["url"]})


class WildBenchV2(_GitExternal):
    key = "wildbench_v2"
    label = "WildBench v2"
    repo = "allenai/WildBench"
    path = "eval_results/v2.0625/score.v2/eval=gpt-4o-2024-05-13"
    manifest_name = "wildbench_v2"
    reference_snapshot_id = "20240801_d6b8dca"

    def fetch(self, ctx: Context, revision: Revision | None) -> Snapshot:
        rev = revision or self.latest_revision(ctx)
        repo = self._clone(ctx, rev.revision)
        listing = git("ls-tree", "-r", "--name-only", rev.revision, self.path + "/", cwd=repo)
        rows = []
        for p in sorted(x for x in listing.splitlines() if x.endswith(".json")):
            for i, rec in enumerate(json.loads(git("show", f"{rev.revision}:{p}", cwd=repo))):
                rows.append({"model": Path(p).stem, "record_index": i, "session_id": str(rec["session_id"]),
                             "score": rec.get("score")})
        raw_dir = ctx.cache / "raw" / self.key
        _write_gz(pd.DataFrame(rows).sort_values(["model", "record_index"], kind="stable"),
                  raw_dir / "wildbench_v2_scores.csv.gz")
        return self._from_dir(raw_dir, self._snapshot_id(rev), rev.as_dict())

    def _from_dir(self, data_dir: Path, sid: str, upstream: dict) -> Snapshot:
        obs, ids, tasks, meta = load_wildbench(data_dir)
        return _matrix_snapshot(self.key, sid, obs, ids, tasks, {}, upstream, {"excluded": meta["excluded_models"]})

    def reference_snapshot(self) -> Snapshot:
        src = _reference_source(self.manifest_name)
        return self._from_dir(EXTERNAL_DIR, self.reference_snapshot_id, {"revision": src["revision"], "url": src["url"]})


class LiveBench(Source):
    key = "livebench"
    label = "LiveBench"
    dataset = "livebench/model_judgment"
    reference_snapshot_id = "20250407_9704e5d"

    def latest_revision(self, ctx: Context) -> Revision:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise SourceUnavailable("needs huggingface_hub") from exc
        info = HfApi().dataset_info(self.dataset)
        return Revision(info.sha, info.last_modified.isoformat() if info.last_modified else None,
                        f"https://huggingface.co/datasets/{self.dataset}/tree/{info.sha}")

    def fetch(self, ctx: Context, revision: Revision | None) -> Snapshot:
        from datasets import load_dataset

        rev = revision or self.latest_revision(ctx)
        frame = load_dataset(self.dataset, revision=rev.revision, split="leaderboard").to_pandas()
        expected = ["question_id", "task", "model", "score", "turn", "tstamp", "category"]
        if list(frame.columns) != expected:
            raise SourceUnavailable(f"unexpected LiveBench columns: {list(frame.columns)}")
        frame = frame.sort_values(["question_id", "model", "tstamp", "turn"], kind="stable")
        raw_dir = ctx.cache / "raw" / self.key
        _write_gz(frame, raw_dir / "livebench_model_judgment.csv.gz")
        return self._from_dir(raw_dir, f"{short_date(rev.date)}_{rev.revision[:7]}", rev.as_dict())

    def _from_dir(self, data_dir: Path, sid: str, upstream: dict) -> Snapshot:
        obs, ids, tasks, categories, meta = load_livebench(data_dir)
        notes = {"coverage_cutoff": meta["coverage_cutoff"]}
        return _matrix_snapshot(self.key, sid, obs, ids, tasks, {"category": categories}, upstream, notes)

    def reference_snapshot(self) -> Snapshot:
        src = _reference_source("livebench")
        return self._from_dir(EXTERNAL_DIR, self.reference_snapshot_id, {"revision": src["revision"], "url": src["url"]})
