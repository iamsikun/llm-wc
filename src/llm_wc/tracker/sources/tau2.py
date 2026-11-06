"""Tau2-Bench (airline, retail, telecom) from the public leaderboard bucket.

Submission metadata is versioned in sierra-research/tau2-bench
(``web/leaderboard/public/submissions``); trajectory files are hosted only in
the public ``sierra-tau-bench-public`` S3 bucket that taubench.com reads.
Each trajectory file is reduced to per-task mean reward, the same reduction as
``scripts/parsers/parse_tau2_trajs.py``; reductions are cached by ETag so a
monthly run downloads only new or changed files.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

import pandas as pd

from llm_wc.benchmarks import TAU2_CSV
from llm_wc.parsers.common import make_unique_slug, slugify
from llm_wc.tracker.snapshot import Snapshot, frame_from_wide, make_snapshot
from llm_wc.tracker.sources.base import Context, Revision, Source, SourceUnavailable, github_latest_commit, short_date

REPO = "sierra-research/tau2-bench"
SUBMISSIONS = "web/leaderboard/public/submissions"
BUCKET = "https://sierra-tau-bench-public.s3.us-west-2.amazonaws.com/submissions"
DOMAINS = ("airline", "retail", "telecom")
REFERENCE_REVISION = "2026-04-10T02:42:36Z"


def _tasks(task_ids: list[str]) -> pd.DataFrame:
    domain = [t.split("-", 1)[0] for t in task_ids]
    sub = [(m.group(1) if (m := re.search(r"\[([^]]+)\]", t)) else "") for t in task_ids]
    return pd.DataFrame({"task_id": task_ids, "domain": domain, "subcategory": sub})


def _get(url: str, ctx: Context, method: str = "GET"):
    return urlopen(Request(url, method=method, headers={"User-Agent": "llm-wc-tracker"}), timeout=ctx.timeout * 3)


def _task_rewards(traj: dict) -> dict[str, float]:
    """Mean reward per task over trials (missing reward counts as 0, as in the parser)."""
    trials: dict[str, list[float]] = {}
    for sim in traj.get("simulations", []):
        if sim.get("task_id") is not None:
            trials.setdefault(str(sim["task_id"]), []).append((sim.get("reward_info") or {}).get("reward", 0.0) or 0.0)
    return {t: sum(r) / len(r) for t, r in trials.items()}


class Tau2(Source):
    key = "tau2"
    label = "Tau2-Bench"
    reference_snapshot_id = "20260410"

    def latest_revision(self, ctx: Context) -> Revision:
        return github_latest_commit(REPO, SUBMISSIONS, ctx)

    def fetch(self, ctx: Context, revision: Revision | None) -> Snapshot:
        manifest = json.loads(_get(f"{BUCKET}/manifest.json", ctx).read())
        folders = list(dict.fromkeys(manifest.get("submissions", []) + manifest.get("legacy_submissions", [])))
        cache_path = ctx.cache / "tau2_rewards.json"
        cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}

        def submission(folder):
            try:
                return folder, json.loads(_get(f"{BUCKET}/{folder}/submission.json", ctx).read())
            except Exception as exc:  # noqa: BLE001
                return folder, {"_error": str(exc)}

        def rewards(url):
            try:
                etag = _get(url, ctx, "HEAD").headers.get("ETag", "")
                hit = cache.get(url)
                if hit and hit["etag"] == etag:
                    return url, hit
                return url, {"etag": etag, "rewards": _task_rewards(json.loads(_get(url, ctx).read()))}
            except Exception as exc:  # noqa: BLE001 - a listed file may not be uploaded yet
                return url, {"error": str(exc)[:120]}

        with ThreadPoolExecutor(8) as pool:
            subs = dict(pool.map(submission, folders))
            files = {}
            for folder, sub in subs.items():
                for domain, name in (sub.get("trajectory_files") or {}).items():
                    if domain in DOMAINS and name and sub.get("trajectories_available", True):
                        files[(folder, domain)] = f"{BUCKET}/{folder}/trajectories/{name}"
            results = dict(pool.map(rewards, sorted(set(files.values()))))
        cache.update({u: r for u, r in results.items() if "error" not in r})
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))

        used: set[str] = set()
        rows, models, skipped = [], [], []
        for folder in folders:
            sub = subs[folder]
            if "_error" in sub:
                skipped.append({"model_id": folder, "reason": f"submission.json unavailable: {sub['_error'][:80]}"})
                continue
            name = sub.get("model_name") or folder
            model_id = make_unique_slug(slugify(name), used)
            used.add(model_id)
            n, missing = 0, []
            for domain in DOMAINS:
                url = files.get((folder, domain))
                if url is None:
                    continue
                if "error" in results[url]:
                    missing.append(domain)
                    continue
                for task_id, reward in results[url]["rewards"].items():
                    rows.append({"task_id": f"{domain}-{task_id}", "model_id": model_id, "score": reward})
                    n += 1
            if n:
                models.append({"model_id": model_id, "model": name, "organization": sub.get("model_organization", ""),
                               "date": sub.get("submission_date", ""), "folder": folder,
                               "user_simulator": (sub.get("methodology") or {}).get("user_simulator", "")})
            else:
                skipped.append({"model_id": model_id, "reason": "no public trajectories for airline, retail, or telecom"})
            if missing:
                skipped.append({"model_id": model_id, "reason": f"trajectory file not found for {', '.join(missing)}"})
        if not rows:
            raise SourceUnavailable("no public trajectory files found")
        scores = pd.DataFrame(rows)
        task_ids = sorted(scores["task_id"].unique())
        rev = revision.as_dict() if revision else {}
        sid = f"{short_date(rev.get('date'))}_{rev.get('revision', '')[:7]}"
        return make_snapshot(self.key, sid, scores, _tasks(task_ids), pd.DataFrame(models),
                             {**rev, "trajectories": BUCKET},
                             {"submissions": len(folders), "with_trajectories": len(models), "excluded": skipped})

    def reference_snapshot(self) -> Snapshot:
        frame = pd.read_csv(TAU2_CSV, dtype={"model": str, "model_id": str})
        task_ids = [c for c in frame if c not in ("model", "model_id")]
        return make_snapshot(self.key, self.reference_snapshot_id, frame_from_wide(frame), _tasks(task_ids),
                             frame[["model_id", "model"]], {"revision": REFERENCE_REVISION, "date": REFERENCE_REVISION})
