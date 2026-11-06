"""Pairwise comparisons: how often system A ranks above system B across rebuilt leaderboards.

Uses the same m-out-of-N resamples (same seed and m) that produce each report's rank
probability matrix, restricted to the top ``K`` systems in observed order. Ties in a
rebuilt leaderboard count one half. The site reports it as a pairwise comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from llm_wc.tracker import registry
from llm_wc.tracker.matrix import Matrix
from llm_wc.tracker.paths import RESULTS
from llm_wc.wc.ruler import TOL, iid_bootstrap_means

K = 30
PAIRWISE = RESULTS / "pairwise"


def pairwise(matrix: Matrix, params: dict, k: int = K) -> dict:
    obs = matrix.obs
    n, j = obs.shape
    m = int(n ** params["gamma"])
    est = iid_bootstrap_means(obs, m, params["n_bootstraps"], params["seed"])
    order = np.argsort(-obs.mean(0), kind="stable")[:k]
    e = est[:, order]
    diff = e[:, :, None] - e[:, None, :]
    p = (diff > TOL).mean(0) + 0.5 * (np.abs(diff) <= TOL).mean(0)
    np.fill_diagonal(p, 0.5)
    return {"model_ids": matrix.model_ids[order].tolist(), "models": matrix.model_names[order].tolist(),
            "m": m, "n_bootstraps": params["n_bootstraps"], "p_above": np.round(p, 4).tolist()}


def path(key: str, snapshot_id: str) -> Path:
    return PAIRWISE / key / f"{snapshot_id}.json"


def write(key: str, snapshot_id: str, doc: dict) -> Path:
    out = path(key, snapshot_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"key": key, "snapshot_id": snapshot_id, **doc}, separators=(",", ":")) + "\n")
    return out


def read(key: str, snapshot_id: str) -> dict | None:
    p = path(key, snapshot_id)
    return json.loads(p.read_text()) if p.is_file() else None


def build_pairwise(keys: list[str] | None = None, force: bool = False) -> None:
    """Compute pairwise files for the reference edition and every live snapshot that lacks one."""
    from llm_wc.tracker.fidelity import reference_matrices
    from llm_wc.tracker.matrix import build_matrix
    from llm_wc.tracker.snapshot import list_snapshots, read_snapshot
    from llm_wc.tracker.sources import SOURCES

    params = registry.analysis_params()
    keys = keys or list(registry.settings())
    ref = None
    for key in keys:
        spec = registry.setting(key)
        src = SOURCES[spec["source"]]
        if spec.get("in_reference") and (force or read(key, src.reference_snapshot_id) is None):
            ref = ref or reference_matrices()
            write(key, src.reference_snapshot_id, pairwise(ref[key], params))
            print(f"  {key} reference", flush=True)
        for d in list_snapshots(spec["source"]):
            if force or read(key, d.name) is None:
                matrix = build_matrix(read_snapshot(d), key, spec)
                if matrix.obs.shape[1] >= 2:
                    write(key, d.name, pairwise(matrix, params))
                    print(f"  {key} {d.name}", flush=True)
