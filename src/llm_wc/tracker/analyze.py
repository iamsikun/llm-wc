"""RULER reports for tracker editions.

``audit_tables`` computes, for one matrix, the same tables that
``scripts/run_ruler_audit.py`` writes (models, position_bias,
rpm, tiers, summary row), with the same functions from ``llm_wc.wc.ruler``.
``reference_tables`` reads those tables back from ``results/ruler_audit``.
``report`` turns either into the JSON document the site and the committed
live results use.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from llm_wc.tracker.matrix import Matrix
from llm_wc.tracker.paths import REFERENCE_AUDIT, RESULTS
from llm_wc.wc.ruler import iid_bootstrap_means, position_bias_and_rpm, ruler, tie_averaged

RPM_DECIMALS = 6


def level(alpha: float) -> str:
    return str(round((1 - alpha) * 100))


def audit_tables(matrix: Matrix, params: dict, label: str) -> dict:
    """Tables for one matrix; identical to run_ruler_audit.audit for the reference settings."""
    gamma, b, seed = params["gamma"], params["n_bootstraps"], params["seed"]
    alphas = tuple(params["alphas"])
    obs = matrix.obs
    n, j = obs.shape
    r = ruler(obs, gamma=gamma, n_bootstraps=b, seed=seed, alphas=alphas)
    ordinary, _ = position_bias_and_rpm(obs, iid_bootstrap_means(obs, n, b, seed), n)
    ord_adj, _, _ = tie_averaged(ordinary, r.rank_start, r.rank_end)

    rank_label = [str(s) if s == e else f"{s}–{e}" for s, e in zip(r.rank_start, r.rank_end)]
    models = pd.DataFrame({
        "position": np.arange(1, j + 1),
        "model_id": matrix.model_ids[r.order],
        "model": matrix.model_names[r.order],
        "raw_score_pct": 100 * r.scores,
        "rank": rank_label,
        "rank_start": r.rank_start,
        "rank_end": r.rank_end,
        "adjustment_pp": 100 * r.adjustment,
        "adjustment_low_pp": 100 * r.adjustment_low,
        "adjustment_high_pp": 100 * r.adjustment_high,
        "ordinary_adjustment_pp": 100 * ord_adj,
        "p_rank_1": r.rpm[:, 0],
        "p_top_5": r.rpm[:, :5].sum(1),
    })
    for alpha in alphas:
        low, high, ids = r.intervals[alpha]
        models[f"rank_low_{level(alpha)}"] = low
        models[f"rank_high_{level(alpha)}"] = high
        models[f"tier_{level(alpha)}"] = ids
    position_bias = pd.DataFrame({"position": np.arange(1, j + 1), "adjustment_pp": 100 * r.position_bias,
                                  "ordinary_adjustment_pp": 100 * ordinary})
    tiers = pd.DataFrame([
        {"rank_mass": level(alpha), **t,
         "observed_rank_start": int(r.rank_start[t["start_position"] - 1]),
         "observed_rank_end": int(r.rank_end[t["end_position"] - 1])}
        for alpha in alphas for t in r.tiers[alpha]
    ])
    summary = {"setting": matrix.key, "label": label, "n_tasks": n, "n_models": j,
               "m": r.m, "n_bootstraps": b, "seed": seed,
               "position_1_adjustment_pp": 100 * r.position_bias[0],
               "leader_adjustment_pp": 100 * r.adjustment[0],
               "leader_adjustment_low_pp": 100 * r.adjustment_low[0],
               "leader_adjustment_high_pp": 100 * r.adjustment_high[0],
               "leader_ordinary_adjustment_pp": 100 * ord_adj[0],
               "leader_p_rank_1": r.rpm[0, 0]}
    for alpha in alphas:
        summary[f"tiers_{level(alpha)}"] = len(r.tiers[alpha])
        summary[f"tier_1_size_{level(alpha)}"] = r.tiers[alpha][0]["size"]
    return {"models": models, "position_bias": position_bias, "rpm": r.rpm, "tiers": tiers, "summary": summary}


def reference_tables(key: str, audit_dir: Path = REFERENCE_AUDIT) -> dict:
    """The committed reference audit tables for one setting."""
    d = Path(audit_dir) / key
    read = dict(float_precision="round_trip")
    summary = pd.read_csv(Path(audit_dir) / "summary.csv", **read).set_index("setting").loc[key].to_dict()
    return {
        "models": pd.read_csv(d / "models.csv", dtype={"rank": str, "model_id": str, "model": str}, **read),
        "position_bias": pd.read_csv(d / "position_bias.csv", **read),
        "rpm": pd.read_csv(d / "rpm.csv", index_col=0, **read).to_numpy(),
        "tiers": pd.read_csv(d / "tiers.csv", dtype={"rank_mass": str}),
        "summary": {"setting": key, **summary},
    }


def _py(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    return [{k: _py(v) for k, v in row.items()} for row in frame.to_dict("records")]


def report(tables: dict, *, key: str, edition: str, snapshot_id: str, source: str,
           upstream: dict | None = None, excluded: list | None = None, params: dict | None = None) -> dict:
    """JSON-ready report. RPM entries are rounded to RPM_DECIMALS; all other numbers are exact."""
    models = tables["models"]
    summary = {k: _py(v) for k, v in tables["summary"].items()}
    n = int(summary["n_tasks"])
    leader_bias = float(tables["position_bias"]["adjustment_pp"].iloc[0])
    return {
        "schema_version": 1,
        "key": key,
        "edition": edition,
        "snapshot_id": snapshot_id,
        "source": source,
        "upstream": upstream or {},
        "params": {"gamma": (params or {}).get("gamma", 0.9), "m": summary["m"],
                   "n_bootstraps": summary["n_bootstraps"], "seed": summary["seed"]},
        "n_tasks": n,
        "n_models": int(summary["n_models"]),
        "summary": summary,
        # Planning scale for the sizing rule N >= (delta / eps)^2 (see the Design page).
        "delta_pp": leader_bias * float(np.sqrt(n)),
        "models": _records(models),
        "position_bias": _records(tables["position_bias"]),
        "tiers": _records(tables["tiers"]),
        "rpm": np.round(np.asarray(tables["rpm"], dtype=float), RPM_DECIMALS).tolist(),
        "excluded": excluded or [],
    }


def result_path(key: str, snapshot_id: str, root: Path = RESULTS) -> Path:
    return Path(root) / key / f"{snapshot_id}.json"


def write_report(doc: dict, root: Path = RESULTS) -> Path:
    path = result_path(doc["key"], doc["snapshot_id"], root)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {**doc, "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat()}
    path.write_text(json.dumps(doc, separators=(",", ":"), ensure_ascii=False) + "\n")
    return path


def latest_live_report(key: str, root: Path = RESULTS) -> dict | None:
    d = Path(root) / key
    files = sorted(d.glob("*.json")) if d.is_dir() else []
    return json.loads(files[-1].read_text()) if files else None
