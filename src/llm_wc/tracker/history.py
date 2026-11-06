"""Per-snapshot RULER summaries over time, one CSV per setting.

Rows come from three places, all with the reference parameters:

- ``archive``: the SWE-bench parser history committed under
  ``resources/swe_bench_history`` (one snapshot per leaderboard commit);
- ``reference``: the reference edition (``results/ruler_audit``);
- ``live``: tracker snapshots, appended by ``update``.
"""
from __future__ import annotations

import pandas as pd

from llm_wc.tracker import registry
from llm_wc.tracker.analyze import reference_tables
from llm_wc.tracker.matrix import build_matrix
from llm_wc.tracker.paths import HISTORY
from llm_wc.tracker.sources import SOURCES
from llm_wc.tracker.sources.swebench import SweBenchClassic, _read_wide, _wide_snapshot
from llm_wc.wc.ruler import ruler

import json

COLUMNS = ["snapshot_id", "date", "edition", "n_tasks", "n_models", "m", "leader_model", "leader_raw_score_pct",
           "leader_adjustment_pp", "position_1_adjustment_pp", "leader_p_rank_1", "tiers_80", "tier_1_size_80"]


def snapshot_date(snapshot_id: str) -> str:
    s = snapshot_id[:8]
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if s.isdigit() else ""


def read_history(key: str) -> pd.DataFrame:
    path = HISTORY / f"{key}.csv"
    if not path.is_file():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={"snapshot_id": str, "date": str, "leader_model": str}, float_precision="round_trip")


def _write(key: str, frame: pd.DataFrame) -> None:
    HISTORY.mkdir(parents=True, exist_ok=True)
    frame = frame.drop_duplicates("snapshot_id", keep="last").sort_values(["date", "snapshot_id"])
    frame[COLUMNS].to_csv(HISTORY / f"{key}.csv", index=False)


def _row(snapshot_id: str, edition: str, n: int, j: int, m: int, leader: str, raw: float, adj: float,
         pos1: float, p1: float, tiers: int, tier1: int) -> dict:
    return dict(zip(COLUMNS, [snapshot_id, snapshot_date(snapshot_id), edition, n, j, m, leader, raw, adj, pos1, p1,
                              tiers, tier1]))


def reference_row(key: str) -> dict:
    t = reference_tables(key)
    s, top = t["summary"], t["models"].iloc[0]
    snapshot_id = SOURCES[registry.setting(key)["source"]].reference_snapshot_id
    return _row(snapshot_id, "reference", int(s["n_tasks"]), int(s["n_models"]), int(s["m"]), str(top["model"]),
                float(top["raw_score_pct"]), float(s["leader_adjustment_pp"]), float(s["position_1_adjustment_pp"]),
                float(s["leader_p_rank_1"]), int(s["tiers_80"]), int(s["tier_1_size_80"]))


def append_live(doc: dict) -> None:
    key, s, top = doc["key"], doc["summary"], doc["models"][0]
    row = _row(doc["snapshot_id"], "live", doc["n_tasks"], doc["n_models"], s["m"], top["model"],
               top["raw_score_pct"], s["leader_adjustment_pp"], s["position_1_adjustment_pp"], s["leader_p_rank_1"],
               s["tiers_80"], s["tier_1_size_80"])
    frame = read_history(key)
    _write(key, pd.concat([frame[frame.snapshot_id != row["snapshot_id"]], pd.DataFrame([row])], ignore_index=True))


def build_history(keys: list[str] | None = None, verbose: bool = True) -> None:
    """Add archive and reference rows for each setting (incremental: known snapshots are skipped)."""
    params = registry.analysis_params()
    for key in keys or list(registry.settings()):
        spec = registry.setting(key)
        frame = read_history(key)
        rows = frame.to_dict("records")
        known = set(frame.snapshot_id)
        src = SOURCES[spec["source"]]
        if isinstance(src, SweBenchClassic):
            for sid, path in src.history_snapshots():
                if sid in known:
                    continue
                snap = _wide_snapshot(src.key, sid, _read_wide(path), {})
                matrix = build_matrix(snap, key, spec)
                if matrix.obs.shape[1] < 2:
                    continue
                r = ruler(matrix.obs, gamma=params["gamma"], n_bootstraps=params["n_bootstraps"], seed=params["seed"],
                          alphas=(params["primary_alpha"],))
                tiers = r.tiers[params["primary_alpha"]]
                rows.append(_row(sid, "archive", r.n_tasks, r.n_models, r.m, str(matrix.model_names[r.order[0]]),
                                 100 * r.scores[0], 100 * r.adjustment[0], 100 * r.position_bias[0], r.rpm[0, 0],
                                 len(tiers), tiers[0]["size"]))
                known.add(sid)
                if verbose:
                    print(f"  {key} {sid}: J={r.n_models}", flush=True)
        if spec.get("in_reference"):
            p = reference_row(key)
            rows = [x for x in rows if x["snapshot_id"] != p["snapshot_id"]] + [p]
        if rows:
            _write(key, pd.DataFrame(rows))


def tiers_path(key: str):
    return HISTORY / f"{key}_tiers.json"


def read_tiers(key: str) -> dict:
    path = tiers_path(key)
    return json.loads(path.read_text()) if path.is_file() else {}


def build_tier_history(keys: list[str] | None = None, verbose: bool = True) -> None:
    """Record, per snapshot, the observed rank-1 models and the top tier (80%) by model ID.

    Uses the SWE-bench parser history plus the tracker's live snapshots, with the reference
    parameters. Incremental: snapshots already recorded are skipped.
    """
    from llm_wc.tracker.snapshot import list_snapshots, read_snapshot

    params = registry.analysis_params()
    alpha = params["primary_alpha"]
    for key in keys or list(registry.settings()):
        spec = registry.setting(key)
        src = SOURCES[spec["source"]]
        if not isinstance(src, SweBenchClassic):
            continue
        out = read_tiers(key)
        snaps = [(sid, lambda p=path, sid=sid: _wide_snapshot(src.key, sid, _read_wide(p), {}))
                 for sid, path in src.history_snapshots()]
        snaps += [(d.name, lambda d=d: read_snapshot(d)) for d in list_snapshots(src.key)]
        for sid, load in snaps:
            if sid in out:
                continue
            matrix = build_matrix(load(), key, spec)
            if matrix.obs.shape[1] < 2:
                continue
            r = ruler(matrix.obs, gamma=params["gamma"], n_bootstraps=params["n_bootstraps"], seed=params["seed"],
                      alphas=(alpha,))
            ids = matrix.model_ids[r.order]
            low, high, tier = r.intervals[alpha]
            out[sid] = {"date": snapshot_date(sid), "n_models": int(r.n_models),
                        "leaders": [str(m) for m in ids[: int(r.rank_end[0])]],
                        "tier1": [str(m) for m, t in zip(ids, tier) if t == 1],
                        "models": sorted(str(m) for m in ids),
                        "names": {str(m): str(n) for m, n in zip(ids[:30], matrix.model_names[r.order][:30])}}
            if verbose:
                print(f"  {key} {sid}: leaders={len(out[sid]['leaders'])} tier1={len(out[sid]['tier1'])}", flush=True)
        HISTORY.mkdir(parents=True, exist_ok=True)
        tiers_path(key).write_text(json.dumps(dict(sorted(out.items())), indent=1) + "\n")


def leader_changes(key: str) -> dict:
    """How often the observed #1 changed, and how often the new #1 was in the same top tier as the old one.

    A change is a snapshot whose rank-1 models share none with the previous snapshot's.
    It is "unresolved" when a previous leader is still in the new snapshot's top tier,
    so the new snapshot's tasks do not separate the new #1 from the old one. When no
    previous leader is on the new board at all (renamed or withdrawn), the change is
    recorded but not classified.
    """
    snaps = [v for _, v in sorted(read_tiers(key).items())]
    events = []
    for prev, cur in zip(snaps, snaps[1:]):
        if set(prev["leaders"]) & set(cur["leaders"]):
            continue
        events.append({"date": cur["date"], "old": prev["leaders"], "new": cur["leaders"],
                       "old_names": [prev["names"].get(m, m) for m in prev["leaders"]],
                       "new_names": [cur["names"].get(m, m) for m in cur["leaders"]],
                       "departed": not set(prev["leaders"]) & set(cur["models"]),
                       "unresolved": bool(set(prev["leaders"]) & set(cur["tier1"]))})
    classified = [e for e in events if not e["departed"]]
    return {"snapshots": len(snaps), "first": snaps[0]["date"] if snaps else None,
            "last": snaps[-1]["date"] if snaps else None, "changes": len(classified),
            "departed": len(events) - len(classified),
            "unresolved": sum(e["unresolved"] for e in classified), "events": events}
