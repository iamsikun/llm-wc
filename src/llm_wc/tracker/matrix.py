"""Build a setting's task-by-model matrix from a snapshot.

A setting spec (from ``tracker/registry.yaml``) selects the matrix in three steps:

1. ``pool``: task filter used to choose models ({column: value}, default all tasks).
2. ``models``: which models to keep on the pool.
   ``complete`` keeps models scored on every pool task;
   ``intersection`` keeps every model and the tasks all of them share;
   ``max_cells`` keeps the model set (by coverage cutoff) whose shared tasks
   maximize models x tasks.
3. ``tasks``: task filter applied after model selection (default none).

Task and model order follow the snapshot's ``tasks`` and ``models`` rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from llm_wc.tracker.snapshot import Snapshot


@dataclass(frozen=True)
class Matrix:
    key: str
    obs: np.ndarray            # (N tasks, J models)
    model_ids: np.ndarray
    model_names: np.ndarray
    task_ids: np.ndarray
    strata: np.ndarray
    stratum_type: str
    clusters: np.ndarray | None = None
    excluded: list[dict] = field(default_factory=list)


def _filter(tasks: pd.DataFrame, spec: dict | None) -> pd.DataFrame:
    for column, value in (spec or {}).items():
        tasks = tasks[tasks[column].astype(str) == str(value)]
    return tasks


def max_cells_block(cover: dict[str, set[str]]) -> tuple[list[str], set[str], int]:
    """Coverage-cutoff model set whose shared tasks maximize models x tasks.

    ``cover`` maps model -> set of covered tasks. Returns (models, tasks, cutoff);
    ties go to more tasks, then to the lowest cutoff.
    """
    counts = pd.Series({m: len(t) for m, t in cover.items()})
    best = None
    for cutoff in sorted(counts.unique()):
        models = list(counts[counts >= cutoff].index)
        common = set.intersection(*(cover[m] for m in models))
        cand = (len(models) * len(common), len(common), int(cutoff), models, common)
        if best is None or cand[:2] > best[:2]:
            best = cand
    return best[3], best[4], best[2]


def build_matrix(snap: Snapshot, key: str, spec: dict) -> Matrix:
    tasks = snap.tasks
    pool = _filter(tasks, spec.get("pool"))
    wide = (snap.scores.pivot(index="task_id", columns="model_id", values="score")
            .reindex(index=pool["task_id"], columns=snap.models["model_id"]))
    rule = spec.get("models", "complete")
    excluded = []
    if rule == "complete":
        keep = wide.notna().all(axis=0)
        for m in wide.columns[~keep]:
            excluded.append({"model_id": m, "reason": f"scored on {int(wide[m].notna().sum())} of {len(wide)} tasks"})
        wide = wide.loc[:, keep]
    elif rule == "intersection":
        wide = wide.loc[:, wide.notna().any(axis=0)]
        wide = wide.loc[wide.notna().all(axis=1)]
    elif rule == "max_cells":
        cover = {m: set(wide.index[wide[m].notna()]) for m in wide.columns if wide[m].notna().any()}
        models, common, cutoff = max_cells_block(cover)
        for m in wide.columns:
            if m not in models:
                excluded.append({"model_id": m, "reason": f"below coverage cutoff {cutoff}"})
        wide = wide.loc[wide.index.isin(common), wide.columns.isin(models)]
    else:
        raise ValueError(f"Unknown model rule: {rule}")
    after = _filter(tasks.set_index("task_id").loc[wide.index].reset_index(), spec.get("tasks"))
    wide = wide.loc[after["task_id"]]
    if wide.isna().any().any():
        raise ValueError(f"{key}: selected matrix is not complete")
    meta = tasks.set_index("task_id").loc[wide.index]
    names = snap.models.set_index("model_id").loc[wide.columns, "model"]
    strata_col, clusters_col = spec.get("strata"), spec.get("clusters")
    n = len(wide)
    return Matrix(
        key=key,
        obs=wide.to_numpy(float),
        model_ids=wide.columns.to_numpy(dtype=str),
        model_names=names.to_numpy(dtype=str),
        task_ids=wide.index.to_numpy(dtype=str),
        strata=meta[strata_col].to_numpy(dtype=str) if strata_col else np.repeat("all", n),
        stratum_type=spec.get("stratum_type", "none"),
        clusters=meta[clusters_col].to_numpy(dtype=str) if clusters_col else None,
        excluded=excluded,
    )
