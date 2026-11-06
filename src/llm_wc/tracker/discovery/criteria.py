"""Feasibility criteria for running RULER on a leaderboard.

RULER needs a task-by-model score matrix: every model scored on the same
tasks, with stable task IDs and per-task scores on a bounded scale. A table
qualifies when:

1. task_level   - scores are reported per task, not only per model;
2. stable_ids   - each (task, model) pair appears once; repeats need review
                  (average trials, or keep one re-submission) but do not
                  rule a table out;
3. shared_pool  - a complete model-by-task block exists (largest block found
                  by coverage cutoff, as for LiveBench in the reference settings);
4. min_tasks    - that block has at least ``min_tasks`` tasks (default 30);
5. min_models   - and at least ``min_models`` models (default 3);
6. score_range  - scores are numeric and bounded: [0, 1] as is, or [0, 100]
                  or [1, 10] after rescaling;
7. variation    - model means in the block are not all equal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from llm_wc.tracker.matrix import max_cells_block

MIN_TASKS = 30
MIN_MODELS = 3


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    blocking: bool = True   # a failed non-blocking check asks for review instead of ruling the table out


@dataclass
class FeasibilityReport:
    target: str
    rows: int
    models: int
    tasks: int
    density: float
    block_models: int
    block_tasks: int
    coverage_cutoff: int
    score_min: float
    score_max: float
    rescale: str | None
    checks: list[Check] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if any(not c.passed and c.blocking for c in self.checks):
            return "not feasible"
        notes = [n for n, cond in (("rescaling", self.rescale is not None),
                                   ("review", any(not c.passed for c in self.checks))) if cond]
        return "feasible" + (f" after {' and '.join(notes)}" if notes else "")

    def as_dict(self) -> dict:
        return {**asdict(self), "verdict": self.verdict}

    def text(self) -> str:
        lines = [f"Feasibility report: {self.target}",
                 f"  rows {self.rows:,}  models {self.models:,}  tasks {self.tasks:,}  density {self.density:.1%}",
                 f"  largest complete block: {self.block_tasks:,} tasks x {self.block_models:,} models"
                 f" (coverage cutoff {self.coverage_cutoff:,})",
                 f"  score range [{self.score_min:g}, {self.score_max:g}]"
                 + (f" -> rescale {self.rescale}" if self.rescale else "")]
        lines += [f"  [{'x' if c.passed else (' ' if c.blocking else '?')}] {c.name:12s} {c.detail}" for c in self.checks]
        lines.append(f"  verdict: {self.verdict}")
        return "\n".join(lines)


def _rescale(lo: float, hi: float) -> tuple[bool, str | None]:
    if lo >= 0 and hi <= 1:
        return True, None
    if lo >= 1 and hi <= 10:
        return True, "x / 10"
    if lo >= 0 and hi <= 100:
        return True, "x / 100"
    return False, None


def assess(long: pd.DataFrame, target: str = "table", min_tasks: int = MIN_TASKS,
           min_models: int = MIN_MODELS) -> FeasibilityReport:
    """Assess a long table with columns task_id, model_id, score."""
    frame = long[["task_id", "model_id", "score"]].copy()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    numeric = frame["score"].notna() & np.isfinite(frame["score"].fillna(0))
    frame = frame[numeric]
    n_models, n_tasks = frame["model_id"].nunique(), frame["task_id"].nunique()
    dupes = int(frame.duplicated(["task_id", "model_id"]).sum())
    pairs = frame.drop_duplicates(["task_id", "model_id"])
    cover = {m: set(g["task_id"]) for m, g in pairs.groupby("model_id")}
    if cover:
        models, tasks, cutoff = max_cells_block(cover)
    else:
        models, tasks, cutoff = [], set(), 0
    lo = float(frame["score"].min()) if len(frame) else float("nan")
    hi = float(frame["score"].max()) if len(frame) else float("nan")
    bounded, rescale = _rescale(lo, hi) if len(frame) else (False, None)
    block = pairs[pairs["model_id"].isin(models) & pairs["task_id"].isin(tasks)]
    spread = block.groupby("model_id")["score"].mean()
    per_model = pairs.groupby("model_id").size()
    checks = [
        Check("task_level", n_tasks >= 2 and bool((per_model > 1).all()) if len(per_model) else False,
              f"{n_tasks:,} distinct tasks; median {int(per_model.median()) if len(per_model) else 0} per model"),
        Check("stable_ids", dupes == 0, "no repeated (task, model) pairs" if dupes == 0
              else f"{dupes:,} repeated (task, model) pairs: decide whether they are trials (average) "
                   "or re-submissions (keep one)", blocking=False),
        Check("shared_pool", len(models) >= 2 and len(tasks) >= 2,
              f"complete block keeps {len(models):,} of {n_models:,} models and {len(tasks):,} of {n_tasks:,} tasks"),
        Check("min_tasks", len(tasks) >= min_tasks, f"{len(tasks):,} tasks in the block (need {min_tasks})"),
        Check("min_models", len(models) >= min_models, f"{len(models):,} models in the block (need {min_models})"),
        Check("score_range", bounded, f"[{lo:g}, {hi:g}]" + (f", rescale {rescale}" if rescale else "")
              if len(frame) else "no numeric scores"),
        Check("variation", len(spread) >= 2 and float(spread.max() - spread.min()) > 0,
              f"model means span {float(spread.max() - spread.min()) if len(spread) else 0:.4g}"),
    ]
    return FeasibilityReport(
        target=target, rows=len(long), models=n_models, tasks=n_tasks,
        density=len(pairs) / (n_models * n_tasks) if n_models and n_tasks else 0.0,
        block_models=len(models), block_tasks=len(tasks), coverage_cutoff=int(cutoff),
        score_min=lo, score_max=hi, rescale=rescale, checks=checks,
    )
