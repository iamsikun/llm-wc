"""Data preparation utilities for logistic models of binary benchmark outcomes.

Handles conversion from wide-format outcome matrices to long-format DataFrames
suitable for model fitting.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def validate_long_format(
    data: pd.DataFrame,
    *,
    y_col: str = "y",
    agent_col: str = "agent",
    task_col: str = "task",
    repo_col: str = "repo",
) -> pd.DataFrame:
    """Validate and rename columns to standard names (y, agent, task, repo).

    Raises KeyError if required columns are missing, ValueError if data
    constraints are violated (nulls, non-binary y, tasks in multiple repos).
    """
    cols = [y_col, agent_col, task_col, repo_col]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = data[cols].copy()
    df = df.rename(
        columns={
            y_col: "y",
            agent_col: "agent",
            task_col: "task",
            repo_col: "repo",
        }
    )

    if df[["agent", "task", "repo"]].isnull().any().any():
        raise ValueError("agent/task/repo columns must not contain null values.")
    if not set(pd.unique(df["y"])) <= {0, 1}:
        raise ValueError("y must be binary and coded as 0/1.")

    df["agent"] = df["agent"].astype(str)
    df["task"] = df["task"].astype(str)
    df["repo"] = df["repo"].astype(str)
    df["y"] = df["y"].astype(int)

    repo_per_task = df.groupby("task")["repo"].nunique()
    bad_tasks = repo_per_task[repo_per_task != 1]
    if len(bad_tasks) > 0:
        raise ValueError(
            "Each task must belong to exactly one repo. Violating tasks: "
            f"{bad_tasks.index.tolist()[:10]}"
        )

    return df


def prepare_long_format(
    obs_matrix: np.ndarray,
    model_names: list[str],
    task_ids: list[str],
    repo_labels: np.ndarray,
) -> tuple[pd.DataFrame, int, int]:
    """Build long-format DataFrame from a wide outcome matrix.

    Disambiguates duplicate model names and filters constant tasks
    (all-zero or all-one across models).

    Args:
        obs_matrix: (N_tasks, N_models) binary outcome matrix.
        model_names: model name strings, length N_models.
        task_ids: task ID strings, length N_tasks.
        repo_labels: repo label per task, length N_tasks.

    Returns:
        (df, n_constant_filtered, n_disambiguated)
    """
    n_tasks, n_models = obs_matrix.shape

    # Disambiguate duplicate model names
    _name_count: dict[str, int] = {}
    unique_agent_names: list[str] = []
    for name in model_names:
        _name_count[name] = _name_count.get(name, 0) + 1
        if _name_count[name] > 1:
            unique_agent_names.append(f"{name} (v{_name_count[name]})")
        else:
            unique_agent_names.append(name)
    n_disambiguated = sum(
        1
        for new, orig in zip(unique_agent_names, model_names)
        if new != orig
    )

    # Filter constant tasks
    nonconstant_mask = ~np.isclose(obs_matrix.var(axis=1), 0.0)
    n_filtered = int((~nonconstant_mask).sum())

    records: list[dict[str, Any]] = []
    for t_idx in range(n_tasks):
        if not nonconstant_mask[t_idx]:
            continue
        for m_idx in range(n_models):
            val = obs_matrix[t_idx, m_idx]
            if np.isnan(val):
                continue
            records.append(
                {
                    "y": int(val),
                    "agent": unique_agent_names[m_idx],
                    "task": task_ids[t_idx],
                    "repo": str(repo_labels[t_idx]),
                }
            )

    return pd.DataFrame(records), n_filtered, n_disambiguated
