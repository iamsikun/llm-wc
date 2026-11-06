"""RULER: Rank Uncertainty and Leaderboard Evaluation Report.

Builds on the m-out-of-N task-row bootstrap:

- position adjustments (per rank position, NumPy sort order for ties);
- a fractional-tie rank probability matrix (RPM): a bootstrap tie of q models
  over positions a..a+q-1 gives each of them mass 1/q at each position;
- central rank-mass intervals [Q(alpha/2), Q(1 - alpha/2)];
- tiers by the observed-order common-intersection rule: walk models in raw
  score order (observed ties as indivisible blocks) and extend the current
  tier while all intervals still share a rank.

``ruler(obs)`` runs all of it. Rows of every per-model output are in raw
score order (stable sort, so observed ties keep the input column order).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TOL = 1e-12


def iid_bootstrap_means(
    obs: np.ndarray, m: int, n_bootstraps: int = 5000, seed: int = 42, chunk: int = 128
) -> np.ndarray:
    """(B, J) means of B resamples of m task rows drawn with replacement.

    Uses the same row-index stream as ``np.random.seed(seed)`` followed by
    ``np.random.randint(0, N, size=(B, m))``, with bounded memory.
    """
    n, j = obs.shape
    rng = np.random.RandomState(seed)
    out = np.empty((n_bootstraps, j))
    for start in range(0, n_bootstraps, chunk):
        size = min(chunk, n_bootstraps - start)
        idx = rng.randint(0, n, size=(size, m))
        counts = np.zeros((size, n), dtype=float)
        np.add.at(counts, (np.arange(size)[:, None], idx), 1)
        out[start:start + size] = counts @ obs / m
    return out


def position_bias_and_rpm(
    obs: np.ndarray, estimates: np.ndarray, m: int
) -> tuple[np.ndarray, np.ndarray]:
    """Position adjustments (J,) and fractional-tie RPM (J models x J ranks).

    ``estimates`` are (B, J) bootstrap means from resamples of size m; the
    adjustment is rescaled by sqrt(m / N). RPM rows follow ``obs`` columns.
    """
    n, j = obs.shape
    emp = obs.mean(0)
    order = np.argsort(-estimates, axis=1)
    sorted_values = np.take_along_axis(estimates, order, axis=1)
    bias = (sorted_values - emp[order]).mean(0) * np.sqrt(m / n)
    rpm = np.zeros((j, j))
    for indices, values in zip(order, sorted_values):
        starts = np.r_[0, np.flatnonzero(np.abs(np.diff(values)) > TOL) + 1]
        ends = np.r_[starts[1:], j]
        for lo, hi in zip(starts, ends):
            rpm[np.ix_(indices[lo:hi], np.arange(lo, hi))] += 1 / (hi - lo)
    rpm /= len(estimates)
    assert np.allclose(rpm.sum(0), 1) and np.allclose(rpm.sum(1), 1)
    return bias, rpm


def rank_intervals_and_tiers(
    rpm: np.ndarray, scores: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Central rank-mass intervals and common-intersection tiers.

    ``rpm`` rows and ``scores`` must be in raw score order. Returns
    (rank_low, rank_high, tier_id, tiers); ranks and tier ids are 1-based.
    Raises if an observed tie block has no common rank.
    """
    cdf = rpm.cumsum(1)
    cdf[:, -1] = 1
    low = (cdf >= alpha / 2 - TOL).argmax(1) + 1
    high = (cdf >= 1 - alpha / 2 - TOL).argmax(1) + 1
    boundaries = np.r_[0, np.flatnonzero(np.abs(np.diff(scores)) > TOL) + 1, len(scores)]
    tiers = []
    begin, left, right = None, None, None
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        block_l, block_u = int(low[start:end].max()), int(high[start:end].min())
        if block_l > block_u:
            raise ValueError(f"Observed tie block {start + 1}:{end} has incompatible rank intervals")
        if begin is not None and max(left, block_l) > min(right, block_u):
            tiers.append(dict(start_position=begin + 1, end_position=int(start),
                              common_rank_low=left, common_rank_high=right))
            begin = None
        if begin is None:
            begin, left, right = int(start), block_l, block_u
        else:
            left, right = max(left, block_l), min(right, block_u)
    tiers.append(dict(start_position=begin + 1, end_position=len(scores),
                      common_rank_low=left, common_rank_high=right))
    labels = np.zeros(len(scores), dtype=int)
    for t, tier in enumerate(tiers, 1):
        tier["tier"] = t
        tier["size"] = tier["end_position"] - tier["start_position"] + 1
        labels[tier["start_position"] - 1:tier["end_position"]] = t
    return low, high, labels, tiers


def observed_tie_blocks(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1-based (rank_start, rank_end) of each model's observed tie block."""
    boundaries = np.r_[0, np.flatnonzero(np.abs(np.diff(scores)) > TOL) + 1, len(scores)]
    start = np.empty(len(scores), dtype=int)
    end = np.empty(len(scores), dtype=int)
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        start[lo:hi], end[lo:hi] = lo + 1, hi
    return start, end


def tie_averaged(position_values: np.ndarray, rank_start: np.ndarray, rank_end: np.ndarray):
    """Mean, min, and max of position values over each model's tie block."""
    blocks = [position_values[s - 1:e] for s, e in zip(rank_start, rank_end)]
    return (np.array([b.mean() for b in blocks]), np.array([b.min() for b in blocks]),
            np.array([b.max() for b in blocks]))


@dataclass
class RulerResult:
    """RULER for one leaderboard. Per-model arrays are in raw score order."""

    n_tasks: int
    n_models: int
    m: int
    n_bootstraps: int
    seed: int
    order: np.ndarray            # (J,) column index of the model at each raw position
    scores: np.ndarray           # (J,) raw scores, descending
    rank_start: np.ndarray       # (J,) observed tie block, 1-based
    rank_end: np.ndarray
    position_bias: np.ndarray    # (J,) adjustment at each rank position
    adjustment: np.ndarray       # (J,) tie-averaged displayed adjustment
    adjustment_low: np.ndarray
    adjustment_high: np.ndarray
    rpm: np.ndarray              # (J models in raw order, J ranks)
    intervals: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)
    tiers: dict[float, list[dict]] = field(default_factory=dict)


def ruler(
    obs: np.ndarray,
    *,
    gamma: float = 0.9,
    m: int | None = None,
    n_bootstraps: int = 5000,
    seed: int = 42,
    alphas: tuple[float, ...] = (0.2, 0.1, 0.05, 0.01),
) -> RulerResult:
    """Run the m-out-of-N bootstrap (m = floor(N**gamma) unless given) and build RULER."""
    n, j = obs.shape
    m = int(n ** gamma) if m is None else m
    bias, rpm = position_bias_and_rpm(obs, iid_bootstrap_means(obs, m, n_bootstraps, seed), m)
    emp = obs.mean(0)
    order = np.argsort(-emp, kind="stable")
    scores = emp[order]
    rpm = rpm[order]
    start, end = observed_tie_blocks(scores)
    adj, adj_lo, adj_hi = tie_averaged(bias, start, end)
    result = RulerResult(n, j, m, n_bootstraps, seed, order, scores, start, end,
                         bias, adj, adj_lo, adj_hi, rpm)
    for alpha in alphas:
        low, high, ids, tiers = rank_intervals_and_tiers(rpm, scores, alpha)
        result.intervals[alpha] = (low, high, ids)
        result.tiers[alpha] = tiers
    return result
