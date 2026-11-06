"""Lookup table of the resample exponent gamma for the m-out-of-n correction.

J models answer N tasks with independent Bernoulli outcomes. Their true means
are evenly spaced over a range of k * sigma centred on p (so p is their mean and
median), sigma = sqrt(p (1 - p) / N): "tied" k = 0 (every model has mean p),
"close" k = 2 (a crowded top of a leaderboard), "spread" k = 15 (a typical whole
leaderboard; the reference settings span 3-51 sigma). For each gamma in {0.1, ..., 0.9, 0.95, 1}: m = max(2, floor(N^gamma)), gamma = 1
(m = N) is the ordinary bootstrap. The corrected score at position k is the
k-th observed score minus boot_wc[k], computed exactly as bootstrap_correction
(iid / moon) does; the error at position k is the corrected score minus the true mean of the
model observed at position k. The root-mean-square error (RMSE) averages its
square over the J positions and the benchmark draws; the mean absolute error
(MAE) averages its absolute value. "none" is the uncorrected observed score.

Grid: p in {0.5, 0.75}, N in {500, 2000}, J in {2, 5, 10, 20, 50, 100}.
R = 1000 benchmark draws, B = 2000 resamples. Within a draw every (p, spread) shares the
uniforms of a given J, and every (p, spread, J) shares the resample indices of a
given gamma.

Writes results/gamma_table/summary.csv, one row per (N, p, spread, J, gamma).

    uv run python scripts/run_gamma_table.py
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "results" / "gamma_table"
GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
PS = (0.5, 0.75)
NS = (500, 2000)
JS = (2, 5, 10, 20, 50, 100)
SPREADS = {"tied": 0.0, "close": 2.0, "spread": 15.0}  # range of true means, in sigma
R, B, SEED = 1000, 2000, 20261103


def resample_size(n: int, gamma: float) -> int:
    return min(max(2, int(n ** gamma)), n)


def position_bias(boot_est: np.ndarray, emp_est: np.ndarray, m: int, n: int) -> np.ndarray:
    """boot_wc of bootstrap_correction for boot_est (B, J) from m-row resamples."""
    boot_rank = np.argsort(-boot_est, axis=1)
    wc = np.mean(np.take_along_axis(boot_est, boot_rank, axis=1) - emp_est[boot_rank], axis=0)
    return wc * np.sqrt(m / n) if m < n else wc


def resample_sums(obs: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Column sums of obs[idx[b]] for each b, exact for 0/1 obs (float32 integers < 2^24)."""
    b, n = idx.shape[0], obs.shape[0]
    flat = (idx + n * np.arange(b)[:, None]).ravel()
    counts = np.bincount(flat, minlength=b * n).reshape(b, n).astype(np.float32)
    return (counts @ obs.astype(np.float32)).astype(np.float64)


def blocks(n: int) -> list[tuple[float, str, int, slice, slice, np.ndarray]]:
    """(p, spread, J, column slice of obs, column slice of the shared uniforms, true means) per block."""
    out, col, ucol = [], 0, {}
    for j in JS:
        ucol[j] = slice(sum(JS[:JS.index(j)]), sum(JS[:JS.index(j) + 1]))
    for p in PS:
        sigma = np.sqrt(p * (1 - p) / n)
        for name, k in SPREADS.items():
            for j in JS:
                out.append((p, name, j, slice(col, col + j), ucol[j], p + k * sigma * np.linspace(-.5, .5, j)))
                col += j
    return out


def one_draw(args: tuple[int, np.random.SeedSequence]) -> np.ndarray:
    """Errors for one benchmark draw: (len(blocks), 1 + len(GAMMAS), 3) of mean |err|, |err| at the top, and mean err^2."""
    n, ss = args
    rng = np.random.default_rng(ss)
    bl = blocks(n)
    u = rng.random((n, sum(JS)))
    obs = np.concatenate([u[:, uc] < mu for _, _, _, _, uc, mu in bl], axis=1).astype(np.float64)
    emp = obs.mean(axis=0)
    order = [np.argsort(-emp[c]) for _, _, _, c, _, _ in bl]
    out = np.empty((len(bl), 1 + len(GAMMAS), 3))

    def record(bi, gi, wc):
        _, _, _, c, _, mu = bl[bi]
        err = emp[c][order[bi]] - wc - mu[order[bi]]
        out[bi, gi] = (np.abs(err).mean(), abs(err[0]), (err ** 2).mean())

    for bi in range(len(bl)):
        record(bi, 0, 0.0)
    for gi, gamma in enumerate(GAMMAS, start=1):
        m = resample_size(n, gamma)
        sums = resample_sums(obs, rng.integers(0, n, size=(B, m)))
        for bi, (_, _, _, c, _, _) in enumerate(bl):
            record(bi, gi, position_bias(sums[:, c] / m, emp[c], m, n))
    return out


def main() -> None:
    rows = []
    for n in NS:
        seeds = np.random.SeedSequence([SEED, n]).spawn(R)
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
            draws = np.stack(list(pool.map(one_draw, [(n, s) for s in seeds], chunksize=8)))  # (R, blocks, G, 3)
        mean, se = draws.mean(axis=0), draws.std(axis=0, ddof=1) / np.sqrt(R)
        for bi, (p, name, j, _, _, _) in enumerate(blocks(n)):
            for gi, gamma in enumerate((None,) + GAMMAS):
                rows.append({"n_tasks": n, "p": p, "spread": name, "range_sigma": SPREADS[name], "n_models": j,
                             "gamma": gamma, "m": None if gamma is None else resample_size(n, gamma),
                             "mae": mean[bi, gi, 0], "mae_se": se[bi, gi, 0], "mae_top": mean[bi, gi, 1],
                             # delta method: se(sqrt(MSE)) = se(MSE) / (2 sqrt(MSE))
                             "rmse": np.sqrt(mean[bi, gi, 2]), "rmse_se": se[bi, gi, 2] / (2 * np.sqrt(mean[bi, gi, 2]))})
        print(f"N = {n} done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
