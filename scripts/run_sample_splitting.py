"""Half-split validation across the 14 settings.

Each split draws N_A = floor(N/2) tasks (balanced across strata where the
setting has them). Half A ranks the models and estimates the m-out-of-N
position adjustment (m = floor(N_A^0.9), B = 2000); half B scores the same
selected models. For positions 1-5 the script records

  split_error = score_A - score_B of the model half A placed there
  adjustment  = bootstrap position adjustment estimated on half A

Seeds: split seeds come from default_rng(42); split r uses bootstrap seed
42 + 1,000,000 + 10,000 r + 4 (the alpha = 0.90 slot of the original grid).

Writes results/sample_splitting/splits.csv (one row per setting, split, position)
and settings.csv.

    uv run python scripts/run_sample_splitting.py [--n-jobs -1]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from llm_wc.benchmarks import SETTINGS, load_settings

OUT = Path(__file__).resolve().parents[1] / "results" / "sample_splitting"


def stratified_half_split(strata: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Exact half split with near-equal allocation in every stratum."""
    unique_strata = np.unique(strata)
    members_by_stratum = [np.flatnonzero(strata == s) for s in unique_strata]
    target_a = len(strata) // 2

    # Start from the lower integer allocation in every stratum, then randomly
    # give the remaining slots to strata whose 50/50 quota is fractional.
    # Singleton strata may land on either side; larger strata get >= 1 task per half.
    lower, upper = [], []
    for members in members_by_stratum:
        size = len(members)
        if size == 1:
            lower.append(0)
            upper.append(1)
        else:
            lower.append(max(1, min(size // 2, size - 1)))
            upper.append(max(1, min((size + 1) // 2, size - 1)))
    allocation = np.array(lower, dtype=int)
    remaining = target_a - int(allocation.sum())
    candidates = np.flatnonzero(np.array(upper) > allocation)
    if not 0 <= remaining <= len(candidates):
        raise ValueError(f"Cannot split: target={target_a}, lower={allocation.sum()}, candidates={len(candidates)}")
    if remaining:
        allocation[rng.choice(candidates, size=remaining, replace=False)] += 1

    idx_a, idx_b = [], []
    for members, n_a in zip(members_by_stratum, allocation, strict=True):
        permuted = rng.permutation(members)
        idx_a.extend(permuted[:n_a].tolist())
        idx_b.extend(permuted[n_a:].tolist())
    return np.array(idx_a, dtype=int), np.array(idx_b, dtype=int)


def bootstrap_position_bias(obs: np.ndarray, *, n_bootstraps: int, boot_size: int, seed: int,
                            chunk_size: int) -> np.ndarray:
    """m-out-of-N position adjustment without materializing B x m x J."""
    n_tasks, n_models = obs.shape
    empirical = obs.mean(axis=0)
    bias_sum = np.zeros(n_models)
    rng = np.random.RandomState(seed)
    for start in range(0, n_bootstraps, chunk_size):
        chunk = min(chunk_size, n_bootstraps - start)
        estimates = obs[rng.randint(0, n_tasks, size=(chunk, boot_size))].mean(axis=1)
        ranks = np.argsort(-estimates, axis=1)
        bias_sum += np.sum(np.take_along_axis(estimates, ranks, axis=1) - empirical[ranks], axis=0)
    return bias_sum / float(n_bootstraps) * math.sqrt(float(boot_size) / float(n_tasks))


def run_one_split(obs, strata, *, split_index, split_seed, n_bootstraps, gamma, top_k, base_seed, chunk_size):
    idx_a, idx_b = stratified_half_split(strata, np.random.default_rng(split_seed))
    score_a, score_b = obs[idx_a].mean(0), obs[idx_b].mean(0)
    selected = np.argsort(-score_a)[:top_k]
    m = int(len(idx_a) ** gamma)
    adjustment = bootstrap_position_bias(obs[idx_a], n_bootstraps=n_bootstraps, boot_size=m,
                                         seed=base_seed + 1_000_000 + split_index * 10_000 + 4,
                                         chunk_size=chunk_size)[:top_k]
    return [{"split": split_index, "position": k + 1, "split_error": score_a[j] - score_b[j],
             "adjustment": adjustment[k], "n_a": len(idx_a), "n_b": len(idx_b), "m": m}
            for k, j in enumerate(selected)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-splits", type=int, default=200)
    ap.add_argument("--n-bootstraps", type=int, default=2000)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--chunk-size", type=int, default=64)
    args = ap.parse_args()

    split_seeds = np.random.default_rng(args.seed).integers(0, np.iinfo(np.int32).max, size=args.n_splits,
                                                            dtype=np.int64)
    rows, settings = [], []
    for key, s in load_settings().items():
        print(f"{key}: N={s.obs.shape[0]} J={s.obs.shape[1]} strata={s.stratum_type}", flush=True)
        # Threads share the task matrix; the NumPy kernels release the GIL.
        results = Parallel(n_jobs=args.n_jobs, prefer="threads")(
            delayed(run_one_split)(s.obs, s.strata, split_index=r, split_seed=int(split_seeds[r]),
                                   n_bootstraps=args.n_bootstraps, gamma=args.gamma, top_k=args.top_k,
                                   base_seed=args.seed, chunk_size=args.chunk_size)
            for r in range(args.n_splits))
        rows += [{"setting": key, **row} for split in results for row in split]
        settings.append({"setting": key, "n_tasks": s.obs.shape[0], "n_models": s.obs.shape[1],
                         "stratification": s.stratum_type, "n_strata": len(np.unique(s.strata)),
                         "n_splits": args.n_splits, "n_bootstraps": args.n_bootstraps,
                         "gamma": args.gamma, "seed": args.seed})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "splits.csv", index=False)
    pd.DataFrame(settings).to_csv(OUT / "settings.csv", index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
