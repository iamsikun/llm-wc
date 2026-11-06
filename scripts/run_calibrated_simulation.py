"""Calibrated simulation with known model means.

1. Fit the logistic DGP Y_ij ~ Bernoulli(sigmoid(alpha_j + T_i)), T_i ~ N(0, sigma_T^2)
   to SWE-bench Verified (constant tasks dropped: N = 472, J = 173).
2. Draw 100 synthetic benchmarks (seed 42 + r). On each, record the score error of
   the models at positions 1-5 and three adjustments computed on the same draw:
   m-out-of-N bootstrap (m = floor(N^0.9)), ordinary bootstrap (m = N), the
   beta-binomial empirical-Bayes posterior-mean shrinkage, and sample splitting
   (selection-half minus evaluation-half score of the selection-half ranks, averaged
   over 200 random half-splits and rescaled by sqrt(N_S / N)). B = 2000.

Writes results/calibrated_simulation/:
  calibration.npz   fitted alpha, sigma, marginal means (taus), task effects
  replications.csv  one row per (replication, position)

    uv run python scripts/run_calibrated_simulation.py
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from llm_wc.benchmarks import load_setting
from llm_wc.simulation.calibration import beta_binomial_eb, calibrate_logistic
from llm_wc.simulation.dgp import simulate_bernoulli_logit_fe
from llm_wc.wc.ruler import iid_bootstrap_means

OUT = Path(__file__).resolve().parents[1] / "results" / "calibrated_simulation"
N_REPS, B, SEED, TOP_K, N_SPLITS = 100, 2000, 42, 5, 200


def replicate(rep: int, alpha: np.ndarray, sigma: float, taus: np.ndarray, n: int) -> list[dict]:
    obs = simulate_bernoulli_logit_fe(alpha=alpha, sigma_t=sigma, n_tasks=n, rng=np.random.default_rng(SEED + rep))
    empirical = obs.mean(0)
    order = np.argsort(-empirical)
    out = {"raw_error": empirical[order[:TOP_K]] - taus[order[:TOP_K]]}
    for name, m, offset in [("moon", int(n ** .9), 5_000_000), ("ordinary", n, 7_000_000)]:
        boot = iid_bootstrap_means(obs, m, B, seed=SEED + rep + offset)
        rank = np.argsort(-boot, axis=1)
        out[name] = ((np.take_along_axis(boot, rank, axis=1) - empirical[rank]).mean(0) * np.sqrt(m / n))[:TOP_K]
    a, b = beta_binomial_eb(obs.sum(0), n)
    posterior = (obs.sum(0) + a) / (n + a + b)
    out["empirical_bayes"] = (empirical - posterior)[order[:TOP_K]]
    split_rng, n_s, split = np.random.default_rng(SEED + rep + 9_000_000), n // 2, np.zeros(TOP_K)
    for _ in range(N_SPLITS):
        perm = split_rng.permutation(n)
        sel, ev = obs[perm[:n_s]].mean(0), obs[perm[n_s:]].mean(0)
        top = np.argsort(-sel)[:TOP_K]
        split += sel[top] - ev[top]
    out["sample_splitting"] = split / N_SPLITS * np.sqrt(n_s / n)
    return [{"replication": rep, "position": k + 1, "model": int(order[k]), "true_mean": taus[order[k]],
             "raw_error": out["raw_error"][k], "moon": out["moon"][k], "ordinary": out["ordinary"][k],
             "empirical_bayes": out["empirical_bayes"][k], "sample_splitting": out["sample_splitting"][k],
             "eb_a": a, "eb_b": b}
            for k in range(TOP_K)]


def main() -> None:
    s = load_setting("swebench_verified")
    cal = calibrate_logistic(s.obs, list(s.model_names), list(s.task_ids))
    print(f"Calibration: {cal.n_tasks} tasks, {cal.n_constant} constant rows excluded, sigma={cal.sigma:.4f}",
          flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "calibration.npz", alpha=cal.alpha, sigma=cal.sigma, taus=cal.taus,
             task_effect=cal.task_effect, n_eff=cal.n_tasks, n_constant=cal.n_constant,
             model_order=np.asarray(cal.model_order, dtype=str))
    with ThreadPoolExecutor(max_workers=3) as pool:
        reps = list(pool.map(lambda r: replicate(r, cal.alpha, cal.sigma, cal.taus, cal.n_tasks), range(N_REPS)))
    frame = pd.DataFrame([row for rep in reps for row in rep])
    frame.to_csv(OUT / "replications.csv", index=False)
    first = frame[frame.position == 1].mean(numeric_only=True)
    print(f"Position 1 (pp): raw bias {100 * first.raw_error:.2f}; adjustments moon {100 * first.moon:.2f}, "
          f"ordinary {100 * first.ordinary:.2f}, EB {100 * first.empirical_bayes:.2f}, "
          f"sample splitting {100 * first.sample_splitting:.2f}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
