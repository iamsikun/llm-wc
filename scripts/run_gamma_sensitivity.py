"""Sensitivity of the position-1 adjustment to the resample exponent.

For each of the 14 settings and gamma in {0.1, ..., 0.9, 0.95, 1}: m = max(2,
floor(N^gamma)); gamma = 1 (m = N) is the ordinary bootstrap. B = 5000; the
global NumPy seed is reset to 42 before every run.

Writes results/gamma_sensitivity/summary.csv, one row per (setting, gamma).

    uv run python scripts/run_gamma_sensitivity.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from llm_wc.benchmarks import load_settings
from llm_wc.wc.core import bootstrap_correction

OUT = Path(__file__).resolve().parents[1] / "results" / "gamma_sensitivity"
GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
B, SEED = 5000, 42


def main() -> None:
    rows = []
    for key, s in load_settings().items():
        n, j = s.obs.shape
        for gamma in GAMMAS:
            m = max(2, int(n ** gamma))
            method = "iid" if m >= n else "moon"
            np.random.seed(SEED)
            res = bootstrap_correction(obs=s.obs, n_bootstraps=B, compute_partial_order=False,
                                       bootstrap_method=method, boot_size=min(m, n))
            rows.append({"setting": key, "gamma": gamma, "bootstrap_method": method, "n_tasks": n,
                         "n_models": j, "m": min(m, n), "position_1_adjustment_pp": 100 * res.boot_wc[0]})
        print(f"{key}: " + " ".join(f"{r['position_1_adjustment_pp']:.2f}" for r in rows[-len(GAMMAS):]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
