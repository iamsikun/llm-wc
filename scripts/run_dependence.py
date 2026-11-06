"""Task-dependence stress tests.

For the five SWE-bench variants (repository labels) and Tau2 telecom (three
issue subcategories), compare three resampling schemes:
  pooled      m-out-of-N task rows (as in the audit)
  stratified  m-out-of-N rows allocated proportionally within each cluster
  wild        wild cluster bootstrap, one Webb multiplier per cluster
B = 5000, seed 42, m = floor(N^0.9). The wild bootstrap uses no sqrt(m/N) rescaling.

Writes results/dependence/summary.csv, one row per (setting, scheme).

    uv run python scripts/run_dependence.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from llm_wc.benchmarks import load_settings
from llm_wc.wc.core import _stratified_moon_bootstrap_estimates, _wild_cluster_bootstrap_estimates
from llm_wc.wc.ruler import iid_bootstrap_means, position_bias_and_rpm, rank_intervals_and_tiers

OUT = Path(__file__).resolve().parents[1] / "results" / "dependence"
SETTINGS = ("swebench_bash_only", "swebench_verified", "swebench_multimodal",
            "swebench_multilingual", "swebench_pro", "tau2_telecom")
B, SEED, GAMMA = 5000, 42, 0.9


def main() -> None:
    rows = []
    for key, s in load_settings(SETTINGS).items():
        obs, labels = s.obs, s.clusters
        n = obs.shape[0]
        m = int(n ** GAMMA)
        order = np.argsort(-obs.mean(0), kind="stable")
        scores = obs.mean(0)[order]
        schemes = {"pooled": (iid_bootstrap_means(obs, m, B, SEED), m)}
        if key.startswith("swebench_"):
            np.random.seed(SEED)
            boot, actual_m = _stratified_moon_bootstrap_estimates(obs, labels, B, m)
            schemes["stratified"] = (boot, actual_m)
        np.random.seed(SEED)
        schemes["wild"] = (_wild_cluster_bootstrap_estimates(obs, labels, B, "webb", obs.mean(0)), n)
        for scheme, (boot, size) in schemes.items():
            bias, rpm = position_bias_and_rpm(obs, boot, size)
            rpm = rpm[order]
            row = {"setting": key, "scheme": scheme, "n_tasks": n, "clusters": len(set(labels)),
                   "resample_size": size, "position_1_adjustment_pp": 100 * bias[0]}
            row.update({f"p_rank_1_position_{i + 1}": rpm[i, 0] for i in range(3)})
            try:
                _, _, _, tiers = rank_intervals_and_tiers(rpm, scores, 0.2)
                row.update(tiers_80=len(tiers), tier_1_size_80=tiers[0]["size"])
            except ValueError:  # an observed tie block without a common rank
                row.update(tiers_80=None, tier_1_size_80=None)
            rows.append(row)
            print(f"{key} {scheme}: m={size} position-1={row['position_1_adjustment_pp']:.3f} pp "
                  f"tier1={row['tier_1_size_80']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
