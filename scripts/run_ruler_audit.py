"""RULER audit of the 14 leaderboard settings, with an interactive report.

For every setting: m-out-of-N bootstrap with m = floor(N^0.9), B = 5000, seed 42;
position adjustments, tie-averaged displayed adjustments, the fractional-tie
rank probability matrix, central rank-mass intervals and tiers at 80/90/95/99%
rank mass. The ordinary N-out-of-N bootstrap adjustment is saved alongside.

Writes results/ruler_audit/:
  summary.csv                one row per setting
  <setting>/models.csv       one row per model, raw score order
  <setting>/position_bias.csv
  <setting>/rpm.csv          models (raw order) x ranks
  <setting>/tiers.csv

    uv run python scripts/run_ruler_audit.py [--settings swebench_verified ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from llm_wc.benchmarks import SETTINGS, SHORT_LABELS, load_settings
from llm_wc.wc.ruler import iid_bootstrap_means, position_bias_and_rpm, ruler, tie_averaged

OUT = Path(__file__).resolve().parents[1] / "results" / "ruler_audit"
GAMMA, B, SEED = 0.9, 5000, 42
ALPHAS = (0.2, 0.1, 0.05, 0.01)


def level(alpha: float) -> str:
    return str(round((1 - alpha) * 100))


def audit(setting) -> dict:
    obs = setting.obs
    n, j = obs.shape
    r = ruler(obs, gamma=GAMMA, n_bootstraps=B, seed=SEED, alphas=ALPHAS)
    ordinary, _ = position_bias_and_rpm(obs, iid_bootstrap_means(obs, n, B, SEED), n)
    ord_adj, ord_lo, ord_hi = tie_averaged(ordinary, r.rank_start, r.rank_end)

    out = OUT / setting.key
    out.mkdir(parents=True, exist_ok=True)
    rank_label = [str(s) if s == e else f"{s}–{e}" for s, e in zip(r.rank_start, r.rank_end)]
    models = pd.DataFrame({
        "position": np.arange(1, j + 1),
        "model_id": setting.model_ids[r.order],
        "model": setting.model_names[r.order],
        "raw_score_pct": 100 * r.scores,
        "rank": rank_label,
        "rank_start": r.rank_start,
        "rank_end": r.rank_end,
        "adjustment_pp": 100 * r.adjustment,
        "adjustment_low_pp": 100 * r.adjustment_low,
        "adjustment_high_pp": 100 * r.adjustment_high,
        "ordinary_adjustment_pp": 100 * ord_adj,
        "p_rank_1": r.rpm[:, 0],
        "p_top_5": r.rpm[:, :5].sum(1),
    })
    for alpha in ALPHAS:
        low, high, ids = r.intervals[alpha]
        models[f"rank_low_{level(alpha)}"] = low
        models[f"rank_high_{level(alpha)}"] = high
        models[f"tier_{level(alpha)}"] = ids
    models.to_csv(out / "models.csv", index=False)
    pd.DataFrame({"position": np.arange(1, j + 1), "adjustment_pp": 100 * r.position_bias,
                  "ordinary_adjustment_pp": 100 * ordinary}).to_csv(out / "position_bias.csv", index=False)
    pd.DataFrame(r.rpm, index=pd.Index(models.model_id, name="model_id"),
                 columns=[str(k) for k in range(1, j + 1)]).to_csv(out / "rpm.csv")
    tiers = []
    for alpha in ALPHAS:
        for t in r.tiers[alpha]:
            tiers.append({"rank_mass": level(alpha), **t,
                          "observed_rank_start": int(r.rank_start[t["start_position"] - 1]),
                          "observed_rank_end": int(r.rank_end[t["end_position"] - 1])})
    pd.DataFrame(tiers).to_csv(out / "tiers.csv", index=False)

    row = {"setting": setting.key, "label": SHORT_LABELS[setting.key], "n_tasks": n, "n_models": j,
           "m": r.m, "n_bootstraps": B, "seed": SEED,
           "position_1_adjustment_pp": 100 * r.position_bias[0],
           "leader_adjustment_pp": 100 * r.adjustment[0],
           "leader_adjustment_low_pp": 100 * r.adjustment_low[0],
           "leader_adjustment_high_pp": 100 * r.adjustment_high[0],
           "leader_ordinary_adjustment_pp": 100 * ord_adj[0],
           "leader_p_rank_1": r.rpm[0, 0]}
    for alpha in ALPHAS:
        row[f"tiers_{level(alpha)}"] = len(r.tiers[alpha])
        row[f"tier_1_size_{level(alpha)}"] = r.tiers[alpha][0]["size"]
    print(f"{setting.key}: N={n} J={j} m={r.m} position-1={row['position_1_adjustment_pp']:.3f} pp "
          f"tiers={row['tiers_80']} tier1={row['tier_1_size_80']}", flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=SETTINGS)
    args = ap.parse_args()
    rows = [audit(s) for s in load_settings(args.settings).values()]
    path = OUT / "summary.csv"
    if path.exists() and set(args.settings) != set(SETTINGS):
        old = pd.read_csv(path)
        rows = old[~old.setting.isin(args.settings)].to_dict("records") + rows
        rows.sort(key=lambda x: SETTINGS.index(x["setting"]))
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
