"""Two-model illustration.

Two models with independent Bernoulli outcomes on N = 500 tasks, success
probabilities (0.500, 0.505) ("small gap") or (0.500, 0.600) ("large gap").
K = 3000 benchmark draws, B = 2000 bootstrap resamples per draw.

  gamma = 0.9 (m = 268), seed 20260502: the main version. Resampling
      rows is done by multinomial sampling of the four joint-outcome cells,
      which is exactly equivalent in distribution. Observed ties are broken
      uniformly; bootstrap ties get fractional rank mass and their selection
      gap is averaged over tied winners.
  gamma = 0.6 (m = 41), seed 20260502: an alternative version.

Writes results/two_model_toy/:
  gamma09_draws.csv    one row per (gap, draw): winner, raw, adjusted, truth, P*(model 2 at rank 1)
  gamma06_summary.json scenario summaries of the gamma = 0.6 run

    uv run python scripts/run_two_model_toy.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "results" / "two_model_toy"
N, K, B, P1, SEED = 500, 3000, 2000, 0.5, 20260502


def run_gamma09() -> pd.DataFrame:
    m = int(N ** .9)
    rng = np.random.default_rng(SEED)
    patterns = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    rows = []
    for name, gap in [("small_gap", .005), ("large_gap", .1)]:
        p1, p2 = P1, P1 + gap
        probs = np.array([(1 - p1) * (1 - p2), (1 - p1) * p2, p1 * (1 - p2), p1 * p2])
        for draw in range(K):
            counts = rng.multinomial(N, probs)
            emp = counts @ patterns / N
            winner = int(rng.integers(2)) if emp[0] == emp[1] else int(emp.argmax())
            boot = rng.multinomial(m, counts / N, size=B) @ patterns / m
            order = np.argsort(-boot, axis=1)
            ties = np.abs(boot[:, 0] - boot[:, 1]) < 1e-12
            selected_base = emp[order[:, 0]]
            selected_base[ties] = emp.mean()
            adjustment = np.sqrt(m / N) * np.mean(boot.max(1) - selected_base)
            q = np.mean((boot[:, 1] > boot[:, 0]).astype(float) + .5 * ties)
            rows.append({"scenario": name, "p1": p1, "p2": p2, "draw": draw, "winner": winner,
                         "raw": emp[winner], "adjustment": adjustment, "adjusted": emp[winner] - adjustment,
                         "selected_true": [p1, p2][winner], "p_model2_rank1": q})
    return pd.DataFrame(rows)


def run_gamma06() -> dict:
    m = int(np.floor(N ** 0.6))
    rng = np.random.default_rng(SEED)

    def one_trial(eps):
        p2 = P1 + eps
        y = rng.binomial(1, [P1, p2], size=(N, 2))
        p_hat = y.mean(axis=0)
        winner = int(rng.integers(0, 2)) if p_hat[0] == p_hat[1] else int(np.argmax(p_hat))
        p_boot = y[rng.integers(0, N, size=(B, m))].mean(axis=1)
        diff = p_boot[:, 1] - p_boot[:, 0]
        ties = diff == 0
        win_b = (diff > 0).astype(int)
        if ties.any():
            win_b[ties] = rng.integers(0, 2, size=ties.sum())
        los_b = 1 - win_b
        bias1 = np.sqrt(m / N) * (p_boot[np.arange(B), win_b] - p_hat[win_b]).mean()
        bias2 = np.sqrt(m / N) * (p_boot[np.arange(B), los_b] - p_hat[los_b]).mean()
        rpm = np.array([[(win_b == j).mean(), (los_b == j).mean()] for j in range(2)])
        loser = 1 - winner
        return dict(winner=winner, raw1=p_hat[winner], raw2=p_hat[loser], true1=(P1, p2)[winner],
                    true2=(P1, p2)[loser], corr1=p_hat[winner] - bias1, corr2=p_hat[loser] - bias2, R=rpm)

    scenarios = {}
    for label, eps in [("unresolved", 0.005), ("decisive", 0.1)]:
        rows = [one_trial(eps) for _ in range(K)]
        mean = {k: float(np.mean([r[k] for r in rows])) for k in ["raw1", "raw2", "true1", "true2", "corr1", "corr2"]}
        rpm = np.array([r["R"] for r in rows])
        won = np.array([r["winner"] == 1 for r in rows])
        scenarios[label] = dict(
            eps=eps, sel=float(won.mean()), **mean,
            bias1=mean["raw1"] - mean["true1"], bias2=mean["raw2"] - mean["true2"],
            resid1=mean["corr1"] - mean["true1"], resid2=mean["corr2"] - mean["true2"],
            n_true=int(won.sum()), n_false=int((~won).sum()), R_marg=rpm.mean(0).tolist(),
            R_true=rpm[won].mean(0).tolist(), R_false=rpm[~won].mean(0).tolist())
    return {"config": {"N": N, "m": m, "m_exponent": 0.6, "B": B, "K": K, "p1": P1, "seed": SEED},
            "scenarios": scenarios}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    draws = run_gamma09()
    draws.to_csv(OUT / "gamma09_draws.csv", index=False)
    print(draws.groupby("scenario")[["raw", "adjusted", "selected_true", "p_model2_rank1"]].mean(), flush=True)
    (OUT / "gamma06_summary.json").write_text(json.dumps(run_gamma06(), indent=2) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
