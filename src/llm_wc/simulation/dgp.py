from __future__ import annotations

import numpy as np


def simulate_bernoulli_logit_fe(
    alpha: np.ndarray,
    sigma_t: float,
    n_tasks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Y_ij ~ Bernoulli(sigmoid(alpha_j + T_i)),  T_i ~ N(0, sigma_t^2).

    Lightweight DGP that takes already-fitted parameters: no method-of-moments
    calibration, no link inversion. Use the fitted agent log-odds and the SD
    of fitted task effects from a logistic regression on long-format data.
    """
    if sigma_t < 0:
        raise ValueError(f"`sigma_t` must be non-negative, got {sigma_t}.")
    if n_tasks <= 0:
        raise ValueError(f"`n_tasks` must be positive, got {n_tasks}.")

    j = len(alpha)
    t = rng.normal(loc=0.0, scale=sigma_t, size=n_tasks)
    latent = alpha.reshape(1, j) + t.reshape(-1, 1)
    p = 1.0 / (1.0 + np.exp(-np.clip(latent, -500.0, 500.0)))
    return (rng.random(size=(n_tasks, j)) < p).astype(float)
