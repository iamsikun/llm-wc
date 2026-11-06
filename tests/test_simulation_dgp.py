from __future__ import annotations

import numpy as np

from llm_wc.simulation.dgp import simulate_bernoulli_logit_fe


def test_bernoulli_logit_fe_shape_and_marginals() -> None:
    """simulate_bernoulli_logit_fe returns a binary (n_tasks, J) matrix with
    column means close to E_T[sigmoid(alpha_j + T)] under T ~ N(0, sigma_t^2)."""
    rng = np.random.default_rng(2026)
    alpha = np.array([-2.0, 0.0, 1.5], dtype=float)
    sigma_t = 1.2
    n_tasks = 8000

    obs = simulate_bernoulli_logit_fe(
        alpha=alpha,
        sigma_t=sigma_t,
        n_tasks=n_tasks,
        rng=rng,
    )
    assert obs.shape == (n_tasks, len(alpha))
    assert set(np.unique(obs).tolist()) <= {0.0, 1.0}

    # Reference marginals via independent large MC over T.
    rng_ref = np.random.default_rng(99)
    t_ref = rng_ref.normal(0.0, sigma_t, size=200_000)
    p_ref = 1.0 / (1.0 + np.exp(-(alpha[None, :] + t_ref[:, None])))
    expected = p_ref.mean(axis=0)
    np.testing.assert_allclose(obs.mean(axis=0), expected, atol=0.02)
