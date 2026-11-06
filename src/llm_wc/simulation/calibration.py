"""Logistic DGP calibrated to an observed binary leaderboard, plus the
beta-binomial empirical-Bayes baseline for comparison.

DGP: Y_ij | T_i ~ Bernoulli(sigmoid(alpha_j + T_i)), T_i ~ N(0, sigma_T^2).
``calibrate_logistic`` fits model and task fixed effects, centers the task
effects, and returns the marginal means tau_j = E_T[sigmoid(alpha_j + T)].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import betaln, digamma, expit

from llm_wc.dgp._data import prepare_long_format


def _disambiguate_model_names(model_names: list[str]) -> list[str]:
    """Match prepare_long_format's duplicate-name disambiguation."""
    counts: dict[str, int] = {}
    out: list[str] = []
    for name in model_names:
        counts[name] = counts.get(name, 0) + 1
        out.append(name if counts[name] == 1 else f"{name} (v{counts[name]})")
    return out


def fit_logit_fe(
    obs: np.ndarray, model_names: list[str], task_ids: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str], int, int]:
    """Fit logit P(Y=1) = alpha_j + T_i by binomial GLM; constant tasks are dropped.

    Returns (alpha (J,) in obs column order, task effects (N_used,) with the
    reference task at 0, disambiguated model names, N_used, n_constant).
    """
    df, n_constant, _ = prepare_long_format(
        obs, model_names, task_ids, np.array(["all"] * obs.shape[0])
    )
    result = sm.GLM.from_formula(
        "y ~ 0 + C(agent) + C(task)", data=df, family=sm.families.Binomial()
    ).fit()
    agent_fe: dict[str, float] = {}
    task_fe: dict[str, float] = {}
    for name, coef in zip(result.model.exog_names, result.params):
        for prefix, target in (("C(agent)[", agent_fe), ("C(task)[", task_fe)):
            if name.startswith(prefix):
                level = name[len(prefix):-1]
                target[level[2:] if level.startswith("T.") else level] = float(coef)
    model_order = _disambiguate_model_names(model_names)
    alpha = np.array([agent_fe[m] for m in model_order], dtype=float)
    task_effect = np.array([task_fe.get(t, 0.0) for t in sorted(df["task"].unique())], dtype=float)
    return alpha, task_effect, model_order, df["task"].nunique(), n_constant


@dataclass
class LogisticCalibration:
    alpha: np.ndarray        # (J,) model intercepts, absorbing the mean task effect
    sigma: float             # sd of centered task effects
    taus: np.ndarray         # (J,) marginal means by Gauss-Hermite quadrature
    task_effect: np.ndarray  # (N_used,) centered task effects
    n_tasks: int             # nonconstant tasks used in the fit
    n_constant: int          # constant tasks dropped
    model_order: list[str]


def marginal_means(alpha: np.ndarray, sigma: float, nodes: int = 120) -> np.ndarray:
    """tau_j = E[sigmoid(alpha_j + T)], T ~ N(0, sigma^2), by Gauss-Hermite quadrature."""
    z, w = hermgauss(nodes)
    return (expit(alpha[None, :] + np.sqrt(2) * sigma * z[:, None]) * w[:, None]).sum(0) / np.sqrt(np.pi)


def calibrate_logistic(obs: np.ndarray, model_names: list[str], task_ids: list[str]) -> LogisticCalibration:
    alpha, task_effect, model_order, n_used, n_constant = fit_logit_fe(obs, model_names, task_ids)
    shift = task_effect.mean()
    alpha = alpha + shift
    task_effect = task_effect - shift
    sigma = task_effect.std(ddof=1)
    return LogisticCalibration(alpha, sigma, marginal_means(alpha, sigma), task_effect,
                               n_used, n_constant, model_order)


def beta_binomial_eb(counts: np.ndarray, n: int) -> tuple[float, float]:
    """Marginal-likelihood (a, b) of a Beta(a, b) prior for J binomial counts out of n.

    Optimizes (log a, log b) in [-10, 14]^2 with analytic gradients from a
    moment-based start and from (1, 1); keeps the best successful fit and
    falls back to Powell if neither converges.
    """
    def objective(log_ab):
        a, b = np.exp(log_ab)
        value = -np.sum(betaln(counts + a, n - counts + b) - betaln(a, b))
        da = -np.sum(digamma(counts + a) - digamma(n + a + b) - digamma(a) + digamma(a + b)) * a
        db = -np.sum(digamma(n - counts + b) - digamma(n + a + b) - digamma(b) + digamma(a + b)) * b
        return value, np.array([da, db])

    mean = np.clip((counts / n).mean(), 1e-5, 1 - 1e-5)
    var = max(float((counts / n).var()), 1e-6)
    concentration = max(mean * (1 - mean) / var - 1, .1)
    bounds = [(-10, 14), (-10, 14)]
    fits = [minimize(objective, np.log(x), method="L-BFGS-B", jac=True, bounds=bounds,
                     options={"ftol": 1e-11, "gtol": 1e-6})
            for x in [(mean * concentration, (1 - mean) * concentration), (1., 1.)]]
    good = [f for f in fits if f.success and np.isfinite(f.fun)]
    if not good:
        fallback = minimize(lambda x: objective(x)[0], min(fits, key=lambda x: x.fun).x, method="Powell",
                            bounds=bounds, options={"ftol": 1e-10, "xtol": 1e-8})
        if not fallback.success or not np.isfinite(fallback.fun):
            raise RuntimeError(fallback.message)
        good = [fallback]
    a, b = np.exp(min(good, key=lambda x: x.fun).x)
    return float(a), float(b)
