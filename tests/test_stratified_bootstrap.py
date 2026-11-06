"""Unit tests for the stratified task bootstrap in wc/core.py."""

from __future__ import annotations

import numpy as np
import pytest

from llm_wc.wc.core import (
    BootstrapResult,
    bootstrap_correction,
    _stratified_bootstrap_estimates,
)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


def _make_stratified_data(rng, repo_sizes=(50, 30, 20), J=5, cluster_effect=0.3):
    """Generate binary data with repo structure."""
    N = sum(repo_sizes)
    cluster_labels = np.repeat(np.arange(len(repo_sizes)), repo_sizes)
    abilities = np.linspace(0.3, 0.7, J)
    obs = np.zeros((N, J), dtype=float)
    for c, n_c in enumerate(repo_sizes):
        mask = cluster_labels == c
        shift = (c - len(repo_sizes) / 2) * cluster_effect
        probs = np.clip(abilities + shift, 0.05, 0.95)
        obs[mask] = (rng.rand(n_c, J) < probs[None, :]).astype(float)
    return obs, cluster_labels


class TestStratifiedBootstrapShape:
    def test_basic_shape(self, rng):
        obs, labels = _make_stratified_data(rng)
        r = bootstrap_correction(
            obs=obs, n_bootstraps=500, cluster_labels=labels,
            bootstrap_method="stratified", compute_partial_order=False,
        )
        assert r.boot_wc.shape == (5,)
        assert r.corr_rank_est.shape == (5,)
        assert r.bootstrap_method == "stratified"
        assert r.n_clusters == 3

    def test_full_pipeline_with_tiers(self, rng):
        obs, labels = _make_stratified_data(rng)
        r = bootstrap_correction(
            obs=obs, n_bootstraps=2000, cluster_labels=labels,
            bootstrap_method="stratified",
        )
        assert r.tiers is not None
        assert r.dominance is not None
        assert r.gap_se is not None
        assert r.max_t_crit > 0


class TestStratifiedCompositionPreservation:
    def test_mean_preservation(self, rng):
        """E[boot_est] should be close to emp_est."""
        obs, labels = _make_stratified_data(rng, repo_sizes=(60, 40, 50), J=4)
        emp_est = obs.mean(axis=0)
        np.random.seed(42)
        boot_est = _stratified_bootstrap_estimates(obs, labels, 10000)
        np.testing.assert_allclose(boot_est.mean(axis=0), emp_est, atol=0.01)

    def test_reduces_to_iid_single_cluster(self, rng):
        """With one repo, stratified should behave like IID bootstrap."""
        N, J = 200, 4
        obs = (rng.rand(N, J) > 0.5).astype(float)
        # All tasks in one repo — but bootstrap_correction requires >=2 clusters,
        # so test the internal function directly
        labels = np.zeros(N, dtype=int)
        np.random.seed(42)
        boot_strat = _stratified_bootstrap_estimates(obs, labels, 5000)
        np.random.seed(42)
        from llm_wc.wc.core import _iid_bootstrap_estimates
        boot_iid = _iid_bootstrap_estimates(obs, 5000, N)
        # Same seed, same resampling → should be identical
        # (single cluster means we resample N from N, same as IID)
        np.testing.assert_array_equal(boot_strat, boot_iid)


class TestStratifiedErrors:
    def test_requires_cluster_labels(self):
        obs = np.random.rand(50, 3)
        with pytest.raises(ValueError, match="requires cluster_labels"):
            bootstrap_correction(
                obs=obs, n_bootstraps=100, bootstrap_method="stratified",
            )

    def test_single_cluster_error(self, rng):
        obs = rng.rand(50, 3)
        labels = np.zeros(50, dtype=int)
        with pytest.raises(ValueError, match="at least 2 clusters"):
            bootstrap_correction(
                obs=obs, n_bootstraps=100, cluster_labels=labels,
                bootstrap_method="stratified",
            )
