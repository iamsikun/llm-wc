"""Tests for cluster bootstrap and wild cluster bootstrap methods in wc/core.py."""

from __future__ import annotations

import numpy as np
import pytest

from llm_wc.wc.core import bootstrap_correction, BootstrapResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.RandomState(42)


def _make_clustered_data(rng, N=500, J=10, C=25, cluster_effect=0.15):
    """Generate clustered binary data with within-cluster correlation."""
    tasks_per_cluster = N // C
    cluster_labels = np.repeat(np.arange(C), tasks_per_cluster)
    N_actual = len(cluster_labels)
    # True model abilities
    abilities = np.linspace(0.3, 0.7, J)
    obs = np.zeros((N_actual, J), dtype=float)
    for c in range(C):
        mask = cluster_labels == c
        n_c = mask.sum()
        # Shared cluster effect per model
        shared = rng.randn(J) * cluster_effect
        probs = np.clip(abilities + shared, 0.05, 0.95)
        obs[mask] = (rng.rand(n_c, J) < probs[None, :]).astype(float)
    return obs, cluster_labels


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_no_cluster_args_same_as_before(self, rng):
        """Without cluster args, output should be identical to old behavior."""
        np.random.seed(123)
        obs = (np.random.rand(200, 5) > 0.5).astype(float)

        np.random.seed(999)
        r1 = bootstrap_correction(obs=obs, n_bootstraps=500, compute_partial_order=False)

        np.random.seed(999)
        r2 = bootstrap_correction(obs=obs, n_bootstraps=500, compute_partial_order=False)

        np.testing.assert_array_equal(r1.boot_wc, r2.boot_wc)
        np.testing.assert_array_equal(r1.corr_rank_est, r2.corr_rank_est)
        assert r1.bootstrap_method == "iid"
        assert r1.n_clusters is None
        assert r1.boot_cluster_size is None
        assert r1.wild_weight_dist is None


# ---------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------

class TestClusterBootstrap:
    def test_shape_and_range(self, rng):
        obs, labels = _make_clustered_data(rng, N=500, J=8, C=25)
        r = bootstrap_correction(
            obs=obs, n_bootstraps=1000, cluster_labels=labels,
            bootstrap_method="cluster", compute_partial_order=False,
        )
        assert r.boot_wc.shape == (8,)
        assert r.corr_rank_est.shape == (8,)
        assert r.bootstrap_method == "cluster"
        assert r.n_clusters == 25
        assert r.boot_cluster_size == 25  # default m=C for 20<=C<50

    def test_boot_size_override(self, rng):
        obs, labels = _make_clustered_data(rng, N=500, J=5, C=25)
        r = bootstrap_correction(
            obs=obs, n_bootstraps=500, cluster_labels=labels,
            bootstrap_method="cluster", boot_size=20,
            compute_partial_order=False,
        )
        assert r.boot_cluster_size == 20

    def test_full_pipeline(self, rng):
        """Cluster bootstrap with partial order, tiers, etc."""
        obs, labels = _make_clustered_data(rng, N=500, J=6, C=25)
        r = bootstrap_correction(
            obs=obs, n_bootstraps=2000, cluster_labels=labels,
            bootstrap_method="cluster",
        )
        assert r.tiers is not None
        assert r.dominance is not None
        assert r.gap_se is not None
        assert r.max_t_crit > 0
        assert len(r.tiers) >= 1
        # rank bounds should be valid
        for j in range(6):
            assert 1 <= r.rank_lower[j] <= r.rank_upper[j] <= 6


# ---------------------------------------------------------------------------
# Wild cluster bootstrap
# ---------------------------------------------------------------------------

class TestWildClusterBootstrap:
    def test_shape(self, rng):
        obs, labels = _make_clustered_data(rng, N=100, J=5, C=10)
        r = bootstrap_correction(
            obs=obs, n_bootstraps=1000, cluster_labels=labels,
            bootstrap_method="wild_cluster", compute_partial_order=False,
        )
        assert r.boot_wc.shape == (5,)
        assert r.bootstrap_method == "wild_cluster"
        assert r.n_clusters == 10
        assert r.wild_weight_dist == "webb"

    @pytest.mark.parametrize("dist", ["rademacher", "mammen", "webb"])
    def test_weight_distributions(self, rng, dist):
        obs, labels = _make_clustered_data(rng, N=100, J=4, C=10)
        r = bootstrap_correction(
            obs=obs, n_bootstraps=2000, cluster_labels=labels,
            bootstrap_method="wild_cluster", wild_weight_dist=dist,
            compute_partial_order=False,
        )
        assert r.wild_weight_dist == dist
        assert r.corr_rank_est.shape == (4,)

    def test_mean_preservation(self, rng):
        """Wild bootstrap should have E[boot_est] ≈ emp_est (since E[w]=0)."""
        obs, labels = _make_clustered_data(rng, N=200, J=4, C=10)
        from llm_wc.wc.core import _wild_cluster_bootstrap_estimates
        emp_est = obs.mean(axis=0)
        np.random.seed(42)
        boot_est = _wild_cluster_bootstrap_estimates(
            obs, labels, 10000, "rademacher", emp_est
        )
        np.testing.assert_allclose(boot_est.mean(axis=0), emp_est, atol=0.01)

    def test_full_pipeline(self, rng):
        obs, labels = _make_clustered_data(rng, N=100, J=5, C=10)
        r = bootstrap_correction(
            obs=obs, n_bootstraps=2000, cluster_labels=labels,
            bootstrap_method="wild_cluster",
        )
        assert r.tiers is not None
        assert r.dominance is not None


# ---------------------------------------------------------------------------
# Auto-selection
# ---------------------------------------------------------------------------

class TestAutoSelection:
    def test_auto_small_c(self, rng):
        """C < 20 → wild_cluster."""
        obs, labels = _make_clustered_data(rng, N=100, J=4, C=10)
        r = bootstrap_correction(
            obs=obs, n_bootstraps=500, cluster_labels=labels,
            compute_partial_order=False,
        )
        assert r.bootstrap_method == "wild_cluster"

    def test_auto_medium_c(self, rng):
        """20 <= C < 50 → cluster with m=C."""
        obs, labels = _make_clustered_data(rng, N=600, J=4, C=30)
        r = bootstrap_correction(
            obs=obs, n_bootstraps=500, cluster_labels=labels,
            compute_partial_order=False,
        )
        assert r.bootstrap_method == "cluster"
        assert r.boot_cluster_size == 30

    def test_auto_large_c(self, rng):
        """C >= 50 → cluster with m=floor(C^0.95)."""
        obs, labels = _make_clustered_data(rng, N=1000, J=4, C=100)
        r = bootstrap_correction(
            obs=obs, n_bootstraps=500, cluster_labels=labels,
            compute_partial_order=False,
        )
        assert r.bootstrap_method == "cluster"
        assert r.boot_cluster_size == int(100 ** 0.95)


# ---------------------------------------------------------------------------
# Cluster-robust SE
# ---------------------------------------------------------------------------

class TestClusterRobustSE:
    def test_larger_than_naive_for_correlated_data(self, rng):
        """Cluster-robust SE should be >= naive SE for positively correlated clusters."""
        obs, labels = _make_clustered_data(rng, N=500, J=5, C=25, cluster_effect=0.3)
        from llm_wc.wc.core import _compute_gap_se
        se_naive = _compute_gap_se(obs, None, 1e-12)
        se_robust = _compute_gap_se(obs, labels, 1e-12)
        # On average, cluster-robust SE should be larger
        mask = ~np.isnan(se_naive)
        assert np.nanmean(se_robust[mask]) > np.nanmean(se_naive[mask])

    def test_one_task_per_cluster_matches_naive(self, rng):
        """When each task is its own cluster, cluster-robust SE ≈ naive SE."""
        N, J = 200, 4
        obs = (rng.rand(N, J) > 0.5).astype(float)
        labels = np.arange(N)  # each task is its own cluster
        from llm_wc.wc.core import _compute_gap_se
        se_naive = _compute_gap_se(obs, None, 1e-12)
        se_robust = _compute_gap_se(obs, labels, 1e-12)
        # Should be close (not exactly equal due to ddof differences)
        mask = ~np.isnan(se_naive)
        np.testing.assert_allclose(
            se_robust[mask], se_naive[mask], rtol=0.15
        )


# ---------------------------------------------------------------------------
# Edge cases and errors
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_cluster_error(self, rng):
        obs = rng.rand(50, 3)
        labels = np.zeros(50, dtype=int)  # all in one cluster
        with pytest.raises(ValueError, match="at least 2 clusters"):
            bootstrap_correction(
                obs=obs, n_bootstraps=100, cluster_labels=labels,
            )

    def test_label_length_mismatch(self, rng):
        obs = rng.rand(50, 3)
        labels = np.arange(40)
        with pytest.raises(ValueError, match="cluster_labels.*must have shape"):
            bootstrap_correction(
                obs=obs, n_bootstraps=100, cluster_labels=labels,
            )

    def test_method_without_labels(self):
        obs = np.random.rand(50, 3)
        with pytest.raises(ValueError, match="requires cluster_labels"):
            bootstrap_correction(
                obs=obs, n_bootstraps=100, bootstrap_method="cluster",
            )

    def test_iid_with_labels(self, rng):
        obs, labels = _make_clustered_data(rng, N=100, J=3, C=10)
        with pytest.raises(ValueError, match="Cannot use.*iid.*with cluster_labels"):
            bootstrap_correction(
                obs=obs, n_bootstraps=100, cluster_labels=labels,
                bootstrap_method="iid",
            )

    def test_unknown_method(self):
        obs = np.random.rand(50, 3)
        with pytest.raises(ValueError, match="requires cluster_labels"):
            bootstrap_correction(
                obs=obs, n_bootstraps=100, bootstrap_method="foobar",
            )

    def test_unknown_weight_dist(self, rng):
        obs, labels = _make_clustered_data(rng, N=100, J=3, C=10)
        with pytest.raises(ValueError, match="Unknown wild_weight_dist"):
            bootstrap_correction(
                obs=obs, n_bootstraps=100, cluster_labels=labels,
                bootstrap_method="wild_cluster", wild_weight_dist="bad",
            )

    def test_moon_requires_m_lt_n(self):
        obs = np.random.rand(50, 3)
        with pytest.raises(ValueError, match="m-out-of-n.*boot_size.*< sample_size"):
            bootstrap_correction(
                obs=obs, n_bootstraps=100, bootstrap_method="moon", boot_size=50,
            )

    def test_moon_with_cluster_labels_error(self, rng):
        obs, labels = _make_clustered_data(rng, N=100, J=3, C=10)
        with pytest.raises(ValueError, match="Cannot use.*moon.*with cluster_labels"):
            bootstrap_correction(
                obs=obs, n_bootstraps=100, cluster_labels=labels,
                bootstrap_method="moon",
            )

    def test_c_equals_n(self, rng):
        """Each task is its own cluster → cluster bootstrap ≈ i.i.d."""
        N, J = 50, 3
        obs = (rng.rand(N, J) > 0.5).astype(float)
        labels = np.arange(N)
        r = bootstrap_correction(
            obs=obs, n_bootstraps=500, cluster_labels=labels,
            bootstrap_method="cluster", compute_partial_order=False,
        )
        assert r.bootstrap_method == "cluster"
        assert r.n_clusters == N


# ---------------------------------------------------------------------------
# Tied models
# ---------------------------------------------------------------------------

class TestTiedModels:
    def test_tied_models_use_position_level_correction(self):
        """Models with identical empirical scores receive position-level corrections.

        The winner's-curse correction is defined per rank position, not per model.
        When two models share an empirical score, numpy's stable sort places them
        at adjacent rank positions in original-index order, and each position
        carries its own selection-bias estimate. The two corrected scores are
        therefore expected to differ: the earlier rank position absorbs a larger
        selection bias than the later one on average, so the higher-position
        (more conservative) corrected score is lower.
        """
        rng = np.random.RandomState(99)
        N, J = 300, 6
        obs = (rng.rand(N, J) > 0.5).astype(float)
        # Force exact tie: copy column 0 into column 1
        obs[:, 1] = obs[:, 0]

        result = bootstrap_correction(obs=obs, n_bootstraps=2000, compute_partial_order=False)

        emp_est = obs.mean(axis=0)
        tied_val = emp_est[0]
        tied_mask = emp_est[result.emp_rank] == tied_val
        # Two columns constructed to tie must land in the tied group.
        assert tied_mask.sum() == 2
        tied_emp = result.emp_rank_est[tied_mask]
        np.testing.assert_array_equal(tied_emp, np.full(2, tied_val))
        # Position-level corrections are not averaged; the earlier-position
        # score is corrected more aggressively.
        corrected_tied = result.corr_rank_est[tied_mask]
        assert corrected_tied[0] <= corrected_tied[1] + 1e-9


# ---------------------------------------------------------------------------
# m-out-of-n bootstrap
# ---------------------------------------------------------------------------

class TestMoonBootstrap:
    def test_shape_and_metadata(self, rng):
        N, J = 200, 5
        obs = (rng.rand(N, J) > 0.5).astype(float)
        m = 50
        r = bootstrap_correction(
            obs=obs, n_bootstraps=1000, bootstrap_method="moon",
            boot_size=m, compute_partial_order=False,
        )
        assert r.boot_wc.shape == (J,)
        assert r.corr_rank_est.shape == (J,)
        assert r.bootstrap_method == "moon"
        assert r.boot_m == m
        assert r.n_clusters is None

    def test_missing_boot_size_error(self):
        """moon requires explicit boot_size."""
        obs = np.random.rand(100, 4)
        with pytest.raises(ValueError, match="requires an explicit.*boot_size"):
            bootstrap_correction(
                obs=obs, n_bootstraps=500, bootstrap_method="moon",
                compute_partial_order=False,
            )

    def test_bias_rescaling(self, rng):
        """moon bias correction should be smaller than iid (boot_wc rescaled by √(m/N))."""
        N, J = 400, 10
        np.random.seed(42)
        # True abilities with clear gaps
        abilities = np.linspace(0.3, 0.7, J)
        obs = (rng.rand(N, J) < abilities[None, :]).astype(float)

        np.random.seed(99)
        r_iid = bootstrap_correction(
            obs=obs, n_bootstraps=5000, bootstrap_method="iid",
            compute_partial_order=False,
        )

        m = 100
        np.random.seed(99)
        r_moon = bootstrap_correction(
            obs=obs, n_bootstraps=5000, bootstrap_method="moon",
            boot_size=m, compute_partial_order=False,
        )

        # moon WC bias (after rescaling) should be on the same order as iid,
        # not inflated by √(N/m)
        ratio = np.mean(np.abs(r_moon.boot_wc)) / np.mean(np.abs(r_iid.boot_wc))
        assert 0.3 < ratio < 3.0, f"Bias ratio moon/iid = {ratio:.2f}, expected ~1"

    def test_full_pipeline(self, rng):
        """moon with partial order, tiers, etc."""
        N, J = 200, 6
        obs = (rng.rand(N, J) > 0.5).astype(float)
        r = bootstrap_correction(
            obs=obs, n_bootstraps=2000, bootstrap_method="moon",
            boot_size=80,
        )
        assert r.tiers is not None
        assert r.dominance is not None
        assert r.gap_se is not None
        assert r.max_t_crit > 0
        for j in range(J):
            assert 1 <= r.rank_lower[j] <= r.rank_upper[j] <= J


# ---------------------------------------------------------------------------
# Stratified m-out-of-n bootstrap
# ---------------------------------------------------------------------------

class TestStratifiedMoonBootstrap:
    def test_shape_and_metadata(self, rng):
        obs, labels = _make_clustered_data(rng, N=500, J=8, C=25)
        r = bootstrap_correction(
            obs=obs, n_bootstraps=1000, cluster_labels=labels,
            bootstrap_method="stratified_moon", boot_size=200,
            compute_partial_order=False,
        )
        assert r.boot_wc.shape == (8,)
        assert r.corr_rank_est.shape == (8,)
        assert r.bootstrap_method == "stratified_moon"
        assert r.n_clusters == 25
        assert r.boot_m is not None
        assert r.boot_m < 500
        assert r.boot_m >= 25

    def test_missing_boot_size_error(self, rng):
        obs, labels = _make_clustered_data(rng, N=500, J=4, C=25)
        with pytest.raises(ValueError, match="requires an explicit.*boot_size"):
            bootstrap_correction(
                obs=obs, n_bootstraps=100, cluster_labels=labels,
                bootstrap_method="stratified_moon",
            )

    def test_boot_size_too_large_error(self, rng):
        obs, labels = _make_clustered_data(rng, N=500, J=4, C=25)
        with pytest.raises(ValueError, match="< sample_size"):
            bootstrap_correction(
                obs=obs, n_bootstraps=100, cluster_labels=labels,
                bootstrap_method="stratified_moon", boot_size=500,
            )

    def test_boot_size_too_small_error(self, rng):
        obs, labels = _make_clustered_data(rng, N=500, J=4, C=25)
        with pytest.raises(ValueError, match=">=.*number of clusters"):
            bootstrap_correction(
                obs=obs, n_bootstraps=100, cluster_labels=labels,
                bootstrap_method="stratified_moon", boot_size=10,
            )

    def test_requires_cluster_labels(self):
        obs = np.random.rand(100, 4)
        with pytest.raises(ValueError, match="requires cluster_labels"):
            bootstrap_correction(
                obs=obs, n_bootstraps=100, bootstrap_method="stratified_moon",
                boot_size=50,
            )

    def test_bias_rescaling(self, rng):
        """stratified_moon WC should be on same order as stratified."""
        obs, labels = _make_clustered_data(rng, N=500, J=10, C=25)

        np.random.seed(99)
        r_strat = bootstrap_correction(
            obs=obs, n_bootstraps=5000, cluster_labels=labels,
            bootstrap_method="stratified", compute_partial_order=False,
        )

        np.random.seed(99)
        r_sm = bootstrap_correction(
            obs=obs, n_bootstraps=5000, cluster_labels=labels,
            bootstrap_method="stratified_moon", boot_size=200,
            compute_partial_order=False,
        )

        ratio = np.mean(np.abs(r_sm.boot_wc)) / max(np.mean(np.abs(r_strat.boot_wc)), 1e-10)
        assert 0.2 < ratio < 5.0, f"Bias ratio stratified_moon/stratified = {ratio:.2f}"

    def test_full_pipeline(self, rng):
        obs, labels = _make_clustered_data(rng, N=500, J=6, C=25)
        r = bootstrap_correction(
            obs=obs, n_bootstraps=2000, cluster_labels=labels,
            bootstrap_method="stratified_moon", boot_size=200,
        )
        assert r.tiers is not None
        assert r.dominance is not None
        assert r.gap_se is not None
        assert r.max_t_crit > 0
        for j in range(6):
            assert 1 <= r.rank_lower[j] <= r.rank_upper[j] <= 6

    def test_proportional_allocation(self, rng):
        """Unequal cluster sizes → actual_m close to requested."""
        labels = np.concatenate([
            np.full(100, 0), np.full(50, 1), np.full(50, 2),
            np.full(25, 3), np.full(25, 4),
        ])
        N = 250
        J = 4
        obs = rng.rand(N, J)
        r = bootstrap_correction(
            obs=obs, n_bootstraps=500, cluster_labels=labels,
            bootstrap_method="stratified_moon", boot_size=125,
            compute_partial_order=False,
        )
        assert r.boot_m is not None
        assert abs(r.boot_m - 125) <= 5
