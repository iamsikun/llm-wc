import numpy as np
import pytest

from llm_wc.benchmarks import load_setting
from llm_wc.wc.ruler import (
    iid_bootstrap_means,
    observed_tie_blocks,
    position_bias_and_rpm,
    rank_intervals_and_tiers,
    ruler,
)


def test_bootstrap_ties_share_rank_mass():
    obs = np.array([[1., 1., 0.], [0., 0., 1.]])
    # Resample 1: three-way tie. Resample 2: models 0 and 1 tie for first.
    _, rpm = position_bias_and_rpm(obs, np.array([[.5, .5, .5], [1, 1, 0.]]), 2)
    expected = np.array([[5 / 12, 5 / 12, 1 / 6], [5 / 12, 5 / 12, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
    np.testing.assert_allclose(rpm, expected)


def test_separated_tiers_recovered():
    rpm = np.zeros((5, 5))
    rpm[:3, :3] = 1 / 3
    rpm[3:, 3:] = 1 / 2
    _, _, ids, tiers = rank_intervals_and_tiers(rpm, np.array([.8, .8, .8, .2, .2]), .1)
    assert ids.tolist() == [1, 1, 1, 2, 2]
    assert [t["size"] for t in tiers] == [3, 2]


def test_quantile_boundary_is_not_spilled_by_roundoff():
    # The CDF of row 0 reaches exactly 0.95 at rank 2.
    rpm = np.array([[.05, .90, .05], [.90, .05, .05], [.05, .05, .90]])
    _, high, _, _ = rank_intervals_and_tiers(rpm, np.array([.9, .8, .1]), .1)
    assert high[0] == 2


def test_incompatible_observed_tie_is_flagged():
    with pytest.raises(ValueError):
        rank_intervals_and_tiers(np.eye(2), np.array([.5, .5]), .1)


def test_observed_tie_blocks():
    start, end = observed_tie_blocks(np.array([.9, .9, .8, .7, .7, .7]))
    assert start.tolist() == [1, 1, 3, 4, 4, 4]
    assert end.tolist() == [2, 2, 3, 6, 6, 6]


def test_iid_bootstrap_matches_global_numpy_stream():
    obs = np.random.default_rng(0).random((40, 4))
    np.random.seed(7)
    idx = np.random.randint(0, 40, size=(300, 12))
    np.testing.assert_allclose(iid_bootstrap_means(obs, 12, 300, seed=7), obs[idx].mean(1))


def test_ruler_invariants():
    rng = np.random.default_rng(1)
    obs = (rng.random((200, 8)) < np.linspace(.4, .6, 8)).astype(float)
    obs[:, 7] = obs[:, 6]  # an exact observed tie
    r = ruler(obs, n_bootstraps=500)
    assert r.m == int(200 ** .9)
    np.testing.assert_allclose(r.rpm.sum(0), 1)
    np.testing.assert_allclose(r.rpm.sum(1), 1)
    assert np.all(np.diff(r.scores) <= 0)
    tied = np.flatnonzero(r.order == 6)[0], np.flatnonzero(r.order == 7)[0]
    assert r.adjustment[tied[0]] == r.adjustment[tied[1]]
    for alpha, (low, high, ids) in r.intervals.items():
        assert all(r.rpm[i, low[i] - 1:high[i]].sum() >= 1 - alpha - 1e-10 for i in range(8))
        for t in r.tiers[alpha]:
            rows = slice(t["start_position"] - 1, t["end_position"])
            assert low[rows].max() <= high[rows].min()
        assert ids[tied[0]] == ids[tied[1]]


def test_load_setting_verified():
    s = load_setting("swebench_verified")
    assert s.obs.shape == (500, 173)
    assert len(set(s.strata)) == 12 and s.clusters is not None
    assert set(np.unique(s.obs)) <= {0.0, 1.0}
