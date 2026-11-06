"""Pairwise comparisons and the leader-change count used on the home page."""
import json

import numpy as np

from llm_wc.tracker import history
from llm_wc.tracker.matrix import Matrix
from llm_wc.tracker.pairwise import pairwise


def test_pairwise_is_complementary_and_orders_clear_gaps():
    rng = np.random.default_rng(0)
    obs = np.column_stack([rng.random(200) < p for p in (0.9, 0.5, 0.5, 0.1)]).astype(float)
    m = Matrix("toy", obs, np.array(list("abcd")), np.array(list("ABCD")), np.arange(200).astype(str),
               np.repeat("all", 200), "none")
    out = pairwise(m, {"gamma": 0.9, "n_bootstraps": 400, "seed": 1})
    p = np.array(out["p_above"])
    assert out["model_ids"][0] == "a" and out["model_ids"][-1] == "d"
    assert np.allclose(p + p.T, 1, atol=2e-4)
    assert p[0, -1] > 0.99


def test_leader_changes_classifies_unresolved_resolved_and_departed(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HISTORY", tmp_path)
    snaps = {
        "20250101_a": {"date": "2025-01-01", "leaders": ["x"], "tier1": ["x", "y"], "models": ["x", "y", "z"], "names": {}},
        "20250201_b": {"date": "2025-02-01", "leaders": ["y"], "tier1": ["y", "x"], "models": ["x", "y", "z"], "names": {}},
        "20250301_c": {"date": "2025-03-01", "leaders": ["z"], "tier1": ["z"], "models": ["x", "y", "z"], "names": {}},
        "20250401_d": {"date": "2025-04-01", "leaders": ["w"], "tier1": ["w"], "models": ["w", "x"], "names": {}},
    }
    (tmp_path / "toy_tiers.json").write_text(json.dumps(snaps))
    c = history.leader_changes("toy")
    assert (c["changes"], c["unresolved"], c["departed"]) == (2, 1, 1)
    assert [e["unresolved"] for e in c["events"] if not e["departed"]] == [True, False]
