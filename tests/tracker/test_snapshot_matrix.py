import numpy as np
import pandas as pd
import pytest

from llm_wc.tracker.matrix import build_matrix, max_cells_block
from llm_wc.tracker.snapshot import (
    make_snapshot, read_snapshot, validate, validate_dir, write_snapshot,
)


def _snap(scores, tasks=None, models=None, **kw):
    scores = pd.DataFrame(scores, columns=["task_id", "model_id", "score"])
    tasks = tasks if tasks is not None else pd.DataFrame({"task_id": sorted(scores.task_id.unique())})
    models = models if models is not None else pd.DataFrame(
        {"model_id": sorted(scores.model_id.unique()), "model": sorted(scores.model_id.unique())})
    return make_snapshot("toy", "20260101_abc", scores, tasks, models, **kw)


def test_round_trip_is_exact_and_deterministic(tmp_path):
    snap = _snap([("t1", "a", 0.1), ("t1", "b", 1 / 3), ("t2", "a", 1.0), ("t2", "b", 0.0)])
    out = write_snapshot(snap, tmp_path)
    back = read_snapshot(out)
    pd.testing.assert_frame_equal(back.scores, snap.scores)
    first = (out / "scores.csv.gz").read_bytes()
    write_snapshot(snap, tmp_path)
    assert (out / "scores.csv.gz").read_bytes() == first
    assert validate_dir(out) == []


def test_validate_rejects_bad_snapshots():
    assert validate(_snap([("t1", "a", 1.5)]))
    dup = _snap([("t1", "a", 1.0)])
    dup.scores = pd.concat([dup.scores, dup.scores])
    assert any("duplicate" in p for p in validate(dup))


def test_tampering_is_detected(tmp_path):
    out = write_snapshot(_snap([("t1", "a", 1.0), ("t2", "a", 0.0)]), tmp_path)
    (out / "models.csv").write_text("model_id,model\na,other\n")
    assert validate_dir(out)


def test_fingerprint_ignores_model_metadata():
    base = _snap([("t1", "a", 1.0)])
    meta = _snap([("t1", "a", 1.0)], models=pd.DataFrame({"model_id": ["a"], "model": ["a"], "date": ["2026-01-01"]}))
    assert base.fingerprint() == meta.fingerprint()


def test_complete_rule_keeps_order_and_drops_incomplete():
    tasks = pd.DataFrame({"task_id": ["t2", "t1", "t3"], "domain": ["x", "x", "y"]})
    models = pd.DataFrame({"model_id": ["b", "a", "c"], "model": ["B", "A", "C"]})
    snap = _snap([("t1", "a", 1), ("t2", "a", 0), ("t1", "b", 1), ("t2", "b", 1), ("t1", "c", 1), ("t3", "a", 1)],
                 tasks, models)
    m = build_matrix(snap, "toy", {"pool": {"domain": "x"}, "models": "complete"})
    assert list(m.task_ids) == ["t2", "t1"]
    assert list(m.model_ids) == ["b", "a"]
    assert list(m.model_names) == ["B", "A"]
    assert m.obs.tolist() == [[1, 0], [1, 1]]
    assert [x["model_id"] for x in m.excluded] == ["c"]


def test_max_cells_block_prefers_more_cells_then_tasks():
    cover = {"a": {1, 2, 3, 4}, "b": {1, 2, 3, 4}, "c": {1, 2}, "d": {1, 2}}
    models, tasks, cutoff = max_cells_block(cover)
    assert (sorted(models), tasks, cutoff) == (["a", "b"], {1, 2, 3, 4}, 4)
    cover["e"] = {1, 2}
    models, tasks, _ = max_cells_block(cover)
    assert sorted(models) == ["a", "b", "c", "d", "e"] and tasks == {1, 2}


def test_max_cells_rule_then_task_filter():
    tasks = pd.DataFrame({"task_id": ["t1", "t2", "t3"], "category": ["x", "y", "x"]})
    snap = _snap([(t, m, 1.0) for t in ("t1", "t2", "t3") for m in ("a", "b")] + [("t1", "c", 0.0)], tasks)
    m = build_matrix(snap, "toy", {"models": "max_cells", "tasks": {"category": "x"}})
    assert list(m.model_ids) == ["a", "b"] and list(m.task_ids) == ["t1", "t3"]


def test_unknown_rule_raises():
    with pytest.raises(ValueError):
        build_matrix(_snap([("t1", "a", 1.0)]), "toy", {"models": "nope"})


def test_strata_and_clusters_come_from_task_columns():
    tasks = pd.DataFrame({"task_id": ["t1", "t2"], "repository": ["r1", "r2"]})
    m = build_matrix(_snap([("t1", "a", 1), ("t2", "a", 0)], tasks), "toy",
                     {"strata": "repository", "stratum_type": "repository", "clusters": "repository"})
    assert np.array_equal(m.strata, ["r1", "r2"]) and np.array_equal(m.clusters, ["r1", "r2"])
