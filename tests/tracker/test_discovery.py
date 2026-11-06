"""Criteria C1 and C2: feasibility checks and probing tables."""
import numpy as np
import pandas as pd

from llm_wc.tracker.discovery.criteria import assess
from llm_wc.tracker.discovery.probe import probe, to_long


def _long(n_tasks=40, n_models=4, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = [(f"t{t}", f"m{m}", scale * float(rng.random() < 0.3 + 0.1 * m))
            for t in range(n_tasks) for m in range(n_models)]
    return pd.DataFrame(rows, columns=["task_id", "model_id", "score"])


def test_complete_binary_table_is_feasible():
    report = assess(_long())
    assert report.verdict == "feasible"
    assert (report.block_tasks, report.block_models) == (40, 4)


def test_percent_scores_need_rescaling():
    report = assess(_long(scale=100.0))
    assert report.verdict == "feasible after rescaling" and report.rescale == "x / 100"


def test_too_few_tasks_or_models_is_not_feasible():
    assert assess(_long(n_tasks=10)).verdict == "not feasible"
    assert assess(_long(n_models=2)).verdict == "not feasible"


def test_aggregate_only_table_is_not_feasible():
    frame = pd.DataFrame({"task_id": ["overall"] * 5, "model_id": list("abcde"), "score": [0.1, 0.2, 0.3, 0.4, 0.5]})
    report = assess(frame)
    assert report.verdict == "not feasible"
    assert not next(c for c in report.checks if c.name == "task_level").passed


def test_repeated_pairs_ask_for_review_but_do_not_rule_out():
    frame = pd.concat([_long(), _long().head(5)])
    assert assess(frame).verdict == "feasible after review"


def test_ragged_table_uses_largest_complete_block():
    frame = _long(n_models=6)
    frame = frame[~((frame.model_id == "m5") & (frame.task_id.isin([f"t{i}" for i in range(30)])))]
    report = assess(frame)
    assert (report.block_models, report.block_tasks) == (5, 40)


def test_probe_reads_long_and_wide_files(tmp_path):
    long = _long()
    long.rename(columns={"model_id": "model", "score": "correct"}).to_csv(tmp_path / "long.csv", index=False)
    assert probe(str(tmp_path / "long.csv")).verdict == "feasible"
    wide = long.pivot(index="model_id", columns="task_id", values="score").reset_index()
    wide.to_csv(tmp_path / "wide.csv", index=False)
    report = probe(str(tmp_path / "wide.csv"))
    assert (report.models, report.tasks, report.verdict) == (4, 40, "feasible")


def test_to_long_converts_booleans():
    frame = pd.DataFrame({"question_id": ["q1", "q2"], "model": ["a", "a"], "is_correct": [True, False]})
    assert to_long(frame)["score"].tolist() == [1.0, 0.0]
