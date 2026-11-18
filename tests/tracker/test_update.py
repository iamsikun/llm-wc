"""Criterion B4: updates are idempotent, compare with the reference input, and isolate failures."""
import pandas as pd
import pytest

from llm_wc.tracker import update as upd
from llm_wc.tracker.snapshot import list_snapshots, make_snapshot
from llm_wc.tracker.sources import SOURCES, Context, Revision, Source, SourceUnavailable


class Fake(Source):
    def __init__(self, key, revision="r1", score=1.0, reference_revision=None, error=None):
        self.key, self.revision, self.score, self.reference_revision, self.error = key, revision, score, reference_revision, error
        self.fetches = 0

    def latest_revision(self, ctx):
        if self.error:
            raise self.error
        return Revision(self.revision, "2026-10-01T00:00:00Z")

    def _snap(self, revision, sid):
        scores = pd.DataFrame({"task_id": ["t1", "t2"], "model_id": ["a", "a"], "score": [self.score, 0.0]})
        return make_snapshot(self.key, sid, scores, pd.DataFrame({"task_id": ["t1", "t2"]}),
                             pd.DataFrame({"model_id": ["a"], "model": ["A"]}), {"revision": revision})

    def fetch(self, ctx, revision):
        self.fetches += 1
        return self._snap(revision.revision, f"20261001_{revision.revision}")

    def reference_snapshot(self):
        return self._snap(self.reference_revision, "reference") if self.reference_revision else None


@pytest.fixture
def fakes(monkeypatch):
    added = {}

    def add(src):
        monkeypatch.setitem(SOURCES, src.key, src)
        added[src.key] = src
        return src
    return add


def _run(tmp_path, keys):
    return upd.update(keys, Context(cache=tmp_path / "cache"), data_root=tmp_path / "data",
                      results_root=tmp_path / "results", status_path=tmp_path / "status.json")


def test_second_run_with_unchanged_upstream_writes_nothing(tmp_path, fakes):
    src = fakes(Fake("fake_a"))
    assert _run(tmp_path, ["fake_a"])["fake_a"]["result"] == "updated"
    before = sorted(p.name for p in (tmp_path / "data").rglob("*"))
    assert _run(tmp_path, ["fake_a"])["fake_a"]["result"] == "unchanged"
    assert sorted(p.name for p in (tmp_path / "data").rglob("*")) == before
    assert src.fetches == 1


def test_new_revision_with_same_content_is_not_stored(tmp_path, fakes):
    src = fakes(Fake("fake_b"))
    _run(tmp_path, ["fake_b"])
    src.revision = "r2"
    assert _run(tmp_path, ["fake_b"])["fake_b"]["result"] == "unchanged"
    assert len(list_snapshots("fake_b", tmp_path / "data")) == 1
    src.score = 0.5
    src.revision = "r3"
    assert _run(tmp_path, ["fake_b"])["fake_b"]["result"] == "updated"
    assert len(list_snapshots("fake_b", tmp_path / "data")) == 2


def test_upstream_at_reference_revision_is_not_refetched(tmp_path, fakes):
    src = fakes(Fake("fake_c", revision="p1", reference_revision="p1"))
    assert _run(tmp_path, ["fake_c"])["fake_c"]["result"] == "same_as_reference"
    assert src.fetches == 0 and not list_snapshots("fake_c", tmp_path / "data")


def test_one_failing_source_does_not_stop_the_others(tmp_path, fakes):
    fakes(Fake("fake_bad", error=RuntimeError("network down")))
    fakes(Fake("fake_secret", error=SourceUnavailable("needs TOKEN")))
    fakes(Fake("fake_ok"))
    status = _run(tmp_path, ["fake_bad", "fake_secret", "fake_ok"])
    assert status["fake_bad"]["result"] == "failed" and "network down" in status["fake_bad"]["reason"]
    assert status["fake_secret"] == {**status["fake_secret"], "result": "unavailable", "reason": "needs TOKEN"}
    assert status["fake_ok"]["result"] == "updated"


def test_retired_source_is_not_checked_and_keeps_its_data(tmp_path, fakes, monkeypatch):
    src = fakes(Fake("fake_r"))
    first = _run(tmp_path, ["fake_r"])["fake_r"]
    monkeypatch.setattr(upd.registry, "registry", lambda: {"sources": {"fake_r": {"retired": "board removed"}}})
    src.error = RuntimeError("must not be called")
    again = _run(tmp_path, ["fake_r"])["fake_r"]
    assert again["result"] == "retired" and again["reason"] == "board removed"
    assert again["snapshot_id"] == first["snapshot_id"] and again["checked_at"] == first["checked_at"]
    assert src.fetches == 1


def test_failure_keeps_last_good_snapshot(tmp_path, fakes):
    src = fakes(Fake("fake_d"))
    first = _run(tmp_path, ["fake_d"])["fake_d"]
    src.error = RuntimeError("boom")
    again = _run(tmp_path, ["fake_d"])["fake_d"]
    assert again["result"] == "failed" and again["snapshot_id"] == first["snapshot_id"]
    assert again["last_success"] == first["last_success"]
    assert len(list_snapshots("fake_d", tmp_path / "data")) == 1
