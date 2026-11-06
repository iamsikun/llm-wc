from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path
from typing import Any


def _load_history_parser_module() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "parsers" / "parse_swebench_history_trajs.py"
    spec = importlib.util.spec_from_file_location("parse_swebench_history_trajs", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


history = _load_history_parser_module()


def test_supported_leaderboards_include_expected_six_values() -> None:
    assert history.SUPPORTED_LEADERBOARDS == (
        "verified",
        "bash-only",
        "multimodal",
        "lite",
        "multilingual",
        "test",
    )


def _commit(sha: str, iso_dt: str, message: str = "") -> history.GitHubCommit:
    return history.GitHubCommit(
        sha=sha,
        committed_at=history._parse_iso8601_utc(iso_dt),
        message=message,
        html_url=None,
    )


def test_fetch_commit_items_for_path_paginates_until_no_next(monkeypatch) -> None:
    first_url = (
        "https://api.github.com/repos/example/repo/commits"
        "?path=data/leaderboards.json&per_page=100"
    )
    second_url = "https://api.github.com/repositories/123/commits?page=2"
    calls: list[str] = []

    page_1 = [{"sha": "newer"}, {"sha": "middle"}]
    page_2 = [{"sha": "older"}]

    def fake_read_json_url(url: str, **_: Any) -> tuple[Any, dict[str, str]]:
        calls.append(url)
        if url == first_url:
            return (
                page_1,
                {"Link": f'<{second_url}>; rel="next", <{second_url}>; rel="last"'},
            )
        if url == second_url:
            return page_2, {}
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(history, "_read_json_url", fake_read_json_url)
    items = history._fetch_commit_items_for_path(
        "example/repo",
        "data/leaderboards.json",
        token=None,
        cache_dir=None,
        timeout=10,
    )

    assert calls == [first_url, second_url]
    assert items == [*page_1, *page_2]


def test_align_experiments_commit_uses_nearest_prior_or_oldest_fallback() -> None:
    experiments = [
        _commit("a", "2025-01-01T00:00:00Z"),
        _commit("b", "2025-01-10T00:00:00Z"),
        _commit("c", "2025-01-20T00:00:00Z"),
    ]

    selected, used_fallback = history._align_experiments_commit(
        _commit("lb", "2025-01-12T00:00:00Z"),
        experiments,
    )
    assert selected.sha == "b"
    assert used_fallback is False

    selected, used_fallback = history._align_experiments_commit(
        _commit("lb-early", "2024-12-25T00:00:00Z"),
        experiments,
    )
    assert selected.sha == "a"
    assert used_fallback is True


def test_filter_commits_applies_date_range_then_max_commits_from_newest() -> None:
    commits = [
        _commit("c1", "2025-01-01T00:00:00Z"),
        _commit("c2", "2025-01-02T00:00:00Z"),
        _commit("c3", "2025-01-03T00:00:00Z"),
        _commit("c4", "2025-01-04T00:00:00Z"),
        _commit("c5", "2025-01-05T00:00:00Z"),
    ]

    filtered = history._filter_commits(
        commits,
        since_date=date.fromisoformat("2025-01-02"),
        until_date=date.fromisoformat("2025-01-05"),
        max_commits=2,
    )

    assert [commit.sha for commit in filtered] == ["c4", "c5"]


def test_extract_instances_from_entry_prefers_embedded_per_instance_details() -> None:
    entry = {
        "name": "Example Model",
        "folder": "20250101_example",
        "per_instance_details": {
            "task_1": {"resolved": True},
            "task_2": {"resolved": False},
        },
    }
    cache: dict[tuple[str, str, str], tuple[dict[str, int], str | None]] = {}

    instances, traj_url, error = history._extract_instances_from_entry(
        entry,
        model_name="Example Model",
        leaderboard_name="verified",
        experiments_sha="exp-sha",
        experiments_base="https://example.com/experiments",
        instance_cache=cache,
        cache_dir=None,
        timeout=10,
        token=None,
    )

    assert error is None
    assert traj_url == "leaderboard:per_instance_details"
    assert instances == {"task_1": 1, "task_2": 0}


def test_extract_instance_results_supports_record_payload_schema() -> None:
    payload = {
        "results": [
            {"instance_id": "a", "passed": True},
            {"instance_id": "b", "passed": False},
        ]
    }
    assert history.extract_instance_results(payload) == {"a": 1, "b": 0}


def test_build_trajectory_history_rows_has_stable_sorted_columns_and_zero_fill() -> None:
    snap_1 = history.SnapshotResult(
        snapshot_id="s1",
        snapshot_dir=Path("/tmp/s1"),
        leaderboard_name="verified",
        leaderboard_commit=_commit("lb1", "2025-01-01T00:00:00Z", "m1"),
        experiments_commit=_commit("ex1", "2025-01-01T00:00:00Z", "e1"),
        experiments_fallback=False,
        entries=[],
        results=[
            history.TrajectoryResults(
                source=history.TrajectorySource(
                    model_name="Model A",
                    model_id="model-a",
                    leaderboard_name="verified",
                    trajectory_url="https://example.com/a",
                    raw_entry={},
                ),
                instances={"task_1": 1},
            )
        ],
        failures=[],
        status="ok",
        status_message=None,
    )
    snap_2 = history.SnapshotResult(
        snapshot_id="s2",
        snapshot_dir=Path("/tmp/s2"),
        leaderboard_name="verified",
        leaderboard_commit=_commit("lb2", "2025-01-02T00:00:00Z", "m2"),
        experiments_commit=_commit("ex2", "2025-01-02T00:00:00Z", "e2"),
        experiments_fallback=False,
        entries=[],
        results=[
            history.TrajectoryResults(
                source=history.TrajectorySource(
                    model_name="Model B",
                    model_id="model-b",
                    leaderboard_name="verified",
                    trajectory_url="https://example.com/b",
                    raw_entry={},
                ),
                instances={"task_2": 1},
            )
        ],
        failures=[],
        status="ok",
        status_message=None,
    )

    rows, fieldnames = history._build_trajectory_history_rows([snap_1, snap_2])

    assert fieldnames[-2:] == ["task_1", "task_2"]
    row_by_model = {row["model_id"]: row for row in rows}

    assert row_by_model["model-a"]["snapshot_id"] == "s1"
    assert row_by_model["model-a"]["task_1"] == 1
    assert row_by_model["model-a"]["task_2"] == 0

    assert row_by_model["model-b"]["snapshot_id"] == "s2"
    assert row_by_model["model-b"]["task_1"] == 0
    assert row_by_model["model-b"]["task_2"] == 1
