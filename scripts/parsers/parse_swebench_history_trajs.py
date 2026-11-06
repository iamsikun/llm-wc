#!/usr/bin/env python3
"""
Parse SWE-Bench leaderboard history and per-task outcomes from GitHub commit history.

This script:
1. Fetches commit history for `data/leaderboards.json` from `swe-bench.github.io`.
2. Fetches commit history for `evaluation/<leaderboard>` from `SWE-bench/experiments`.
3. For each historical leaderboard commit (oldest -> newest):
   - Loads the leaderboard snapshot at that commit.
   - Aligns it to the nearest prior experiments commit by commit timestamp.
   - Extracts per-model per-instance outcomes via:
     a) embedded `per_instance_details`,
     b) explicit trajectory URL in the entry,
     c) candidate files in the aligned experiments tree.
4. Writes per-snapshot artifacts under:
   - `snapshots/<commit_date>_<shortsha>/index.json`
   - `snapshots/<commit_date>_<shortsha>/leaderboard.csv`
   - `snapshots/<commit_date>_<shortsha>/trajectory.csv`
5. Writes top-level history summaries:
   - `history_index.json`
   - `leaderboard_history.csv`
   - `model_history.csv`
   - `trajectory_history.csv`
   - `failures.csv`
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

# Ensure the src directory is importable (workaround for macOS UF_HIDDEN .pth bug)
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from llm_wc.parsers.common import (
    CANDIDATE_TRAJ_FILES,
    DownloadError,
    JsonDict,
    JsonValue,
    TrajectoryResults,
    TrajectorySource,
    _extract_from_mapping,
    coerce_csv_value,
    csv_loads,
    extract_instance_results,
    extract_model_name,
    get_entry_folder,
    jsonl_loads,
    make_unique_slug,
    parse_any_payload,
    pick_leaderboard,
    slugify,
    write_csv,
    write_json,
    write_leaderboard_csv,
    write_matrix_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_LEADERBOARD_REPO = "SWE-bench/swe-bench.github.io"
DEFAULT_EXPERIMENTS_REPO = "SWE-bench/experiments"
DEFAULT_LEADERBOARD_PATH = "data/leaderboards.json"
SUPPORTED_LEADERBOARDS = (
    "verified",
    "bash-only",
    "multimodal",
    "lite",
    "multilingual",
    "test",
)
DEFAULT_CACHE_DIR = "resources/.cache/swe_bench_history"
DEFAULT_OUTPUT_BASE = "resources/swe_bench_history"
DEFAULT_TIMEOUT = 30


@dataclass(frozen=True)
class GitHubCommit:
    """Commit metadata from the GitHub commits API."""

    sha: str
    committed_at: datetime
    message: str
    html_url: str | None



@dataclass
class SnapshotResult:
    """Aggregated result for one leaderboard commit snapshot."""

    snapshot_id: str
    snapshot_dir: Path
    leaderboard_name: str
    leaderboard_commit: GitHubCommit
    experiments_commit: GitHubCommit
    experiments_fallback: bool
    entries: list[JsonDict]
    results: list[TrajectoryResults]
    failures: list[JsonDict]
    status: str
    status_message: str | None = None



def _parse_iso8601_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_date_arg(value: str) -> date:
    return date.fromisoformat(value)



def _default_request_headers(token: str | None) -> dict[str, str]:
    headers: dict[str, str] = {"User-Agent": "llm-wc-swebench-history-parser"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_api_headers(token: str | None) -> dict[str, str]:
    headers = _default_request_headers(token)
    headers["Accept"] = "application/vnd.github+json"
    headers["X-GitHub-Api-Version"] = "2022-11-28"
    return headers


def _http_get_text(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    headers: dict[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    req_headers = _default_request_headers(None)
    if headers:
        req_headers.update(headers)
    request = Request(url, headers=req_headers)
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()
        response_headers = {k: v for k, v in response.headers.items()}
    return payload.decode("utf-8"), response_headers


def _cache_paths(cache_dir: Path, url: str) -> tuple[Path, Path]:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    suffix = Path(url.split("?", 1)[0]).suffix or ".txt"
    payload_path = cache_dir / f"{digest}{suffix}"
    headers_path = cache_dir / f"{digest}.headers.json"
    return payload_path, headers_path


def _read_with_cache(
    url: str,
    *,
    cache_dir: Path | None,
    timeout: int = DEFAULT_TIMEOUT,
    headers: dict[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    if cache_dir is None:
        return _http_get_text(url, timeout=timeout, headers=headers)

    cache_dir.mkdir(parents=True, exist_ok=True)
    payload_path, headers_path = _cache_paths(cache_dir, url)

    if payload_path.exists():
        payload_text = payload_path.read_text(encoding="utf-8")
        if headers_path.exists():
            try:
                cached_headers = json.loads(headers_path.read_text(encoding="utf-8"))
                if isinstance(cached_headers, dict):
                    return payload_text, {
                        str(k): str(v) for k, v in cached_headers.items()
                    }
            except json.JSONDecodeError:
                pass
        return payload_text, {}

    payload_text, response_headers = _http_get_text(
        url, timeout=timeout, headers=headers
    )
    payload_path.write_text(payload_text, encoding="utf-8")
    headers_path.write_text(
        json.dumps(response_headers, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload_text, response_headers


def _read_json_url(
    url: str,
    *,
    cache_dir: Path | None,
    timeout: int = DEFAULT_TIMEOUT,
    headers: dict[str, str] | None = None,
) -> tuple[JsonValue, dict[str, str]]:
    text, response_headers = _read_with_cache(
        url, cache_dir=cache_dir, timeout=timeout, headers=headers
    )
    payload = json.loads(text)
    return payload, response_headers


def _parse_link_header(link_header: str | None) -> dict[str, str]:
    if not link_header:
        return {}
    links: dict[str, str] = {}
    for part in link_header.split(","):
        section = part.strip()
        if not section:
            continue
        if ";" not in section:
            continue
        raw_url, raw_rel = section.split(";", 1)
        raw_url = raw_url.strip()
        raw_rel = raw_rel.strip()
        if not (raw_url.startswith("<") and raw_url.endswith(">")):
            continue
        if not raw_rel.startswith("rel="):
            continue
        rel = raw_rel.split("=", 1)[1].strip().strip('"')
        links[rel] = raw_url[1:-1]
    return links


def _parse_commit_item(item: JsonDict) -> GitHubCommit | None:
    sha = item.get("sha")
    commit = item.get("commit")
    if not isinstance(sha, str) or not isinstance(commit, dict):
        return None

    inner_author = commit.get("author")
    inner_committer = commit.get("committer")
    committed_at = None
    if isinstance(inner_author, dict) and isinstance(inner_author.get("date"), str):
        committed_at = _parse_iso8601_utc(inner_author["date"])
    elif isinstance(inner_committer, dict) and isinstance(
        inner_committer.get("date"), str
    ):
        committed_at = _parse_iso8601_utc(inner_committer["date"])
    else:
        return None

    message = commit.get("message")
    if not isinstance(message, str):
        message = ""
    html_url = item.get("html_url")
    if not isinstance(html_url, str):
        html_url = None

    return GitHubCommit(
        sha=sha,
        committed_at=committed_at,
        message=message,
        html_url=html_url,
    )


def _fetch_commit_items_for_path(
    owner_repo: str,
    path: str,
    *,
    token: str | None,
    cache_dir: Path | None,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[JsonDict]:
    encoded_path = quote(path, safe="/")
    next_url = (
        f"https://api.github.com/repos/{owner_repo}/commits"
        f"?path={encoded_path}&per_page=100"
    )
    headers = _github_api_headers(token)
    items: list[JsonDict] = []

    while next_url:
        logger.debug("Fetching commits page: %s", next_url)
        payload, response_headers = _read_json_url(
            next_url,
            cache_dir=cache_dir,
            timeout=timeout,
            headers=headers,
        )
        if not isinstance(payload, list):
            raise TypeError(f"Unexpected commits payload type: {type(payload)}")
        for value in payload:
            if isinstance(value, dict):
                items.append(value)

        link_header = response_headers.get("Link") or response_headers.get("link")
        next_url = _parse_link_header(link_header).get("next")

    return items


def _fetch_commits_for_path(
    owner_repo: str,
    path: str,
    *,
    token: str | None,
    cache_dir: Path | None,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[GitHubCommit]:
    items = _fetch_commit_items_for_path(
        owner_repo,
        path,
        token=token,
        cache_dir=cache_dir,
        timeout=timeout,
    )
    commits: list[GitHubCommit] = []
    for item in items:
        parsed = _parse_commit_item(item)
        if parsed is not None:
            commits.append(parsed)

    # API returns newest-first; reorder for chronological processing.
    commits.sort(key=lambda commit: commit.committed_at)
    return commits


def _filter_commits(
    commits: list[GitHubCommit],
    *,
    since_date: date | None,
    until_date: date | None,
    max_commits: int | None,
) -> list[GitHubCommit]:
    filtered: list[GitHubCommit] = []
    for commit in commits:
        commit_date = commit.committed_at.date()
        if since_date and commit_date < since_date:
            continue
        if until_date and commit_date > until_date:
            continue
        filtered.append(commit)

    if max_commits is not None and max_commits > 0 and len(filtered) > max_commits:
        filtered = filtered[-max_commits:]

    return filtered


def _align_experiments_commit(
    leaderboard_commit: GitHubCommit, experiments_commits: list[GitHubCommit]
) -> tuple[GitHubCommit, bool]:
    if not experiments_commits:
        raise ValueError("No experiments commits available for alignment.")

    dates = [commit.committed_at for commit in experiments_commits]
    idx = bisect.bisect_right(dates, leaderboard_commit.committed_at) - 1
    if idx >= 0:
        return experiments_commits[idx], False
    return experiments_commits[0], True


def _iter_entries(leaderboard: JsonDict) -> list[JsonDict]:
    """Wrapper returning list instead of iterator for history snapshot processing."""
    for key in ("entries", "models", "results", "rows"):
        value = leaderboard.get(key)
        if isinstance(value, list):
            return [entry for entry in value if isinstance(entry, dict)]
    raise KeyError(
        "Could not find leaderboard entries list. Checked keys: "
        "entries, models, results, rows"
    )


def _extract_traj_url(entry: JsonDict) -> str | None:
    folder = get_entry_folder(entry)

    def is_candidate(key: str, value: Any) -> bool:
        if not isinstance(value, str):
            return False
        if not value.startswith(("http://", "https://")):
            return False
        key_lower = key.lower()
        if (
            "traj" not in key_lower
            and "trajectory" not in key_lower
            and key_lower not in {"results_url", "result_url", "instance_results_url"}
        ):
            return False
        if "docent" in key_lower:
            return False
        if folder and folder not in value:
            return False
        return True

    for key, value in entry.items():
        if is_candidate(key, value):
            return value

    links = entry.get("links")
    if isinstance(links, dict):
        for key, value in links.items():
            if is_candidate(key, value):
                return value

    return None


def _build_candidate_urls(entry: JsonDict, experiments_base: str) -> list[str]:
    folder = get_entry_folder(entry)
    if not folder:
        return []
    return [f"{experiments_base}/{folder}/{name}" for name in CANDIDATE_TRAJ_FILES]


def _load_trajectory_url(
    url: str,
    *,
    cache_dir: Path | None,
    timeout: int = DEFAULT_TIMEOUT,
    token: str | None = None,
) -> JsonValue:
    text, _ = _read_with_cache(
        url,
        cache_dir=cache_dir,
        timeout=timeout,
        headers=_default_request_headers(token),
    )
    return parse_any_payload(text, url)


def _resolve_candidate_trajectory_url(
    entry: JsonDict,
    *,
    experiments_base: str,
    cache_dir: Path | None,
    timeout: int = DEFAULT_TIMEOUT,
    token: str | None = None,
) -> str:
    candidate_urls = _build_candidate_urls(entry, experiments_base)
    if not candidate_urls:
        raise DownloadError(
            "No candidate trajectory URL could be built (missing folder)."
        )

    for url in candidate_urls:
        try:
            _read_with_cache(
                url,
                cache_dir=cache_dir,
                timeout=timeout,
                headers=_default_request_headers(token),
            )
        except (HTTPError, URLError):
            continue
        return url

    raise DownloadError("No candidate trajectory URL was accessible.")


def _extract_instances_from_entry(
    entry: JsonDict,
    *,
    model_name: str,
    leaderboard_name: str,
    experiments_sha: str,
    experiments_base: str,
    instance_cache: dict[tuple[str, str, str], tuple[dict[str, int], str | None]],
    cache_dir: Path | None,
    timeout: int = DEFAULT_TIMEOUT,
    token: str | None = None,
) -> tuple[dict[str, int] | None, str | None, str | None]:
    folder = get_entry_folder(entry) or ""
    explicit_url = _extract_traj_url(entry)
    cache_key = (experiments_sha, folder, explicit_url or "")

    cached = instance_cache.get(cache_key)
    if cached is not None:
        cached_instances, cached_url = cached
        return dict(cached_instances), cached_url, None

    details = entry.get("per_instance_details")
    if isinstance(details, dict):
        try:
            details_instances = _extract_from_mapping(details)
            if details_instances:
                traj_url = "leaderboard:per_instance_details"
                return details_instances, traj_url, None
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Failed parsing per_instance_details for %s on %s: %s",
                model_name,
                leaderboard_name,
                exc,
            )

    explicit_error: str | None = None
    if explicit_url:
        try:
            payload = _load_trajectory_url(
                explicit_url, cache_dir=cache_dir, timeout=timeout, token=token
            )
            instances = extract_instance_results(payload)
            instance_cache[cache_key] = (dict(instances), explicit_url)
            return instances, explicit_url, None
        except Exception as exc:  # noqa: BLE001
            explicit_error = f"explicit_url_failed: {exc}"

    candidate_error: str | None = None
    try:
        candidate_url = _resolve_candidate_trajectory_url(
            entry,
            experiments_base=experiments_base,
            cache_dir=cache_dir,
            timeout=timeout,
            token=token,
        )
        payload = _load_trajectory_url(
            candidate_url, cache_dir=cache_dir, timeout=timeout, token=token
        )
        instances = extract_instance_results(payload)
        instance_cache[cache_key] = (dict(instances), candidate_url)
        return instances, candidate_url, None
    except Exception as exc:  # noqa: BLE001
        candidate_error = f"candidate_scan_failed: {exc}"

    errors = [part for part in (explicit_error, candidate_error) if part]
    if not errors:
        errors = ["No per-task outcomes found in entry or candidate files."]
    return None, explicit_url, "; ".join(errors)



def _snapshot_id_for_commit(commit: GitHubCommit) -> str:
    stamp = commit.committed_at.strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{commit.sha[:7]}"


def _experiments_raw_base(leaderboard_name: str, experiments_sha: str) -> str:
    return (
        "https://raw.githubusercontent.com/"
        f"{DEFAULT_EXPERIMENTS_REPO}/{experiments_sha}/evaluation/{leaderboard_name}"
    )


def _leaderboards_raw_url(leaderboard_sha: str) -> str:
    return (
        "https://raw.githubusercontent.com/"
        f"{DEFAULT_LEADERBOARD_REPO}/{leaderboard_sha}/{DEFAULT_LEADERBOARD_PATH}"
    )


def _process_snapshot(
    *,
    leaderboard_name: str,
    leaderboard_commit: GitHubCommit,
    experiments_commit: GitHubCommit,
    experiments_fallback: bool,
    snapshots_root: Path,
    instance_cache: dict[tuple[str, str, str], tuple[dict[str, int], str | None]],
    cache_dir: Path | None,
    timeout: int,
    token: str | None,
) -> SnapshotResult:
    snapshot_id = _snapshot_id_for_commit(leaderboard_commit)
    snapshot_dir = snapshots_root / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_raw_url = _leaderboards_raw_url(leaderboard_commit.sha)
    experiments_base = _experiments_raw_base(leaderboard_name, experiments_commit.sha)

    entries: list[JsonDict] = []
    results: list[TrajectoryResults] = []
    failures: list[JsonDict] = []
    status = "ok"
    status_message: str | None = None

    try:
        payload, _ = _read_json_url(
            leaderboard_raw_url,
            cache_dir=cache_dir,
            timeout=timeout,
            headers=_default_request_headers(token),
        )
    except Exception as exc:  # noqa: BLE001
        status = "snapshot_error"
        status_message = f"Failed to load leaderboard snapshot: {exc}"
        failures.append(
            {
                "model": "",
                "model_id": "",
                "error": status_message,
                "trajectory_url": None,
                "entry": {},
            }
        )
        logger.warning(
            "Snapshot %s (%s) failed to load: %s",
            snapshot_id,
            leaderboard_commit.sha,
            exc,
        )
    else:
        try:
            leaderboard = pick_leaderboard(payload, leaderboard_name)
            entries = _iter_entries(leaderboard)
        except Exception as exc:  # noqa: BLE001
            status = "missing_leaderboard"
            status_message = str(exc)
            logger.warning(
                "Snapshot %s (%s) missing leaderboard '%s': %s",
                snapshot_id,
                leaderboard_commit.sha,
                leaderboard_name,
                exc,
            )
        else:
            used_model_ids: set[str] = set()
            for entry in entries:
                model_name = extract_model_name(entry)
                model_id = make_unique_slug(slugify(model_name), used_model_ids)

                instances, traj_url, error = _extract_instances_from_entry(
                    entry,
                    model_name=model_name,
                    leaderboard_name=leaderboard_name,
                    experiments_sha=experiments_commit.sha,
                    experiments_base=experiments_base,
                    instance_cache=instance_cache,
                    cache_dir=cache_dir,
                    timeout=timeout,
                    token=token,
                )

                if instances is None:
                    logger.warning(
                        "Model unresolved in snapshot %s: %s (%s)",
                        snapshot_id,
                        model_name,
                        error,
                    )
                    failures.append(
                        {
                            "model": model_name,
                            "model_id": model_id,
                            "error": error or "No results found",
                            "trajectory_url": traj_url,
                            "entry": entry,
                        }
                    )
                    continue

                source = TrajectorySource(
                    model_name=model_name,
                    model_id=model_id,
                    leaderboard_name=leaderboard_name,
                    trajectory_url=traj_url,
                    raw_entry=entry,
                )
                results.append(TrajectoryResults(source=source, instances=instances))

    index_payload: JsonDict = {
        "leaderboard": leaderboard_name,
        "snapshot_id": snapshot_id,
        "status": status,
        "status_message": status_message,
        "source": {
            "leaderboard_repo": DEFAULT_LEADERBOARD_REPO,
            "leaderboard_path": DEFAULT_LEADERBOARD_PATH,
            "leaderboard_commit": {
                "sha": leaderboard_commit.sha,
                "date": leaderboard_commit.committed_at.isoformat(),
                "message": leaderboard_commit.message,
                "html_url": leaderboard_commit.html_url,
                "raw_url": leaderboard_raw_url,
            },
            "experiments_repo": DEFAULT_EXPERIMENTS_REPO,
            "experiments_commit": {
                "sha": experiments_commit.sha,
                "date": experiments_commit.committed_at.isoformat(),
                "message": experiments_commit.message,
                "html_url": experiments_commit.html_url,
            },
            "experiments_base": experiments_base,
            "experiments_fallback": experiments_fallback,
        },
        "counts": {
            "entries": len(entries),
            "parsed_models": len(results),
            "failures": len(failures),
        },
        "models": [
            {
                "model": item.source.model_name,
                "model_id": item.source.model_id,
                "trajectory_url": item.source.trajectory_url,
                "counts": {"total": item.total, "resolved": item.resolved},
            }
            for item in results
        ],
        "failures": failures,
    }
    write_json(snapshot_dir / "index.json", index_payload)
    write_leaderboard_csv(snapshot_dir, entries)
    write_matrix_csv(snapshot_dir, results)
    # Keep snapshot artifact shape stable even when no rows are available.
    if not (snapshot_dir / "leaderboard.csv").exists():
        (snapshot_dir / "leaderboard.csv").write_text("", encoding="utf-8")
    if not (snapshot_dir / "trajectory.csv").exists():
        (snapshot_dir / "trajectory.csv").write_text(
            "model,model_id\n", encoding="utf-8"
        )

    return SnapshotResult(
        snapshot_id=snapshot_id,
        snapshot_dir=snapshot_dir,
        leaderboard_name=leaderboard_name,
        leaderboard_commit=leaderboard_commit,
        experiments_commit=experiments_commit,
        experiments_fallback=experiments_fallback,
        entries=entries,
        results=results,
        failures=failures,
        status=status,
        status_message=status_message,
    )


def _build_leaderboard_history_rows(
    snapshots: list[SnapshotResult],
) -> tuple[list[dict[str, Any]], list[str]]:
    metadata_fields = [
        "snapshot_id",
        "leaderboard",
        "leaderboard_commit_sha",
        "leaderboard_commit_date",
        "experiments_commit_sha",
        "experiments_commit_date",
        "experiments_fallback",
    ]
    dynamic_fields: list[str] = []
    rows: list[dict[str, Any]] = []

    for snapshot in snapshots:
        for entry in snapshot.entries:
            row: dict[str, Any] = {
                "snapshot_id": snapshot.snapshot_id,
                "leaderboard": snapshot.leaderboard_name,
                "leaderboard_commit_sha": snapshot.leaderboard_commit.sha,
                "leaderboard_commit_date": snapshot.leaderboard_commit.committed_at.isoformat(),
                "experiments_commit_sha": snapshot.experiments_commit.sha,
                "experiments_commit_date": snapshot.experiments_commit.committed_at.isoformat(),
                "experiments_fallback": snapshot.experiments_fallback,
            }
            for key, value in entry.items():
                if key not in dynamic_fields:
                    dynamic_fields.append(key)
                row[key] = coerce_csv_value(value)
            rows.append(row)

    return rows, metadata_fields + dynamic_fields


def _build_model_history_rows(
    snapshots: list[SnapshotResult],
) -> tuple[list[dict[str, Any]], list[str]]:
    fieldnames = [
        "snapshot_id",
        "leaderboard",
        "leaderboard_commit_sha",
        "leaderboard_commit_date",
        "experiments_commit_sha",
        "experiments_commit_date",
        "experiments_fallback",
        "model",
        "model_id",
        "trajectory_url",
        "entry_date",
        "entry_folder",
        "resolved",
        "total",
        "resolved_rate",
    ]
    rows: list[dict[str, Any]] = []

    for snapshot in snapshots:
        for item in snapshot.results:
            total = item.total
            resolved = item.resolved
            resolved_rate = (resolved / total) if total else 0.0
            rows.append(
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "leaderboard": snapshot.leaderboard_name,
                    "leaderboard_commit_sha": snapshot.leaderboard_commit.sha,
                    "leaderboard_commit_date": snapshot.leaderboard_commit.committed_at.isoformat(),
                    "experiments_commit_sha": snapshot.experiments_commit.sha,
                    "experiments_commit_date": snapshot.experiments_commit.committed_at.isoformat(),
                    "experiments_fallback": snapshot.experiments_fallback,
                    "model": item.source.model_name,
                    "model_id": item.source.model_id,
                    "trajectory_url": item.source.trajectory_url,
                    "entry_date": coerce_csv_value(item.source.raw_entry.get("date")),
                    "entry_folder": coerce_csv_value(
                        item.source.raw_entry.get("folder")
                    ),
                    "resolved": resolved,
                    "total": total,
                    "resolved_rate": resolved_rate,
                }
            )

    return rows, fieldnames


def _build_trajectory_history_rows(
    snapshots: list[SnapshotResult],
) -> tuple[list[dict[str, Any]], list[str]]:
    metadata_fields = [
        "snapshot_id",
        "leaderboard",
        "leaderboard_commit_sha",
        "leaderboard_commit_date",
        "experiments_commit_sha",
        "experiments_commit_date",
        "experiments_fallback",
        "model",
        "model_id",
        "trajectory_url",
    ]

    all_instances: set[str] = set()
    for snapshot in snapshots:
        for item in snapshot.results:
            all_instances.update(item.instances.keys())
    instance_ids = sorted(all_instances)
    fieldnames = metadata_fields + instance_ids

    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        for item in snapshot.results:
            row: dict[str, Any] = {
                "snapshot_id": snapshot.snapshot_id,
                "leaderboard": snapshot.leaderboard_name,
                "leaderboard_commit_sha": snapshot.leaderboard_commit.sha,
                "leaderboard_commit_date": snapshot.leaderboard_commit.committed_at.isoformat(),
                "experiments_commit_sha": snapshot.experiments_commit.sha,
                "experiments_commit_date": snapshot.experiments_commit.committed_at.isoformat(),
                "experiments_fallback": snapshot.experiments_fallback,
                "model": item.source.model_name,
                "model_id": item.source.model_id,
                "trajectory_url": item.source.trajectory_url,
            }
            for instance_id in instance_ids:
                row[instance_id] = item.instances.get(instance_id, 0)
            rows.append(row)

    return rows, fieldnames


def _build_failure_rows(
    snapshots: list[SnapshotResult],
) -> tuple[list[dict[str, Any]], list[str]]:
    fieldnames = [
        "snapshot_id",
        "leaderboard",
        "leaderboard_commit_sha",
        "leaderboard_commit_date",
        "experiments_commit_sha",
        "experiments_commit_date",
        "experiments_fallback",
        "model",
        "model_id",
        "error",
        "trajectory_url",
        "folder",
        "entry_date",
        "entry_json",
    ]
    rows: list[dict[str, Any]] = []

    for snapshot in snapshots:
        for failure in snapshot.failures:
            entry = failure.get("entry")
            if not isinstance(entry, dict):
                entry = {}
            rows.append(
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "leaderboard": snapshot.leaderboard_name,
                    "leaderboard_commit_sha": snapshot.leaderboard_commit.sha,
                    "leaderboard_commit_date": snapshot.leaderboard_commit.committed_at.isoformat(),
                    "experiments_commit_sha": snapshot.experiments_commit.sha,
                    "experiments_commit_date": snapshot.experiments_commit.committed_at.isoformat(),
                    "experiments_fallback": snapshot.experiments_fallback,
                    "model": coerce_csv_value(failure.get("model")),
                    "model_id": coerce_csv_value(failure.get("model_id")),
                    "error": coerce_csv_value(failure.get("error")),
                    "trajectory_url": coerce_csv_value(failure.get("trajectory_url")),
                    "folder": coerce_csv_value(entry.get("folder")),
                    "entry_date": coerce_csv_value(entry.get("date")),
                    "entry_json": json.dumps(entry, sort_keys=True),
                }
            )

    return rows, fieldnames


def _write_history_outputs(
    *,
    output_dir: Path,
    snapshots: list[SnapshotResult],
    leaderboard_name: str,
    selected_commits_count: int,
    all_leaderboard_commits_count: int,
    all_experiments_commits_count: int,
    args: argparse.Namespace,
) -> None:
    history_index = {
        "leaderboard": leaderboard_name,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "filters": {
            "since_date": args.since_date,
            "until_date": args.until_date,
            "max_commits": args.max_commits,
        },
        "source": {
            "leaderboard_repo": DEFAULT_LEADERBOARD_REPO,
            "leaderboard_path": DEFAULT_LEADERBOARD_PATH,
            "experiments_repo": DEFAULT_EXPERIMENTS_REPO,
            "experiments_path": f"evaluation/{leaderboard_name}",
        },
        "counts": {
            "all_leaderboard_commits": all_leaderboard_commits_count,
            "all_experiments_commits": all_experiments_commits_count,
            "selected_leaderboard_commits": selected_commits_count,
            "snapshots_written": len(snapshots),
            "parsed_models": sum(len(snapshot.results) for snapshot in snapshots),
            "failures": sum(len(snapshot.failures) for snapshot in snapshots),
        },
        "snapshots": [
            {
                "snapshot_id": snapshot.snapshot_id,
                "path": f"snapshots/{snapshot.snapshot_id}",
                "status": snapshot.status,
                "status_message": snapshot.status_message,
                "leaderboard_commit": {
                    "sha": snapshot.leaderboard_commit.sha,
                    "date": snapshot.leaderboard_commit.committed_at.isoformat(),
                    "message": snapshot.leaderboard_commit.message,
                    "html_url": snapshot.leaderboard_commit.html_url,
                },
                "experiments_commit": {
                    "sha": snapshot.experiments_commit.sha,
                    "date": snapshot.experiments_commit.committed_at.isoformat(),
                    "message": snapshot.experiments_commit.message,
                    "html_url": snapshot.experiments_commit.html_url,
                    "fallback_used": snapshot.experiments_fallback,
                },
                "counts": {
                    "entries": len(snapshot.entries),
                    "parsed_models": len(snapshot.results),
                    "failures": len(snapshot.failures),
                },
            }
            for snapshot in snapshots
        ],
    }
    write_json(output_dir / "history_index.json", history_index)

    leaderboard_rows, leaderboard_fields = _build_leaderboard_history_rows(snapshots)
    write_csv(
        output_dir / "leaderboard_history.csv", leaderboard_rows, leaderboard_fields
    )

    model_rows, model_fields = _build_model_history_rows(snapshots)
    write_csv(output_dir / "model_history.csv", model_rows, model_fields)

    trajectory_rows, trajectory_fields = _build_trajectory_history_rows(snapshots)
    write_csv(
        output_dir / "trajectory_history.csv", trajectory_rows, trajectory_fields
    )

    failure_rows, failure_fields = _build_failure_rows(snapshots)
    write_csv(output_dir / "failures.csv", failure_rows, failure_fields)


def regenerate_summaries_from_snapshots(
    run_dir: Path,
    leaderboard_name: str,
) -> None:
    """Rebuild top-level history summaries from snapshot directories.

    Reads each ``snapshots/<snapshot_id>/index.json`` (and its sibling CSVs)
    and regenerates ``history_index.json``, ``leaderboard_history.csv``,
    ``model_history.csv``, ``trajectory_history.csv``, and ``failures.csv``.
    """
    snapshots_root = run_dir / "snapshots"
    if not snapshots_root.is_dir():
        raise FileNotFoundError(f"No snapshots directory in {run_dir}")

    snapshot_dirs = sorted(d for d in snapshots_root.iterdir() if d.is_dir())
    if not snapshot_dirs:
        raise FileNotFoundError(f"No snapshot directories in {snapshots_root}")

    history_snapshots: list[JsonDict] = []
    leaderboard_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    leaderboard_dynamic_fields: list[str] = []
    all_instance_ids: set[str] = set()

    for snap_dir in snapshot_dirs:
        index_path = snap_dir / "index.json"
        if not index_path.exists():
            continue

        with open(index_path, encoding="utf-8") as fh:
            idx = json.load(fh)

        snapshot_id = idx["snapshot_id"]
        source = idx.get("source", {})
        lb_commit = source.get("leaderboard_commit", {})
        exp_commit = source.get("experiments_commit", {})
        exp_fallback = source.get("experiments_fallback", False)
        status = idx.get("status", "ok")
        status_message = idx.get("status_message")

        meta = {
            "snapshot_id": snapshot_id,
            "leaderboard": leaderboard_name,
            "leaderboard_commit_sha": lb_commit.get("sha", ""),
            "leaderboard_commit_date": lb_commit.get("date", ""),
            "experiments_commit_sha": exp_commit.get("sha", ""),
            "experiments_commit_date": exp_commit.get("date", ""),
            "experiments_fallback": exp_fallback,
        }

        counts = idx.get("counts", {})
        history_snapshots.append({
            "snapshot_id": snapshot_id,
            "path": f"snapshots/{snapshot_id}",
            "status": status,
            "status_message": status_message,
            "leaderboard_commit": {
                "sha": lb_commit.get("sha", ""),
                "date": lb_commit.get("date", ""),
                "message": lb_commit.get("message", ""),
                "html_url": lb_commit.get("html_url"),
            },
            "experiments_commit": {
                "sha": exp_commit.get("sha", ""),
                "date": exp_commit.get("date", ""),
                "message": exp_commit.get("message", ""),
                "html_url": exp_commit.get("html_url"),
                "fallback_used": exp_fallback,
            },
            "counts": {
                "entries": counts.get("entries", 0),
                "parsed_models": counts.get("parsed_models", 0),
                "failures": counts.get("failures", 0),
            },
        })

        # leaderboard_history.csv
        lb_csv_path = snap_dir / "leaderboard.csv"
        if lb_csv_path.exists():
            with open(lb_csv_path, encoding="utf-8") as fh:
                lb_reader = csv.DictReader(fh)
                if lb_reader.fieldnames:
                    for field in lb_reader.fieldnames:
                        if field not in leaderboard_dynamic_fields:
                            leaderboard_dynamic_fields.append(field)
                    for lb_entry in lb_reader:
                        row = dict(meta)
                        for k, v in lb_entry.items():
                            row[k] = coerce_csv_value(v)
                        leaderboard_rows.append(row)

        # model_history.csv
        for model in idx.get("models", []):
            m_counts = model.get("counts", {})
            total = m_counts.get("total", 0)
            resolved = m_counts.get("resolved", 0)
            model_rows.append({
                **meta,
                "model": model.get("model", ""),
                "model_id": model.get("model_id", ""),
                "trajectory_url": model.get("trajectory_url", ""),
                "entry_date": "",
                "entry_folder": "",
                "resolved": resolved,
                "total": total,
                "resolved_rate": (resolved / total) if total else 0.0,
            })

        # trajectory_history.csv
        traj_path = snap_dir / "trajectory.csv"
        if traj_path.exists():
            with open(traj_path, encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames:
                    instance_cols = [
                        c for c in reader.fieldnames if c not in ("model", "model_id")
                    ]
                    all_instance_ids.update(instance_cols)
                    for trow in reader:
                        traj_row = dict(meta)
                        traj_row["model"] = trow.get("model", "")
                        traj_row["model_id"] = trow.get("model_id", "")
                        traj_row["trajectory_url"] = ""
                        for col in instance_cols:
                            traj_row[col] = trow.get(col, 0)
                        trajectory_rows.append(traj_row)

        # failures.csv
        for failure in idx.get("failures", []):
            f_entry = failure.get("entry", {})
            if not isinstance(f_entry, dict):
                f_entry = {}
            failure_rows.append({
                **meta,
                "model": coerce_csv_value(failure.get("model")),
                "model_id": coerce_csv_value(failure.get("model_id")),
                "error": coerce_csv_value(failure.get("error")),
                "trajectory_url": coerce_csv_value(failure.get("trajectory_url")),
                "folder": coerce_csv_value(f_entry.get("folder")),
                "entry_date": coerce_csv_value(f_entry.get("date")),
                "entry_json": json.dumps(f_entry, sort_keys=True),
            })

    # Write history_index.json
    total_parsed = sum(s["counts"]["parsed_models"] for s in history_snapshots)
    total_failures = sum(s["counts"]["failures"] for s in history_snapshots)

    history_index = {
        "leaderboard": leaderboard_name,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "filters": {"since_date": None, "until_date": None, "max_commits": None},
        "source": {
            "leaderboard_repo": DEFAULT_LEADERBOARD_REPO,
            "leaderboard_path": DEFAULT_LEADERBOARD_PATH,
            "experiments_repo": DEFAULT_EXPERIMENTS_REPO,
            "experiments_path": f"evaluation/{leaderboard_name}",
        },
        "counts": {
            "all_leaderboard_commits": len(history_snapshots),
            "all_experiments_commits": 0,
            "selected_leaderboard_commits": len(history_snapshots),
            "snapshots_written": len(history_snapshots),
            "parsed_models": total_parsed,
            "failures": total_failures,
        },
        "snapshots": history_snapshots,
    }
    write_json(run_dir / "history_index.json", history_index)

    # Write CSVs
    lb_meta_fields = [
        "snapshot_id", "leaderboard", "leaderboard_commit_sha",
        "leaderboard_commit_date", "experiments_commit_sha",
        "experiments_commit_date", "experiments_fallback",
    ]
    write_csv(
        run_dir / "leaderboard_history.csv",
        leaderboard_rows,
        lb_meta_fields + leaderboard_dynamic_fields,
    )

    model_fields = [
        "snapshot_id", "leaderboard", "leaderboard_commit_sha",
        "leaderboard_commit_date", "experiments_commit_sha",
        "experiments_commit_date", "experiments_fallback",
        "model", "model_id", "trajectory_url",
        "entry_date", "entry_folder",
        "resolved", "total", "resolved_rate",
    ]
    write_csv(run_dir / "model_history.csv", model_rows, model_fields)

    traj_meta_fields = [
        "snapshot_id", "leaderboard", "leaderboard_commit_sha",
        "leaderboard_commit_date", "experiments_commit_sha",
        "experiments_commit_date", "experiments_fallback",
        "model", "model_id", "trajectory_url",
    ]
    instance_ids_sorted = sorted(all_instance_ids)
    write_csv(
        run_dir / "trajectory_history.csv",
        trajectory_rows,
        traj_meta_fields + instance_ids_sorted,
    )

    failure_fields = [
        "snapshot_id", "leaderboard", "leaderboard_commit_sha",
        "leaderboard_commit_date", "experiments_commit_sha",
        "experiments_commit_date", "experiments_fallback",
        "model", "model_id", "error", "trajectory_url",
        "folder", "entry_date", "entry_json",
    ]
    write_csv(run_dir / "failures.csv", failure_rows, failure_fields)

    logger.info(
        "Regenerated summaries: %d snapshots, %d models, %d failures",
        len(history_snapshots), total_parsed, total_failures,
    )


def _existing_snapshot_shas(run_dir: Path) -> set[str]:
    """Return the set of 7-char short SHAs already present in snapshots."""
    snapshots_root = run_dir / "snapshots"
    if not snapshots_root.is_dir():
        return set()
    shas: set[str] = set()
    for snap_dir in snapshots_root.iterdir():
        if snap_dir.is_dir():
            # Format: 20250507T093905Z_1a0866b -> 1a0866b
            name = snap_dir.name
            if "_" in name:
                shas.add(name.rsplit("_", 1)[-1])
    return shas


def _cleanup_cache_dir(cache_dir: Path | None) -> None:
    if cache_dir is None or not cache_dir.exists():
        return

    try:
        resolved = cache_dir.resolve()
        if resolved in {Path("/").resolve(), Path.cwd().resolve()}:
            logger.warning("Refusing to delete suspicious cache path: %s", cache_dir)
            return

        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
        else:
            cache_dir.unlink()
        logger.info("Removed cache directory: %s", cache_dir)

        parent = cache_dir.parent
        if parent.name == ".cache":
            try:
                parent.rmdir()
            except OSError:
                pass
    except OSError as exc:
        logger.warning("Failed to remove cache directory %s: %s", cache_dir, exc)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Parse SWE-Bench leaderboard history and per-task outcomes from "
            "GitHub commit history."
        )
    )
    parser.add_argument(
        "--leaderboard-name",
        default="verified",
        choices=SUPPORTED_LEADERBOARDS,
        help="Leaderboard name to parse (default: verified).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Default is "
            "resources/swe_bench_history/<leaderboard-name>. "
            "Snapshots and summary files are written directly here."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help=(
            "URL cache directory for remote payloads. Use 'none' to disable "
            f"(default: {DEFAULT_CACHE_DIR})."
        ),
    )
    parser.add_argument(
        "--max-commits",
        type=int,
        default=None,
        help="Optional limit: process only the newest N commits after date filtering.",
    )
    parser.add_argument(
        "--since-date",
        default=None,
        help="Include commits on/after this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--until-date",
        default=None,
        help="Include commits on/before this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--github-token",
        default=None,
        help="GitHub API token (or set GITHUB_TOKEN env var).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=True,
        help=(
            "Incremental mode (default). Only fetch commits not already "
            "present in the latest run directory. New snapshots are added "
            "to the existing run dir and summaries are regenerated."
        ),
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Force a full parse, creating a new run directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    if args.no_incremental:
        args.incremental = False

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    github_token = args.github_token or os.getenv("GITHUB_TOKEN")
    cache_dir = None
    if isinstance(args.cache_dir, str) and args.cache_dir.lower() != "none":
        cache_dir = Path(args.cache_dir)

    try:
        since_date = _parse_date_arg(args.since_date) if args.since_date else None
        until_date = _parse_date_arg(args.until_date) if args.until_date else None
        if since_date and until_date and since_date > until_date:
            logger.error("--since-date cannot be after --until-date.")
            return 1

        logger.info("Fetching leaderboard commit history from GitHub API...")
        try:
            leaderboard_commits = _fetch_commits_for_path(
                DEFAULT_LEADERBOARD_REPO,
                DEFAULT_LEADERBOARD_PATH,
                token=github_token,
                cache_dir=cache_dir,
                timeout=args.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch leaderboard commits: %s", exc)
            return 1

        experiments_path = f"evaluation/{args.leaderboard_name}"
        logger.info("Fetching experiments commit history for %s...", experiments_path)
        try:
            experiments_commits = _fetch_commits_for_path(
                DEFAULT_EXPERIMENTS_REPO,
                experiments_path,
                token=github_token,
                cache_dir=cache_dir,
                timeout=args.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch experiments commits: %s", exc)
            experiments_commits = []

        if not leaderboard_commits:
            logger.error("No commits found for %s", DEFAULT_LEADERBOARD_PATH)
            return 1
        if not experiments_commits:
            logger.warning(
                "No experiments commits found for %s; "
                "will rely on per_instance_details from leaderboard entries",
                experiments_path,
            )

        selected_commits = _filter_commits(
            leaderboard_commits,
            since_date=since_date,
            until_date=until_date,
            max_commits=args.max_commits,
        )
        if not selected_commits:
            logger.error("No leaderboard commits matched the selected filters.")
            return 1

        logger.info(
            "Selected %d/%d leaderboard commits (oldest=%s newest=%s)",
            len(selected_commits),
            len(leaderboard_commits),
            selected_commits[0].committed_at.isoformat(),
            selected_commits[-1].committed_at.isoformat(),
        )

        if args.output_dir is None:
            output_dir = Path(DEFAULT_OUTPUT_BASE) / args.leaderboard_name
        else:
            output_dir = Path(args.output_dir)

        # ── Incremental mode: skip commits already in snapshots/ ──
        existing_shas: set[str] = set()

        if args.incremental:
            existing_shas = _existing_snapshot_shas(output_dir)
            if existing_shas:
                logger.info(
                    "Incremental mode: found %d existing snapshots in %s",
                    len(existing_shas),
                    output_dir,
                )

        # Filter out commits whose short SHA is already present
        if existing_shas:
            new_commits = [
                c for c in selected_commits
                if c.sha[:7] not in existing_shas
            ]
            skipped = len(selected_commits) - len(new_commits)
            if skipped:
                logger.info(
                    "Incremental: skipping %d already-parsed commit(s), "
                    "%d new commit(s) to process",
                    skipped,
                    len(new_commits),
                )
            selected_commits = new_commits

        if not selected_commits:
            logger.info("No new commits to process. Up to date.")
            return 0

        snapshots_root = output_dir / "snapshots"
        snapshots_root.mkdir(parents=True, exist_ok=True)

        logger.info("Writing history outputs under %s", output_dir)

        instance_cache: dict[
            tuple[str, str, str], tuple[dict[str, int], str | None]
        ] = {}
        snapshots: list[SnapshotResult] = []

        # When no experiments commits exist (e.g. multilingual), use the
        # leaderboard commit itself as a placeholder.  Trajectory resolution
        # will rely entirely on per_instance_details embedded in entries.
        no_experiments = not experiments_commits

        for idx, leaderboard_commit in enumerate(selected_commits, start=1):
            if no_experiments:
                experiments_commit = leaderboard_commit
                fallback_used = True
            else:
                experiments_commit, fallback_used = _align_experiments_commit(
                    leaderboard_commit, experiments_commits
                )
            if fallback_used:
                logger.warning(
                    "Snapshot %s predates available experiments commits; "
                    "falling back to oldest experiments commit %s",
                    leaderboard_commit.sha,
                    experiments_commit.sha,
                )

            logger.info(
                "[%d/%d] Processing leaderboard commit %s (%s)",
                idx,
                len(selected_commits),
                leaderboard_commit.sha,
                leaderboard_commit.committed_at.isoformat(),
            )
            snapshot = _process_snapshot(
                leaderboard_name=args.leaderboard_name,
                leaderboard_commit=leaderboard_commit,
                experiments_commit=experiments_commit,
                experiments_fallback=fallback_used,
                snapshots_root=snapshots_root,
                instance_cache=instance_cache,
                cache_dir=cache_dir,
                timeout=args.timeout,
                token=github_token,
            )
            snapshots.append(snapshot)

        # In incremental mode we always regenerate summaries from all
        # snapshots on disk (old + new).  In full mode we can use the
        # in-memory snapshots directly.
        if args.incremental and existing_shas:
            regenerate_summaries_from_snapshots(output_dir, args.leaderboard_name)
        else:
            _write_history_outputs(
                output_dir=output_dir,
                snapshots=snapshots,
                leaderboard_name=args.leaderboard_name,
                selected_commits_count=len(selected_commits),
                all_leaderboard_commits_count=len(leaderboard_commits),
                all_experiments_commits_count=len(experiments_commits),
                args=args,
            )

        total_snapshots = len(list(snapshots_root.iterdir()))
        logger.info("Done. Total snapshots: %d (new: %d)", total_snapshots, len(snapshots))
        logger.info(
            "Parsed models (new): %d | Failures (new): %d",
            sum(len(snapshot.results) for snapshot in snapshots),
            sum(len(snapshot.failures) for snapshot in snapshots),
        )
        logger.info("Top-level index: %s", output_dir / "history_index.json")
        return 0
    finally:
        _cleanup_cache_dir(cache_dir)


if __name__ == "__main__":
    raise SystemExit(main())
