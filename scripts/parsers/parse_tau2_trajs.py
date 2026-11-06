#!/usr/bin/env python3
"""
Parse Tau2-Bench trajectory results from GitHub or a local clone.

Fetches submission data from the sierra-research/tau2-bench repository
(GitHub-first with local fallback) and produces normalized outputs:

-   ``index.json``:  Metadata index with model list, counts, and failures.
-   ``leaderboard.csv``:  Leaderboard summary from submission metadata.
-   ``trajectory.csv``:  Dense (Model x Task) matrix with pass rates.
-   ``trajectories/<model_id>.json``:  Detailed per-model results and
    conversation logs.

Supports incremental updates: on re-run, only new submissions are fetched
and processed; top-level CSVs are regenerated from all data.

Usage:
    # Fetch from GitHub (default)
    uv run python scripts/parsers/parse_tau2_trajs.py

    # Incremental update (skip already-processed submissions)
    uv run python scripts/parsers/parse_tau2_trajs.py --verbose

    # Full re-parse
    uv run python scripts/parsers/parse_tau2_trajs.py --no-incremental

    # Local clone fallback
    uv run python scripts/parsers/parse_tau2_trajs.py --repo-path /path/to/tau2-bench
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Ensure the src directory is importable
_src = str(Path(__file__).resolve().parent.parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from llm_wc.parsers.common import (
    JsonDict,
    TrajectoryResults,
    TrajectorySource,
    make_unique_slug,
    read_with_cache,
    slugify,
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAU2_REPO = "sierra-research/tau2-bench"
TAU2_RAW_BASE = f"https://raw.githubusercontent.com/{TAU2_REPO}/main"
TAU2_SUBMISSIONS_PATH = "web/leaderboard/public/submissions"
DOMAINS = ["airline", "retail", "telecom"]


# ---------------------------------------------------------------------------
# GitHub fetching helpers
# ---------------------------------------------------------------------------


def _github_headers(token: str | None) -> dict[str, str]:
    """Build HTTP headers for GitHub API requests."""
    headers = {"User-Agent": "llm-wc-tau2-parser"}
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _read_json_local(path: Path) -> Any:
    """Read and parse a JSON file from disk."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fetch_text(
    url: str,
    cache_dir: Path | None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> str:
    """Fetch text from a URL with optional caching and custom headers.

    Uses common.read_with_cache for raw.githubusercontent.com URLs (no auth
    needed). For API URLs requiring custom headers, handles caching manually.
    """
    if headers and any(k.lower() == "authorization" for k in headers):
        # API call with auth — cache manually
        from llm_wc.parsers.common import cache_path as _cache_path

        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached = _cache_path(cache_dir, url)
            if cached.exists():
                logger.debug(f"Cache hit: {url}")
                return cached.read_text(encoding="utf-8")

        req = Request(url, headers=headers)
        with urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")

        if cache_dir is not None:
            cached.write_text(text, encoding="utf-8")
        return text

    return read_with_cache(url, cache_dir)


def _fetch_json(
    relative_path: str,
    cache_dir: Path | None,
    repo_path: Path | None,
    token: str | None = None,
    timeout: int = 30,
) -> Any:
    """Fetch JSON from GitHub raw URL, falling back to local clone."""
    url = f"{TAU2_RAW_BASE}/{relative_path}"
    try:
        text = _fetch_text(url, cache_dir, timeout=timeout)
        return json.loads(text)
    except (HTTPError, URLError, OSError) as exc:
        if repo_path is not None:
            local_path = repo_path / relative_path
            if local_path.exists():
                logger.info(f"GitHub fetch failed, using local: {local_path}")
                return _read_json_local(local_path)
        raise RuntimeError(f"Failed to fetch {relative_path}: {exc}") from exc


def _list_trajectory_files(
    folder_name: str,
    submission_data: JsonDict,
    cache_dir: Path | None,
    repo_path: Path | None,
    token: str | None = None,
    timeout: int = 30,
) -> list[dict[str, str]]:
    """List trajectory files for a submission folder.

    Returns list of dicts with 'name' and 'download_url' keys.

    Strategy (in order):
    1. Use ``trajectory_files`` from submission.json (exact filenames)
    2. GitHub Contents API (requires token for higher rate limits)
    3. Local clone glob
    """
    base_path = f"{TAU2_SUBMISSIONS_PATH}/{folder_name}/trajectories"

    # Strategy 1: Exact filenames from submission metadata
    traj_file_map = submission_data.get("trajectory_files")
    if isinstance(traj_file_map, dict) and traj_file_map:
        results = []
        for domain, filename in traj_file_map.items():
            if filename and isinstance(filename, str):
                url = f"{TAU2_RAW_BASE}/{base_path}/{filename}"
                results.append({"name": filename, "download_url": url})
        if results:
            logger.debug(
                f"Using trajectory_files from submission.json: {len(results)} files"
            )
            return results

    # Strategy 2: GitHub Contents API
    api_url = (
        f"https://api.github.com/repos/{TAU2_REPO}/contents/{base_path}"
    )
    try:
        headers = _github_headers(token)
        text = _fetch_text(api_url, cache_dir, headers=headers, timeout=timeout)
        entries = json.loads(text)
        if isinstance(entries, list):
            results = [
                {"name": e["name"], "download_url": e["download_url"]}
                for e in entries
                if isinstance(e, dict) and e.get("name", "").endswith(".json")
            ]
            if results:
                return results
    except (HTTPError, URLError, OSError) as exc:
        logger.debug(f"GitHub API failed for {folder_name}/trajectories: {exc}")

    # Strategy 3: Local clone glob
    if repo_path is not None:
        traj_dir = repo_path / base_path
        if traj_dir.is_dir():
            return [
                {"name": f.name, "download_url": str(f)}
                for f in sorted(traj_dir.glob("*.json"))
            ]

    return []


def _fetch_trajectory_data(
    file_info: dict[str, str],
    cache_dir: Path | None,
    repo_path: Path | None,
    timeout: int = 30,
) -> Any:
    """Fetch and parse a trajectory JSON file."""
    url_or_path = file_info["download_url"]
    # Local path (from fallback)
    if url_or_path.startswith("/"):
        return _read_json_local(Path(url_or_path))
    # Remote URL
    text = _fetch_text(url_or_path, cache_dir, timeout=timeout)
    return json.loads(text)


# ---------------------------------------------------------------------------
# Trajectory processing
# ---------------------------------------------------------------------------


def _process_domain_trajectories(
    traj_data: Any,
    domain: str,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Parse a domain trajectory file into pass rates and summary.

    Returns:
        instances: {"{domain}-{task_id}": pass_rate}
        summary: {task_id: {"trials": int, "pass_rate": float}}
    """
    simulations = traj_data.get("simulations", [])
    task_trials: dict[str, list[float]] = {}

    for sim in simulations:
        task_id = sim.get("task_id")
        if task_id is None:
            continue
        reward = sim.get("reward_info", {}).get("reward", 0.0)
        task_trials.setdefault(task_id, []).append(reward)

    instances: dict[str, float] = {}
    summary: dict[str, dict[str, Any]] = {}
    for task_id, rewards in task_trials.items():
        pass_rate = sum(rewards) / len(rewards) if rewards else 0.0
        instances[f"{domain}-{task_id}"] = pass_rate
        summary[task_id] = {"trials": len(rewards), "pass_rate": pass_rate}

    return instances, summary


def _build_leaderboard_entry(submission_data: JsonDict) -> JsonDict:
    """Build a leaderboard entry dict from submission metadata."""
    entry: JsonDict = {
        "model": submission_data.get("model_name", ""),
        "organization": submission_data.get("model_organization", ""),
        "date": submission_data.get("submission_date", ""),
    }
    results_summary = submission_data.get("results", {})
    for domain in DOMAINS:
        domain_stats = results_summary.get(domain, {})
        for k, v in domain_stats.items():
            entry[f"{domain}_{k}"] = v
    return entry


# ---------------------------------------------------------------------------
# Incremental: reload previously processed results
# ---------------------------------------------------------------------------


def _reload_result(
    output_dir: Path,
    model_meta: JsonDict,
) -> TrajectoryResults | None:
    """Reload a TrajectoryResults from a previously saved per-model JSON."""
    traj_path = output_dir / model_meta.get("path", "")
    if not traj_path.exists():
        logger.warning(f"Cannot reload {model_meta.get('model')}: {traj_path} missing")
        return None

    detail = _read_json_local(traj_path)
    model_name = detail.get("model", model_meta.get("model", "unknown"))
    model_id = detail.get("model_id", model_meta.get("model_id", "unknown"))

    source = TrajectorySource(
        model_name=model_name,
        model_id=model_id,
        leaderboard_name="tau2-bench",
        raw_entry=model_meta.get("raw_entry", {}),
    )

    instances: dict[str, float] = {}
    for domain_name, domain_data in detail.get("domains", {}).items():
        for task_id, task_info in domain_data.get("summary", {}).items():
            instances[f"{domain_name}-{task_id}"] = task_info.get("pass_rate", 0.0)

    return TrajectoryResults(
        source=source,
        instances=instances,  # type: ignore[arg-type]
        scores=dict(instances),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse Tau2-Bench trajectory results from GitHub or local clone."
    )
    parser.add_argument(
        "--repo-path",
        default=None,
        help="Path to a local tau2-bench clone (optional fallback).",
    )
    parser.add_argument(
        "--output-dir",
        default="resources/tau2_bench",
        help="Output directory (default: resources/tau2_bench).",
    )
    parser.add_argument(
        "--cache-dir",
        default="resources/.cache/tau2_bench",
        help='HTTP cache directory. Use "none" to disable.',
    )
    parser.add_argument(
        "--github-token",
        default=None,
        help="GitHub API token (or set GITHUB_TOKEN env var).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=True,
        help="Skip already-processed submissions (default).",
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        default=False,
        help="Force full re-parse, ignoring existing results.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trajectories").mkdir(parents=True, exist_ok=True)

    cache_dir: Path | None = None
    if args.cache_dir.lower() != "none":
        cache_dir = Path(args.cache_dir)

    repo_path: Path | None = None
    if args.repo_path:
        repo_path = Path(args.repo_path)
        if not repo_path.is_dir():
            logger.error(f"--repo-path does not exist: {repo_path}")
            return 1

    token = args.github_token or os.getenv("GITHUB_TOKEN")
    incremental = args.incremental and not args.no_incremental

    # -----------------------------------------------------------------------
    # 1. Fetch manifest
    # -----------------------------------------------------------------------
    try:
        manifest = _fetch_json(
            f"{TAU2_SUBMISSIONS_PATH}/manifest.json",
            cache_dir, repo_path, token, args.timeout,
        )
    except Exception as exc:
        logger.error(f"Failed to fetch manifest: {exc}")
        return 1

    # Collect all submission folders (active + legacy + voice)
    submission_folders: list[str] = []
    for key in ("submissions", "legacy_submissions", "voice_submissions"):
        entries = manifest.get(key, [])
        submission_folders.extend(entries)
        if entries:
            logger.info(f"  {key}: {len(entries)} entries")
    logger.info(f"Manifest: {len(submission_folders)} total submissions")

    # -----------------------------------------------------------------------
    # 2. Determine which submissions to process
    # -----------------------------------------------------------------------
    existing_index: JsonDict = {}
    existing_models: list[JsonDict] = []
    existing_folders: set[str] = set()

    if incremental:
        index_path = output_dir / "index.json"
        if index_path.exists():
            existing_index = _read_json_local(index_path)
            existing_models = existing_index.get("models", [])
            existing_folders = {m["folder_name"] for m in existing_models if "folder_name" in m}
            logger.info(f"Incremental: {len(existing_folders)} existing submissions")

    folders_to_process = [
        f for f in submission_folders if f not in existing_folders
    ]
    logger.info(f"New submissions to process: {len(folders_to_process)}")

    # -----------------------------------------------------------------------
    # 3. Reload previously processed results (for CSV regeneration)
    # -----------------------------------------------------------------------
    all_results: list[TrajectoryResults] = []
    all_lb_entries: list[JsonDict] = []
    all_model_meta: list[JsonDict] = []
    all_folder_names: list[str] = []
    used_slugs: set[str] = set()
    failures: list[JsonDict] = []

    if incremental and existing_models:
        for model_meta in existing_models:
            prev_result = _reload_result(output_dir, model_meta)
            if prev_result is not None:
                all_results.append(prev_result)
                all_lb_entries.append(
                    _build_leaderboard_entry(model_meta.get("raw_entry", model_meta))
                )
                all_model_meta.append(model_meta)
                all_folder_names.append(model_meta.get("folder_name", ""))
                used_slugs.add(prev_result.source.model_id)

    # -----------------------------------------------------------------------
    # 4. Process new submissions
    # -----------------------------------------------------------------------
    for folder_name in folders_to_process:
        logger.info(f"Processing: {folder_name}")

        # 4a. Fetch submission.json
        try:
            submission_data = _fetch_json(
                f"{TAU2_SUBMISSIONS_PATH}/{folder_name}/submission.json",
                cache_dir, repo_path, token, args.timeout,
            )
        except Exception as exc:
            logger.error(f"Failed to fetch submission for {folder_name}: {exc}")
            failures.append({
                "folder": folder_name,
                "stage": "fetch_submission",
                "error": str(exc),
            })
            continue

        model_name = submission_data.get("model_name", folder_name)
        model_id = make_unique_slug(slugify(model_name), used_slugs)
        used_slugs.add(model_id)

        # 4b. Build TrajectorySource
        source = TrajectorySource(
            model_name=model_name,
            model_id=model_id,
            leaderboard_name="tau2-bench",
            trajectory_url=(
                f"https://github.com/{TAU2_REPO}/tree/main/"
                f"{TAU2_SUBMISSIONS_PATH}/{folder_name}"
            ),
            raw_entry=submission_data,
        )

        # 4c. Fetch and parse trajectory files per domain
        instances: dict[str, float] = {}
        model_detail: JsonDict = {
            "model": model_name,
            "model_id": model_id,
            "domains": {},
        }

        traj_files = _list_trajectory_files(
            folder_name, submission_data, cache_dir, repo_path, token,
            args.timeout,
        )

        for domain in DOMAINS:
            matching = [f for f in traj_files if domain in f["name"]]
            if not matching:
                logger.warning(
                    f"  No trajectory for {domain} in {folder_name}"
                )
                continue

            try:
                traj_data = _fetch_trajectory_data(
                    matching[0], cache_dir, repo_path, args.timeout,
                )
            except Exception as exc:
                logger.error(
                    f"  Failed to fetch {domain} trajectory for {folder_name}: {exc}"
                )
                continue

            domain_instances, domain_summary = _process_domain_trajectories(
                traj_data, domain,
            )
            instances.update(domain_instances)

            model_detail["domains"][domain] = {
                "tasks": traj_data.get("tasks", []),
                "simulations": traj_data.get("simulations", []),
                "summary": domain_summary,
            }
            logger.info(
                f"  {domain}: {len(domain_summary)} tasks, "
                f"{sum(len(v) for v in [s.get('simulations', []) for s in [traj_data]])} simulations"
            )

        # 4d. Build TrajectoryResults
        result = TrajectoryResults(
            source=source,
            instances=instances,  # type: ignore[arg-type]
            scores=dict(instances),
        )

        # 4e. If no trajectory data was fetched but we have an existing
        #     per-model JSON with data, reload from it instead of overwriting.
        traj_json_path = output_dir / "trajectories" / f"{model_id}.json"
        if result.total == 0 and traj_json_path.exists():
            reloaded = _reload_result(output_dir, {
                "model": model_name, "model_id": model_id,
                "path": f"trajectories/{model_id}.json",
            })
            if reloaded and reloaded.total > 0:
                logger.info(
                    f"  Reusing existing trajectory data ({reloaded.total} tasks)"
                )
                result = reloaded

        # Write per-model trajectory JSON only if we have new data
        if result.total > 0 and instances:
            write_json(traj_json_path, model_detail)

        # 4f. Accumulate
        all_results.append(result)
        all_lb_entries.append(_build_leaderboard_entry(submission_data))

        model_meta: JsonDict = {
            "model": model_name,
            "model_id": model_id,
            "folder_name": folder_name,
            "path": f"trajectories/{model_id}.json",
            "n_tasks": result.total,
            "raw_entry": submission_data,
        }
        all_model_meta.append(model_meta)
        all_folder_names.append(folder_name)

        logger.info(
            f"  Done: {model_name} ({model_id}), {result.total} tasks"
        )

    # -----------------------------------------------------------------------
    # 5. Write combined outputs
    # -----------------------------------------------------------------------
    # Only include models with actual trajectory data in the matrix CSV.
    # Models without trajectories would appear as all-zeros (indistinguishable
    # from genuine 0 scores), breaking downstream NaN-based filtering.
    results_with_data = [r for r in all_results if r.total > 0]
    if results_with_data:
        write_matrix_csv(output_dir, results_with_data)
        logger.info(
            f"Wrote trajectory.csv ({len(results_with_data)} models with data, "
            f"{len(all_results) - len(results_with_data)} skipped)"
        )

    if all_lb_entries:
        write_leaderboard_csv(output_dir, all_lb_entries)
        logger.info(f"Wrote leaderboard.csv ({len(all_lb_entries)} entries)")

    # Build index (exclude raw_entry from stored metadata to keep index small)
    index_models = []
    for meta in all_model_meta:
        index_models.append({
            "model": meta["model"],
            "model_id": meta["model_id"],
            "folder_name": meta.get("folder_name", ""),
            "path": meta["path"],
            "n_tasks": meta.get("n_tasks", 0),
            "raw_entry": meta.get("raw_entry", {}),
        })

    write_json(output_dir / "index.json", {
        "benchmark": "tau2-bench",
        "repo": TAU2_REPO,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "domains": DOMAINS,
        "counts": {
            "models": len(all_results),
            "failures": len(failures),
        },
        "models": index_models,
        "failures": failures,
    })
    logger.info(
        f"Wrote index.json ({len(all_results)} models, {len(failures)} failures)"
    )

    if failures:
        logger.warning(f"Failures: {len(failures)}")
        for f in failures:
            logger.warning(f"  {f['folder']}: {f['error']}")

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
