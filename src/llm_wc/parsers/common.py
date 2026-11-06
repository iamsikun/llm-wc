"""
Shared utilities for benchmark parser scripts.

Provides common I/O helpers, slug generation, output writers, and
SWE-bench-specific constants/functions used across all parsers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, TypeAlias
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
JsonDict: TypeAlias = dict[str, Any]
JsonList: TypeAlias = list[Any]
JsonValue: TypeAlias = Any

# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------

def slugify(value: str) -> str:
    """
    Normalize model names into safe, stable, URL-friendly IDs.

    Example: "Claude 3.5 Sonnet" -> "claude-3-5-sonnet"
    """
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "model"


def make_unique_slug(base_slug: str, used: set[str]) -> str:
    """Return a unique slug by appending numeric suffixes when needed."""
    if base_slug not in used:
        used.add(base_slug)
        return base_slug

    idx = 2
    while True:
        candidate = f"{base_slug}-{idx}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        idx += 1


# ---------------------------------------------------------------------------
# JSON / CSV I/O
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: JsonValue) -> None:
    """Write JSON to a file with pretty printing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write list of dicts to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def coerce_csv_value(value: Any) -> str | int | float | None:
    """Format a value for CSV output."""
    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        return value
    return json.dumps(value, sort_keys=True)


# ---------------------------------------------------------------------------
# Standard output writers
# ---------------------------------------------------------------------------

@dataclass
class TrajectorySource:
    """Metadata about the source of a trajectory."""

    model_name: str
    model_id: str
    leaderboard_name: str
    trajectory_url: str | None = None
    raw_entry: JsonDict = field(default_factory=dict)
    result_files: list[str] = field(default_factory=list)
    result_path: str | None = None


@dataclass
class TrajectoryResults:
    """Parsed results for a single model."""

    source: TrajectorySource
    instances: dict[str, int]  # instance_id -> 1 (solved) or 0 (unsolved)
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    verdicts: dict[str, str | None] = field(default_factory=dict)

    @property
    def resolved(self) -> int:
        """Count of resolved instances."""
        return sum(self.instances.values())

    @property
    def total(self) -> int:
        """Total number of instances."""
        return len(self.instances)


def write_matrix_csv(output_dir: Path, results: list[TrajectoryResults]) -> None:
    """
    Write a dense CSV matrix where rows are models and columns are instances.
    Values are 1 (pass) or 0 (fail).
    """
    if not results:
        return

    all_ids: set[str] = set()
    for item in results:
        all_ids.update(item.instances.keys())
    instance_ids = sorted(all_ids)
    fieldnames = ["model", "model_id"] + instance_ids

    rows: list[dict[str, Any]] = []
    for item in results:
        row: dict[str, Any] = {
            "model": item.source.model_name,
            "model_id": item.source.model_id,
        }
        for instance_id in instance_ids:
            row[instance_id] = item.instances.get(instance_id, 0)
        rows.append(row)

    write_csv(output_dir / "trajectory.csv", rows, fieldnames)


def write_leaderboard_csv(output_dir: Path, entries: list[JsonDict]) -> None:
    """Write the raw leaderboard entries to a CSV file."""
    if not entries:
        return

    fieldnames: list[str] = []
    include_model = any("model" in entry or "name" in entry for entry in entries)
    if include_model:
        fieldnames.append("model")

    for entry in entries:
        for key in entry.keys():
            if key == "model" and "model" in fieldnames:
                continue
            if key not in fieldnames:
                fieldnames.append(key)

    rows: list[dict[str, Any]] = []
    for entry in entries:
        row = {key: coerce_csv_value(entry.get(key)) for key in fieldnames}
        if "model" in fieldnames:
            row["model"] = coerce_csv_value(
                entry.get("model") or entry.get("name") or extract_model_name(entry)
            )
        rows.append(row)

    write_csv(output_dir / "leaderboard.csv", rows, fieldnames)


# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------

def make_output_dir(output_parent: str | Path) -> Path:
    """
    Create a timestamped output directory under *output_parent*.

    Uses ``<YYYYMMDD>``; falls back to ``<YYYYMMDD_HHMMSS>`` if that date
    directory already exists.
    """
    run_at = datetime.now()
    output_parent = Path(output_parent)
    run_date = run_at.strftime("%Y%m%d")
    output_dir = output_parent / run_date
    if output_dir.exists():
        output_dir = output_parent / run_at.strftime("%Y%m%d_%H%M%S")
    return output_dir


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------

def load_json_records(path: Path) -> list[dict[str, Any]]:
    """Load records from JSONL / JSON / CSV as list[dict]."""
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
        return records

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    if isinstance(payload, dict):
        for key in ("judgments", "results", "data", "items", "rows"):
            value = payload.get(key)
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return value
    raise ValueError(f"{path}: unsupported file format for records.")


# ---------------------------------------------------------------------------
# Remote fetching (used by SWE-bench parsers)
# ---------------------------------------------------------------------------

def download_text(url: str, timeout: int = 30) -> str:
    """Download content from a URL and return it as a UTF-8 string."""
    logger.debug(f"Downloading: {url}")
    req = Request(url, headers={"User-Agent": "llm-wc-swebench-parser"})
    with urlopen(req, timeout=timeout) as response:
        payload = response.read()
    return payload.decode("utf-8")


def cache_path(cache_dir: Path, url: str) -> Path:
    """Generate a consistent cache file path for a given URL."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    suffix = Path(url).suffix or ".txt"
    return cache_dir / f"{digest}{suffix}"


def read_with_cache(url: str, cache_dir: Path | None) -> str:
    """
    Read content from a URL, using a local file cache if configured.
    """
    if cache_dir is None:
        return download_text(url)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_path(cache_dir, url)

    if cached.exists():
        logger.debug(f"Cache hit for {url} at {cached}")
        return cached.read_text(encoding="utf-8")

    text = download_text(url)
    cached.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# Payload parsing
# ---------------------------------------------------------------------------

def jsonl_loads(text: str) -> list[JsonDict]:
    """Parse line-delimited JSON (JSONL) payload."""
    records: list[JsonDict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def csv_loads(text: str) -> list[JsonDict]:
    """Parse CSV payload into a list of dictionaries."""
    reader = csv.DictReader(text.splitlines())
    return [row for row in reader]


def parse_any_payload(text: str, url: str) -> JsonValue:
    """Parse a text payload based on the URL extension or content heuristics."""
    suffix = Path(url.split("?", 1)[0]).suffix.lower()
    if suffix == ".jsonl":
        return jsonl_loads(text)
    if suffix == ".csv":
        return csv_loads(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return jsonl_loads(text)


def load_url(url: str, cache_dir: Path | None) -> JsonValue:
    """Download (with caching) and parse the file at *url*."""
    text = read_with_cache(url, cache_dir)
    return parse_any_payload(text, url)


# ---------------------------------------------------------------------------
# SWE-bench leaderboard constants & helpers
# ---------------------------------------------------------------------------

DEFAULT_LEADERBOARD_URL = (
    "https://raw.githubusercontent.com/SWE-bench/swe-bench.github.io"
    "/master/data/leaderboards.json"
)

CANDIDATE_TRAJ_FILES = (
    "traj.json",
    "traj.jsonl",
    "trajs.json",
    "trajs.jsonl",
    "trajectories.json",
    "trajectories.jsonl",
    "results.json",
    "results.jsonl",
    "results/results.json",
    "results/results.jsonl",
    "results/instance_results.jsonl",
    "results/instances.jsonl",
    "results/resolved.json",
    "results/resolved.jsonl",
    "resolved.json",
    "resolved.jsonl",
    "instance_results.jsonl",
    "instances.json",
    "analysis/instance_results.jsonl",
    "analysis/results.json",
    "analysis/instances.json",
)

INSTANCE_ID_KEYS = (
    "instance_id",
    "id",
    "task_id",
    "problem_id",
    "name",
)

SOLVED_KEYS = (
    "resolved",
    "success",
    "passed",
    "is_solved",
    "solved",
    "correct",
)

STATUS_KEYS = (
    "status",
    "result",
    "outcome",
)

STATUS_TRUE = {"resolved", "success", "passed", "pass", "correct", "fixed"}
STATUS_FALSE = {"failed", "fail", "error", "incorrect", "unresolved"}


class DownloadError(RuntimeError):
    """Raised when no candidate trajectory file could be downloaded."""


def normalize_experiments_base(experiments_base: str) -> str:
    """
    Normalize GitHub URLs to ensure they point to raw content.

    Converts 'github.com/.../tree/...' to 'raw.githubusercontent.com/...'.
    """
    base = experiments_base.rstrip("/")
    tree_prefix = "https://github.com/SWE-bench/experiments/tree/"
    if base.startswith(tree_prefix):
        suffix = base[len(tree_prefix):]
        return f"https://raw.githubusercontent.com/SWE-bench/experiments/{suffix}"
    return base


def pick_leaderboard(payload: JsonValue, leaderboard_name: str) -> JsonDict:
    """
    Traverse the JSON payload to find the specific leaderboard dictionary.
    """
    name_key = leaderboard_name.strip().lower()
    alt_keys = {name_key, name_key.replace(" ", "-"), name_key.replace("-", " ")}

    if isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            entry_name = str(entry.get("name", "")).strip().lower()
            if entry_name in alt_keys:
                return entry
        raise KeyError(
            f"No leaderboard named '{leaderboard_name}' found in list payload"
        )

    if isinstance(payload, dict):
        if "leaderboards" in payload and isinstance(payload["leaderboards"], list):
            return pick_leaderboard(payload["leaderboards"], leaderboard_name)

        for key, value in payload.items():
            key_norm = key.strip().lower()
            if key_norm in alt_keys and isinstance(value, dict):
                return value

        raise KeyError(
            f"No leaderboard named '{leaderboard_name}' found in dict payload"
        )

    raise TypeError(f"Unexpected leaderboards payload format: {type(payload)}")


def iter_entries(leaderboard: JsonDict) -> Iterable[JsonDict]:
    """Yield model entries from the leaderboard dictionary."""
    for key in ("entries", "models", "results", "rows"):
        value = leaderboard.get(key)
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, dict):
                    yield entry
            return
    raise KeyError(
        "Could not find leaderboard entries list. "
        "Checked keys: entries, models, results, rows"
    )


def extract_model_name(entry: JsonDict) -> str:
    """Extract a human-readable model name from a leaderboard entry."""
    for key in ("model", "name", "submission", "agent", "system"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown-model"


def get_entry_folder(entry: JsonDict) -> str | None:
    """Return the normalized folder/run name from the entry."""
    folder = entry.get("folder") or entry.get("path") or entry.get("run")
    if not isinstance(folder, str) or not folder.strip():
        return None
    return folder.strip().strip("/")


# ---------------------------------------------------------------------------
# SWE-bench instance extraction
# ---------------------------------------------------------------------------

def _get_first_key(record: JsonDict, keys: tuple[str, ...]) -> Any:
    """Return the value of the first key found in the record."""
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _normalize_status(value: str) -> bool | None:
    """Convert status string to boolean resolved state."""
    normalized = value.strip().lower()
    if normalized in STATUS_TRUE:
        return True
    if normalized in STATUS_FALSE:
        return False
    return None


def _get_nested_resolved(record: JsonDict) -> bool | None:
    """Check for resolved status in nested metadata or scores."""
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return None
    scores = metadata.get("scores")
    if not isinstance(scores, dict):
        return None
    value = scores.get("resolved")
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return _normalize_status(value)
    return None


def _get_solved_value(record: JsonDict) -> bool | None:
    """Determine if a record represents a solved instance."""
    for key in SOLVED_KEYS:
        if key in record:
            value = record.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value > 0
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in STATUS_TRUE:
                    return True
                if normalized in STATUS_FALSE:
                    return False

    nested = _get_nested_resolved(record)
    if nested is not None:
        return nested

    for key in STATUS_KEYS:
        if key in record and isinstance(record[key], str):
            normalized = record[key].strip().lower()
            return _normalize_status(normalized)

    return None


def _extract_from_records(records: list[Any]) -> dict[str, int]:
    """Extract results from a list of record dictionaries."""
    results: dict[str, int] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        instance_id = _get_first_key(record, INSTANCE_ID_KEYS)
        if not instance_id:
            continue
        instance_id = str(instance_id)
        solved = _get_solved_value(record)
        if solved is None:
            continue
        results[instance_id] = 1 if solved else 0

    if not results:
        raise ValueError("No instance-level results found in records")
    return results


def _extract_from_mapping(mapping: dict[str, Any]) -> dict[str, int]:
    """Extract results from a dictionary keyed by instance ID."""
    results: dict[str, int] = {}
    for instance_id, record in mapping.items():
        if isinstance(record, dict):
            solved = _get_solved_value(record)
            if solved is None:
                continue
            results[str(instance_id)] = 1 if solved else 0
        elif isinstance(record, bool):
            results[str(instance_id)] = 1 if record else 0
        elif isinstance(record, (int, float)):
            results[str(instance_id)] = 1 if record > 0 else 0
        elif isinstance(record, str):
            solved = _normalize_status(record)
            if solved is None:
                continue
            results[str(instance_id)] = 1 if solved else 0
    return results


def extract_instance_results(payload: JsonValue) -> dict[str, int]:
    """
    Normalize the parsed payload into a dictionary of {instance_id: resolved_status}.

    resolved_status is 1 for Solved, 0 for Unsolved.
    """
    if isinstance(payload, list):
        return _extract_from_records(payload)

    if isinstance(payload, dict):
        for key in ("instances", "results", "items", "data", "predictions"):
            value = payload.get(key)
            if isinstance(value, list):
                return _extract_from_records(value)
            if isinstance(value, dict):
                extracted = _extract_from_mapping(value)
                if extracted:
                    return extracted

        resolved_list = payload.get("resolved_instances") or payload.get("resolved")
        unresolved_list = payload.get("unresolved_instances") or payload.get(
            "unresolved"
        )

        if isinstance(resolved_list, list):
            instances: dict[str, int] = {str(item): 1 for item in resolved_list}
            if isinstance(unresolved_list, list):
                instances.update({str(item): 0 for item in unresolved_list})
            return instances

        instance_map = payload.get("instance_results") or payload.get("results")
        if isinstance(instance_map, dict):
            extracted = _extract_from_mapping(instance_map)
            if extracted:
                return extracted

    raise ValueError(
        "Unsupported trajectory payload structure: could not identify instance results"
    )


def _extract_traj_url(entry: JsonDict) -> str | None:
    """Identify the most likely trajectory URL from the entry's fields."""
    folder = get_entry_folder(entry)

    def is_candidate(k: str, v: Any) -> bool:
        if not isinstance(v, str):
            return False
        lower_k = k.lower()
        if "traj" not in lower_k and "trajectory" not in lower_k:
            return False
        if not v.startswith(("http://", "https://")):
            return False
        if "docent" in lower_k:
            return False
        if folder and folder not in v:
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
    """Construct a list of possible URLs for the trajectory file."""
    folder = get_entry_folder(entry)
    if not folder:
        return []

    experiments_base = normalize_experiments_base(experiments_base)
    return [
        f"{experiments_base}/{folder}/{filename}" for filename in CANDIDATE_TRAJ_FILES
    ]


def resolve_trajectory(
    entry: JsonDict,
    experiments_base: str,
    cache_dir: Path | None,
    model_name: str,
) -> str:
    """Determine the final URL for the trajectory file."""
    traj_url = _extract_traj_url(entry)

    if traj_url:
        return traj_url

    candidate_urls = _build_candidate_urls(entry, experiments_base)
    if not candidate_urls:
        raise DownloadError(
            f"No trajectory URL found and no folder info for model '{model_name}'"
        )

    logger.info(f"Scanning {len(candidate_urls)} candidate URLs for {model_name}...")

    for url in candidate_urls:
        try:
            read_with_cache(url, cache_dir)
        except (HTTPError, URLError):
            continue
        else:
            return url

    raise DownloadError("No candidate trajectory URL was accessible.")


# ---------------------------------------------------------------------------
# History snapshot resolution
# ---------------------------------------------------------------------------

DEFAULT_HISTORY_ROOT = "resources/swe_bench_history"


def get_leaderboard_dir(history_root: str | Path, leaderboard: str) -> Path:
    """Return the leaderboard directory containing ``history_index.json``.

    Layout: ``history_root/<leaderboard>/history_index.json``
    """
    history_root = Path(history_root)
    leaderboard_root = history_root / leaderboard

    if not leaderboard_root.is_dir():
        raise FileNotFoundError(
            f"No history directory for leaderboard '{leaderboard}' at {leaderboard_root}"
        )

    index_path = leaderboard_root / "history_index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"No history_index.json in {leaderboard_root}"
        )

    return leaderboard_root


def get_latest_snapshot_dir(history_root: str | Path, leaderboard: str) -> Path:
    """Return the most recent snapshot directory for *leaderboard*.

    Snapshots are sorted chronologically by their directory name
    (``<ISO-timestamp>_<short-sha>``).  Returns the last one.
    """
    run_dir = get_leaderboard_dir(history_root, leaderboard)
    snapshots_root = run_dir / "snapshots"
    if not snapshots_root.is_dir():
        raise FileNotFoundError(f"No snapshots directory in {run_dir}")
    snapshot_dirs = sorted(
        d for d in snapshots_root.iterdir() if d.is_dir()
    )
    if not snapshot_dirs:
        raise FileNotFoundError(f"No snapshot directories in {snapshots_root}")
    return snapshot_dirs[-1]


def get_snapshot_at_date(
    history_root: str | Path,
    leaderboard: str,
    target_date: str,
) -> Path:
    """Return the latest snapshot committed on or before *target_date*.

    Parameters
    ----------
    target_date : str
        ISO date string (``YYYY-MM-DD``).  The snapshot whose leaderboard
        commit timestamp is closest to (but not after) this date is returned.
    """
    run_dir = get_leaderboard_dir(history_root, leaderboard)
    index_path = run_dir / "history_index.json"
    with open(index_path, encoding="utf-8") as fh:
        index = json.load(fh)

    snapshots_meta = index.get("snapshots", [])
    if not snapshots_meta:
        raise FileNotFoundError(f"No snapshots in {index_path}")

    best: dict[str, Any] | None = None
    for meta in snapshots_meta:
        commit = meta.get("leaderboard_commit", {})
        commit_date = commit.get("date", "")[:10]
        if commit_date and commit_date <= target_date:
            if best is None or commit_date > best.get("_date", ""):
                best = {**meta, "_date": commit_date}

    if best is None:
        raise FileNotFoundError(
            f"No snapshot on or before {target_date} for leaderboard '{leaderboard}'"
        )

    snapshot_dir = run_dir / "snapshots" / best["snapshot_id"]
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"Snapshot directory not found: {snapshot_dir}")
    return snapshot_dir


def get_latest_trajectory_csv(history_root: str | Path, leaderboard: str) -> Path:
    """Shorthand: ``get_latest_snapshot_dir(...) / 'trajectory.csv'``."""
    return get_latest_snapshot_dir(history_root, leaderboard) / "trajectory.csv"


def get_trajectory_csv_at_date(
    history_root: str | Path,
    leaderboard: str,
    target_date: str,
) -> Path:
    """Shorthand: ``get_snapshot_at_date(...) / 'trajectory.csv'``."""
    return get_snapshot_at_date(history_root, leaderboard, target_date) / "trajectory.csv"


def extract_instances_from_entry(
    entry: JsonDict,
    model_name: str,
    experiments_base: str,
    cache_dir: Path | None,
) -> tuple[dict[str, int] | None, str | None, str | None]:
    """
    Main logic to extract instance results for a single SWE-bench entry.

    Returns:
        (instances_dict, trajectory_url, error_message)
    """
    instances = None
    traj_url = None
    error = None
    folder = get_entry_folder(entry)

    details = entry.get("per_instance_details")
    if isinstance(details, dict):
        try:
            instances = _extract_from_mapping(details)
            if not instances:
                instances = None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Failed to parse per_instance_details for {model_name}: {exc}"
            )
            instances = None

        if instances is not None:
            traj_url = _extract_traj_url(entry)
            if not traj_url:
                for key in ("trajs", "logs", "trajectory", "traj_url"):
                    value = entry.get(key)
                    if isinstance(value, str) and value.startswith(
                        ("http://", "https://")
                    ):
                        if folder and folder not in value:
                            continue
                        traj_url = value
                        break
            if not traj_url:
                traj_url = "leaderboard"

    if instances is None:
        try:
            traj_url = resolve_trajectory(
                entry, experiments_base, cache_dir, model_name
            )
            logger.info(f"Resolved trajectory for {model_name} at {traj_url}")
            payload = load_url(traj_url, cache_dir)
            instances = extract_instance_results(payload)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            logger.warning(f"Failed to process {model_name}: {exc}")

    return instances, traj_url, error
