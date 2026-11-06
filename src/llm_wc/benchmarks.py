"""Task-by-model score matrices for the 14 reference leaderboard settings.

Every setting is read from a frozen snapshot under ``resources/``:

- SWE-bench Bash-only / Verified / Multimodal / Multilingual: leaderboard
  commit ``7c4289f`` (snapshot ``SWE_SNAPSHOT``). Multimodal's result files
  list only resolved issues, so its matrix uses the official test split
  (``parsers.swebench_tasks``) with unlisted issues scored 0. Multilingual
  drops one malformed upstream ID.
- SWE-bench Pro: ``resources/swe_bench_pro/20260305``. Several systems lack
  records for some tasks, so the matrix keeps the tasks every system has.
- Tau2-Bench airline / retail / telecom: ``resources/tau2_bench``.
- AlpacaEval 2, WildBench v2, LiveBench (pooled and three categories):
  ``resources/external`` (checksummed; see its README).

Usage::

    from llm_wc.benchmarks import SETTINGS, load_settings
    data = load_settings()            # {key: Setting}
    obs = data["swebench_verified"].obs   # (N tasks, J models)
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from llm_wc.parsers.swebench_tasks import load_swebench_task_ids

RESOURCES = Path(__file__).resolve().parents[2] / "resources"
EXTERNAL_DIR = RESOURCES / "external"
SWE_SNAPSHOT = "20260227T165949Z_7c4289f"
SWE_PRO_DIR = RESOURCES / "swe_bench_pro" / "20260305"
TAU2_CSV = RESOURCES / "tau2_bench" / "trajectory.csv"

SWE_VARIANTS = ("bash-only", "verified", "multimodal", "multilingual")
# Upstream task IDs that are not SWE-bench issues
SWE_EXCLUDED_TASKS = {"multilingual": ("mrdoob__three",)}
TAU2_DOMAINS = ("airline", "retail", "telecom")
LIVEBENCH_CATEGORIES = ("coding", "language", "instruction_following")

# Reporting order.
SETTINGS = (
    "swebench_bash_only",
    "swebench_verified",
    "swebench_multimodal",
    "swebench_multilingual",
    "swebench_pro",
    "tau2_airline",
    "tau2_retail",
    "tau2_telecom",
    "alpaca_eval_2",
    "wildbench_v2",
    "livebench_pooled",
    "livebench_coding",
    "livebench_language",
    "livebench_instruction_following",
)

# Short labels for tables.
SHORT_LABELS = {
    "swebench_bash_only": "SWE Bash-only",
    "swebench_verified": "SWE Verified",
    "swebench_multimodal": "SWE Multimodal",
    "swebench_multilingual": "SWE Multilingual",
    "swebench_pro": "SWE Pro",
    "tau2_airline": "Tau2 Airline",
    "tau2_retail": "Tau2 Retail",
    "tau2_telecom": "Tau2 Telecom",
    "alpaca_eval_2": "AlpacaEval 2",
    "wildbench_v2": "WildBench v2",
    "livebench_pooled": "LiveBench pooled",
    "livebench_coding": "LiveBench Coding",
    "livebench_language": "LiveBench Language",
    "livebench_instruction_following": "LiveBench Instr.",
}

_HASH_SUFFIX_RE = re.compile(r"-[0-9a-f]{40}(?:-|$)")


@dataclass(frozen=True)
class Setting:
    """One leaderboard setting.

    Attributes:
        key: setting key (see ``SETTINGS``).
        obs: (N, J) task-level scores in [0, 1].
        model_ids: (J,) stable model identifiers.
        model_names: (J,) leaderboard display names.
        task_ids: (N,) task identifiers.
        strata: (N,) labels balanced by the half-split validation
            ("all" when the split is unstratified).
        stratum_type: description of ``strata``.
        clusters: (N,) labels for the dependence stress tests (SWE-bench
            repositories, Tau2 telecom subcategories), else None.
    """

    key: str
    obs: np.ndarray
    model_ids: np.ndarray
    model_names: np.ndarray
    task_ids: np.ndarray
    strata: np.ndarray
    stratum_type: str
    clusters: np.ndarray | None = None


def swe_trajectory_csv(variant: str) -> Path:
    return RESOURCES / "swe_bench_history" / variant / "snapshots" / SWE_SNAPSHOT / "trajectory.csv"


def extract_repo(task_id: str) -> str:
    """Repository name from classic (``owner__repo-123``) and Pro task IDs."""
    clean = task_id.removeprefix("instance_")
    if "__" not in clean:
        return clean.rsplit("-", 1)[0]
    owner, rest = clean.split("__", 1)
    match = _HASH_SUFFIX_RE.search(rest)
    repo = rest[: match.start()] if match else rest.rsplit("-", 1)[0]
    return f"{owner}__{repo}"


def _load_swe(key: str, variant: str) -> Setting:
    path = swe_trajectory_csv(variant)
    frame = pd.read_csv(path).drop(columns=list(SWE_EXCLUDED_TASKS.get(variant, ())))
    official = load_swebench_task_ids(variant)
    if official is not None:
        listed = set(frame.columns) - {"model", "model_id"}
        if not listed <= set(official):
            raise ValueError(f"{path} lists tasks outside the official {variant} split")
        frame = frame.reindex(columns=["model", "model_id", *official], fill_value=0)
    return _swe_setting(key, frame, path)


def _load_swe_pro(key: str) -> Setting:
    """SWE-bench Pro on the tasks for which every system has a released record."""
    frame = pd.read_csv(SWE_PRO_DIR / "trajectory.csv")[["model", "model_id"]]
    records = {m: json.loads((SWE_PRO_DIR / "trajectories" / f"{m}.json").read_text())["instances"]
               for m in frame.model_id}
    common = sorted(set.intersection(*(set(r) for r in records.values())))
    scores = pd.DataFrame([[records[m][t] for t in common] for m in frame.model_id], columns=common)
    return _swe_setting(key, pd.concat([frame, scores], axis=1), SWE_PRO_DIR)


def _swe_setting(key: str, frame: pd.DataFrame, path: Path) -> Setting:
    task_ids = [c for c in frame if c not in {"model", "model_id"}]
    values = frame[task_ids].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if np.isnan(values).any():
        raise ValueError(f"Missing values in {path}")
    repos = np.array([extract_repo(t) for t in task_ids])
    return Setting(
        key=key,
        obs=values.T,
        model_ids=frame["model_id"].astype(str).to_numpy(),
        model_names=frame["model"].astype(str).to_numpy(),
        task_ids=np.array(task_ids),
        strata=repos,
        stratum_type="repository",
        clusters=repos,
    )


def _load_tau2(domain: str) -> Setting:
    frame = pd.read_csv(TAU2_CSV)
    task_ids = [c for c in frame if c.startswith(domain + "-")]
    values = frame[task_ids].apply(pd.to_numeric, errors="coerce")
    complete = values.notna().all(axis=1)
    clusters = None
    if domain == "telecom":
        clusters = np.array([re.search(r"\[([^]]+)\]", t).group(1) for t in task_ids])
    return Setting(
        key=f"tau2_{domain}",
        obs=values.loc[complete].to_numpy(float).T,
        model_ids=frame.loc[complete, "model_id"].astype(str).to_numpy(),
        model_names=frame.loc[complete, "model"].astype(str).to_numpy(),
        task_ids=np.array(task_ids),
        strata=np.repeat("all", len(task_ids)),
        stratum_type="none",
        clusters=clusters,
    )


# --- External judge-scored benchmarks ----------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_external_manifest(data_dir: Path = EXTERNAL_DIR) -> dict[str, Any]:
    """Read ``source_manifest.json`` and verify every file's checksum."""
    manifest = json.loads((data_dir / "source_manifest.json").read_text(encoding="utf-8"))
    for filename, metadata in manifest["files"].items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256_file(path)
        if actual != metadata["sha256"]:
            raise ValueError(
                f"Checksum mismatch for {filename}: expected {metadata['sha256']}, got {actual}"
            )
    return manifest


def load_alpaca_eval(
    data_dir: Path = EXTERNAL_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """AlpacaEval 2 raw weighted win rate: preference - 1 in [0, 1].

    Returns (obs, model_ids, task_ids, metadata). Keeps models with all 805
    valid preferences; drops the NullModel control.
    """
    frame = pd.read_csv(data_dir / "alpaca_eval_2_annotations.csv.gz", float_precision="round_trip")
    records_by_model: dict[str, dict[str, float]] = {}
    dataset_by_id: dict[str, str] = {}
    excluded: list[dict[str, Any]] = []

    for model, records in frame.groupby("model", sort=True):
        if model == "NullModel":
            excluded.append({"model": model, "n_records": len(records), "reason": "explicit_adversarial_control"})
            continue
        if len(records) != 805:
            excluded.append({"model": model, "n_records": len(records), "reason": "record_count"})
            continue
        missing = int(records["preference"].isna().sum())
        if missing:
            excluded.append({"model": model, "n_records": len(records), "reason": "missing_preference",
                             "missing_preferences": missing})
            continue
        preferences = records["preference"].astype(float)
        invalid = sorted(preferences[(preferences < 1.0) | (preferences > 2.0)].unique())
        if invalid:
            excluded.append({"model": model, "n_records": len(records), "reason": "invalid_preference",
                             "invalid_preferences": [float(v) for v in invalid]})
            continue
        if records["task_id"].duplicated().any():
            raise ValueError(f"Duplicate AlpacaEval task for {model}")
        records_by_model[str(model)] = dict(
            zip(records["task_id"].astype(str), preferences.to_numpy(dtype=float) - 1.0)
        )
        for task_id, dataset in zip(records["task_id"], records["dataset"]):
            # Task alignment uses the hashed instruction; the source label is metadata only.
            dataset_by_id[str(task_id)] = str(dataset)

    common_tasks = set.intersection(*(set(o) for o in records_by_model.values()))
    if len(common_tasks) != 805:
        raise ValueError(f"Expected 805 shared AlpacaEval tasks, got {len(common_tasks)}")
    task_ids = sorted(common_tasks)
    model_ids = np.array(sorted(records_by_model), dtype=str)
    obs = np.array([[records_by_model[m][t] for m in model_ids] for t in task_ids], dtype=float)
    metadata = {
        "complete_models": len(model_ids),
        "tasks": len(task_ids),
        "excluded_incomplete_models": excluded,
        "dataset_counts": pd.Series([dataset_by_id[t] for t in task_ids]).value_counts().to_dict(),
    }
    return obs, model_ids, np.array(task_ids), metadata


def alpaca_source_strata(task_ids: np.ndarray, data_dir: Path = EXTERNAL_DIR) -> np.ndarray:
    """Source dataset of each AlpacaEval instruction (five strata)."""
    frame = pd.read_csv(data_dir / "alpaca_eval_2_annotations.csv.gz", usecols=["task_id", "dataset"])
    meta = (
        frame.drop_duplicates("task_id", keep="last")
        .assign(task_id=lambda d: d["task_id"].astype(str))
        .set_index("task_id")
    )
    if len(meta) != len(task_ids):
        raise ValueError(f"Expected {len(task_ids)} AlpacaEval tasks, found {len(meta)} metadata rows")
    return meta.loc[list(task_ids), "dataset"].astype(str).to_numpy()


def load_wildbench(
    data_dir: Path = EXTERNAL_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """WildBench v2 checklist score / 10 on the prompts shared by all models."""
    frame = pd.read_csv(data_dir / "wildbench_v2_scores.csv.gz", float_precision="round_trip")
    records_by_model: dict[str, dict[str, float]] = {}
    excluded: list[dict[str, Any]] = []
    for model, records in frame.groupby("model", sort=True):
        scores = pd.to_numeric(records["score"], errors="coerce")
        invalid = int((scores.isna() | (scores < 0.0) | (scores > 10.0)).sum())
        if invalid:
            excluded.append({"model": model, "n_records": len(records), "invalid_scores": invalid})
            continue
        if records["session_id"].duplicated().any():
            raise ValueError(f"Duplicate WildBench task for {model}")
        records_by_model[str(model)] = dict(
            zip(records["session_id"].astype(str), scores.to_numpy(dtype=float) / 10.0)
        )
    task_sets = [set(o) for o in records_by_model.values()]
    task_ids = sorted(set.intersection(*task_sets))
    model_ids = np.array(sorted(records_by_model), dtype=str)
    obs = np.array([[records_by_model[m][t] for m in model_ids] for t in task_ids], dtype=float)
    metadata = {
        "complete_models": len(model_ids),
        "tasks": len(task_ids),
        "per_model_task_count_min": min(map(len, task_sets)),
        "per_model_task_count_max": max(map(len, task_sets)),
        "excluded_models": excluded,
    }
    return obs, model_ids, np.array(task_ids), metadata


def load_livebench(
    data_dir: Path = EXTERNAL_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """LiveBench task-level accuracy on the complete model x task block.

    Among coverage thresholds, keeps the model set whose common tasks maximize
    the number of complete cells. Returns (obs, model_ids, task_ids,
    categories, metadata).
    """
    raw = pd.read_csv(data_dir / "livebench_model_judgment.csv.gz", float_precision="round_trip")
    frame = (
        raw.sort_values("tstamp")
        .drop_duplicates(["model", "question_id"], keep="last")
        .reset_index(drop=True)
    )
    counts = frame.groupby("model")["question_id"].nunique()
    candidates = []
    for cutoff in sorted(counts.unique()):
        models = counts[counts >= cutoff].index
        common = set.intersection(*(set(frame.loc[frame["model"] == m, "question_id"]) for m in models))
        candidates.append({"coverage_cutoff": int(cutoff), "n_tasks": len(common),
                           "n_cells": len(models) * len(common), "models": models, "tasks": common})
    best = max(candidates, key=lambda c: (c["n_cells"], c["n_tasks"]))
    selected = frame[frame["model"].isin(pd.Index(best["models"])) & frame["question_id"].isin(best["tasks"])]
    matrix = selected.pivot(index="question_id", columns="model", values="score").sort_index().sort_index(axis=1)
    if matrix.isna().any().any():
        raise ValueError("Selected LiveBench matrix is not complete")
    task_meta = (
        selected[["question_id", "category", "task"]]
        .drop_duplicates("question_id")
        .set_index("question_id")
        .loc[matrix.index]
    )
    metadata = {
        "coverage_cutoff": best["coverage_cutoff"],
        "complete_models": matrix.shape[1],
        "tasks": matrix.shape[0],
        "task_counts": task_meta.groupby(["category", "task"]).size().sort_index().to_dict(),
    }
    return (
        matrix.to_numpy(dtype=float),
        matrix.columns.to_numpy(dtype=str),
        matrix.index.to_numpy(dtype=str),
        task_meta["category"].to_numpy(dtype=str),
        metadata,
    )


def _external(key: str, obs, model_ids, task_ids, strata=None, stratum_type="none") -> Setting:
    return Setting(
        key=key,
        obs=obs,
        model_ids=model_ids,
        model_names=model_ids,
        task_ids=task_ids,
        strata=np.repeat("all", len(task_ids)) if strata is None else strata,
        stratum_type=stratum_type,
    )


def load_settings(keys: tuple[str, ...] | list[str] = SETTINGS) -> dict[str, Setting]:
    """Load the requested settings, in the given order."""
    unknown = set(keys) - set(SETTINGS)
    if unknown:
        raise KeyError(f"Unknown settings: {sorted(unknown)}")
    out: dict[str, Setting] = {}
    external = [k for k in keys if not k.startswith(("swebench_", "tau2_"))]
    if external:
        load_external_manifest()
    cache: dict[str, Any] = {}
    for key in keys:
        if key == "swebench_pro":
            out[key] = _load_swe_pro(key)
        elif key.startswith("swebench_"):
            out[key] = _load_swe(key, key.removeprefix("swebench_").replace("_", "-"))
        elif key.startswith("tau2_"):
            out[key] = _load_tau2(key.removeprefix("tau2_"))
        elif key == "alpaca_eval_2":
            obs, ids, tasks, _ = load_alpaca_eval()
            out[key] = _external(key, obs, ids, tasks, alpaca_source_strata(tasks), "source dataset")
        elif key == "wildbench_v2":
            obs, ids, tasks, _ = load_wildbench()
            out[key] = _external(key, obs, ids, tasks)
        else:
            if "livebench" not in cache:
                cache["livebench"] = load_livebench()
            obs, ids, tasks, categories, _ = cache["livebench"]
            if key == "livebench_pooled":
                out[key] = _external(key, obs, ids, tasks, categories, "category")
            else:
                mask = categories == key.removeprefix("livebench_")
                out[key] = _external(key, obs[mask], ids, tasks[mask])
    return out


def load_setting(key: str) -> Setting:
    return load_settings([key])[key]
