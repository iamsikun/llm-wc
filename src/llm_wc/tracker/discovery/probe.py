"""Load a score table from a file, URL, or Hugging Face dataset and assess it.

Long tables need a task column, a model column, and a score column; names
are guessed from common conventions unless given. Wide tables have one row
per model and one numeric column per task.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llm_wc.tracker.discovery.criteria import FeasibilityReport, assess

TASK_NAMES = ("task_id", "question_id", "instance_id", "problem_id", "sample_id", "doc_id", "example_id",
              "prompt_id", "session_id", "item_id", "task", "id")
MODEL_NAMES = ("model_id", "model", "model_name", "system", "agent", "submission", "model_name_or_path")
SCORE_NAMES = ("score", "correct", "is_correct", "resolved", "passed", "pass", "reward", "acc", "accuracy",
               "exact_match", "em", "preference", "label")
MAX_HF_ROWS = 2_000_000


def _guess(columns, names) -> str | None:
    lower = {c.lower(): c for c in columns}
    return next((lower[n] for n in names if n in lower), None)


def load_table(target: str, split: str | None = None, config: str | None = None) -> pd.DataFrame:
    if target.startswith("hf:"):
        from datasets import get_dataset_split_names, load_dataset

        name = target[3:]
        split = split or get_dataset_split_names(name, config)[0]
        return load_dataset(name, config, split=f"{split}[:{MAX_HF_ROWS}]").to_pandas()
    lower = target.lower()
    if lower.endswith((".parquet", ".pq")):
        return pd.read_parquet(target)
    if lower.endswith((".jsonl", ".jsonl.gz", ".ndjson")):
        return pd.read_json(target, lines=True)
    if lower.endswith((".json", ".json.gz")):
        if Path(target).is_file():
            data = json.loads(Path(target).read_text())
        else:
            return pd.read_json(target)
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [data])
        return pd.json_normalize(data)
    if lower.endswith((".tsv", ".tsv.gz")):
        return pd.read_csv(target, sep="\t")
    return pd.read_csv(target)


def to_long(frame: pd.DataFrame, task_col=None, model_col=None, score_col=None, layout="auto") -> pd.DataFrame:
    task_col = task_col or _guess(frame.columns, TASK_NAMES)
    model_col = model_col or _guess(frame.columns, MODEL_NAMES)
    score_col = score_col or _guess(frame.columns, SCORE_NAMES)
    if layout == "long" or (layout == "auto" and task_col and model_col and score_col and task_col != model_col):
        missing = [n for n, c in (("task", task_col), ("model", model_col), ("score", score_col)) if not c]
        if missing:
            raise ValueError(f"Cannot find {', '.join(missing)} column(s); pass --task-col/--model-col/--score-col")
        long = frame[[task_col, model_col, score_col]].copy()
        long.columns = ["task_id", "model_id", "score"]
    else:
        model_col = model_col or frame.columns[0]
        values = frame.drop(columns=[model_col]).apply(pd.to_numeric, errors="coerce")
        values = values.loc[:, values.notna().any()]
        values.index = frame[model_col].astype(str)
        long = values.stack().rename("score").rename_axis(["model_id", "task_id"]).reset_index()
    if long["score"].dtype == bool:
        long["score"] = long["score"].astype(float)
    long["task_id"] = long["task_id"].astype(str)
    long["model_id"] = long["model_id"].astype(str)
    return long


def probe(target: str, task_col=None, model_col=None, score_col=None, layout="auto", split=None,
          config=None) -> FeasibilityReport:
    frame = load_table(target, split=split, config=config)
    return assess(to_long(frame, task_col, model_col, score_col, layout), target=target)
