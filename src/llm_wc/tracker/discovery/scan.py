"""Search Hugging Face and GitHub for leaderboards that might release per-task scores.

The scan only proposes. New hits go to ``tracker/discovery_inbox.json`` for a
person to review with ``probe``; nothing is added to the registry or to
``tracker/candidates.yaml`` automatically.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.parse import quote

from llm_wc.tracker import registry
from llm_wc.tracker.paths import INBOX
from llm_wc.tracker.sources.base import Context, http_json

HF_QUERIES = ("leaderboard", "model_judgment", "eval results", "evaluation results", "per-sample",
              "details", "trajectories", "submissions")
NOISE_SUFFIXES = ("/requests", "/contents", "-requests", "_requests")  # submission queues and aggregate tables
GITHUB_QUERIES = ("leaderboard per-instance results", "benchmark leaderboard submissions results json",
                  "agent benchmark trajectories leaderboard", "llm leaderboard per task scores")


def _known() -> set[str]:
    known = set()
    for c in registry.candidates():
        known.update(str(x).lower() for x in (c.get("id"), c.get("url"), *c.get("aliases", [])) if x)
    for s in registry.registry()["sources"].values():
        known.update(str(x).lower() for x in (s.get("homepage"), s.get("upstream")) if x)
    if INBOX.is_file():
        known.update(x["id"].lower() for x in json.loads(INBOX.read_text()))
    return known


def scan_hf(limit: int) -> list[dict]:
    from huggingface_hub import HfApi

    api, hits = HfApi(), {}
    for q in HF_QUERIES:
        for d in api.list_datasets(search=q, limit=limit, sort="downloads"):
            hits.setdefault(d.id, {"id": d.id, "where": "huggingface", "url": f"https://huggingface.co/datasets/{d.id}",
                                   "downloads": getattr(d, "downloads", None), "query": q,
                                   "last_modified": str(getattr(d, "last_modified", "") or "")})
    return list(hits.values())


def scan_github(limit: int, ctx: Context) -> list[dict]:
    hits = {}
    for q in GITHUB_QUERIES:
        url = f"https://api.github.com/search/repositories?q={quote(q)}&sort=stars&per_page={min(limit, 100)}"
        for r in http_json(url, ctx, github=True).get("items", []):
            hits.setdefault(r["full_name"], {"id": r["full_name"], "where": "github", "url": r["html_url"],
                                             "stars": r["stargazers_count"], "query": q,
                                             "description": (r.get("description") or "")[:200],
                                             "last_modified": r.get("pushed_at")})
    return list(hits.values())


def scan(limit: int = 30, write: bool = True, ctx: Context | None = None) -> list[dict]:
    ctx = ctx or Context()
    found = []
    for name, fn in (("huggingface", lambda: scan_hf(limit)), ("github", lambda: scan_github(limit, ctx))):
        try:
            found += fn()
        except Exception as exc:  # noqa: BLE001 - one search backend failing should not stop the other
            print(f"{name} search failed: {exc}")
    known = _known()
    stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    new = [{**h, "found_at": stamp, "status": "to review"} for h in found
           if h["id"].lower() not in known and h["url"].lower() not in known
           and not h["id"].lower().endswith(NOISE_SUFFIXES)]
    if write and new:
        existing = json.loads(INBOX.read_text()) if INBOX.is_file() else []
        INBOX.write_text(json.dumps(existing + new, indent=2) + "\n")
    return new
