"""Build the static site into ``_site/``.

Inputs (all read only): ``tracker/registry.yaml``, the reference edition in
``results/ruler_audit``, live reports in ``tracker/results``, history CSVs,
``tracker/status.json``, and the templates and
assets in ``tracker/site``. Output is plain HTML, CSS, JS, JSON, and CSV with
relative links, so it works on localhost and under a GitHub Pages subpath.

After writing, the build checks that the reference edition matches
``results/ruler_audit`` exactly, that every internal link resolves, that
public text never says "moon", and that pages stay small.
"""
from __future__ import annotations

import functools
import hashlib
import http.server
import json
import re
import shutil
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from llm_wc.tracker import registry
from llm_wc.tracker.analyze import RPM_DECIMALS, latest_live_report, reference_tables, report
from llm_wc.tracker.discovery.criteria import MIN_MODELS, MIN_TASKS
from llm_wc.tracker import pairwise
from llm_wc.tracker.history import leader_changes, read_history
from llm_wc.tracker.paths import BUILD, REFERENCE_AUDIT, SITE_SRC
from llm_wc.tracker.sources import SOURCES
from llm_wc.tracker.site import share
from llm_wc.tracker.site.texmath import tex
from llm_wc.tracker.update import load_status

MAX_PAGE_BYTES = 2_000_000


def _fmt_date(value: str | None) -> str:
    if not value:
        return ""
    s = str(value)
    m = re.match(r"(\d{4})-?(\d{2})-?(\d{2})", s)
    if not m:
        return s
    return date(int(m[1]), int(m[2]), int(m[3])).strftime("%b %-d, %Y")


def _edition_meta(doc: dict) -> dict:
    s, top = doc["summary"], doc["models"][0]
    tier_rows = [t for t in doc["tiers"] if str(t["rank_mass"]) == "80"]
    up = doc.get("upstream") or {}
    return {
        "edition": doc["edition"], "snapshot_id": doc["snapshot_id"],
        "date": _fmt_date(up.get("date") or doc["snapshot_id"]),
        "n_tasks": doc["n_tasks"], "n_models": doc["n_models"], "m": doc["params"]["m"],
        "leader": top["model"], "leader_raw": top["raw_score_pct"], "leader_rank": top["rank"],
        "leader_tied": top["rank_start"] != top["rank_end"],
        "leader_adjustment": s["leader_adjustment_pp"],
        "leader_adjusted": top["raw_score_pct"] - s["leader_adjustment_pp"],
        "position_1_adjustment": s["position_1_adjustment_pp"],
        "leader_p_rank_1": s["leader_p_rank_1"],
        "tiers_80": s["tiers_80"], "tier_1_size_80": s["tier_1_size_80"],
        "tier_1_interval": [tier_rows[0]["common_rank_low"], tier_rows[0]["common_rank_high"]] if tier_rows else None,
        "delta_pp": doc["delta_pp"],
    }


def _reference_upstream(source: str, cache: dict) -> dict:
    if source not in cache:
        snap = SOURCES[source].reference_snapshot()
        cache[source] = snap.upstream if snap is not None else {}
    return cache[source]


def _write_json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(",", ":"), ensure_ascii=False, allow_nan=False))


def _leaderboard_csv(doc: dict) -> str:
    return pd.DataFrame(doc["models"]).to_csv(index=False)


def _rpm_csv(doc: dict) -> str:
    ids = [m["model_id"] for m in doc["models"]]
    frame = pd.DataFrame(doc["rpm"], index=pd.Index(ids, name="model_id"),
                         columns=[str(k) for k in range(1, len(ids) + 1)])
    return frame.to_csv()


def collect(out: Path) -> dict:
    """Write per-setting data files and return the template context."""
    reg = registry.registry()
    status = load_status()
    reference_up: dict = {}
    benchmarks, all_docs = [], {}
    for key, spec in reg["settings"].items():
        src = SOURCES[spec["source"]]
        editions, docs = [], {}
        live = latest_live_report(key)
        if live is not None:
            docs["live"] = live
        if spec.get("in_reference"):
            docs["reference"] = report(reference_tables(key), key=key, edition="reference", snapshot_id=src.reference_snapshot_id,
                                   source=spec["source"], upstream=_reference_upstream(spec["source"], reference_up))
        for name, doc in docs.items():
            base = out / "data" / key
            _write_json(base / f"{name}.json", doc)
            if name == "reference":
                shutil.copy(REFERENCE_AUDIT / key / "models.csv", base / "reference-leaderboard.csv")
                shutil.copy(REFERENCE_AUDIT / key / "rpm.csv", base / "reference-rpm.csv")
                shutil.copy(REFERENCE_AUDIT / key / "tiers.csv", base / "reference-tiers.csv")
            else:
                (base / f"{name}-leaderboard.csv").write_text(_leaderboard_csv(doc))
                (base / f"{name}-rpm.csv").write_text(_rpm_csv(doc))
                pd.DataFrame(doc["tiers"]).to_csv(base / f"{name}-tiers.csv", index=False)
            pw = pairwise.read(key, doc["snapshot_id"])
            if pw is not None:
                _write_json(base / f"{name}-pairwise.json", pw)
            editions.append({**_edition_meta(doc), "json": f"data/{key}/{name}.json",
                             "csv": f"data/{key}/{name}-leaderboard.csv", "rpm_csv": f"data/{key}/{name}-rpm.csv",
                             "tiers_csv": f"data/{key}/{name}-tiers.csv", "upstream": doc.get("upstream") or {},
                             "pairwise": f"data/{key}/{name}-pairwise.json" if pw is not None else None})
        if not editions:
            continue
        all_docs[key] = docs
        hist = read_history(key)
        hist = hist[hist["n_models"] >= 2] if len(hist) else hist
        history = json.loads(hist.to_json(orient="records")) if len(hist) else []
        if history:
            _write_json(out / "data" / key / "history.json", history)
        st = status.get(spec["source"], {})
        benchmarks.append({
            "key": key, **{k: spec.get(k) for k in ("label", "short_label", "family", "summary", "description",
                                                     "metric", "task_population", "resampling_unit",
                                                     "overview_url", "limitations", "in_reference", "stratum_type")},
            "source": spec["source"], "source_meta": reg["sources"].get(spec["source"], {}),
            "editions": editions, "default": editions[0], "history_points": len(history),
            "status": st,
        })
    featured = reg["site"].get("featured")
    story = leader_changes(featured) if featured in all_docs else None
    if story and not story["changes"]:
        story = None
    return {"site": reg["site"], "analysis": reg["analysis"], "benchmarks": benchmarks, "docs": all_docs,
            "story": story,
            "registry_sources": list(reg["sources"].items()),
            "criteria": {"min_tasks": MIN_TASKS, "min_models": MIN_MODELS}}


def _summary_payload(ctx: dict) -> list[dict]:
    """Small cross-benchmark payload embedded in the overview pages."""
    rows = []
    for b in ctx["benchmarks"]:
        row = {k: b[k] for k in ("key", "label", "short_label", "family", "in_reference", "history_points")}
        row["editions"] = {e["edition"]: {k: v for k, v in e.items() if k != "upstream"} for e in b["editions"]}
        rows.append(row)
    return rows


def _env() -> Environment:
    env = Environment(loader=FileSystemLoader(SITE_SRC / "templates"), autoescape=select_autoescape(["html"]),
                      trim_blocks=True, lstrip_blocks=True)
    env.filters["pp"] = lambda v, d=2: f"{v:.{d}f}".replace("-", "\u2212")  # true minus sign
    env.filters["pct"] = lambda v, d=1: f"{v:.{d}f}"
    env.filters["num"] = lambda v: f"{int(v):,}"
    # JSON for inline <script> blocks: "<" is escaped so data can never close the tag.
    env.filters["tojson_compact"] = lambda v: Markup(
        json.dumps(v, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c"))
    env.filters["date"] = _fmt_date
    env.filters["tex"] = tex
    env.filters["month"] = lambda v: date.fromisoformat(str(v)[:10]).strftime("%B %Y") if v else ""
    return env


def _share_assets(out: Path, ctx: dict) -> None:
    """Link-preview images (og/<key>.png, og/home.png) and badges (badges/<key>.svg)."""
    site = ctx["site"]
    for b in ctx["benchmarks"]:
        d = ctx["docs"][b["key"]].get("live") or ctx["docs"][b["key"]]["reference"]
        share.og_image(d, b["label"], out / "og" / f"{b['key']}.png", site.get("headline") or site["title"],
                       site["base_url"] + f"benchmarks/{b['key']}.html")
        (out / "badges").mkdir(exist_ok=True)
        (out / "badges" / f"{b['key']}.svg").write_text(share.badge_svg(d))
    featured = site.get("featured")
    if (out / "og" / f"{featured}.png").is_file():
        shutil.copy(out / "og" / f"{featured}.png", out / "og" / "home.png")


PAGES = [
    ("index.html", "index.html", "home"),
    ("benchmarks.html", "benchmarks.html", "benchmarks"),
    ("methods.html", "methods.html", "methods"),
    ("design.html", "design.html", "design"),
    ("about.html", "about.html", "about"),
    ("404.html", "404.html", ""),
]


def build(out: str | Path | None = None) -> Path:
    out = Path(out) if out else BUILD
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(SITE_SRC / "static", out / "static")
    ctx = collect(out)
    summary = _summary_payload(ctx)
    ref = [b for b in ctx["benchmarks"] if b["in_reference"]]
    pe = [next(e for e in b["editions"] if e["edition"] == "reference") for b in ref]
    headline = {
        "n_settings": len(ref),
        "pos1_min": min(e["position_1_adjustment"] for e in pe),
        "pos1_max": max(e["position_1_adjustment"] for e in pe),
        "over_1pp": sum(e["position_1_adjustment"] > 1 for e in pe),
        "multi_tier1": sum(e["tier_1_size_80"] > 1 for e in pe),
        "families": len({b["family"] for b in ref}),
    }
    digest = hashlib.sha256(b"".join(f.read_bytes() for f in sorted((SITE_SRC / "static").iterdir()))).hexdigest()
    common = {**ctx, "summary": summary, "headline": headline, "build_date": date.today().isoformat(), "v": digest[:10],
              "build_id": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
              "n_live": sum(any(e["edition"] == "live" for e in b["editions"]) for b in ctx["benchmarks"])}
    env = _env()
    for template, target, nav in PAGES:
        html = env.get_template(template).render(**common, root="", nav=nav, page=target)
        (out / target).write_text(html)
    tmpl = env.get_template("benchmark.html")
    for b in ctx["benchmarks"]:
        html = tmpl.render(**common, b=b, root="../", nav="benchmarks", page=f"benchmarks/{b['key']}.html",
                           og_path=f"og/{b['key']}.png")
        (out / "benchmarks").mkdir(exist_ok=True)
        (out / "benchmarks" / f"{b['key']}.html").write_text(html)
    (out / ".nojekyll").write_text("")
    _share_assets(out, ctx)
    problems = check(out, ctx)
    if problems:
        raise SystemExit("Site checks failed:\n  " + "\n  ".join(problems))
    print(f"Built {sum(1 for _ in out.rglob('*.html'))} pages into {out} (all checks passed)")
    return out


# --- Checks -------------------------------------------------------------------


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[str] = []
        self.text: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        for name, value in attrs:
            if name in ("href", "src") and value:
                self.links.append(value)

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.text.append(data)


def check(out: Path, ctx: dict) -> list[str]:
    problems = []
    # The reference edition equals results/ruler_audit.
    for b in ctx["benchmarks"]:
        if not b["in_reference"]:
            continue
        doc = json.loads((out / "data" / b["key"] / "reference.json").read_text())
        t = reference_tables(b["key"])
        ours = pd.DataFrame(doc["models"])[list(t["models"].columns)]
        try:
            pd.testing.assert_frame_equal(ours, t["models"], check_exact=True, check_dtype=False)
        except AssertionError as exc:
            problems.append(f"{b['key']}: reference leaderboard differs ({str(exc).splitlines()[0]})")
        if np.max(np.abs(np.array(doc["rpm"]) - t["rpm"])) > 0.5 * 10 ** -RPM_DECIMALS + 1e-15:
            problems.append(f"{b['key']}: reference RPM differs beyond rounding")
        for k, v in t["summary"].items():
            if doc["summary"].get(k) != v:
                problems.append(f"{b['key']}: summary.{k} differs")
    # D4 and D5: links, wording, size.
    for page in out.rglob("*.html"):
        parser = _Links()
        html = page.read_text()
        parser.feed(html)
        for link in parser.links:
            if re.match(r"^(https?:|mailto:|#|data:|javascript:)", link):
                continue
            target = (page.parent / link.split("#")[0].split("?")[0]).resolve()
            if not target.exists():
                problems.append(f"{page.relative_to(out)}: broken link {link}")
        if re.search(r"\bmoon\b", " ".join(parser.text), re.IGNORECASE):
            problems.append(f"{page.relative_to(out)}: says 'moon'")
        if len(html.encode()) > MAX_PAGE_BYTES:
            problems.append(f"{page.relative_to(out)}: page larger than {MAX_PAGE_BYTES:,} bytes")
    for data in (out / "data").rglob("*.json"):
        if data.stat().st_size > MAX_PAGE_BYTES:
            problems.append(f"{data.relative_to(out)}: larger than {MAX_PAGE_BYTES:,} bytes")
    return problems


def serve(out: str | Path | None = None, port: int = 8000) -> None:
    out = Path(out) if out else BUILD
    if not (out / "index.html").is_file():
        build(out)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"Serving {out} at http://127.0.0.1:{port}/", flush=True)
        httpd.serve_forever()
