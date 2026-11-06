"""Command line: ``uv run python -m llm_wc.tracker <command>``.

  update      check upstream sources; store and analyze new snapshots
  history     add archive and reference rows to the per-setting history
  pairwise    "A ranks above B" shares for the top systems of each edition
  validate    check every stored snapshot and live report
  status      print the last update result for each source
  check-reference verify that the adapters rebuild the reference matrices and numbers
  build       write the static website to _site/
  serve       serve _site/ on localhost
  probe       feasibility report for a score table (file or hf:<dataset>)
  scan        search Hugging Face and GitHub for candidate benchmarks
"""
from __future__ import annotations

import argparse
import json
import sys


def _update(args) -> int:
    from llm_wc.tracker.sources import SOURCES
    from llm_wc.tracker.update import update

    keys = args.sources or list(SOURCES)
    unknown = set(keys) - set(SOURCES)
    if unknown:
        print(f"Unknown sources: {sorted(unknown)}", file=sys.stderr)
        return 2
    status = update(keys, force=args.force)
    return 1 if args.strict and any(status[k]["result"] == "failed" for k in keys) else 0


def _history(args) -> int:
    from llm_wc.tracker.history import build_history, build_tier_history

    build_history(args.settings)
    build_tier_history(args.settings)
    return 0


def _pairwise(args) -> int:
    from llm_wc.tracker.pairwise import build_pairwise

    build_pairwise(args.settings, force=args.force)
    return 0


def _validate(args) -> int:
    from llm_wc.tracker.paths import DATA, RESULTS
    from llm_wc.tracker.snapshot import validate_dir

    bad = 0
    snaps = sorted(DATA.glob("*/*/manifest.json"))
    for manifest in snaps:
        problems = validate_dir(manifest.parent)
        print(f"{'ok  ' if not problems else 'FAIL'} {manifest.parent.relative_to(DATA)}"
              + (f": {'; '.join(problems)}" if problems else ""))
        bad += bool(problems)
    reports = sorted(p for p in RESULTS.glob("*/*.json") if p.parent.name not in ("history", "pairwise"))
    for path in reports:
        doc = json.loads(path.read_text())
        ok = doc.get("schema_version") == 1 and len(doc["models"]) == doc["n_models"] == len(doc["rpm"])
        print(f"{'ok  ' if ok else 'FAIL'} {path.relative_to(RESULTS)}")
        bad += not ok
    print(f"{len(snaps)} snapshots, {len(reports)} reports, {bad} problems")
    return 1 if bad else 0


def _status(args) -> int:
    from llm_wc.tracker.update import load_status

    for key, entry in load_status().items():
        print(f"{key:24s} {entry['result']:14s} {entry.get('snapshot_id', ''):28s} {entry['checked_at']}"
              + (f"  {entry['reason']}" if entry.get("reason") else ""))
    return 0


def _check_reference(args) -> int:
    from llm_wc.tracker.fidelity import check_reference

    return 0 if check_reference(analysis=not args.matrices_only) else 1


def _build(args) -> int:
    from llm_wc.tracker.site.build import build

    build(args.out)
    return 0


def _bundle(args) -> int:
    from llm_wc.tracker.paths import BUILD
    from llm_wc.tracker.site.build import build
    from llm_wc.tracker.site.bundle import bundle

    site = build(args.site) if not args.no_build else (args.site or BUILD)
    out = bundle(site, args.out)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


def _serve(args) -> int:
    from llm_wc.tracker.site.build import serve

    serve(args.out, args.port)
    return 0


def _probe(args) -> int:
    from llm_wc.tracker.discovery.probe import probe

    report = probe(args.target, task_col=args.task_col, model_col=args.model_col, score_col=args.score_col,
                   layout=args.layout, split=args.split, config=args.config)
    print(report.text())
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    return 0


def _scan(args) -> int:
    from llm_wc.tracker.discovery.scan import scan

    found = scan(limit=args.limit, write=not args.dry_run)
    print(f"{len(found)} new candidates" + ("" if args.dry_run else " written to tracker/discovery_inbox.json"))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m llm_wc.tracker", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("update", help="check upstream and store new snapshots")
    p.add_argument("--sources", nargs="+")
    p.add_argument("--force", action="store_true", help="fetch and store even if upstream looks unchanged")
    p.add_argument("--strict", action="store_true", help="exit 1 if any source failed")
    p.set_defaults(func=_update)

    p = sub.add_parser("history", help="add archive and reference rows to per-setting history")
    p.add_argument("--settings", nargs="+")
    p.set_defaults(func=_history)

    p = sub.add_parser("pairwise", help="pairwise 'A ranks above B' shares for the top systems")
    p.add_argument("--settings", nargs="+")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_pairwise)

    sub.add_parser("validate", help="validate stored snapshots and reports").set_defaults(func=_validate)
    sub.add_parser("status", help="last update result per source").set_defaults(func=_status)

    p = sub.add_parser("check-reference", help="adapters reproduce the reference matrices and numbers")
    p.add_argument("--matrices-only", action="store_true")
    p.set_defaults(func=_check_reference)

    p = sub.add_parser("build", help="build the static site")
    p.add_argument("--out", default=None)
    p.set_defaults(func=_build)

    p = sub.add_parser("bundle", help="build the site and pack it into one HTML file for private sharing")
    p.add_argument("--site", default=None, help="built site directory (default _site/)")
    p.add_argument("--out", default=None, help="output file (default _bundle/sota-or-luck.html)")
    p.add_argument("--no-build", action="store_true", help="pack an existing build")
    p.set_defaults(func=_bundle)

    p = sub.add_parser("serve", help="serve the built site on localhost")
    p.add_argument("--out", default=None)
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_serve)

    p = sub.add_parser("probe", help="feasibility report for a score table")
    p.add_argument("target", help="CSV/JSON/JSONL/Parquet path or URL, or hf:<dataset>")
    p.add_argument("--task-col")
    p.add_argument("--model-col")
    p.add_argument("--score-col")
    p.add_argument("--layout", choices=["auto", "long", "wide"], default="auto")
    p.add_argument("--split", default=None, help="Hugging Face split")
    p.add_argument("--config", default=None, help="Hugging Face config")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_probe)

    p = sub.add_parser("scan", help="search for candidate benchmarks")
    p.add_argument("--limit", type=int, default=30, help="results per query")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=_scan)

    args = ap.parse_args(argv)
    return args.func(args)
