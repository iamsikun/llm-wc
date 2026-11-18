"""Check upstream sources, store new snapshots, and analyze them.

For each source: ask for the upstream revision (cheap); if it matches the
latest stored snapshot (or the pinned reference input, when nothing is stored
yet), do nothing. Otherwise fetch, validate, and compare the content
fingerprint; only new content is written, as a snapshot under
``tracker/data`` plus one live report per setting under ``tracker/results``.
Each source runs on its own: a failure is recorded in ``tracker/status.json``
and the next source still runs.
"""
from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone

from llm_wc.tracker import registry
from llm_wc.tracker.analyze import audit_tables, report, write_report
from llm_wc.tracker.history import append_live
from llm_wc.tracker.matrix import build_matrix
from llm_wc.tracker.paths import DATA, RESULTS, STATUS
from llm_wc.tracker.snapshot import Snapshot, latest_snapshot_dir, read_manifest, read_snapshot, write_snapshot
from llm_wc.tracker.sources import SOURCES, Context, SourceUnavailable


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_status(path=STATUS) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def save_status(status: dict, path=STATUS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")


def analyze_snapshot(snap: Snapshot, data_root=DATA, results_root=RESULTS) -> list[str]:
    """Write one live report per setting built from this snapshot; return the paths."""
    params = registry.analysis_params()
    paths = []
    for key in registry.settings_for_source(snap.source):
        spec = registry.setting(key)
        matrix = build_matrix(snap, key, spec)
        if matrix.obs.shape[1] < 2:
            continue
        tables = audit_tables(matrix, params, spec["short_label"])
        doc = report(tables, key=key, edition="live", snapshot_id=snap.snapshot_id, source=snap.source,
                     upstream=snap.upstream, excluded=matrix.excluded + snap.notes.get("excluded", []), params=params)
        paths.append(str(write_report(doc, results_root)))
        append_live(doc)
        if results_root == RESULTS:
            from llm_wc.tracker import pairwise

            pairwise.write(key, snap.snapshot_id, pairwise.pairwise(matrix, params))
    return paths


def update_source(key: str, ctx: Context, force: bool = False, data_root=DATA, results_root=RESULTS) -> dict:
    src = SOURCES[key]
    entry: dict = {"checked_at": _now()}
    last_dir = latest_snapshot_dir(key, data_root)
    last = read_manifest(last_dir) if last_dir else None
    rev = src.latest_revision(ctx)
    if rev is not None:
        entry["upstream"] = rev.as_dict()
    ref = None
    if rev is not None and not force:
        if last and last["upstream"].get("revision") == rev.revision:
            return {**entry, "result": "unchanged", "snapshot_id": last["snapshot_id"]}
        if not last:
            ref = src.reference_snapshot()
            if ref is not None and ref.upstream.get("revision") == rev.revision:
                return {**entry, "result": "same_as_reference"}
    snap = src.fetch(ctx, rev)
    fp = snap.fingerprint()
    if last and last["fingerprint"] == fp and not force:
        return {**entry, "result": "unchanged", "snapshot_id": last["snapshot_id"]}
    if not last and not force:
        ref = ref or src.reference_snapshot()
        if ref is not None and ref.fingerprint() == fp:
            return {**entry, "result": "same_as_reference"}
    out = write_snapshot(snap, data_root)
    reports = analyze_snapshot(read_snapshot(out), data_root, results_root)
    return {**entry, "result": "updated", "snapshot_id": snap.snapshot_id, "reports": len(reports)}


def _retired(key: str) -> str | None:
    """The registry's ``retired`` reason for source ``key``, if the source is no longer checked."""
    return registry.registry().get("sources", {}).get(key, {}).get("retired")


def update(keys: list[str] | None = None, ctx: Context | None = None, force: bool = False,
           data_root=DATA, results_root=RESULTS, status_path=STATUS) -> dict:
    ctx = ctx or Context()
    status = load_status(status_path)
    for key in keys or list(SOURCES):
        print(f"[{key}] checking ...", flush=True)
        prev = status.get(key, {})
        retired = _retired(key)
        if retired:
            # Not checked: keep the last check's date and data, and record why.
            entry = {"checked_at": prev.get("checked_at", _now()), "result": "retired", "reason": retired}
            for field in ("last_success", "snapshot_id"):
                if field in prev:
                    entry[field] = prev[field]
            status[key] = entry
            save_status(status, status_path)
            print(f"[{key}] retired: {retired}", flush=True)
            continue
        try:
            entry = update_source(key, ctx, force, data_root, results_root)
            entry["last_success"] = entry["checked_at"]
        except SourceUnavailable as exc:
            entry = {"checked_at": _now(), "result": "unavailable", "reason": str(exc)}
        except Exception as exc:  # noqa: BLE001 - one source must not stop the others
            entry = {"checked_at": _now(), "result": "failed",
                     "reason": f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"[:300]}
            traceback.print_exc()
        for field in ("last_success", "snapshot_id"):
            if field not in entry and field in prev:
                entry[field] = prev[field]
        status[key] = entry
        save_status(status, status_path)
        print(f"[{key}] {entry['result']}" + (f": {entry.get('reason')}" if entry.get("reason") else ""), flush=True)
    return status
