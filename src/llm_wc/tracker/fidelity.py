"""Check that the tracker reproduces the reference results.

``reference_matrices`` rebuilds each reference setting from the adapters' reference
snapshots (stored and re-read, so the snapshot format is exercised too).
``check_reference`` compares them with ``llm_wc.benchmarks.load_settings()`` and,
optionally, their RULER tables with ``results/ruler_audit``.
"""
from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd

from llm_wc.tracker import registry
from llm_wc.tracker.analyze import audit_tables, reference_tables
from llm_wc.tracker.matrix import Matrix, build_matrix
from llm_wc.tracker.snapshot import read_snapshot, write_snapshot
from llm_wc.tracker.sources import SOURCES

# Float tolerance for the RULER tables, in their own units (scores and pp). Integer
# outputs (ranks, tiers, the RPM) must match exactly. The tolerance absorbs
# last-bit rounding between platforms' linear-algebra libraries (macOS vs Linux).
FLOAT_ATOL = 1e-9


def reference_matrices() -> dict[str, Matrix]:
    out, snaps = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for key, spec in registry.settings().items():
            if not spec.get("in_reference"):
                continue
            src = spec["source"]
            if src not in snaps:
                snaps[src] = read_snapshot(write_snapshot(SOURCES[src].reference_snapshot(), tmp))
            out[key] = build_matrix(snaps[src], key, spec)
    return out


def matrix_differences(matrix: Matrix, setting) -> list[str]:
    diffs = []
    for field in ("obs", "model_ids", "model_names", "task_ids", "strata"):
        if not np.array_equal(getattr(matrix, field), getattr(setting, field)):
            diffs.append(field)
    if matrix.stratum_type != setting.stratum_type:
        diffs.append("stratum_type")
    if (matrix.clusters is None) != (setting.clusters is None) or (
            matrix.clusters is not None and not np.array_equal(matrix.clusters, setting.clusters)):
        diffs.append("clusters")
    return diffs


def table_differences(ours: dict, ref: dict) -> list[str]:
    diffs = []
    for name in ("models", "position_bias", "tiers"):
        try:
            pd.testing.assert_frame_equal(ours[name].reset_index(drop=True), ref[name],
                                          check_exact=False, rtol=0, atol=FLOAT_ATOL, check_dtype=False)
        except AssertionError:
            diffs.append(name)
    if not np.array_equal(ours["rpm"], ref["rpm"]):
        diffs.append("rpm")
    diffs += [f"summary.{k}" for k, v in ref["summary"].items() if not _same(ours["summary"].get(k), v)]
    return diffs


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= FLOAT_ATOL or (np.isnan(a) and np.isnan(b))
    return a == b


def check_reference(analysis: bool = True) -> bool:
    from llm_wc.benchmarks import load_settings

    reference = load_settings()
    params = registry.analysis_params()
    ok = True
    for key, matrix in reference_matrices().items():
        diffs = matrix_differences(matrix, reference[key])
        if analysis and not diffs:
            diffs = table_differences(audit_tables(matrix, params, registry.setting(key)["short_label"]),
                                      reference_tables(key))
        print(f"{'ok  ' if not diffs else 'FAIL'} {key} {matrix.obs.shape}" + (f": {diffs}" if diffs else ""),
              flush=True)
        ok &= not diffs
    return ok
