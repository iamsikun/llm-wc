"""The SWE-bench fetch path builds the same task set as ``llm_wc.benchmarks``.

Upstream result files for some boards (Multimodal) list only the tasks some system
resolved, so the adapter must add the rest of the official task list as unresolved.
"""
import pytest

from llm_wc.benchmarks import SETTINGS, load_settings, swe_trajectory_csv
from llm_wc.tracker import registry
from llm_wc.tracker.fidelity import matrix_differences
from llm_wc.tracker.matrix import build_matrix
from llm_wc.tracker.sources import SOURCES
from llm_wc.tracker.sources.swebench import SweBenchClassic, _read_wide, _wide_snapshot

KEYS = [k for k in registry.settings()
        if isinstance(SOURCES[registry.setting(k)["source"]], SweBenchClassic) and k in SETTINGS]


@pytest.mark.parametrize("key", KEYS)
def test_fetch_path_rebuilds_the_pinned_matrix(key):
    spec = registry.setting(key)
    src = SOURCES[spec["source"]]
    path = swe_trajectory_csv(src.variant)
    snap = _wide_snapshot(src.key, path.parent.name, _read_wide(path, src.variant), {})
    assert matrix_differences(build_matrix(snap, key, spec), load_settings([key])[key]) == []
