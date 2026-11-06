"""Criteria B1 and B2: the tracker reproduces the reference matrices and RULER tables exactly."""
import pytest

from llm_wc.benchmarks import SETTINGS, load_settings
from llm_wc.tracker import registry
from llm_wc.tracker.analyze import audit_tables, reference_tables
from llm_wc.tracker.fidelity import matrix_differences, reference_matrices, table_differences


@pytest.fixture(scope="module")
def matrices():
    return reference_matrices()


@pytest.fixture(scope="module")
def reference():
    return load_settings()


def test_every_reference_setting_is_registered(matrices):
    assert sorted(matrices) == sorted(SETTINGS)


@pytest.mark.parametrize("key", SETTINGS)
def test_adapter_matrix_matches_reference(key, matrices, reference):
    assert matrix_differences(matrices[key], reference[key]) == []


@pytest.mark.parametrize("key", SETTINGS)
def test_analysis_matches_reference_audit(key, matrices):
    ours = audit_tables(matrices[key], registry.analysis_params(), registry.setting(key)["short_label"])
    assert table_differences(ours, reference_tables(key)) == []
