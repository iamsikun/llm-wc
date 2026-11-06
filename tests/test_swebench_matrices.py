"""SWE-bench matrices follow the official task sets and record coverage."""
import json

import numpy as np

from llm_wc.benchmarks import SWE_PRO_DIR, load_settings
from llm_wc.parsers.swebench_tasks import load_swebench_task_ids


def test_multimodal_uses_official_test_split():
    s = load_settings(["swebench_multimodal"])["swebench_multimodal"]
    official = load_swebench_task_ids("multimodal")
    assert list(s.task_ids) == official and len(official) == 510
    # Result files list resolved issues only, so no system can have a listed failure
    assert s.obs.sum(0).max() == 186


def test_multilingual_has_300_issues():
    s = load_settings(["swebench_multilingual"])["swebench_multilingual"]
    assert s.obs.shape[0] == 300 and "mrdoob__three" not in s.task_ids


def test_pro_keeps_tasks_with_every_record():
    s = load_settings(["swebench_pro"])["swebench_pro"]
    for j, model_id in enumerate(s.model_ids):
        records = json.loads((SWE_PRO_DIR / "trajectories" / f"{model_id}.json").read_text())["instances"]
        assert set(s.task_ids) <= set(records)
        assert np.array_equal(s.obs[:, j], [records[t] for t in s.task_ids])
    assert s.obs.shape == (395, 14)
