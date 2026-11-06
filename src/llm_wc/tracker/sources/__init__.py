"""Source adapters, keyed by the ``source`` field in ``tracker/registry.yaml``."""
from __future__ import annotations

from llm_wc.tracker.sources.base import Context, Revision, Source, SourceUnavailable
from llm_wc.tracker.sources.external import AlpacaEval2, LiveBench, WildBenchV2
from llm_wc.tracker.sources.swebench import SweBenchClassic, SweBenchPro
from llm_wc.tracker.sources.tau2 import Tau2

SOURCES: dict[str, Source] = {s.key: s for s in [
    SweBenchClassic("bash-only"),
    SweBenchClassic("verified"),
    SweBenchClassic("multimodal"),
    SweBenchClassic("multilingual"),
    SweBenchClassic("lite", in_reference=False),
    SweBenchClassic("test", in_reference=False),
    SweBenchPro(),
    Tau2(),
    AlpacaEval2(),
    WildBenchV2(),
    LiveBench(),
]}

__all__ = ["SOURCES", "Context", "Revision", "Source", "SourceUnavailable"]
