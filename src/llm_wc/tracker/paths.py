"""Filesystem locations used by the tracker."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRACKER = ROOT / "tracker"
DATA = TRACKER / "data"
RESULTS = TRACKER / "results"
HISTORY = RESULTS / "history"
REGISTRY = TRACKER / "registry.yaml"
CANDIDATES = TRACKER / "candidates.yaml"
STATUS = TRACKER / "status.json"
SITE_SRC = TRACKER / "site"
INBOX = TRACKER / "discovery_inbox.json"

# Reference inputs and outputs (read only).
RESOURCES = ROOT / "resources"
REFERENCE_AUDIT = ROOT / "results" / "ruler_audit"

# Not committed.
CACHE = ROOT / ".cache" / "tracker"
BUILD = ROOT / "_site"
