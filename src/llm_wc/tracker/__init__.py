"""Leaderboard tracker: live RULER reports for public LLM leaderboards.

The tracker is separate from the experiment scripts. It reads their
pinned results (``results/ruler_audit``) for the reference edition and never
writes to ``resources/`` or ``results/``. Live data lives under ``tracker/``.

Layout:

- ``snapshot``: the canonical snapshot format (scores, tasks, models, manifest).
- ``matrix``: building a setting's task-by-model matrix from a snapshot.
- ``registry``: ``tracker/registry.yaml`` (sources, settings, site metadata).
- ``sources``: one adapter per upstream leaderboard.
- ``analyze``: RULER reports (the same ``llm_wc.wc.ruler.ruler`` the experiments use).
- ``update``: check upstream, fetch, validate, store, analyze.
- ``discovery``: feasibility criteria, probing tables, scanning for candidates.
- ``site``: the static website builder.

Run ``uv run python -m llm_wc.tracker --help``.
"""
