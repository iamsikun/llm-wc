"""Interactive RULER report: a self-contained HTML page covering every setting.

Numbers come from ``results/ruler_audit``; descriptive text (sources,
populations, limitations) and the page itself come from
``resources/ruler_report/{settings.json,template.html}``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESOURCES = Path(__file__).resolve().parents[3] / "resources" / "ruler_report"
LEVELS = (("80", 0.2), ("90", 0.1), ("95", 0.05), ("99", 0.01))


def load_audit(audit_dir: Path, key: str) -> dict:
    """Read one setting's audit outputs: models, position_bias, rpm, tiers."""
    d = Path(audit_dir) / key
    return {
        "models": pd.read_csv(d / "models.csv", dtype={"rank": str}, float_precision="round_trip"),
        "position_bias": pd.read_csv(d / "position_bias.csv", float_precision="round_trip"),
        "rpm": pd.read_csv(d / "rpm.csv", index_col=0, float_precision="round_trip").to_numpy(),
        "tiers": pd.read_csv(d / "tiers.csv", dtype={"rank_mass": str}),
    }


def report_data(audit_dir: Path, settings_meta: dict | None = None) -> dict:
    meta = settings_meta or json.loads((RESOURCES / "settings.json").read_text())
    benchmarks = []
    for static in meta["settings"]:
        a = load_audit(audit_dir, static["key"])
        models, rpm = a["models"], a["rpm"]
        bias = a["position_bias"]
        leader = slice(int(models.rank_start[0]) - 1, int(models.rank_end[0]))
        ordinary = bias.ordinary_adjustment_pp.to_numpy()[leader]
        rows = []
        for r in models.itertuples():
            rows.append({
                "adjustment_high_pp": r.adjustment_high_pp, "adjustment_low_pp": r.adjustment_low_pp,
                "adjustment_pp": r.adjustment_pp, "model": r.model, "model_id": r.model_id,
                "p_top_5": r.p_top_5, "rank": r.rank, "rank_end": int(r.rank_end), "rank_start": int(r.rank_start),
                "raw_score_pct": r.raw_score_pct,
                "tiering": {k: {"rank_high": int(getattr(r, f"rank_high_{k}")),
                                "rank_low": int(getattr(r, f"rank_low_{k}")),
                                "tier": int(getattr(r, f"tier_{k}"))} for k, _ in LEVELS},
            })
        tier_levels = []
        for key, alpha in LEVELS:
            t = a["tiers"][a["tiers"].rank_mass == key].drop(columns="rank_mass")
            tiers = [{c: int(v) for c, v in row.items()} for row in t.to_dict("records")]
            tier_levels.append({"alpha": alpha, "key": key, "rank_mass": 1 - alpha, "tiers": tiers,
                                "tier_count": len(tiers), "tier_1_size": tiers[0]["size"]})
        benchmarks.append({
            **static,
            "n_models": len(models),
            "rpm": rpm.tolist(),
            "position_bias_pp": bias.adjustment_pp.tolist(),
            "rows": rows,
            "tier_levels": tier_levels,
            "primary_tier_count": tier_levels[0]["tier_count"],
            "primary_tier_1_size": tier_levels[0]["tier_1_size"],
            "moon_rank_1_adjustment_pp": rows[0]["adjustment_pp"],
            "moon_rank_1_adjustment_low_pp": rows[0]["adjustment_low_pp"],
            "moon_rank_1_adjustment_high_pp": rows[0]["adjustment_high_pp"],
            "ordinary_rank_1_adjustment_pp": float(np.mean(ordinary)),
            "ordinary_rank_1_adjustment_low_pp": float(np.min(ordinary)),
            "ordinary_rank_1_adjustment_high_pp": float(np.max(ordinary)),
        })
    summary = pd.read_csv(Path(audit_dir) / "summary.csv", float_precision="round_trip").set_index("setting")
    for b in benchmarks:
        b["m"] = int(summary.loc[b["key"], "m"])
        b["n_tasks"] = int(summary.loc[b["key"], "n_tasks"])
    return {"benchmarks": benchmarks, "method": meta["method"], "schema_version": meta["schema_version"],
            "title": meta["title"]}


def write_ruler_report(audit_dir: Path, output_path: Path) -> dict:
    data = report_data(audit_dir)
    html = (RESOURCES / "template.html").read_text()
    html = html.replace("const REPORT_DATA = null;", "const REPORT_DATA = " + json.dumps(data, ensure_ascii=False) + ";", 1)
    Path(output_path).write_text(html)
    return data
