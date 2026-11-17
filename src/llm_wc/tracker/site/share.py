"""Shareable assets: link-preview images (1200x630 PNG) and embeddable badges (SVG)."""
from __future__ import annotations

from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, ACCENT, WASH, GRID, BG = "#121211", "#76756f", "#2a78d6", "#e6f0fc", "#e9e8e3", "#fcfcfb"


def _rank_text(m: dict) -> str:
    lo, hi = m["rank_low_80"], m["rank_high_80"]
    return f"#{lo}" if lo == hi else f"#{lo}–{hi}"


def og_image(doc: dict, label: str, path: Path, title: str, url: str) -> None:
    """Title, leaderboard name, and the two-panel figure for the top rows."""
    models = doc["models"]
    tier1 = sum(m["tier_80"] == 1 for m in models)
    k = min(doc["n_models"], max(6, min(10, tier1 + 2)))
    rows = models[:k]
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=BG)
    fig.text(0.05, 0.9, title, fontsize=28, fontweight="bold", color=INK, family="serif")
    top = models[0]
    adj = top["raw_score_pct"] - top["adjustment_pp"]
    fig.text(0.05, 0.82, f"{label}: top score {top['raw_score_pct']:.1f}% → {adj:.1f}% adjusted · "
             f"rank range {_rank_text(top)} · {tier1} of {doc['n_models']} in the top tier",
             fontsize=15, color=MUTED)
    ax1 = fig.add_axes([0.30, 0.12, 0.30, 0.6], facecolor=BG)
    ax2 = fig.add_axes([0.66, 0.12, 0.30, 0.6], facecolor=BG)
    ys = list(range(k))
    raw = [m["raw_score_pct"] for m in rows]
    adjs = [m["raw_score_pct"] - m["adjustment_pp"] for m in rows]
    for ax in (ax1, ax2):
        ax.axhspan(-0.5, min(tier1, k) - 0.5, color=WASH, zorder=0)
        ax.set_ylim(k - 0.5, -0.5)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=11)
        ax.grid(axis="x", color=GRID, lw=1)
        ax.set_axisbelow(True)
    for y, r, a in zip(ys, raw, adjs):
        ax1.annotate("", xy=(a, y), xytext=(r, y), zorder=2,
                     arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2, mutation_scale=12, shrinkA=0, shrinkB=5))
    ax1.scatter(raw, ys, s=55, facecolor=BG, edgecolor=INK, lw=1.6, zorder=3)
    ax1.scatter(adjs, ys, s=60, color=ACCENT, zorder=4)
    ax1.set_title("Score Adjustment", loc="left", fontsize=15, fontweight="bold", color=INK)
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    for y, m in zip(ys, rows):
        ax2.plot([m["rank_low_80"], m["rank_high_80"]], [y, y], color=ACCENT if m["tier_80"] == 1 else MUTED,
                 lw=7, solid_capstyle="round", alpha=1 if m["tier_80"] == 1 else 0.5, zorder=2)
        ax2.scatter([m["rank_start"]], [y], s=30, color=INK, zorder=3)
    rmax = max(k, max(m["rank_high_80"] for m in rows))
    ax2.set_xlim(0.5, rmax + 0.5)
    step = max(1, int(round(rmax / 4 / 5)) * 5) if rmax > 8 else 1
    ax2.set_xticks([1] + list(range(step, rmax + 1, step))[(1 if step == 1 else 0):])
    ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"#{int(v)}" if v >= 1 else ""))
    ax2.set_title("Rank Interval", loc="left", fontsize=15, fontweight="bold", color=INK)
    for y, m in zip(ys, rows):
        name = m["model"] if len(m["model"]) <= 30 else m["model"][:29] + "…"
        fig.text(0.285, 0.12 + 0.6 * (k - 0.5 - y) / k, name, ha="right", va="center", fontsize=11.5, color=INK)
    fig.text(0.05, 0.04, url, fontsize=12, color=MUTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=BG, metadata={"Software": None})
    plt.close(fig)


def badge_svg(doc: dict) -> str:
    """Shields-style badge: 'RULER | 79.2% → 78.3% · top tier 15/173'."""
    top = doc["models"][0]
    tier1 = sum(m["tier_80"] == 1 for m in doc["models"])
    left = "RULER"
    right = f"{top['raw_score_pct']:.1f}% → {top['raw_score_pct'] - top['adjustment_pp']:.1f}% · top tier {tier1}/{doc['n_models']}"
    lw, rw = 12 + 7 * len(left), 12 + 6.4 * len(right)
    w = lw + rw
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="20" role="img" aria-label="{escape(left)}: {escape(right)}">'
            f'<title>{escape(left)}: {escape(right)}</title><clipPath id="r"><rect width="{w:.0f}" height="20" rx="3"/></clipPath>'
            f'<g clip-path="url(#r)"><rect width="{lw:.0f}" height="20" fill="#121211"/><rect x="{lw:.0f}" width="{rw:.0f}" height="20" fill="#2a78d6"/></g>'
            f'<g fill="#fff" font-family="Verdana,DejaVu Sans,sans-serif" font-size="11">'
            f'<text x="{lw / 2:.1f}" y="14" text-anchor="middle">{escape(left)}</text>'
            f'<text x="{lw + rw / 2:.1f}" y="14" text-anchor="middle">{escape(right)}</text></g></svg>')
