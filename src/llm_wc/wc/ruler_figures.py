"""Figures for RULER reports.

Each function draws under its own rc context, so output does not depend on
notebook or backend rc settings.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

FIGURE_RC = {"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "pdf.fonttype": 42, "ps.fonttype": 42}
TIER_COLOR = "#d64a3a"
NO_DATE = {"CreationDate": None}  # reproducible PDFs: reruns leave the files unchanged

_HEADLINE_NAMES = {
    "live-SWE-agent + Claude 4.5 Opus medium (20251101)": "live-SWE-agent + Claude 4.5 Opus",
    "live-SWE-agent + Gemini 3 Pro Preview (2025-11-18)": "live-SWE-agent + Gemini 3 Pro",
    "Atlassian Rovo Dev (2025-09-02)": "Atlassian Rovo Dev",
    "EPAM AI/Run Developer Agent v20250719 + Claude 4 Sonnet": "EPAM AI/Run Developer Agent",
    "mini-SWE-agent + Claude 4.5 Opus (high reasoning)": "mini-SWE-agent + Claude 4.5 Opus",
    "mini-SWE-agent + MiniMax M2.5 (high reasoning)": "mini-SWE-agent + MiniMax M2.5",
    "mini-SWE-agent + Gemini 3 Flash (high reasoning)": "mini-SWE-agent + Gemini 3 Flash",
}
_INTERVAL_NAMES = {
    **_HEADLINE_NAMES,
    "Sonar Foundation Agent + Claude 4.5 Opus": "Sonar Foundation + Claude 4.5 Opus",
    "EPAM AI/Run Developer Agent v20250719 + Claude 4 Sonnet": "EPAM AI/Run Agent",
}


class _rc:
    """rc context starting from matplotlib defaults."""

    def __init__(self, params: dict):
        self.params = params

    def __enter__(self):
        self.ctx = mpl.rc_context()
        self.ctx.__enter__()
        mpl.rcdefaults()
        mpl.rcParams.update(self.params)

    def __exit__(self, *exc):
        return self.ctx.__exit__(*exc)


def _compact(name: str, replacements: dict, width: int) -> str:
    cleaned = replacements.get(str(name), str(name))
    cleaned = re.sub(r"\s*\((?:20\d{2}[-/]?\d{2}[-/]?\d{2}|20\d{6,}|high reasoning|medium reasoning)\)", "", cleaned)
    cleaned = re.sub(r"\s+Preview\b", "", cleaned)
    return "\n".join(textwrap.wrap(cleaned, width=width, break_long_words=False))


def _callout(fig, *, x, y, width, height, text, fontsize=13):
    fig.patches.append(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.01", transform=fig.transFigure,
        facecolor="#F8FAFC", edgecolor="#3F5F99", linewidth=1.1))
    fig.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize, color="#1F2937")


def plot_ruler_headline(
    models,
    rpm: np.ndarray,
    *,
    title: str,
    tier_1_size: int,
    rank_mass: float = 0.8,
    top_k: int = 10,
    output_path: Path | None = None,
    dpi: int = 220,
):
    """Headline figure: top-k RPM heatmap and 80% rank-mass intervals.

    ``models`` is a raw-order table with columns model, rank_start, rank_end,
    rank_low_80, rank_high_80; ``rpm`` is its (J x J) RPM. Saves PDF and PNG
    when ``output_path`` is given; returns the figure otherwise.
    """
    top = models.iloc[:top_k]
    assert top.rank_end.iloc[-1] == top_k, "Do not split an observed score tie."
    heatmap = rpm[:top_k, :top_k]
    lo, hi = top.rank_low_80.to_numpy(), top.rank_high_80.to_numpy()
    interval_max_rank = int(np.ceil(max(top_k, hi.max()) / 5) * 5)
    p11 = rpm[0, 0]
    with _rc({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig = plt.figure(figsize=(15.4, 8.0), dpi=dpi)
        gs = fig.add_gridspec(2, 2, left=0.075, right=0.975, top=0.855, bottom=0.255, width_ratios=[1.02, 1.18],
                              height_ratios=[1.0, 0.09], wspace=0.25, hspace=0.34)
        ax_heat, ax_int = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        cax, legend_ax = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
        legend_ax.axis("off")
        fig.suptitle(title, y=0.965, fontsize=19.5, fontweight="bold", color="#111111")

        cmap = LinearSegmentedColormap.from_list("rpm_blues", ["#F8FBFF", "#DDEAF7", "#8CB6DD", "#2A5AA5"])
        image = ax_heat.imshow(heatmap, cmap=cmap, vmin=0, vmax=max(0.40, float(np.nanmax(heatmap))), aspect="auto")
        ax_heat.set_title("(a) Rank probability matrix", fontsize=12.5, fontweight="bold", pad=14)
        ax_heat.set_xlabel("Bootstrap rank", fontsize=12)
        ax_heat.set_ylabel(f"Empirical top-{top_k} models", fontsize=12)
        ax_heat.set_xticks(np.arange(top_k))
        ax_heat.set_xticklabels([str(i) for i in range(1, top_k + 1)], fontsize=10)
        ax_heat.set_yticks(np.arange(top_k))
        ax_heat.set_yticklabels([_compact(m, _HEADLINE_NAMES, 22) for m in top.model], fontsize=9)
        ax_heat.tick_params(length=0)
        ax_heat.set_xticks(np.arange(-0.5, top_k, 1), minor=True)
        ax_heat.set_yticks(np.arange(-0.5, top_k, 1), minor=True)
        ax_heat.grid(which="minor", color="#B8B8B8", linewidth=0.75, alpha=0.55)
        for spine in ax_heat.spines.values():
            spine.set_color("#AFAFAF")
            spine.set_linewidth(0.8)
        for row in range(top_k):
            for col in range(top_k):
                value = float(heatmap[row, col])
                if value >= 0.01:
                    ax_heat.text(col, row, f"{100 * value:.1f}%", ha="center", va="center", fontsize=8.4,
                                 fontweight="bold", color="white" if value >= 0.22 else "#222222")
        cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
        cbar.set_ticks([0, 0.1, 0.2, 0.3, 0.4])
        cbar.set_ticklabels(["0%", "10%", "20%", "30%", "40%"])

        ax_int.set_title("(b) Rank uncertainty intervals", fontsize=12.5, fontweight="bold", pad=14)
        ax_int.set_xlabel("Rank", fontsize=12)
        ax_int.set_xlim(0.4, interval_max_rank + 0.6)
        ax_int.set_ylim(-0.6, top_k - 0.4)
        ax_int.invert_yaxis()
        ax_int.set_yticks(np.arange(top_k))
        ax_int.set_yticklabels([_compact(m, _INTERVAL_NAMES, 18) for m in top.model], fontsize=8.4)
        ax_int.set_xticks(np.arange(1, interval_max_rank + 1) if interval_max_rank <= 12
                          else [1] + list(range(5, interval_max_rank + 1, 5)))
        ax_int.grid(axis="x", linestyle=(0, (4, 3)), color="#D0D0D0", linewidth=0.8, alpha=0.85)
        ax_int.tick_params(axis="y", length=0, pad=3)
        ax_int.tick_params(axis="x", labelsize=10)
        for side in ("top", "right"):
            ax_int.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax_int.spines[side].set_color("#B5B5B5")
            ax_int.spines[side].set_linewidth(0.8)
        for i in range(top_k):
            ax_int.hlines(i, lo[i], hi[i], color="#A8A8A8", linewidth=1.0, zorder=1)
            ax_int.vlines([lo[i], hi[i]], i - 0.09, i + 0.09, color="#A8A8A8", linewidth=1.0, zorder=1)
            ax_int.scatter(top.rank_start.iloc[i], i, s=64, color="#173A80", edgecolor="white", linewidth=1.0,
                           zorder=3)

        legend_ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="square,pad=0.0",
                                           transform=legend_ax.transAxes, facecolor="white", edgecolor="#D0D0D0",
                                           linewidth=1.0, clip_on=False, zorder=0))
        legend_ax.legend(handles=[
            Line2D([0], [0], marker="o", markersize=8.5, linestyle="None", color="#173A80", label="Observed rank"),
            Line2D([0], [0], color="#A8A8A8", linewidth=1.4, marker="|", markersize=10,
                   label=f"{100 * rank_mass:.0f}% rank-mass interval"),
        ], loc="center", bbox_to_anchor=(0.0, 0.0, 1.0, 1.0), ncol=2, frameon=False, fontsize=10.6,
            handlelength=2.2, handletextpad=0.8, columnspacing=1.8, borderpad=0.4, borderaxespad=0.0)

        heat_pos, int_pos = ax_heat.get_position(), ax_int.get_position()
        _callout(fig, x=heat_pos.x0, y=0.12, width=heat_pos.width, height=0.055, fontsize=11.4,
                 text=rf"Published #1 remains #1 in only $\bf{{{100 * p11:.1f}\%}}$ of bootstrap samples.")
        _callout(fig, x=int_pos.x0, y=0.12, width=int_pos.width, height=0.055, fontsize=11.4,
                 text=rf"Top $\bf{{{tier_1_size}}}$ models share one unresolved tier "
                      rf"($\bf{{{100 * rank_mass:.0f}\%}}$ rank mass).")
        if output_path is None:
            return fig
        output_path = Path(output_path)
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", metadata=NO_DATE)
        fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", facecolor="white", dpi=dpi)
        plt.close(fig)


def plot_rpm_with_tiers(
    rpm: np.ndarray,
    tier_ids: np.ndarray,
    *,
    title: str,
    ylabels: list[str] | None = None,
    top: int | None = None,
    vmax: float = 1.0,
    figsize=(7.5, 4.0),
    output_path: Path | None = None,
):
    """RPM heatmap (models in raw order x bootstrap ranks) with tier boundaries in red.

    With ``ylabels`` the rows are labeled by model; otherwise by raw-order index
    with ticks at 1, 5, 10, ....
    """
    k = top or rpm.shape[0]
    with _rc(FIGURE_RC):
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        im = ax.imshow(rpm[:k, :k], cmap="Blues", vmin=0, vmax=vmax)
        if ylabels is not None:
            ax.set(yticks=range(k), yticklabels=ylabels, xticks=range(k), xticklabels=range(1, k + 1),
                   xlabel="Bootstrap rank", title=title)
            lw = 1.7
        else:
            ax.set(xlabel="Bootstrap rank", ylabel="Raw-order model index", title=title)
            ticks = [0] + list(range(4, k, 5))
            ax.set_xticks(ticks, [str(x + 1) for x in ticks])
            ax.set_yticks(ticks, [str(x + 1) for x in ticks])
            lw = 1.8
        for i in range(k - 1):
            if tier_ids[i] != tier_ids[i + 1]:
                ax.axhline(i + .5, c=TIER_COLOR, lw=lw)
        fig.colorbar(im, ax=ax, shrink=.8, label="Fractional rank mass")
        if output_path is None:
            return fig
        fig.savefig(output_path, metadata=NO_DATE)
        plt.close(fig)


def plot_rpm_panels(
    panels: list[tuple[np.ndarray, np.ndarray, list[str], str]],
    *,
    figsize=(10.5, 3.8),
    output_path: Path | None = None,
):
    """Side-by-side labeled RPM heatmaps sharing one colorbar.

    Each panel is ``(rpm, tier_ids, ylabels, title)``; model labels are rotated
    45 degrees so the panels fit across the page.
    """
    with _rc(FIGURE_RC):
        fig, axs = plt.subplots(1, len(panels), figsize=figsize, layout="constrained")
        for ax, (letter, (rpm, tier_ids, ylabels, title)) in zip(axs, zip("abcdefgh", panels)):
            k = len(ylabels)
            im = ax.imshow(rpm[:k, :k], cmap="Blues", vmin=0, vmax=1)
            ax.set(xticks=range(k), xticklabels=range(1, k + 1), xlabel="Bootstrap rank", title=f"({letter}) {title}")
            ax.set_yticks(range(k), ylabels, rotation=45, ha="right", va="center", rotation_mode="anchor", fontsize=9.5)
            for i in range(k - 1):
                if tier_ids[i] != tier_ids[i + 1]:
                    ax.axhline(i + .5, c=TIER_COLOR, lw=1.7)
        fig.colorbar(im, ax=axs, shrink=.85, aspect=25, pad=.01, label="Fractional rank mass")
        fig.get_layout_engine().set(w_pad=.02, wspace=.01)
        if output_path is None:
            return fig
        fig.savefig(output_path, metadata=NO_DATE)
        plt.close(fig)


def plot_two_model_toy(raw, adjusted, selected_true_mean, rpm_small, rpm_large, *, output_path: Path | None = None):
    """Figure 2: small-gap winner scores (raw vs adjusted) and the two averaged RPMs."""
    with _rc(FIGURE_RC):
        fig, axs = plt.subplots(1, 3, figsize=(9.1, 3.1), gridspec_kw={"width_ratios": [1.5, 1, 1]},
                                layout="constrained")
        bins = np.linspace(44, 59, 36)
        axs[0].hist(100 * raw, bins=bins, density=True, alpha=.6, label="Raw winner", color="#d65f5f")
        axs[0].hist(100 * adjusted, bins=bins, density=True, alpha=.6, label="Adjusted", color="#477ca8")
        axs[0].axvline(100 * selected_true_mean, color="black", ls="--", lw=1.5, label="Selected truth")
        axs[0].set(xlabel="Winner score (%)", ylabel="Density", title="A. Small gap: 0.5 pp")
        axs[0].legend(fontsize=8.5)
        for ax, a, title in zip(axs[1:], [rpm_small, rpm_large], ["B. Small gap RPM", "C. Large gap RPM"]):
            a = np.asarray(a)
            ax.imshow(a, vmin=0, vmax=1, cmap="Blues")
            ax.set(xticks=[0, 1], xticklabels=["1", "2"], yticks=[0, 1], yticklabels=["Model 1", "Model 2"],
                   xlabel="Bootstrap rank", title=title)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{100 * a[i, j]:.1f}%", ha="center", va="center",
                            color="white" if a[i, j] > .6 else "black", fontsize=11)
        if output_path is None:
            return fig
        fig.savefig(output_path, metadata=NO_DATE)
        plt.close(fig)
