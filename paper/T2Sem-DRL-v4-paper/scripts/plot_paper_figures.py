"""Publication figures — blue-only palette, Times New Roman (see t2sem-paper-revision figure-style.md)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

C_NAVY = "#0B3D91"
C_BLUE = "#1D4ED8"
C_SKY = "#3B82F6"
C_LIGHT = "#93C5FD"
C_PALE = "#DBEAFE"
C_GRID = "#E2E8F0"

CMAP_BLUES = LinearSegmentedColormap.from_list("blues", [C_PALE, C_NAVY])

# Typography: Times New Roman, 2pt smaller than prior defaults
FS = 9
FS_LABEL = 10
FS_TITLE = 10
FS_TICK = 8
FS_SMALL = 7
FS_HEAD = 11

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": FS,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_TITLE,
    "legend.fontsize": FS_SMALL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})


def _save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.2)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _clean_ax(ax, grid: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_NAVY)
    ax.spines["bottom"].set_color(C_NAVY)
    ax.tick_params(colors=C_NAVY, labelsize=FS_TICK)
    ax.set_axisbelow(True)
    if grid == "y":
        ax.yaxis.grid(True, color=C_GRID, lw=0.8)
        ax.xaxis.grid(False)
    elif grid == "x":
        ax.xaxis.grid(True, color=C_GRID, lw=0.8)
        ax.yaxis.grid(False)
    elif grid == "both":
        ax.grid(True, color=C_GRID, lw=0.7)


def fig0_pipeline() -> None:
    """Fig. 1 — architecture (unchanged layout, more vertical spacing)."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 5)
    ax.axis("off")

    blues = [C_PALE, C_LIGHT, C_SKY, C_BLUE, C_NAVY, C_SKY, C_LIGHT]
    boxes = [
        (0.3, 2.3, 1.5, 1.0, "UE Terminals"),
        (2.2, 2.3, 1.7, 1.0, "Feasibility Mask"),
        (4.2, 3.35, 1.7, 0.9, "Short PPO"),
        (4.2, 1.35, 1.7, 0.9, "Long DDPG"),
        (6.5, 2.3, 1.4, 1.0, "UAV MEC"),
        (8.3, 2.3, 1.4, 1.0, "HAP MEC"),
        (5.5, 0.2, 3.0, 0.7, "Per-tier CMDP  (λκ dual ascent)"),
    ]
    for i, (x, y, w, h, txt) in enumerate(boxes):
        col = blues[i]
        edge = C_NAVY if col in (C_BLUE, C_NAVY) else C_SKY
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=col, edgecolor=edge, lw=1.3,
        ))
        tc = "white" if col in (C_BLUE, C_NAVY) else C_NAVY
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=FS, color=tc, fontweight="bold",
                fontfamily="Times New Roman")

    for a, b in [
        ((1.8, 2.8), (2.2, 2.8)), ((3.9, 2.8), (4.2, 3.75)),
        ((3.9, 2.8), (4.2, 1.85)), ((5.9, 3.75), (6.5, 3.1)),
        ((5.9, 1.85), (6.5, 2.55)), ((7.9, 2.8), (8.3, 2.8)),
        ((5.3, 1.35), (6.8, 0.9)),
    ]:
        ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle="-|>", color=C_NAVY, lw=1.2, mutation_scale=13,
        ))
    ax.text(5.5, 4.55, "T2Sem-DRL v4 Architecture", ha="center",
            fontsize=FS_HEAD, fontweight="bold", color=C_NAVY,
            fontfamily="Times New Roman")
    _save(fig, "fig0_pipeline")


def fig_main_sla() -> None:
    """Fig. 2 — horizontal dot plot (forest style): best for method ranking."""
    df = pd.read_csv(DATA / "experiment_summary.csv")
    order = [
        ("random", "Random"),
        ("greedy", "Greedy"),
        ("tjcct", "TJCCT"),
        ("local_only", "Local-only"),
        ("T2Sem-DRL", "T2Sem-DRL"),
    ]
    labels, means, stds = [], [], []
    for m, lab in order:
        sub = df[df["method"] == m]
        labels.append(lab)
        means.append(sub["sla_rate_mean"].mean() * 100)
        stds.append(sub["sla_rate_std"].mean() * 100)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    y = np.arange(len(labels))
    colors = [C_LIGHT] * 4 + [C_NAVY]

    ax.axvline(80, color=C_SKY, ls="--", lw=1.2, zorder=0)
    # 80% reference label on x-axis (no in-plot legend → avoids bottom-right overlap)
    ax.text(80, 1.03, "80% plateau", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=FS_SMALL, color=C_SKY,
            fontfamily="Times New Roman")

    for yi, mu, sd, ci in zip(y, means, stds, colors):
        ax.plot([0, mu], [yi, yi], color=C_PALE, lw=5, solid_capstyle="round", zorder=1)
        ax.errorbar(mu, yi, xerr=sd, fmt="o", color=ci, ms=9, capsize=4,
                    markeredgecolor="white", markeredgewidth=1.5, ecolor=ci, zorder=3)
        # Fixed right margin column for all value labels
        ax.text(98.5, yi, f"{mu:.1f}%", ha="right", va="center",
                fontsize=FS_TICK, color=C_NAVY, fontfamily="Times New Roman")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS_TICK, fontfamily="Times New Roman")
    ax.set_xlabel("Overall SLA Satisfaction Rate (%)", fontfamily="Times New Roman")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    _clean_ax(ax, grid="x")
    fig.subplots_adjust(left=0.22, right=0.88, top=0.95, bottom=0.12)
    _save(fig, "fig2_main_sla")


def fig_per_tier_sla() -> None:
    """Fig. 3 — small multiples: one horizontal dot panel per tier (no line crossing)."""
    df = pd.read_csv(DATA / "per_tier_sla.csv")
    key_map = {"T2Sem-DRL": "T2Sem-DRL", "Greedy": "greedy", "Random": "random",
               "Local-only": "local_only"}
    methods = ["T2Sem-DRL", "Greedy", "Random", "Local-only"]
    tiers = ["critical", "interactive", "best_effort"]
    titles = ["(a) Critical tier", "(b) Interactive tier", "(c) Best-effort tier"]
    palette = [C_NAVY, C_BLUE, C_SKY, C_LIGHT]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    fig.subplots_adjust(wspace=0.42, left=0.10, right=0.98, top=0.90, bottom=0.14)

    for ax, tier, title in zip(axes, tiers, titles):
        vals = [df[df["method"] == key_map[m]][tier].values[0] * 100 for m in methods]
        y = np.arange(len(methods))
        for yi, v, m, c in zip(y, vals, methods, palette):
            ax.scatter(v, yi, s=110, color=c, edgecolors=C_NAVY, linewidths=1.2, zorder=3)
            ax.text(v + 1.8, yi, f"{v:.1f}%", va="center", fontsize=FS_TICK, color=C_NAVY,
                    fontfamily="Times New Roman")
        ax.axvline(80, color=C_PALE, ls="--", lw=1.0, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels(methods, fontsize=FS_TICK, fontfamily="Times New Roman")
        ax.set_xlabel("SLA Rate (%)", fontfamily="Times New Roman")
        ax.set_title(title, loc="left", fontsize=FS_TITLE, color=C_NAVY, pad=6,
                     fontfamily="Times New Roman")
        ax.set_xlim(35 if tier == "critical" else 45, 108)
        ax.invert_yaxis()
        _clean_ax(ax, grid="x")

    _save(fig, "fig3_per_tier_sla")


def fig_pareto() -> None:
    """Fig. 4 — scatter only; numbered markers + external legend (no on-plot text)."""
    df = pd.read_csv(DATA / "pareto_points.csv")
    drl = df[df["point"].str.startswith("P")].groupby("point").agg(
        sla=("sla", "mean"), energy=("energy", "mean"),
    ).reset_index()

    # Connect in order of decreasing energy (P4 → P1) for meaningful frontier
    order = ["P4_high", "P3_v4", "P2_low", "P1_minpressure"]
    drl["point"] = pd.Categorical(drl["point"], categories=order, ordered=True)
    drl = drl.sort_values("point")

    bl = df[df["point"].isin(["random", "greedy", "local_only"])].groupby("point").agg(
        sla=("sla", "mean"), energy=("energy", "mean"),
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ex = drl["energy"].values
    sy = drl["sla"].values * 100
    ax.plot(ex, sy, "-", color=C_LIGHT, lw=2, zorder=2)
    nums = ["P4", "P3", "P2", "P1"]
    names = ["P4 high pressure", "P3 v4 default", "P2 low", "P1 min pressure"]
    for i, (_, row) in enumerate(drl.iterrows()):
        ax.scatter(row["energy"], row["sla"] * 100, s=220, color=C_NAVY,
                   edgecolors="white", linewidths=2, zorder=4)
        ax.text(row["energy"], row["sla"] * 100, nums[i], ha="center", va="center",
                fontsize=FS_TICK, color="white", fontweight="bold", zorder=5,
                fontfamily="Times New Roman")

    bl_style = [
        ("random", "s", C_PALE, "Random"),
        ("greedy", "D", C_LIGHT, "Greedy"),
        ("local_only", "o", C_SKY, "Local-only"),
    ]
    for name, mk, col, lab in bl_style:
        ax.scatter(bl.loc[name, "energy"], bl.loc[name, "sla"] * 100,
                   marker=mk, s=110, color=col, edgecolors=C_NAVY, lw=1.2,
                   zorder=3, label=lab)

    ax.set_xlabel("Total Energy per Episode (J)")
    ax.set_ylabel("SLA Satisfaction Rate (%)")
    ax.set_xscale("log")
    ax.set_ylim(76, 95)
    _clean_ax(ax, grid="both")

    # Legend below plot — avoids all on-chart overlap
    pareto_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_NAVY,
               markersize=10, label=f"{n}")
        for n in names
    ]
    leg1 = ax.legend(handles=pareto_handles, loc="upper center",
                     bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False,
                     title="T2Sem-DRL sweep points (markers labeled P1–P4 on chart)",
                     title_fontsize=FS_SMALL, prop={"family": "Times New Roman"})
    ax.add_artist(leg1)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor=C_GRID,
              fontsize=FS_TICK, title="Masked baselines",
              prop={"family": "Times New Roman"})

    fig.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.26)
    _save(fig, "fig4_pareto")


def fig_mask_sensitivity() -> None:
    """Fig. 5 — three stacked panels (3×1), each full width; no horizontal squeeze."""
    df = pd.read_csv(DATA / "mask_sensitivity.csv")
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 10.5))
    fig.subplots_adjust(hspace=0.62, left=0.14, right=0.96, top=0.97, bottom=0.06)

    # (a) delay margin — line chart
    ax = axes[0]
    d = df[df["dim"] == "delay_margin"].sort_values("value")
    ax.plot(d["value"], d["overall_sla"] * 100, "o-", color=C_NAVY, lw=2.2, ms=8,
            markerfacecolor=C_SKY, markeredgecolor=C_NAVY)
    ax.axvline(0.8, color=C_BLUE, ls=":", lw=1.8)
    ax.text(0.8, 94.8, " default α=0.8", fontsize=FS_TICK, color=C_BLUE, va="top",
            fontfamily="Times New Roman")
    ax.set_xlim(0.58, 1.02)
    ax.set_ylim(81, 96)
    ax.set_xticks(d["value"].values)
    ax.set_ylabel("Overall SLA (%)")
    ax.set_xlabel(r"Delay margin $\alpha$")
    ax.set_title("(a) Delay margin sensitivity", loc="left", fontsize=FS_TITLE,
                 color=C_NAVY, pad=8, fontfamily="Times New Roman")
    _clean_ax(ax)

    # (b) channel — horizontal bar (categorical data)
    ax = axes[1]
    d = df[df["dim"] == "channel"].set_index("value").loc[["los", "expected", "nlos"]].reset_index()
    vals = d["overall_sla"].values * 100
    labels = ["LoS only", "Expected (default)", "NLoS only"]
    colors = [C_SKY, C_NAVY, C_LIGHT]
    y = np.arange(len(labels))
    bars = ax.barh(y, vals, height=0.55, color=colors, edgecolor=C_NAVY, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS_TICK, fontfamily="Times New Roman")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Overall SLA (%)", fontfamily="Times New Roman")
    ax.set_title("(b) Channel-estimate mode", loc="left", fontsize=FS_TITLE,
                 color=C_NAVY, pad=8, fontfamily="Times New Roman")
    for bar, v in zip(bars, vals):
        ax.text(v + 1.2, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=FS_TICK, color=C_NAVY,
                fontfamily="Times New Roman")
    _clean_ax(ax, grid="x")
    ax.invert_yaxis()

    # (c) similarity margin — line chart
    ax = axes[2]
    d = df[df["dim"] == "sim_margin"].sort_values("value")
    ax.plot(d["value"], d["overall_sla"] * 100, "s-", color=C_BLUE, lw=2.2, ms=8,
            markerfacecolor=C_PALE, markeredgecolor=C_NAVY)
    ax.axvline(0.05, color=C_BLUE, ls=":", lw=1.8)
    ax.text(0.05, 94.8, " default δs=0.05", fontsize=FS_TICK, color=C_BLUE, va="top",
            fontfamily="Times New Roman")
    ax.set_xlim(-0.005, 0.16)
    ax.set_ylim(79, 96)
    ax.set_xticks(d["value"].values)
    ax.set_ylabel("Overall SLA (%)")
    ax.set_xlabel(r"Similarity margin $\delta_s$")
    ax.set_title("(c) Similarity-margin sensitivity", loc="left", fontsize=FS_TITLE,
                 color=C_NAVY, pad=8, fontfamily="Times New Roman")
    _clean_ax(ax)

    _save(fig, "fig5_mask_sensitivity")


def fig_ablation() -> None:
    """Fig. 6 — dumbbell chart (gap to full model)."""
    df = pd.read_csv(DATA / "ablation_summary.csv")
    show = [
        ("mask_off", "w/o mask"),
        ("no_semantic", "w/o semantic"),
        ("pure_drl", "Pure DRL"),
        ("fixed_mu", "Fixed μ"),
        ("no_cmdp", "w/o CMDP"),
        ("single_timescale", "Single-timescale"),
    ]
    full_sla = df[df["ablation"] == "full"]["overall_sla"].values[0] * 100
    rows = []
    for key, lab in show:
        sla = df[df["ablation"] == key]["overall_sla"].values[0] * 100
        rows.append({"label": lab, "sla": sla, "delta": sla - full_sla})
    plot_df = pd.DataFrame(rows).sort_values("sla", ascending=True)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    y = np.arange(len(plot_df))

    ax.axvline(full_sla, color=C_NAVY, ls="--", lw=1.5, zorder=0)
    ax.text(0.98, 1.02, f"Full model = {full_sla:.1f}%", transform=ax.transAxes,
            fontsize=FS_TICK, color=C_NAVY, ha="right", va="bottom",
            fontfamily="Times New Roman")
    for yi, row in zip(y, plot_df.itertuples()):
        ax.scatter(row.sla, yi, s=100, color=C_SKY, edgecolors=C_NAVY, zorder=3)
        ax.text(row.sla + 1.0, yi, f"{row.sla:.1f}%", ha="left", va="center",
                fontsize=FS_TICK, color=C_NAVY, fontfamily="Times New Roman")

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"], fontsize=FS_TICK, fontfamily="Times New Roman")
    ax.set_xlabel("Overall SLA Satisfaction Rate (%)", fontfamily="Times New Roman")
    ax.set_xlim(40, 98)
    _clean_ax(ax, grid="x")
    fig.subplots_adjust(left=0.26, right=0.95, top=0.95, bottom=0.10)
    _save(fig, "fig6_ablation")


def fig_delay_energy() -> None:
    """Fig. 7 — two side-by-side metrics (delay | energy), shared methods; no bubble overlap."""
    df = pd.read_csv(DATA / "experiment_summary.csv")
    order = [
        ("random", "Random"),
        ("greedy", "Greedy"),
        ("local_only", "Local-only"),
        ("T2Sem-DRL", "T2Sem-DRL"),
    ]
    rows = []
    for m, lab in order:
        sub = df[df["method"] == m]
        rows.append({
            "method": lab,
            "delay": sub["avg_delay_mean"].mean(),
            "energy": sub["total_energy_mean"].mean() / 1000,
            "sla": sub["sla_rate_mean"].mean() * 100,
        })
    plot_df = pd.DataFrame(rows)
    palette = [C_LIGHT, C_SKY, C_BLUE, C_NAVY]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fig.subplots_adjust(wspace=0.45, left=0.18, right=0.97, top=0.88, bottom=0.12)

    y = np.arange(len(plot_df))
    for ax, col, xlabel, title, use_log in [
        (ax1, "delay", "Avg. E2E Delay (s)", "(a) Latency", False),
        (ax2, "energy", "Total Energy (kJ, log scale)", "(b) Propulsion energy", True),
    ]:
        vals = plot_df[col].values
        for yi, v, c in zip(y, vals, palette):
            ax.barh(yi, v, height=0.5, color=c, edgecolor=C_NAVY, linewidth=0.6)
            pad = v * 0.08 if use_log else max(vals) * 0.03
            ax.text(v + pad, yi, f"{v:.3f}" if col == "delay" else f"{v:.1f}",
                    va="center", fontsize=FS_TICK, color=C_NAVY,
                    fontfamily="Times New Roman")
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["method"], fontsize=FS_TICK, fontfamily="Times New Roman")
        ax.set_xlabel(xlabel, fontfamily="Times New Roman")
        ax.set_title(title, loc="left", fontsize=FS_TITLE, color=C_NAVY, pad=6,
                     fontfamily="Times New Roman")
        ax.invert_yaxis()
        _clean_ax(ax, grid="x")
        if use_log:
            ax.set_xscale("log")
            ax.set_xlim(2, 80)
        else:
            ax.set_xlim(0, max(vals) * 1.25)

    # SLA annotation in suptitle area — not overlapping bars
    sla_txt = "  |  ".join(f"{r.method}: {r.sla:.1f}% SLA" for r in plot_df.itertuples())
    fig.suptitle(f"Per-method service quality: {sla_txt}", fontsize=FS_TICK,
                 color=C_NAVY, y=0.98, fontfamily="Times New Roman")

    _save(fig, "fig7_delay_energy")


def main() -> None:
    fig0_pipeline()
    fig_main_sla()
    fig_per_tier_sla()
    fig_pareto()
    fig_mask_sensitivity()
    fig_ablation()
    fig_delay_energy()
    print(f"Wrote 7 figure sets to {FIG}")


if __name__ == "__main__":
    main()
