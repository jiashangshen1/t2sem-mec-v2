"""Plot per-tier lambda and violation curves from train_metrics.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Fig. 8 is the only figure allowed to use cross-hue colors because the three
# tier trajectories overlap; color contrast is more important than palette
# uniformity for this diagnostic convergence plot.
C_NAVY = "#0B3D91"
C_ORANGE = "#D55E00"
C_GREEN = "#009E73"
C_GRID = "#E2E8F0"
TIER_COLORS = {"critical": C_NAVY, "interactive": C_ORANGE, "best_effort": C_GREEN}
TIER_STYLES = {"critical": "-", "interactive": "--", "best_effort": "-."}

FS = 9
FS_LABEL = 10
FS_TICK = 8
FS_TITLE = 10

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": FS,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_TITLE,
    "legend.fontsize": FS_TICK,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})


def _clean_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_NAVY)
    ax.spines["bottom"].set_color(C_NAVY)
    ax.tick_params(colors=C_NAVY, labelsize=FS_TICK)
    ax.yaxis.grid(True, color=C_GRID, lw=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="train_metrics.csv path")
    parser.add_argument("--out", required=True, help="output png or pdf path")
    args = parser.parse_args()

    df = pd.read_csv(args.log)
    tiers = [c.replace("lambda_", "") for c in df.columns if c.startswith("lambda_")]

    fig, axes = plt.subplots(2, 1, figsize=(5.2, 4.2), sharex=True)
    for tier in tiers:
        color = TIER_COLORS.get(tier, C_NAVY)
        linestyle = TIER_STYLES.get(tier, "-")
        viol_col = f"viol_{tier}"
        lam_col = f"lambda_{tier}"
        if viol_col in df.columns:
            axes[0].plot(
                df["episode"], df[viol_col], label=tier, linewidth=1.6,
                color=color, linestyle=linestyle,
            )
        if lam_col in df.columns:
            axes[1].plot(
                df["episode"], df[lam_col], label=tier, linewidth=1.6,
                color=color, linestyle=linestyle,
            )

    axes[0].set_ylabel("Violation (target − rate)", fontfamily="Times New Roman")
    axes[0].legend(
        fontsize=FS_TICK, loc="upper right",
        prop={"family": "Times New Roman"},
        frameon=True, edgecolor=C_GRID,
    )
    axes[1].set_ylabel("λκ", fontfamily="Times New Roman")
    axes[1].set_xlabel("Episode", fontfamily="Times New Roman")
    axes[1].legend(
        fontsize=FS_TICK, loc="upper left",
        prop={"family": "Times New Roman"},
        frameon=True, edgecolor=C_GRID,
    )
    for ax in axes:
        _clean_ax(ax)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily("Times New Roman")

    fig.suptitle(
        "Per-tier CMDP dual convergence",
        fontsize=FS_TITLE, y=1.02, color=C_NAVY,
        fontfamily="Times New Roman",
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    if out.suffix.lower() == ".png":
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
