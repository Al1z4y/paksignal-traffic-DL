"""
generate_demo_images.py
Run this once to produce the demo PNG assets referenced in README.md.

    python scripts/generate_demo_images.py

Output folder: docs/images/
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(OUT, exist_ok=True)


# ─── colour palette ───────────────────────────────────────────────────────────
BG      = "#0f172a"   # dark navy
PANEL   = "#1e293b"
ACCENT  = "#7c3aed"   # purple
GREEN   = "#16a34a"
RED     = "#dc2626"
AMBER   = "#d97706"
CYAN    = "#0891b2"
TEXT    = "#e2e8f0"
MUTED   = "#94a3b8"

LANE_COLORS = ["#7c3aed", "#0891b2", "#16a34a", "#d97706"]
LANES = ["North", "South", "East", "West"]


def _fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(BG)
    return fig, ax


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=10)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(MUTED)
        spine.set_linewidth(0.6)
    if title:
        ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=10)
    ax.grid(axis="y", color="#334155", linewidth=0.5, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PCU Score Comparison
# ─────────────────────────────────────────────────────────────────────────────
def pcu_chart():
    pcu_scores = [18.5, 6.0, 14.0, 8.5]

    fig, ax = _fig(9, 4.5)
    bars = ax.bar(LANES, pcu_scores, color=LANE_COLORS, width=0.5, zorder=3,
                  edgecolor="#1e293b", linewidth=1.2)
    _style_ax(ax,
              title="PCU-Weighted Traffic Load per Lane (sample)",
              xlabel="Direction",
              ylabel="PCU Score")

    for bar, val in zip(bars, pcu_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}",
                ha="center", va="bottom", color=TEXT, fontsize=11, fontweight="bold")

    ax.set_ylim(0, 24)
    fig.tight_layout(pad=1.5)
    path = os.path.join(OUT, "pcu_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Static vs Adaptive Green Time
# ─────────────────────────────────────────────────────────────────────────────
def green_time_chart():
    static   = [30, 30, 30, 30]
    adaptive = [45, 15, 38, 22]
    x = np.arange(len(LANES))
    w = 0.35

    fig, ax = _fig(9, 4.5)
    b1 = ax.bar(x - w/2, static,   w, label="Static (30s fixed)",       color=MUTED,    zorder=3, edgecolor="#0f172a", linewidth=1)
    b2 = ax.bar(x + w/2, adaptive, w, label="Adaptive (PCU-weighted)", color=LANE_COLORS, zorder=3, edgecolor="#0f172a", linewidth=1)
    _style_ax(ax,
              title="Static vs Adaptive Green Signal Time (sample)",
              xlabel="Direction",
              ylabel="Green Time (seconds)")

    ax.set_xticks(x)
    ax.set_xticklabels(LANES)
    ax.set_ylim(0, 55)

    for bar, val in zip(b1, static):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.8, f"{val}s",
                ha="center", va="bottom", color=TEXT, fontsize=9)
    for bar, val in zip(b2, adaptive):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.8, f"{val}s",
                ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")

    ax.axhline(30, color=MUTED, linewidth=0.8, linestyle=":", alpha=0.5)
    ax.legend(facecolor=BG, edgecolor=MUTED, labelcolor=TEXT, fontsize=10)

    fig.tight_layout(pad=1.5)
    path = os.path.join(OUT, "green_time_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Vehicle Class Distribution (stacked bar)
# ─────────────────────────────────────────────────────────────────────────────
def vehicle_distribution():
    classes = ["bike", "car", "rickshaw", "htv", "emv"]
    cls_colors = ["#7c3aed", "#0891b2", "#d97706", "#dc2626", "#16a34a"]
    counts = {
        "North": [6, 4, 2, 1, 0],
        "South": [2, 2, 1, 0, 0],
        "East":  [4, 3, 2, 1, 0],
        "West":  [3, 2, 1, 0, 1],
    }
    x = np.arange(len(LANES))

    fig, ax = _fig(9, 4.5)
    bottoms = np.zeros(len(LANES))
    for cls, col, idx in zip(classes, cls_colors, range(len(classes))):
        vals = [counts[l][idx] for l in LANES]
        ax.bar(x, vals, bottom=bottoms, label=cls.capitalize(),
               color=col, zorder=3, edgecolor="#0f172a", linewidth=0.8)
        bottoms += np.array(vals)

    _style_ax(ax,
              title="Vehicle Class Distribution per Lane (sample)",
              xlabel="Direction",
              ylabel="Vehicle Count")
    ax.set_xticks(x)
    ax.set_xticklabels(LANES)
    ax.set_ylim(0, 16)
    ax.legend(facecolor=BG, edgecolor=MUTED, labelcolor=TEXT, fontsize=10,
              loc="upper right")

    fig.tight_layout(pad=1.5)
    path = os.path.join(OUT, "vehicle_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. System Architecture Flow Diagram
# ─────────────────────────────────────────────────────────────────────────────
def architecture_diagram():
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, label, sublabel="", color=ACCENT):
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.08",
            facecolor=color, edgecolor=TEXT, linewidth=1.2, zorder=3
        )
        ax.add_patch(rect)
        ax.text(x, y + (0.1 if sublabel else 0), label,
                ha="center", va="center", color="white",
                fontsize=9, fontweight="bold", zorder=4)
        if sublabel:
            ax.text(x, y - 0.22, sublabel,
                    ha="center", va="center", color="#cbd5e1",
                    fontsize=7.5, zorder=4)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED,
                                   lw=1.5, mutation_scale=14), zorder=2)

    # ── Row 1: Four feeds ────────────────────────────────────────────────────
    feed_y = 5.1
    feeds = [("North\nFeed", 1.5), ("South\nFeed", 3.5),
             ("East\nFeed",  6.5), ("West\nFeed",  9.0)]
    for label, x in feeds:
        box(x, feed_y, 1.5, 0.7, label, color=CYAN)

    # ── YOLOv8 detector boxes ─────────────────────────────────────────────
    det_y = 3.7
    for _, x in feeds:
        box(x, det_y, 1.5, 0.7, "YOLOv8n", "Detector", color="#1d4ed8")
        arrow(x, feed_y - 0.35, x, det_y + 0.35)

    # ── PCU scoring ──────────────────────────────────────────────────────
    pcu_y = 2.4
    for _, x in feeds:
        box(x, pcu_y, 1.5, 0.7, "PCU", "Scoring", color=ACCENT)
        arrow(x, det_y - 0.35, x, pcu_y + 0.35)
        # converge to centre
        arrow(x, pcu_y - 0.35, 5.3, 1.3 + 0.35)

    # ── Signal controller ────────────────────────────────────────────────
    box(5.3, 1.3, 2.2, 0.72, "Signal Controller", "Green Time Allocator", color=GREEN)

    # ── EMV Priority ─────────────────────────────────────────────────────
    box(8.8, 1.3, 2.0, 0.72, "EMV Priority", "Emergency Override", color=RED)
    arrow(5.3 + 1.1, 1.3, 8.8 - 1.0, 1.3)

    # ── Dashboard ────────────────────────────────────────────────────────
    box(5.3, 0.35, 5.6, 0.6, "Streamlit Dashboard  ·  CSV Export  ·  Env. Impact", color="#374151")
    arrow(5.3, 1.3 - 0.36, 5.3, 0.35 + 0.3)
    arrow(8.8, 1.3 - 0.36, 6.6, 0.35 + 0.3)

    ax.set_title("PakSignal — System Architecture", color=TEXT,
                 fontsize=14, fontweight="bold", pad=6)

    fig.tight_layout(pad=0.5)
    path = os.path.join(OUT, "architecture.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PCU Weight reference card
# ─────────────────────────────────────────────────────────────────────────────
def pcu_weight_card():
    classes = ["bike", "car", "rickshaw", "htv / bus", "emv"]
    weights = [0.5, 1.0, 1.5, 3.0, 2.0]
    colors  = ["#7c3aed", "#0891b2", "#d97706", "#dc2626", "#16a34a"]

    fig, ax = _fig(8, 3.8)
    bars = ax.barh(classes[::-1], weights[::-1], color=colors[::-1],
                   height=0.55, zorder=3, edgecolor="#0f172a", linewidth=1)
    _style_ax(ax,
              title="PCU Weights — IRC:106-1990 Reference",
              xlabel="PCU Weight",
              ylabel="")

    for bar, val in zip(bars, weights[::-1]):
        ax.text(val + 0.04, bar.get_y() + bar.get_height()/2,
                f"× {val}", va="center", color=TEXT, fontsize=11, fontweight="bold")

    ax.set_xlim(0, 3.8)
    ax.grid(axis="x", color="#334155", linewidth=0.5, linestyle="--", alpha=0.7)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=11, labelcolor=TEXT)

    fig.tight_layout(pad=1.5)
    path = os.path.join(OUT, "pcu_weights.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved {path}")


if __name__ == "__main__":
    print("Generating demo images …")
    pcu_chart()
    green_time_chart()
    vehicle_distribution()
    architecture_diagram()
    pcu_weight_card()
    print("Done. All images saved to docs/images/")
