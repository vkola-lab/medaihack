"""
Generate figures/architecture.png — the hero diagram embedded in the
top-level README. Run once; output is checked into the repo.

    python figures/_make_architecture.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── Layout config ──────────────────────────────────────────────────────────
W, H = 14.0, 11.5                     # figure size in inches
COL_X = {"main": 5.5, "side": 11.0}   # x-centers for main column / FiLM column
BOX_W, BOX_H = 5.2, 0.62              # default box width / height
GAP = 0.18                            # gap between successive boxes

PALETTE = {
    "input":   "#E3F2FD",   # light blue
    "norm":    "#FFE0B2",   # peach   (TracerNorm)
    "stem":    "#C8E6C9",   # green
    "stage":   "#BBDEFB",   # blue    (ResNet stages)
    "film":    "#FFF59D",   # yellow  (FiLM)
    "pool":    "#D1C4E9",   # purple
    "head":    "#F8BBD0",   # pink
    "output":  "#FFCDD2",   # light red
    "emb":     "#FFE082",   # amber   (tracer embedding)
}

EDGE = "#37474F"


def box(ax, y, color, text, x=COL_X["main"], w=BOX_W, h=BOX_H, fontsize=10, weight="normal"):
    """Draw a rounded box centered at (x, y) with text inside."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.2, edgecolor=EDGE, facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color="#212121")
    return y - h / 2  # bottom edge


def arrow(ax, y_from, y_to, x=COL_X["main"], label=None, ls="-"):
    a = FancyArrowPatch(
        (x, y_from - 0.02), (x, y_to + 0.02),
        arrowstyle="-|>", mutation_scale=14, linewidth=1.3,
        color=EDGE, linestyle=ls,
    )
    ax.add_patch(a)
    if label:
        ax.text(x + 0.18, (y_from + y_to) / 2, label,
                fontsize=8.5, color="#37474F", va="center")


def lateral_arrow(ax, y, x_from, x_to, label=None):
    """FiLM conditioning arrow from side column into the main column."""
    a = FancyArrowPatch(
        (x_from - 1.0, y), (x_to + BOX_W / 2 + 0.05, y),
        arrowstyle="-|>", mutation_scale=12, linewidth=1.1,
        color="#FBC02D", linestyle="--",
    )
    ax.add_patch(a)
    if label:
        ax.text((x_from + x_to) / 2, y + 0.15, label,
                fontsize=8, color="#F57F17", ha="center")


# ── Build the figure ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")

# Title
ax.text(W / 2, H - 0.25,
        "mygo_centiloid — TracerNorm + FiLM-conditioned 3D ResNet-18 for Centiloid Regression",
        ha="center", fontsize=12.5, fontweight="bold", color="#0D47A1")

# Stack of boxes (top → bottom)
y = H - 0.95

# Inputs
y0 = box(ax, y, PALETTE["input"],
         "Input PET volume (B, 1, 128, 128, 128)  +  tracer_id (B,)",
         fontsize=10.5, weight="bold")
y -= BOX_H + GAP + 0.05

arrow(ax, y0, y + BOX_H / 2)
y0 = box(ax, y, PALETTE["norm"],
         "TracerNorm    →   per-tracer (γ, β) intensity rescale",
         fontsize=10)
y -= BOX_H + GAP

arrow(ax, y0, y + BOX_H / 2)
y0 = box(ax, y, PALETTE["stem"],
         "Stem: Conv3d(1→64, 7³, s=2)  →  BN  →  ReLU  →  MaxPool3d(s=2)",
         fontsize=9.5)
ax.text(COL_X["main"] + BOX_W / 2 + 0.18, y, "(B, 64, 32, 32, 32)",
        fontsize=8.5, color="#455A64", va="center")
y -= BOX_H + GAP

# Stages with FiLM
stages = [
    ("Stage 1: ResBlock × 2 (s=1)  +  FiLM",  "(B,  64, 32, 32, 32)"),
    ("Stage 2: ResBlock × 2 (s=2)  +  FiLM",  "(B, 128, 16, 16, 16)"),
    ("Stage 3: ResBlock × 2 (s=2)  +  FiLM",  "(B, 256,  8,  8,  8)"),
    ("Stage 4: ResBlock × 2 (s=2)  +  FiLM",  "(B, 512,  4,  4,  4)"),
]
stage_ys = []
for label, shape in stages:
    arrow(ax, y0, y + BOX_H / 2)
    y0 = box(ax, y, PALETTE["stage"], label, fontsize=9.5)
    ax.text(COL_X["main"] + BOX_W / 2 + 0.18, y, shape,
            fontsize=8.5, color="#455A64", va="center")
    stage_ys.append(y)
    y -= BOX_H + GAP

# GAP
arrow(ax, y0, y + BOX_H / 2)
y0 = box(ax, y, PALETTE["pool"],
         "Global Average Pool 3D",
         fontsize=10)
ax.text(COL_X["main"] + BOX_W / 2 + 0.18, y, "(B, 512)",
        fontsize=8.5, color="#455A64", va="center")
y -= BOX_H + GAP

# Concat
arrow(ax, y0, y + BOX_H / 2)
y0 = box(ax, y, PALETTE["pool"],
         "Concat[ image_feat (512) ‖ tracer_emb (32) ]",
         fontsize=10)
ax.text(COL_X["main"] + BOX_W / 2 + 0.18, y, "(B, 544)",
        fontsize=8.5, color="#455A64", va="center")
y -= BOX_H + GAP

# Head
arrow(ax, y0, y + BOX_H / 2)
y0 = box(ax, y, PALETTE["head"],
         "FC(544→256) → BN → GELU → Dropout(0.4)",
         fontsize=9.5)
y -= BOX_H + GAP

arrow(ax, y0, y + BOX_H / 2)
y0 = box(ax, y, PALETTE["head"],
         "FC(256→64) → BN → GELU → Dropout(0.2)",
         fontsize=9.5)
y -= BOX_H + GAP

arrow(ax, y0, y + BOX_H / 2)
y0 = box(ax, y, PALETTE["head"],
         "FC(64→1)   ←  linear  (CL can be negative)",
         fontsize=9.5)
y -= BOX_H + GAP

# Output
arrow(ax, y0, y + BOX_H / 2)
y0 = box(ax, y, PALETTE["output"],
         "Centiloid prediction  (B,)",
         fontsize=10.5, weight="bold")

# ── FiLM side column: Tracer embedding → 4 stages ──────────────────────────
emb_y = (stage_ys[0] + stage_ys[-1]) / 2 + (stage_ys[0] - stage_ys[-1]) * 0.12
box(ax, emb_y, PALETTE["emb"],
    "TracerEmbedding\n(4 → 32)",
    x=COL_X["side"], w=2.6, h=1.1, fontsize=9.5, weight="bold")

for sy in stage_ys:
    lateral_arrow(ax, sy, COL_X["side"], COL_X["main"])

# Concat also takes tracer_emb — show that branch
ax.text(COL_X["side"], emb_y - 1.05,
        "(also concatenated\n into the head input)",
        ha="center", fontsize=8.2, color="#6D4C41", style="italic")
# dashed connector down to the concat box
concat_y = stage_ys[-1] - 2 * (BOX_H + GAP)
a = FancyArrowPatch(
    (COL_X["side"], emb_y - 0.7), (COL_X["main"] + BOX_W / 2 + 0.05, concat_y),
    connectionstyle="arc3,rad=-0.18",
    arrowstyle="-|>", mutation_scale=12, linewidth=1.0,
    color="#FBC02D", linestyle="--",
)
ax.add_patch(a)

# Legend
legend_handles = [
    mpatches.Patch(facecolor=PALETTE["norm"],   edgecolor=EDGE, label="TracerNorm"),
    mpatches.Patch(facecolor=PALETTE["stem"],   edgecolor=EDGE, label="Stem"),
    mpatches.Patch(facecolor=PALETTE["stage"],  edgecolor=EDGE, label="ResNet stage + FiLM"),
    mpatches.Patch(facecolor=PALETTE["emb"],    edgecolor=EDGE, label="Tracer embedding"),
    mpatches.Patch(facecolor=PALETTE["pool"],   edgecolor=EDGE, label="Pooling / Concat"),
    mpatches.Patch(facecolor=PALETTE["head"],   edgecolor=EDGE, label="Regression head"),
]
ax.legend(handles=legend_handles, loc="lower left",
          bbox_to_anchor=(0.005, 0.005), fontsize=8.5,
          ncol=3, frameon=True, edgecolor="#B0BEC5")

out_path = os.path.join(os.path.dirname(__file__), "architecture.png")
plt.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
