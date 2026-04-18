"""
Shared constants and helpers for the EDA suite.

Single source of truth for cross-script constants (amyloid threshold,
tracer ordering, color palette, plot style). Importing from this module
eliminates the drift risk of redefining these in every script.

Usage in any eda/NN_*.py:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _common import (
        AMYLOID_POS_THRESHOLD, TRACER_ORDER, TRACER_NAMES,
        TRACER_COLORS, FOREGROUND_THRESH, setup_style, set_run_tag,
    )
"""

from datetime import datetime, timezone
from functools import wraps

import matplotlib.pyplot as plt
import seaborn as sns

# ── Clinical constants ────────────────────────────────────────────────────────

# Centiloid threshold for amyloid positivity.
# Source: Klunk et al., Alzheimer's & Dementia 2015 (≥24.4 CL ≈ moderate plaque).
AMYLOID_POS_THRESHOLD: float = 24.4

# Foreground voxel threshold used for "brain" mask in calibration analysis.
# Volumes are min-max normalized to [0,1]; voxels above 0.05 are treated as
# brain tissue rather than air / zero-padded background.
FOREGROUND_THRESH: float = 0.05


# ── Tracer metadata ───────────────────────────────────────────────────────────

TRACER_ORDER: list[str] = ["FBP", "FBB", "NAV", "PIB"]

TRACER_NAMES: dict[str, str] = {
    "FBP": "Florbetapir",
    "FBB": "Florbetaben",
    "NAV": "Florbetanav",
    "PIB": "Pittsburgh\nCompound B",
}

TRACER_COLORS: dict[str, tuple] = dict(
    zip(TRACER_ORDER, sns.color_palette("Set1", 4))
)


# ── Plot style ────────────────────────────────────────────────────────────────

def setup_style(font_scale: float = 1.2) -> None:
    """Apply the project-wide seaborn theme. Call once per script in main()."""
    sns.set_theme(style="whitegrid", font_scale=font_scale)


# ── Figure header (tag + UTC timestamp on every saved figure) ────────────────

_RUN_TAG: str | None = None


def set_run_tag(tag: str | None) -> None:
    """Set the identifier stamped at the top of every saved figure.

    Typically called once in a script's main() with a run/experiment label
    (e.g. the output directory basename, or a --tag CLI argument). All
    matplotlib figures saved afterwards include `<tag>  ·  <UTC time>`
    centered above the figure — useful when the same script is run for
    multiple experiments (baseline vs ablation) and you need to tell the
    resulting PNGs apart at a glance.
    """
    global _RUN_TAG
    _RUN_TAG = tag


def _stamp_header(fig) -> None:
    if not _RUN_TAG:
        return
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fig.text(0.5, 1.00, f"{_RUN_TAG}  ·  {ts}",
             ha="center", va="top", fontsize=10,
             color="#444", fontweight="normal")


def _install_savefig_hook() -> None:
    """Wrap plt.savefig / Figure.savefig so every saved figure gets stamped.

    Safe: the stamp is idempotent per-call and only fires when a run tag
    has been set. All four EDA scripts already use `bbox_inches='tight'`,
    so the header at y=1.00 is captured without clipping.
    """
    if getattr(plt.Figure.savefig, "_mygo_wrapped", False):
        return  # already installed

    orig_plt_savefig = plt.savefig
    orig_fig_savefig = plt.Figure.savefig

    @wraps(orig_plt_savefig)
    def _plt_savefig(*args, **kwargs):
        _stamp_header(plt.gcf())
        return orig_plt_savefig(*args, **kwargs)

    @wraps(orig_fig_savefig)
    def _fig_savefig(self, *args, **kwargs):
        _stamp_header(self)
        return orig_fig_savefig(self, *args, **kwargs)

    _plt_savefig._mygo_wrapped = True   # type: ignore[attr-defined]
    _fig_savefig._mygo_wrapped = True   # type: ignore[attr-defined]
    plt.savefig = _plt_savefig
    plt.Figure.savefig = _fig_savefig


_install_savefig_hook()


# ── Logger ────────────────────────────────────────────────────────────────────

def log(msg: str, level: str = "info") -> None:
    """Lightweight logger used across EDA scripts (avoids stdlib logging setup)."""
    prefix = {
        "info": "   ",
        "ok"  : "  ✓",
        "warn": "  ⚠",
        "save": "  →",
        "skip": "  ·",
    }.get(level, "   ")
    print(f"{prefix} {msg}")
