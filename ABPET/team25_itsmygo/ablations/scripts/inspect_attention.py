"""
inspect_attention.py  —  EXPERIMENTAL, not part of the main pipeline

Visualize the stage-4 spatial attention map learned by PETResNetAttn.
The attention module is a CBAM-style (avg, max) → Conv3D → sigmoid gate at
4×4×4 resolution. This script upsamples it to PET resolution (trilinear)
and overlays on axial / coronal / sagittal slices — same layout as the
HiResCAM panels produced by gradcam.py for easy visual comparison.

Sample picking:
    By default picks 3 representative samples with CL ≈ 0 (young-control),
    CL ≈ 30 (just-positive), CL ≈ 100 (AD-typical). Override with --cl_targets
    "0,30,100,150" for a different spread.

Usage:
    python ablations/scripts/inspect_attention.py \\
        --config dev/config/predict.yaml \\
        --checkpoint checkpoints/stage3_attn/best_model.pt \\
        --out_dir results/attention/stage3_attn
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mygo_centiloid import PETDataset, PETResNetAttn


# ─────────────────────────────────────────────────────────────────────────────
# Config + model
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    name = ckpt.get("model", "petresnet_attn")
    if name != "petresnet_attn":
        raise ValueError(
            f"inspect_attention expects a petresnet_attn checkpoint, got {name!r}. "
            f"For {name!r} use dev/experimental_fet/gradcam.py instead."
        )

    model = PETResNetAttn(
        num_tracers      = ckpt["num_tracers"],
        dropout_high     = ckpt.get("dropout_high", 0.4),
        dropout_low      = ckpt.get("dropout_low",  0.2),
        attention_kernel = ckpt.get("attention_kernel", 7),
    ).to(device)

    state = {k.replace("_orig_mod.", ""): v
             for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Sample picking  (nearest-CL to each target)
# ─────────────────────────────────────────────────────────────────────────────

def pick_by_cl_targets(dataset, cl_targets: list[float]) -> list[int]:
    """Pick one index per target CL — the dataset sample closest to it."""
    if not dataset.has_targets:
        raise ValueError("inspect_attention needs a CSV with CENTILOIDS column.")
    cl = dataset.centiloids.numpy()
    used: set[int] = set()
    picks: list[int] = []
    for target in cl_targets:
        order = np.argsort(np.abs(cl - target))
        for idx in order:
            if idx not in used:
                picks.append(int(idx))
                used.add(int(idx))
                break
    return picks


# ─────────────────────────────────────────────────────────────────────────────
# Attention compute
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def attention_for_sample(model, x: torch.Tensor, tracer_idx: torch.Tensor
                         ) -> tuple[float, np.ndarray]:
    """One forward pass → (y_pred, attention map upsampled to x spatial shape)."""
    y, attn = model(x, tracer_idx, return_attention=True)
    # attn: (B=1, 1, d, h, w) at stage-4 resolution (4³ for 128³ input)
    attn_up = F.interpolate(attn, size=x.shape[2:],
                            mode="trilinear", align_corners=False)
    # Normalize 0..1 for visualization
    a = attn_up - attn_up.amin(dim=(2, 3, 4), keepdim=True)
    a = a / (a.amax(dim=(2, 3, 4), keepdim=True) + 1e-8)
    return float(y.item()), a.squeeze().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation  (mirrors dev/experimental_fet/gradcam.py save_overlay)
# ─────────────────────────────────────────────────────────────────────────────

def save_overlay(volume: np.ndarray, heatmap: np.ndarray, meta: dict,
                 out_path: str) -> None:
    vol = volume.squeeze()
    D, H, W = vol.shape
    slices = {
        "Axial (z)":    (vol[D // 2, :, :], heatmap[D // 2, :, :]),
        "Coronal (y)":  (vol[:, H // 2, :], heatmap[:, H // 2, :]),
        "Sagittal (x)": (vol[:, :, W // 2], heatmap[:, :, W // 2]),
    }

    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    fig.suptitle(
        f"SpatialAttention  |  stage4 (kernel={meta['kernel']})  |  model={meta['model']}\n"
        f"tracer={meta['tracer']}  |  CL_true={meta['cl_true']:+.1f}  "
        f"CL_pred={meta['cl_pred']:+.1f}  |  |err|={meta['abs_err']:.1f} CL",
        fontsize=11, fontweight="bold",
    )

    for row, (title, (bg, hm)) in enumerate(slices.items()):
        axes[row, 0].imshow(bg, cmap="gray", origin="lower")
        axes[row, 0].set_title(f"{title} — PET", fontsize=10)
        axes[row, 1].imshow(hm, cmap="hot", origin="lower", vmin=0, vmax=1)
        axes[row, 1].set_title(f"{title} — attn", fontsize=10)
        axes[row, 2].imshow(bg, cmap="gray", origin="lower")
        axes[row, 2].imshow(hm, cmap="hot", origin="lower",
                            alpha=0.45, vmin=0, vmax=1)
        axes[row, 2].set_title(f"{title} — overlay", fontsize=10)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def _filename(rank: int, tracer: str, cl_true: float, cl_pred: float,
              abs_err: float) -> str:
    def _s(v): return f"{v:+06.1f}".replace("+", "p").replace("-", "n").replace(".", "_")
    return (f"{rank:02d}_{tracer}_"
            f"cl{_s(cl_true)}_pred{_s(cl_pred)}_err{abs_err:04.1f}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config",     default="dev/config/predict.yaml")
    ap.add_argument("--csv",        default=None, help="Override paths.csv")
    ap.add_argument("--checkpoint", default=None, help="Override paths.checkpoint")
    ap.add_argument("--out_dir",    default="results/attention")
    ap.add_argument("--cl_targets", default="0,30,100",
                    help="Comma-separated CL targets — one sample picked per target, "
                         "nearest by absolute CL distance. Default: '0,30,100'.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    csv_path  = args.csv        or cfg["paths"]["csv"]
    ckpt_path = args.checkpoint or cfg["paths"]["checkpoint"]
    cl_targets = [float(t) for t in args.cl_targets.split(",") if t.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────
    model, ckpt = load_model(ckpt_path, device)
    kernel = ckpt.get("attention_kernel", 7)
    print(f"Model        : petresnet_attn (kernel={kernel})")
    print(f"Checkpoint   : {ckpt_path}")

    # ── Data ──────────────────────────────────────────────────────────────
    dataset = PETDataset(csv_path, tracer_map=ckpt["tracer_map"])
    inv_tracer = {v: k for k, v in dataset.tracer_map.items()}

    indices = pick_by_cl_targets(dataset, cl_targets)
    print(f"\nPicked {len(indices)} samples for CL targets {cl_targets}:")

    for rank, idx in enumerate(indices):
        image, cl_true, tid = dataset[idx]
        cl_true = float(cl_true)
        x = image.unsqueeze(0).to(device).float()
        t = tid.unsqueeze(0).to(device)

        y_pred, heatmap = attention_for_sample(model, x, t)
        tracer_name = inv_tracer[int(tid.item())]
        abs_err = abs(y_pred - cl_true)

        out_name = _filename(rank, tracer_name, cl_true, y_pred, abs_err)
        save_overlay(
            volume=x.squeeze(0).cpu().numpy(),
            heatmap=heatmap,
            meta={"model": "petresnet_attn", "kernel": kernel,
                  "tracer": tracer_name, "cl_true": cl_true,
                  "cl_pred": y_pred, "abs_err": abs_err},
            out_path=os.path.join(args.out_dir, out_name),
        )
        print(f"  [{rank+1}/{len(indices)}] {tracer_name:>3}  "
              f"true={cl_true:+7.2f}  pred={y_pred:+7.2f}  |err|={abs_err:5.2f}  "
              f"→ {out_name}")

    print(f"\n✓ Done → {args.out_dir}")


if __name__ == "__main__":
    main()
