"""
gradcam.py  —  EXPERIMENTAL, not part of the main pipeline

3D Grad-CAM overlays for any PETResNet variant checkpoint
(PETResNet, PETResNetNoGAP, PETResNetNoFiLM, PETResNetTracerNormOnly).

Methods (select with --method):
    hirescam         HiResCAM at a single layer.  Keeps `grad ⊙ activation`
                     before spatial aggregation — sharper & more faithful
                     than vanilla Grad-CAM at the same layer (Draelos 2021).
    gradcam          Vanilla Grad-CAM — gradients spatially averaged per
                     channel before weighting. Coarser; keep for sanity.
    fused  (default) HiResCAM at stages 2/3/4 fused via geometric mean.
                     Each layer contributes at its natural scale; overlap
                     concentrates where coarse (semantic) and fine
                     (structural) attention agree → finest map overall.

Accuracy stratification (--split_by_accuracy):
    Runs prediction on the full dataset, ranks by |error|, picks the best
    and worst N *per tracer* so no single tracer dominates either group,
    and writes them to  <out_dir>/best/  and  <out_dir>/worst/.
    Side-by-side `diff` of the two folders is the failure-mode diagnostic.

Usage:
    # Finest fused CAM on val set, split by accuracy, 6 best + 6 worst per tracer
    python dev/gradcam.py \\
        --config dev/config/predict.yaml \\
        --split_by_accuracy --n_samples 6 \\
        --out_dir results/gradcam/petresnet

    # Single-layer HiResCAM at stage3 (medium detail)
    python dev/gradcam.py --method hirescam --target_layer stage3

    # Vanilla Grad-CAM for comparison
    python dev/gradcam.py --method gradcam
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mygo_centiloid import (
    PETDataset,
    PETResNet, PETResNetNoGAP, PETResNetNoFiLM, PETResNetTracerNormOnly,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config + model
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    name = ckpt.get("model", "petresnet")
    num_tracers = ckpt["num_tracers"]

    if name == "petresnet_no_gap":
        model = PETResNetNoGAP(
            num_tracers  = num_tracers,
            emb_dim      = ckpt.get("emb_dim", 32),
            dropout_high = ckpt.get("dropout_high", 0.4),
            dropout_low  = ckpt.get("dropout_low",  0.2),
        ).to(device)
    elif name == "petresnet_no_film":
        model = PETResNetNoFiLM(
            num_tracers  = num_tracers,
            emb_dim      = ckpt.get("emb_dim", 32),
            dropout_high = ckpt.get("dropout_high", 0.4),
            dropout_low  = ckpt.get("dropout_low",  0.2),
        ).to(device)
    elif name == "petresnet_tracer_norm_only":
        model = PETResNetTracerNormOnly(
            num_tracers  = num_tracers,
            dropout_high = ckpt.get("dropout_high", 0.4),
            dropout_low  = ckpt.get("dropout_low",  0.2),
        ).to(device)
    else:
        model = PETResNet(
            num_tracers  = num_tracers,
            emb_dim      = ckpt.get("emb_dim", 32),
            dropout_high = ckpt.get("dropout_high", 0.4),
            dropout_low  = ckpt.get("dropout_low",  0.2),
        ).to(device)

    state = {k.replace("_orig_mod.", ""): v
             for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    model.eval()
    return model, name, ckpt


def resolve_target_layers(model, model_name: str,
                          single_layer: str | None,
                          fused: bool) -> dict:
    """Return {name: module} for one or more hook targets."""
    if fused:
        # Final feature at each "depth" — film* if present, else stage*.
        names = (["film2", "film3", "film4"] if model_name == "petresnet"
                 else ["stage2", "stage3", "stage4"])
    else:
        default = "film4" if model_name == "petresnet" else "stage4"
        names = [single_layer or default]

    missing = [n for n in names if not hasattr(model, n)]
    if missing:
        available = [n for n in ("film1", "film2", "film3", "film4",
                                 "stage1", "stage2", "stage3", "stage4")
                     if hasattr(model, n)]
        raise ValueError(f"Model {model_name} has no attribute(s) {missing}. "
                         f"Available: {available}")
    return {n: getattr(model, n) for n in names}


# ─────────────────────────────────────────────────────────────────────────────
# CAM core — HiResCAM + Grad-CAM, optional multi-layer fusion
# ─────────────────────────────────────────────────────────────────────────────

class CAM3D:
    """Hooks a set of target modules and computes a fused CAM per sample."""

    def __init__(self, model, target_modules: dict, method: str = "hirescam"):
        assert method in ("gradcam", "hirescam")
        self.model   = model
        self.method  = method
        self.targets = target_modules
        self._act: dict[str, torch.Tensor] = {}
        self._grad: dict[str, torch.Tensor] = {}
        self._hooks: list = []

        for name, mod in target_modules.items():
            self._hooks.append(mod.register_forward_hook(self._fwd(name)))
            self._hooks.append(mod.register_full_backward_hook(self._bwd(name)))

    def _fwd(self, name):
        def hook(_m, _i, out): self._act[name] = out.detach()
        return hook

    def _bwd(self, name):
        def hook(_m, _gi, go): self._grad[name] = go[0].detach()
        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()

    # ── single-layer CAM ──
    def _layer_cam(self, A: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        if self.method == "gradcam":
            alpha = G.mean(dim=(2, 3, 4), keepdim=True)
            cam = F.relu((alpha * A).sum(dim=1, keepdim=True))
        else:  # hirescam
            cam = F.relu((G * A).sum(dim=1, keepdim=True))
        return cam

    @staticmethod
    def _norm01(cam: torch.Tensor) -> torch.Tensor:
        cam = cam - cam.amin(dim=(2, 3, 4), keepdim=True)
        cam = cam / (cam.amax(dim=(2, 3, 4), keepdim=True) + 1e-8)
        return cam

    def compute(self, x: torch.Tensor, tracer_idx: torch.Tensor):
        """One forward+backward pass → (y_pred, heatmap numpy of shape x.shape[2:])."""
        self.model.zero_grad(set_to_none=True)
        y = self.model(x, tracer_idx)
        y.sum().backward()

        cams: list[torch.Tensor] = []
        for name in self.targets:
            cam = self._layer_cam(self._act[name], self._grad[name])
            cam = F.interpolate(cam, size=x.shape[2:],
                                mode="trilinear", align_corners=False)
            cams.append(self._norm01(cam))

        if len(cams) == 1:
            fused = cams[0]
        else:
            # Geometric mean — log-sum-exp for numerical safety.
            log_stack = torch.stack([torch.log(c + 1e-8) for c in cams], dim=0)
            fused = torch.exp(log_stack.mean(dim=0))
            fused = self._norm01(fused)

        return float(y.item()), fused.squeeze().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
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
        f"{meta['method']}  |  target={meta['target']}  |  model={meta['model']}\n"
        f"tracer={meta['tracer']}  |  CL_true={meta['cl_true']:+.1f}  "
        f"CL_pred={meta['cl_pred']:+.1f}  |  |err|={meta['abs_err']:.1f} CL",
        fontsize=11, fontweight="bold",
    )

    for row, (title, (bg, hm)) in enumerate(slices.items()):
        axes[row, 0].imshow(bg, cmap="gray", origin="lower")
        axes[row, 0].set_title(f"{title} — PET", fontsize=10)
        axes[row, 1].imshow(hm, cmap="hot", origin="lower", vmin=0, vmax=1)
        axes[row, 1].set_title(f"{title} — CAM", fontsize=10)
        axes[row, 2].imshow(bg, cmap="gray", origin="lower")
        axes[row, 2].imshow(hm, cmap="hot", origin="lower",
                            alpha=0.45, vmin=0, vmax=1)
        axes[row, 2].set_title(f"{title} — overlay", fontsize=10)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Sample picking
# ─────────────────────────────────────────────────────────────────────────────

def pick_spread_indices(dataset, n_samples: int) -> list[int]:
    """Evenly spaced across the CL distribution (fallback when no GT)."""
    if not dataset.has_targets:
        return list(range(min(n_samples, len(dataset))))
    cl = dataset.centiloids.numpy()
    order = np.argsort(cl)
    if n_samples >= len(dataset):
        return order.tolist()
    picks = np.linspace(0, len(order) - 1, n_samples).round().astype(int)
    return order[picks].tolist()


@torch.no_grad()
def predict_full_dataset(model, dataset, device, batch_size: int, num_workers: int):
    """Return (y_true, y_pred, tracer_ids) as numpy arrays."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    model.eval()
    y_true_chunks, y_pred_chunks, tid_chunks = [], [], []
    for batch in loader:
        if len(batch) == 3:
            images, targets, tids = batch
            y_true_chunks.append(targets.numpy())
        else:
            images, tids = batch
            y_true_chunks.append(np.full(len(images), np.nan, dtype=np.float32))
        images_d = images.to(device, non_blocking=True)
        tids_d   = tids.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type,
                                enabled=(device.type == "cuda")):
            preds = model(images_d, tids_d)
        y_pred_chunks.append(preds.float().cpu().numpy())
        tid_chunks.append(tids.numpy())
    return (np.concatenate(y_true_chunks),
            np.concatenate(y_pred_chunks),
            np.concatenate(tid_chunks))


def pick_best_worst_per_tracer(y_true, y_pred, tracer_ids, n_per_tracer: int):
    """Return (best_idx, worst_idx) stratified per tracer."""
    abs_err = np.abs(y_pred - y_true)
    best, worst = [], []
    for t in sorted(set(tracer_ids.tolist())):
        mask = tracer_ids == t
        idxs = np.where(mask)[0]
        order = idxs[np.argsort(abs_err[mask])]  # ascending |err|
        take = min(n_per_tracer, len(order) // 2)
        if take == 0:
            continue
        best.extend(order[:take].tolist())
        worst.extend(order[-take:].tolist())
    return best, worst, abs_err


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _filename(rank: int, tracer: str, cl_true: float, cl_pred: float,
              abs_err: float) -> str:
    def _s(v): return f"{v:+06.1f}".replace("+", "p").replace("-", "n").replace(".", "_")
    return (f"{rank:02d}_{tracer}_"
            f"cl{_s(cl_true)}_pred{_s(cl_pred)}_err{abs_err:04.1f}.png")


def run_group(cam: CAM3D, model, dataset, indices, device,
              meta_static: dict, group_dir: str,
              abs_err_all: np.ndarray | None) -> None:
    os.makedirs(group_dir, exist_ok=True)
    inv_tracer = {v: k for k, v in dataset.tracer_map.items()}
    print(f"\n  → {group_dir}  (n={len(indices)})")
    for rank, idx in enumerate(indices):
        sample = dataset[idx]
        if len(sample) == 3:
            image, cl_true, tid = sample
            cl_true = float(cl_true)
        else:
            image, tid = sample
            cl_true = float("nan")

        x = image.unsqueeze(0).to(device).float()
        t = tid.unsqueeze(0).to(device)
        y_pred, heatmap = cam.compute(x, t)
        tracer_name = inv_tracer[int(tid.item())]
        abs_err = (abs_err_all[idx] if abs_err_all is not None
                   else abs(y_pred - cl_true))

        out_name = _filename(rank, tracer_name, cl_true, y_pred, abs_err)
        save_overlay(
            volume=x.squeeze(0).detach().cpu().numpy(),
            heatmap=heatmap,
            meta={**meta_static, "tracer": tracer_name,
                  "cl_true": cl_true, "cl_pred": y_pred, "abs_err": abs_err},
            out_path=os.path.join(group_dir, out_name),
        )
        print(f"    [{rank+1:>2}/{len(indices)}] {tracer_name:>3}  "
              f"true={cl_true:+7.2f}  pred={y_pred:+7.2f}  |err|={abs_err:5.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config",       default="dev/config/predict.yaml")
    ap.add_argument("--csv",          default=None, help="Override paths.csv")
    ap.add_argument("--checkpoint",   default=None, help="Override paths.checkpoint")
    ap.add_argument("--out_dir",      default="results/gradcam")
    ap.add_argument("--n_samples",    type=int, default=6,
                    help="# samples per group (per tracer when --split_by_accuracy)")
    ap.add_argument("--method",       choices=["gradcam", "hirescam", "fused"],
                    default="fused",
                    help="CAM method. 'fused' = HiResCAM across stages 2/3/4.")
    ap.add_argument("--target_layer", default=None,
                    help="Single-layer target (ignored when method=fused). "
                         "Default: film4 (petresnet) / stage4 (no_film).")
    ap.add_argument("--split_by_accuracy", action="store_true",
                    help="Render top-N best and top-N worst predicted samples "
                         "per tracer, into <out_dir>/best/ and <out_dir>/worst/.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    csv_path   = args.csv        or cfg["paths"]["csv"]
    ckpt_path  = args.checkpoint or cfg["paths"]["checkpoint"]
    batch_size = cfg["inference"]["batch_size"]
    num_workers = cfg["inference"]["num_workers"]

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────
    model, model_name, ckpt = load_model(ckpt_path, device)
    fused = (args.method == "fused")
    method_core = "hirescam" if fused else args.method
    target_modules = resolve_target_layers(
        model, model_name, args.target_layer, fused
    )
    target_label = "+".join(target_modules.keys())
    method_label = "HiResCAM-fused" if fused else method_core.upper()

    print(f"Model        : {model_name}")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"Method       : {method_label}")
    print(f"Target layer : {target_label}")

    # ── Data ──────────────────────────────────────────────────────────────
    dataset = PETDataset(csv_path, tracer_map=ckpt["tracer_map"])

    meta_static = {
        "method": method_label, "target": target_label, "model": model_name,
    }

    cam = CAM3D(model, target_modules, method=method_core)
    try:
        if args.split_by_accuracy:
            if not dataset.has_targets:
                raise ValueError("--split_by_accuracy requires the CSV to have "
                                 "a CENTILOIDS column (ground truth).")
            print("\nPredicting full dataset for accuracy stratification...")
            y_true, y_pred, tids = predict_full_dataset(
                model, dataset, device, batch_size, num_workers
            )
            best, worst, abs_err = pick_best_worst_per_tracer(
                y_true, y_pred, tids, args.n_samples
            )
            print(f"  best : {len(best)} samples "
                  f"(mean |err|={abs_err[best].mean():.2f} CL)")
            print(f"  worst: {len(worst)} samples "
                  f"(mean |err|={abs_err[worst].mean():.2f} CL)")
            run_group(cam, model, dataset, best, device, meta_static,
                      os.path.join(args.out_dir, "best"), abs_err)
            run_group(cam, model, dataset, worst, device, meta_static,
                      os.path.join(args.out_dir, "worst"), abs_err)
        else:
            indices = pick_spread_indices(dataset, args.n_samples)
            run_group(cam, model, dataset, indices, device, meta_static,
                      args.out_dir, None)
    finally:
        cam.remove()

    print(f"\n✓ Done → {args.out_dir}")


if __name__ == "__main__":
    main()
