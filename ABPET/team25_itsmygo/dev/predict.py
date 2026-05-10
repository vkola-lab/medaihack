"""
Prediction script for Amyloid PET Centiloid Prediction.

Entry point:
    python dev/predict.py --config dev/config/predict.yaml

The YAML holds all defaults. --csv / --checkpoint / --output override the
corresponding YAML fields — used by predict.sh (the judge entry point).
Model architecture is restored from the checkpoint itself.
"""

import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mygo_centiloid import (
    PETDataset,
    PETResNet, PETResNetNoGAP, PETResNetNoFiLM, PETResNetTracerNormOnly,
)
from mygo_centiloid.utils import (
    make_run_dir, write_config, write_metrics, append_registry,
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    required = {
        "paths":     ["csv", "checkpoint", "output", "log_dir"],
        "run":       ["name"],
        "inference": ["batch_size", "num_workers"],
    }
    for section, keys in required.items():
        if section not in cfg:
            raise KeyError(f"Config missing section: [{section}]")
        for k in keys:
            if k not in cfg[section]:
                raise KeyError(f"Config missing key: [{section}].{k}")
    return cfg


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds = []

    for batch in loader:
        images, tracers = batch[0], batch[-1]
        images  = images.to(device,  non_blocking=True)
        tracers = tracers.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type,
                                enabled=(device.type == "cuda")):
            preds = model(images, tracers)

        all_preds.append(preds.cpu())

    return torch.cat(all_preds).numpy()


def main():
    parser = argparse.ArgumentParser(description="Predict centiloid scores.")
    parser.add_argument("--config", default="dev/config/predict.yaml",
                        help="Path to YAML inference config.")
    # Judge-entry overrides (also used for ad-hoc runs)
    parser.add_argument("--csv",        default=None, help="Override paths.csv")
    parser.add_argument("--checkpoint", default=None, help="Override paths.checkpoint")
    parser.add_argument("--output",     default=None, help="Override paths.output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    P, R, I = cfg["paths"], cfg["run"], cfg["inference"]

    csv_path   = args.csv        or P["csv"]
    ckpt_path  = args.checkpoint or P["checkpoint"]
    output     = args.output     or P["output"]
    batch_size = I["batch_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    tracer_map   = ckpt["tracer_map"]
    num_tracers  = ckpt["num_tracers"]
    model_name   = ckpt.get("model", "petresnet")
    emb_dim      = ckpt.get("emb_dim", 32)
    dropout_high = ckpt.get("dropout_high", 0.4)
    dropout_low  = ckpt.get("dropout_low",  0.2)

    # ── Run folder + registry setup ───────────────────────────────────────
    run_dir = make_run_dir(
        run_type="predict", model=model_name,
        base=P["log_dir"], run_name=R["name"],
    )
    write_config(run_dir, {
        "run_type": "predict", "model": model_name,
        "config_path": args.config, "csv": csv_path,
        "checkpoint": ckpt_path, "output": output, **cfg,
    })
    print(f"Run dir: {run_dir}")
    print(f"Config : {args.config}\n")

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = PETDataset(csv_path, tracer_map=tracer_map)
    loader  = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=I["num_workers"], pin_memory=True,
    )

    # ── Model (dispatch on checkpoint's recorded architecture) ────────────
    if model_name == "petresnet_no_gap":
        model = PETResNetNoGAP(
            num_tracers  = num_tracers,
            emb_dim      = emb_dim,
            dropout_high = dropout_high,
            dropout_low  = dropout_low,
        ).to(device)
    elif model_name == "petresnet_no_film":
        model = PETResNetNoFiLM(
            num_tracers  = num_tracers,
            emb_dim      = emb_dim,
            dropout_high = dropout_high,
            dropout_low  = dropout_low,
        ).to(device)
    elif model_name == "petresnet_tracer_norm_only":
        model = PETResNetTracerNormOnly(
            num_tracers  = num_tracers,
            dropout_high = dropout_high,
            dropout_low  = dropout_low,
        ).to(device)
    else:
        model = PETResNet(
            num_tracers  = num_tracers,
            emb_dim      = emb_dim,
            dropout_high = dropout_high,
            dropout_low  = dropout_low,
        ).to(device)
    print(f"Model: {model_name}")
    if ckpt.get("freeze_tracer_norm"):
        print("TracerNorm: was frozen at identity during training (ablation ckpt)")
    print()

    # Strip torch.compile prefix if present
    state_dict = {k.replace("_orig_mod.", ""): v
                  for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {ckpt_path}\n")

    model.summary(input_size=(batch_size, 1, 128, 128, 128))

    # ── Predict ───────────────────────────────────────────────────────────
    predictions = predict(model, loader, device)

    out_df = dataset.df[["npy_path", "TRACER.AMY"]].copy()
    if "ID" in dataset.df.columns:
        out_df.insert(0, "ID", dataset.df["ID"])
    out_df["PREDICTED_CENTILOIDS"] = predictions

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    out_df.to_csv(output, index=False)
    print(f"Saved {len(out_df)} predictions → {output}")

    # ── Registry ──────────────────────────────────────────────────────────
    metrics = {
        "run_type":   "predict",
        "model":      model_name,
        "n_samples":  int(len(out_df)),
        "checkpoint": ckpt_path,
        "output":     output,
    }
    write_metrics(run_dir, metrics)
    append_registry({
        "type":    "predict",
        "model":   model_name,
        "run_dir": str(run_dir),
        "ckpt":    ckpt_path,
        "output":  output,
        "metrics": {"n_samples": int(len(out_df))},
    })


if __name__ == "__main__":
    main()
