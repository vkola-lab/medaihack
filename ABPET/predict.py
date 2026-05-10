"""
Prediction script for Amyloid PET Centiloid Prediction.

Loads a trained checkpoint and runs inference on a CSV of samples,
outputting predicted centiloid scores.

Usage:
    python predict.py --csv /projectnb/medaihack/ABPET/data/val.csv --checkpoint /projectnb/medaihack/ABPET/medaihack/ABPET/checkpoints/best_model.pt --output predictions.csv
"""

import argparse
import torch
from torch.utils.data import DataLoader

from dataset import PETDataset
# TODO: if you replace BaselineCNN with your own architecture, update this import
# and the model instantiation below (search "# MODEL").
from model import BaselineCNN


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds = []

    for batch in loader:
        images, tracers = batch[0], batch[-1]
        images = images.to(device)
        tracers = tracers.to(device)

        with torch.amp.autocast(device.type):
            preds = model(images, tracers)

        all_preds.append(preds.cpu())

    return torch.cat(all_preds).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True,
                        help="CSV with npy_path and TRACER.AMY columns")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default="predictions.csv",
                        help="Path to save predictions")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Load checkpoint ---
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tracer_map = checkpoint["tracer_map"]
    num_tracers = checkpoint["num_tracers"]

    # --- Data ---
    dataset = PETDataset(args.csv, tracer_map=tracer_map)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # --- Model --- # MODEL
    model = BaselineCNN(num_tracers=num_tracers).to(device)

    # Handle torch.compile prefix in state_dict keys
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f"Loaded model from {args.checkpoint}\n")

    # --- Predict ---
    predictions = predict(model, loader, device)

    # --- Save ---
    out_df = dataset.df[["npy_path", "TRACER.AMY"]].copy()
    if "ID" in dataset.df.columns:
        out_df.insert(0, "ID", dataset.df["ID"])
    out_df["PREDICTED_CENTILOIDS"] = predictions

    out_df.to_csv(args.output, index=False)
    print(f"Saved {len(out_df)} predictions to {args.output}")


if __name__ == "__main__":
    main()
