"""
inspect_tracer_norm.py  —  EXPERIMENTAL, not part of the main pipeline

Dumps the learned per-tracer scale (γ) / shift (β) from a trained checkpoint.
Works for both PETResNet (with FiLM) and PETResNetNoFiLM — both expose
`tracer_norm.scale.weight` and `tracer_norm.shift.weight`.

Usage:
    python ablations/scripts/inspect_tracer_norm.py --ckpt checkpoints/no_film/best_model.pt
"""

import argparse
import sys
import torch


# Alphabetical order — matches mygo_centiloid/data/dataset.py default,
# which builds tracer_map via sorted(df["TRACER.AMY"].unique()).
DEFAULT_TRACER_MAP = {"FBB": 0, "FBP": 1, "NAV": 2, "PIB": 3}


def load_state(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ("model_state_dict", "state_dict", "model"):
        if key in ckpt:
            return ckpt[key]
    return ckpt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")
    ap.add_argument("--tracer-map", default=None,
                    help="Optional override, e.g. 'FBB:0,FBP:1,NAV:2,PIB:3'")
    args = ap.parse_args()

    state = load_state(args.ckpt)

    scale_key = "tracer_norm.scale.weight"
    shift_key = "tracer_norm.shift.weight"
    missing = [k for k in (scale_key, shift_key) if k not in state]
    if missing:
        print(f"[!] Checkpoint does not contain TracerNorm params: {missing}")
        print(f"    Available top-level keys (sample): "
              f"{list(state.keys())[:8]}{'...' if len(state) > 8 else ''}")
        sys.exit(1)

    scale = state[scale_key].flatten().tolist()
    shift = state[shift_key].flatten().tolist()

    if args.tracer_map:
        tracer_map = {kv.split(":")[0]: int(kv.split(":")[1])
                      for kv in args.tracer_map.split(",")}
    else:
        tracer_map = DEFAULT_TRACER_MAP

    n = len(scale)
    oob = [name for name, idx in tracer_map.items() if idx >= n]
    if oob:
        print(f"[!] Tracer indices {oob} exceed embedding size {n}. "
              f"Check your tracer_map.")
        sys.exit(1)

    print(f"\nCheckpoint: {args.ckpt}")
    print(f"TracerNorm embedding size: {n}\n")
    print(f"{'Tracer':<6} {'scale (γ)':>12} {'shift (β)':>12}")
    print("-" * 32)
    for name, idx in tracer_map.items():
        print(f"{name:<6} {scale[idx]:>12.4f} {shift[idx]:>12.4f}")


if __name__ == "__main__":
    main()
