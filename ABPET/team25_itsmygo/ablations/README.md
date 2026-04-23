# Ablations

Post-hackathon factorial ablation over the three tracer-conditioning sites
in `PETResNet` (TracerNorm, per-stage FiLM, tracer-embedding GAP concat).
The canonical submission (`PETResNet` with all three sites active —
**leaderboard MAE 11.7916 CL**, internal 11.79) remains the reference;
everything here is the post-hoc factorial reported in §3.5 / §4.3 of the
MYGO-Centiloid report.

Model variants (`petresnet_no_gap.py`, `petresnet_no_film.py`,
`petresnet_tracer_norm_only.py`) live under
[`mygo_centiloid/model/`](../mygo_centiloid/model/) so that `dev/train.py`
can import them by name from the package. Only ablation-specific
**configs**, **scripts**, and **write-ups** live here.

## Results

All runs use the same data splits, 100 epochs, AdamW 1e-4, batch 4,
inverse-frequency sampler, and seed 42. Metric is Val MAE in CL (lower is
better) on the 500-sample validation set, scored with internal
`dev/evaluate.py` for apples-to-apples comparison against the submission's
internal number (11.79).

| Variant | TracerNorm | FiLM | GAP concat | Model class | Val MAE (internal) | Δ vs submission |
|---|---|---|---|---|---|---|
| Submission (reference) | ✓ | ✓ | ✓ | `PETResNet` | 11.79 (leaderboard: 11.7916) | — |
| −GAP | ✓ | ✓ | — | `PETResNetNoGAP` | 10.83 | −0.96 |
| −FiLM | ✓ | — | ✓ | `PETResNetNoFiLM` | 10.49 | −1.30 |
| TracerNorm-only | ✓ | — | — | `PETResNetTracerNormOnly` | **9.03** | **−2.76** |

Lower MAE is better. Numbers reproduce report Table 4 (`Val MAE (CL)` row).
See [`studies/tracer_conditioning.md`](studies/tracer_conditioning.md) for
the mechanistic interpretation (FiLM is the dominant capacity sink; GAP
concat is a secondary modulator).

## Layout

```text
ablations/
├── README.md                           this file — index + headline numbers
├── configs/
│   ├── no_gap.yaml                     −GAP      (TracerNorm + FiLM)
│   ├── no_film.yaml                    −FiLM     (TracerNorm + GAP concat)
│   └── tracer_norm_only.yaml           TN-only   (TracerNorm only)
├── scripts/
│   ├── ablate_tracer_norm.py           toggle TracerNorm → identity at inference
│   └── inspect_tracer_norm.py          dump learned γ / β per tracer
└── studies/
    └── tracer_conditioning.md          4-variant factorial write-up
```

## Running an ablation

All commands are relative to the `team25_itsmygo/` folder and assume the
package is installed (`pip install -e .`) and data is linked at `./data/`.

```bash
# −GAP — TracerNorm + FiLM, no tracer-embedding concat
bash dev/train.sh ablations/configs/no_gap.yaml

# −FiLM — TracerNorm + tracer-embedding concat, no FiLM
bash dev/train.sh ablations/configs/no_film.yaml

# TracerNorm-only — TracerNorm as the only conditioning site
bash dev/train.sh ablations/configs/tracer_norm_only.yaml
```

Post-training inspection:

```bash
# Compare "with TracerNorm" vs "TracerNorm → identity" at inference
python ablations/scripts/ablate_tracer_norm.py \
    --csv        data/val.csv \
    --checkpoint checkpoints/tracer_norm_only/best_model.pt

# Dump the learned per-tracer γ / β (Table 4 of the report)
python ablations/scripts/inspect_tracer_norm.py \
    --ckpt checkpoints/tracer_norm_only/best_model.pt
```

## Scope and caveats

- **Single seed (42).** Numbers above are from one seed; the study page
  flags multi-seed variance as future work.
- **No change to the canonical submission.** These results are post-hoc
  and exist to document the factorial decoupling between predictive
  performance and physical interpretability (report §4.3).
- **Experimental scripts.** Scripts under `scripts/` carry an
  "EXPERIMENTAL, not part of the main pipeline" banner in their
  docstrings and are not wired into the judges' `predict.sh` path.
