# Ablations

Post-hackathon studies that probe which components of `PETResNet` actually
drive the competition result. The canonical submission (`PETResNet` with
TracerNorm + per-stage FiLM + head-level tracer embedding, **leaderboard
MAE 11.7916 CL** — final competition score; 11.73 internal) remains the
reference; everything here is exploratory.

Model variants (`petresnet_no_film.py`, `petresnet_attn.py`,
`petresnet_attn_gated.py`) live under
[`mygo_centiloid/model/`](../mygo_centiloid/model/) so that
`dev/train.py` can import them by name from the package. Only
ablation-specific **configs**, **scripts**, and **write-ups** live here.

## Results

All runs use the same data splits, 100 epochs, AdamW 1e-4, batch 4,
inverse-frequency sampler, and seed 42. Metric is Val MAE in CL (lower
is better) on the 500-sample validation set, scored with internal
`dev/evaluate.py` for apples-to-apples comparison against the submission's
internal number (11.73). The submission's **final competition score** on
the hackathon leaderboard is **11.7916 CL**.

| # | Study | Model variant | Val MAE (internal) | Δ vs submission | Notes |
|---|-------|---------------|--------------------|-----------------|-------|
| 0 | Submission (reference) | `PETResNet` (TracerNorm + FiLM + tracer emb) | **11.73** (leaderboard: 11.7916) | — | canonical |
| 1 | [Tracer conditioning](studies/tracer_conditioning.md) | `PETResNetNoFiLM` (TracerNorm only) | **9.03** | −2.70 | dropping FiLM + head emb helps |
| 2 | [Spatial attention](studies/attention_study.md) | `PETResNetAttn` (stage-4 CBAM, on No-FiLM base) | 10.02 | −1.71 | ~2× faster early convergence; no final gain |
| 2b | [Spatial attention (gated)](studies/attention_study.md) | `PETResNetAttnGated` (residual zero-init) | _pending_ | _pending_ | stabilizes epoch-1 Pearson r |

Lower MAE is better. Per-tracer breakdowns and seed ablations are in the
individual study write-ups.

## Layout

```text
ablations/
├── README.md                           this file — index + headline numbers
├── configs/
│   ├── train_no_tracernorm.yaml        no-FiLM / TracerNorm-frozen variant
│   ├── stage3_attn.yaml                No-FiLM + stage-4 CBAM attention
│   └── stage3_attn_gated.yaml          same, residual + zero-init
├── scripts/
│   ├── ablate_tracer_norm.py           toggle TracerNorm → identity at inference
│   ├── inspect_tracer_norm.py          dump learned γ / β per tracer
│   └── inspect_attention.py            render stage-4 attention maps over PET slices
└── studies/
    ├── tracer_conditioning.md          No-FiLM write-up (method, results, interpretation)
    └── attention_study.md              Stage-4 CBAM write-up
```

## Running an ablation

All commands are relative to the ABPET folder and assume the package is
installed (`pip install -e .`) and data is linked at `./data/`.

```bash
# 1. Tracer conditioning — train No-FiLM variant
bash dev/train.sh ablations/configs/train_no_tracernorm.yaml

# 2. Stage-4 spatial attention — train CBAM variant on No-FiLM base
bash dev/train.sh ablations/configs/stage3_attn.yaml

# 2b. Residual / zero-init attention
bash dev/train.sh ablations/configs/stage3_attn_gated.yaml
```

Post-training inspection:

```bash
# Compare "with TracerNorm" vs "TracerNorm → identity" at inference
python ablations/scripts/ablate_tracer_norm.py \
    --csv        data/val.csv \
    --checkpoint checkpoints/no_film/best_model.pt

# Dump the learned per-tracer γ / β
python ablations/scripts/inspect_tracer_norm.py \
    --ckpt checkpoints/no_film/best_model.pt

# Render stage-4 attention maps on representative samples
python ablations/scripts/inspect_attention.py \
    --config     dev/config/predict.yaml \
    --checkpoint checkpoints/stage3_attn/best_model.pt \
    --out_dir    results/attention/stage3_attn
```

## Scope and caveats

- **Single seed (42).** Numbers above are from one seed; the study pages
  report multi-seed results where available.
- **No change to the canonical submission.** These results are not used
  to retroactively pick a winner — they exist to document what we
  learned during post-competition analysis.
- **Exploratory.** Scripts under `scripts/` carry an "EXPERIMENTAL, not
  part of the main pipeline" banner in their docstrings and are not
  wired into the judges' `predict.sh` path.
