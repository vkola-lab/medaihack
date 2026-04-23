# Tracer Conditioning Ablation

> Matches the four-variant factorial reported in §3.5 / §4.3 of the
> MYGO-Centiloid report.

## Question

The canonical submission conditions on tracer identity in three places:

1. **Input-level** — `TracerNorm`: a per-tracer learned (γ_t, β_t) affine
   rescale on the raw volume (8 learned scalars total).
2. **Per-stage feature-level** — `FiLMBlock` at every residual stage
   (feature-wise linear modulation driven by a shared tracer embedding).
3. **Head-level** — tracer embedding concatenated to the GAP output
   before the regression head.

How much of the accuracy comes from each site?

## Design

We run a four-variant factorial over (FiLM × GAP-concat) on top of a
TracerNorm backbone. All runs share the same data splits, sampler, loss,
optimizer, scheduler, augmentation, and seed (42).

| Variant | TracerNorm | FiLM | GAP concat | Model class | Config |
|---|---|---|---|---|---|
| Submission | ✓ | ✓ | ✓ | `PETResNet` | [`dev/config/train.yaml`](../../dev/config/train.yaml) |
| −GAP | ✓ | ✓ | — | `PETResNetNoGAP` | [`../configs/no_gap.yaml`](../configs/no_gap.yaml) |
| −FiLM | ✓ | — | ✓ | `PETResNetNoFiLM` | [`../configs/no_film.yaml`](../configs/no_film.yaml) |
| TracerNorm-only | ✓ | — | — | `PETResNetTracerNormOnly` | [`../configs/tracer_norm_only.yaml`](../configs/tracer_norm_only.yaml) |

Model sources:

- [`mygo_centiloid/model/petresnet_film.py`](../../mygo_centiloid/model/petresnet_film.py) — submission
- [`mygo_centiloid/model/petresnet_no_gap.py`](../../mygo_centiloid/model/petresnet_no_gap.py) — `−GAP`
- [`mygo_centiloid/model/petresnet_no_film.py`](../../mygo_centiloid/model/petresnet_no_film.py) — `−FiLM`
- [`mygo_centiloid/model/petresnet_tracer_norm_only.py`](../../mygo_centiloid/model/petresnet_tracer_norm_only.py) — TracerNorm-only

## Result

All MAE numbers below are from internal `dev/evaluate.py` on the 500-subject
validation split (same scorer used for the submission's 11.73 internal
number). The submission's **final competition score** on the hackathon
leaderboard is 11.7916 CL.

| Variant | TracerNorm | FiLM | GAP concat | Val MAE (internal) | Δ vs submission |
|---|---|---|---|---|---|
| Submission (`PETResNet`) | ✓ | ✓ | ✓ | 11.79 | — |
| `PETResNetNoGAP` (`−GAP`) | ✓ | ✓ | — | 10.83 | −0.96 |
| `PETResNetNoFiLM` (`−FiLM`) | ✓ | — | ✓ | 10.49 | −1.30 |
| `PETResNetTracerNormOnly` | ✓ | — | — | **9.03** | **−2.76** |

Numbers reproduce report Table 4 (`Val MAE (CL)` row).

## Interpretation

The Centiloid scale is explicitly designed to be tracer-agnostic at the
**output** — once expressed in CL, a number means the same thing regardless
of which of FBP / FBB / NAV / PIB was used. The only place tracer identity
genuinely matters is at the **input**, where raw intensity scales differ.

Input-level `TracerNorm` handles that calibration cleanly. Adding per-layer
FiLM and a head-level tracer embedding on top provides additional capacity
that, under end-to-end optimization on 2k training samples, absorbs
tracer-specific signal that would otherwise sharpen TracerNorm's alignment
with the published CL-per-SUVR conversion formulas (report §4.3, Fig. 6).

FiLM is the dominant capacity sink: removing it alone recovers most of the
physical alignment (Pearson r of learned γ_t vs published slope jumps from
−0.71 to +0.69). Removing the GAP concat in addition sharpens alignment to
r = +0.93.

## Secondary check

[`ablate_tracer_norm.py`](../scripts/ablate_tracer_norm.py) takes any of the
four checkpoints and re-runs inference with `TracerNorm` forced to identity
(γ=1, β=0). The MAE delta quantifies how much the **learned** per-tracer
calibration contributes on top of the per-volume min-max normalization
applied during preprocessing.

[`inspect_tracer_norm.py`](../scripts/inspect_tracer_norm.py) dumps the
learned (γ_t, β_t) values for side-by-side comparison with the published
CL-per-SUVR parameters (report Table 4).

## Limitations

- Single seed (42). Multi-seed reruns over {0, 1, 42, 2026} are future
  work; report §6.3 notes seed-variance analysis is a natural next step.
- Val set only. A held-out test MAE would strengthen the claim but the
  judges' test split is not available to participants.
- Does **not** retroactively change the competition submission.
