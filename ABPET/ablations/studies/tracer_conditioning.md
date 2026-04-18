# Tracer Conditioning Ablation

> **Status:** draft — numbers confirmed, prose to finalize.

## Question

The canonical submission conditions on tracer identity in three places:

1. **Input-level** — `TracerNorm`: a per-tracer learned (γ, β) affine
   rescale on the raw volume.
2. **Per-stage** — `FiLMBlock` at every residual stage (feature-wise
   linear modulation using a tracer embedding).
3. **Head-level** — tracer embedding concatenated into the regression
   head.

How much of the accuracy comes from each level?

## Design

We train `PETResNetNoFiLM` — identical to `PETResNet` except that
(2) and (3) are removed while (1) `TracerNorm` is retained. Everything
else is held constant: same data splits, sampler, loss, optimizer,
scheduler, seed (42).

Config: [`../configs/train_no_tracernorm.yaml`](../configs/train_no_tracernorm.yaml).

Model: [`mygo_centiloid/model/petresnet_no_film.py`](../../mygo_centiloid/model/petresnet_no_film.py).

## Result

All MAE numbers below are from internal `dev/evaluate.py` (same scorer for
both rows). The submission's **final competition score** on the hackathon
leaderboard is 11.7916 CL.

| Variant | FiLM | Head emb | TracerNorm | Val MAE (internal) | Val Pearson r |
|---|---|---|---|---|---|
| Submission (`PETResNet`) | ✓ | ✓ | ✓ | 11.73 | 0.936 |
| `PETResNetNoFiLM` | — | — | ✓ | **9.03** | _tbd_ |

Removing the deeper conditioning pathways **improves** internal val MAE
by −2.70 CL.

## Interpretation

The Centiloid scale is explicitly designed to be tracer-agnostic at the
**output** — once expressed in CL, a number means the same thing
regardless of which of FBP / FBB / NAV / PIB was used. The only place
tracer identity genuinely matters is at the **input**, where raw
intensity scales differ.

Input-level `TracerNorm` handles that calibration cleanly. Adding
per-layer FiLM and a head-level tracer embedding on top gives the model
shortcut features (e.g. learning "this tracer usually has high CL
patients") that help in-sample but don't generalize. With only 2k
training samples, the extra capacity amplifies this shortcut more than
it helps extract signal.

## Secondary check

`ablations/scripts/ablate_tracer_norm.py` takes a trained
`PETResNetNoFiLM` checkpoint and re-runs inference with `TracerNorm`
forced to identity (γ=1, β=0). The MAE delta between the two runs
quantifies how much the **learned** per-tracer calibration contributes
on top of a simple min-max norm. Numbers pending.

## Limitations

- Single seed. We plan to re-run with seeds {0, 1, 42, 2026} and report
  mean ± std.
- Val set only. A held-out test MAE would strengthen the claim but we
  do not have access to the judges' test split.
- Does **not** retroactively change the competition submission.
