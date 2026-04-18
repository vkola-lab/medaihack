# Stage-4 Spatial Attention

> **Status:** draft — numbers confirmed for vanilla CBAM, gated variant pending.

## Question

The Klunk Centiloid project defines a fixed cortical target VOI (CTX)
for the SUVR that the scale is derived from. A natural question is
whether giving the network an explicit, learnable analog of that
region-selection mechanism helps accuracy on top of the No-FiLM backbone.

## Design

We take the No-FiLM backbone (`PETResNetNoFiLM`, the current best
variant — see [tracer_conditioning.md](tracer_conditioning.md)) and
insert a CBAM-style spatial attention module on the **stage-4** output
(the final `(B, 512, 4, 4, 4)` feature map before global pooling):

```
spatial_attn(x) = sigmoid( Conv3d( concat[mean(x), max(x)] ) )
out = x * spatial_attn(x)                     # vanilla gate
out = x * (1 + spatial_attn(x))               # gated / residual variant
```

- **Vanilla** — `PETResNetAttn` in
  [`mygo_centiloid/model/petresnet_attn.py`](../../mygo_centiloid/model/petresnet_attn.py),
  config [`../configs/stage3_attn.yaml`](../configs/stage3_attn.yaml).
- **Gated (zero-init)** — `PETResNetAttnGated` in
  [`mygo_centiloid/model/petresnet_attn_gated.py`](../../mygo_centiloid/model/petresnet_attn_gated.py),
  config [`../configs/stage3_attn_gated.yaml`](../configs/stage3_attn_gated.yaml).
  Starts as an effective identity map so the module cannot hurt
  epoch-1 behavior.

## Result

MAE from internal `dev/evaluate.py`; same scorer as the submission's
internal number (11.73). The submission's final leaderboard score is
11.7916 CL.

| Variant | Early convergence | Best Val MAE (internal) |
|---|---|---|
| No-FiLM baseline | reference | **9.03** |
| + Stage-4 CBAM (vanilla) | ~2× faster to crossing 15 CL | 10.02 |
| + Stage-4 CBAM (gated, zero-init) | stable from epoch 1 | _pending_ |

## Interpretation

The vanilla variant **accelerates early training** — by roughly epoch 10
it reaches the MAE the No-FiLM baseline needs ~20 epochs for — which is
consistent with the attention gate acting as a strong inductive bias
toward cortex-like regions. But by the time both runs finish, the
baseline has caught up and passed it.

Reading: at 2k samples the backbone already learns a region-selective
prior implicitly; the hard-coded attention module doesn't add
information, and the extra sigmoid gate on a residual path introduces
late-training oscillation (visible after epoch ~55 in the
`stage3_attn` run) that the zero-init gated variant was designed to
fix.

## Visualization

`ablations/scripts/inspect_attention.py` renders the stage-4
attention map (upsampled to PET resolution) over axial / coronal /
sagittal slices. Default target CLs are {0, 30, 100} — a young
control, a just-positive case, and an AD-typical case. Qualitatively
the high-CL attention concentrates on bilateral frontal / precuneus
cortex, consistent with the Klunk CTX VOI.

## Limitations

- Single seed. Multi-seed re-run queued.
- The "2× faster convergence" claim is from one run each; could be
  noise in the early-training regime.
- Vanilla gate multiplies by a value in [0, 1], which can only
  *suppress* features — hence the zero-init gated variant, which
  multiplies by [1, 2] and is guaranteed to start as identity. We
  expect the gated variant to close the gap but not beat the baseline.
