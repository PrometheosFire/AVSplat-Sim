# scene_084 ablation study — what is capping reconstruction quality?

## Why this study exists

Two full dynamic Difix-loop runs on scene_084 reached **24.24 dB PSNR** on
held-out frames, against the 26+ commonly reported by AV 3DGS papers. Before
chasing that gap with guesses, four measurements narrowed the search:

**1. Capacity is not the limit.** Raising the MCMC cap from 250k to 2M Gaussians
bought only +1.6 dB (22.31 → 23.92), the saturating end of the curve. 2M
Gaussians over an 86 m path is already dense.

> ⚠️ **This premise was wrong, and the study disproved it.** `abl_cap3m` turned
> out to be the best single lever (+0.237 dB). +1.6 dB for 8× capacity was a
> saturating curve read too early, not a ceiling. See Analysis §2.

**2. The validation split is easy.** Consecutive frames are 0.24 m apart, so a
held-out frame sits ~0.24 m from its nearest training view. 24 dB on an
interpolation split that tight is genuinely improvable, not a hard ceiling.

**3. The error is not uniform across cameras.** Per-camera PSNR on the 45k
no-bank model spans **3.2 dB**:

| camera block | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| PSNR | 25.36 | 25.33 | 24.46 | **22.17** | 24.35 |

The best camera is already at 25.4. Most of the headroom is in dragging the
worst cameras up, not in lifting everything.

**4. It is not a global colour offset.** Fitting a per-channel affine between
render and ground truth (the multiNeRF metric convention) *lowers* PSNR by
0.24–0.42 dB; quadratic gains only ~0.1 dB. If a systematic per-image colour or
exposure error were costing PSNR, this would have recovered it.

| model | PSNR | cc-PSNR (affine) | cc-PSNR (quadratic) |
|---|---|---|---|
| A r3 (30k, bank=3) | 23.925 | 23.648 | 24.028 |
| B r0 (45k, no bank) | 24.336 | 24.093 | 24.444 |
| B r3 (45k, bank=3) | 24.236 | 23.995 | 24.367 |
| C r3 (250k cap) | 22.314 | 21.898 | 22.426 |

### Where that leaves the hypotheses

| hypothesis | status |
|---|---|
| global per-image colour / exposure offset | **ruled out** by (4) |
| spatially varying per-camera effects (vignetting, response curve) | **open** — an affine cannot express a radial vignette, and (3) is consistent with it |
| geometry / detail | **leading suspect** — what remains after (1), (2) and (4) |

The study is built around that: the geometry levers and the *per-camera*
photometric lever get priority, and the global-per-image correctors are included
as controls to confirm the negative rather than because they are expected to
win.

## How to run

**14 experiments: 10 single-lever variants + a 4-point step-count sweep**, plus
a follow-up push run. All run sequentially in one Hydra multirun.

```bash
PYTHONPATH=. envs/envs/env_gsplat/bin/python src/gsplat_training/train_splats.py -m \
  +experiment=abl_baseline,abl_depth,abl_ppisp,abl_reg_low,abl_steps_45k,abl_steps_60k,abl_steps_75k,abl_steps_90k,abl_ppisp_ctrl,abl_bilateral,abl_app,abl_antialiased,abl_depth_ppisp,abl_cap3m \
  hydra.sweep.dir=multirun/ablation_scene084 \
  'hydra.sweep.subdir=${exp_name}'
```

Verified end to end with `max_steps=1`: Hydra launches 14 jobs, all 14 produce
stats in named subdirectories.

The order is deliberate. `abl_baseline` first, since it is the reference for
every lever *and* the 30k point of the step curve. Then the three highest-prior
levers, so the core answers exist ~3 h in. Then the step sweep. Then the
controls. `abl_cap3m` last, as the most likely variant to OOM — Hydra's basic
launcher logs a failed job and continues, so a failure there costs only itself.

`PYTHONPATH=.` is required: `train_splats.py` imports `src.gsplat_training.*`,
and the orchestrators normally set it for their subprocesses. Without it the
sweep dies immediately on `ModuleNotFoundError: No module named 'src'`.

The push run is a separate launch. `PYTORCH_CUDA_ALLOC_CONF` is required — see
"Two memory bugs this exposed":

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
envs/envs/env_gsplat/bin/python src/gsplat_training/train_splats.py -m \
  +experiment=abl_push \
  hydra.sweep.dir=multirun/ablation_scene084_push \
  'hydra.sweep.subdir=${exp_name}'
```

Then collect the tables below with:

```bash
envs/envs/env_gsplat/bin/python scripts/experiments/collect_ablation.py \
  multirun/ablation_scene084
```

Results land in `multirun/ablation_scene084/<exp_name>/`, each with
`stats/val_step29999.json`, `renders/`, `ckpts/` and the resolved
`.hydra/config.yaml`.

## What is held constant

scene_084 · 5 cameras · `data_factor=2` (960×540) · `test_every=16` ·
`enable_dynamic=true` · MCMC `cap_max=2M` · **30 000 steps** · `steps_scaler=1.0`
· no pseudo-view bank.

30k rather than the earlier 45k so the lever variants stay affordable. Every
comparison here is **internal** — `abl_baseline` is the reference, not any
earlier run — so the absolute step count only has to be consistent, and it is.

The step-count sweep is the one deliberate exception: it varies `steps_scaler`
(and therefore the whole densification schedule) precisely to find out whether
30k is a fair operating point for the rest.

## VRAM

Peak memory per variant, as reported by the `train/mem` scalar:

| variant | Gaussians | peak |
|---|---|---|
| `abl_baseline` | 2M | 2.73 GB |
| `abl_bilateral` | 2M | 3.12 GB |
| `abl_app` | 2M | 5.09 GB |
| `abl_cap3m` | 3M | 5.09 GB |
| `abl_push` | 4M | 5.32 GB |

**Do not size a run from this table.** `train/mem` reports torch *allocated*
memory, and the binding constraint is *reserved* — the first push attempt OOMed
with 4.53 GiB allocated but 2.26 GiB sitting reserved-but-unallocated to
fragmentation. See "Two memory bugs this exposed"; anything approaching the
device limit needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and the
4M figure above was measured with it set.

Rigid Gaussians are the other term, and they scale with `steps_scaler` rather
than with the background cap — see the same section.

## The experiments

| # | variant | change | what it tests |
|---|---|---|---|
| 1 | `abl_baseline` | — | reference for every delta below |
| 2 | `abl_depth` | `depth_loss=true` | **geometry.** 200k COLMAP points and a precomputed `point_visibility.npz` already sit unused in the ncore dataset. Free supervision aimed straight at the leading suspect. |
| 3 | `abl_ppisp` | `post_processing=ppisp`, controller off | **per-camera photometry.** The only module with *spatially varying, per-camera* terms (vignetting, response curve), and the only one that corrects anything at eval — it detects a novel view and still applies the per-camera part. Targets finding (3) directly. |
| 4 | `abl_ppisp_ctrl` | + controller, activation 24k | whether a network predicting exposure/colour from the render beats identity on held-out frames — and what it costs, since the runner **freezes the background** past the activation step (24k geometry + 6k controller-only). |
| 5 | `abl_bilateral` | `post_processing=bilateral_grid` | **control.** Global-per-image, the class ruled out by (4), and skipped entirely at eval. Can only win by absorbing photometric variation during training and leaving cleaner geometry. |
| 6 | `abl_app` | `app_opt=true` | **control + diagnostic.** Also global-per-image, and falls back to a zero embedding at eval. Its MLP is expressive enough to absorb *geometry* error into appearance, so a large train/val divergence here indicates overfitting rather than a photometric gap. Newly possible — this used to raise `NotImplementedError` with rigid nodes. |
| 7 | `abl_antialiased` | `antialiased=true` | near-free; plausible on fisheye, where a Gaussian's projected footprint varies sharply across the frame. |
| 8 | `abl_reg_low` | `opacity_reg`, `scale_reg` 0.01 → 0.001 | **geometry.** Both regularisers push toward fewer, smaller, more transparent Gaussians. The background saturates `cap_max` exactly every run, so the question is not whether there are enough Gaussians but whether they are allowed to be sharp. |
| 9 | `abl_depth_ppisp` | 2 + 3 | are the two leading hypotheses complementary? Only interpretable next to 2 and 3 alone. |
| 11-14 | `abl_steps_45k` `abl_steps_60k` `abl_steps_75k` `abl_steps_90k` | `steps_scaler` 1.5 / 2 / 2.5 / 3 | **training length.** The cheapest explanation for 24 dB, and untested: is the model simply undertrained? Read as a curve against `abl_baseline` (30k). A flattening slope means training length is exhausted and the gap is structural; a steep one means every other variant here is being measured at the wrong operating point. `steps_scaler` rather than raw `max_steps` so the densification schedule stretches with it. |
| 10 | `abl_cap3m` | `cap_max` 2M → 3M | is capacity still binding? The cap is hit exactly every run, but (1) says returns are saturating. **Run last — a larger cap is the most likely variant to OOM**, and Hydra continues past a failed job (verified), so a failure there costs only itself. |

## Deliberately excluded

**`data_factor=1` (full 1920×1080).** Tempting, since "detail-limited" is the
leading hypothesis, but it does not belong in *this* sweep: changing the
resolution changes the **evaluation** resolution too, so its PSNR is not
comparable to the other ten — higher resolution generally scores lower simply
because there is more high-frequency detail to miss. It needs its own paired
comparison (train at factor 1, evaluate at factor 2 by downsampling, or compare
factor-1 vs factor-2 models on identical downsampled targets), plus ~3–4× the
runtime. Worth doing as a follow-up; not worth a slot here.

Also note it cannot currently be used with the Difix loop at all, so even a win
would not transfer to the pseudo-view pipeline without further work.

**`pose_opt`.** Retired feature, not planned for use.

---

# Results

All 14 completed. Collected with
`scripts/experiments/collect_ablation.py multirun/ablation_scene084`.

## Headline metrics

| variant | PSNR | Δ | cc-PSNR | SSIM | LPIPS | Δ LPIPS | bg #GS | rigid #GS | time |
|---|---|---|---|---|---|---|---|---|---|
| `abl_baseline` | 24.197 | — | 23.273 | 0.7962 | 0.2469 | — | 2,000,000 | 49,060 | 00:46:46 |
| `abl_depth` | 22.958 | -1.238 | 22.002 | 0.7756 | 0.2670 | +0.0201 | 2,000,000 | 44,492 | 00:49:52 |
| `abl_ppisp` | 24.094 | -0.103 | 23.384 | 0.8021 | 0.2362 | -0.0107 | 2,000,000 | 46,478 | 00:47:55 |
| `abl_ppisp_ctrl` | 23.827 | -0.370 | 22.859 | 0.7881 | 0.2574 | +0.0106 | 2,000,000 | 121,812 | 00:42:29 |
| `abl_bilateral` | 22.550 | -1.646 | 22.600 | 0.8042 | 0.2361 | -0.0108 | 2,000,000 | 45,825 | 01:29:58 |
| `abl_app` | 23.125 | -1.072 | 22.601 | 0.7926 | 0.2458 | -0.0011 | 2,000,000 | 57,093 | 01:52:02 |
| `abl_antialiased` | 24.201 | +0.005 | 23.281 | 0.7944 | 0.2514 | +0.0046 | 2,000,000 | 48,332 | 00:47:00 |
| `abl_reg_low` | 23.867 | -0.329 | 22.961 | 0.7868 | 0.2595 | +0.0126 | 2,000,000 | 49,174 | 00:47:04 |
| `abl_depth_ppisp` | 23.076 | -1.121 | 22.266 | 0.7820 | 0.2580 | +0.0112 | 2,000,000 | 42,886 | 00:50:45 |
| `abl_cap3m` | 24.434 | +0.237 | 23.515 | 0.8059 | 0.2298 | -0.0171 | 3,000,000 | 43,824 | 01:01:50 |

## Step-count sweep

_Same recipe as `abl_baseline`, trained longer. Read as a curve: a flattening slope means the model has saturated and the remaining gap is structural._

| variant | steps | PSNR | Δ vs 30k | Δ vs previous | LPIPS | bg #GS | time | min/1k steps |
|---|---|---|---|---|---|---|---|---|
| `abl_baseline` | 30000 | 24.197 | — | — | 0.2469 | 2,000,000 | 00:46:46 | 1.56 |
| `abl_steps_45k` | 45000 | 24.387 | +0.191 | +0.191 | 0.2378 | 2,000,000 | 01:10:01 | 1.56 |
| `abl_steps_60k` | 60000 | 24.496 | +0.299 | +0.109 | 0.2321 | 2,000,000 | 01:32:06 | 1.53 |
| `abl_steps_75k` | 75000 | 24.646 | +0.450 | +0.150 | 0.2250 | 2,000,000 | 01:54:08 | 1.52 |
| `abl_steps_90k` | 90000 | 24.692 | +0.496 | +0.046 | 0.2217 | 2,000,000 | 02:17:15 | 1.52 |

## Per-camera PSNR

| variant | cam 0 | cam 1 | cam 2 | cam 3 | cam 4 | spread |
|---|---|---|---|---|---|---|
| baseline (45k ref) | 25.36 | 25.33 | 24.46 | 22.17 | 24.35 | 3.19 |
| `abl_baseline` | 25.59 | 25.23 | 24.35 | 21.85 | 24.03 | 3.74 |
| `abl_depth` | 23.74 | 24.35 | 23.01 | 20.92 | 22.79 | 3.43 |
| `abl_ppisp` | 25.18 | 25.32 | 24.39 | 21.78 | 23.88 | 3.53 |
| `abl_ppisp_ctrl` | 25.29 | 25.11 | 23.87 | 21.43 | 23.53 | 3.86 |
| `abl_bilateral` | 22.28 | 23.34 | 22.80 | 21.39 | 22.78 | 1.95 |
| `abl_app` | 24.51 | 24.76 | 21.72 | 21.50 | 23.06 | 3.27 |
| `abl_antialiased` | 25.60 | 25.19 | 24.39 | 21.77 | 24.08 | 3.84 |
| `abl_reg_low` | 25.02 | 25.23 | 24.06 | 21.35 | 23.72 | 3.88 |
| `abl_depth_ppisp` | 24.00 | 24.41 | 23.25 | 20.99 | 22.80 | 3.42 |
| `abl_cap3m` | 25.80 | 25.37 | 24.55 | 22.14 | 24.35 | 3.66 |
| `abl_steps_45k` | 25.41 | 25.24 | 24.45 | 22.42 | 24.42 | 2.99 |
| `abl_steps_60k` | 25.71 | 25.62 | 24.58 | 22.46 | 24.19 | 3.25 |
| `abl_steps_75k` | 25.78 | 25.66 | 24.81 | 22.60 | 24.44 | 3.18 |
| `abl_steps_90k` | 25.74 | 25.66 | 24.88 | 22.70 | 24.53 | 3.03 |

## Training fit vs. generalisation

The headline table alone is misleading for the appearance variants, so this is
the training-side counterpart, read from TensorBoard. `train l1` is the final
training L1; `val PSNR` is the held-out metric.

| variant | train l1 | val PSNR | peak mem |
|---|---|---|---|
| `abl_baseline` | 0.04765 | 24.197 | 2.73 GB |
| `abl_app` | **0.02999** | 23.125 | 5.09 GB |
| `abl_bilateral` | 0.04523 | 22.550 | 3.12 GB |
| `abl_ppisp` | 0.04671 | 24.094 | — |
| `abl_cap3m` | 0.04503 | 24.434 | 5.09 GB |
| `abl_reg_low` | 0.04765 | 23.867 | — |
| `abl_depth` | 0.05376 (depth term **2.302**) | 22.958 | — |
| `abl_steps_45k` | 0.02984 | **24.387** | 2.86 GB |
| `abl_steps_90k` | 0.02978 | **24.692** | 2.93 GB |

# Analysis

## 1. Training length is the biggest lever, and it has not saturated

30k → 90k gives **+0.496 dB and −0.0252 LPIPS**, monotonically. That LPIPS gain
is larger than any single lever produced, and the curve is still rising at 90k
(+0.046 on the last step). `min/1k steps` stays flat at 1.52–1.56, so the
curvature is genuine saturation behaviour, not a throughput artefact.

**Every other result in this study is therefore measured at an operating point
that understates it.** The levers were compared at 30k, where the model is
visibly undertrained.

## 2. Capacity is still binding — this corrects a premise of the study

`abl_cap3m` is the **best single lever at +0.237 dB**, with the best SSIM
(0.8059) and best LPIPS (0.2298) of any 30k variant. The background hits
`cap_max` exactly in every run, and raising it still pays.

Finding (1) in "Why this study exists" — that capacity is not the limit, inferred
from 250k → 2M returning only +1.6 dB — was **wrong**. That was the shape of a
saturating curve read too early. Peak VRAM at 3M was 5.09 GB, leaving room, so
**4M is worth testing** — see the push run.

## 3. Every per-image appearance module hurts, and the mechanism is proven

`app_opt` and `bilateral_grid` both fit the **training** set better than baseline
and generalise **worse**:

| | train l1 | val PSNR |
|---|---|---|
| `abl_steps_45k` | 0.02984 | 24.387 |
| `abl_app` (30k) | 0.02999 | **23.125** |

Identical training fit, **1.26 dB apart on validation**. `app_opt` buys its
training fit by memorising per-image appearance, which is unavailable at eval
(zero embedding) — so the geometry it distorted to accommodate that appearance is
all that remains. This is exactly the diagnostic the variant was included for,
and it answers question 4 unambiguously.

`bilateral_grid` shows the same pattern (train l1 0.04523 vs baseline 0.04765,
val −1.65 dB). Its apparent spread collapse to 1.95 is not a win: it drags the
good cameras *down* (cam 0: 25.59 → 22.28) rather than lifting camera 3.

Both are also expensive — `abl_app` took 1:52:02 and `abl_bilateral` 1:29:58
against a 46:46 baseline. Worse, and 2–2.4× the cost.

## 4. PPISP is the only appearance module that does not hurt

| | PSNR | cc-PSNR | SSIM | LPIPS | time |
|---|---|---|---|---|---|
| `abl_baseline` | 24.197 | 23.273 | 0.7962 | 0.2469 | 46:46 |
| `abl_ppisp` | 24.094 | **23.384** | **0.8021** | **0.2362** | 47:55 |

Raw PSNR is −0.103 (noise), but SSIM, LPIPS and colour-corrected PSNR all
improve, and it is the **only** variant whose cc-PSNR beats baseline's. It is
also nearly free (+1 min). That matches the prediction: PPISP is the only module
that still applies its per-camera terms on a novel view, where `app_opt` falls
back to a zero embedding and the bilateral grid is skipped entirely.

Modest, but real and free — and unlike the others it does not trade
generalisation for training fit (train l1 0.04671, slightly *better* than
baseline, with val essentially unchanged).

## 5. Depth supervision — the result was invalid, not informative

`abl_depth` came out **−1.238 dB**, the second-worst variant. Chasing it down
found two bugs in the depth ground truth (below), so this number measured broken
data rather than depth supervision. `abl_depth_ppisp` (−1.121) is void for the
same reason. See "Depth supervision investigation".

## 6. The controller's freeze side effect is confirmed

`abl_ppisp_ctrl` produced **121,812 rigid Gaussians** against ~45k everywhere
else — 2.7×. `freeze_gaussians()` freezes only `self.splats`, so with the
background frozen after step 24k the rigid densifier kept growing unchecked for
6k steps. It was also *faster* than baseline (42:29 vs 46:46) for the same
reason, and −0.370 dB worse.

This is a genuine implementation issue, not just a study artefact: any run using
PPISP distillation gets uncontrolled rigid growth.

## 7. The camera-3 deficit is a convergence problem, not a photometric one

This resolves the study's central open hypothesis. Camera 3 sits ~3 dB below the
best camera, and **only longer training moves it**:

| | cam 3 | spread |
|---|---|---|
| `abl_baseline` (30k) | 21.85 | 3.74 |
| `abl_ppisp` | 21.78 | 3.53 |
| `abl_cap3m` | 22.14 | 3.66 |
| `abl_steps_45k` | 22.42 | 2.99 |
| `abl_steps_90k` | **22.70** | **3.03** |

No appearance module lifts it. The per-camera vignetting/response hypothesis —
left open because an affine could not express it — is **not supported**: PPISP
has exactly those terms and moved camera 3 by −0.07.

## 8. Null results

- `abl_antialiased` +0.005 dB — noise. No effect on fisheye here.
- `abl_reg_low` −0.329 dB with **train l1 identical to baseline** (0.04765).
  The regularisers were not binding on the photometric fit at all, so weakening
  them only removed useful constraint.

# Off-trajectory quality (KID)

Everything above measures **on-trajectory** validation. The Difix loop exists to
improve views *away* from the recorded path, and the two are not the same
question — the loop slightly costs on-trajectory PSNR while substantially
improving off-trajectory KID. So each variant was rendered along the ego
trajectory plus −1/−2/−3 m lateral shifts (5 cameras × 200 frames, rigid objects
on) and scored against the real-frame bank.

KID ×1000, lower is closer to the real image distribution. **slope** is KID
growth per metre of lateral shift — flatter means the model generalises better
away from the path, which is the property that matters for simulation.

| variant | ego | −1 m | −2 m | −3 m | slope | Δ ego | Δ −3 m |
|---|---|---|---|---|---|---|---|
| `abl_baseline` | 16.14±0.78 | 17.46±0.88 | 22.19±1.07 | 28.83±1.25 | 4.28 | +0.00 | +0.00 |
| `abl_depth` | 17.13±0.90 | 18.46±1.04 | 25.90±1.41 | 34.39±1.60 | 5.92 | +1.00 | +5.55 |
| `abl_ppisp` | — | — | — | — | *render failed: PPISP no-controller state_dict* | |
| `abl_ppisp_ctrl` | 16.27±0.80 | 17.88±0.86 | 21.03±0.91 | 26.62±1.13 | 3.42 | +0.14 | -2.21 |
| `abl_bilateral` | — | — | — | — | *render failed: bilateral state read as PPISP* | |
| `abl_app` | — | — | — | — | *render failed: KeyError 'sh0'* | |
| `abl_antialiased` | 18.57±0.85 | 19.09±0.97 | 22.94±1.08 | 28.14±1.20 | 3.26 | +2.43 | -0.69 |
| `abl_reg_low` | 16.91±0.78 | 17.93±0.85 | 22.42±1.09 | 30.07±1.33 | 4.40 | +0.77 | +1.24 |
| `abl_depth_ppisp` | — | — | — | — | *render failed: PPISP no-controller state_dict* | |
| `abl_cap3m` | 14.05±0.68 | 15.50±0.80 | 19.37±0.99 | 24.92±1.17 | 3.65 | -2.09 | -3.91 |
| `abl_steps_45k` | 16.37±0.71 | 18.11±0.86 | 22.77±1.04 | 28.32±1.24 | 4.05 | +0.24 | -0.51 |
| `abl_steps_60k` | 15.48±0.74 | 17.23±0.86 | 21.17±1.03 | 26.77±1.22 | 3.78 | -0.66 | -2.06 |
| `abl_steps_75k` | 14.85±0.65 | 16.72±0.85 | 21.03±1.08 | 27.35±1.25 | 4.18 | -1.29 | -1.48 |
| `abl_steps_90k` | 14.45±0.67 | 16.40±0.84 | 20.59±1.09 | 25.41±1.22 | 3.71 | -1.69 | -3.42 |

Renders were produced from each checkpoint, featurised, and deleted per variant
to keep peak disk at ~3.6 GB rather than ~50 GB. Total 29.6 min.

## Four variants could not be rendered at all

This is a finding in its own right, not a harness problem. `render_standalone`
fails on three of the configurations this study introduced:

| variant(s) | error | cause |
|---|---|---|
| `abl_app` | `KeyError: 'sh0'` | with `app_opt` the trainer stores `features` + `colors` instead of `sh0`/`shN`; the renderer only knows the SH layout |
| `abl_ppisp`, `abl_depth_ppisp` | missing `controllers.*` keys | `PPISP.from_state_dict` rebuilds with the config default `use_controller=True`, so a checkpoint trained *without* the controller fails a strict `load_state_dict` |
| `abl_bilateral` | `KeyError: 'crf_params'` | the bilateral grid's module state is passed to `PPISP.from_state_dict` — the renderer assumes any post-processing state is PPISP |

**These configurations train but cannot be deployed.** The Difix loop renders
every round from a checkpoint, so none of them can currently feed it whatever
their metrics say. That directly affects the recommendation below: the
`ppisp` + `use_controller: false` setting recommended from the on-trajectory
results is exactly one of the broken cases.

## What KID says

**1. `abl_cap3m` wins off-trajectory too, and by more.** Best at every shift
level: ego 14.05 (−2.09 vs baseline) and −3 m 24.92 (−3.91). It was the best
on-trajectory lever at +0.237 dB; off-trajectory the margin is proportionally
larger. Capacity is the clearest win in the whole study.

**2. Longer training helps off-trajectory, but less cleanly.** Ego KID improves
monotonically (16.14 → 14.45 from 30k to 90k), but the −3 m column is noisy
(28.32 → 26.77 → 27.35 → 25.41) and the slope does not flatten monotonically
(4.05, 3.78, 4.18, 3.71). Longer training makes the model uniformly better, it
does not specifically make it generalise further off-path — consistent with the
earlier finding that the *pseudo-view bank*, not step count, is what flattens
the slope.

**3. Depth supervision is worse off-trajectory than on.** `abl_depth` is +1.00
ego but **+5.55 at −3 m**, with the steepest slope in the study (5.92 vs 4.28).
Whatever the depth term is doing to the geometry, it degrades most where the
geometry is least constrained. This reinforces that the current depth
configuration is actively harmful rather than merely unhelpful.

**4. `abl_antialiased` is not the null result it looked like.** On-trajectory it
was +0.005 dB — pure noise. Off-trajectory it has the **worst ego KID of any
variant (18.57, +2.43)** while having the flattest slope (3.26). It makes every
view distributionally worse but degrades more slowly. That is a real effect the
PSNR comparison could not see, and a caution against reading a single metric.

**5. `abl_ppisp_ctrl` looks better here than on-trajectory.** It was −0.370 dB on
PSNR, but off-trajectory it is −2.21 at −3 m with the second-flattest slope
(3.42). Since PPISP applies its per-camera terms on novel views, this is the
expected direction — though it is entangled with the background freeze and the
runaway rigid densification, so it is not a clean measurement.

**6. `abl_reg_low` is confirmed harmful** — worse at every shift, and worse at
−3 m (+1.24) than at ego (+0.77).

## Consequences for the recommendation

The on-trajectory recipe was *more steps + more capacity + PPISP*. KID revises it:

- **Capacity is now the clear first priority** — it wins both metrics, by the
  largest margin of anything tested, and 3M peaked at 5.09 GB.
- **Steps remain worthwhile** but their value is mostly uniform quality, not
  off-path generalisation. For the Difix loop specifically, the pseudo-view bank
  remains the thing that flattens the slope.
- **PPISP cannot be adopted yet.** The no-controller configuration is
  unrenderable, and the with-controller configuration carries the rigid
  densification bug. Both must be fixed before either can enter the loop.

# Depth supervision investigation

Two bugs in the depth ground truth, both fixed in `datasets/ncore.py`:

1. **Pinhole projection on fisheye cameras** — targets were projected with
   `K @ points_cam` while the renderer rasterises `opencv_fisheye`, so
   `grid_sample` read the depth of a different scene point (median 11 px off on
   the forward camera, 26 px on a side camera, p90 up to 95 px). Fixed by using
   the camera's own `camera_rays_to_image_points`.
2. **Wrong coordinate frame** — `point_visibility.npz` holds raw COLMAP-world
   coordinates and only ever received `self.transform`, which is built for the
   NCore scene frame. Same points, same order, offset by a constant 0.203 units.
   Fixed by taking coordinates from `self.points`.

Together these raised rendered-vs-GT depth correlation from 0.18–0.36 to
0.37–0.72 and collapsed the per-frame depth ratio from a wild 1.34–2.88 to a
consistent 1.11–1.22.

| run | depth term | train l1 | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| baseline (no depth) | — | 0.04765 | **24.197** | 0.7962 | 0.2469 |
| original (2 bugs) | 2.3018 | 0.05376 | 22.958 | 0.7756 | 0.2670 |
| **both fixed** | **0.3077** | **0.04740** | 23.673 | **0.7994** | **0.2403** |

Depth term down 87%, training fit now better than baseline, and **SSIM and
LPIPS both beat baseline** — only PSNR still trails, by 0.524 dB.

**There is probably still room here.** A residual ~15% bias remains between
rendered and GT depth. It is a property of what `RGB+ED` returns, not of the
ground truth, so any further fix belongs on the loss side. `depth_lambda` was
also never meaningfully tuned — it was set while the targets were broken, and
the term has since shrunk 87%.

A third change was tested and **rejected**: switching targets to ray distance
(`np.linalg.norm(points_cam)`) made everything worse (PSNR 23.67 → 23.59, LPIPS
0.240 → 0.263). The projection geometry could not distinguish a ray-distance
convention from a z convention with ~15% accumulated-depth overshoot; training
did, and z is correct.

Both fixes live inside `if self.load_depths`, so every run with
`depth_loss=false` is unaffected. Bug 2 affects any scene where NCore's
world→scene mapping is non-identity, not just fisheye ones.

# Push run: 4M Gaussians, 150k steps

Combining the only two levers the study found to work, and nothing else.
`multirun/ablation_scene084_push/abl_push` — 06:04:13, peak 5.32 GB allocated.

| step | PSNR | Δ prev | cc-PSNR | SSIM | LPIPS | num_GS |
|---|---|---|---|---|---|---|
| 30 000 | 21.657 | — | 20.601 | 0.6702 | 0.4352 | 2 960 567 |
| 60 000 | 22.888 | +1.231 | 21.922 | 0.7384 | 0.3305 | 4 000 000 |
| 90 000 | 23.960 | +1.072 | 23.039 | 0.7927 | 0.2515 | 4 000 000 |
| 120 000 | 24.708 | +0.748 | 23.799 | 0.8246 | 0.1996 | 4 000 000 |
| **150 000** | **25.075** | +0.367 | **24.165** | **0.8368** | **0.1789** | 4 000 000 |

Against the study's previous best:

| model | PSNR | SSIM | LPIPS |
|---|---|---|---|
| `abl_baseline` (2M, 30k) | 24.197 | 0.7962 | 0.2469 |
| `abl_cap3m` (3M, 30k) | 24.434 | 0.8059 | 0.2298 |
| `abl_steps_90k` (2M, 90k) | 24.692 | 0.7994 | 0.2217 |
| **`abl_push` (4M, 150k)** | **25.075** | **0.8368** | **0.1789** |

**+0.878 dB and −27.5% LPIPS over baseline**, and +0.383 dB over the best
single-lever result. The two levers compose: neither alone got past 24.7.

LPIPS is the headline. 0.2469 → 0.1789 is a far larger relative move than the
PSNR gain, which fits everything else the study found — this scene is limited by
geometric detail, not by photometry, and detail is what LPIPS measures.

**Not saturated.** The increments roughly halve every 30k (+1.231, +1.072,
+0.748, +0.367), so extrapolation suggests an asymptote near 25.4–25.5 rather
than a wall. More steps still buy quality; they just buy it slowly.

Note the 30k row is *not* comparable to `abl_baseline`'s 30k. `steps_scaler`
stretches the learning-rate, SH and densification schedules by 5×, so at step
30k this model is only a fifth of the way through its recipe (21.657 vs 24.197).
Only the endpoints compare.

## Off-trajectory (KID)

Rendered ego plus −1/−2/−3 m from the 150k checkpoint and scored against the
real-frame bank, same protocol as the ablation KID table.

| model | ego | −1 m | −2 m | −3 m | slope | KID(−3m)/KID(ego) |
|---|---|---|---|---|---|---|
| `abl_baseline` (2M, 30k) | 16.14 | 17.46 | 22.19 | 28.83 | 4.28 | 1.79 |
| `abl_steps_90k` | 14.45 | 16.40 | 20.59 | 25.41 | 3.71 | 1.76 |
| `abl_cap3m` | 14.05 | 15.50 | 19.37 | 24.92 | 3.65 | 1.77 |
| **`abl_push` (4M, 150k)** | **9.59** | **11.35** | **15.90** | **22.53** | 4.34 | **2.35** |

Enormous in absolute terms — **ego −41%, −3 m −22%** against baseline, well past
any other variant. But the **slope is unchanged** (4.28 → 4.34) and the relative
degradation gets *worse*: the −41% gain at ego has shrunk to −22% by −3 m.

**A better-fit model overfits the trajectory harder.** Capacity and steps buy
quality uniformly; they buy nothing for generalisation away from the path.

### This settles how the two halves of the project relate

Put beside the earlier controlled measurement on this scene — run B, where every
round trained for the same 45k steps so only the pseudo-view bank varied:

| lever | ego KID | −3 m KID | slope |
|---|---|---|---|
| capacity + steps (`abl_push`) | **−41%** | −22% | 4.28 → **4.34** (no change) |
| pseudo-view bank (run B, r0→r3) | +0.7% | **−30%** | 4.26 → **1.27** (−70%) |

They are near-orthogonal. Capacity and steps lower the whole curve without
tilting it; the Difix bank tilts the curve without lowering it. Neither
substitutes for the other, and the obvious next experiment is both at once — a
4M/150k model trained with an accumulated pseudo-view bank.

Rough arithmetic on what that would give: applying the bank's measured slope
flattening (×0.30) to `abl_push`'s intercept puts −3 m near **13–14 KID**,
against 22.53 now and 28.83 for the original baseline.

## Two memory bugs this exposed

The first attempt OOMed at step 32 500 — exactly where the background cap is
reached. Two independent causes:

**1. `train/mem` measures the wrong thing.** It reports torch *allocated*
memory. At failure 4.53 GiB was allocated while **2.26 GiB sat
reserved-but-unallocated** to fragmentation, against a real capacity of 7.62 GiB
(not the 8.19 GiB the device advertises). A probe validated on `train/mem`
looked safe and was not. Fixed with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; peak allocated afterwards was
5.32 GB and the run completed.

**2. `steps_scaler` causes runaway rigid densification.** `rigid_cap_max`
defaults to **1 000 000**, and `adjust_steps` stretches `rigid_refine_stop_iter`
along with everything else. Rigid Gaussians sat at 83 907 and were still growing
~5% per refine with 92k steps of growth remaining — on track for the full 1M,
against ~50k in a 30k run. That is ~100k Gaussians *per vehicle* for 10
instances: far past useful, and it starves the background.

Capped at 150 000 for this run; it finished at 89 999, so the cap bounded the
growth without binding hard.

**This is a general defect, not a quirk of this experiment.** Any long run with
`steps_scaler` inherits it, and `rigid_cap_max: 1_000_000` is a poor default at
any step count for a handful of vehicles. Worth changing in `train.yaml`.

# Decisions taken

**The recipe: more capacity + more steps.** Validated together in the push run
— 4M Gaussians and 150k steps reached **25.075 dB / 0.1789 LPIPS**, against a
24.197 / 0.2469 baseline. The two levers compose; neither alone passed 24.7.

1. **Raise `cap_max`.** The best single lever on both on- and off-trajectory
   metrics, and still paying at 4M.
2. **Raise the step count.** Not saturated at 150k — increments roughly halve
   every 30k, extrapolating to ~25.4-25.5.
3. **PPISP is promising but blocked.** It improves SSIM/LPIPS/cc-PSNR at no
   PSNR cost and is the only appearance module that survives evaluation, and
   with the controller it also improves off-trajectory KID. But
   `use_controller: false` produces checkpoints `render_standalone` cannot load,
   and `use_controller: true` carries the rigid-densification freeze bug. **Fix
   both before adopting**; neither can feed the Difix loop today.
4. **Do not use `app_opt` or `bilateral_grid`.** Both demonstrably trade
   generalisation for training fit, and both cost ~2× the runtime.
5. **`depth_loss` is now usable and probably still has room.** Two ground-truth
   bugs made the −1.238 dB result a measurement of broken data. Corrected, it
   beats baseline on SSIM and LPIPS while costing 0.52 dB PSNR, with a residual
   ~15% bias and an untuned `depth_lambda` still outstanding.
6. **Scaling does not replace the Difix loop.** Capacity and steps lower the
   whole KID curve without tilting it (slope 4.28 → 4.34); the pseudo-view bank
   tilts it without lowering it (4.26 → 1.27). Near-orthogonal — the next
   experiment is both at once.

## Follow-ups

- **Fix the three `render_standalone` load paths** (`app_opt` splat layout,
  PPISP controller-optional state dict, bilateral state misread as PPISP).
  Until then those three configurations are training-only and cannot be used
  anywhere downstream.
- **Fix `freeze_gaussians`** to freeze rigid nodes too (or document why not).
- **Re-run the lever comparison at 90k**, since everything here was measured
  where the model is undertrained; a lever's ranking at 30k need not hold at
  convergence.
- **Re-run KID for the four unrenderable variants** once the loader is fixed.
  `abl_app` and `abl_bilateral` are the interesting ones: both fit training
  better than baseline and lost on validation only because their per-image
  correction is unavailable at eval, so their underlying geometry may be fine.
  KID would settle it.
- **Combine the push recipe with the Difix loop.** The single highest-value
  follow-up. Budget matters: ~6 h per round at 4M/150k, so a schedule like
  `[30k, 30k, 30k, 150k]` is more sensible than four full rounds — returns from
  the bank are heavily front-loaded, and 7k-model pseudo-GT measured only ~4%
  worse than 45k-model pseudo-GT.
- **Change `rigid_cap_max`.** The 1,000,000 default lets rigid densification run
  away on any long run; 150k was ample here (finished at 89,999).
- **`data_factor=1`** remains untested and now more interesting: if quality is
  convergence- and capacity-limited rather than photometric, resolution is the
  next structural lever. Needs its own paired design (see "Deliberately
  excluded").
