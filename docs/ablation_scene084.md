# scene_084 ablation study — what is capping reconstruction quality?

## Why this study exists

Two full dynamic Difix-loop runs on scene_084 reached **24.24 dB PSNR** on
held-out frames, against the 26+ commonly reported by AV 3DGS papers. Before
chasing that gap with guesses, four measurements narrowed the search:

**1. Capacity is not the limit.** Raising the MCMC cap from 250k to 2M Gaussians
bought only +1.6 dB (22.31 → 23.92), the saturating end of the curve. 2M
Gaussians over an 86 m path is already dense.

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

**14 experiments total: 10 single-lever variants + a 4-point step-count sweep.**
At measured speeds (below) that is ~15 h, which does not fit one night — so it
is split into two sessions. Night 1 carries the step sweep plus the variants
with the strongest priors, because "the model is simply undertrained" is the
cheapest explanation for 24 dB and nothing has ruled it out.

### Measured timings

From this scene's completed runs (`.success` markers): **30k steps = 00:49:02**,
**45k = 01:07:49**. Fitting those gives ~11 min fixed overhead plus 1.27 min per
1k steps.

| steps | est. time | | variants at 30k | est. time |
|---|---|---|---|---|
| 45k | 1:08 | | each ~0:49 | |
| 60k | 1:27 | | `abl_app` (+27%/img) | ~0:55 |
| 75k | 1:46 | | `abl_cap3m` (3M) | ~1:10 |
| 90k | 2:05 | | | |

### Night 1 — step sweep + high-prior levers (~9:15)

```bash
PYTHONPATH=. envs/envs/env_gsplat/bin/python src/gsplat_training/train_splats.py -m \
  +experiment=abl_baseline,abl_depth,abl_ppisp,abl_reg_low,abl_steps_45k,abl_steps_60k,abl_steps_75k,abl_steps_90k \
  hydra.sweep.dir=multirun/ablation_scene084 \
  'hydra.sweep.subdir=${exp_name}'
```

`abl_baseline` runs first because it is both the reference for every lever and
the 30k point of the step curve.

### Night 2 — remaining levers and controls (~5:40)

```bash
PYTHONPATH=. envs/envs/env_gsplat/bin/python src/gsplat_training/train_splats.py -m \
  +experiment=abl_ppisp_ctrl,abl_bilateral,abl_app,abl_antialiased,abl_depth_ppisp,abl_cap3m \
  hydra.sweep.dir=multirun/ablation_scene084 \
  'hydra.sweep.subdir=${exp_name}'
```

Same sweep directory, so `collect_ablation.py` picks up both sessions together.
`abl_cap3m` is last because 3M Gaussians may OOM on 8 GB.

### Everything in one go (~15 h)

If you would rather not split it, concatenate both `+experiment=` lists. Order
is deliberate either way — the variant most likely to fail is last.

`PYTHONPATH=.` is required: `train_splats.py` imports `src.gsplat_training.*`,
and the orchestrators normally set it for their subprocesses. Without it the
sweep dies immediately on `ModuleNotFoundError: No module named 'src'`.

Then collect the tables below with:

```bash
envs/envs/env_gsplat/bin/python scripts/experiments/collect_ablation.py \
  multirun/ablation_scene084
```

Results land in `multirun/ablation_scene084/<exp_name>/`, each with
`stats/val_step29999.json`, `renders/`, `ckpts/` and the resolved
`.hydra/config.yaml`.

**Disk: ~20 GB** for all 14 runs. Checkpoints are kept (1.4 GB each — needed
for rigid Gaussian counts and to continue from a winner); PLYs are disabled
(451 MB each, and nothing in this study renders from one). ~150 GB free at the
time of writing, so this is comfortable.

### Things already handled

- `refine_task.user_refinement.enabled: false` — refinement never runs (the
  tracks JSON is passed directly), but its interactive loop would block an
  unattended run if it ever did.
- **A failed variant does not stop the sweep.** Verified: Hydra's basic launcher
  logs the failure and continues to the next job.
- `use_color_correction_metric: true` — every run reports `cc_psnr`/`cc_ssim`/
  `cc_lpips` beside the raw metrics, so comparisons against published numbers
  are explicit about which convention they use.
- No tracking or refinement runs; the sweep reuses scene_084's existing
  `03_ncore_ego_1dcbcd85` and `15_refine_27ea80ce`.

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
| 10 | `abl_cap3m` | `cap_max` 2M → 3M | is capacity still binding? The cap is hit exactly every run, but (1) says returns are saturating. **Last, because 3M may OOM on 8 GB.** |

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

Fill in from `multirun/ablation_scene084/<exp_name>/stats/val_step29999.json`.

## Headline metrics

| variant | PSNR | Δ | cc-PSNR | SSIM | LPIPS | Δ LPIPS | bg #GS | rigid #GS | time |
|---|---|---|---|---|---|---|---|---|---|
| `abl_baseline` | | — | | | | — | | | |
| `abl_depth` | | | | | | | | | |
| `abl_ppisp` | | | | | | | | | |
| `abl_ppisp_ctrl` | | | | | | | | | |
| `abl_bilateral` | | | | | | | | | |
| `abl_app` | | | | | | | | | |
| `abl_antialiased` | | | | | | | | | |
| `abl_reg_low` | | | | | | | | | |
| `abl_depth_ppisp` | | | | | | | | | |
| `abl_cap3m` | | | | | | | | | |

## Step-count sweep

| variant | steps | PSNR | Δ vs 30k | Δ vs previous | LPIPS | bg #GS | time | min/1k steps |
|---|---|---|---|---|---|---|---|---|
| `abl_baseline` | 30000 | | — | — | | | | |
| `abl_steps_45k` | 45000 | | | | | | | |
| `abl_steps_60k` | 60000 | | | | | | | |
| `abl_steps_75k` | 75000 | | | | | | | |
| `abl_steps_90k` | 90000 | | | | | | | |

## Per-camera PSNR

The 3.2 dB spread is the specific defect this study targets. A variant that
lifts the overall mean by raising camera 3 is a different (and better) result
than one that lifts all cameras equally.

| variant | cam 0 | cam 1 | cam 2 | cam 3 | cam 4 | spread |
|---|---|---|---|---|---|---|
| baseline (45k ref) | 25.36 | 25.33 | 24.46 | 22.17 | 24.35 | 3.19 |
| `abl_baseline` | | | | | | |
| `abl_depth` | | | | | | |
| `abl_ppisp` | | | | | | |
| `abl_ppisp_ctrl` | | | | | | |
| `abl_bilateral` | | | | | | |
| `abl_app` | | | | | | |
| `abl_antialiased` | | | | | | |
| `abl_reg_low` | | | | | | |
| `abl_depth_ppisp` | | | | | | |
| `abl_cap3m` | | | | | | |

## Questions to answer from the table

1. **Is it geometry?** Do `abl_depth` and `abl_reg_low` move PSNR more than the
   photometric variants? If yes, the diagnosis holds and effort belongs in
   supervision and regularisation, not appearance.
2. **Does the per-camera spread close?** Does `abl_ppisp` specifically lift
   camera 3, or does it raise everything uniformly? Only the former confirms the
   vignetting/response hypothesis.
3. **Do the controls stay flat?** `abl_bilateral` and `abl_app` should be near
   baseline if measurement (4) generalises. If either wins substantially, the
   "not photometric" conclusion needs revisiting.
4. **Does `abl_app` diverge train vs val?** A better training loss with equal or
   worse val PSNR means the MLP is absorbing geometry error — informative about
   overfitting regardless of the headline number.
5. **Is the controller worth its geometry cost?** `abl_ppisp_ctrl` vs
   `abl_ppisp`: does predicted per-frame correction repay 6k fewer geometry
   steps?
6. **Is capacity still binding?** Does `abl_cap3m` gain enough to justify the
   VRAM, or has it saturated as (1) suggests?
7. **Is it undertrained?** Where does the step curve flatten? If 30k → 90k
   buys under ~0.3 dB the model has saturated and the levers above are the whole
   story. If it buys more than a dB, training length is the cheapest win and
   every 30k comparison here understates what the levers would do at
   convergence. Also worth checking against cost: `min/1k steps` should be
   roughly flat, so any curvature in PSNR is real saturation rather than a
   throughput artefact.
8. **Do the winners compose?** `abl_depth_ppisp` versus the sum of the two
   individual deltas.

## Analysis

_(to fill in)_

## Decisions taken

_(to fill in — which settings become the new default, what moves into the Difix
loop, what gets a follow-up study)_
