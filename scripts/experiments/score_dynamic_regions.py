"""Dynamic-region (vehicle) metrics for a trained dynamic run — TODO item 3.

Global metrics are nearly blind to vehicles, which occupy a few percent of the
frame. Measured: a configuration change that removed 5-6 whole vehicles moved
global PSNR by ~0.17 dB, and across the 7-cell reset sweep global PSNR spanned
0.066 dB while rigid Gaussians spanned 43x. So "modelling vehicles beats masking
them" (thesis section 6.3) is unmeasurable without this.

What it reports, and what it deliberately does not:

  * **masked PSNR** inside the projected vehicle boxes, pooled and per instance.
    PSNR is the only metric exactly definable under a mask. SSIM is patch-based
    (an 11x11 window straddles the boundary) and LPIPS is a whole-image deep
    feature distance with no principled per-pixel restriction, so reporting three
    quantities as if identically defined would be false precision.
  * **surviving instance count and opacity-weighted capacity**, which is what
    actually tracked vehicle quality in practice: 662,811 Gaussians at median
    opacity 0.006 carry ~10k of capacity against 499,145 at 0.944 carrying ~319k.

The mask comes from the BOXES, never from the model's own rigid alpha. A
model-derived mask shrinks wherever the model failed to render a vehicle, so a
model that lost a car would score better on the cars it kept.

The mask is a full cuboid, so roughly 30-40% of it is not vehicle. That bleed is
identical across every arm compared, so relative comparisons hold; absolute values
read as "box-region" rather than "vehicle-pixel" PSNR and the thesis should say so.

GT comes from the stage 035 real-frame bank, which is already at render resolution
and byte-identical to the training input, so no parser is constructed and no
resize enters the comparison.

Usage:
    PYTHONPATH=src/gsplat_training envs/envs/env_gsplat/bin/python \\
        scripts/experiments/score_dynamic_regions.py \\
        --run multirun/rigid_reset_study/abl_rigid_r_runs/abl_rigid_r0_p020_f100 \\
        --real-bank results/extended_4dgs/wayve101/scene_084/035_real_frames_3e3e1542
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath("src/gsplat_training"))

from render_standalone import (  # noqa: E402
    CameraMetadata, StandaloneRenderer, load_all_cameras,
    load_rigid_state_from_checkpoint, load_splats_from_checkpoint,
    rigid_box_solid_gaussians,
)


def psnr_masked(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """PSNR over masked pixels only, on [0, 255] uint8 inputs.

    Returns None when the mask is empty; a caller must not average that as zero.
    """
    if not mask.any():
        return None
    d = (gt[mask].astype(np.float64) - pred[mask].astype(np.float64)) ** 2
    mse = float(d.mean())
    return float("inf") if mse == 0 else 10.0 * float(np.log10(255.0**2 / mse))


def load_val_frames(real_bank_dir: str) -> Dict[str, List[dict]]:
    """Held-out frames per camera, from the bank's index.json.

    ``is_val`` in the index is the authoritative record of the split: the
    validation phase rotates by 8 frames at each camera boundary because
    200 % 16 == 8, so it is NOT "every 16th frame per camera".
    """
    with open(os.path.join(real_bank_dir, "index.json")) as fp:
        index = json.load(fp)
    out: Dict[str, List[dict]] = {}
    for cam, rec in index["cameras"].items():
        out[cam] = [f for f in rec["frames"] if f.get("is_val")]
    return out


def score_run(
    run_dir: str,
    real_bank_dir: str,
    device: str = "cuda",
    per_instance: bool = True,
    mask_thresh: int = 250,
    limit: Optional[int] = None,
) -> Dict:
    ckpts = sorted(glob.glob(os.path.join(run_dir, "ckpts", "*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint under {run_dir}/ckpts")
    ckpt_path = ckpts[-1]

    # Returns (splats, post_processing_state). PPISP is deliberately NOT passed to
    # the renderer: it is a per-camera/per-frame photometric correction, and
    # applying it here while the GT bank is uncorrected would compare a corrected
    # render to an uncorrected photograph. The runs this scores have
    # post_processing: null anyway.
    splats, _pp_state = load_splats_from_checkpoint(ckpt_path, device=device)
    rigid_state = load_rigid_state_from_checkpoint(ckpt_path, device=device)
    if rigid_state is None:
        raise ValueError(
            f"{ckpt_path} has no rigid_nodes; this scores dynamic runs only. A "
            "background-only run is the MASKING arm of section 6.3 and must be "
            "scored with the same masks from the dynamic run's tracks."
        )

    cams = {c.camera_id: c for c in load_all_cameras(os.path.join(run_dir, "camera_paths"))}
    val = load_val_frames(real_bank_dir)
    n_inst = int(rigid_state["instances_fv"].shape[1])

    # Two renderers over the same Gaussians: the scene, and boxes-only on white.
    # rigid_white empties the background and composites caller-supplied rigid
    # Gaussians verbatim, which is exactly the mask.
    scene = StandaloneRenderer(splats, rigid_state=rigid_state,
                               render_mode="full", device=device)
    boxes = StandaloneRenderer(splats, rigid_state=rigid_state,
                               render_mode="rigid_white", device=device)

    def render(r: StandaloneRenderer, cam: CameraMetadata, fi: int, **kw) -> np.ndarray:
        return r.render_frame(
            camtoworld=cam.camtoworlds[fi], K=cam.K, width=cam.width,
            height=cam.height, camera_model=cam.camera_model,
            radial_coeffs=cam.radial_coeffs, tangential_coeffs=cam.tangential_coeffs,
            thin_prism_coeffs=cam.thin_prism_coeffs, ftheta_coeffs=cam.ftheta_coeffs,
            camera_idx=cam.camera_index, **kw,
        )

    # Accumulate squared error over PIXELS, not PSNR over frames: averaging
    # per-frame PSNR weights a frame with 200 masked pixels the same as one with
    # 20,000, and dynamic-region masks vary enormously in size between frames.
    pooled_num = 0.0
    pooled_px = 0
    total_px = 0
    per_inst_mse = {m: [0.0, 0] for m in range(n_inst)}
    global_num, global_px = 0.0, 0
    frames_scored = frames_no_mask = 0

    t0 = time.perf_counter()
    for cam_id in sorted(val):
        cam = cams.get(cam_id)
        if cam is None:
            continue
        for rec in (val[cam_id][:limit] if limit else val[cam_id]):
            fi = int(rec["frame_idx"])
            gt = np.asarray(Image.open(
                os.path.join(real_bank_dir, rec["image"])).convert("RGB"))
            # Rigid poses are per timestep and shared across cameras, so the
            # per-camera frame index IS the rigid frame index.
            pred = render(scene, cam, fi, rigid_frame_idx=fi)
            pred = np.clip(pred, 0, 1) if pred.dtype != np.uint8 else pred
            if pred.dtype != np.uint8:
                pred = (pred * 255).astype(np.uint8)
            if pred.shape[:2] != gt.shape[:2]:
                raise ValueError(
                    f"render {pred.shape[:2]} != bank {gt.shape[:2]} for "
                    f"{cam_id} frame {fi}; the bank must match the render "
                    "resolution or PSNR is comparing resampled pixels"
                )

            solid = rigid_box_solid_gaussians(rigid_state, fi)
            if solid is None:
                frames_no_mask += 1
                continue
            mimg = render(boxes, cam, fi, rigid_gaussians=solid)
            if mimg.dtype != np.uint8:
                mimg = (np.clip(mimg, 0, 1) * 255).astype(np.uint8)
            mask = mimg.min(axis=-1) < mask_thresh      # anything not white

            # Global (whole-frame) MSE on the same frames, so the pair is
            # comparable rather than being read against stats/val_step*.json,
            # which averages PSNR per frame instead of pooling pixels.
            if not mask.any():
                # Boxes exist at this timestep but project nowhere in THIS camera --
                # a vehicle ahead of the ego is off-screen for the rear cameras.
                # Such a frame must not enter either total, or the masked-pixel
                # fraction is diluted by frames that could never contribute. That
                # bug understated the fraction by 1.58x (1.54% reported against a
                # true 2.44%) before being caught by disagreeing with eval()'s
                # independent implementation. PSNR was unaffected -- it pools only
                # masked pixels -- but the fraction is quoted, so it matters.
                frames_no_mask += 1
                continue

            global_num += float(((gt.astype(np.float64) - pred.astype(np.float64)) ** 2).sum())
            global_px += int(gt.size)
            total_px += int(gt.size)

            d = (gt[mask].astype(np.float64) - pred[mask].astype(np.float64)) ** 2
            pooled_num += float(d.sum()); pooled_px += int(d.size)
            frames_scored += 1

            if per_instance:
                for m in range(n_inst):
                    s1 = rigid_box_solid_gaussians(rigid_state, fi, instance_ids=[m])
                    if s1 is None:
                        continue
                    im = render(boxes, cam, fi, rigid_gaussians=s1)
                    if im.dtype != np.uint8:
                        im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
                    mk = im.min(axis=-1) < mask_thresh
                    if not mk.any():
                        continue
                    d = (gt[mk].astype(np.float64) - pred[mk].astype(np.float64)) ** 2
                    per_inst_mse[m][0] += float(d.sum()); per_inst_mse[m][1] += int(d.size)

    def to_psnr(num: float, px: int) -> Optional[float]:
        if px == 0:
            return None
        mse = num / px
        return float("inf") if mse == 0 else 10.0 * float(np.log10(255.0**2 / mse))

    # Opacity-weighted capacity per instance: the count alone inverts the
    # ordering (see TODO item 19).
    pid = rigid_state["point_ids"]
    rop = torch.sigmoid(rigid_state["gauss.opacities"].flatten())
    inst = []
    for m in range(n_inst):
        sel = pid == m
        p = to_psnr(*per_inst_mse[m]) if per_inst_mse[m][1] else None
        inst.append({
            "instance": m,
            "n_gaussians": int(sel.sum()),
            "effective_capacity": float(rop[sel].sum()) if sel.any() else 0.0,
            "mean_opacity": float(rop[sel].mean()) if sel.any() else 0.0,
            "masked_psnr": p,
            "mask_px": per_inst_mse[m][1],
        })

    return {
        "schema_version": 1,
        "config": {
            "run_dir": run_dir, "checkpoint": os.path.basename(ckpt_path),
            "real_bank_dir": real_bank_dir, "mask": "solid projected 3D boxes",
            "mask_threshold": mask_thresh, "per_instance": per_instance,
            "note": "full-cuboid mask; ~30-40% of masked pixels are not vehicle",
        },
        "frames_scored": frames_scored,
        "frames_without_mask": frames_no_mask,
        "dynamic_psnr": to_psnr(pooled_num, pooled_px),
        # Same frames, same pixel-pooled estimator, so the two are directly
        # comparable. NOT read from stats/val_step*.json, which averages PSNR per
        # frame over a different frame set.
        "global_psnr_same_frames": to_psnr(global_num, global_px),
        "masked_pixel_fraction": (pooled_px / total_px) if total_px else None,
        "instances": inst,
        "n_instances_total": n_inst,
        "n_instances_with_gaussians": int(len(torch.unique(pid))),
        "effective_capacity_total": float(rop.sum()),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a training output dir (with ckpts/, camera_paths/)")
    ap.add_argument("--real-bank", required=True, help="035_real_frames_<h> directory")
    ap.add_argument("--out", default=None, help="default: <run>/metrics_dynamic/dynamic_regions.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-per-instance", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="val frames per camera (smoke test)")
    a = ap.parse_args()

    res = score_run(a.run, a.real_bank, device=a.device,
                    per_instance=not a.no_per_instance, limit=a.limit)

    out = a.out or os.path.join(a.run, "metrics_dynamic", "dynamic_regions.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fp:
        json.dump(res, fp, indent=2)

    print(f"\n{os.path.basename(a.run)}")
    print(f"  dynamic-region PSNR  {res['dynamic_psnr']:.3f}" if res["dynamic_psnr"]
          else "  dynamic-region PSNR  n/a")
    if res["global_psnr_same_frames"]:
        print(f"  global PSNR (same frames) {res['global_psnr_same_frames']:.3f}")
    print(f"  masked pixel fraction {100 * (res['masked_pixel_fraction'] or 0):.2f}%")
    print(f"  instances with Gaussians {res['n_instances_with_gaussians']}/{res['n_instances_total']}"
          f"   effective capacity {res['effective_capacity_total']:,.0f}")
    print(f"  frames scored {res['frames_scored']} (no mask: {res['frames_without_mask']})"
          f"   {res['seconds']}s")
    ranked = sorted((i for i in res["instances"] if i["masked_psnr"]),
                    key=lambda i: i["masked_psnr"])
    if ranked:
        print("  worst instances by masked PSNR:")
        for i in ranked[:4]:
            print(f"    inst {i['instance']:2d}  PSNR {i['masked_psnr']:6.2f}  "
                  f"{i['n_gaussians']:>8,} GS  eff {i['effective_capacity']:>9,.0f}  "
                  f"op {i['mean_opacity']:.3f}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
