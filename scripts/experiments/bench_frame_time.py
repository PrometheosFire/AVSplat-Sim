"""Frame-time benchmark for the simulator's rendering path — TODO item 16, thesis section 6.5.

Section 6.5 claims real-time closed-loop simulation against a 30 Hz target and
**has never been measured**. The only datapoint on record is an extrapolation
(38 ms rasterise at 960x540 for 1.5 M Gaussians), which puts a 2.05 M-Gaussian
dynamic scene below 26 fps.

Measured **scripted and offline**, deliberately not inside the interactive loop:
that loop adds viser transport and a ``max_fps: 10`` cap, so timing it would floor
every result at 10 fps and measure the cap rather than the renderer. What is timed
here is one ``render_frame`` call -- rasterise, composite the rigid Gaussians for
the frame, and copy back to host -- which is what a simulator tick pays before
display. Viser transport sits on top and is excluded.

The levers are Gaussian count and resolution, so those are what it sweeps. Existing
checkpoints already span them without any new training: the rigid reset sweep
covers 15,558 -> 662,811 rigid Gaussians at a fixed 2 M background, and
``abl_cap3m`` vs ``abl_baseline`` covers 2 M -> 3 M background.

Usage:
    PYTHONPATH=. envs/envs/env_gsplat/bin/python scripts/experiments/bench_frame_time.py \\
        --runs multirun/rigid_reset_study/abl_rigid_r_runs/* \\
        --resolutions 960x540 1920x1080
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.abspath("src/gsplat_training"))

from render_standalone import (  # noqa: E402
    CameraMetadata, StandaloneRenderer, load_all_cameras,
    load_rigid_state_from_checkpoint, load_splats_from_checkpoint,
)


def parse_res(spec: str) -> Tuple[int, int]:
    w, _, h = spec.lower().partition("x")
    return int(w), int(h)


def scaled_camera(cam: CameraMetadata, width: int, height: int) -> Tuple[np.ndarray, int, int]:
    """Intrinsics rescaled to a target resolution.

    K must scale with the image or the field of view changes and the Gaussian
    count in frame changes with it, which would confound a resolution sweep with a
    visibility sweep.
    """
    sx, sy = width / cam.width, height / cam.height
    K = cam.K.copy()
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    return K, width, height


def bench_run(
    run_dir: str,
    resolutions: List[Tuple[int, int]],
    frames: int = 60,
    warmup: int = 10,
    camera_id: Optional[str] = None,
    device: str = "cuda",
    with_rigid: bool = True,
) -> List[Dict]:
    ckpts = sorted(glob.glob(os.path.join(run_dir, "ckpts", "*.pt")))
    if not ckpts:
        return []
    splats, _pp = load_splats_from_checkpoint(ckpts[-1], device=device)
    rigid_state = load_rigid_state_from_checkpoint(ckpts[-1], device=device)
    cams = load_all_cameras(os.path.join(run_dir, "camera_paths"))
    cam = next((c for c in cams if c.camera_id == camera_id), cams[0])

    n_bg = int(splats["means"].shape[0])
    n_rigid = int(rigid_state["gauss.means"].shape[0]) if rigid_state is not None else 0
    n_frames_avail = cam.camtoworlds.shape[0]

    out = []
    for (w, h) in resolutions:
        K, width, height = scaled_camera(cam, w, h)
        r = StandaloneRenderer(
            splats, rigid_state=rigid_state if with_rigid else None,
            render_mode="full", device=device,
        )

        def one(i: int):
            return r.render_frame(
                camtoworld=cam.camtoworlds[i % n_frames_avail], K=K,
                width=width, height=height, camera_model=cam.camera_model,
                radial_coeffs=cam.radial_coeffs,
                tangential_coeffs=cam.tangential_coeffs,
                thin_prism_coeffs=cam.thin_prism_coeffs,
                ftheta_coeffs=cam.ftheta_coeffs, camera_idx=cam.camera_index,
                rigid_frame_idx=(i % n_frames_avail) if (with_rigid and rigid_state is not None)
                else None,
            )

        # A high resolution with a large Gaussian set can OOM on an 8 GB card, and
        # that must cost one measurement rather than the whole sweep.
        try:
            # Warm up: first calls pay CUDA context setup, kernel autotuning and
            # lazy allocation, and would otherwise dominate a short measurement.
            for i in range(warmup):
                one(i)
            torch.cuda.synchronize()

            times = []
            for i in range(frames):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                one(i)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM: {os.path.basename(run_dir)} at {w}x{h} "
                  f"({n_bg + n_rigid:,} Gaussians) -- recorded as a failure")
            out.append({
                "run": os.path.basename(run_dir.rstrip("/")), "width": w, "height": h,
                "n_background": n_bg, "n_rigid": n_rigid,
                "n_total": n_bg + (n_rigid if with_rigid else 0),
                "with_rigid": with_rigid and rigid_state is not None,
                "oom": True, "ms_median": None, "fps_median": None,
            })
            del r
            torch.cuda.empty_cache()
            continue

        times.sort()
        med = statistics.median(times)
        out.append({
            "run": os.path.basename(run_dir.rstrip("/")),
            "width": width, "height": height,
            "n_background": n_bg, "n_rigid": n_rigid,
            "n_total": n_bg + (n_rigid if with_rigid else 0),
            "with_rigid": with_rigid and rigid_state is not None,
            "ms_median": round(med, 2),
            "ms_p95": round(times[int(0.95 * (len(times) - 1))], 2),
            "ms_min": round(times[0], 2),
            "fps_median": round(1000.0 / med, 1),
            "frames": frames,
        })
        del r
        torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--resolutions", nargs="+", default=["960x540"])
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-rigid", action="store_true",
                    help="also time background-only, to isolate the rigid cost")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    res = [parse_res(s) for s in a.resolutions]
    rows: List[Dict] = []
    for run in a.runs:
        if not os.path.isdir(os.path.join(run, "ckpts")):
            continue
        rows += bench_run(run, res, a.frames, a.warmup, a.camera, a.device, with_rigid=True)
        if a.no_rigid:
            rows += bench_run(run, res, a.frames, a.warmup, a.camera, a.device, with_rigid=False)

    if not rows:
        print("no runs with checkpoints found", file=sys.stderr)
        return 1

    print(f"\n{'run':30s}{'res':>11}{'rigid':>9}{'total GS':>11}"
          f"{'ms med':>9}{'ms p95':>9}{'fps':>8}{'30Hz':>7}")
    print("-" * 94)
    for r in sorted(rows, key=lambda r: (r["width"], r["n_total"])):
        name = r["run"][:28] + ("" if r["with_rigid"] else " (bg only)")
        res = f"{r['width']}x{r['height']}"
        rigid = r["n_rigid"] if r["with_rigid"] else 0
        if r.get("oom"):
            print(f"{name:30s}{res:>11}{rigid:>9,}{r['n_total']:>11,}"
                  f"{'OOM':>9}{'':>9}{'':>8}{'--':>7}")
            continue
        verdict = "OK" if r["fps_median"] >= 30 else "MISS"
        print(f"{name:30s}{res:>11}{rigid:>9,}{r['n_total']:>11,}"
              f"{r['ms_median']:>9.2f}{r['ms_p95']:>9.2f}{r['fps_median']:>8.1f}"
              f"{verdict:>7}")

    print("\nExcludes viser transport and the interactive loop's max_fps cap; this is the "
          "offline cost of one tick's render.")

    out = a.out or "multirun/frame_time_benchmark.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fp:
        json.dump({"schema_version": 1,
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "note": "one render_frame call: rasterise + rigid composite + host copy; "
                           "excludes viser transport and the loop's max_fps cap",
                   "rows": rows}, fp, indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
