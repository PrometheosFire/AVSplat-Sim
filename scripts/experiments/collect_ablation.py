"""Collect an ablation sweep into the markdown tables in docs/ablation_scene084.md.

Reads every variant directory under a Hydra sweep dir and prints two tables:
headline metrics as deltas against the baseline, and the per-camera PSNR
breakdown, which is where this study's specific defect lives (a 3.2 dB spread
across cameras). Paste the output into the Results section.

Per-camera numbers come from the saved [GT | pred] validation canvases rather
than the stats file, because eval only records the aggregate. Validation frames
are drawn camera-major, so the canvases split into equal per-camera blocks; the
last block absorbs any remainder.

Usage:
    envs/envs/env_gsplat/bin/python scripts/experiments/collect_ablation.py \\
        multirun/ablation_scene084
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

# Order matters: the first entry is the reference every delta is measured against.
LEVERS = [
    "abl_baseline", "abl_depth", "abl_ppisp", "abl_ppisp_ctrl", "abl_bilateral",
    "abl_app", "abl_antialiased", "abl_reg_low", "abl_depth_ppisp", "abl_cap3m",
]
# Same recipe as abl_baseline, trained longer. Reported as their own table
# because they are a curve, not independent levers.
STEPS = ["abl_steps_45k", "abl_steps_60k", "abl_steps_75k", "abl_steps_90k"]
ORDER = LEVERS + STEPS
N_CAMERAS = 5


def _fmt_hms(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_stats(variant_dir: str):
    """Latest val stats for one variant, or None if it never got that far."""
    files = sorted(glob.glob(os.path.join(variant_dir, "stats", "val_step*.json")))
    if not files:
        return None
    with open(files[-1]) as fp:
        return json.load(fp)


def rigid_count(variant_dir: str):
    """Rigid Gaussians from the checkpoint. Absent when save_steps was empty."""
    cks = sorted(glob.glob(os.path.join(variant_dir, "ckpts", "*.pt")))
    if not cks:
        return None
    try:
        import torch
        ck = torch.load(cks[-1], map_location="cpu", weights_only=False)
        rn = ck.get("rigid_nodes")
        return int(rn["gauss.means"].shape[0]) if rn else 0
    except Exception:
        return None


def train_time(variant_dir: str):
    """Wall time and step count, read from the marker train_splats.py writes.

    ``.success`` carries both, e.g.::

        GSplat training completed successfully (45000 steps).
        duration: 01:07:49

    That is authoritative -- it is measured around the training call itself --
    so it is preferred over any timestamp arithmetic. Falls back to file mtimes
    only when the marker is missing, which means the run did not finish.
    """
    marker = os.path.join(variant_dir, ".success")
    if os.path.exists(marker):
        duration, steps = None, None
        with open(marker) as fp:
            for line in fp:
                line = line.strip()
                if line.startswith("duration:"):
                    duration = line.split(":", 1)[1].strip()
                m = re.search(r"\((\d+) steps\)", line)
                if m:
                    steps = int(m.group(1))
        if duration:
            return duration, steps

    logs = glob.glob(os.path.join(variant_dir, "*.log"))
    stats = glob.glob(os.path.join(variant_dir, "stats", "val_step*.json"))
    if not logs or not stats:
        return None, None
    start = min(os.path.getctime(p) for p in logs)
    end = max(os.path.getmtime(p) for p in stats)
    return (_fmt_hms(end - start) + "*", None) if end > start else (None, None)


def per_camera_psnr(variant_dir: str):
    """Mean PSNR per camera-major block of the saved validation canvases."""
    try:
        import cv2
    except ImportError:
        return None
    fs = sorted(
        f for f in glob.glob(os.path.join(variant_dir, "renders", "val_step*.png"))
        if "_boxes" not in f
    )
    if not fs:
        return None
    vals = []
    for f in fs:
        im = cv2.imread(f)
        if im is None:
            continue
        w = im.shape[1] // 2
        gt, pr = im[:, :w].astype(np.float64), im[:, w:].astype(np.float64)
        mse = ((gt - pr) ** 2).mean()
        vals.append(10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf"))
    if not vals:
        return None
    vals = np.array(vals)
    per = len(vals) // N_CAMERAS
    if per == 0:
        return None
    return [
        float(vals[i * per:(i + 1) * per].mean()) if i < N_CAMERAS - 1
        else float(vals[(N_CAMERAS - 1) * per:].mean())
        for i in range(N_CAMERAS)
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", help="e.g. multirun/ablation_scene084")
    args = ap.parse_args()

    rows = {}
    for name in ORDER:
        d = os.path.join(args.sweep_dir, name)
        if not os.path.isdir(d):
            continue
        rows[name] = {
            "stats": load_stats(d),
            "rigid": rigid_count(d),
            "time": train_time(d),
            "cams": per_camera_psnr(d),
        }
    if not rows:
        print(f"No variant directories found under {args.sweep_dir}", file=sys.stderr)
        return 1

    base = rows.get(ORDER[0], {}).get("stats")

    def delta(v, key, fmt="{:+.3f}"):
        if base is None or v is None or key not in v or key not in base:
            return ""
        return fmt.format(v[key] - base[key])

    def metric_row(name, r):
        s = r["stats"]
        d_psnr = "—" if name == ORDER[0] else delta(s, "psnr")
        d_lpips = "—" if name == ORDER[0] else delta(s, "lpips", "{:+.4f}")
        cc = f"{s['cc_psnr']:.3f}" if "cc_psnr" in s else ""
        rigid = f"{r['rigid']:,}" if r["rigid"] is not None else ""
        return (f"| `{name}` | {s['psnr']:.3f} | {d_psnr} | {cc} | {s['ssim']:.4f} | "
                f"{s['lpips']:.4f} | {d_lpips} | {s['num_GS']:,} | {rigid} | "
                f"{r['time'][0] or ''} |")

    print("## Headline metrics\n")
    print("| variant | PSNR | Δ | cc-PSNR | SSIM | LPIPS | Δ LPIPS | bg #GS | rigid #GS | time |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name in LEVERS:
        r = rows.get(name)
        if r is None:
            print(f"| `{name}` | *not run* | | | | | | | | |")
        elif r["stats"] is None:
            print(f"| `{name}` | **failed** | | | | | | | | |")
        else:
            print(metric_row(name, r))

    print("\n## Step-count sweep\n")
    print("_Same recipe as `abl_baseline`, trained longer. Read as a curve: a "
          "flattening slope means the model has saturated and the remaining gap "
          "is structural._\n")
    print("| variant | steps | PSNR | Δ vs 30k | Δ vs previous | LPIPS | bg #GS | time | min/1k steps |")
    print("|---|---|---|---|---|---|---|---|---|")
    prev = None
    for name in [ORDER[0]] + STEPS:
        r = rows.get(name)
        if r is None or r["stats"] is None:
            print(f"| `{name}` | | *not run* | | | | | | |")
            continue
        s = r["stats"]
        steps = r["time"][1]
        dur = r["time"][0] or ""
        rate = ""
        if steps and dur and ":" in dur and not dur.endswith("*"):
            h, m, sec = (int(x) for x in dur.split(":"))
            rate = f"{(h * 60 + m + sec / 60) / (steps / 1000):.2f}"
        d_base = "—" if name == ORDER[0] else delta(s, "psnr")
        d_prev = "—" if prev is None else f"{s['psnr'] - prev:+.3f}"
        print(f"| `{name}` | {steps or ''} | {s['psnr']:.3f} | {d_base} | {d_prev} | "
              f"{s['lpips']:.4f} | {s['num_GS']:,} | {dur} | {rate} |")
        prev = s["psnr"]

    print("\n## Per-camera PSNR\n")
    print("| variant | cam 0 | cam 1 | cam 2 | cam 3 | cam 4 | spread |")
    print("|---|---|---|---|---|---|---|")
    print("| baseline (45k ref) | 25.36 | 25.33 | 24.46 | 22.17 | 24.35 | 3.19 |")
    for name in ORDER:
        r = rows.get(name)
        if r is None or not r["cams"]:
            continue
        c = r["cams"]
        cells = " | ".join(f"{x:.2f}" for x in c)
        print(f"| `{name}` | {cells} | {max(c) - min(c):.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
