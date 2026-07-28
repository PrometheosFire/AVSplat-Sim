#!/usr/bin/env python3
"""Focused diagnostic for a single (class, id) track: why the bicycle gate
accepts or rejects it. Default target: scene_099 bus id=30."""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.tracking.bicycle_fit import fit_track
from src.tracking.bicycle_kinematics import rollout, wheelbase_from_size
from src.tracking.refine_tracks import (
    _bicycle_fit_config, _track_class, _preclean_fit_inputs, _fit_with_trimming,
)

from scripts.experiments.alpha_sweep import (
    _extract_inputs, _find_predictions, _fixed_fit_tracks, _load_cfg,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene_099")
    ap.add_argument("--cls", default="bus")
    ap.add_argument("--tid", type=int, default=30)
    args = ap.parse_args()

    cfg = _load_cfg()
    bcfg = cfg.get("bicycle_fit", {}) or {}
    fit_cfg = _bicycle_fit_config(bcfg)
    dt = float(bcfg.get("dt", 0.1))
    alpha = float(bcfg.get("wheelbase_alpha", 0.45))
    max_pos_rmse = float(bcfg.get("max_fit_pos_rmse", 3.0))
    lr_ratio = float(fit_cfg.lr_ratio)
    print(f"alpha={alpha} dt={dt} segment_len={fit_cfg.segment_len} "
          f"max_fit_pos_rmse={max_pos_rmse}")

    data = json.load(open(_find_predictions(args.scene)))
    results = data.get("results", data)
    tracks = _fixed_fit_tracks(results, cfg)

    match = None
    for tid, track in tracks.items():
        if int(tid) == args.tid and _track_class(track) == args.cls:
            match = (tid, track)
            break
    if match is None:
        avail = sorted((int(t), _track_class(tr)) for t, tr in tracks.items()
                       if _track_class(tr) == args.cls)
        print(f"{args.cls}_{args.tid} not in fittable set. Available {args.cls}: {avail}")
        return

    tid, track = match
    inp = _extract_inputs(track)
    pos = inp["positions"]
    yaws = inp["yaws"]
    offs = inp["offsets"]
    L = wheelbase_from_size(inp["size_med"], alpha)
    span = offs[-1] - offs[0]
    print(f"\n{args.cls}_{tid}: n_obs={pos.shape[0]} span={span} "
          f"n_steps={inp['n_steps']} L={L:.2f} size_med={np.round(inp['size_med'],2)}")

    # per-frame observation step sizes (detect teleports / noisy jumps)
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    dframe = np.diff(offs)
    speed = steps / (dframe * dt)
    print(f"obs step: median={np.median(steps):.2f}m max={steps.max():.2f}m "
          f"@offset {offs[int(np.argmax(steps))]} | "
          f"implied speed median={np.median(speed):.1f} max={speed.max():.1f} m/s")

    res = fit_track(offs, pos, yaws, n_steps=inp["n_steps"], dt=dt,
                    wheelbase=L, cfg=fit_cfg)
    print(f"fit: success={res.success} cost={res.cost:.1f} "
          f"pos_rmse(ms)={res.pos_rmse:.2f} yaw_rmse={res.yaw_rmse:.3f}")

    # single continuous rollout from persisted params (what gets baked)
    traj = rollout(res.states[0], res.accel, res.steer, dt, L, lr_ratio)
    ss_err = traj[offs, :2] - pos
    ss_dist = np.linalg.norm(ss_err, axis=1)
    ss_rmse = float(np.sqrt(np.mean(ss_dist**2)))
    gate = "ACCEPT" if (np.isfinite(ss_rmse) and ss_rmse <= max_pos_rmse) else "REJECT"
    print(f"[RAW]      ssRMSE={ss_rmse:.2f}m  max_err={ss_dist.max():.2f}m "
          f"-> gate={gate}")

    # ---- NEW: pre-clean (yaw unflip + spike drop) then fit ---------------- #
    c_off, c_pos, c_yaw, n_drop = _preclean_fit_inputs(offs, pos, yaws, bcfg)
    print(f"\npre-clean: dropped {n_drop} spike frame(s); "
          f"kept {c_off.size}/{offs.size} obs")
    res2 = fit_track(c_off, c_pos, c_yaw, n_steps=inp["n_steps"], dt=dt,
                     wheelbase=L, cfg=fit_cfg)
    traj2 = rollout(res2.states[0], res2.accel, res2.steer, dt, L, lr_ratio)
    e2 = np.linalg.norm(traj2[c_off, :2] - c_pos, axis=1)
    rmse2 = float(np.sqrt(np.mean(e2**2)))
    gate2 = "ACCEPT" if (np.isfinite(rmse2) and rmse2 <= max_pos_rmse) else "REJECT"
    print(f"[PRECLEAN] ssRMSE={rmse2:.2f}m -> gate={gate2}")

    # ---- trimmed refit fallback if still failing -------------------------- #
    if gate2 == "REJECT":
        res3, traj3, rmse3, t_off, t_pos = _fit_with_trimming(
            c_off, c_pos, c_yaw, inp["n_steps"], dt, L, fit_cfg,
            res2, max_pos_rmse, bcfg,
        )
        gate3 = "ACCEPT" if (np.isfinite(rmse3) and rmse3 <= max_pos_rmse) else "REJECT"
        print(f"[TRIMFIT]  ssRMSE={rmse3:.2f}m  kept {t_off.size}/{c_off.size} obs "
              f"-> gate={gate3}")
        final = gate3
    else:
        final = gate2

    print(f"\nFINAL decision: {args.cls}_{tid} -> "
          f"{'bicycle model kept' if final=='ACCEPT' else 'REJECTED, legacy smoothing'}")
    return


if __name__ == "__main__":
    main()
