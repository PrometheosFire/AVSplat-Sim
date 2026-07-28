#!/usr/bin/env python3
"""Diagnostic for large-vehicle bicycle fits.

Reproduces & quantifies the two reported failure modes:
  (1) multiple-shooting segment handoff — discontinuities at segment boundaries
      in the baked trajectory, and single-shooting-from-params divergence.
  (2) sensitivity to early noisy predictions.

Run::
    PYTHONPATH="$PWD" envs/env_cc3dt/bin/python scripts/experiments/bicycle_diag.py \
        --scenes scene_020 scene_096
"""
from __future__ import annotations

import argparse

import numpy as np

from src.tracking.bicycle_fit import BicycleFitConfig, fit_track
from src.tracking.bicycle_kinematics import rollout, wheelbase_from_size
from src.tracking.refine_tracks import _bicycle_fit_config, _track_class

# reuse extraction helpers from the alpha sweep
from scripts.experiments.alpha_sweep import (
    _extract_inputs,
    _find_predictions,
    _fixed_fit_tracks,
    _load_cfg,
    LARGE_VEHICLES,
)
import json


def _seg_boundary_jumps(states: np.ndarray, node_frames: np.ndarray) -> np.ndarray:
    """Ground-position step size at each interior segment boundary vs its neighbours."""
    pos = states[:, :2]
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)  # per-frame step (T-1,)
    jumps = []
    for nf in node_frames[1:-1]:
        nf = int(nf)
        # step INTO the node (nf-1 -> nf) compared to the local median step
        lo = max(nf - 4, 0)
        hi = min(nf + 4, len(steps))
        local = np.median(steps[lo:hi]) if hi > lo else 0.0
        jumps.append(steps[nf - 1] - local)
    return np.asarray(jumps)


def _single_shoot(state0, accel, steer, dt, wheelbase, lr_ratio):
    return rollout(np.asarray(state0, float), np.asarray(accel, float),
                   np.asarray(steer, float), dt, wheelbase, lr_ratio)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["scene_020", "scene_096"])
    ap.add_argument("--alpha", type=float, default=None)
    args = ap.parse_args()

    cfg = _load_cfg()
    bicycle_cfg = cfg.get("bicycle_fit", {}) or {}
    fit_cfg = _bicycle_fit_config(bicycle_cfg)
    dt = float(bicycle_cfg.get("dt", 0.1))
    alpha = args.alpha if args.alpha is not None else float(bicycle_cfg.get("wheelbase_alpha", 0.45))
    lr_ratio = float(fit_cfg.lr_ratio)

    print(f"alpha={alpha} dt={dt} segment_len={fit_cfg.segment_len} lr_ratio={lr_ratio}")
    print(f"{'scene/cls/tid':28s} {'n':>4s} {'span':>4s} {'L':>5s} "
          f"{'msRMSE':>6s} {'ssRMSE':>6s} {'maxJump':>7s} {'SSmax':>7s} {'gate':>6s}")

    max_pos_rmse = float(bicycle_cfg.get("max_fit_pos_rmse", 3.0))
    for scene in args.scenes:
        pred = _find_predictions(scene)
        if not pred:
            print(f"[{scene}] no predictions")
            continue
        with open(pred) as f:
            data = json.load(f)
        results = data.get("results", data)
        tracks = _fixed_fit_tracks(results, cfg)

        for tid, track in tracks.items():
            cls = _track_class(track)
            if cls not in LARGE_VEHICLES:
                continue
            inp = _extract_inputs(track)
            L = wheelbase_from_size(inp["size_med"], alpha)
            res = fit_track(inp["offsets"], inp["positions"], inp["yaws"],
                            n_steps=inp["n_steps"], dt=dt, wheelbase=L, cfg=fit_cfg)

            jumps = _seg_boundary_jumps(res.states, res.node_frames)
            max_jump = float(np.max(np.abs(jumps))) if jumps.size else 0.0

            # single-shoot from persisted params (state0 + full controls)
            ss = _single_shoot(res.states[0], res.accel, res.steer, dt, L, lr_ratio)
            ss_max = float(np.linalg.norm(ss[:, :2] - res.states[:, :2], axis=1).max())

            # NEW gate: single-shoot RMSE vs observations (what gets baked now)
            ss_err = ss[inp["offsets"], :2] - inp["positions"]
            ss_rmse = float(np.sqrt(np.mean(np.sum(ss_err**2, axis=1))))
            gate = "ACCEPT" if ss_rmse <= max_pos_rmse else "reject"

            print(f"{scene[-3:]}/{cls[:10]:10s}/{tid:<5d} "
                  f"{inp['positions'].shape[0]:4d} {inp['n_steps']:4d} {L:5.2f} "
                  f"{res.pos_rmse:6.2f} {ss_rmse:6.2f} {max_jump:7.3f} {ss_max:7.3f} "
                  f"{gate:>6s}")


if __name__ == "__main__":
    main()
