#!/usr/bin/env python3
"""Compare segment_len for the bicycle fit: time + fit quality on long tracks."""
from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np

from src.tracking.bicycle_fit import fit_track
from src.tracking.bicycle_kinematics import rollout, wheelbase_from_size
from src.tracking.refine_tracks import _bicycle_fit_config, _track_class
from scripts.experiments.alpha_sweep import (
    _extract_inputs, _find_predictions, _fixed_fit_tracks, _load_cfg, LARGE_VEHICLES,
)

cfg = _load_cfg()
bcfg = cfg.get("bicycle_fit", {}) or {}
base = _bicycle_fit_config(bcfg)
dt = float(bcfg.get("dt", 0.1))
alpha = float(bcfg.get("wheelbase_alpha", 0.45))

# gather large-vehicle tracks from scene_096 (has the long ones)
results = json.load(open(_find_predictions("scene_096")))
results = results.get("results", results)
tracks = _fixed_fit_tracks(results, cfg)
targets = []
for tid, tr in tracks.items():
    if _track_class(tr) in LARGE_VEHICLES:
        inp = _extract_inputs(tr)
        if inp["n_steps"] >= 20:  # only the long ones
            targets.append((tid, _track_class(tr), inp))
targets.sort(key=lambda x: -x[2]["n_steps"])


def run(seglen):
    cfgL = replace(base, segment_len=seglen)
    rows = []
    t0 = time.perf_counter()
    for tid, cls, inp in targets:
        L = wheelbase_from_size(inp["size_med"], alpha)
        t1 = time.perf_counter()
        res = fit_track(inp["offsets"], inp["positions"], inp["yaws"],
                        n_steps=inp["n_steps"], dt=dt, wheelbase=L, cfg=cfgL)
        dt_fit = time.perf_counter() - t1
        traj = rollout(res.states[0], res.accel, res.steer, dt, L, base.lr_ratio)
        err = traj[inp["offsets"], :2] - inp["positions"]
        ss_rmse = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
        rows.append((tid, cls, inp["n_steps"], dt_fit, ss_rmse))
    total = time.perf_counter() - t0
    return total, rows


for seglen in (10, 100):
    total, rows = run(seglen)
    print(f"\n=== segment_len={seglen} | total {total:.1f}s for {len(rows)} long tracks ===")
    print(f"{'tid':>6} {'cls':10} {'span':>4} {'fit_s':>7} {'ssRMSE':>7} {'gate':>7}")
    for tid, cls, span, dt_fit, ss_rmse in rows:
        gate = "ACCEPT" if ss_rmse <= 3.0 else "reject"
        print(f"{tid:>6} {cls[:10]:10} {span:>4} {dt_fit:7.2f} {ss_rmse:7.2f} {gate:>7}")
