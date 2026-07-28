#!/usr/bin/env python3
"""Verify parallel bicycle fitting matches serial output and measure speedup."""
from __future__ import annotations

import copy
import json
import time

from src.tracking.refine_tracks import (
    apply_bicycle_fit, build_tracks, _frame_index, tracks_to_results,
    select_bicycle_track_ids,
)
from scripts.experiments.alpha_sweep import (
    _find_predictions, _fixed_fit_tracks, _load_cfg,
)

cfg = _load_cfg()
bcfg = dict(cfg.get("bicycle_fit", {}) or {})

results = json.load(open(_find_predictions("scene_096")))
results = results.get("results", results)
frame_keys, _ = _frame_index(results)
kept = _fixed_fit_tracks(results, cfg)  # fittable vehicle tracks
print(f"{len(kept)} fittable tracks")


def run(workers):
    tracks = copy.deepcopy(kept)
    c = dict(bcfg); c["fit_workers"] = workers
    t0 = time.perf_counter()
    n, rep = apply_bicycle_fit(tracks, frame_keys, c)
    dt = time.perf_counter() - t0
    # serialize the baked boxes for exact comparison
    payload = json.dumps(tracks_to_results(tracks, frame_keys), sort_keys=True)
    return dt, n, sorted(rep["rejected_ids"]), payload


dt_s, n_s, rej_s, pay_s = run(1)
print(f"serial : {dt_s:6.1f}s  fitted={n_s} rejected={len(rej_s)}")
dt_p, n_p, rej_p, pay_p = run(0)
print(f"parallel: {dt_p:6.1f}s  fitted={n_p} rejected={len(rej_p)}")
print(f"speedup : {dt_s/dt_p:.1f}x")
print(f"identical output: {pay_s == pay_p}  (rejected match: {rej_s == rej_p})")
