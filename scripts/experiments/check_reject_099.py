#!/usr/bin/env python3
"""Reproduce the base auto-refine for scene_099 using the exact config of run
15_refine_9157231b and report which vehicle tracks the bicycle fit rejected."""
from __future__ import annotations

import json

from omegaconf import OmegaConf

from src.tracking.refine_tracks import refine, build_tracks
from scripts.experiments.alpha_sweep import _find_predictions

RUN = "results/4dgs/wayve101/scene_099/15_refine_9157231b"
cfg = OmegaConf.load(f"{RUN}/.hydra/config.yaml")
refine_task = cfg.refine_task

results = json.load(open(_find_predictions("scene_099")))
results = results.get("results", results)

# class/id per track (from raw predictions) for readable labels
tracks = build_tracks(results)
label = {}
for tid, tr in tracks.items():
    cls = None
    for b in tr.values():
        cls = b.get("category") or b.get("class") or b.get("label")
        if cls:
            break
    label[int(tid)] = cls

refined, report = refine(results, refine_task)
brep = report.get("bicycle_fit", {}) or {}
rejected = [int(x) for x in brep.get("rejected_ids", [])]

print("tracks_fitted:", brep.get("tracks_fitted"))
print("rejected_ids :", sorted(rejected))
print("rejected labels:")
for tid in sorted(rejected):
    print(f"  {label.get(tid)}_{tid}")

# focus on the bus track(s)
buses = [tid for tid, c in label.items() if c == "bus"]
print("\nbus tracks:", sorted(buses))
for tid in sorted(buses):
    print(f"  bus_{tid}: {'REJECTED -> legacy smoothing' if tid in rejected else 'bicycle-fit accepted'}")
