#!/usr/bin/env python3
"""Wheelbase-alpha sweep for the kinematic bicycle track fit (parallel).

Question this answers
---------------------
The wheelbase is estimated as ``L = wheelbase_alpha * max(size_x, size_y)`` with
a single global ``alpha`` for every vehicle class. This script sweeps ``alpha``
and reports the resulting per-track fit residuals (``pos_rmse`` / ``yaw_rmse``),
aggregated per class, so we can see:

  * whether the fit quality actually depends on alpha at all, and
  * whether cars vs. larger vehicles (truck/bus/trailer/construction_vehicle)
    prefer *different* alphas (i.e. would per-class alpha help?).

Because :func:`fit_track` runs independently per track, one sweep broken down by
class is sufficient to read off the best alpha for each class separately.

Design
------
Fusion + static filtering run once per scene (default config) so the fitted set
is fixed. Per-track fit *inputs* are extracted once; only the wheelbase changes
with alpha. All (track, alpha) fits are dispatched across a process pool.

Run (env_cc3dt has numpy + scipy)::

    PYTHONPATH="$PWD" envs/env_cc3dt/bin/python scripts/experiments/alpha_sweep.py \
        --scenes scene_020 scene_096 --workers 8
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml

from src.tracking.bicycle_fit import BicycleFitConfig, fit_track
from src.tracking.bicycle_kinematics import wheelbase_from_size
from src.tracking.refine_tracks import (
    apply_fusion,
    build_tracks,
    filter_static,
    fuse_tracks,
    select_bicycle_track_ids,
    _bicycle_fit_config,
    _frame_index,
    _track_class,
    _yaw_from_box_rotation,
)
from src.tracking.track_geometry import GROUND_AXES

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REFINE_CFG = os.path.join(REPO, "configs", "pipeline", "refine_tracks.yaml")

LARGE_VEHICLES = {"truck", "bus", "trailer", "construction_vehicle"}

# Identical across all fits; built once per worker process.
_FIT_CFG: BicycleFitConfig | None = None
_DT: float = 0.1


def _find_predictions(scene: str) -> str | None:
    pat = os.path.join(
        REPO, "results", "4dgs", "wayve101", scene,
        "10_tracking_*", "eval", "track_3d_predictions_colmap.json",
    )
    hits = sorted(glob.glob(pat), key=os.path.getmtime)
    return hits[-1] if hits else None


def _load_cfg() -> dict:
    with open(REFINE_CFG, "r") as f:
        return yaml.safe_load(f)


def _fixed_fit_tracks(results: dict, cfg: dict) -> dict:
    """Fusion + filter once; return dict{tid: track} of fittable vehicle tracks."""
    frame_keys, _ = _frame_index(results)
    tracks = build_tracks(results)
    bicycle_cfg = cfg.get("bicycle_fit", {}) or {}

    mapping = fuse_tracks(
        tracks,
        iou_thr=float(cfg.get("iou_thr", 0.3)),
        min_overlap_frames=int(cfg.get("min_overlap_frames", 1)),
        max_gap=int(cfg.get("max_gap", 5)),
        gap_dist=float(cfg.get("gap_dist", 2.0)),
        class_groups=cfg.get("class_groups", None),
        merge_dist=float(cfg.get("merge_dist", 2.0)),
        persistent_overlap_frames=int(cfg.get("persistent_overlap_frames", 15)),
        persistent_merge_dist=float(cfg.get("persistent_merge_dist", 3.5)),
        containment_frac=float(cfg.get("containment_frac", 0.7)),
        divergence_cap=float(cfg.get("divergence_cap", 6.0)),
        model_assisted_gap=False,  # keep fusion alpha-independent
    )
    fused, _ = apply_fusion(
        tracks, mapping,
        pose_filter_cfg=cfg.get("fusion_pose_filter", {}) or {},
        frame_keys=frame_keys,
    )
    kept, _ = filter_static(
        fused,
        min_displacement=float(cfg.get("min_displacement", 2.5)),
        min_track_length=int(cfg.get("min_track_length", 3)),
        displacement_percentile=float(cfg.get("displacement_percentile", 90.0)),
        displacement_mode=str(cfg.get("displacement_mode", "net")),
        displacement_smooth=int(cfg.get("displacement_smooth", 3)),
        min_rel_displacement=float(cfg.get("min_rel_displacement", 0.0)),
    )
    fit_ids = set(select_bicycle_track_ids(kept, bicycle_cfg))
    return {tid: kept[tid] for tid in fit_ids}


def _extract_inputs(track: dict) -> dict:
    """Per-track fit inputs (offsets, positions, yaws, size_med, n_steps)."""
    ax0, ax1 = GROUND_AXES
    frames = sorted(track)
    first, last = frames[0], frames[-1]
    n_steps = last - first
    offsets = np.asarray([f - first for f in frames], dtype=np.int64)
    positions = np.asarray(
        [[track[f]["translation"][ax0], track[f]["translation"][ax1]] for f in frames],
        dtype=np.float64,
    )
    yaws = np.asarray([_yaw_from_box_rotation(track[f]) for f in frames], dtype=np.float64)
    sizes = np.asarray([track[f]["size"] for f in frames], dtype=np.float64)
    size_med = np.median(sizes, axis=0)
    return {
        "offsets": offsets,
        "positions": positions,
        "yaws": yaws,
        "size_med": size_med,
        "n_steps": int(n_steps),
    }


def _worker_init(fit_cfg_kwargs: dict, dt: float) -> None:
    global _FIT_CFG, _DT
    _FIT_CFG = BicycleFitConfig(**fit_cfg_kwargs)
    _DT = dt


def _fit_one(task: tuple) -> tuple:
    """One (scene, tid, class, alpha, inputs) fit -> (alpha, class, pos_rmse, yaw_rmse, n_obs)."""
    _scene, _tid, cls, alpha, inp = task
    wheelbase = wheelbase_from_size(inp["size_med"], alpha)
    res = fit_track(
        inp["offsets"], inp["positions"], inp["yaws"],
        n_steps=inp["n_steps"], dt=_DT, wheelbase=wheelbase, cfg=_FIT_CFG,
    )
    return (alpha, cls, float(res.pos_rmse), float(res.yaw_rmse), int(res.n_obs))


def sweep(scenes: list[str], alphas: list[float], workers: int) -> None:
    cfg = _load_cfg()
    bicycle_cfg = cfg.get("bicycle_fit", {}) or {}
    fit_cfg = _bicycle_fit_config(bicycle_cfg)
    fit_cfg_kwargs = fit_cfg.__dict__.copy()
    dt = float(bicycle_cfg.get("dt", 0.1))

    tasks: list[tuple] = []
    for scene in scenes:
        pred_path = _find_predictions(scene)
        if not pred_path:
            print(f"[skip] {scene}: no track_3d_predictions_colmap.json found", flush=True)
            continue
        with open(pred_path, "r") as f:
            data = json.load(f)
        results = data.get("results", data)
        fit_tracks = _fixed_fit_tracks(results, cfg)
        n_large = sum(_track_class(t) in LARGE_VEHICLES for t in fit_tracks.values())
        n_car = sum(_track_class(t) == "car" for t in fit_tracks.values())
        print(f"[{scene}] fitted tracks: {len(fit_tracks)} ({n_large} large / {n_car} car)", flush=True)
        for tid, track in fit_tracks.items():
            cls = _track_class(track)
            inp = _extract_inputs(track)
            for alpha in alphas:
                tasks.append((scene, tid, cls, float(alpha), inp))

    print(f"\nDispatching {len(tasks)} fits across {workers} workers...", flush=True)

    records: dict[float, dict[str, list]] = {a: defaultdict(list) for a in alphas}
    done = 0
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init, initargs=(fit_cfg_kwargs, dt)
    ) as ex:
        futs = [ex.submit(_fit_one, t) for t in tasks]
        for fut in as_completed(futs):
            alpha, cls, pos_rmse, yaw_rmse, n_obs = fut.result()
            records[alpha][cls].append((pos_rmse, yaw_rmse, n_obs))
            done += 1
            if done % 25 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} fits done", flush=True)

    _print_tables(records, alphas)


def _agg(rows: list) -> tuple[float, float, float, int]:
    """mean pos, median pos, mean yaw(deg), n_tracks."""
    if not rows:
        return float("nan"), float("nan"), float("nan"), 0
    pos = [r[0] for r in rows if not np.isnan(r[0])]
    yaw = [np.degrees(r[1]) for r in rows if not np.isnan(r[1])]
    pm = statistics.fmean(pos) if pos else float("nan")
    pmed = statistics.median(pos) if pos else float("nan")
    ym = statistics.fmean(yaw) if yaw else float("nan")
    return pm, pmed, ym, len(rows)


def _group(records_for_alpha: dict[str, list], names) -> list:
    if isinstance(names, str):
        return list(records_for_alpha.get(names, []))
    out: list = []
    for n in names:
        out.extend(records_for_alpha.get(n, []))
    return out


def _print_tables(records: dict[float, dict[str, list]], alphas: list[float]) -> None:
    def section(title: str, selector) -> None:
        print(f"\n=== {title} ===")
        print(f"{'alpha':>6} | {'pos_mean(m)':>11} | {'pos_med(m)':>10} | {'yaw_mean(deg)':>13} | {'n':>4}")
        print("-" * 60)
        best_pos = (float("inf"), None)
        best_yaw = (float("inf"), None)
        for a in alphas:
            rows = selector(records[a])
            pm, pmed, ym, n = _agg(rows)
            if not np.isnan(pm) and pm < best_pos[0]:
                best_pos = (pm, a)
            if not np.isnan(ym) and ym < best_yaw[0]:
                best_yaw = (ym, a)
            print(f"{a:>6.2f} | {pm:>11.4f} | {pmed:>10.4f} | {ym:>13.3f} | {n:>4}")
        print(f"  -> best pos-mean alpha={best_pos[1]}   best yaw-mean alpha={best_yaw[1]}")

    section("ALL vehicles", lambda r: _group(r, {"car"} | LARGE_VEHICLES))
    section("car", lambda r: _group(r, "car"))
    section("large (truck/bus/trailer/construction)", lambda r: _group(r, LARGE_VEHICLES))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", nargs="+", default=["scene_020", "scene_096"])
    ap.add_argument(
        "--alphas", nargs="+", type=float,
        default=[0.35, 0.45, 0.55, 0.60, 0.70, 0.85, 1.00],
    )
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()
    print(f"Scenes: {args.scenes}")
    print(f"Alphas: {args.alphas}")
    print(f"Workers: {args.workers}")
    sweep(args.scenes, args.alphas, args.workers)


if __name__ == "__main__":
    main()
