"""Step 1.5: track refinement (fuse split tracks, then filter static objects).

Operates on the COLMAP-frame predictions written by the tracker
(``track_3d_predictions_colmap.json``) and produces a refined JSON with:

1. Fusion: track-ID switches for the same physical object are merged via a
   union-find over two signals — (a) bird's eye IoU overlap when two tracks are
   co-present in shared frames, and (b) a velocity-predicted gap handoff when a
   track briefly disappears and another appears nearby. Fusion is transitive
   (A~B, B~C => A~B~C) and allowed across configurable class groups (e.g. a
   pickup flickering between ``car`` and ``truck``).
2. Filtering: tracks whose centroid never moves beyond a threshold (parked /
   static objects) are dropped, along with too-short noise tracks.

Run standalone or via the 4DGS orchestrator. Input/output are resolved from
``cfg.refine_task``.
"""
from __future__ import annotations

import copy
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace as _dc_replace

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.tracking.track_geometry import (
    bev_iou,
    box_center_ground,
    quat_wxyz_to_matrix,
    GROUND_AXES,
    UP_AXIS,
)
from src.tracking.bicycle_fit import BicycleFitConfig, fit_track
from src.tracking.bicycle_kinematics import rollout as _bicycle_rollout
from src.tracking.bicycle_kinematics import wheelbase_from_size
from src.tracking.bicycle_kinematics import STATE_THETA, STATE_V

try:  # Optional progress bar; falls back to a plain loop if unavailable.
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - tqdm is an optional convenience dep
    _tqdm = None


# --------------------------------------------------------------------------- #
# Data loading / track assembly
# --------------------------------------------------------------------------- #
def _frame_index(results: dict) -> tuple[list[str], dict[str, int]]:
    """Order frame tokens by their trailing integer timestamp."""
    keys = sorted(results.keys(), key=lambda k: int(k.split("_")[-1]))
    return keys, {k: i for i, k in enumerate(keys)}


def build_tracks(results: dict) -> dict[int, dict[int, dict]]:
    """Group boxes by tracking id: ``{track_id: {frame_index: box}}``."""
    keys, fidx = _frame_index(results)
    tracks: dict[int, dict[int, dict]] = defaultdict(dict)
    for token in keys:
        for box in results[token]:
            tracks[int(box["tracking_id"])][fidx[token]] = box
    return tracks


# --------------------------------------------------------------------------- #
# Union-find
# --------------------------------------------------------------------------- #
class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# --------------------------------------------------------------------------- #
# Class-group compatibility
# --------------------------------------------------------------------------- #
def _track_class(track: dict[int, dict]) -> str:
    """Score-weighted majority class of a track."""
    votes: Counter = Counter()
    for box in track.values():
        votes[box.get("tracking_name", "unknown")] += float(
            box.get("tracking_score", 1.0)
        )
    return votes.most_common(1)[0][0]


def _class_group_lookup(class_groups):
    """Map each class name to a group id. Returns (lookup, unrestricted)."""
    if not class_groups:
        return {}, True
    lookup: dict[str, int] = {}
    for gid, group in enumerate(class_groups):
        for name in group:
            lookup[str(name)] = gid
    return lookup, False


def _classes_compatible(cls_a, cls_b, lookup, unrestricted) -> bool:
    if unrestricted:
        return True
    if cls_a == cls_b:
        return True
    ga, gb = lookup.get(cls_a), lookup.get(cls_b)
    return ga is not None and ga == gb


# --------------------------------------------------------------------------- #
# Fusion signals
# --------------------------------------------------------------------------- #
def _estimate_ground_velocity(track: dict[int, dict], window: int = 3):
    """Per-frame ground velocity from the track's most recent centroids."""
    frames = sorted(track)
    if len(frames) < 2:
        return np.zeros(2)
    last = frames[-1]
    prev = frames[max(0, len(frames) - 1 - window)]
    span = last - prev
    if span <= 0:
        return np.zeros(2)
    p_last = box_center_ground(track[last])
    p_prev = box_center_ground(track[prev])
    return (p_last - p_prev) / float(span)


def _bicycle_gap_prediction(
    track: dict[int, dict],
    gap: int,
    dt: float,
    wheelbase_alpha: float,
    lr_ratio: float,
    window: int = 4,
) -> np.ndarray | None:
    """Extrapolate a track's endpoint across a gap with the bicycle model.

    Estimates the tail speed and (constant) yaw-rate from the last few centroids
    and rolls the CoG bicycle model forward ``gap`` frames. Unlike a
    constant-velocity guess this follows a turn through the gap, so a vehicle
    that reappears after briefly disappearing mid-corner is still matched.
    Returns the predicted ground center ``(x, z)``, or ``None`` when the tail is
    too short/slow to extrapolate reliably.
    """
    frames = sorted(track)
    if len(frames) < 3:
        return None
    last = frames[-1]
    prev = frames[max(0, len(frames) - 1 - window)]
    span = last - prev
    if span <= 0:
        return None

    p_last = box_center_ground(track[last])
    p_prev = box_center_ground(track[prev])
    step = (p_last - p_prev) / float(span)  # per-frame displacement
    speed = float(np.linalg.norm(step)) / max(dt, 1e-6)
    if speed < 0.3:  # essentially stopped: constant-velocity handles it
        return None

    heading = float(np.arctan2(step[1], step[0]))

    # Constant yaw-rate from the change of motion heading between the first and
    # second half of the tail window.
    mid_idx = max(0, len(frames) - 1 - window // 2)
    mid = frames[mid_idx]
    yaw_rate = 0.0
    if prev < mid < last:
        h1 = box_center_ground(track[mid]) - p_prev
        h2 = p_last - box_center_ground(track[mid])
        if np.linalg.norm(h1) > 1e-6 and np.linalg.norm(h2) > 1e-6:
            a1 = np.arctan2(h1[1], h1[0])
            a2 = np.arctan2(h2[1], h2[0])
            d_yaw = float(np.arctan2(np.sin(a2 - a1), np.cos(a2 - a1)))
            yaw_rate = d_yaw / (0.5 * span * max(dt, 1e-6))

    size = np.asarray(track[last].get("size", [0.0, 0.0, 0.0]), dtype=np.float64)
    wheelbase = wheelbase_from_size(size, wheelbase_alpha)
    lr = max(lr_ratio * wheelbase, 1e-6)
    sin_beta = float(np.clip(yaw_rate * lr / max(speed, 1e-6), -0.99, 0.99))
    beta = np.arcsin(sin_beta)
    steer = float(np.arctan(np.tan(beta) / max(lr_ratio, 1e-6)))

    state0 = np.array([p_last[0], p_last[1], heading, speed], dtype=np.float64)
    accel = np.zeros(gap)
    steer_seq = np.full(gap, steer)
    states = _bicycle_rollout(state0, accel, steer_seq, dt, wheelbase, lr_ratio)
    return states[-1, :2]



def fuse_tracks(
    tracks: dict[int, dict[int, dict]],
    iou_thr: float,
    min_overlap_frames: int,
    max_gap: int,
    gap_dist: float,
    class_groups,
    merge_dist: float = 2.0,
    persistent_overlap_frames: int = 15,
    persistent_merge_dist: float = 3.5,
    containment_frac: float = 0.7,
    divergence_cap: float = 6.0,
    model_assisted_gap: bool = False,
    gap_model_alpha: float = 0.6,
    gap_model_lr_ratio: float = 0.5,
    gap_model_dt: float = 0.1,
) -> dict[int, int]:
    """Return a mapping ``track_id -> canonical_track_id`` after fusion."""
    ids = list(tracks)
    uf = _UnionFind(ids)
    lookup, unrestricted = _class_group_lookup(class_groups)
    cls = {tid: _track_class(tr) for tid, tr in tracks.items()}
    frames_of = {tid: set(tr.keys()) for tid, tr in tracks.items()}
    span_of = {
        tid: (min(tr), max(tr)) if tr else (0, -1) for tid, tr in tracks.items()
    }

    # Precompute the max co-present distance between every pair of tracks that
    # share at least one frame. Two fragments of one object stay close (<~6 m
    # even under depth jitter); two distinct vehicles that briefly coincide
    # diverge far. This drives a divergence guard that stops transitive chains
    # from pulling distinct cars into one group through a long "hub" track.
    co_max: dict[tuple[int, int], float] = {}
    frame_boxes: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for tid, tr in tracks.items():
        for f, box in tr.items():
            frame_boxes[f].append((tid, box_center_ground(box)))
    for lst in frame_boxes.values():
        for p in range(len(lst)):
            ta, ca = lst[p]
            for q in range(p + 1, len(lst)):
                tb, cb = lst[q]
                key = (ta, tb) if ta < tb else (tb, ta)
                dd = float(np.linalg.norm(ca - cb))
                if dd > co_max.get(key, 0.0):
                    co_max[key] = dd

    group_members: dict[int, set[int]] = {tid: {tid} for tid in ids}

    def guarded_union(a: int, b: int) -> None:
        """Union a and b unless it would group two divergent co-present tracks."""
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            return
        ma = group_members.get(ra, {ra})
        mb = group_members.get(rb, {rb})
        for x in ma:
            for y in mb:
                key = (x, y) if x < y else (y, x)
                if co_max.get(key, 0.0) > divergence_cap:
                    return  # refuse: distinct cars would be chained together
        uf.union(a, b)
        nr = uf.find(a)
        merged = ma | mb
        group_members[nr] = merged
        for r in (ra, rb):
            if r != nr and r in group_members:
                del group_members[r]

    for i in range(len(ids)):
        a = ids[i]
        for j in range(i + 1, len(ids)):
            b = ids[j]
            if uf.find(a) == uf.find(b):
                continue
            if not _classes_compatible(cls[a], cls[b], lookup, unrestricted):
                continue

            shared = frames_of[a] & frames_of[b]
            if len(shared) >= min_overlap_frames:
                # Signal 1: co-present overlap. Mean BEV IoU catches well-aligned
                # duplicates; a small min centroid distance catches depth-jitter
                # duplicates whose footprints miss in many frames (dragging the
                # mean down) but repeatedly coincide. The divergence guard in
                # guarded_union keeps single-frame coincidences of distinct cars
                # from merging.
                shared_dists = [
                    float(
                        np.linalg.norm(
                            box_center_ground(tracks[a][f])
                            - box_center_ground(tracks[b][f])
                        )
                    )
                    for f in shared
                ]
                ious = [bev_iou(tracks[a][f], tracks[b][f]) for f in shared]
                min_centroid = min(shared_dists)
                # Signal 1b: persistent proximity. Two tracks that stay close
                # over a long stretch, OR where the shorter is mostly nested
                # inside the longer (a fragment of it), are the same object
                # split by depth error. The median-distance guard keeps real
                # vehicles that merely pass each other (close once, then
                # diverge -> high median) from merging.
                containment = len(shared) / max(
                    1, min(len(frames_of[a]), len(frames_of[b]))
                )
                persistent = (
                    (
                        len(shared) >= persistent_overlap_frames
                        or containment >= containment_frac
                    )
                    and float(np.median(shared_dists)) <= persistent_merge_dist
                )
                if (
                    float(np.mean(ious)) >= iou_thr
                    or min_centroid <= merge_dist
                    or persistent
                ):
                    guarded_union(a, b)
                continue

            # Signal 2: velocity-predicted gap handoff (disjoint in time). Accept
            # if either the velocity-predicted position or the raw endpoint lands
            # within gap_dist (robust to noisy short-track velocity estimates).
            a_start, a_end = span_of[a]
            b_start, b_end = span_of[b]
            if a_end < b_start:
                end, start = a, b
                gap = b_start - a_end
            elif b_end < a_start:
                end, start = b, a
                gap = a_start - b_end
            else:
                continue
            if not (1 <= gap <= max_gap):
                continue
            e_end = max(tracks[end])
            s_start = min(tracks[start])
            vel = _estimate_ground_velocity(tracks[end])
            end_center = box_center_ground(tracks[end][e_end])
            actual = box_center_ground(tracks[start][s_start])
            pred = end_center + vel * gap
            pred_dist = float(np.linalg.norm(pred - actual))
            raw_dist = float(np.linalg.norm(end_center - actual))
            candidates = [pred_dist, raw_dist]
            # Model-assisted handoff: a bicycle (constant-turn) extrapolation of
            # the ending track predicts the reappearance point better than a
            # straight-line guess when the vehicle is cornering through the gap.
            if model_assisted_gap:
                bike_pred = _bicycle_gap_prediction(
                    tracks[end], gap, gap_model_dt, gap_model_alpha, gap_model_lr_ratio
                )
                if bike_pred is not None:
                    candidates.append(float(np.linalg.norm(bike_pred - actual)))
            if min(candidates) <= gap_dist:
                guarded_union(a, b)

    # Canonical id per group = member with the most frames.
    groups: dict[int, list[int]] = defaultdict(list)
    for tid in ids:
        groups[uf.find(tid)].append(tid)
    mapping: dict[int, int] = {}
    for members in groups.values():
        canonical = max(members, key=lambda t: len(tracks[t]))
        for tid in members:
            mapping[tid] = canonical
    return mapping


def apply_fusion(
    tracks: dict[int, dict[int, dict]],
    mapping: dict[int, int],
    pose_filter_cfg: dict | None = None,
    frame_keys: list[str] | None = None,
) -> tuple[dict[int, dict[int, dict]], dict]:
    """Merge tracks per ``mapping`` with optional pose-aware overlap filtering."""
    pose_filter_cfg = pose_filter_cfg or {}
    enabled = bool(pose_filter_cfg.get("enabled", False))
    prefer_canonical = bool(pose_filter_cfg.get("prefer_canonical", True))
    drop_if_all_outliers = bool(pose_filter_cfg.get("drop_if_all_outliers", True))
    max_center_dist_m = float(pose_filter_cfg.get("max_center_dist_m", 2.5))
    max_yaw_deg = float(pose_filter_cfg.get("max_yaw_deg", 25.0))
    max_size_rel = float(pose_filter_cfg.get("max_size_rel", 0.35))
    yaw_sensitive_classes = {
        str(c)
        for c in (pose_filter_cfg.get("yaw_sensitive_classes", []) or [
            "car", "truck", "bus", "trailer", "construction_vehicle", "motorcycle", "bicycle"
        ])
    }
    max_yaw_rad = np.deg2rad(max_yaw_deg)

    def _box_yaw(box: dict) -> float:
        rot = quat_wxyz_to_matrix(box["rotation"])
        fwd = rot[:, 0]
        return float(np.arctan2(fwd[2], fwd[0]))

    def _wrap_angle(a: float) -> float:
        return float(np.arctan2(np.sin(a), np.cos(a)))

    def _expected_from_canonical(
        canonical_track: dict[int, dict],
        canonical_frames: list[int],
        fidx: int,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        if not canonical_frames:
            return None
        if fidx in canonical_track:
            b = canonical_track[fidx]
            return (
                np.asarray(box_center_ground(b), dtype=np.float64),
                np.asarray(b.get("size", [0.0, 0.0, 0.0]), dtype=np.float64),
                _box_yaw(b),
            )

        prev = [f for f in canonical_frames if f < fidx]
        nxt = [f for f in canonical_frames if f > fidx]
        if prev and nxt:
            fp, fn = prev[-1], nxt[0]
            bp, bn = canonical_track[fp], canonical_track[fn]
            a = (fidx - fp) / float(fn - fp)
            cp = np.asarray(box_center_ground(bp), dtype=np.float64)
            cn = np.asarray(box_center_ground(bn), dtype=np.float64)
            sp = np.asarray(bp.get("size", [0.0, 0.0, 0.0]), dtype=np.float64)
            sn = np.asarray(bn.get("size", [0.0, 0.0, 0.0]), dtype=np.float64)
            yp, yn = _box_yaw(bp), _box_yaw(bn)
            yd = _wrap_angle(yn - yp)
            return ((1 - a) * cp + a * cn, (1 - a) * sp + a * sn, yp + a * yd)
        if prev:
            b = canonical_track[prev[-1]]
            return (
                np.asarray(box_center_ground(b), dtype=np.float64),
                np.asarray(b.get("size", [0.0, 0.0, 0.0]), dtype=np.float64),
                _box_yaw(b),
            )
        if nxt:
            b = canonical_track[nxt[0]]
            return (
                np.asarray(box_center_ground(b), dtype=np.float64),
                np.asarray(b.get("size", [0.0, 0.0, 0.0]), dtype=np.float64),
                _box_yaw(b),
            )
        return None

    def _passes_pose_gate(box: dict, expected, fused_name: str) -> bool:
        if not enabled or expected is None:
            return True
        exp_center, exp_size, exp_yaw = expected
        cand_center = np.asarray(box_center_ground(box), dtype=np.float64)
        if float(np.linalg.norm(cand_center - exp_center)) > max_center_dist_m:
            return False

        cand_size = np.asarray(box.get("size", [0.0, 0.0, 0.0]), dtype=np.float64)
        denom = np.maximum(np.abs(exp_size), 1e-6)
        size_rel = float(np.max(np.abs(cand_size - exp_size) / denom))
        if size_rel > max_size_rel:
            return False

        if fused_name in yaw_sensitive_classes:
            cand_yaw = _box_yaw(box)
            if abs(_wrap_angle(cand_yaw - exp_yaw)) > max_yaw_rad:
                return False
        return True

    groups: dict[int, list[int]] = defaultdict(list)
    for tid, canonical in mapping.items():
        groups[canonical].append(tid)

    merged: dict[int, dict[int, dict]] = defaultdict(dict)
    overlap_decisions: list[dict] = []
    for canonical, members in groups.items():
        canonical_track = tracks.get(canonical, {})
        canonical_frames = sorted(canonical_track)

        frames_all: set[int] = set()
        for tid in members:
            frames_all |= set(tracks[tid].keys())

        fused_name = _track_class(canonical_track) if canonical_track else _track_class(
            {f: b for tid in members for f, b in tracks[tid].items()}
        )

        for fidx in sorted(frames_all):
            candidates: list[tuple[int, dict]] = []
            for tid in members:
                b = tracks[tid].get(fidx)
                if b is not None:
                    candidates.append((tid, b))
            if not candidates:
                continue

            expected = _expected_from_canonical(canonical_track, canonical_frames, fidx)
            valid: list[tuple[int, dict]] = []
            rejected_ids: list[int] = []
            for tid, b in candidates:
                if _passes_pose_gate(b, expected, fused_name):
                    valid.append((tid, b))
                else:
                    rejected_ids.append(int(tid))

            chosen: dict | None = None
            chosen_tid: int | None = None
            if prefer_canonical:
                can_pair = next(((tid, b) for tid, b in valid if tid == canonical), None)
                can_box = can_pair[1] if can_pair is not None else None
                if can_box is not None:
                    chosen = can_box
                    chosen_tid = int(canonical)

            pool = valid if valid else ([] if drop_if_all_outliers else candidates)
            if chosen is None and pool:
                best_tid, best_box = max(
                    pool, key=lambda tb: float(tb[1].get("tracking_score", 1.0))
                )
                chosen = best_box
                chosen_tid = int(best_tid)

            if chosen is not None:
                merged[canonical][fidx] = chosen

            if len(candidates) > 1 or rejected_ids:
                entry = {
                    "canonical_id": int(canonical),
                    "frame_index": int(fidx),
                    "token": frame_keys[fidx] if frame_keys and 0 <= fidx < len(frame_keys) else None,
                    "candidate_ids": [int(tid) for tid, _ in candidates],
                    "rejected_ids": rejected_ids,
                    "selected_id": chosen_tid,
                    "dropped_frame": chosen is None,
                }
                overlap_decisions.append(entry)

    # Unify id + class label across each merged track.
    for canonical, track in merged.items():
        name = _track_class(track)
        for box in track.values():
            box["tracking_id"] = int(canonical)
            box["tracking_name"] = name

    fusion_groups = [
        {
            "canonical_id": int(canonical),
            "members": sorted(int(t) for t in members),
        }
        for canonical, members in groups.items()
    ]
    fusion_groups.sort(key=lambda g: g["canonical_id"])

    fusion_report = {
        "groups": fusion_groups,
        "merged_groups": [g for g in fusion_groups if len(g["members"]) > 1],
        "overlap_decisions": overlap_decisions,
    }
    return merged, fusion_report


# --------------------------------------------------------------------------- #
# Static filtering
# --------------------------------------------------------------------------- #
def track_displacement(
    track: dict[int, dict],
    percentile: float,
    mode: str = "net",
    smooth: int = 3,
) -> float:
    """Ground displacement of a track.

    Boxes are in the static COLMAP world, so a parked object stays put while a
    moving one translates. Two modes:

    * ``net`` (default): net travel = distance between the smoothed start and
      end positions (mean of the first/last ``smooth`` centroids). Robust to
      per-frame jitter — a parked car whose box wobbles still reads ~0.
    * ``span``: legacy point-cloud diameter ``2 * percentile(|c - median|)``,
      which is inflated by jitter and tends to keep wobbling parked cars.
    """
    centers = np.array([box_center_ground(b) for b in track.values()])
    if len(centers) < 2:
        return 0.0
    if mode == "span":
        median = np.median(centers, axis=0)
        dists = np.linalg.norm(centers - median, axis=1)
        return float(2.0 * np.percentile(dists, percentile))
    k = max(1, min(int(smooth), len(centers) // 2))
    start = centers[:k].mean(axis=0)
    end = centers[-k:].mean(axis=0)
    return float(np.linalg.norm(end - start))


def filter_static(
    tracks: dict[int, dict[int, dict]],
    min_displacement: float,
    min_track_length: int,
    displacement_percentile: float,
    displacement_mode: str = "net",
    displacement_smooth: int = 3,
    min_rel_displacement: float = 0.0,
) -> tuple[dict[int, dict[int, dict]], list[dict]]:
    """Drop static / too-short tracks. Returns (kept, dropped_report).

    A track is kept only if it both travels >= ``min_displacement`` meters and,
    when ``min_rel_displacement`` > 0, travels >= that fraction of its distance
    from the origin (ego start). The relative gate removes far objects whose
    apparent motion is depth-estimation jitter (which scales with range): a
    static object 200 m away can "move" ~10 m radially yet only ~5% of its
    range, while real movers exceed ~45%.
    """
    kept: dict[int, dict[int, dict]] = {}
    dropped: list[dict] = []
    for tid, track in tracks.items():
        disp = track_displacement(
            track, displacement_percentile, displacement_mode, displacement_smooth
        )
        centers = np.array([box_center_ground(b) for b in track.values()])
        rng = float(np.linalg.norm(centers.mean(axis=0))) if len(centers) else 0.0
        rel = disp / rng if rng > 1e-6 else 0.0
        if len(track) < min_track_length:
            dropped.append({"id": tid, "reason": "short", "frames": len(track), "disp": disp})
        elif disp < min_displacement or rel < min_rel_displacement:
            dropped.append(
                {"id": tid, "reason": "static", "frames": len(track),
                 "disp": disp, "rel": round(rel, 3)}
            )
        else:
            kept[tid] = track
    return kept, dropped


# --------------------------------------------------------------------------- #
# Gap filling (interpolate missing frames inside each track's lifespan)
# --------------------------------------------------------------------------- #
def _slerp_wxyz(q0, q1, alpha: float) -> np.ndarray:
    """Spherical linear interpolation between two ``[w, x, y, z]`` quaternions."""
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    dot = float(np.dot(a, b))
    if dot < 0.0:  # take the shortest arc (q and -q are the same rotation)
        b = -b
        dot = -dot
    if dot > 0.9995:  # nearly parallel -> normalized lerp avoids div-by-zero
        q = (1.0 - alpha) * a + alpha * b
        return q / (np.linalg.norm(q) + 1e-12)
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta0 * alpha
    perp = b - a * dot
    perp = perp / (np.linalg.norm(perp) + 1e-12)
    return a * np.cos(theta) + perp * np.sin(theta)


def _interp_box(box_a: dict, box_b: dict, alpha: float) -> dict:
    """Interpolate a box between two keyframes (lerp position/size, slerp rot)."""
    import copy

    nb = copy.deepcopy(box_a)
    ta = np.asarray(box_a["translation"], dtype=np.float64)
    tb = np.asarray(box_b["translation"], dtype=np.float64)
    nb["translation"] = ((1.0 - alpha) * ta + alpha * tb).tolist()
    if "size" in box_a and "size" in box_b:
        sa = np.asarray(box_a["size"], dtype=np.float64)
        sb = np.asarray(box_b["size"], dtype=np.float64)
        nb["size"] = ((1.0 - alpha) * sa + alpha * sb).tolist()
    nb["rotation"] = _slerp_wxyz(box_a["rotation"], box_b["rotation"], alpha).tolist()
    if box_a.get("velocity") is not None and box_b.get("velocity") is not None:
        va = np.asarray(box_a["velocity"], dtype=np.float64)
        vb = np.asarray(box_b["velocity"], dtype=np.float64)
        nb["velocity"] = ((1.0 - alpha) * va + alpha * vb).tolist()
    sca = float(box_a.get("tracking_score", 1.0))
    scb = float(box_b.get("tracking_score", 1.0))
    nb["tracking_score"] = (1.0 - alpha) * sca + alpha * scb
    nb["interpolated"] = True  # mark synthetic boxes for debugging/inspection
    return nb


def fill_track_gaps(
    tracks: dict[int, dict[int, dict]], frame_keys: list[str]
) -> tuple[int, list[dict]]:
    """Fill interior frame gaps of each track by interpolation, in place.

    For every track, any frame missing between its first and last observed
    frame is synthesized by interpolating the two surrounding keyframes
    (linear on translation/size, SLERP on rotation). This makes each track
    gap-free within its lifespan so the 4DGS rigid node is rendered continuously
    (the tracker often drops individual frames). Returns the number of frames
    filled. Object identity (``tracking_id``/``tracking_name``) and per-instance
    size are preserved; frames outside ``[first, last]`` are untouched.
    """
    n_filled = 0
    details: list[dict] = []
    n_keys = len(frame_keys)
    for track in tracks.values():
        frames = sorted(track)
        if len(frames) < 2:
            continue
        for a, b in zip(frames[:-1], frames[1:]):
            if b - a <= 1:
                continue
            box_a, box_b = track[a], track[b]
            for f in range(a + 1, b):
                new_box = _interp_box(box_a, box_b, (f - a) / (b - a))
                if 0 <= f < n_keys and "sample_token" in new_box:
                    new_box["sample_token"] = frame_keys[f]
                track[f] = new_box
                n_filled += 1
                details.append(
                    {
                        "tracking_id": int(new_box.get("tracking_id", -1)),
                        "frame_index": int(f),
                        "token": frame_keys[f] if 0 <= f < n_keys else None,
                        "between": [int(a), int(b)],
                    }
                )
    return n_filled, details


# --------------------------------------------------------------------------- #
# End extension (extrapolate a bounded number of frames past each track's ends)
# --------------------------------------------------------------------------- #
def _endpoint_velocity_3d(
    track,
    frames,
    at_start: bool,
    window: int,
) -> np.ndarray:
    """3D translation velocity (per frame) at a track endpoint.

    Vertical (up-axis) component is zeroed so extrapolation stays on the ground
    plane and vehicles don't drift up/down from tracker jitter.
    """
    if len(frames) < 2:
        return np.zeros(3)
    if at_start:
        i0, i1 = 0, min(window, len(frames) - 1)
    else:
        i1, i0 = len(frames) - 1, max(0, len(frames) - 1 - window)
    f0, f1 = frames[i0], frames[i1]
    span = f1 - f0
    if span <= 0:
        return np.zeros(3)
    t0 = np.asarray(track[f0]["translation"], dtype=np.float64)
    t1 = np.asarray(track[f1]["translation"], dtype=np.float64)
    vel = (t1 - t0) / float(span)
    vel[UP_AXIS] = 0.0
    return vel


def _extrap_box(base_box: dict, new_trans: np.ndarray, token: str | None) -> dict:
    """Copy a keyframe box at a new translation (rotation/size held constant)."""
    import copy

    nb = copy.deepcopy(base_box)
    nb["translation"] = np.asarray(new_trans, dtype=np.float64).tolist()
    nb["interpolated"] = True
    nb["extrapolated"] = True  # synthesized beyond the observed track span
    if token is not None and "sample_token" in nb:
        nb["sample_token"] = token
    return nb


def extend_track_ends(
    tracks: dict[int, dict[int, dict]],
    frame_keys: list[str],
    extend_frames: int,
    vel_window: int = 3,
) -> tuple[int, list[dict]]:
    """Extrapolate up to ``extend_frames`` frames before/after each track, in place.

    The tracker runs on undistorted crops with a narrower FOV than the fisheye
    cameras, so objects entering/leaving frame (or getting far) stop being
    detected while still visible in the training images. Extend each track with
    constant ground-plane velocity (rotation and size held), bounded by
    ``extend_frames`` and the clip range. Returns the number of frames added.
    """
    if extend_frames <= 0:
        return 0, []
    n_keys = len(frame_keys)
    n_ext = 0
    details: list[dict] = []
    for track in tracks.values():
        frames = sorted(track)
        if len(frames) < 2:
            continue
        first, last = frames[0], frames[-1]

        # Forward from the last observed frame.
        v_end = _endpoint_velocity_3d(
            track,
            frames,
            at_start=False,
            window=vel_window,
        )
        base_box = track[last]
        base_t = np.asarray(base_box["translation"], dtype=np.float64)
        for k in range(1, extend_frames + 1):
            f = last + k
            if f >= n_keys or f in track:
                break
            token = frame_keys[f] if 0 <= f < n_keys else None
            track[f] = _extrap_box(base_box, base_t + v_end * k, token)
            n_ext += 1
            details.append(
                {
                    "tracking_id": int(track[f].get("tracking_id", -1)),
                    "frame_index": int(f),
                    "token": token,
                    "direction": "after",
                }
            )

        # Backward from the first observed frame.
        v_start = _endpoint_velocity_3d(
            track,
            frames,
            at_start=True,
            window=vel_window,
        )
        base_box = track[first]
        base_t = np.asarray(base_box["translation"], dtype=np.float64)
        for k in range(1, extend_frames + 1):
            f = first - k
            if f < 0 or f in track:
                break
            token = frame_keys[f] if 0 <= f < n_keys else None
            track[f] = _extrap_box(base_box, base_t - v_start * k, token)
            n_ext += 1
            details.append(
                {
                    "tracking_id": int(track[f].get("tracking_id", -1)),
                    "frame_index": int(f),
                    "token": token,
                    "direction": "before",
                }
            )
    return n_ext, details


# --------------------------------------------------------------------------- #
# Per-instance height plane fitting (RANSAC, pre-training step)
# --------------------------------------------------------------------------- #
def fit_instance_heights_ransac(
    tracks: dict[int, dict[int, dict]],
    outlier_threshold: float = 0.5,
    ransac_iters: int = 100,
    min_inliers: int = 3,
    rng_seed: int = 42,
) -> dict:
    """Fit a per-instance ground plane Y = a·X + b·Z + c to each track.

    For every kept track, collects its valid per-frame box centers in COLMAP
    world space (X, Y, Z where Y=UP_AXIS=1 is height), then uses RANSAC to
    robustly fit a plane Y = a·X + b·Z + c, ignoring frames where the tracker
    produced a wildly wrong height estimate.

    After fitting, replaces each frame's box ``translation[1]`` (Y) with the
    plane-predicted value ``a*X + b*Z + c``, effectively removing per-frame
    height jitter while preserving the vehicle's true elevation and any gentle
    slope along its path.

    Falls back to a median-based constant (a=b=0, c=median(Y)) when the track
    has fewer frames than ``min_inliers``.

    Args:
        tracks: ``{track_id: {frame_idx: box}}`` — mutated in-place.
        outlier_threshold: Maximum |residual| (meters) for a frame to count as
            an inlier during RANSAC.
        ransac_iters: Number of random 3-point trials per instance.
        min_inliers: Minimum valid frames required to attempt RANSAC; shorter
            tracks fall back to the median.
        rng_seed: Seed for reproducible RANSAC sampling.

    Returns:
        Report dict with per-track plane coefficients and inlier counts.
    """
    rng = np.random.default_rng(rng_seed)
    report: dict[str, dict] = {}

    for tid, track in tracks.items():
        frames = sorted(track.keys())
        if not frames:
            continue

        # Collect (X, Y, Z) for each frame's box center.
        pts = np.array(
            [np.asarray(track[f]["translation"], dtype=np.float64) for f in frames]
        )  # (N, 3)
        X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
        N = len(pts)

        if N < min_inliers:
            # Fallback: constant height at median Y.
            c = float(np.median(Y))
            a, b = 0.0, 0.0
            best_inliers = N
        else:
            # RANSAC: repeatedly sample 3 points, fit a plane, count inliers.
            best_inliers = 0
            best_abc = (0.0, 0.0, float(np.median(Y)))

            # Design matrix [X, Z, 1] for least-squares fit Y = [X,Z,1] @ [a,b,c].
            A_full = np.column_stack([X, Z, np.ones(N)])

            for _ in range(ransac_iters):
                sample = rng.choice(N, size=3, replace=False)
                A_s = A_full[sample]
                Y_s = Y[sample]
                # Solve the 3×3 system exactly (3 points determine a plane).
                try:
                    abc, _, _, _ = np.linalg.lstsq(A_s, Y_s, rcond=None)
                except np.linalg.LinAlgError:
                    continue

                residuals = np.abs(A_full @ abc - Y)
                n_inliers = int((residuals < outlier_threshold).sum())

                if n_inliers > best_inliers:
                    best_inliers = n_inliers
                    # Re-fit on the full inlier set for a more stable estimate.
                    inlier_mask = residuals < outlier_threshold
                    A_in = A_full[inlier_mask]
                    Y_in = Y[inlier_mask]
                    abc_refined, _, _, _ = np.linalg.lstsq(A_in, Y_in, rcond=None)
                    best_abc = tuple(float(v) for v in abc_refined)

            a, b, c = best_abc

        # Apply: replace each frame's Y with the plane prediction.
        for i, f in enumerate(frames):
            y_fitted = a * X[i] + b * Z[i] + c
            t = list(track[f]["translation"])
            t[UP_AXIS] = float(y_fitted)
            track[f]["translation"] = t

        report[str(tid)] = {
            "plane": {"a": round(a, 6), "b": round(b, 6), "c": round(c, 6)},
            "n_frames": N,
            "n_inliers": best_inliers,
            "fallback_median": N < min_inliers,
        }

    return report


# --------------------------------------------------------------------------- #
# Translation smoothing (denoise per-frame tracker jitter)
# --------------------------------------------------------------------------- #
def smooth_track_translations(
    tracks: dict[int, dict[int, dict]], window: int
) -> int:
    """Low-pass each track's per-frame translation with a centered moving average.

    The tracker's per-frame box centers jitter (depth/position noise), which
    shows up as shaky rigid-object motion. A centered moving average over
    ``window`` frames denoises the translation while preserving overall motion;
    the window shrinks at track ends so endpoints aren't dragged. Rotation and
    size are left untouched (rotation smoothing is intentionally not applied).
    Operates in place; returns the number of frames modified.
    """
    if window <= 1:
        return 0
    half = window // 2
    n_smoothed = 0
    for track in tracks.values():
        frames = sorted(track)
        if len(frames) < 3:
            continue
        trans = np.asarray(
            [track[f]["translation"] for f in frames], dtype=np.float64
        )  # (T, 3)
        smoothed = np.empty_like(trans)
        T = trans.shape[0]
        for i in range(T):
            lo = max(0, i - half)
            hi = min(T, i + half + 1)
            smoothed[i] = trans[lo:hi].mean(axis=0)
        for i, f in enumerate(frames):
            track[f]["translation"] = smoothed[i].tolist()
            n_smoothed += 1
    return n_smoothed


def _yaw_from_box_rotation(box: dict) -> float:
    """Extract ground-plane yaw from a box quaternion via its local +x axis."""
    rot = quat_wxyz_to_matrix(box["rotation"])
    forward = rot[:, 0]
    heading = np.asarray([forward[0], forward[2]], dtype=np.float64)
    norm = float(np.linalg.norm(heading))
    if norm <= 1e-12:
        return 0.0
    heading /= norm
    return float(np.arctan2(heading[1], heading[0]))


def _yaw_matrix(yaw: float) -> np.ndarray:
    """Ground-plane yaw rotation matrix consistent with :func:`_yaw_from_box_rotation`.

    The pipeline measures heading as ``atan2(dz, dx)`` of a box's local +x axis
    (see :func:`_yaw_from_box_rotation`), so a box pointing at heading ``yaw`` has
    local +x equal to ``[cos yaw, 0, sin yaw]``. That corresponds to a rotation of
    ``-yaw`` about the world +y axis (a positive ``atan2(dz, dx)`` yaw turns +x
    toward +z, opposite to the right-handed ``R_y(+yaw)``). Building the matrix
    this way makes the round trip exact:
    ``_yaw_from_box_rotation(_yaw_matrix(yaw)) == yaw``. Callers therefore apply a
    delta ``target_yaw - base_yaw`` and get back a box at ``target_yaw``.
    """
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.array(
        [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=np.float64
    )


def _matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    m = np.asarray(rot, dtype=np.float64)
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def _smooth_scalar_series(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average over a 1D series, shrinking at the ends."""
    if window <= 1 or values.size == 0:
        return values.copy()
    half = window // 2
    out = np.empty_like(values, dtype=np.float64)
    for i in range(values.shape[0]):
        lo = max(0, i - half)
        hi = min(values.shape[0], i + half + 1)
        out[i] = values[lo:hi].mean()
    return out


def _tangent_yaws_from_centers(
    centers: np.ndarray,
    fallback_yaws: np.ndarray,
    min_speed: float,
) -> np.ndarray:
    """Estimate heading from the local translation tangent, with yaw fallback."""
    T = centers.shape[0]
    yaws = np.empty(T, dtype=np.float64)
    for i in range(T):
        lo = max(0, i - 1)
        hi = min(T - 1, i + 1)
        if hi == lo:
            yaws[i] = fallback_yaws[i]
            continue
        delta = centers[hi] - centers[lo]
        speed = float(np.linalg.norm(delta)) / float(hi - lo)
        if speed < min_speed:
            yaws[i] = fallback_yaws[i]
            continue
        yaws[i] = float(np.arctan2(delta[1], delta[0]))
    return yaws


def smooth_track_rotations(
    tracks: dict[int, dict[int, dict]],
    window: int,
    mode: str,
    min_speed: float = 0.05,
) -> int:
    """Stabilize box yaw while preserving curved trajectories.

    Modes:
    * ``box_yaw``: smooth the raw per-frame box yaw sequence.
    * ``tangent_yaw``: derive yaw from the local tangent of the already-smoothed
      ground-plane trajectory, falling back to raw box yaw when motion is too
      small/noisy.

    We preserve each box's non-yaw orientation basis and only adjust heading
    around the world up axis. This avoids breaking the dataset's box convention
    (e.g. making upright pedestrians appear horizontal).
    """
    if window <= 0:
        return 0
    mode = str(mode).strip().lower()
    if mode not in {"box_yaw", "tangent_yaw"}:
        raise ValueError(
            "pose_smooth_rotation_mode must be 'box_yaw' or 'tangent_yaw'."
        )

    n_smoothed = 0
    for track in tracks.values():
        frames = sorted(track)
        if len(frames) < 2:
            continue

        raw_rots = np.asarray(
            [quat_wxyz_to_matrix(track[f]["rotation"]) for f in frames], dtype=np.float64
        )
        raw_yaws = np.asarray([_yaw_from_box_rotation(track[f]) for f in frames], dtype=np.float64)
        raw_yaws = np.unwrap(raw_yaws)

        if mode == "box_yaw":
            base_yaws = raw_yaws
        else:
            centers = np.asarray(
                [box_center_ground(track[f]) for f in frames], dtype=np.float64
            )
            base_yaws = _tangent_yaws_from_centers(centers, raw_yaws, min_speed)
            base_yaws = np.unwrap(base_yaws)

        smoothed_yaws = _smooth_scalar_series(base_yaws, window)
        for i, f in enumerate(frames):
            # Apply only a world-up yaw delta on top of the raw box rotation,
            # so roll/pitch (and axis conventions) are preserved.
            delta = float(smoothed_yaws[i] - raw_yaws[i])
            rot_new = _yaw_matrix(delta) @ raw_rots[i]
            track[f]["rotation"] = _matrix_to_quat_wxyz(rot_new).tolist()
            n_smoothed += 1
    return n_smoothed


# --------------------------------------------------------------------------- #
# Kinematic bicycle-model fitting (vehicle classes)
# --------------------------------------------------------------------------- #
DEFAULT_BICYCLE_CLASSES = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
]


def _bicycle_fit_config(cfg: dict) -> BicycleFitConfig:
    """Build a :class:`BicycleFitConfig` from the refine-task ``bicycle_fit`` block."""
    defaults = BicycleFitConfig()
    return BicycleFitConfig(
        lr_ratio=float(cfg.get("lr_ratio", defaults.lr_ratio)),
        segment_len=int(cfg.get("segment_len", defaults.segment_len)),
        pos_scale=float(cfg.get("pos_scale", defaults.pos_scale)),
        yaw_scale=float(cfg.get("yaw_scale", defaults.yaw_scale)),
        huber_delta=float(cfg.get("huber_delta", defaults.huber_delta)),
        cauchy_c=float(cfg.get("cauchy_c", defaults.cauchy_c)),
        irls_iters=int(cfg.get("irls_iters", defaults.irls_iters)),
        lambda_accel=float(cfg.get("lambda_accel", defaults.lambda_accel)),
        lambda_steer=float(cfg.get("lambda_steer", defaults.lambda_steer)),
        lambda_accel_mag=float(cfg.get("lambda_accel_mag", defaults.lambda_accel_mag)),
        lambda_steer_mag=float(cfg.get("lambda_steer_mag", defaults.lambda_steer_mag)),
        defect_weight=float(cfg.get("defect_weight", defaults.defect_weight)),
        min_heading_speed=float(cfg.get("min_heading_speed", defaults.min_heading_speed)),
        max_steer=float(cfg.get("max_steer", defaults.max_steer)),
    )


def select_bicycle_track_ids(
    tracks: dict[int, dict[int, dict]], cfg: dict
) -> list[int]:
    """Track ids eligible for bicycle fitting (vehicle class + long enough)."""
    classes = set(cfg.get("classes", DEFAULT_BICYCLE_CLASSES))
    min_frames = int(cfg.get("min_track_frames", 3))
    ids = []
    for tid, track in tracks.items():
        if len(track) < max(min_frames, 2):
            continue
        if _track_class(track) in classes:
            ids.append(tid)
    return ids


def _bicycle_fit_worker(task: tuple) -> tuple:
    """Process-pool worker: fit one track and return ``(tid, result)``.

    Module-level (picklable) so :func:`apply_bicycle_fit` can dispatch the
    independent per-track fits across a :class:`ProcessPoolExecutor`.
    """
    tid, offsets, positions, yaws, n_steps, dt, wheelbase, cfg = task
    res = fit_track(
        offsets, positions, yaws, n_steps=n_steps, dt=dt,
        wheelbase=wheelbase, cfg=cfg,
    )
    return tid, res


def _preclean_fit_inputs(
    offsets: np.ndarray,
    positions: np.ndarray,
    yaws: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Remove the two size-scaled detection artefacts before bicycle fitting.

    Large / elongated vehicles suffer two failures the kinematic model cannot
    represent and that blow up the fit:

    * 180-degree heading flips (front/back box ambiguity) — resolved by pointing
      each heading along the direction of motion (local where the vehicle moves,
      the net start->end direction where it is momentarily too slow to trust the
      local tangent). Heading is ``atan2(dz, dx)`` and ground positions are
      ``(x, z)``, so motion direction is directly comparable to the box yaw.
    * lateral position spikes (single-frame bad boxes) — dropped when a point
      deviates from the offset-interpolation of its temporal neighbours by more
      than ``preclean_outlier_dist`` meters.

    Returns cleaned ``(offsets, positions, yaws, n_dropped)``. The first and last
    observations and a minimum of two points are always preserved (they anchor
    the fit span).
    """
    offsets = np.asarray(offsets, dtype=np.int64).copy()
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2).copy()
    yaws = np.asarray(yaws, dtype=np.float64).reshape(-1).copy()
    n = offsets.size
    if n < 3:
        return offsets, positions, yaws, 0

    def _wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    # --- 1. heading unflip -------------------------------------------------- #
    if bool(cfg.get("preclean_yaw_flip", True)):
        min_move = float(cfg.get("preclean_min_move", 0.3))
        d = np.zeros_like(positions)
        d[1:-1] = positions[2:] - positions[:-2]
        d[0] = positions[1] - positions[0]
        d[-1] = positions[-1] - positions[-2]
        mag = np.linalg.norm(d, axis=1)
        local_dir = np.arctan2(d[:, 1], d[:, 0])
        net = positions[-1] - positions[0]
        global_dir = float(np.arctan2(net[1], net[0]))
        for i in range(n):
            ref = local_dir[i] if mag[i] >= min_move else global_dir
            if abs(_wrap(yaws[i] - ref)) > (np.pi / 2.0):
                yaws[i] = _wrap(yaws[i] + np.pi)

    # --- 2. position spike rejection --------------------------------------- #
    outlier_dist = float(cfg.get("preclean_outlier_dist", 2.0))
    n_dropped = 0
    if outlier_dist > 0.0:
        dev = np.zeros(n)
        for i in range(1, n - 1):
            span = offsets[i + 1] - offsets[i - 1]
            t = (offsets[i] - offsets[i - 1]) / span if span > 0 else 0.5
            pred = positions[i - 1] * (1.0 - t) + positions[i + 1] * t
            dev[i] = float(np.linalg.norm(positions[i] - pred))
        flagged = dev > outlier_dist  # interior only; endpoints (i=0,n-1) kept
        # Cap removal so a genuinely wiggly track is not gutted: keep at most the
        # worst 40% flagged, and never drop below two observations.
        max_drop = int(np.floor(0.4 * n))
        if int(flagged.sum()) > max_drop and max_drop >= 0:
            worst = np.argsort(dev)[::-1][:max_drop]
            keep_flag = np.zeros(n, dtype=bool)
            keep_flag[worst] = True
            flagged = flagged & keep_flag
        if flagged.any() and (n - int(flagged.sum())) >= 2:
            keep = ~flagged
            offsets = offsets[keep]
            positions = positions[keep]
            yaws = yaws[keep]
            n_dropped = int(flagged.sum())

    return offsets, positions, yaws, n_dropped


def _fit_with_trimming(
    offsets: np.ndarray,
    positions: np.ndarray,
    yaws: np.ndarray,
    n_steps: int,
    dt: float,
    wheelbase: float,
    fit_cfg: "BicycleFitConfig",
    res0,
    max_pos_rmse: float,
    cfg: dict,
) -> tuple:
    """Least-trimmed-squares recovery for a fit that failed the consistency gate.

    Iteratively drops the worst single-shoot-residual observations and refits so
    the continuous rollout can track the retained inliers. Scattered single-frame
    outliers (the dominant large-vehicle failure) are removed one small batch at
    a time; the span ``n_steps`` is held fixed so the baked trajectory still
    covers every original frame (the model fills the dropped ones).

    Returns ``(res, traj, ss_rmse, offsets, positions)`` for the best attempt by
    RMSE (which may still exceed the gate, leaving the caller to reject).
    """
    lr = fit_cfg.lr_ratio

    # Bound each refit: with the finite-difference Jacobian an uncapped solve is
    # ``100 * n_params`` function evals, which — on a large, now under-constrained
    # (trimmed) track — grinds for minutes in the trust-region solver. A finite
    # cap and fewer IRLS passes keep every refit fast; a partial solve is fine
    # for a fallback attempt.
    max_nfev = int(cfg.get("trim_refit_max_nfev", 400))
    refit_cfg = _dc_replace(
        fit_cfg,
        solver_max_nfev=max_nfev,
        irls_iters=min(fit_cfg.irls_iters, 2),
    )

    def _eval(res, offs, pos):
        traj = _bicycle_rollout(res.states[0], res.accel, res.steer, dt, wheelbase, lr)
        resid = np.linalg.norm(traj[offs, :2] - pos, axis=1)
        rmse = float(np.sqrt(np.mean(resid**2))) if resid.size else float("inf")
        return traj, resid, rmse

    cur_off, cur_pos, cur_yaw = offsets, positions, yaws
    res = res0
    traj, resid, rmse = _eval(res, cur_off, cur_pos)
    best = (res, traj, rmse, cur_off, cur_pos)

    iters = int(cfg.get("trim_refit_iters", 3))
    thr = float(cfg.get("trim_refit_resid", 0.0)) or max_pos_rmse
    min_obs = 4
    for _ in range(max(iters, 0)):
        if rmse <= max_pos_rmse:
            break
        n_cur = cur_off.size
        if n_cur <= min_obs:
            break
        over = int((resid > thr).sum())
        if over == 0:
            over = 1  # RMSE high but no single frame over threshold: drop worst
        # Drop the worst residuals, at most ~1/3 of the points per iteration and
        # never below ``min_obs``.
        budget = min(over, max(n_cur - min_obs, 0), max(int(np.ceil(n_cur / 3)), 1))
        if budget <= 0:
            break
        drop_idx = np.argsort(resid)[::-1][:budget]
        keep = np.ones(n_cur, dtype=bool)
        keep[drop_idx] = False
        if int(keep.sum()) < 2:
            break
        cur_off, cur_pos, cur_yaw = cur_off[keep], cur_pos[keep], cur_yaw[keep]
        res = fit_track(
            cur_off, cur_pos, cur_yaw, n_steps=n_steps, dt=dt,
            wheelbase=wheelbase, cfg=refit_cfg,
        )
        traj, resid, rmse = _eval(res, cur_off, cur_pos)
        if rmse < best[2]:
            best = (res, traj, rmse, cur_off, cur_pos)

    return best


def apply_bicycle_fit(
    tracks: dict[int, dict[int, dict]],
    frame_keys: list[str],
    cfg: dict,
) -> tuple[int, dict]:
    """Fit a kinematic bicycle model to each vehicle track and write it back.

    For every track passed in (already restricted to vehicle classes by
    :func:`select_bicycle_track_ids`), fits a CoG bicycle model to the observed
    ground positions ``(x, z)`` and headings, then overwrites the track's dense
    per-frame poses with the model rollout. This subsumes gap-filling and
    translation/rotation smoothing for vehicles: the result is gap-free,
    denoised, and kinematically consistent.

    * Ground position ``(x, z)`` comes from the fitted state; the up axis ``y``
      (height) is kept from the tracker predictions, linearly interpolated
      across gap frames (RANSAC height-plane fitting is handled separately and
      currently off).
    * Heading is written as a world-up yaw delta on top of the nearest measured
      box rotation (preserving roll/pitch); frames whose fitted speed is below
      ``min_heading_speed`` keep the measured heading.

    Mutates ``tracks`` in place (adds synthesized gap boxes). Returns
    ``(n_tracks_fitted, report)`` where ``report['params']`` holds the fitted
    model parameters per track for downstream persistence, and
    ``report['rejected_ids']`` lists tracks whose fitted rollout failed the
    consistency gate (``max_fit_pos_rmse``) — those keep their raw boxes so the
    caller can route them to the legacy fill/smooth path.

    The baked trajectory is a single kinematic rollout from the fitted
    ``(state0, accel, steer)`` (not the multiple-shooting per-segment states),
    so the training boxes are free of segment-boundary handoff jumps and are
    identical to what the persisted params reproduce during extrapolation.
    """
    fit_cfg = _bicycle_fit_config(cfg)
    alpha = float(cfg.get("wheelbase_alpha", 0.6))
    dt = float(cfg.get("dt", 0.1))
    # Max RMSE (meters) between the single kinematic rollout and the observed
    # positions for a fit to be accepted. Unconverged / noise-blown-up fits
    # (common on long or large-vehicle tracks) exceed this and are rejected so
    # the track falls back to the legacy fill/smooth path instead of emitting a
    # broken trajectory.
    max_pos_rmse = float(cfg.get("max_fit_pos_rmse", 3.0))
    n_keys = len(frame_keys)
    ax0, ax1 = GROUND_AXES

    n_fitted = 0
    n_filled = 0
    params: dict[str, dict] = {}
    per_track: dict[str, dict] = {}
    rejected: list[int] = []

    # Only tracks with >= 2 observed frames are actually fittable.
    fit_items = [(tid, track) for tid, track in tracks.items() if len(track) >= 2]

    # --- Phase 1: gather per-track fit inputs (cheap, serial) -------------- #
    tasks: list[tuple] = []
    prep_by_tid: dict[int, dict] = {}
    n_precleaned = 0
    for tid, track in fit_items:
        frames = sorted(track)
        first, last = frames[0], frames[-1]
        n_steps = last - first
        offsets = np.asarray([f - first for f in frames], dtype=np.int64)
        positions = np.asarray(
            [
                [track[f]["translation"][ax0], track[f]["translation"][ax1]]
                for f in frames
            ],
            dtype=np.float64,
        )
        yaws = np.asarray(
            [_yaw_from_box_rotation(track[f]) for f in frames], dtype=np.float64
        )
        sizes = np.asarray([track[f]["size"] for f in frames], dtype=np.float64)
        wheelbase = wheelbase_from_size(np.median(sizes, axis=0), alpha)

        # Pre-clean size-scaled detection artefacts (180-degree heading flips and
        # single-frame position spikes) so the kinematic fit is not blown up by
        # them. The cleaned arrays are reused verbatim by the gate/bake phase.
        c_off, c_pos, c_yaw, n_drop = _preclean_fit_inputs(offsets, positions, yaws, cfg)
        n_precleaned += n_drop
        prep_by_tid[tid] = {
            "offsets": c_off,
            "positions": c_pos,
            "yaws": c_yaw,
            "n_steps": n_steps,
            "wheelbase": wheelbase,
            "first": first,
            "frames": frames,
        }
        tasks.append((tid, c_off, c_pos, c_yaw, n_steps, dt, wheelbase, fit_cfg))

    # --- Phase 2: fit (parallel across processes; serial for tiny batches) - #
    # ``fit_track`` is independent per track, so the solves parallelize cleanly.
    # fit_workers: 0 = auto (cpu-1, capped to #tasks); 1 = serial; N = N workers.
    cfg_workers = int(cfg.get("fit_workers", 0))
    if cfg_workers <= 0:
        workers = min(len(tasks), max((os.cpu_count() or 1) - 1, 1))
    else:
        workers = min(cfg_workers, len(tasks)) if tasks else 0

    results_by_tid: dict = {}
    use_pool = workers > 1 and len(tasks) > 1

    if use_pool:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_bicycle_fit_worker, t) for t in tasks]
            done_iter = as_completed(futures)
            if _tqdm is not None:
                done_iter = _tqdm(
                    done_iter, total=len(futures),
                    desc=f"🚗 Fitting bicycle model ({workers}w)",
                    unit="track", dynamic_ncols=True,
                )
            for fut in done_iter:
                tid, res = fut.result()
                results_by_tid[tid] = res
    else:
        serial_iter = tasks
        if _tqdm is not None and tasks:
            serial_iter = _tqdm(
                tasks, desc="🚗 Fitting bicycle model",
                unit="track", dynamic_ncols=True,
            )
        for t in serial_iter:
            tid, res = _bicycle_fit_worker(t)
            results_by_tid[tid] = res

    # --- Phase 3: consistency gate + bake into the track boxes (serial) --- #
    n_trimmed_tracks = 0
    trim_enabled = bool(cfg.get("trim_refit", True))
    for tid, track in fit_items:
        res = results_by_tid[tid]
        prep = prep_by_tid[tid]
        offsets = prep["offsets"]
        positions = prep["positions"]
        yaws = prep["yaws"]
        n_steps = prep["n_steps"]
        wheelbase = prep["wheelbase"]
        first = prep["first"]
        frames = prep["frames"]

        # Bake the SINGLE continuous rollout from the fitted (state0, controls)
        # rather than the multiple-shooting per-segment states. Multiple shooting
        # is used only to stabilize the optimization; its per-segment states
        # contain continuity-defect jumps at each segment boundary and, more
        # importantly, disagree with what the persisted (state0, accel, steer)
        # params reproduce. Baking the rollout removes the boundary "handoff"
        # jumps and makes the training boxes identical to the simulator's
        # on-demand extrapolation.
        traj = _bicycle_rollout(
            res.states[0], res.accel, res.steer, dt, wheelbase, fit_cfg.lr_ratio
        )

        # Consistency / sanity gate: the single rollout must track the (cleaned)
        # observations within max_pos_rmse. If not, attempt a least-trimmed-
        # squares recovery — drop the worst-residual observations and refit —
        # before giving up. Rejected tracks fall back to the legacy path via the
        # caller.
        ss_err = traj[offsets, :2] - positions
        ss_rmse = float(np.sqrt(np.mean(np.sum(ss_err**2, axis=1))))
        if (not np.isfinite(ss_rmse) or ss_rmse > max_pos_rmse) and trim_enabled:
            n_before = offsets.size
            res, traj, ss_rmse, offsets, positions = _fit_with_trimming(
                offsets, positions, yaws, n_steps, dt, wheelbase, fit_cfg,
                res, max_pos_rmse, cfg,
            )
            if offsets.size < n_before:
                n_trimmed_tracks += 1

        traj_pos = traj[:, :2]
        traj_yaw = traj[:, STATE_THETA]
        traj_valid = traj[:, STATE_V] >= fit_cfg.min_heading_speed

        if not np.isfinite(ss_rmse) or ss_rmse > max_pos_rmse:
            rejected.append(int(tid))
            continue

        # Height kept from predictions, interpolated across gaps. Uses the full
        # observed frames (height is untouched by the position/yaw pre-clean and
        # trimming, which only drop points from the ground-plane fit inputs).
        obs_off = np.asarray([f - first for f in frames], dtype=np.int64)
        obs_y = np.asarray(
            [track[f]["translation"][UP_AXIS] for f in frames], dtype=np.float64
        )
        span = np.arange(n_steps + 1)
        y_interp = np.interp(span, obs_off, obs_y)

        observed = set(frames)
        for off in range(n_steps + 1):
            f = first + off
            if f in observed:
                box = track[f]
                base_rot = box["rotation"]
            else:
                nearest = min(observed, key=lambda o: abs(o - f))
                box = copy.deepcopy(track[nearest])
                base_rot = box["rotation"]
                box["interpolated"] = True
                if 0 <= f < n_keys and "sample_token" in box:
                    box["sample_token"] = frame_keys[f]
                track[f] = box
                n_filled += 1

            t = list(box["translation"])
            t[ax0] = float(traj_pos[off, 0])
            t[ax1] = float(traj_pos[off, 1])
            t[UP_AXIS] = float(y_interp[off])
            box["translation"] = t

            if bool(traj_valid[off]):
                base_yaw = _yaw_from_box_rotation({"rotation": base_rot})
                delta = float(traj_yaw[off] - base_yaw)
                rot_new = _yaw_matrix(delta) @ quat_wxyz_to_matrix(base_rot)
                box["rotation"] = _matrix_to_quat_wxyz(rot_new).tolist()

        n_fitted += 1
        params[str(tid)] = {
            "wheelbase": float(wheelbase),
            "lr_ratio": float(fit_cfg.lr_ratio),
            "dt": float(dt),
            "first_frame": int(first),
            "n_steps": int(n_steps),
            "state0": [float(v) for v in res.states[0]],
            "accel": [float(v) for v in res.accel],
            "steer": [float(v) for v in res.steer],
        }
        per_track[str(tid)] = {
            "class": _track_class(track),
            "n_obs": int(res.n_obs),
            "success": bool(res.success),
            "pos_rmse": float(ss_rmse),
            "yaw_rmse": float(res.yaw_rmse),
            "wheelbase": float(wheelbase),
            "first_frame": int(first),
            "n_frames": int(n_steps + 1),
        }

    report = {
        "tracks_fitted": n_fitted,
        "frames_filled": n_filled,
        "tracks_precleaned_frames": n_precleaned,
        "tracks_trimmed": n_trimmed_tracks,
        "tracks": per_track,
        "params": params,
        "rejected_ids": rejected,
    }
    return n_fitted, report


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def tracks_to_results(
    tracks: dict[int, dict[int, dict]], frame_keys: list[str]
) -> dict:
    """Rebuild the ``{token: [boxes]}`` structure from track timelines."""
    results: dict[str, list[dict]] = {k: [] for k in frame_keys}
    for track in tracks.values():
        for fidx, box in track.items():
            results[frame_keys[fidx]].append(box)
    return results


def _write_user_iteration(
    root_dir: str,
    base_payload: dict,
    tracks: dict[int, dict[int, dict]],
    frame_keys: list[str],
    iteration_idx: int,
    command: str,
    data_root: str,
    camera_names: list[str],
    project_cfg: dict,
    bicycle_params: dict | None = None,
    bicycle_alpha: float = 0.6,
) -> str:
    """Write one numbered user-refinement snapshot and return its directory."""
    it_dir = os.path.join(root_dir, f"{iteration_idx:03d}")
    os.makedirs(it_dir, exist_ok=True)

    out = dict(base_payload)
    out["results"] = tracks_to_results(tracks, frame_keys)
    out_json = os.path.join(it_dir, "track_3d_refined_colmap.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f)

    report = {
        "iteration": iteration_idx,
        "command": command,
        "tracks": len(tracks),
    }
    with open(os.path.join(it_dir, "refine_report_user.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Persist the fitted bicycle-model params for this iteration (reflecting any
    # manual edits) so an interrupted interactive session and the simulator can
    # read the newest parameters without waiting for the loop to finish with
    # 'done'. Written both into the iteration snapshot and refreshed at the
    # top-level refine dir (which is what resume and load_rigid_tracks read).
    if bicycle_params is not None:
        sidecar = {
            "frame_keys": frame_keys,
            "wheelbase_alpha": float(bicycle_alpha),
            "tracks": bicycle_params,
        }
        with open(os.path.join(it_dir, "bicycle_params.json"), "w", encoding="utf-8") as f:
            json.dump(sidecar, f)
        top_dir = os.path.dirname(root_dir)
        with open(os.path.join(top_dir, "bicycle_params.json"), "w", encoding="utf-8") as f:
            json.dump(sidecar, f)

    with open(os.path.join(it_dir, ".success"), "w", encoding="utf-8") as f:
        f.write("User refinement iteration finished successfully.")

    with open(os.path.join(root_dir, "latest.txt"), "w", encoding="utf-8") as f:
        f.write(f"{iteration_idx:03d}\n")

    proj_report = _project_results_snapshot(
        results=out["results"],
        data_root=data_root,
        output_dir=os.path.join(it_dir, "projected"),
        camera_names=camera_names,
        project_cfg=project_cfg,
    )

    with open(os.path.join(it_dir, "project_report.json"), "w", encoding="utf-8") as f:
        json.dump(proj_report, f, indent=2)

    frame_mapping = proj_report.get("frame_mapping", {})
    return it_dir, frame_mapping


def _project_results_snapshot(
    results: dict,
    data_root: str,
    output_dir: str,
    camera_names: list[str],
    project_cfg: dict,
) -> dict:
    """Project one refined snapshot to camera frames and return a report."""
    from src.tracking.project_tracks import project_tracks

    os.makedirs(output_dir, exist_ok=True)
    return project_tracks(
        results=results,
        data_root=to_absolute_path(data_root),
        output_dir=output_dir,
        camera_names=list(camera_names),
        box_width=int(project_cfg.get("box_width", 2)),
        subdiv=int(project_cfg.get("subdiv", 12)),
        max_frames=int(project_cfg.get("max_frames", 0)) or None,
        max_view_angle=float(project_cfg.get("max_view_angle", 80.0)),
    )


def _parse_track_selector(text: str) -> tuple[str, int]:
    """Parse ``<class>_<id>`` selectors (class can include underscores)."""
    token = text.strip()
    cls, sep, raw_id = token.rpartition("_")
    if not sep or not cls:
        raise ValueError(f"Invalid selector '{text}'. Expected '<class>_<id>'.")
    try:
        tid = int(raw_id)
    except ValueError as exc:
        raise ValueError(f"Invalid track id in selector '{text}'.") from exc
    return cls, tid


def _validate_selector(
    tracks: dict[int, dict[int, dict]],
    selector: str,
) -> tuple[int, str]:
    """Resolve selector to (track_id, current_track_class), validating class."""
    cls, tid = _parse_track_selector(selector)
    if tid not in tracks:
        raise ValueError(f"Track id {tid} not found.")
    current = _track_class(tracks[tid])
    if cls != current:
        raise ValueError(
            f"Selector class mismatch for id {tid}: got '{cls}', current is '{current}'."
        )
    return tid, current


def _manual_fuse_tracks(
    tracks: dict[int, dict[int, dict]], selectors: list[str]
) -> tuple[int, int]:
    """Fuse selected tracks in place. Returns (canonical_id, removed_count)."""
    resolved = [_validate_selector(tracks, s) for s in selectors]
    ids = [tid for tid, _ in resolved]
    uniq_ids = list(dict.fromkeys(ids))
    if len(uniq_ids) < 2:
        raise ValueError("Fuse requires at least two distinct tracks.")

    canonical = max(uniq_ids, key=lambda t: len(tracks[t]))
    merged_track = copy.deepcopy(tracks[canonical])

    for tid in uniq_ids:
        if tid == canonical:
            continue
        for fidx, box in tracks[tid].items():
            existing = merged_track.get(fidx)
            if existing is None or float(box.get("tracking_score", 1.0)) > float(
                existing.get("tracking_score", 1.0)
            ):
                merged_track[fidx] = copy.deepcopy(box)

    name = _track_class(merged_track)
    for box in merged_track.values():
        box["tracking_id"] = int(canonical)
        box["tracking_name"] = name

    tracks[canonical] = merged_track
    removed = 0
    for tid in uniq_ids:
        if tid != canonical and tid in tracks:
            del tracks[tid]
            removed += 1

    return canonical, removed


def _extend_single_track(
    track: dict[int, dict],
    frame_keys: list[str],
    amount: int,
    vel_window: int,
) -> int:
    """Extend exactly one track on one side. Positive=end, negative=start."""
    n_keys = len(frame_keys)
    frames = sorted(track)
    if len(frames) < 2 or amount == 0:
        return 0

    n_ext = 0
    if amount > 0:
        last = frames[-1]
        vel = _endpoint_velocity_3d(
            track,
            frames,
            at_start=False,
            window=vel_window,
        )
        base_box = track[last]
        base_t = np.asarray(base_box["translation"], dtype=np.float64)
        for k in range(1, amount + 1):
            f = last + k
            if f >= n_keys or f in track:
                break
            token = frame_keys[f]
            track[f] = _extrap_box(base_box, base_t + vel * k, token)
            n_ext += 1
        return n_ext

    first = frames[0]
    vel = _endpoint_velocity_3d(
        track,
        frames,
        at_start=True,
        window=vel_window,
    )
    base_box = track[first]
    base_t = np.asarray(base_box["translation"], dtype=np.float64)
    for k in range(1, abs(amount) + 1):
        f = first - k
        if f < 0 or f in track:
            break
        token = frame_keys[f]
        track[f] = _extrap_box(base_box, base_t - vel * k, token)
        n_ext += 1
    return n_ext


def _extend_single_track_stopped(
    track: dict[int, dict],
    frame_keys: list[str],
    amount: int,
) -> int:
    """Extend one track by copying endpoint boxes without motion."""
    n_keys = len(frame_keys)
    frames = sorted(track)
    if len(frames) < 1 or amount == 0:
        return 0

    n_ext = 0
    if amount > 0:
        last = frames[-1]
        base_box = track[last]
        base_t = np.asarray(base_box["translation"], dtype=np.float64)
        for k in range(1, amount + 1):
            f = last + k
            if f >= n_keys or f in track:
                break
            token = frame_keys[f]
            track[f] = _extrap_box(base_box, base_t, token)
            n_ext += 1
        return n_ext

    first = frames[0]
    base_box = track[first]
    base_t = np.asarray(base_box["translation"], dtype=np.float64)
    for k in range(1, abs(amount) + 1):
        f = first - k
        if f < 0 or f in track:
            break
        token = frame_keys[f]
        track[f] = _extrap_box(base_box, base_t, token)
        n_ext += 1
    return n_ext


def _remove_track_frames(track: dict[int, dict], frame_range: tuple[int, int]) -> int:
    """Remove frames in the given frame index range (inclusive). Returns count removed."""
    frames = sorted(track)
    if not frames:
        return 0
    start, end = frame_range
    # Clamp to valid frame indices in the track
    start = max(start, min(frames))
    end = min(end, max(frames))
    targets = [f for f in frames if start <= f <= end]
    for f in targets:
        track.pop(f, None)
    return len(targets)


def _should_bike_fit(track: dict[int, dict], bicycle_cfg: dict) -> bool:
    """True if a track is a bicycle-fit-eligible vehicle (class + length)."""
    classes = set(bicycle_cfg.get("classes", DEFAULT_BICYCLE_CLASSES))
    min_frames = int(bicycle_cfg.get("min_track_frames", 3))
    return _track_class(track) in classes and len(track) >= max(min_frames, 2)


def _refit_track_bicycle(
    tid: int,
    track: dict[int, dict],
    frame_keys: list[str],
    bicycle_cfg: dict,
    bicycle_params: dict,
) -> None:
    """Re-fit one vehicle track's bicycle model in place and refresh its params.

    Re-runs :func:`apply_bicycle_fit` on the single track so its dense per-frame
    poses and persisted parameters stay mutually consistent after a structural
    edit (fuse / model extend). Updates ``bicycle_params[str(tid)]`` in place,
    dropping the entry if the fit no longer applies.
    """
    _, rep = apply_bicycle_fit({tid: track}, frame_keys, bicycle_cfg)
    entry = rep.get("params", {}).get(str(tid))
    if entry is not None:
        bicycle_params[str(tid)] = entry
    else:
        bicycle_params.pop(str(tid), None)


def _extend_single_track_bicycle(
    track: dict[int, dict],
    params_entry: dict,
    frame_keys: list[str],
    amount: int,
) -> int:
    """Extend one bicycle-fitted track using the kinematic model rollout.

    Rolls the fitted CoG bicycle model past the observed span with constant
    control (zero acceleration, held steering) so the extension follows the
    vehicle's last curvature instead of a straight constant-velocity vector.
    Positive ``amount`` extends after the last frame, negative before the first.
    Height (up axis) is held at the endpoint value; heading follows the model.
    Returns the number of frames added.
    """
    frames = sorted(track)
    if len(frames) < 1 or amount == 0:
        return 0

    wheelbase = float(params_entry["wheelbase"])
    lr_ratio = float(params_entry["lr_ratio"])
    dt = float(params_entry["dt"])
    state0 = np.asarray(params_entry["state0"], dtype=np.float64)
    accel = np.asarray(params_entry["accel"], dtype=np.float64)
    steer = np.asarray(params_entry["steer"], dtype=np.float64)
    states = _bicycle_rollout(state0, accel, steer, dt, wheelbase, lr_ratio)

    n_keys = len(frame_keys)
    ax0, ax1 = GROUND_AXES

    def _write_box(f: int, state: np.ndarray, base_box: dict) -> None:
        box = copy.deepcopy(base_box)
        box["extrapolated"] = True
        if 0 <= f < n_keys and "sample_token" in box:
            box["sample_token"] = frame_keys[f]
        t = list(box["translation"])
        t[ax0] = float(state[0])
        t[ax1] = float(state[1])
        t[UP_AXIS] = float(base_box["translation"][UP_AXIS])
        box["translation"] = t
        base_yaw = _yaw_from_box_rotation({"rotation": base_box["rotation"]})
        delta = float(state[2] - base_yaw)
        rot_new = _yaw_matrix(delta) @ quat_wxyz_to_matrix(base_box["rotation"])
        box["rotation"] = _matrix_to_quat_wxyz(rot_new).tolist()
        track[f] = box

    if amount > 0:
        last = frames[-1]
        n_add = 0
        for k in range(1, amount + 1):
            f = last + k
            if f >= n_keys or f in track:
                break
            n_add += 1
        if n_add == 0:
            return 0
        end_state = states[-1]
        last_steer = float(steer[-1]) if steer.size else 0.0
        ext = _bicycle_rollout(
            end_state,
            np.zeros(n_add),
            np.full(n_add, last_steer),
            dt,
            wheelbase,
            lr_ratio,
        )
        base_box = track[last]
        for k in range(1, n_add + 1):
            _write_box(last + k, ext[k], base_box)
        return n_add

    first = frames[0]
    amt = abs(amount)
    n_add = 0
    for k in range(1, amt + 1):
        f = first - k
        if f < 0 or f in track:
            break
        n_add += 1
    if n_add == 0:
        return 0
    first_steer = float(steer[0]) if steer.size else 0.0
    # Integrate backward in time by rolling the model with a negative timestep.
    back = _bicycle_rollout(
        states[0],
        np.zeros(n_add),
        np.full(n_add, first_steer),
        -dt,
        wheelbase,
        lr_ratio,
    )
    base_box = track[first]
    for k in range(1, n_add + 1):
        _write_box(first - k, back[k], base_box)
    return n_add


def _apply_user_command(
    tracks: dict[int, dict[int, dict]],
    cmd: str,
    args: str,
    default_extend: int,
    vel_window: int,
    frame_keys: list[str],
    bicycle_params: dict | None = None,
    bicycle_cfg: dict | None = None,
    refit_bicycle: bool = True,
    frame_mapping: dict[str, int] | None = None,
) -> str:
    """Apply one interactive user-refinement command in place.

    When ``bicycle_params`` is provided and bicycle fitting is enabled, edits are
    kept consistent with the persisted per-track model parameters: filtering
    drops the track's params, fusing re-fits the merged vehicle track, and
    extending a fitted vehicle track uses the kinematic model rollout.

    ``refit_bicycle`` gates the expensive per-track bicycle re-fit. During
    staging (dry-run preview) it is ``False`` so typing a command does not run
    the fit; the fit runs once when the batch is applied.
    """
    bicycle_cfg = bicycle_cfg or {}
    bike_enabled = (
        bicycle_params is not None and bool(bicycle_cfg.get("enabled", False))
    )

    if cmd == "filter":
        if not args:
            raise ValueError("Filter expects 'Filter: <class>_<id>'.")
        tid, cls = _validate_selector(tracks, args)
        del tracks[tid]
        if bicycle_params is not None:
            bicycle_params.pop(str(tid), None)
        return f"🗑️ Staged filter for {cls}_{tid}."

    if cmd == "fuse":
        specs = [s.strip() for s in args.split(",") if s.strip()]
        if len(specs) < 2:
            raise ValueError("Fuse expects at least two selectors.")
        pre_ids = [_validate_selector(tracks, s)[0] for s in specs]
        canonical, removed = _manual_fuse_tracks(tracks, specs)
        refit_note = ""
        if bike_enabled:
            for tid in set(pre_ids):
                if tid != canonical:
                    bicycle_params.pop(str(tid), None)
            if _should_bike_fit(tracks[canonical], bicycle_cfg):
                if refit_bicycle:
                    _refit_track_bicycle(
                        canonical, tracks[canonical], frame_keys, bicycle_cfg, bicycle_params
                    )
                refit_note = " (re-fit bicycle model)"
            else:
                bicycle_params.pop(str(canonical), None)
        return f"🔗 Staged fuse into id {canonical}; removed {removed} tracks{refit_note}."

    if cmd == "extend":
        parts = [s.strip() for s in args.split(",") if s.strip()]
        if not parts:
            raise ValueError("Extend expects 'Extend: <class>_<id>[, <int>]'.")
        selector = parts[0]
        amount = default_extend
        if len(parts) >= 2:
            amount = int(parts[1].strip().rstrip(";"))
        tid, cls = _validate_selector(tracks, selector)
        frames_before = sorted(tracks[tid])
        first_before = frames_before[0] if frames_before else 0
        last_before = frames_before[-1] if frames_before else -1
        use_model = bike_enabled and str(tid) in bicycle_params
        if use_model:
            added = _extend_single_track_bicycle(
                track=tracks[tid],
                params_entry=bicycle_params[str(tid)],
                frame_keys=frame_keys,
                amount=amount,
            )
            if added > 0 and refit_bicycle and _should_bike_fit(tracks[tid], bicycle_cfg):
                _refit_track_bicycle(
                    tid, tracks[tid], frame_keys, bicycle_cfg, bicycle_params
                )
        else:
            added = _extend_single_track(
                track=tracks[tid],
                frame_keys=frame_keys,
                amount=amount,
                vel_window=vel_window,
            )
        mode = " (bicycle model)" if use_model else ""
        if added == 0:
            if amount > 0 and last_before >= (len(frame_keys) - 1):
                return (
                    f"↕️ Staged extend for {cls}_{tid} by {amount}; adds 0 frames "
                    "(track already reaches the last frame; use negative amount to extend earlier frames)."
                )
            if amount < 0 and first_before <= 0:
                return (
                    f"↕️ Staged extend for {cls}_{tid} by {amount}; adds 0 frames "
                    "(track already starts at the first frame; use positive amount to extend later frames)."
                )
        return f"↕️ Staged extend for {cls}_{tid} by {amount}; adds {added} frames{mode}."

    if cmd in {"extend_stoped", "extend_stopped"}:
        parts = [s.strip() for s in args.split(",") if s.strip()]
        if len(parts) < 2:
            raise ValueError(
                "Extend_stopped expects 'Extend_stopped: <class>_<id>, <int>'."
            )
        selector = parts[0]
        amount = int(parts[1].strip().rstrip(";"))
        if amount == 0:
            raise ValueError("Extend_stopped amount must be non-zero.")
        tid, cls = _validate_selector(tracks, selector)
        frames_before = sorted(tracks[tid])
        first_before = frames_before[0] if frames_before else 0
        last_before = frames_before[-1] if frames_before else -1
        added = _extend_single_track_stopped(
            track=tracks[tid],
            frame_keys=frame_keys,
            amount=amount,
        )
        if added == 0:
            if amount > 0 and last_before >= (len(frame_keys) - 1):
                return (
                    f"🧱 Staged stopped-extend for {cls}_{tid} by {amount}; adds 0 frames "
                    "(track already reaches the last frame; use negative amount to extend earlier frames)."
                )
            if amount < 0 and first_before <= 0:
                return (
                    f"🧱 Staged stopped-extend for {cls}_{tid} by {amount}; adds 0 frames "
                    "(track already starts at the first frame; use positive amount to extend later frames)."
                )
        return f"🧱 Staged stopped-extend for {cls}_{tid} by {amount}; adds {added} frames."

    if cmd == "remove":
        parts = [s.strip() for s in args.split(",") if s.strip()]
        if len(parts) < 2:
            raise ValueError("Remove expects 'Remove: <class>_<id>, <start>-<end>'.")
        selector = parts[0]
        range_str = parts[1].strip().rstrip(";")
        if "-" not in range_str:
            raise ValueError("Remove range must be 'start-end' (e.g., '10-50').")
        try:
            user_start, user_end = map(int, range_str.split("-"))
        except ValueError:
            raise ValueError(f"Invalid frame range '{range_str}'; expected 'start-end'.")
        if user_start > user_end:
            user_start, user_end = user_end, user_start
        
        # Collect all sequence indices for the requested range
        frame_indices_to_remove = set()
        if frame_mapping:
            for seq_idx in range(user_start, user_end + 1):
                ts = frame_mapping.get(str(seq_idx))
                if ts is None:
                    raise ValueError(
                        f"Frame index {seq_idx} out of bounds. "
                        f"Valid range: 0-{max(int(k) for k in frame_mapping.keys()) if frame_mapping else 'unknown'}."
                    )
                # frame_mapping maps seq_idx -> timestamp, but tracks are keyed by seq_idx
                frame_indices_to_remove.add(seq_idx)
        else:
            # Fallback: use sequence indices directly
            for seq_idx in range(user_start, user_end + 1):
                frame_indices_to_remove.add(seq_idx)
        
        tid, cls = _validate_selector(tracks, selector)
        
        # Debug: print what we're trying to remove
        track_frames = sorted(tracks[tid].keys())
        #print(f"🔍 DEBUG Remove: track {cls}_{tid} has frames {track_frames[:10]}... (total: {len(track_frames)})")
        #print(f"🔍 DEBUG Remove: trying to remove frame indices {sorted(frame_indices_to_remove)}")
        #print(f"🔍 DEBUG Remove: frame_mapping keys range: {sorted(int(k) for k in frame_mapping.keys())[:5]}...{sorted(int(k) for k in frame_mapping.keys())[-5:] if frame_mapping else 'None'}")
        
        # Remove specific frames by sequence index
        removed = 0
        for frame_idx in frame_indices_to_remove:
            if frame_idx in tracks[tid]:
                del tracks[tid][frame_idx]
                removed += 1
        
        if not tracks[tid]:
            del tracks[tid]
            return (
                f"✂️ Staged remove for {cls}_{tid} frames {user_start}-{user_end}; removed {removed} frames "
                "(track became empty and was deleted)."
            )
        return f"✂️ Staged remove for {cls}_{tid} frames {user_start}-{user_end}; removed {removed} frames."

    raise ValueError(
        "Unknown command. Use Filter/Fuse/Extend/Extend_stopped/Remove/apply/undo/done."
    )


def _replay_pending_commands(
    tracks: dict[int, dict[int, dict]],
    params: dict,
    pending_commands: list[str],
    default_extend: int,
    vel_window: int,
    frame_keys: list[str],
    bicycle_cfg: dict,
    refit_bicycle: bool = True,
    frame_mapping: dict[str, int] | None = None,
) -> tuple[dict[int, dict[int, dict]], dict, list[str]]:
    """Return preview tracks/params/messages after replaying staged commands."""
    preview = copy.deepcopy(tracks)
    preview_params = copy.deepcopy(params)
    messages: list[str] = []
    for raw in pending_commands:
        raw_clean = raw.strip().rstrip(";").strip()
        if ":" in raw_clean:
            head, tail = raw_clean.split(":", 1)
            cmd = head.strip().lower()
            args = tail.strip()
        else:
            parts = raw_clean.split(None, 1)
            cmd = parts[0].strip().lower()
            args = parts[1].strip() if len(parts) > 1 else ""
        messages.append(
            _apply_user_command(
                preview,
                cmd,
                args,
                default_extend,
                vel_window,
                frame_keys,
                preview_params,
                bicycle_cfg,
                refit_bicycle=refit_bicycle,
                frame_mapping=frame_mapping,
            )
        )
    return preview, preview_params, messages


def _run_user_refinement_loop(
    refined_results: dict,
    frame_keys: list[str],
    cfg: dict,
    output_dir: str,
    payload_template: dict,
    data_root: str,
    camera_names: list[str],
    bicycle_params: dict | None = None,
    bicycle_cfg: dict | None = None,
    allow_refit_bicycle: bool = False,
) -> tuple[dict, dict, dict]:
    """Interactive post-refinement loop with undo and numbered snapshots.

    Returns ``(final_results, summary, final_bicycle_params)``. Manual edits keep
    the persisted bicycle-model parameters consistent: filtering drops a track's
    params, fusing re-fits the merged vehicle track, and extending a fitted
    vehicle track rolls the model forward/backward.
    """
    user_cfg = cfg.get("user_refinement", {}) or {}
    bicycle_cfg = dict(bicycle_cfg or {})
    params: dict = copy.deepcopy(bicycle_params) if bicycle_params else {}
    bike_alpha = float(bicycle_cfg.get("wheelbase_alpha", 0.6))
    if not bool(user_cfg.get("enabled", False)):
        return refined_results, {
            "enabled": False,
            "iterations": 0,
            "commands_applied": 0,
        }, params

    history: list[tuple[dict[int, dict[int, dict]], dict]] = []
    commands: list[str] = []
    applied_batch_sizes: list[int] = []
    pending_commands: list[str] = []
    command_log_lines: list[str] = []
    vel_window = int(user_cfg.get("extend_velocity_window", cfg.get("extend_velocity_window", 3)))
    default_extend = int(user_cfg.get("default_extend", 5))
    project_cfg = cfg.get("project", {}) or {}

    root_dir = output_dir
    #root_dir = os.path.join(output_dir, str(user_cfg.get("output_dir", "user_refinement_what?")))
    os.makedirs(root_dir, exist_ok=True)
    command_log_path = os.path.join(root_dir, "commands_applied.txt")
    existing_iters: list[int] = []
    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)
        if not os.path.isdir(path):
            continue
        try:
            existing_iters.append(int(name))
        except ValueError:
            continue

    # Resume: if a previous session left numbered snapshots in this (config-
    # hashed) refine dir, continue editing from the latest one instead of
    # discarding those manual edits and restarting from the automatic base.
    resume = bool(user_cfg.get("resume_from_latest", True))
    resumed_from: int | None = None
    if resume and existing_iters:
        latest = max(existing_iters)
        snap_json = os.path.join(root_dir, f"{latest:03d}", "track_3d_refined_colmap.json")
        if os.path.exists(snap_json):
            try:
                with open(snap_json, encoding="utf-8") as f:
                    snap_data = json.load(f)
                resumed_results = snap_data.get("results", snap_data)
                tracks = build_tracks(copy.deepcopy(resumed_results))
                resumed_from = latest
                # Prefer the previous session's persisted bicycle params (they
                # reflect its manual edits) over this run's fresh auto fit.
                sidecar_path = os.path.join(output_dir, "bicycle_params.json")
                if os.path.exists(sidecar_path):
                    with open(sidecar_path, encoding="utf-8") as f:
                        sidecar = json.load(f)
                    saved_params = sidecar.get("tracks", {}) or {}
                    if saved_params:
                        params = copy.deepcopy(saved_params)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"⚠️ Could not resume from {snap_json}: {exc}. Starting fresh.")
                resumed_from = None

    if resumed_from is None:
        tracks = build_tracks(copy.deepcopy(refined_results))

    start_idx = (max(existing_iters) + 1) if existing_iters else 0
    iter_idx = start_idx
    first_dir, frame_mapping = _write_user_iteration(
        root_dir=root_dir,
        base_payload=payload_template,
        tracks=tracks,
        frame_keys=frame_keys,
        iteration_idx=iter_idx,
        command="initial",
        data_root=data_root,
        camera_names=camera_names,
        project_cfg=project_cfg,
        bicycle_params=params,
        bicycle_alpha=bike_alpha,
    )

    print("\n🧭 Interactive user refinement enabled.")
    if resumed_from is not None:
        print(
            f"♻️ Resumed from previous user-refinement snapshot "
            f"{resumed_from:03d} ({len(tracks)} tracks)."
        )
    print(f"📂 Iteration {iter_idx:03d} saved to: {first_dir}")
    print("Commands:")
    print("  Filter: <class>_<id>")
    print("  Fuse: <class>_<id>, <class>_<id> [, ...]")
    print(f"  Extend: <class>_<id>[, <int>] (default int={default_extend})")
    print("  Extend_stopped: <class>_<id>, <int>")
    max_frame_idx = max(int(k) for k in frame_mapping) if frame_mapping else 0
    print(f"  Remove: <class>_<id>, <start>-<end> (frame indices 0-{max_frame_idx})")
    print("  apply")
    print("  undo")
    print("  done")

    while True:
        raw = input("user-refine> ").strip()
        raw = raw.rstrip(";").strip()
        if not raw:
            continue

        if ":" in raw:
            head, tail = raw.split(":", 1)
            cmd = head.strip().lower()
            args = tail.strip()
        else:
            parts = raw.strip().split(None, 1)
            cmd = parts[0].strip().lower()
            args = parts[1].strip() if len(parts) > 1 else ""

        try:
            if cmd == "done":
                if pending_commands:
                    print("⚠️ Pending staged commands exist. Use 'apply' or 'undo' before 'done'.")
                    continue
                break

            if cmd == "undo":
                if pending_commands:
                    dropped = pending_commands.pop()
                    print(f"↩️ Removed staged command: {dropped}")
                    continue
                if not history:
                    print("⚠️ Nothing to undo.")
                    continue
                tracks, params = history.pop()
                if applied_batch_sizes:
                    n_drop = applied_batch_sizes.pop()
                    if n_drop > 0:
                        del commands[-n_drop:]
                iter_idx += 1
                out_dir, frame_mapping = _write_user_iteration(
                    root_dir=root_dir,
                    base_payload=payload_template,
                    tracks=tracks,
                    frame_keys=frame_keys,
                    iteration_idx=iter_idx,
                    command="undo",
                    data_root=data_root,
                    camera_names=camera_names,
                    project_cfg=project_cfg,
                    bicycle_params=params,
                    bicycle_alpha=bike_alpha,
                )
                command_log_lines.append(f"iter {iter_idx:03d} | undo")
                print(f"↩️ Undo applied. Snapshot: {out_dir}")
                continue

            if cmd == "apply":
                if not pending_commands:
                    print("⚠️ No staged commands to apply.")
                    continue
                preview_tracks, preview_params, _ = _replay_pending_commands(
                    tracks,
                    params,
                    pending_commands,
                    default_extend,
                    vel_window,
                    frame_keys,
                    bicycle_cfg,
                    frame_mapping=frame_mapping,
                    refit_bicycle=allow_refit_bicycle,
                )
                history.append((copy.deepcopy(tracks), copy.deepcopy(params)))
                tracks = preview_tracks
                params = preview_params
                commands.extend(pending_commands)
                applied_batch = list(pending_commands)
                applied_batch_sizes.append(len(applied_batch))
                pending_commands.clear()
                iter_idx += 1
                out_dir, frame_mapping = _write_user_iteration(
                    root_dir=root_dir,
                    base_payload=payload_template,
                    tracks=tracks,
                    frame_keys=frame_keys,
                    iteration_idx=iter_idx,
                    command="apply | " + " ; ".join(applied_batch),
                    data_root=data_root,
                    camera_names=camera_names,
                    project_cfg=project_cfg,
                    bicycle_params=params,
                    bicycle_alpha=bike_alpha,
                )
                command_log_lines.append(
                    f"iter {iter_idx:03d} | apply | " + " ; ".join(applied_batch)
                )
                print(f"✅ Applied {len(applied_batch)} staged command(s). Snapshot: {out_dir}")
                continue

            if cmd not in {"filter", "fuse", "extend", "extend_stoped", "extend_stopped", "remove"}:
                raise ValueError(
                    "Unknown command. Use Filter/Fuse/Extend/Extend_stopped/Remove/apply/undo/done."
                )

            preview_tracks, preview_params, _ = _replay_pending_commands(
                tracks,
                params,
                pending_commands,
                default_extend,
                vel_window,
                frame_keys,
                bicycle_cfg,
                refit_bicycle=allow_refit_bicycle,
                frame_mapping=frame_mapping,
            )
            stage_msg = _apply_user_command(
                preview_tracks,
                cmd,
                args,
                default_extend,
                vel_window,
                frame_keys,
                preview_params,
                bicycle_cfg,
                refit_bicycle=allow_refit_bicycle,
                frame_mapping=frame_mapping,
            )
            pending_commands.append(raw)
            print(stage_msg)
            print(
                f"📝 {len(pending_commands)} command(s) staged. Use 'apply' to write a new iteration."
            )

        except Exception as exc:
            print(f"❌ {exc}")

    final_results = tracks_to_results(tracks, frame_keys)
    summary = {
        "enabled": True,
        "iterations": iter_idx - start_idx + 1,
        "commands_applied": len(commands),
        "history_depth": len(history),
        "output_root": root_dir,
        "command_log": command_log_path,
    }

    with open(command_log_path, "a", encoding="utf-8") as f:
        f.write("=== user refinement session ===\n")
        f.write(f"start_iteration: {start_idx:03d}\n")
        for line in command_log_lines:
            f.write(line + "\n")
        f.write(f"commands_applied_total: {len(commands)}\n")
        f.write("\n")

    return final_results, summary, params


def _build_track_change_summary(
    fusion_report: dict,
    kept_ids: list[int],
    dropped: list[dict],
    filled_details: list[dict],
    extended_details: list[dict],
) -> list[dict]:
    """Build compact per-track summary from detailed refinement change logs."""
    summary: dict[int, dict] = {}

    def ensure(tid: int) -> dict:
        if tid not in summary:
            summary[tid] = {
                "tracking_id": int(tid),
                "kept": None,
                "drop_reason": None,
                "fusion": {
                    "canonical_id": int(tid),
                    "group_size": 1,
                    "merged_from_ids": [],
                    "merged_into": None,
                    "overlap_candidate_frames": 0,
                    "overlap_selected_frames": 0,
                    "overlap_rejected_frames": 0,
                    "overlap_dropped_frames": 0,
                },
                "filled_frames": 0,
                "extended_before": 0,
                "extended_after": 0,
            }
        return summary[tid]

    for group in fusion_report.get("groups", []):
        canonical = int(group.get("canonical_id", -1))
        members = [int(m) for m in group.get("members", [])]
        for m in members:
            e = ensure(m)
            e["fusion"]["canonical_id"] = canonical
            e["fusion"]["group_size"] = len(members)
            if m == canonical:
                e["fusion"]["merged_from_ids"] = [x for x in members if x != canonical]
            else:
                e["fusion"]["merged_into"] = canonical

    for d in fusion_report.get("overlap_decisions", []):
        cand = [int(t) for t in d.get("candidate_ids", [])]
        rej = {int(t) for t in d.get("rejected_ids", [])}
        sel = d.get("selected_id", None)
        sel = int(sel) if sel is not None else None
        dropped_frame = bool(d.get("dropped_frame", False))
        for tid in cand:
            e = ensure(tid)
            e["fusion"]["overlap_candidate_frames"] += 1
            if tid in rej:
                e["fusion"]["overlap_rejected_frames"] += 1
            if sel is not None and tid == sel:
                e["fusion"]["overlap_selected_frames"] += 1
            if dropped_frame:
                e["fusion"]["overlap_dropped_frames"] += 1

    for tid in kept_ids:
        e = ensure(int(tid))
        e["kept"] = True

    for d in dropped:
        tid = int(d.get("id", -1))
        if tid < 0:
            continue
        e = ensure(tid)
        e["kept"] = False
        e["drop_reason"] = d.get("reason", "unknown")

    for d in filled_details:
        tid = int(d.get("tracking_id", -1))
        if tid >= 0:
            ensure(tid)["filled_frames"] += 1

    for d in extended_details:
        tid = int(d.get("tracking_id", -1))
        if tid < 0:
            continue
        e = ensure(tid)
        if d.get("direction") == "before":
            e["extended_before"] += 1
        else:
            e["extended_after"] += 1

    return [summary[k] for k in sorted(summary)]


def refine(results: dict, cfg: dict, output_dir: str = "", data_root: str = "") -> tuple[dict, dict]:
    """Run fusion then filtering. Returns (refined_results, report).

    Operates on a deep copy so the caller's ``results`` (and its box dicts) are
    never mutated \u2014 fusion rewrites ``tracking_id``/``tracking_name`` in place.
    """
    results = copy.deepcopy(results)
    frame_keys, _ = _frame_index(results)
    tracks = build_tracks(results)
    n_in = len(tracks)

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
        model_assisted_gap=bool(bicycle_cfg.get("model_assisted_gap", False)),
        gap_model_alpha=float(bicycle_cfg.get("wheelbase_alpha", 0.6)),
        gap_model_lr_ratio=float(bicycle_cfg.get("lr_ratio", 0.5)),
        gap_model_dt=float(bicycle_cfg.get("dt", 0.1)),
    )
    fused, fusion_report = apply_fusion(
        tracks,
        mapping,
        pose_filter_cfg=cfg.get("fusion_pose_filter", {}) or {},
        frame_keys=frame_keys,
    )
    n_fused = len(fused)

    kept, dropped = filter_static(
        fused,
        min_displacement=float(cfg.get("min_displacement", 2.5)),
        min_track_length=int(cfg.get("min_track_length", 3)),
        displacement_percentile=float(cfg.get("displacement_percentile", 90.0)),
        displacement_mode=str(cfg.get("displacement_mode", "net")),
        displacement_smooth=int(cfg.get("displacement_smooth", 3)),
        min_rel_displacement=float(cfg.get("min_rel_displacement", 0.0)),
    )

    # Optional interactive user edits before bicycle fitting to remove outliers.
    refined_kept = kept
    if bool(cfg.get("user_refinement", {}).get("enabled", False)) and output_dir:
        cameras = list(cfg.get("project", {}).get("cameras", []) or [])
        bicycle_cfg_user = cfg.get("bicycle_fit", {}) or {}
        user_refinement_dir = os.path.join(output_dir, "user_refinement")
        refined_kept_results, user_report_pre, _ = _run_user_refinement_loop(
            refined_results=tracks_to_results(kept, frame_keys),
            frame_keys=frame_keys,
            cfg=cfg,
            output_dir=user_refinement_dir,
            payload_template={},
            data_root=data_root,
            camera_names=cameras,
            bicycle_params={},
            bicycle_cfg=bicycle_cfg_user,
            allow_refit_bicycle=False,
        )
        refined_kept = build_tracks(refined_kept_results)
        kept = refined_kept

    # Vehicle tracks are refined by a per-track kinematic bicycle-model fit
    # (gap-fill + denoise + kinematic consistency in one), which supersedes the
    # heuristic fill/extend/smooth stages for those classes. Non-vehicle tracks
    # (and vehicle tracks too short to fit) keep the legacy path below.
    bicycle_enabled = bool(bicycle_cfg.get("enabled", False))
    bicycle_report: dict = {}
    fit_ids: set[int] = set()
    if bicycle_enabled:
        fit_ids = set(select_bicycle_track_ids(kept, bicycle_cfg))
        fit_kept = {tid: kept[tid] for tid in fit_ids}
        _, bicycle_report = apply_bicycle_fit(fit_kept, frame_keys, bicycle_cfg)
        # Fits rejected by the consistency gate keep their raw boxes (never
        # baked) — route them to the legacy fill/smooth path below.
        rejected_ids = set(bicycle_report.get("rejected_ids", []))
        params = bicycle_report.get("params", {})
        fit_ids -= rejected_ids

    # Optional interactive user edits after bicycle fitting to refine kinematic results.
    refined_kept = kept
    if bool(cfg.get("user_refinement", {}).get("enabled", False)) and bool(cfg.get("user_refinement", {}).get("post_bicycle_fit", False)) and output_dir:
        cameras = list(cfg.get("project", {}).get("cameras", []) or [])
        bicycle_cfg_user = cfg.get("bicycle_fit", {}) or {}
        user_refinement_post_dir = os.path.join(output_dir, "user_refinement_post_bicycle")
        refined_kept_results, user_report_post, _ = _run_user_refinement_loop(
            refined_results=tracks_to_results(kept, frame_keys),
            frame_keys=frame_keys,
            cfg=cfg,
            output_dir=user_refinement_post_dir,
            payload_template={},
            data_root=data_root,
            camera_names=cameras,
            bicycle_params=params,
            bicycle_cfg=bicycle_cfg_user,
            allow_refit_bicycle=True,
        )
        refined_kept = build_tracks(refined_kept_results)
        kept = refined_kept

    # Legacy heuristic path, applied only to tracks NOT handled by the fit.
    legacy_kept = {tid: t for tid, t in kept.items() if tid not in fit_ids}

    # Interpolate interior frame gaps so each kept track is continuous within its
    # lifespan (the tracker frequently drops single frames). Done after filtering
    # so we never waste work on tracks that get dropped.
    n_filled = 0
    filled_details: list[dict] = []
    if bool(cfg.get("fill_gaps", True)):
        n_filled, filled_details = fill_track_gaps(legacy_kept, frame_keys)

    # Extrapolate a bounded number of frames past each track's ends to cover the
    # fisheye FOV edges / far-away dropouts the (narrower-FOV) tracker misses.
    n_extended = 0
    extended_details: list[dict] = []
    if int(cfg.get("extend_frames", 0)) > 0:
        n_extended, extended_details = extend_track_ends(
            legacy_kept,
            frame_keys,
            extend_frames=int(cfg.get("extend_frames", 0)),
            vel_window=int(cfg.get("extend_velocity_window", 3)),
        )

    # Low-pass the (now dense) per-frame translations to remove tracker jitter,
    # which otherwise shows up as shaky rigid-object motion in the 4DGS render.
    n_smoothed = 0
    if int(cfg.get("pose_smooth_window", 0)) > 1:
        n_smoothed = smooth_track_translations(
            legacy_kept, window=int(cfg.get("pose_smooth_window", 0))
        )

    # Optional yaw-only pose smoothing. Can either smooth the raw box yaw, or
    # derive yaw from the local tangent of the already-smoothed translation path
    # so headings stay stable while still following curved trajectories.
    n_rot_smoothed = 0
    if bool(cfg.get("pose_smooth_rotation", False)):
        n_rot_smoothed = smooth_track_rotations(
            legacy_kept,
            window=int(
                cfg.get(
                    "pose_smooth_rotation_window", cfg.get("pose_smooth_window", 0)
                )
            ),
            mode=str(cfg.get("pose_smooth_rotation_mode", "tangent_yaw")),
            min_speed=float(cfg.get("pose_smooth_rotation_min_speed", 0.05)),
        )

    report = {
        "tracks_in": n_in,
        "tracks_after_fusion": n_fused,
        "merges": n_in - n_fused,
        "fusion": fusion_report,
        "tracks_kept": len(kept),
        "tracks_dropped": len(dropped),
        "filtering": {
            "dropped": dropped,
            "kept_ids": sorted(int(k) for k in kept.keys()),
        },
        "dropped_static": sum(1 for d in dropped if d["reason"] == "static"),
        "dropped_short": sum(1 for d in dropped if d["reason"] == "short"),
        "frames_filled": n_filled,
        "filled_frames": filled_details,
        "frames_extended": n_extended,
        "extended_frames": extended_details,
        "frames_smoothed": n_smoothed,
        "rotations_smoothed": n_rot_smoothed,
        "bicycle_fit": bicycle_report,
        "track_summary": _build_track_change_summary(
            fusion_report=fusion_report,
            kept_ids=sorted(int(k) for k in kept.keys()),
            dropped=dropped,
            filled_details=filled_details,
            extended_details=extended_details,
        ),
    }
    return tracks_to_results(kept, frame_keys), report


def _resolve_input(cfg: DictConfig) -> str:
    """Resolve the input predictions JSON path from config."""
    candidate = cfg.refine_task.get("input_json", "") or ""
    if candidate:
        return to_absolute_path(candidate)
    input_dir = cfg.refine_task.get("input_dir", "") or ""
    if input_dir:
        return os.path.join(
            to_absolute_path(input_dir),
            "eval",
            "track_3d_predictions_colmap.json",
        )
    raise ValueError(
        "refine_task requires either input_json or input_dir to be set."
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== AVSplat-Sim: Track Refinement (Step 1.5) ===")

    if not cfg.refine_task.get("enabled", True):
        print("⏭️ Refinement disabled via cfg.refine_task.enabled=false")
        return

    input_json = _resolve_input(cfg)
    output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"📁 Input:  {input_json}")
    print(f"📁 Output: {output_dir}")
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Predictions not found: {input_json}")

    with open(input_json, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", data)

    # Prepare data_root and cameras for user_refinement loop inside refine()
    data_root = cfg.refine_task.get("data_root", "") or cfg.dataset.base_dir
    data_root = to_absolute_path(data_root)
    cameras = list(cfg.refine_task.get("project", {}).get("cameras", []) or []) or list(cfg.dataset.cameras)

    refined_results, report = refine(results, cfg.refine_task, output_dir, data_root)

    # Note: user_refinement now runs BEFORE bicycle_fit inside refine() to allow
    # manual curation of outliers before fitting to a kinematic model.
    frame_keys, _ = _frame_index(refined_results)

    # Per-instance ground-plane height fitting (RANSAC).
    # Removes tracker height jitter by fitting Y = a·X + b·Z + c per vehicle
    # track, replacing every frame's Y with the plane-predicted value.
    # Run AFTER user refinement so manual edits are respected.
    height_fit_report: dict = {}
    if bool(cfg.refine_task.get("height_plane_fit", True)):
        height_tracks = build_tracks(refined_results)
        height_fit_report = fit_instance_heights_ransac(
            height_tracks,
            outlier_threshold=float(
                cfg.refine_task.get("height_plane_outlier_thr", 0.5)
            ),
            ransac_iters=int(cfg.refine_task.get("height_plane_ransac_iters", 100)),
            min_inliers=int(cfg.refine_task.get("height_plane_min_inliers", 3)),
        )
        frame_keys_final, _ = _frame_index(refined_results)
        refined_results = tracks_to_results(height_tracks, frame_keys_final)
        n_fitted = len(height_fit_report)
        n_ransac = sum(1 for v in height_fit_report.values() if not v["fallback_median"])
        print(
            f"📐 Height plane fit: {n_fitted} tracks "
            f"({n_ransac} RANSAC, {n_fitted - n_ransac} median fallback)"
        )
    report["height_plane_fit"] = height_fit_report

    # Always render projections for the final refined output so refinement
    # quality can be inspected visually, even outside the orchestrator.
    proj_report = _project_results_snapshot(
        results=refined_results,
        data_root=to_absolute_path(data_root),
        output_dir=os.path.join(output_dir, "projected"),
        camera_names=cameras,
        project_cfg=cfg.refine_task.get("project", {}) or {},
    )
    report["projection"] = proj_report

    out = dict(data)
    out["results"] = refined_results
    out_json = os.path.join(output_dir, "track_3d_refined_colmap.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f)

    # Persist the fitted bicycle-model parameters to a sidecar so the simulator
    # can extrapolate trajectories on demand. Kept separate from the refined-box
    # JSON (whose contract downstream loaders depend on) and out of the main
    # report (whose per-frame control arrays would bloat it). ``bicycle_params``
    # was extracted before the interactive loop and reflects any manual edits.
    n_bike = report.get("bicycle_fit", {}).get("tracks_fitted", 0)
    if report.get("bicycle_fit", {}).get("tracks_fitted", 0) > 0:
        sidecar = {
            "frame_keys": frame_keys,
            "wheelbase_alpha": float(
                (cfg.refine_task.get("bicycle_fit", {}) or {}).get("wheelbase_alpha", 0.6)
            ),
            "tracks": report.get("bicycle_fit", {}).get("tracks", {}),
        }
        with open(
            os.path.join(output_dir, "bicycle_params.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(sidecar, f)

    with open(os.path.join(output_dir, "refine_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"✅ Refined: {report['tracks_in']} -> {report['tracks_after_fusion']} "
        f"(merged {report['merges']}) -> kept {report['tracks_kept']} "
        f"(dropped {report['dropped_static']} static, {report['dropped_short']} short) "
        f"| filled {report['frames_filled']} gap frames"
        f", extended {report['frames_extended']} end frames"
        f", smoothed {report['frames_smoothed']} frames"
        f", rotation-smoothed {report['rotations_smoothed']} frames"
        f", bicycle-fitted {n_bike} vehicle tracks"
    )
    with open(os.path.join(output_dir, ".success"), "w", encoding="utf-8") as f:
        f.write("Refinement finished successfully.")


if __name__ == "__main__":
    main()
