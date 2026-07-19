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

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.tracking.track_geometry import (
    bev_iou,
    box_center_ground,
    quat_wxyz_to_matrix,
    UP_AXIS,
)


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
            if min(pred_dist, raw_dist) <= gap_dist:
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
    """World-frame rotation matrix for a yaw rotation about +y."""
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64
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

    return it_dir


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


def _remove_track_frames(track: dict[int, dict], amount: int) -> int:
    """Remove endpoint predictions. Positive=end, negative=start."""
    frames = sorted(track)
    if not frames or amount == 0:
        return 0
    n = abs(int(amount))
    targets = frames[-n:] if amount > 0 else frames[:n]
    for f in targets:
        track.pop(f, None)
    return len(targets)


def _apply_user_command(
    tracks: dict[int, dict[int, dict]],
    cmd: str,
    args: str,
    default_extend: int,
    vel_window: int,
    frame_keys: list[str],
) -> str:
    """Apply one interactive user-refinement command in place."""
    if cmd == "filter":
        if not args:
            raise ValueError("Filter expects 'Filter: <class>_<id>'.")
        tid, cls = _validate_selector(tracks, args)
        del tracks[tid]
        return f"🗑️ Staged filter for {cls}_{tid}."

    if cmd == "fuse":
        specs = [s.strip() for s in args.split(",") if s.strip()]
        if len(specs) < 2:
            raise ValueError("Fuse expects at least two selectors.")
        canonical, removed = _manual_fuse_tracks(tracks, specs)
        return f"🔗 Staged fuse into id {canonical}; removed {removed} tracks."

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
        added = _extend_single_track(
            track=tracks[tid],
            frame_keys=frame_keys,
            amount=amount,
            vel_window=vel_window,
        )
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
        return f"↕️ Staged extend for {cls}_{tid} by {amount}; adds {added} frames."

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
            raise ValueError("Remove expects 'Remove: <class>_<id>, <int>'.")
        selector = parts[0]
        amount = int(parts[1].strip().rstrip(";"))
        if amount == 0:
            raise ValueError("Remove amount must be non-zero.")
        tid, cls = _validate_selector(tracks, selector)
        removed = _remove_track_frames(tracks[tid], amount)
        if not tracks[tid]:
            del tracks[tid]
            return (
                f"✂️ Staged remove for {cls}_{tid} by {amount}; removed {removed} frames "
                "(track became empty and was deleted)."
            )
        return f"✂️ Staged remove for {cls}_{tid} by {amount}; removed {removed} frames."

    raise ValueError(
        "Unknown command. Use Filter/Fuse/Extend/Extend_stopped/Remove/apply/undo/done."
    )


def _replay_pending_commands(
    tracks: dict[int, dict[int, dict]],
    pending_commands: list[str],
    default_extend: int,
    vel_window: int,
    frame_keys: list[str],
) -> tuple[dict[int, dict[int, dict]], list[str]]:
    """Return preview tracks/messages after replaying staged commands."""
    preview = copy.deepcopy(tracks)
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
            )
        )
    return preview, messages


def _run_user_refinement_loop(
    refined_results: dict,
    frame_keys: list[str],
    cfg: dict,
    output_dir: str,
    payload_template: dict,
    data_root: str,
    camera_names: list[str],
) -> tuple[dict, dict]:
    """Interactive post-refinement loop with undo and numbered snapshots."""
    user_cfg = cfg.get("user_refinement", {}) or {}
    if not bool(user_cfg.get("enabled", False)):
        return refined_results, {
            "enabled": False,
            "iterations": 0,
            "commands_applied": 0,
        }

    tracks = build_tracks(copy.deepcopy(refined_results))
    history: list[dict[int, dict[int, dict]]] = []
    commands: list[str] = []
    applied_batch_sizes: list[int] = []
    pending_commands: list[str] = []
    command_log_lines: list[str] = []
    vel_window = int(user_cfg.get("extend_velocity_window", cfg.get("extend_velocity_window", 3)))
    default_extend = int(user_cfg.get("default_extend", 5))
    project_cfg = cfg.get("project", {}) or {}

    root_dir = os.path.join(output_dir, str(user_cfg.get("output_dir", "user_refinement")))
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
    start_idx = (max(existing_iters) + 1) if existing_iters else 0
    iter_idx = start_idx
    first_dir = _write_user_iteration(
        root_dir=root_dir,
        base_payload=payload_template,
        tracks=tracks,
        frame_keys=frame_keys,
        iteration_idx=iter_idx,
        command="initial",
        data_root=data_root,
        camera_names=camera_names,
        project_cfg=project_cfg,
    )

    print("\n🧭 Interactive user refinement enabled.")
    print(f"📂 Iteration {iter_idx:03d} saved to: {first_dir}")
    print("Commands:")
    print("  Filter: <class>_<id>")
    print("  Fuse: <class>_<id>, <class>_<id> [, ...]")
    print(f"  Extend: <class>_<id>[, <int>] (default int={default_extend})")
    print("  Extend_stopped: <class>_<id>, <int>")
    print("  Remove: <class>_<id>, <int> (positive=end, negative=start)")
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
                tracks = history.pop()
                if applied_batch_sizes:
                    n_drop = applied_batch_sizes.pop()
                    if n_drop > 0:
                        del commands[-n_drop:]
                iter_idx += 1
                out_dir = _write_user_iteration(
                    root_dir=root_dir,
                    base_payload=payload_template,
                    tracks=tracks,
                    frame_keys=frame_keys,
                    iteration_idx=iter_idx,
                    command="undo",
                    data_root=data_root,
                    camera_names=camera_names,
                    project_cfg=project_cfg,
                )
                command_log_lines.append(f"iter {iter_idx:03d} | undo")
                print(f"↩️ Undo applied. Snapshot: {out_dir}")
                continue

            if cmd == "apply":
                if not pending_commands:
                    print("⚠️ No staged commands to apply.")
                    continue
                preview_tracks, _ = _replay_pending_commands(
                    tracks,
                    pending_commands,
                    default_extend,
                    vel_window,
                    frame_keys,
                )
                history.append(copy.deepcopy(tracks))
                tracks = preview_tracks
                commands.extend(pending_commands)
                applied_batch = list(pending_commands)
                applied_batch_sizes.append(len(applied_batch))
                pending_commands.clear()
                iter_idx += 1
                out_dir = _write_user_iteration(
                    root_dir=root_dir,
                    base_payload=payload_template,
                    tracks=tracks,
                    frame_keys=frame_keys,
                    iteration_idx=iter_idx,
                    command="apply | " + " ; ".join(applied_batch),
                    data_root=data_root,
                    camera_names=camera_names,
                    project_cfg=project_cfg,
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

            preview_tracks, _ = _replay_pending_commands(
                tracks,
                pending_commands,
                default_extend,
                vel_window,
                frame_keys,
            )
            stage_msg = _apply_user_command(
                preview_tracks,
                cmd,
                args,
                default_extend,
                vel_window,
                frame_keys,
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

    return final_results, summary


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


def refine(results: dict, cfg: dict) -> tuple[dict, dict]:
    """Run fusion then filtering. Returns (refined_results, report).

    Operates on a deep copy so the caller's ``results`` (and its box dicts) are
    never mutated \u2014 fusion rewrites ``tracking_id``/``tracking_name`` in place.
    """
    results = copy.deepcopy(results)
    frame_keys, _ = _frame_index(results)
    tracks = build_tracks(results)
    n_in = len(tracks)

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

    # Interpolate interior frame gaps so each kept track is continuous within its
    # lifespan (the tracker frequently drops single frames). Done after filtering
    # so we never waste work on tracks that get dropped.
    n_filled = 0
    filled_details: list[dict] = []
    if bool(cfg.get("fill_gaps", True)):
        n_filled, filled_details = fill_track_gaps(kept, frame_keys)

    # Extrapolate a bounded number of frames past each track's ends to cover the
    # fisheye FOV edges / far-away dropouts the (narrower-FOV) tracker misses.
    n_extended = 0
    extended_details: list[dict] = []
    if int(cfg.get("extend_frames", 0)) > 0:
        n_extended, extended_details = extend_track_ends(
            kept,
            frame_keys,
            extend_frames=int(cfg.get("extend_frames", 0)),
            vel_window=int(cfg.get("extend_velocity_window", 3)),
        )

    # Low-pass the (now dense) per-frame translations to remove tracker jitter,
    # which otherwise shows up as shaky rigid-object motion in the 4DGS render.
    n_smoothed = 0
    if int(cfg.get("pose_smooth_window", 0)) > 1:
        n_smoothed = smooth_track_translations(
            kept, window=int(cfg.get("pose_smooth_window", 0))
        )

    # Optional yaw-only pose smoothing. Can either smooth the raw box yaw, or
    # derive yaw from the local tangent of the already-smoothed translation path
    # so headings stay stable while still following curved trajectories.
    n_rot_smoothed = 0
    if bool(cfg.get("pose_smooth_rotation", False)):
        n_rot_smoothed = smooth_track_rotations(
            kept,
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

    refined_results, report = refine(results, cfg.refine_task)

    # Optional interactive user edits (filter/fuse/extend/undo) after the
    # automatic refinement. Writes numbered snapshots under user_refinement/.
    frame_keys, _ = _frame_index(results)
    data_root = cfg.refine_task.get("data_root", "") or cfg.dataset.base_dir
    cameras = list(cfg.refine_task.get("project", {}).get("cameras", []) or []) or list(cfg.dataset.cameras)
    refined_results, user_report = _run_user_refinement_loop(
        refined_results=refined_results,
        frame_keys=frame_keys,
        cfg=cfg.refine_task,
        output_dir=output_dir,
        payload_template=dict(data),
        data_root=to_absolute_path(data_root),
        camera_names=cameras,
    )
    report["user_refinement"] = user_report

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
    )
    with open(os.path.join(output_dir, ".success"), "w", encoding="utf-8") as f:
        f.write("Refinement finished successfully.")


if __name__ == "__main__":
    main()
