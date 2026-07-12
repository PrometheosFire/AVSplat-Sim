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
    tracks: dict[int, dict[int, dict]], mapping: dict[int, int]
) -> dict[int, dict[int, dict]]:
    """Merge tracks per ``mapping``; on shared frames keep the higher score."""
    merged: dict[int, dict[int, dict]] = defaultdict(dict)
    for tid, track in tracks.items():
        canonical = mapping[tid]
        for fidx, box in track.items():
            existing = merged[canonical].get(fidx)
            if existing is None or float(box["tracking_score"]) > float(
                existing["tracking_score"]
            ):
                merged[canonical][fidx] = box
    # Unify id + class label across each merged track.
    for canonical, track in merged.items():
        name = _track_class(track)
        for box in track.values():
            box["tracking_id"] = int(canonical)
            box["tracking_name"] = name
    return merged


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
) -> int:
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
    return n_filled


# --------------------------------------------------------------------------- #
# End extension (extrapolate a bounded number of frames past each track's ends)
# --------------------------------------------------------------------------- #
def _endpoint_velocity_3d(track, frames, at_start: bool, window: int) -> np.ndarray:
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
) -> int:
    """Extrapolate up to ``extend_frames`` frames before/after each track, in place.

    The tracker runs on undistorted crops with a narrower FOV than the fisheye
    cameras, so objects entering/leaving frame (or getting far) stop being
    detected while still visible in the training images. Extend each track with
    constant ground-plane velocity (rotation and size held), bounded by
    ``extend_frames`` and the clip range. Returns the number of frames added.
    """
    if extend_frames <= 0:
        return 0
    n_keys = len(frame_keys)
    n_ext = 0
    for track in tracks.values():
        frames = sorted(track)
        if len(frames) < 2:
            continue
        first, last = frames[0], frames[-1]

        # Forward from the last observed frame.
        v_end = _endpoint_velocity_3d(track, frames, at_start=False, window=vel_window)
        base_box = track[last]
        base_t = np.asarray(base_box["translation"], dtype=np.float64)
        for k in range(1, extend_frames + 1):
            f = last + k
            if f >= n_keys or f in track:
                break
            token = frame_keys[f] if 0 <= f < n_keys else None
            track[f] = _extrap_box(base_box, base_t + v_end * k, token)
            n_ext += 1

        # Backward from the first observed frame.
        v_start = _endpoint_velocity_3d(track, frames, at_start=True, window=vel_window)
        base_box = track[first]
        base_t = np.asarray(base_box["translation"], dtype=np.float64)
        for k in range(1, extend_frames + 1):
            f = first - k
            if f < 0 or f in track:
                break
            token = frame_keys[f] if 0 <= f < n_keys else None
            track[f] = _extrap_box(base_box, base_t - v_start * k, token)
            n_ext += 1
    return n_ext


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


def refine(results: dict, cfg: dict) -> tuple[dict, dict]:
    """Run fusion then filtering. Returns (refined_results, report).

    Operates on a deep copy so the caller's ``results`` (and its box dicts) are
    never mutated \u2014 fusion rewrites ``tracking_id``/``tracking_name`` in place.
    """
    import copy

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
    fused = apply_fusion(tracks, mapping)
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
    if bool(cfg.get("fill_gaps", True)):
        n_filled = fill_track_gaps(kept, frame_keys)

    # Extrapolate a bounded number of frames past each track's ends to cover the
    # fisheye FOV edges / far-away dropouts the (narrower-FOV) tracker misses.
    n_extended = 0
    if int(cfg.get("extend_frames", 0)) > 0:
        n_extended = extend_track_ends(
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

    report = {
        "tracks_in": n_in,
        "tracks_after_fusion": n_fused,
        "merges": n_in - n_fused,
        "tracks_kept": len(kept),
        "tracks_dropped": len(dropped),
        "dropped_static": sum(1 for d in dropped if d["reason"] == "static"),
        "dropped_short": sum(1 for d in dropped if d["reason"] == "short"),
        "frames_filled": n_filled,
        "frames_extended": n_extended,
        "frames_smoothed": n_smoothed,
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
    )
    with open(os.path.join(output_dir, ".success"), "w", encoding="utf-8") as f:
        f.write("Refinement finished successfully.")


if __name__ == "__main__":
    main()
