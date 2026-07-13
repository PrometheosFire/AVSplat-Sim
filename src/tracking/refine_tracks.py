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
        vel = _endpoint_velocity_3d(track, frames, at_start=False, window=vel_window)
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
    vel = _endpoint_velocity_3d(track, frames, at_start=True, window=vel_window)
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
            amount = int(parts[1])
        tid, cls = _validate_selector(tracks, selector)
        added = _extend_single_track(
            track=tracks[tid],
            frame_keys=frame_keys,
            amount=amount,
            vel_window=vel_window,
        )
        return f"↕️ Staged extend for {cls}_{tid} by {amount}; adds {added} frames."

    raise ValueError("Unknown command. Use Filter/Fuse/Extend/apply/undo/done.")


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
        if ":" in raw:
            head, tail = raw.split(":", 1)
            cmd = head.strip().lower()
            args = tail.strip()
        else:
            parts = raw.strip().split(None, 1)
            cmd = parts[0].strip().lower()
            args = parts[1].strip() if len(parts) > 1 else ""
        messages.append(
            _apply_user_command(preview, cmd, args, default_extend, vel_window, frame_keys)
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
    print("  apply")
    print("  undo")
    print("  done")

    while True:
        raw = input("user-refine> ").strip()
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

            if cmd not in {"filter", "fuse", "extend"}:
                raise ValueError(
                    "Unknown command. Use Filter/Fuse/Extend/apply/undo/done."
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
    )
    with open(os.path.join(output_dir, ".success"), "w", encoding="utf-8") as f:
        f.write("Refinement finished successfully.")


if __name__ == "__main__":
    main()
