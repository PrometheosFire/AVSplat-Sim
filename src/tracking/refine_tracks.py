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
                # mean down) but repeatedly coincide. Distinct vehicles never get
                # within merge_dist, so this stays safe.
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
                # for a long co-present stretch are the same object split by
                # depth error (distinct vehicles drift apart). Requires a long
                # overlap so brief coincidences of real neighbors don't trigger.
                persistent = (
                    len(shared) >= persistent_overlap_frames
                    and float(np.median(shared_dists)) <= persistent_merge_dist
                )
                if (
                    float(np.mean(ious)) >= iou_thr
                    or min_centroid <= merge_dist
                    or persistent
                ):
                    uf.union(a, b)
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
                uf.union(a, b)

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
) -> tuple[dict[int, dict[int, dict]], list[dict]]:
    """Drop static / too-short tracks. Returns (kept, dropped_report)."""
    kept: dict[int, dict[int, dict]] = {}
    dropped: list[dict] = []
    for tid, track in tracks.items():
        disp = track_displacement(
            track, displacement_percentile, displacement_mode, displacement_smooth
        )
        if len(track) < min_track_length:
            dropped.append({"id": tid, "reason": "short", "frames": len(track), "disp": disp})
        elif disp < min_displacement:
            dropped.append({"id": tid, "reason": "static", "frames": len(track), "disp": disp})
        else:
            kept[tid] = track
    return kept, dropped


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
    )

    report = {
        "tracks_in": n_in,
        "tracks_after_fusion": n_fused,
        "merges": n_in - n_fused,
        "tracks_kept": len(kept),
        "tracks_dropped": len(dropped),
        "dropped_static": sum(1 for d in dropped if d["reason"] == "static"),
        "dropped_short": sum(1 for d in dropped if d["reason"] == "short"),
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
        f"(dropped {report['dropped_static']} static, {report['dropped_short']} short)"
    )
    with open(os.path.join(output_dir, ".success"), "w", encoding="utf-8") as f:
        f.write("Refinement finished successfully.")


if __name__ == "__main__":
    main()
