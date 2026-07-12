"""Load refined 3D vehicle tracks into per-instance, per-frame rigid poses.

Consumes the tracker's ``track_3d_refined_colmap.json`` (boxes in COLMAP world)
and produces arrays the trainer can use directly: for every rigid instance and
every global frame, a pose ``(quaternion, translation)`` expressed in the
trainer's normalized frame, plus a validity mask and per-instance box sizes.

The scene similarity transform (COLMAP -> training) is baked in: scale is folded
into the box sizes and translations so each per-frame pose is a pure ``SE(3)``
rotation+translation, matching the OmniRe rigid-node formulation
``x_world = R[f] @ x_local + t[f]``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .alignment import SimilarityTransform

# Default rigid classes: the vehicle group (excludes pedestrians, bicycles, etc.)
DEFAULT_RIGID_CLASSES: Tuple[str, ...] = (
    "car",
    "truck",
    "bus",
)


# ---------------------------------------------------------------------------
# Quaternion helpers (wxyz convention, matching src/tracking/track_geometry.py)
# ---------------------------------------------------------------------------
def quat_wxyz_to_matrix(quat: Sequence[float]) -> np.ndarray:
    """Convert a ``[w, x, y, z]`` quaternion to a ``3x3`` rotation matrix."""
    w, x, y, z = (float(v) for v in quat)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a ``3x3`` rotation matrix to a ``[w, x, y, z]`` quaternion."""
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        r = np.sqrt(1.0 + trace)
        w = 0.5 * r
        f = 0.5 / r
        x = (m[2, 1] - m[1, 2]) * f
        y = (m[0, 2] - m[2, 0]) * f
        z = (m[1, 0] - m[0, 1]) * f
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        if i == 0:
            r = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            x = 0.5 * r
            f = 0.5 / r
            w = (m[2, 1] - m[1, 2]) * f
            y = (m[0, 1] + m[1, 0]) * f
            z = (m[0, 2] + m[2, 0]) * f
        elif i == 1:
            r = np.sqrt(1.0 - m[0, 0] + m[1, 1] - m[2, 2])
            y = 0.5 * r
            f = 0.5 / r
            w = (m[0, 2] - m[2, 0]) * f
            x = (m[0, 1] + m[1, 0]) * f
            z = (m[1, 2] + m[2, 1]) * f
        else:
            r = np.sqrt(1.0 - m[0, 0] - m[1, 1] + m[2, 2])
            z = 0.5 * r
            f = 0.5 / r
            w = (m[1, 0] - m[0, 1]) * f
            x = (m[0, 2] + m[2, 0]) * f
            y = (m[1, 2] + m[2, 1]) * f
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    if q[0] < 0:  # canonical hemisphere
        q = -q
    return q


# ---------------------------------------------------------------------------
# Rigid track container
# ---------------------------------------------------------------------------
@dataclass
class RigidTracks:
    """Per-instance, per-frame rigid poses in the trainer's frame.

    Shapes use ``T`` = number of global frames and ``M`` = number of instances.
    """

    instance_ids: List[int]  # length M, original tracking ids
    class_names: List[str]  # length M
    sizes: np.ndarray  # (M, 3) box [l, w, h] in training units
    trans: np.ndarray  # (T, M, 3) translation, training frame
    quats: np.ndarray  # (T, M, 4) wxyz box->world(training)
    valid: np.ndarray  # (T, M) bool, frame-validity per instance
    frame_timestamps_us: np.ndarray  # (T,) sorted unique parser timestamps

    @property
    def num_frames(self) -> int:
        return self.trans.shape[0]

    @property
    def num_instances(self) -> int:
        return self.trans.shape[1]

    def frame_index_from_timestamp(self, timestamp_us: int) -> int:
        """Map a parser (synthetic) timestamp to its global frame index."""
        idx = int(
            np.searchsorted(self.frame_timestamps_us, int(timestamp_us), side="left")
        )
        # Guard against off-by-one from exact-match semantics.
        if idx >= len(self.frame_timestamps_us):
            idx = len(self.frame_timestamps_us) - 1
        return idx


def load_rigid_tracks(
    tracks_json_path: str,
    transform: SimilarityTransform,
    frame_timestamps_us: Sequence[int],
    rigid_classes: Optional[Sequence[str]] = None,
    min_score: float = 0.0,
) -> RigidTracks:
    """Load refined tracks and convert to per-instance rigid poses.

    Args:
        tracks_json_path: Path to ``track_3d_refined_colmap.json``.
        transform: COLMAP -> training similarity transform.
        frame_timestamps_us: The parser's per-frame synthetic timestamps. Their
            sorted-unique values define the global frame ordering ``T``; the
            tracking frames (sorted by their own timestamps) are matched to this
            ordering by rank.
        rigid_classes: Class names to keep. Defaults to the vehicle group.
        min_score: Drop boxes with ``tracking_score`` below this threshold.

    Returns:
        A populated :class:`RigidTracks`.
    """
    rigid_classes = set(rigid_classes or DEFAULT_RIGID_CLASSES)

    with open(tracks_json_path) as f:
        payload = json.load(f)
    results: Dict[str, list] = payload["results"]

    # Global frame ordering from the parser's unique timestamps.
    sorted_ts = np.array(sorted(set(int(t) for t in frame_timestamps_us)), dtype=np.int64)
    num_frames = len(sorted_ts)

    # Tracking frames in temporal order (their real timestamps differ from the
    # parser's synthetic ones, but the ordering corresponds rank-by-rank).
    track_tokens = sorted(results.keys(), key=lambda k: int(k.split("_")[-1]))
    if len(track_tokens) != num_frames:
        raise ValueError(
            f"Frame count mismatch: parser has {num_frames} unique frames but "
            f"tracks JSON has {len(track_tokens)}. Ensure the tracking run and "
            f"the training sequence cover the same frames (no duration/seek "
            f"subsetting)."
        )

    # First pass: collect instances passing the class/score filter.
    id_to_class: Dict[int, str] = {}
    id_to_size_sum: Dict[int, np.ndarray] = {}
    id_to_count: Dict[int, int] = {}
    for token in track_tokens:
        for box in results[token]:
            if box["tracking_name"] not in rigid_classes:
                continue
            if float(box.get("tracking_score", 1.0)) < min_score:
                continue
            tid = int(box["tracking_id"])
            id_to_class.setdefault(tid, box["tracking_name"])
            # Tracker stores size as [width, length, height] (Vis4D convention),
            # with length->local x, width->local y, height->local z (see
            # src/tracking/track_geometry.py). Reorder to [length, width, height]
            # so it matches the local (x, y, z) axes the rigid Gaussians and box
            # wireframes use — otherwise each box is rotated 90 deg about its up
            # axis relative to the vehicle heading.
            size = np.asarray(box["size"], dtype=np.float64)[[1, 0, 2]]
            id_to_size_sum[tid] = id_to_size_sum.get(tid, np.zeros(3)) + size
            id_to_count[tid] = id_to_count.get(tid, 0) + 1

    instance_ids = sorted(id_to_class.keys())
    if not instance_ids:
        raise ValueError(
            f"No rigid instances found in {tracks_json_path} for classes "
            f"{sorted(rigid_classes)}."
        )
    id_to_col = {tid: m for m, tid in enumerate(instance_ids)}
    num_inst = len(instance_ids)

    class_names = [id_to_class[tid] for tid in instance_ids]
    # Per-instance size = mean box size over its observations, scaled to training.
    sizes = np.stack(
        [id_to_size_sum[tid] / max(id_to_count[tid], 1) for tid in instance_ids]
    )
    sizes = (sizes * transform.scale).astype(np.float32)

    # Allocate per-frame pose arrays.
    trans = np.zeros((num_frames, num_inst, 3), dtype=np.float32)
    quats = np.zeros((num_frames, num_inst, 4), dtype=np.float32)
    quats[..., 0] = 1.0  # identity quaternion default
    valid = np.zeros((num_frames, num_inst), dtype=bool)

    # Second pass: fill poses in training frame.
    for frame_idx, token in enumerate(track_tokens):
        for box in results[token]:
            tid = int(box["tracking_id"])
            if tid not in id_to_col:
                continue
            if float(box.get("tracking_score", 1.0)) < min_score:
                continue
            col = id_to_col[tid]
            center_colmap = np.asarray(box["translation"], dtype=np.float64)
            rot_colmap = quat_wxyz_to_matrix(box["rotation"])
            center_train = transform.transform_points(center_colmap)[0]
            rot_train = transform.transform_box_rotation(rot_colmap)
            trans[frame_idx, col] = center_train.astype(np.float32)
            quats[frame_idx, col] = matrix_to_quat_wxyz(rot_train).astype(np.float32)
            valid[frame_idx, col] = True

    return RigidTracks(
        instance_ids=instance_ids,
        class_names=class_names,
        sizes=sizes,
        trans=trans,
        quats=quats,
        valid=valid,
        frame_timestamps_us=sorted_ts,
    )
