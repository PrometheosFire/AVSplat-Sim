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
import os
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
    # Optional per-instance kinematic bicycle parameters (COLMAP frame) fitted at
    # refinement time (Step 1.5), keyed by instance column ``m``. Present only
    # when a ``bicycle_params.json`` sidecar sits next to the tracks JSON. Used
    # to *extrapolate* poses past the observed span (simulator/renderer).
    bicycle_params: Optional[Dict[int, dict]] = None
    # Components of the COLMAP -> training similarity transform, kept so the
    # bicycle rollout (which runs in COLMAP world) can be mapped to the training
    # frame identically to the baked poses. ``None`` when no sidecar was loaded.
    transform_scale: Optional[float] = None
    transform_rotation: Optional[np.ndarray] = None  # (3, 3)
    transform_translation: Optional[np.ndarray] = None  # (3,)

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


# ---------------------------------------------------------------------------
# Kinematic bicycle extrapolation (COLMAP rollout -> training pose)
# ---------------------------------------------------------------------------
# Sidecar written next to ``track_3d_refined_colmap.json`` by the refinement step.
BICYCLE_SIDECAR_NAME = "bicycle_params.json"

# Ground-plane / up axes in COLMAP world (matches src/tracking/bicycle_kinematics).
_GROUND_AXES = (0, 2)
_UP_AXIS = 1


def _yaw_from_rotation_colmap(rot: np.ndarray) -> float:
    """Ground-plane yaw of a COLMAP box rotation (its local +x axis).

    Mirrors ``src.tracking.refine_tracks._yaw_from_box_rotation``: heading is the
    (x, z) projection of the box's local +x axis, ``yaw = atan2(z, x)``.
    """
    forward = np.asarray(rot, dtype=np.float64)[:, 0]
    heading = np.array([forward[0], forward[2]], dtype=np.float64)
    norm = float(np.linalg.norm(heading))
    if norm <= 1e-12:
        return 0.0
    return float(np.arctan2(heading[1] / norm, heading[0] / norm))


def _yaw_matrix_colmap(yaw: float) -> np.ndarray:
    """Ground-plane yaw rotation matrix consistent with ``_yaw_from_rotation_colmap``.

    Mirrors ``src.tracking.refine_tracks._yaw_matrix``: heading is ``atan2(dz, dx)``
    of the box's local +x axis, so a box at heading ``yaw`` has local +x equal to
    ``[cos yaw, 0, sin yaw]`` (a rotation of ``-yaw`` about world +y). The round
    trip is exact: ``_yaw_from_rotation_colmap(_yaw_matrix_colmap(yaw)) == yaw``.
    """
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=np.float64)


def bicycle_pose_at_frame(
    entry: dict,
    transform_scale: float,
    transform_rotation: np.ndarray,
    transform_translation: np.ndarray,
    frame_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extrapolate a fitted bicycle model to a global frame, in the training frame.

    Rolls the kinematic bicycle model out in COLMAP world (the frame it was
    fitted in), then maps the resulting box pose to the training frame with the
    same similarity transform used for the baked poses. Frames inside the fitted
    span reproduce the fit; frames past the end continue with zero acceleration
    and the last steering angle; frames before the start integrate backwards.

    Args:
        entry: Per-instance params — ``wheelbase``, ``lr_ratio``, ``dt``,
            ``first_frame``, ``n_steps``, ``state0`` (4,), ``accel`` (n,),
            ``steer`` (n,), plus reference height ``ref_y`` and rotation
            ``ref_rot`` (3x3, COLMAP) taken from the last observed frame.
        transform_scale/rotation/translation: COLMAP -> training similarity
            (``x' = s * R @ x + t``).
        frame_idx: Global (training) frame index to evaluate.

    Returns:
        ``(trans (3,), quat_wxyz (4,))`` in the training frame.
    """
    from src.tracking.bicycle_kinematics import rollout as _roll

    wheelbase = float(entry["wheelbase"])
    lr_ratio = float(entry["lr_ratio"])
    dt = float(entry["dt"])
    first = int(entry["first_frame"])
    n_steps = int(entry["n_steps"])
    state0 = np.asarray(entry["state0"], dtype=np.float64)
    accel = np.asarray(entry["accel"], dtype=np.float64)
    steer = np.asarray(entry["steer"], dtype=np.float64)

    # States across the fitted span: shape (n_steps + 1, 4).
    states = _roll(state0, accel, steer, dt, wheelbase, lr_ratio)
    offset = int(frame_idx) - first

    if 0 <= offset <= n_steps:
        st = states[offset]
    elif offset > n_steps:
        extra = offset - n_steps
        last_steer = float(steer[-1]) if steer.size else 0.0
        ext = _roll(
            states[-1],
            np.zeros(extra),
            np.full(extra, last_steer),
            dt,
            wheelbase,
            lr_ratio,
        )
        st = ext[extra]
    else:  # offset < 0 -> integrate backwards from the first state
        back = int(-offset)
        first_steer = float(steer[0]) if steer.size else 0.0
        rev = _roll(
            states[0],
            np.zeros(back),
            np.full(back, first_steer),
            -dt,
            wheelbase,
            lr_ratio,
        )
        st = rev[back]

    # Build the COLMAP box pose: planar (x, z) from the rollout, height held at
    # the reference, heading rotated by the yaw delta about the up axis.
    ref_rot = np.asarray(entry["ref_rot"], dtype=np.float64).reshape(3, 3)
    ref_y = float(entry["ref_y"])
    base_yaw = _yaw_from_rotation_colmap(ref_rot)

    center_colmap = np.zeros(3, dtype=np.float64)
    center_colmap[_GROUND_AXES[0]] = st[0]
    center_colmap[_GROUND_AXES[1]] = st[1]
    center_colmap[_UP_AXIS] = ref_y
    rot_colmap = _yaw_matrix_colmap(float(st[2]) - base_yaw) @ ref_rot

    # Map to training frame identically to the baked poses.
    R = np.asarray(transform_rotation, dtype=np.float64).reshape(3, 3)
    t = np.asarray(transform_translation, dtype=np.float64).reshape(3)
    center_train = float(transform_scale) * (R @ center_colmap) + t
    rot_train = R @ rot_colmap
    return center_train.astype(np.float64), matrix_to_quat_wxyz(rot_train)


def _load_bicycle_sidecar(tracks_json_path: str) -> Optional[dict]:
    """Load the ``bicycle_params.json`` sidecar next to the tracks JSON, if any."""
    sidecar = os.path.join(os.path.dirname(tracks_json_path), BICYCLE_SIDECAR_NAME)
    if not os.path.isfile(sidecar):
        return None
    with open(sidecar) as f:
        return json.load(f)


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
    id_to_sizes: Dict[int, list] = {}  # list of (3,) size arrays per instance
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
            if tid not in id_to_sizes:
                id_to_sizes[tid] = []
            id_to_sizes[tid].append(size)
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
    # Per-instance size = median box size over its observations, scaled to training.
    # Median is more robust than mean to tracker frames with abnormally large/small
    # box estimates, preventing any single bad frame from inflating the box size.
    sizes = np.stack(
        [np.median(np.stack(id_to_sizes[tid], axis=0), axis=0) for tid in instance_ids]
    )
    sizes = (sizes * transform.scale).astype(np.float32)

    # Allocate per-frame pose arrays.
    trans = np.zeros((num_frames, num_inst, 3), dtype=np.float32)
    quats = np.zeros((num_frames, num_inst, 4), dtype=np.float32)
    quats[..., 0] = 1.0  # identity quaternion default
    valid = np.zeros((num_frames, num_inst), dtype=bool)

    # Per-instance reference pose (COLMAP): the last observed frame's up-axis
    # height and rotation matrix, used to anchor bicycle extrapolation (which
    # only models planar x/z/heading). Keyed by column -> (frame_idx, y, rot).
    ref_pose: Dict[int, Tuple[int, float, np.ndarray]] = {}

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
            prev = ref_pose.get(col)
            if prev is None or frame_idx >= prev[0]:
                ref_pose[col] = (
                    frame_idx,
                    float(center_colmap[_UP_AXIS]),
                    rot_colmap.copy(),
                )

    # Optional: attach fitted kinematic-bicycle params for extrapolation.
    bicycle_params, tf_scale, tf_rot, tf_trans = _build_bicycle_params(
        tracks_json_path, track_tokens, id_to_col, ref_pose, transform
    )

    return RigidTracks(
        instance_ids=instance_ids,
        class_names=class_names,
        sizes=sizes,
        trans=trans,
        quats=quats,
        valid=valid,
        frame_timestamps_us=sorted_ts,
        bicycle_params=bicycle_params,
        transform_scale=tf_scale,
        transform_rotation=tf_rot,
        transform_translation=tf_trans,
    )


def _build_bicycle_params(
    tracks_json_path: str,
    track_tokens: List[str],
    id_to_col: Dict[int, int],
    ref_pose: Dict[int, Tuple[int, float, np.ndarray]],
    transform: SimilarityTransform,
) -> Tuple[
    Optional[Dict[int, dict]], Optional[float], Optional[np.ndarray], Optional[np.ndarray]
]:
    """Read the bicycle sidecar and index its params by instance column.

    Returns ``(params_by_col, scale, R, t)`` or ``(None, None, None, None)`` when
    no sidecar is present or its frame ordering does not match this run.
    """
    sidecar = _load_bicycle_sidecar(tracks_json_path)
    if not sidecar or not sidecar.get("tracks"):
        return None, None, None, None

    # The sidecar's frame ordering must match this run's track ordering, else the
    # per-instance ``first_frame`` / ``n_steps`` indices are meaningless here.
    frame_keys = sidecar.get("frame_keys")
    if frame_keys is not None and list(frame_keys) != list(track_tokens):
        return None, None, None, None

    params_by_col: Dict[int, dict] = {}
    for tid_str, raw in sidecar["tracks"].items():
        tid = int(tid_str)
        col = id_to_col.get(tid)
        if col is None:
            continue
        ref = ref_pose.get(col)
        if ref is None:
            continue
        params_by_col[col] = {
            "wheelbase": float(raw["wheelbase"]),
            "lr_ratio": float(raw["lr_ratio"]),
            "dt": float(raw["dt"]),
            "first_frame": int(raw["first_frame"]),
            "n_steps": int(raw["n_steps"]),
            "state0": np.asarray(raw["state0"], dtype=np.float64),
            "accel": np.asarray(raw["accel"], dtype=np.float64),
            "steer": np.asarray(raw["steer"], dtype=np.float64),
            "ref_y": ref[1],
            "ref_rot": ref[2],
        }

    if not params_by_col:
        return None, None, None, None

    return (
        params_by_col,
        float(transform.scale),
        np.asarray(transform.rotation, dtype=np.float64).reshape(3, 3),
        np.asarray(transform.translation, dtype=np.float64).reshape(3),
    )
