"""COLMAP <-> training-frame alignment for dynamic-object annotations.

Tracking boxes are exported in the raw COLMAP world frame, while the gsplat
trainer optimizes in a normalized "scene" frame produced by
:class:`NCoreParser`. This module recovers the similarity transform
``(s, R, t)`` mapping COLMAP world -> training frame by matching camera centers,
so box centers/orientations/sizes can be expressed in the trainer's coordinates.

Matching strategy
-----------------
The NCore converter rewrites frame timestamps to synthetic values
(``frame_index * 1e6``), so timestamps are NOT directly comparable to the real
capture timestamps embedded in COLMAP image filenames. However, both encode the
same rig captures in the same temporal order. We therefore pair frames by
*ordinal rank within each camera*: sort the parser's frames by their synthetic
timestamp and the COLMAP images by their real timestamp, then match rank-by-rank.
The recovered transform's fit residual is asserted to be near zero, which
validates the correspondence.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# COLMAP loading (robust across pycolmap API variants)
# ---------------------------------------------------------------------------
def load_colmap_images(sparse_dir: str) -> list:
    """Return a list of COLMAP image objects across pycolmap API variants."""
    import pycolmap

    if hasattr(pycolmap, "Reconstruction"):
        rec = pycolmap.Reconstruction(sparse_dir)
        return list(rec.images.values())
    manager = pycolmap.SceneManager(sparse_dir)
    manager.load_cameras()
    manager.load_images()
    return list(manager.images.values())


def resolve_colmap_sparse_dir(data_root: str) -> str:
    """Locate the COLMAP sparse reconstruction under a scene root."""
    candidates = (
        Path(data_root) / "colmap_sparse" / "rig",
        Path(data_root) / "sparse" / "0",
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"Could not find COLMAP sparse data under {data_root} "
        f"(looked for colmap_sparse/rig and sparse/0)."
    )


def colmap_camera_center(image) -> np.ndarray:
    """Return the camera center in world frame, robust across pycolmap APIs."""
    bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    if hasattr(image, "cam_from_world"):
        world_to_cam = np.concatenate(
            [np.asarray(image.cam_from_world().matrix(), dtype=np.float64), bottom],
            axis=0,
        )
    else:
        rotation = np.asarray(image.R(), dtype=np.float64)
        translation = np.asarray(image.tvec, dtype=np.float64).reshape(3, 1)
        world_to_cam = np.concatenate(
            [np.concatenate([rotation, translation], axis=1), bottom], axis=0
        )
    cam_to_world = np.linalg.inv(world_to_cam)
    return cam_to_world[:3, 3]


# ---------------------------------------------------------------------------
# Similarity estimation
# ---------------------------------------------------------------------------
def umeyama_similarity(
    src: np.ndarray, dst: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity ``s, R, t`` with ``dst ~= s * R @ src + t``.

    Implements Umeyama (1991). ``src`` and ``dst`` are ``(N, 3)`` arrays of
    corresponding points. Returns scale (float), rotation ``(3, 3)`` and
    translation ``(3,)``.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    var_src = (src_c**2).sum() / n
    scale = float(np.trace(np.diag(D) @ S) / var_src)
    t = mu_dst - scale * R @ mu_src
    return scale, R, t


# ---------------------------------------------------------------------------
# COLMAP -> training-frame transform
# ---------------------------------------------------------------------------
class SimilarityTransform:
    """A similarity transform ``x' = s * R @ x + t`` with helpers for boxes."""

    def __init__(self, scale: float, rotation: np.ndarray, translation: np.ndarray):
        self.scale = float(scale)
        self.rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        self.translation = np.asarray(translation, dtype=np.float64).reshape(3)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Apply to ``(N, 3)`` points."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return (self.scale * (self.rotation @ points.T).T) + self.translation

    def transform_box_rotation(self, box_rotation: np.ndarray) -> np.ndarray:
        """Rotate a box orientation matrix ``(3, 3)`` into the training frame."""
        return self.rotation @ np.asarray(box_rotation, dtype=np.float64).reshape(3, 3)


def _camera_id_to_int(camera_id: str) -> int:
    """Map a parser camera id like ``"camera10"`` to its integer ``10``."""
    digits = "".join(ch for ch in str(camera_id) if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot extract integer camera id from '{camera_id}'.")
    return int(digits)


def compute_colmap_to_training(
    parser,
    colmap_sparse_dir: str,
    residual_rel_tol: float = 1e-2,
) -> SimilarityTransform:
    """Recover the COLMAP-world -> training-frame similarity transform.

    Args:
        parser: An ``NCoreParser`` exposing ``camera_ids``, ``frame_list``,
            ``frame_timestamps_us``, ``camtoworlds`` and ``scene_scale``.
        colmap_sparse_dir: Path to the COLMAP sparse reconstruction directory.
        residual_rel_tol: Maximum allowed mean fit residual relative to
            ``scene_scale``. Exceeding it raises, signalling a bad match.

    Returns:
        A :class:`SimilarityTransform` mapping COLMAP world -> training frame.
    """
    # Parser frames grouped per integer camera id, with their training centers.
    parser_by_cam: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    for idx, ((camera_id, _frame_idx), ts) in enumerate(
        zip(parser.frame_list, parser.frame_timestamps_us)
    ):
        cam_int = _camera_id_to_int(camera_id)
        parser_by_cam[cam_int].append((int(ts), parser.camtoworlds[idx][:3, 3]))

    # COLMAP images grouped per integer camera id, with their world centers.
    images = load_colmap_images(colmap_sparse_dir)
    colmap_by_cam: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    for image in images:
        ts = int(Path(image.name).stem)
        colmap_by_cam[int(image.camera_id)].append((ts, colmap_camera_center(image)))

    # Pair by ordinal rank within each shared camera.
    src_pts: List[np.ndarray] = []  # COLMAP centers (source)
    dst_pts: List[np.ndarray] = []  # training centers (target)
    for cam_int, parser_frames in parser_by_cam.items():
        if cam_int not in colmap_by_cam:
            continue
        colmap_frames = colmap_by_cam[cam_int]
        if len(parser_frames) != len(colmap_frames):
            raise ValueError(
                f"Camera {cam_int}: parser has {len(parser_frames)} frames but "
                f"COLMAP has {len(colmap_frames)}; cannot align by rank."
            )
        parser_sorted = sorted(parser_frames, key=lambda x: x[0])
        colmap_sorted = sorted(colmap_frames, key=lambda x: x[0])
        for (_, train_center), (_, colmap_center) in zip(parser_sorted, colmap_sorted):
            dst_pts.append(train_center)
            src_pts.append(colmap_center)

    if len(src_pts) < 3:
        raise ValueError(
            f"Only {len(src_pts)} camera-center correspondences found between "
            f"parser and COLMAP; need >= 3 for similarity alignment."
        )

    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    scale, rotation, translation = umeyama_similarity(src, dst)

    # Validate the fit: residual should be ~float noise for a correct match.
    pred = (scale * (rotation @ src.T).T) + translation
    residual = float(np.linalg.norm(pred - dst, axis=1).mean())
    scene_scale = float(getattr(parser, "scene_scale", 1.0)) or 1.0
    rel = residual / scene_scale
    print(
        f"[dynamic.alignment] COLMAP->training similarity: scale={scale:.6f}, "
        f"N={len(src)} centers, residual={residual:.3e} "
        f"({rel:.2e} x scene_scale)"
    )
    if rel > residual_rel_tol:
        raise RuntimeError(
            f"COLMAP->training alignment residual too large: {residual:.3e} "
            f"({rel:.2e} x scene_scale > {residual_rel_tol:.2e}). "
            f"Camera-center correspondence is likely wrong."
        )
    return SimilarityTransform(scale, rotation, translation)
