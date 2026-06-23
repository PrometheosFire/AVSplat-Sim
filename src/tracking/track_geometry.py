"""Geometry helpers shared by track refinement and visualization.

All boxes are assumed to be in the COLMAP world frame (x-left, y-up,
z-forward), as written by the tracker's ``*_colmap.json`` export. The bird's
eye (ground) plane is therefore ``(x, z)`` and the up axis is ``y``.
"""
from __future__ import annotations

import numpy as np

# COLMAP world: index 1 (y) is up; the ground/BEV plane is (x, z).
GROUND_AXES = (0, 2)
UP_AXIS = 1


def quat_wxyz_to_matrix(quat: list[float] | np.ndarray) -> np.ndarray:
    """Convert a [w, x, y, z] quaternion to a 3x3 rotation matrix."""
    w, x, y, z = (float(v) for v in quat)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def box_corners_3d(box: dict) -> np.ndarray:
    """Return the 8 world-frame corners of a box. Shape (8, 3).

    Size is ``[w, l, h]`` and, following the Vis4D box convention, the local
    axes are: length (l) along local x, width (w) along local y, height (h)
    along local z. The world frame is whatever the box translation/rotation
    are expressed in (COLMAP here).
    """
    t = np.asarray(box["translation"], dtype=np.float64)
    w, l, h = (float(v) for v in box["size"])
    rot = quat_wxyz_to_matrix(box["rotation"])

    xc = np.array([l, l, -l, -l, l, l, -l, -l]) * 0.5  # length -> local x
    yc = np.array([-w, w, w, -w, -w, w, w, -w]) * 0.5  # width  -> local y
    zc = np.array([-h, -h, -h, -h, h, h, h, h]) * 0.5  # height -> local z
    local = np.stack([xc, yc, zc], axis=1)  # (8, 3)
    return (rot @ local.T).T + t


def box_footprint(box: dict) -> np.ndarray:
    """Return the ground-plane footprint polygon of a box. Shape (M, 2).

    Projects the 8 world corners onto the ground plane ``(x, z)`` and returns
    their convex hull as a float32 polygon suitable for cv2 area/intersection.
    """
    import cv2

    corners = box_corners_3d(box)[:, GROUND_AXES].astype(np.float32)
    hull = cv2.convexHull(corners)
    return hull.reshape(-1, 2)


def polygon_area(poly: np.ndarray) -> float:
    """Area of a 2D polygon via the shoelace formula."""
    p = np.asarray(poly, dtype=np.float64)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def bev_iou(box_a: dict, box_b: dict) -> float:
    """Bird's eye view IoU between two boxes' ground footprints."""
    import cv2

    poly_a = box_footprint(box_a)
    poly_b = box_footprint(box_b)
    area_a = polygon_area(poly_a)
    area_b = polygon_area(poly_b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(poly_a, poly_b)
    inter = float(max(inter, 0.0))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def box_center_ground(box: dict) -> np.ndarray:
    """Return the box center on the ground plane (x, z). Shape (2,)."""
    t = np.asarray(box["translation"], dtype=np.float64)
    return t[list(GROUND_AXES)]
