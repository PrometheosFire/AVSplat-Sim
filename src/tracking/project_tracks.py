"""Project refined 3D tracks onto the original camera frames for validation.

Boxes live in the raw COLMAP world frame (the tracker's ``*_colmap.json``
export), and the COLMAP camera poses are in that same frame, so projection is
direct: transform a box's world corners into each camera, then project with the
camera's true ``OPENCV_FISHEYE`` model (or a pinhole approximation). Each track
is drawn with a stable per-ID color so identity continuity and switches are
obvious across frames.

Driven by ``cfg.refine_task.project``. Renders annotated frames per camera
(no video). Runnable standalone via Hydra overrides.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.model_wrappers.trackers.wayve101_cc3dt_dataset import (
    _extract_cam_to_world,
    _load_colmap_scene,
    _resolve_sparse_dir,
)
from src.tracking.track_geometry import box_corners_3d

# Cuboid edges from the corner order in box_corners_3d:
#   0-3 bottom face (z=-h/2), 4-7 top face (z=+h/2); +x is the box "front".
_BOTTOM = [(0, 1), (1, 2), (2, 3), (3, 0)]
_TOP = [(4, 5), (5, 6), (6, 7), (7, 4)]
_VERT = [(0, 4), (1, 5), (2, 6), (3, 7)]
_FRONT = [(0, 1), (1, 5), (5, 4), (4, 0)]  # face at +x (length forward)
_EDGES = _BOTTOM + _TOP + _VERT


def _id_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR color for a track id (stable across frames)."""
    rng = (1103515245 * (track_id + 1) + 12345) & 0x7FFFFFFF
    r = 80 + (rng & 0xFF) % 176
    g = 80 + ((rng >> 8) & 0xFF) % 176
    b = 80 + ((rng >> 16) & 0xFF) % 176
    return int(b), int(g), int(r)


def _build_camera_tables(cameras, images, camera_names):
    """Return per-camera intrinsics/distortion and (camera, ts) -> pose maps."""
    cam_calib = {}
    for cid, cam in cameras.items():
        p = np.asarray(cam.params, dtype=np.float64)
        k = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]], dtype=np.float64)
        dist = p[4:8] if p.size >= 8 else np.zeros(4)
        model = getattr(getattr(cam, "model", None), "name", "") or str(
            getattr(cam, "model", "")
        )
        cam_calib[cid] = (k, dist, model, int(cam.width), int(cam.height))

    poses: dict[tuple[str, int], tuple] = {}
    for image in images.values():
        name = Path(image.name).parent.as_posix()
        if camera_names and name not in camera_names:
            continue
        ts = int(Path(image.name).stem)
        poses[(name, ts)] = (image.camera_id, _extract_cam_to_world(image))
    return cam_calib, poses


def _project(points_cam: np.ndarray, k, dist, model: str) -> np.ndarray:
    """Project camera-frame points (N, 3), z>0, to pixels (N, 2)."""
    pts = np.ascontiguousarray(points_cam.reshape(-1, 1, 3), dtype=np.float64)
    if "FISHEYE" in model.upper():
        out, _ = cv2.fisheye.projectPoints(
            pts, np.zeros(3), np.zeros(3), k, dist.reshape(4, 1)
        )
        return out.reshape(-1, 2)
    uv = (k @ points_cam.T).T
    return uv[:, :2] / uv[:, 2:3]


def _draw_edge(img, p0, p1, k, dist, model, color, width, subdiv):
    """Draw one cuboid edge, subdivided so it curves under fisheye."""
    if p0[2] <= 0.05 or p1[2] <= 0.05:
        return  # skip edges crossing/behind the image plane
    ts = np.linspace(0.0, 1.0, subdiv).reshape(-1, 1)
    seg = (1 - ts) * p0 + ts * p1
    uv = _project(seg, k, dist, model).astype(np.int32)
    cv2.polylines(img, [uv.reshape(-1, 1, 2)], False, color, width, cv2.LINE_AA)


def draw_box_on_image(
    img, box, k, dist, model, w, h,
    width=2, subdiv=12, label=True, max_view_angle=80.0,
):
    """Draw a 3D box cuboid (world->camera->image) on ``img``. Returns drawn?"""
    corners_w = box_corners_3d(box)  # (8, 3) in COLMAP world
    cam_id_pose = box["_pose"]
    r_w2c, t_w2c = cam_id_pose
    corners_c = (r_w2c @ corners_w.T).T + t_w2c  # camera frame (z forward)

    if np.count_nonzero(corners_c[:, 2] > 0.05) < 8:
        return False  # require fully in front for a clean cuboid

    # Cull boxes whose corners exceed the lens cone. A very close/wide box has
    # corners near or beyond the fisheye FOV half-angle, where projection folds
    # back and produces frame-spanning garbage. atan2(radial, forward) is the
    # angle of each corner from the optical axis.
    angles = np.degrees(
        np.arctan2(np.linalg.norm(corners_c[:, :2], axis=1), corners_c[:, 2])
    )
    if float(angles.max()) > max_view_angle:
        return False

    color = _id_color(int(box["tracking_id"]))
    for i, j in _EDGES:
        ew = width + 1 if (i, j) in _FRONT else width
        _draw_edge(img, corners_c[i], corners_c[j], k, dist, model, color, ew, subdiv)

    if label:
        front_mid = corners_c[[0, 1, 4, 5]].mean(axis=0, keepdims=True)
        uv = _project(front_mid, k, dist, model)[0]
        if 0 <= uv[0] < w and 0 <= uv[1] < h:
            text = f"{box.get('tracking_name','?')} {int(box['tracking_id'])}"
            cv2.putText(
                img, text, (int(uv[0]), int(uv[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
            )
    return True


def _find_image_file(data_root: str, camera: str, ts: int) -> str | None:
    matches = glob.glob(os.path.join(data_root, "images", camera, f"{ts}.*"))
    return matches[0] if matches else None


def project_tracks(
    results: dict,
    data_root: str,
    output_dir: str,
    camera_names,
    box_width: int = 2,
    subdiv: int = 12,
    max_frames: int | None = None,
    max_view_angle: float = 80.0,
) -> dict:
    """Render boxes onto frames per camera. Returns a small summary report."""
    cameras, images = _load_colmap_scene(_resolve_sparse_dir(data_root))
    cam_calib, poses = _build_camera_tables(cameras, images, camera_names)

    tokens = sorted(results.keys(), key=lambda k: int(k.split("_")[-1]))
    if max_frames:
        step = max(1, len(tokens) // max_frames)
        tokens = tokens[::step]

    drawn_total = 0
    frames_written = 0
    cams = camera_names or sorted({n for (n, _) in poses})

    for token in tokens:
        ts = int(token.split("_")[-1])
        boxes = results[token]
        for camera in cams:
            pose_entry = poses.get((camera, ts))
            img_path = _find_image_file(data_root, camera, ts)
            if pose_entry is None or img_path is None:
                continue
            cam_id, c2w = pose_entry
            k, dist, model, w, h = cam_calib[cam_id]
            w2c = np.linalg.inv(c2w)
            r_w2c, t_w2c = w2c[:3, :3], w2c[:3, 3]

            img = cv2.imread(img_path)
            if img is None:
                continue
            for box in boxes:
                box["_pose"] = (r_w2c, t_w2c)
                if draw_box_on_image(
                    img, box, k, dist, model, w, h,
                    box_width, subdiv, True, max_view_angle,
                ):
                    drawn_total += 1
                box.pop("_pose", None)

            cam_out = os.path.join(output_dir, camera)
            os.makedirs(cam_out, exist_ok=True)
            cv2.imwrite(os.path.join(cam_out, f"{ts}.jpg"), img)
            frames_written += 1

    return {
        "frames_written": frames_written,
        "boxes_drawn": drawn_total,
        "cameras": cams,
        "tokens_rendered": len(tokens),
    }


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== AVSplat-Sim: Project Tracks onto Frames ===")
    proj = cfg.refine_task.get("project", {})
    input_json = to_absolute_path(cfg.refine_task.get("project_input_json", "") or "")
    output_dir = to_absolute_path(cfg.refine_task.get("project_output_dir", "") or "")
    data_root = cfg.refine_task.get("data_root", "") or cfg.dataset.base_dir
    data_root = to_absolute_path(data_root)

    if not input_json or not os.path.exists(input_json):
        raise FileNotFoundError(f"project input not found: {input_json}")
    if not output_dir:
        raise ValueError("refine_task.project_output_dir must be set.")

    with open(input_json, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", data)

    cameras = list(proj.get("cameras", []) or []) or list(cfg.dataset.cameras)
    report = project_tracks(
        results,
        data_root=data_root,
        output_dir=output_dir,
        camera_names=cameras,
        box_width=int(proj.get("box_width", 2)),
        subdiv=int(proj.get("subdiv", 12)),
        max_frames=int(proj.get("max_frames", 0)) or None,
        max_view_angle=float(proj.get("max_view_angle", 80.0)),
    )
    print(
        f"🖼️  Projected {report['boxes_drawn']} boxes over "
        f"{report['frames_written']} frames -> {output_dir}"
    )


if __name__ == "__main__":
    main()
