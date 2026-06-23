"""BEV trajectory visualizer for qualitative track validation.

Renders a top-down (bird's eye) view in the COLMAP world frame, where each
selected track's box footprint is drawn across every frame it appears in, so a
full trajectory can be inspected for correctness and ID continuity. Optionally
overlays the ego path reconstructed from COLMAP.

Driven by ``cfg.refine_task.viz``; select tracks with ``ids`` / ``classes``,
color ``by time`` (gradient) or ``by id``. Used by the 4DGS orchestrator to
render raw and/or refined predictions, and runnable standalone via Hydra
overrides.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import to_absolute_path
from matplotlib import colormaps
from omegaconf import DictConfig

from src.tracking.track_geometry import GROUND_AXES, box_center_ground, box_footprint


def _frame_order(results: dict) -> dict[str, int]:
    keys = sorted(results.keys(), key=lambda k: int(k.split("_")[-1]))
    return {k: i for i, k in enumerate(keys)}


def _collect_tracks(results: dict, ids, classes):
    """Return {track_id: [(frame_index, box), ...]} filtered by ids/classes."""
    fidx = _frame_order(results)
    id_set = {int(i) for i in ids} if ids else None
    cls_set = {str(c) for c in classes} if classes else None
    tracks: dict[int, list] = defaultdict(list)
    for token, boxes in results.items():
        for box in boxes:
            tid = int(box["tracking_id"])
            if id_set is not None and tid not in id_set:
                continue
            if cls_set is not None and box.get("tracking_name") not in cls_set:
                continue
            tracks[tid].append((fidx[token], box))
    for tid in tracks:
        tracks[tid].sort(key=lambda fb: fb[0])
    return tracks, max(fidx.values()) + 1 if fidx else 1


def _reconstruct_ego_path(data_root: str, reference_camera: str) -> np.ndarray | None:
    """Reference-camera positions in raw COLMAP frame, projected to (x, z)."""
    try:
        from src.model_wrappers.trackers.wayve101_cc3dt_dataset import (
            _extract_cam_to_world,
            _load_colmap_scene,
            _resolve_sparse_dir,
        )

        sparse = _resolve_sparse_dir(data_root)
        _, images = _load_colmap_scene(sparse)
        poses = []
        for image in images.values():
            if Path(image.name).parent.as_posix() != reference_camera:
                continue
            ts = int(Path(image.name).stem)
            poses.append((ts, _extract_cam_to_world(image)[:3, 3]))
        if not poses:
            return None
        poses.sort(key=lambda p: p[0])
        pts = np.array([p[1] for p in poses])
        return pts[:, list(GROUND_AXES)]
    except Exception as exc:  # pragma: no cover - ego path is optional
        print(f"⚠️  Ego path unavailable: {exc}")
        return None


def render_bev(
    results: dict,
    output_path: str,
    ids=None,
    classes=None,
    color_by: str = "time",
    show_ego: bool = True,
    ego_path: np.ndarray | None = None,
    title: str = "",
    figsize: float = 14.0,
    bounds=None,
    margin: float = 5.0,
    max_aspect: float = 4.0,
) -> None:
    """Render the BEV trajectory plot to ``output_path``.

    Args:
        bounds: Optional ``[xmin, xmax, zmin, zmax]`` crop in COLMAP world
            meters. If omitted, the view auto-fits all drawn points plus
            ``margin``.
        margin: Meters of padding around auto-fit data bounds.
        max_aspect: Cap on the figure's long:short side ratio so an elongated
            scene fills the canvas without becoming an unreadable sliver.
    """
    tracks, n_frames = _collect_tracks(results, ids, classes)

    # Gather ground points (track centroids + ego) to frame the view.
    xs: list[float] = []
    zs: list[float] = []
    for _, seq in tracks.items():
        for _, box in seq:
            cx, cz = box_center_ground(box)
            xs.append(float(cx))
            zs.append(float(cz))
    if show_ego and ego_path is not None and len(ego_path):
        xs.extend(ego_path[:, 0].tolist())
        zs.extend(ego_path[:, 1].tolist())

    if bounds:
        xmin, xmax, zmin, zmax = (float(v) for v in bounds)
    elif xs:
        xmin, xmax = min(xs) - margin, max(xs) + margin
        zmin, zmax = min(zs) - margin, max(zs) + margin
    else:
        xmin, xmax, zmin, zmax = -10.0, 10.0, -10.0, 10.0

    ext_x = max(xmax - xmin, 1e-3)
    ext_z = max(zmax - zmin, 1e-3)

    # Size the figure to the data aspect so an elongated scene fills the canvas
    # instead of collapsing into a thin ribbon. The longer extent maps to
    # ``figsize``; the shorter side is bounded by ``max_aspect``.
    if ext_z >= ext_x:
        fig_h = figsize
        fig_w = figsize * max(ext_x / ext_z, 1.0 / max_aspect)
    else:
        fig_w = figsize
        fig_h = figsize * max(ext_z / ext_x, 1.0 / max_aspect)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = colormaps["viridis"]
    id_cmap = colormaps["tab20"]

    for order, (tid, seq) in enumerate(sorted(tracks.items())):
        centers = np.array([box_center_ground(b) for _, b in seq])
        # Trajectory polyline.
        if len(centers) >= 2:
            ax.plot(
                centers[:, 0], centers[:, 1],
                "-", lw=0.8, alpha=0.5,
                color=id_cmap(order % 20) if color_by == "id" else "0.6",
                zorder=1,
            )
        # Footprints per frame.
        for fi, box in seq:
            poly = box_footprint(box)
            poly = np.vstack([poly, poly[0]])
            if color_by == "id":
                col = id_cmap(order % 20)
            else:
                col = cmap(fi / max(n_frames - 1, 1))
            ax.plot(poly[:, 0], poly[:, 1], "-", lw=1.0, color=col, zorder=2)
        # Label at the first appearance.
        ax.annotate(
            str(tid), centers[0], fontsize=7, color="black",
            ha="center", va="center", zorder=4,
        )

    if show_ego and ego_path is not None and len(ego_path) >= 2:
        ax.plot(
            ego_path[:, 0], ego_path[:, 1],
            "-", color="red", lw=2.0, alpha=0.8, zorder=3, label="ego",
        )
        ax.scatter(*ego_path[0], c="red", s=40, marker="o", zorder=5)

    # COLMAP world +x points LEFT, so put larger x on the left to draw a
    # natural top-down view (forward up, left on the left, right on the right).
    ax.set_xlim(xmax, xmin)
    ax.set_ylim(zmin, zmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("COLMAP x (+left) [m]")
    ax.set_ylabel("COLMAP z (forward) [m]")
    ax.set_title(title or f"BEV tracks ({len(tracks)} shown)")
    ax.grid(True, alpha=0.3)

    if color_by == "time":
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(0, max(n_frames - 1, 1))
        )
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="frame index", shrink=0.6)
    if show_ego and ego_path is not None:
        ax.legend(loc="upper right")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"🖼️  Wrote {output_path} ({len(tracks)} tracks)")


def render_from_file(
    input_json: str, output_path: str, viz: dict, data_root: str = "", title: str = ""
) -> None:
    """Load a predictions JSON and render the BEV per ``viz`` settings."""
    with open(input_json, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", data)

    ego = None
    if viz.get("show_ego", True) and data_root:
        ego = _reconstruct_ego_path(
            data_root, str(viz.get("reference_camera", "front-forward"))
        )
    render_bev(
        results,
        output_path=output_path,
        ids=list(viz.get("ids", []) or []),
        classes=list(viz.get("classes", []) or []),
        color_by=str(viz.get("color_by", "time")),
        show_ego=bool(viz.get("show_ego", True)),
        ego_path=ego,
        title=title,
        figsize=float(viz.get("figsize", 14.0)),
        bounds=list(viz.get("bounds", []) or []) or None,
        margin=float(viz.get("margin", 5.0)),
        max_aspect=float(viz.get("max_aspect", 4.0)),
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== AVSplat-Sim: BEV Track Visualization ===")
    viz = cfg.refine_task.get("viz", {})
    input_json = to_absolute_path(cfg.refine_task.get("viz_input_json", "") or "")
    output_path = to_absolute_path(cfg.refine_task.get("viz_output_path", "") or "")
    data_root = cfg.refine_task.get("data_root", "") or cfg.dataset.base_dir
    data_root = to_absolute_path(data_root)

    if not input_json or not os.path.exists(input_json):
        raise FileNotFoundError(f"viz input not found: {input_json}")
    if not output_path:
        raise ValueError("refine_task.viz_output_path must be set.")

    render_from_file(input_json, output_path, viz, data_root=data_root)


if __name__ == "__main__":
    main()
