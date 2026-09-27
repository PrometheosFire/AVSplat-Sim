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


def _reconstruct_ego_poses(
    data_root: str, reference_camera: str
) -> tuple[list[int], np.ndarray] | None:
    """Reference-camera timestamps and ground positions, sorted by timestamp.

    The variant of :func:`_reconstruct_ego_path` the per-frame renderer needs: it
    must pair each ego pose with the prediction token for the same instant, so the
    timestamps have to come back too rather than just the path.
    """
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
        ts = [p[0] for p in poses]
        pts = np.array([p[1] for p in poses])[:, list(GROUND_AXES)]
        return ts, pts
    except Exception as exc:  # pragma: no cover - ego path is optional
        print(f"⚠️  Ego poses unavailable: {exc}")
        return None


def _ego_headings(path: np.ndarray) -> np.ndarray:
    """Per-frame heading unit vectors from the ego path's tangent.

    Central differences, with one-sided differences at the ends. The COLMAP poses
    carry an orientation, but it is the CAMERA's, which includes the rig mounting
    rotation; the path tangent is the direction the vehicle actually travelled and
    needs no rig calibration to interpret.
    """
    n = len(path)
    if n < 2:
        return np.tile(np.array([0.0, 1.0]), (max(n, 1), 1))
    d = np.empty_like(path, dtype=np.float64)
    d[1:-1] = path[2:] - path[:-2]
    d[0] = path[1] - path[0]
    d[-1] = path[-1] - path[-2]
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    # A stationary frame has no tangent; carry the previous heading forward rather
    # than emitting a zero vector that would collapse the ego frame.
    bad = norms[:, 0] < 1e-9
    d[bad] = np.array([0.0, 1.0])
    norms[bad] = 1.0
    out = d / norms
    for i in range(1, n):
        if bad[i]:
            out[i] = out[i - 1]
    return out


def render_bev_frames(
    results: dict,
    output_dir: str,
    ego_ts: list[int],
    ego_xz: np.ndarray,
    ids=None,
    classes=None,
    max_range: float = 60.0,
    ring_step: float = 10.0,
    trail: int = 8,
    figsize: float = 12.0,
    dpi: int = 130,
    limit: int | None = None,
) -> int:
    """Write one ego-centred BEV per timestep, mirroring the tracker's BEV output.

    The single-image renderer draws every track at every frame into one plot, which
    for a 200-frame scene with tens of tracks is unreadable -- the trajectories
    overlap and nothing localises to a moment. This instead answers "what was around
    the ego at time t", which is the question the refinement is actually inspected
    for.

    Layout follows the tracker's own BEV so the two can be compared side by side:
    ego at the centre pointing up, alternating grey range rings labelled every
    ``ring_step`` metres, one oriented rectangle per object, a centre dot, and a
    short dot trail of that track's recent centres.

    Frames are paired to ego poses by TIMESTAMP parsed from the prediction token,
    not by rank, so a missing or reordered frame cannot silently shift every box
    onto the wrong ego pose.

    Returns:
        Number of frames written.
    """
    from matplotlib.patches import Circle, Polygon

    tracks, _ = _collect_tracks(results, ids, classes)
    if not tracks:
        print("⚠️  No tracks to render.")
        return 0

    # token -> timestamp, and the per-frame box list.
    tokens = sorted(results.keys())
    def _ts_of(token: str) -> int | None:
        tail = str(token).rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else None

    ego_by_ts = {t: i for i, t in enumerate(ego_ts)}
    headings = _ego_headings(np.asarray(ego_xz, dtype=np.float64))

    # Per-track history of (frame_index, centre), for the dot trails.
    # tab10, not tab20: tab20 alternates each hue with a pastel twin, and the
    # pastels are hard to see at the few-pixel sizes distant objects occupy.
    id_colors = colormaps["tab10"]
    order_of = {tid: i for i, tid in enumerate(sorted(tracks.keys()))}
    history: dict[int, list[np.ndarray]] = {tid: [] for tid in tracks}

    os.makedirs(output_dir, exist_ok=True)
    written = 0
    for fi, token in enumerate(tokens):
        if limit is not None and written >= limit:
            break
        ts = _ts_of(token)
        if ts is None or ts not in ego_by_ts:
            continue
        ei = ego_by_ts[ts]
        origin = np.asarray(ego_xz[ei], dtype=np.float64)
        fwd = headings[ei]
        # +x is LEFT and +z FORWARD in this frame, so the right-hand direction is
        # (-f_z, f_x): facing forward (0,1) gives right = (-1,0).
        right = np.array([-fwd[1], fwd[0]])

        def to_ego(pt: np.ndarray) -> np.ndarray:
            d = np.asarray(pt, dtype=np.float64) - origin
            return np.array([float(d @ right), float(d @ fwd)])

        fig, ax = plt.subplots(figsize=(figsize, figsize))
        # Range rings, darkest at the centre, so distance reads at a glance.
        n_rings = max(int(round(max_range / ring_step)), 1)
        for k in range(n_rings, 0, -1):
            r = k * ring_step
            # Darkest at the centre, lightening outward, matching the tracker's
            # BEV so near/far reads the same way in both. The ramp is spread over
            # however many rings there are rather than stepped by a fixed amount:
            # a fixed step saturated against the light end past six rings, so at
            # 100 m every ring beyond 60 m came out the same grey and the bands
            # stopped being readable as distance at all.
            t = (k - 1) / max(n_rings - 1, 1)
            shade = 0.60 + 0.30 * t
            ax.add_patch(Circle((0, 0), r, facecolor=str(shade),
                                edgecolor="none", zorder=0))
            ax.annotate(f"{int(r)} m", (r, 0), fontsize=7, color="white",
                        ha="center", va="center", zorder=6,
                        bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none"))

        # Ego: a fixed rectangle at the origin, pointing up.
        ego_l, ego_w = 4.5, 1.9
        ax.add_patch(Polygon(
            [(-ego_w / 2, -ego_l / 2), (ego_w / 2, -ego_l / 2),
             (ego_w / 2, ego_l / 2), (-ego_w / 2, ego_l / 2)],
            closed=True, fill=False, edgecolor="black", lw=1.6, zorder=5,
        ))

        present = 0
        beyond = 0
        for tid, seq in tracks.items():
            box = next((b for f, b in seq if f == fi), None)
            if box is None:
                continue
            centre = to_ego(box_center_ground(box))
            history[tid].append(centre)
            if np.hypot(*centre) > max_range:
                # Counted and reported: a frame can legitimately look empty while
                # holding several vehicles just outside the ring, and silently
                # dropping them makes that indistinguishable from "nothing here".
                beyond += 1
                continue
            present += 1
            col = id_colors(order_of[tid] % 10)
            poly = np.array([to_ego(c) for c in box_footprint(box)])
            ax.add_patch(Polygon(poly, closed=True, fill=False,
                                 edgecolor=col, lw=1.3, zorder=4))
            ax.scatter(*centre, s=14, color=col, zorder=5)
            tr = history[tid][-trail:]
            if len(tr) > 1:
                t = np.array(tr)
                ax.scatter(t[:-1, 0], t[:-1, 1], s=7, color=col, alpha=0.75, zorder=3)

        lim = max_range * 1.02
        # x_plot is already the ego-RIGHT component, so it grows rightward with no
        # inversion. (The overview plot inverts because it draws COLMAP x directly,
        # where +x points left; that does not apply once the points are in the ego
        # frame.)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        extra = f", {beyond} beyond" if beyond else ""
        ax.set_title(f"frame {fi:03d}   t={ts}   {present} within "
                     f"{int(max_range)} m{extra}", fontsize=9)
        fig.savefig(os.path.join(output_dir, f"{ts}.png"), dpi=dpi,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written += 1

    print(f"🖼️  Wrote {written} per-frame BEVs to {output_dir}")
    return written


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

    ref_cam = str(viz.get("reference_camera", "front-forward"))

    # Per-frame mode: one ego-centred BEV per timestep, written into a directory
    # named after the single-image output. Needs ego poses (for the ego frame), so
    # it falls through to the overview plot when they are unavailable rather than
    # emitting frames in an arbitrary world orientation.
    if bool(viz.get("per_frame", False)) and data_root:
        poses = _reconstruct_ego_poses(data_root, ref_cam)
        if poses is not None:
            ego_ts, ego_xz = poses
            out_dir = os.path.splitext(output_path)[0] + "_frames"
            render_bev_frames(
                results,
                output_dir=out_dir,
                ego_ts=ego_ts,
                ego_xz=ego_xz,
                ids=list(viz.get("ids", []) or []),
                classes=list(viz.get("classes", []) or []),
                max_range=float(viz.get("max_range", 60.0)),
                ring_step=float(viz.get("ring_step", 10.0)),
                trail=int(viz.get("trail", 8)),
                figsize=float(viz.get("frame_figsize", 8.0)),
                limit=(int(viz["frame_limit"]) if viz.get("frame_limit") else None),
            )
        else:
            print("⚠️  per_frame requested but ego poses are unavailable; "
                  "writing the overview plot only")

    ego = None
    if viz.get("show_ego", True) and data_root:
        ego = _reconstruct_ego_path(data_root, ref_cam)
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
