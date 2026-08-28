"""One round of the Difix pseudo-view loop.

Selects a subset of the shifted renders, cleans them with reference-conditioned
Difix (the real frame at the same camera and timestep is the reference), measures
how much the diffusion changed, and writes the manifest the trainer consumes.

Why a dedicated script rather than ``apply_diffusion.py``: that one walks a
directory, has no reference conditioning, no idempotency, and its expected input
path (``<result_dir>/frames``) predates the renderer's per-mode subtree. It is
left untouched for the legacy pipeline.

The ZERO shift is a control, not training data. Real ground truth exists at those
poses, so difixing them measures whether Difix moves a render toward or away from
the truth. Judge that with LPIPS rather than PSNR: Difix is a perceptual
restoration model trained with LPIPS and Gram losses, so it sharpens correctly
while PSNR punishes the resulting sub-pixel misalignment (measured: PSNR -1.92 dB
while LPIPS improved 47% on a visibly better image). Control frames are cleaned
and measured but never enter the manifest.

``pseudo_task.dynamic`` records whether the renders being cleaned contained
rigid objects, and the trainer reads it back to decide whether to composite
vehicles into a pseudo-view's render. It describes the IMAGES, not the intent of
whatever trains on them -- the dynamic loop's ``final_round`` mode deliberately
feeds a bank of static rounds to a dynamic final round.

Run standalone::

    PYTHONPATH=. envs/envs/env_gsplat/bin/python \\
        src/post_processing/difix_pseudo_views.py \\
        model@diffusion=difix_ref \\
        ++pseudo_task.render_dir=<round>/render/full \\
        ++pseudo_task.real_bank_dir=<035_real_frames_*> \\
        ++pseudo_task.output_dir=<round>/difix \\
        ++pseudo_task.frame_stride=6 ++pseudo_task.frame_phase=0
"""

import csv
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

# 2 added the top-level "dynamic" flag. datasets/pseudo.py still reads version 1
# (treating it as static), so banks written before this stay usable.
MANIFEST_SCHEMA_VERSION = 2

# |difix - render| above this (0-255) counts a pixel as changed.
CHANGED_THRESHOLD = 2.0


def _fmt_hms(seconds: float) -> str:
    """Format a duration as HH:MM:SS. Hours accumulate past 24 rather than wrap."""
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _stamp_success(path: str, message: str, elapsed: float, breakdown=None) -> None:
    """Write a .success marker carrying its runtime and optional sub-block times.

    Nothing in the pipeline reads these files -- only their existence is checked
    -- so the extra lines are free to grow.
    """
    lines = [message, f"duration: {_fmt_hms(elapsed)}"]
    if breakdown:
        width = max(len(label) for label, _ in breakdown)
        for label, value in breakdown:
            shown = value if isinstance(value, str) else _fmt_hms(value)
            lines.append(f"  {label:<{width}}  {shown}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def load_real_index(real_bank_dir: str) -> Dict[str, Any]:
    """Read the real-frame bank's index written by ``export_real_frames.py``."""
    index_path = os.path.join(real_bank_dir, "index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"{index_path} not found. Run src/gsplat_training/export_real_frames.py first."
        )
    with open(index_path, "r") as fp:
        return json.load(fp)


def parse_shift_metres(shift_name: str) -> Optional[List[float]]:
    """Invert ``render_standalone._shift_name`` back into ``[x, y, z]`` metres.

    ``"X_-1"`` -> ``[-1, 0, 0]``, ``"X_0.5_Z_-1"`` -> ``[0.5, 0, -1]``,
    ``"original"`` -> ``[0, 0, 0]``. Returns ``None`` if the name does not
    follow that convention; the value is provenance only, so an unparsable
    custom shift name is recorded as null rather than failing the round.
    """
    if shift_name == "original":
        return [0.0, 0.0, 0.0]
    axes = {"X": 0, "Y": 1, "Z": 2}
    out = [0.0, 0.0, 0.0]
    matches = re.findall(r"([XYZ])_(-?\d+(?:\.\d+)?)", shift_name)
    if not matches:
        return None
    for axis, value in matches:
        out[axes[axis]] = float(value)
    return out


def select_frames(
    camera_record: Dict[str, Any], stride: int, phase: int
) -> List[Dict[str, Any]]:
    """Pick this round's frames for one camera.

    Keeps ``frame_idx % stride == phase`` and drops validation frames. This is
    the primary val-exclusion point: the parser's identical check is a backstop
    against a manifest built by some other means.
    """
    return [
        f
        for f in camera_record["frames"]
        if f["frame_idx"] % stride == phase % stride and not f["is_val"]
    ]


def discover_shifts(
    render_dir: str, requested_m: Optional[List[List[float]]] = None
) -> List[str]:
    """Shift directories to clean, optionally filtered to a requested subset.

    The loop renders every shift level each round so successive rounds can be
    compared on the same trajectories, but cleans only the levels scheduled for
    that round. ``requested_m`` is that subset, in metres; matching goes through
    :func:`parse_shift_metres` so the caller never has to reproduce
    ``render_standalone._shift_name``.

    Args:
        render_dir: A render mode directory, e.g. ``<round>/render/full``.
        requested_m: ``[x, y, z]`` triplets to keep. ``None`` or empty keeps all.

    Returns:
        Shift directory names, sorted.

    Raises:
        FileNotFoundError: If ``render_dir`` holds no ``frames/``.
        ValueError: If a requested shift was never rendered.
    """
    frames_root = os.path.join(render_dir, "frames")
    if not os.path.isdir(frames_root):
        raise FileNotFoundError(
            f"{frames_root} not found. Point render_dir at a render mode "
            "directory, e.g. <round>/render/full"
        )
    available = sorted(
        d for d in os.listdir(frames_root) if os.path.isdir(os.path.join(frames_root, d))
    )
    if not requested_m:
        return available

    def matches(name: str, wanted: List[float]) -> bool:
        parsed = parse_shift_metres(name)
        return parsed is not None and all(
            abs(a - b) < 1e-6 for a, b in zip(parsed, wanted)
        )

    selected = [n for n in available if any(matches(n, w) for w in requested_m)]
    unmatched = [w for w in requested_m if not any(matches(n, w) for n in available)]
    if unmatched:
        raise ValueError(
            f"shifts {unmatched} were requested but not rendered under "
            f"{frames_root} (available: {available})"
        )
    return selected


def build_jobs(
    render_dir: str,
    real_bank_dir: str,
    output_dir: str,
    real_index: Dict[str, Any],
    shifts: List[str],
    cameras: List[str],
    stride: int,
    phase: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Pair every selected render with its real reference frame.

    Returns:
        ``(jobs, meta)`` -- ``jobs`` is what ``DifixWrapper.process_pairs``
        consumes; ``meta`` carries the per-view bookkeeping (camera, indices,
        shift) in the same order, for stats and the manifest.
    """
    jobs: List[Dict[str, str]] = []
    meta: List[Dict[str, Any]] = []

    for shift_name in shifts:
        shift_m = parse_shift_metres(shift_name)
        for camera_id in cameras:
            camera_record = real_index["cameras"][camera_id]
            frames_dir = os.path.join(render_dir, "frames", shift_name, camera_id)
            if not os.path.isdir(frames_dir):
                print(f"  [skip] no renders for {shift_name}/{camera_id}")
                continue

            for frame in select_frames(camera_record, stride, phase):
                idx = frame["frame_idx"]
                render_path = os.path.join(frames_dir, f"frame_{idx:05d}.png")
                if not os.path.isfile(render_path):
                    continue
                ref_path = os.path.join(real_bank_dir, frame["image"])
                out_path = os.path.join(
                    output_dir, "frames", shift_name, camera_id, f"frame_{idx:05d}.png"
                )
                jobs.append(
                    {
                        "input_path": render_path,
                        "ref_path": ref_path,
                        "output_path": out_path,
                    }
                )
                meta.append(
                    {
                        "shift_name": shift_name,
                        "shift_m": shift_m,
                        "camera_id": camera_id,
                        "camera_idx": camera_record["camera_index"],
                        "width": camera_record["width"],
                        "height": camera_record["height"],
                        "K": camera_record["K"],
                        "frame_idx": idx,
                        "global_index": frame["global_index"],
                        "timestamp_us": frame["timestamp_us"],
                        "render_path": render_path,
                        "ref_path": ref_path,
                        "image_path": out_path,
                    }
                )

    return jobs, meta


def _read_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return float("inf") if mse == 0 else 10.0 * float(np.log10(255.0**2 / mse))


def compute_stats(
    meta: List[Dict[str, Any]], control_shift_name: str, device: str = "cuda"
) -> List[Dict[str, Any]]:
    """Measure the diffusion correction for every cleaned view.

    Run as a second pass over the written PNGs rather than inside the diffusion
    loop. Decoding 960x540 costs a few milliseconds, so the extra read is worth
    being able to recompute or extend the statistics without paying for
    diffusion again -- which matters because the control statistic is the
    round's go/no-go signal.

    Control views additionally get full-reference metrics against the real
    frame. LPIPS is the signal to trust; PSNR is recorded for diagnosis only.
    """
    lpips = None
    if any(m["shift_name"] == control_shift_name for m in meta):
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device)

    def as_tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).permute(2, 0, 1)[None].float().to(device) / 255.0

    rows: List[Dict[str, Any]] = []
    for entry in meta:
        render = _read_rgb(entry["render_path"]).astype(np.float64)
        difix = _read_rgb(entry["image_path"]).astype(np.float64)
        delta = np.abs(difix - render)

        row: Dict[str, Any] = {
            "shift_name": entry["shift_name"],
            "shift_m_x": (entry["shift_m"] or [None] * 3)[0],
            "shift_m_y": (entry["shift_m"] or [None] * 3)[1],
            "shift_m_z": (entry["shift_m"] or [None] * 3)[2],
            "is_control": entry["shift_name"] == control_shift_name,
            "camera_id": entry["camera_id"],
            "camera_idx": entry["camera_idx"],
            "frame_idx": entry["frame_idx"],
            "global_index": entry["global_index"],
            "timestamp_us": entry["timestamp_us"],
            "mean_abs_delta": round(float(delta.mean()), 4),
            "p95_abs_delta": round(float(np.percentile(delta, 95)), 4),
            "frac_changed": round(float((delta > CHANGED_THRESHOLD).mean()), 5),
            "psnr_render_vs_real": None,
            "psnr_difix_vs_real": None,
            "lpips_render_vs_real": None,
            "lpips_difix_vs_real": None,
        }

        if row["is_control"]:
            real = _read_rgb(entry["ref_path"])
            if real.shape != render.shape:
                real = cv2.resize(
                    real,
                    (render.shape[1], render.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            row["psnr_render_vs_real"] = round(_psnr(render, real), 4)
            row["psnr_difix_vs_real"] = round(_psnr(difix, real), 4)
            if lpips is not None:
                with torch.no_grad():
                    real_t = as_tensor(real)
                    row["lpips_render_vs_real"] = round(
                        float(lpips(as_tensor(render.astype(np.uint8)), real_t)), 5
                    )
                    row["lpips_difix_vs_real"] = round(
                        float(lpips(as_tensor(difix.astype(np.uint8)), real_t)), 5
                    )

        rows.append(row)

    return rows


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-frame rows by shift, and by (shift, camera)."""

    def agg(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"n": len(subset)}
        for key in (
            "mean_abs_delta",
            "p95_abs_delta",
            "frac_changed",
            "psnr_render_vs_real",
            "psnr_difix_vs_real",
            "lpips_render_vs_real",
            "lpips_difix_vs_real",
        ):
            values = [r[key] for r in subset if r[key] is not None]
            out[key] = round(float(np.mean(values)), 5) if values else None
        return out

    by_shift: Dict[str, Any] = {}
    by_shift_camera: Dict[str, Any] = {}
    for shift in sorted({r["shift_name"] for r in rows}):
        subset = [r for r in rows if r["shift_name"] == shift]
        by_shift[shift] = agg(subset)
        for camera in sorted({r["camera_id"] for r in subset}):
            by_shift_camera[f"{shift}/{camera}"] = agg(
                [r for r in subset if r["camera_id"] == camera]
            )
    return {"by_shift": by_shift, "by_shift_camera": by_shift_camera}


def build_manifest(
    meta: List[Dict[str, Any]],
    render_dir: str,
    control_shift_name: str,
    round_idx: int,
    phase: int,
    stride: int,
    test_every: Optional[int],
    dynamic: bool,
) -> Dict[str, Any]:
    """Assemble the trainer-facing manifest, excluding control views.

    Poses come from the ``camtoworlds.npy`` the renderer saved next to the
    frames, so the manifest records the poses that were ACTUALLY rendered rather
    than re-deriving the shift and risking disagreement.

    ``dynamic`` records whether those renders contained rigid objects. It is a
    top-level field because one round cleans one render tree, which is uniformly
    dynamic or not; the trainer's per-view granularity comes from concatenating
    several rounds' manifests, not from variation inside one. It must describe
    the IMAGES -- a manifest that claims the wrong thing makes the trainer
    either paint vehicles into the background or try to erase them, both
    silently.
    """
    pose_cache: Dict[Tuple[str, str], np.ndarray] = {}
    entries: List[Dict[str, Any]] = []

    for entry in meta:
        if entry["shift_name"] == control_shift_name:
            continue
        key = (entry["shift_name"], entry["camera_id"])
        if key not in pose_cache:
            pose_path = os.path.join(
                render_dir, "poses", entry["shift_name"], entry["camera_id"],
                "camtoworlds.npy",
            )
            pose_cache[key] = np.load(pose_path)
        poses = pose_cache[key]
        idx = entry["frame_idx"]
        if idx >= len(poses):
            raise IndexError(
                f"frame_idx {idx} is out of range for {len(poses)} saved poses "
                f"at {key}"
            )

        entries.append(
            {
                "image_path": os.path.abspath(entry["image_path"]),
                "camtoworld": poses[idx].tolist(),
                "K": entry["K"],
                "width": entry["width"],
                "height": entry["height"],
                "camera_id": entry["camera_id"],
                "camera_idx": entry["camera_idx"],
                "frame_idx": idx,
                "global_index": entry["global_index"],
                "timestamp_us": entry["timestamp_us"],
                "shift_name": entry["shift_name"],
                "shift_m": entry["shift_m"],
            }
        )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "round": round_idx,
        "phase": phase,
        "frame_stride": stride,
        "test_every": test_every,
        "dynamic": bool(dynamic),
        "render_dir": os.path.abspath(render_dir),
        "entries": entries,
    }


def write_stats(rows: List[Dict[str, Any]], output_dir: str) -> None:
    """Write stats.csv and stats_summary.json (stdlib csv; no pandas in env)."""
    csv_path = os.path.join(output_dir, "stats.csv")
    with open(csv_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarise(rows)
    with open(os.path.join(output_dir, "stats_summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)

    print("\n=== Difix correction summary ===")
    for shift, stats in summary["by_shift"].items():
        line = (
            f"  {shift:<14s} n={stats['n']:<4d} "
            f"mean|delta|={stats['mean_abs_delta']:>7.3f}  "
            f"changed={stats['frac_changed']:.1%}"
        )
        if stats["lpips_difix_vs_real"] is not None:
            better = stats["lpips_difix_vs_real"] < stats["lpips_render_vs_real"]
            line += (
                f"  | CONTROL lpips {stats['lpips_render_vs_real']:.4f} -> "
                f"{stats['lpips_difix_vs_real']:.4f} "
                f"({'IMPROVED' if better else 'DEGRADED'}), "
                f"psnr {stats['psnr_render_vs_real']:.2f} -> "
                f"{stats['psnr_difix_vs_real']:.2f} dB"
            )
        print(line)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    task = cfg.pseudo_task
    render_dir = task.render_dir
    real_bank_dir = task.real_bank_dir
    output_dir = task.output_dir
    for name, value in (
        ("render_dir", render_dir),
        ("real_bank_dir", real_bank_dir),
        ("output_dir", output_dir),
    ):
        if not value:
            raise ValueError(f"pseudo_task.{name} must be set")

    success_marker = os.path.join(output_dir, ".success")
    if os.path.exists(success_marker) and not task.force:
        print(f"Pseudo-view round already complete at {output_dir}")
        return

    t_start = time.perf_counter()
    phases: List[Tuple[str, float]] = []

    os.makedirs(output_dir, exist_ok=True)
    real_index = load_real_index(real_bank_dir)

    requested_m = [[float(v) for v in s] for s in (task.shifts_m or [])]
    shifts = discover_shifts(render_dir, requested_m)
    cameras = list(task.cameras) or sorted(real_index["cameras"])
    missing = [c for c in cameras if c not in real_index["cameras"]]
    if missing:
        raise ValueError(
            f"cameras {missing} are not in the real-frame bank "
            f"({sorted(real_index['cameras'])})"
        )

    print(
        f"Round {task.round}: shifts={shifts} cameras={cameras} "
        f"stride={task.frame_stride} phase={task.frame_phase} "
        f"dynamic={bool(task.dynamic)}"
    )

    jobs, meta = build_jobs(
        render_dir=render_dir,
        real_bank_dir=real_bank_dir,
        output_dir=output_dir,
        real_index=real_index,
        shifts=shifts,
        cameras=cameras,
        stride=int(task.frame_stride),
        phase=int(task.frame_phase),
    )
    if not jobs:
        raise RuntimeError(
            "No frames selected. Check frame_stride/frame_phase against the "
            "render directory and the real-frame bank."
        )
    n_control = sum(1 for m in meta if m["shift_name"] == task.control_shift_name)
    print(
        f"Selected {len(jobs)} views ({n_control} control, "
        f"{len(jobs) - n_control} pseudo)"
    )

    t_phase = time.perf_counter()
    wrapper = instantiate(cfg.diffusion)
    if not getattr(wrapper, "use_ref", False):
        print(
            "WARNING: the diffusion model is not reference-conditioned. Pass "
            "model@diffusion=difix_ref so the real frame guides the cleaning."
        )
    phases.append(("model load", time.perf_counter() - t_phase))

    # One process_pairs call per shift so each level is timed on its own. Free:
    # build_jobs already emits shift-major order, so the work and its order are
    # unchanged, and the pipeline is built in DifixWrapper.__init__ rather than
    # per call, so the model is loaded exactly once either way.
    for shift_name in shifts:
        shift_jobs = [j for j, m in zip(jobs, meta) if m["shift_name"] == shift_name]
        if not shift_jobs:
            continue
        t_phase = time.perf_counter()
        wrapper.process_pairs(
            shift_jobs,
            prompt=task.prompt,
            timestep=int(task.timestep),
            skip_existing=not task.force,
        )
        elapsed = time.perf_counter() - t_phase
        phases.append((f"difix {shift_name} ({len(shift_jobs)} frames)", elapsed))
        print(f"  {shift_name}: {len(shift_jobs)} frames in {_fmt_hms(elapsed)}")

    t_phase = time.perf_counter()
    rows = compute_stats(meta, task.control_shift_name)
    write_stats(rows, output_dir)
    phases.append(("stats", time.perf_counter() - t_phase))

    t_phase = time.perf_counter()
    manifest = build_manifest(
        meta=meta,
        render_dir=render_dir,
        control_shift_name=task.control_shift_name,
        round_idx=int(task.round),
        phase=int(task.frame_phase),
        stride=int(task.frame_stride),
        test_every=real_index.get("test_every"),
        dynamic=bool(task.dynamic),
    )
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"\nManifest: {len(manifest['entries'])} pseudo-views -> {manifest_path}")
    phases.append(("manifest", time.perf_counter() - t_phase))

    _stamp_success(
        success_marker,
        "Difix pseudo-view round completed successfully.",
        time.perf_counter() - t_start,
        phases,
    )


if __name__ == "__main__":
    main()
