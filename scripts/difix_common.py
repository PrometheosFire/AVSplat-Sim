"""Shared machinery for the Difix pseudo-view loops.

Both loop orchestrators -- ``run_difix_loop.py`` (static scene) and
``run_difix_4dgs.py`` (dynamic rigid objects) -- run the same skeleton: prepare a
dataset, export a bank of real reference frames, then alternate train / render /
diffuse for a schedule of lateral shifts. Everything that does not depend on
which of those two it is lives here, so the dynamic loop extends the static one
rather than forking it.

What stays in the entry scripts is the part that genuinely differs: how a round
is trained (with or without rigid nodes) and how it is rendered (from a PLY, or
from a checkpoint so the rigid nodes come with it).

This module sits in ``scripts/`` on purpose. That directory is ``sys.path[0]``
for both entry scripts, so ``import difix_common`` resolves with no path
manipulation -- and no orchestrator currently imports from ``src/``.

**Cache namespacing.** The two pipelines write under different results roots and
must never read each other's caches. Most step hashes fold in an input path that
already carries the root, so they diverge on their own. One does not:
``ensure_ncore_dataset``'s hash is derived from the *masks* directory, which
lives under ``data/`` and is root-independent. The optional ``hash_prefix``
argument exists for that case. It is applied only when non-empty, so the default
reproduces the static pipeline's existing hashes byte for byte and its caches
keep hitting.
"""

import hashlib
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

SEG_PYTHON = os.path.abspath("envs/env_segmentation/bin/python")
GSPLAT_PYTHON = os.path.abspath("envs/envs/env_gsplat/bin/python")
CC3DT_PYTHON = os.path.abspath("envs/env_cc3dt/bin/python")


# --------------------------------------------------------------------------- #
# Caching, subprocesses, timing                                                #
# --------------------------------------------------------------------------- #


def generate_config_hash(config_subset: dict) -> str:
    """Generates a unique 8-character hash from a dictionary."""
    config_str = json.dumps(config_subset, sort_keys=True)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()[:8]


def _prefixed(payload: dict, hash_prefix: str) -> dict:
    """Namespace a hash payload, leaving it untouched when no prefix is given.

    Adding the key unconditionally would change every existing hash and
    invalidate the static pipeline's caches, so the empty default must be a
    no-op rather than an empty string in the payload.
    """
    if not hash_prefix:
        return payload
    return {**payload, "pipeline": hash_prefix}


def _run(cmd: List[str], env: Dict[str, str]) -> None:
    subprocess.run(cmd, env=env, check=True)


def _done(path: str) -> bool:
    return os.path.exists(os.path.join(path, ".success"))


def _stamp(path: str, message: str) -> None:
    with open(os.path.join(path, ".success"), "w") as fp:
        fp.write(message + "\n")


def _fmt_hms(seconds: float) -> str:
    """Format a duration as HH:MM:SS. Hours accumulate past 24 rather than wrap."""
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _timed(marker_dir: str, fn):
    """Run one loop step and report how long it took, for ``loop_state.json``.

    A cache hit reports ``"cached"`` rather than ``00:00:00`` so the state file
    distinguishes work that was skipped from work that was genuinely instant.
    Either way the step's own ``.success`` file keeps the timing from the run
    that actually produced it, including its sub-block breakdown.
    """
    if _done(marker_dir):
        return fn(), "cached"
    started = time.perf_counter()
    result = fn()
    return result, _fmt_hms(time.perf_counter() - started)


# --------------------------------------------------------------------------- #
# Steps 01-03: masks -> fused masks -> ncore dataset                           #
# Hashing mirrors run_pipeline.py exactly so existing caches are reused.       #
# --------------------------------------------------------------------------- #


def ensure_masks(
    cfg: DictConfig,
    base_dir: str,
    overrides: List[str],
    env,
    hash_prefix: str = "",
) -> str:
    """STEP 01: per-frame object masks from SAM3."""
    step1_params = {
        "dataset": OmegaConf.to_container(cfg.dataset, resolve=True),
        "model": OmegaConf.to_container(cfg.segmenter, resolve=True),
        "pipeline": OmegaConf.to_container(cfg.seg_task, resolve=True),
    }
    masks_dir = os.path.join(
        base_dir,
        f"01_masks_{generate_config_hash(_prefixed(step1_params, hash_prefix))}",
    )
    if _done(masks_dir):
        print(f"[01] cache hit: {masks_dir}")
        return masks_dir

    print(f"[01] extracting object masks -> {masks_dir}")
    _run(
        [
            SEG_PYTHON,
            os.path.abspath("src/pre_training/extract_masks.py"),
            f"hydra.run.dir={masks_dir}",
            *overrides,
        ],
        env,
    )
    return masks_dir


def ensure_fused_masks(
    cfg: DictConfig,
    base_dir: str,
    masks_dir: str,
    overrides: List[str],
    env,
    hash_prefix: str = "",
) -> str:
    """STEP 02: OR the object masks with the ego mask and dilate."""
    mp = cfg.mask_processing
    fuse_str = (
        f"{masks_dir}_{mp.dilation_percentage}_{mp.dilate_ego}_{mp.use_ego_masks}"
    )
    if hash_prefix:
        fuse_str = f"{hash_prefix}_{fuse_str}"
    fuse_hash = hashlib.md5(fuse_str.encode()).hexdigest()[:8]
    fused_dir = os.path.abspath(os.path.join(base_dir, f"02_fused_masks_{fuse_hash}"))
    if _done(fused_dir):
        print(f"[02] cache hit: {fused_dir}")
        return fused_dir

    print(f"[02] fusing ego + object masks -> {fused_dir}")
    _run(
        [
            SEG_PYTHON,
            os.path.abspath("src/pre_training/fuse_masks.py"),
            f"hydra.run.dir={fused_dir}",
            f"+input_masks_dir={masks_dir}",
            *overrides,
        ],
        env,
    )
    return fused_dir


def ensure_ncore_dataset(
    base_dir: str,
    masks_dir: str,
    overrides: List[str],
    env,
    hash_prefix: str = "",
    dir_name: str = "03_ncore_dataset",
    hash_tag: str = "ncore",
) -> str:
    """STEP 03: convert images + masks into the ncore v4 sequence.

    ``masks_dir`` decides what the resulting model can represent, and the two
    pipelines want opposite things from it:

    * SAM-fused masks (``02_fused_masks_*``) hide vehicles, so nothing dynamic
      bakes into the static background as a ghost.
    * the dataset's own ego masks leave vehicles visible, which is what the
      rigid Gaussian nodes learn their appearance from.

    ``dir_name`` and ``hash_tag`` keep those two variants distinguishable when a
    single pipeline builds both (mode 2 of the dynamic loop does).
    """
    ncore_str = f"{hash_tag}_{masks_dir}"
    if hash_prefix:
        ncore_str = f"{hash_prefix}_{ncore_str}"
    ncore_hash = hashlib.md5(ncore_str.encode()).hexdigest()[:8]
    ncore_dir = os.path.abspath(os.path.join(base_dir, f"{dir_name}_{ncore_hash}"))
    if _done(ncore_dir):
        print(f"[03] cache hit: {ncore_dir}")
        return ncore_dir

    print(f"[03] converting to ncore -> {ncore_dir}")
    _run(
        [
            GSPLAT_PYTHON,
            os.path.abspath("src/pre_training/convert_ncore.py"),
            f"hydra.run.dir={ncore_dir}",
            f"+input_masks_dir={masks_dir}",
            *overrides,
        ],
        env,
    )
    return ncore_dir


def resolve_cameras(ncore_dir: str, requested: List[str]) -> List[str]:
    """Camera ids for training, from the list the ncore conversion wrote.

    NCoreParser refuses to auto-detect when a dataset has several cameras, so
    the id list has to be passed explicitly to every downstream step.
    """
    cam_list_path = os.path.join(ncore_dir, "ncore_dataset", "camera_list.yaml")
    available = list(OmegaConf.load(cam_list_path).ncore_camera_ids)
    if not requested:
        return available
    missing = [c for c in requested if c not in available]
    if missing:
        raise ValueError(f"cameras {missing} not in the dataset ({available})")
    return [c for c in available if c in requested]


# --------------------------------------------------------------------------- #
# Step 035: real-frame bank                                                    #
# --------------------------------------------------------------------------- #


def ensure_real_bank(
    base_dir: str,
    ncore_json: str,
    cameras: List[str],
    data_factor: int,
    test_every: int,
    overrides: List[str],
    env,
    hash_prefix: str = "",
) -> str:
    """STEP 035: export the exact pixels the trainer sees, for Difix references.

    The hash covers everything that changes the pixels or the indexing, since
    the bank's frame ordering has to line up with the trainer's split.
    """
    bank_hash = generate_config_hash(
        _prefixed(
            {
                "ncore": ncore_json,
                "cameras": cameras,
                "data_factor": data_factor,
                "test_every": test_every,
            },
            hash_prefix,
        )
    )
    bank_dir = os.path.abspath(os.path.join(base_dir, f"035_real_frames_{bank_hash}"))
    if _done(bank_dir):
        print(f"[035] cache hit: {bank_dir}")
        return bank_dir

    print(f"[035] exporting real frames -> {bank_dir}")
    _run(
        [
            GSPLAT_PYTHON,
            os.path.abspath("src/gsplat_training/export_real_frames.py"),
            f"hydra.run.dir={bank_dir}",
            f"++gaussian_splatting.data_dir={ncore_json}",
            f"++gaussian_splatting.ncore_camera_ids=[{','.join(cameras)}]",
            f"++gaussian_splatting.data_factor={data_factor}",
            f"++gaussian_splatting.test_every={test_every}",
            *overrides,
        ],
        env,
    )
    return bank_dir


# --------------------------------------------------------------------------- #
# Round schedule                                                               #
# --------------------------------------------------------------------------- #


def union_shifts(schedule: List[List[List[float]]]) -> List[List[float]]:
    """Every distinct shift in the schedule, in first-appearance order.

    Used for the comparison renders, which must cover the whole target range in
    every round. Taking ``schedule[-1]`` would only be correct for a cumulative
    schedule; an incremental one ends with just the outermost level.
    """
    seen: List[List[float]] = []
    for round_shifts in schedule:
        for shift in round_shifts:
            if shift not in seen:
                seen.append(shift)
    return seen


def resolve_schedule(loop_cfg: DictConfig) -> List[List[List[float]]]:
    """Shift lists per round: explicit ``shift_schedule``, else a generated ramp.

    ``schedule_mode`` picks how the generated ramp grows, which is a real
    experimental variable rather than a formatting detail:

    * ``cumulative`` -- round r cleans every level up to r+1, so each round's
      pseudo-GT is regenerated by the freshest (best) model. Costs more Difix
      calls, and at ``frame_stride: 1`` it re-cleans the SAME frames, banking
      several versions of one view from models of differing quality.
    * ``incremental`` -- round r cleans only the newest level, reusing earlier
      rounds' output from the bank. Roughly half the Difix cost and no duplicate
      views, but the earliest level keeps pixels made by the weakest model.

    The generated ramp is single-axis and single-direction. A bilateral
    curriculum (say -3 m and +3 m) is expressed by giving ``shift_schedule``
    explicitly; everything downstream is set-based and handles it unchanged.
    """
    if loop_cfg.shift_schedule:
        return [
            [list(map(float, shift)) for shift in round_shifts]
            for round_shifts in loop_cfg.shift_schedule
        ]

    target = [float(v) for v in loop_cfg.target_shift]
    step = [float(v) for v in loop_cfg.step_shift]
    axis = next((i for i, v in enumerate(step) if v != 0.0), None)
    if axis is None:
        raise ValueError("loop_task.step_shift must have a non-zero component")
    n_rounds = int(round(target[axis] / step[axis]))
    if n_rounds < 1:
        raise ValueError(
            f"target_shift {target} is not a positive multiple of step_shift {step}"
        )

    mode = str(loop_cfg.get("schedule_mode", "cumulative")).lower()
    if mode not in ("cumulative", "incremental"):
        raise ValueError(
            f"schedule_mode must be 'cumulative' or 'incremental', got {mode!r}"
        )

    schedule: List[List[List[float]]] = []
    for r in range(n_rounds):
        levels = range(r + 1) if mode == "cumulative" else (r,)
        schedule.append([[step[i] * (k + 1) for i in range(3)] for k in levels])
    return schedule


def derive_step_lists(max_steps: int) -> Dict[str, str]:
    """Keep eval/save/ply aligned with a shortened round.

    A 7000-step round with the default ``[7000, 30000]`` would write no PLY once
    ``adjust_steps`` scaled things, and the next round's render would have
    nothing to load.
    """
    steps = f"[{max_steps}]"
    return {
        "eval_steps": steps,
        "save_steps": steps,
        "ply_steps": steps,
    }


def shift_arg(shifts: List[List[float]]) -> str:
    """Render a shift list as the bracketed literal Hydra expects on the CLI."""
    return "[" + ",".join(
        "[" + ",".join(str(v) for v in s) + "]" for s in shifts
    ) + "]"


# --------------------------------------------------------------------------- #
# Locating a trained round's output                                            #
# --------------------------------------------------------------------------- #


def _latest_ply(train_dir: str) -> str:
    """Highest-step ``.ply`` under ``train_dir/ply``. Background Gaussians only."""
    ply_dir = os.path.join(train_dir, "ply")
    if not os.path.isdir(ply_dir):
        raise RuntimeError(f"no ply/ directory under {train_dir}")

    def step_of(name: str) -> int:
        try:
            return int(name.replace("point_cloud_", "").replace(".ply", ""))
        except ValueError:
            return -1

    plys = sorted([f for f in os.listdir(ply_dir) if f.endswith(".ply")], key=step_of)
    if not plys:
        raise RuntimeError(f"no .ply written under {ply_dir}")
    return os.path.join(ply_dir, plys[-1])


def _latest_checkpoint(ckpt_dir: str) -> str:
    """Highest-step ``*.pt`` checkpoint in ``ckpt_dir`` (or ``""``).

    Checkpoints are named ``ckpt_<step>_rank<r>.pt`` by the trainer, so the step
    is the second underscore-separated field. The dynamic loop renders from this
    rather than from the PLY because rigid nodes are stored in the checkpoint
    only.
    """
    if not os.path.isdir(ckpt_dir):
        return ""

    def step_of(fname: str) -> int:
        try:
            return int(fname.split("_")[1])
        except (IndexError, ValueError):
            return -1

    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")], key=step_of
    )
    return os.path.join(ckpt_dir, ckpts[-1]) if ckpts else ""


# --------------------------------------------------------------------------- #
# The diffusion round                                                          #
# --------------------------------------------------------------------------- #


def run_difix(
    round_dir: str,
    render_dir: str,
    real_bank_dir: str,
    cameras: List[str],
    round_idx: int,
    stride: int,
    phase: int,
    shifts: List[List[float]],
    overrides: List[str],
    env,
    dynamic: bool = False,
) -> str:
    """Clean this round's scheduled shift levels and emit its manifest.

    ``shifts`` is a subset of what was rendered: every round rasterises the full
    target range so its output can be compared like for like across rounds, but
    only the levels on this round's schedule are cleaned and trained on.

    ``dynamic`` is stamped into the manifest and tells the trainer whether to
    composite rigid objects into these pseudo-views. It must describe the
    renders that were cleaned, not the round that will consume them.
    """
    difix_dir = os.path.join(round_dir, "difix")
    manifest = os.path.join(difix_dir, "manifest.json")
    if _done(difix_dir):
        print(f"  [difix] cache hit: {difix_dir}")
        return manifest

    # The zero shift is cleaned as a CONTROL, not as training data: real ground
    # truth exists there, so compute_stats scores it with full-reference metrics
    # (LPIPS is the signal) and that is the round's only go/no-go measurement.
    # build_manifest excludes it from the bank by name, so requesting it adds a
    # measurement without adding pseudo-views.
    #
    # It is requested here rather than by the callers because both render it
    # unconditionally (``all_shifts = [[0, 0, 0]] + shifts``), so it can never
    # be "requested but not rendered". Before this, callers passed only their
    # round's curriculum shifts, which never contain the zero shift, so the
    # control silently stopped being produced when per-round shift selection
    # was added.
    shifts_to_clean = list(shifts)
    if not any(all(abs(v) < 1e-6 for v in s) for s in shifts_to_clean):
        shifts_to_clean.append([0.0, 0.0, 0.0])

    print(
        f"  [difix] shifts={shifts_to_clean} (control included) "
        f"stride={stride} phase={phase} dynamic={dynamic}"
    )
    _run(
        [
            GSPLAT_PYTHON,
            os.path.abspath("src/post_processing/difix_pseudo_views.py"),
            f"hydra.run.dir={os.path.join(difix_dir, 'hydra')}",
            # Reference conditioning is the whole point: the real frame at the
            # same camera and timestep anchors the cleaning.
            "model@diffusion=difix_ref",
            f"++pseudo_task.render_dir={os.path.join(render_dir, 'full')}",
            f"++pseudo_task.real_bank_dir={real_bank_dir}",
            f"++pseudo_task.output_dir={difix_dir}",
            f"++pseudo_task.cameras=[{','.join(cameras)}]",
            f"++pseudo_task.round={round_idx}",
            f"++pseudo_task.frame_stride={stride}",
            f"++pseudo_task.frame_phase={phase}",
            f"++pseudo_task.shifts_m={shift_arg(shifts_to_clean)}",
            f"++pseudo_task.dynamic={str(bool(dynamic)).lower()}",
            *overrides,
        ],
        env,
    )
    return manifest
