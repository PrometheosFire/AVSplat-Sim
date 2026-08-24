"""Iterative Difix pseudo-view loop: a complete, self-contained pipeline.

3DGS reconstructions of AV captures overfit the recorded ego trajectory: all
cameras sweep a single 1D path, so the scene is well constrained near it and
poorly constrained for laterally shifted trajectories. This loop manufactures
the missing supervision -- train, render a shifted trajectory, clean the
artifacts with reference-conditioned Difix3D+, retrain from scratch with the
cleaned frames added -- pushing the shift one step further out each round so
Difix is only ever asked for a small correction on a render the previous round
already partly learned.

Static scene only: dynamic Gaussians, 3D tracking and track refinement are all
off. Dynamic objects are masked out of training by steps 1-2, so the baked model
contains no vehicles for Difix to hallucinate over.

Steps 1-3 duplicate run_pipeline.py's mask extraction, fusion and ncore
conversion, using identical hashing so their caches are shared rather than
recomputed. That duplication is deliberate: this script is meant to run a scene
end to end on its own.

    STEP 01  SAM3 object masks                  01_masks_<h>/
    STEP 02  fuse with ego masks + dilate       02_fused_masks_<h>/
    STEP 03  convert to ncore                   03_ncore_dataset_<h>/
    STEP 035 export real frames (Difix refs)    035_real_frames_<h>/
    STEP 06  the round loop                     06_difix_loop_<h>/
               round_000/{train,render,difix}
               round_001/...
               loop_state.json

Run::

    envs/envs/env_gsplat/bin/python scripts/run_difix_loop.py \\
        dataset.scene=scene_021 \\
        ++loop_task.cameras=[camera1] \\
        ++loop_task.max_steps_per_round=[7000,7000,7000,30000]
"""

import hashlib
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

SEG_PYTHON = os.path.abspath("envs/env_segmentation/bin/python")
GSPLAT_PYTHON = os.path.abspath("envs/envs/env_gsplat/bin/python")


def generate_config_hash(config_subset: dict) -> str:
    """Generates a unique 8-character hash from a dictionary."""
    config_str = json.dumps(config_subset, sort_keys=True)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()[:8]


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


def ensure_masks(cfg: DictConfig, base_dir: str, overrides: List[str], env) -> str:
    """STEP 01: per-frame object masks from SAM3."""
    step1_params = {
        "dataset": OmegaConf.to_container(cfg.dataset, resolve=True),
        "model": OmegaConf.to_container(cfg.segmenter, resolve=True),
        "pipeline": OmegaConf.to_container(cfg.seg_task, resolve=True),
    }
    masks_dir = os.path.join(base_dir, f"01_masks_{generate_config_hash(step1_params)}")
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
    cfg: DictConfig, base_dir: str, masks_dir: str, overrides: List[str], env
) -> str:
    """STEP 02: OR the object masks with the ego mask and dilate."""
    mp = cfg.mask_processing
    fuse_str = (
        f"{masks_dir}_{mp.dilation_percentage}_{mp.dilate_ego}_{mp.use_ego_masks}"
    )
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
    base_dir: str, fused_dir: str, overrides: List[str], env
) -> str:
    """STEP 03: convert images + masks into the ncore v4 sequence."""
    ncore_hash = hashlib.md5(f"ncore_{fused_dir}".encode()).hexdigest()[:8]
    ncore_dir = os.path.abspath(os.path.join(base_dir, f"03_ncore_dataset_{ncore_hash}"))
    if _done(ncore_dir):
        print(f"[03] cache hit: {ncore_dir}")
        return ncore_dir

    print(f"[03] converting to ncore -> {ncore_dir}")
    _run(
        [
            GSPLAT_PYTHON,
            os.path.abspath("src/pre_training/convert_ncore.py"),
            f"hydra.run.dir={ncore_dir}",
            f"+input_masks_dir={fused_dir}",
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
) -> str:
    """STEP 035: export the exact pixels the trainer sees, for Difix references.

    The hash covers everything that changes the pixels or the indexing, since
    the bank's frame ordering has to line up with the trainer's split.
    """
    bank_hash = generate_config_hash(
        {
            "ncore": ncore_json,
            "cameras": cameras,
            "data_factor": data_factor,
            "test_every": test_every,
        }
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


# --------------------------------------------------------------------------- #
# Per-round steps                                                              #
# --------------------------------------------------------------------------- #


def run_training(
    round_dir: str,
    ncore_json: str,
    cameras: List[str],
    manifests: List[str],
    max_steps: int,
    data_factor: int,
    test_every: int,
    overrides: List[str],
    env,
) -> str:
    """Train a FRESH model on the real frames plus the accumulated bank."""
    train_dir = os.path.join(round_dir, "train")
    if _done(train_dir):
        print(f"  [train] cache hit: {train_dir}")
        return train_dir

    manifest_arg = "[" + ",".join(manifests) + "]"
    print(f"  [train] {max_steps} steps, {len(manifests)} manifest(s) in the bank")
    cmd = [
        GSPLAT_PYTHON,
        os.path.abspath("src/gsplat_training/train_splats.py"),
        f"hydra.run.dir={train_dir}",
        f"++gaussian_splatting.data_dir={ncore_json}",
        f"++gaussian_splatting.result_dir={train_dir}",
        f"++gaussian_splatting.ncore_camera_ids=[{','.join(cameras)}]",
        f"++gaussian_splatting.data_factor={data_factor}",
        f"++gaussian_splatting.test_every={test_every}",
        f"++gaussian_splatting.max_steps={max_steps}",
        # Static-only: the pseudo-view guard in Runner.__init__ rejects dynamic.
        "++gaussian_splatting.enable_dynamic=false",
        f"++gaussian_splatting.pseudo_manifests={manifest_arg}",
    ]
    cmd += [f"++gaussian_splatting.{k}={v}" for k, v in derive_step_lists(max_steps).items()]
    cmd += overrides
    _run(cmd, env)
    return train_dir


def _latest_ply(train_dir: str) -> str:
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


def run_render(
    round_dir: str,
    train_dir: str,
    shifts: List[List[float]],
    cameras: List[str],
    overrides: List[str],
    env,
) -> str:
    """Render the ego trajectory plus every shift level, for this round's model.

    Rendering is cheap (~20 ms/frame) next to diffusion, so every round -- the
    final one included -- rasterises the whole set. That is what lets you put
    round 0's and round 3's videos of the SAME trajectory side by side and see
    whether the reconstruction actually improved.

    The ego trajectory is rendered but never difixed: real ground truth already
    exists there, so it is a visual reference, not training data. Selection of
    what gets cleaned happens in run_difix via ``shifts_m``.
    """
    render_dir = os.path.join(round_dir, "render")
    if _done(render_dir):
        print(f"  [render] cache hit: {render_dir}")
        return render_dir

    all_shifts = [[0.0, 0.0, 0.0]] + shifts
    shift_arg = "[" + ",".join(
        "[" + ",".join(str(v) for v in s) + "]" for s in all_shifts
    ) + "]"
    print(f"  [render] shifts={all_shifts}")
    _run(
        [
            GSPLAT_PYTHON,
            os.path.abspath("src/gsplat_training/render_standalone.py"),
            f"hydra.run.dir={render_dir}",
            f"++rendering.ply_path={_latest_ply(train_dir)}",
            f"++rendering.camera_paths_dir={os.path.join(train_dir, 'camera_paths')}",
            f"++rendering.output_dir={render_dir}",
            f"++rendering.cameras_to_render=[{','.join(cameras)}]",
            # Plain colour only: the debug-box and rigid-white modes are not
            # pseudo-GT sources, and there are no rigid nodes in a static bake.
            "++rendering.render_modes=[full]",
            "++rendering.render_dynamic=false",
            f"++rendering.trajectory.shifts={shift_arg}",
            *overrides,
        ],
        env,
    )
    return render_dir


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
) -> str:
    """Clean this round's scheduled shift levels and emit its manifest.

    ``shifts`` is a subset of what was rendered: every round rasterises the full
    target range so its output can be compared like for like across rounds, but
    only the levels on this round's schedule are cleaned and trained on.
    """
    difix_dir = os.path.join(round_dir, "difix")
    manifest = os.path.join(difix_dir, "manifest.json")
    if _done(difix_dir):
        print(f"  [difix] cache hit: {difix_dir}")
        return manifest

    shifts_arg = "[" + ",".join(
        "[" + ",".join(str(v) for v in s) + "]" for s in shifts
    ) + "]"
    print(f"  [difix] shifts={shifts} stride={stride} phase={phase}")
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
            f"++pseudo_task.shifts_m={shifts_arg}",
            *overrides,
        ],
        env,
    )
    return manifest


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Starting the Difix pseudo-view loop")
    overrides = list(HydraConfig.get().overrides.task)
    loop_cfg = cfg.loop_task

    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    base_dir = os.path.abspath(
        f"results/{dataset_cfg['name']}/{dataset_cfg['scene']}"
    )
    os.makedirs(base_dir, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".")

    t_loop = time.perf_counter()

    # --- Data preparation (shares run_pipeline.py's caches) ---
    t_prep = time.perf_counter()
    masks_dir = ensure_masks(cfg, base_dir, overrides, env)
    fused_dir = ensure_fused_masks(cfg, base_dir, masks_dir, overrides, env)
    ncore_dir = ensure_ncore_dataset(base_dir, fused_dir, overrides, env)
    ncore_json = os.path.join(ncore_dir, "ncore_dataset", "staging_symlinks.json")

    cameras = resolve_cameras(ncore_dir, list(loop_cfg.cameras))
    data_factor = int(loop_cfg.data_factor)
    test_every = int(cfg.gaussian_splatting.test_every)
    print(f"Cameras: {cameras} | data_factor={data_factor} test_every={test_every}")

    real_bank_dir = ensure_real_bank(
        base_dir, ncore_json, cameras, data_factor, test_every, overrides, env
    )
    prep_duration = _fmt_hms(time.perf_counter() - t_prep)

    # --- Round schedule ---
    schedule = resolve_schedule(loop_cfg)
    n_rounds = len(schedule) + 1  # the final round trains only
    steps_per_round = (
        [int(s) for s in loop_cfg.max_steps_per_round]
        if loop_cfg.max_steps_per_round
        else [int(cfg.gaussian_splatting.max_steps)] * n_rounds
    )
    if len(steps_per_round) != n_rounds:
        raise ValueError(
            f"max_steps_per_round has {len(steps_per_round)} entries but the "
            f"schedule needs {n_rounds} (len(shift_schedule) + 1)"
        )
    phases = (
        [int(p) for p in loop_cfg.frame_phase_per_round]
        if loop_cfg.frame_phase_per_round
        else [r * 2 for r in range(n_rounds)]
    )
    stride = int(loop_cfg.frame_stride)

    loop_hash = generate_config_hash(
        {
            "ncore": ncore_dir,
            "cameras": cameras,
            "schedule": schedule,
            "steps": steps_per_round,
            "phases": phases[: n_rounds - 1],
            "stride": stride,
            "data_factor": data_factor,
            "gsplat": OmegaConf.to_yaml(cfg.gaussian_splatting),
        }
    )
    # Comparison renders must span the whole target range every round, which is
    # the union of the schedule -- NOT schedule[-1], which is only the full range
    # for a cumulative schedule.
    full_shifts = union_shifts(schedule)

    loop_dir = os.path.join(base_dir, f"06_difix_loop_{loop_hash}")
    os.makedirs(loop_dir, exist_ok=True)
    print(f"\nLoop dir: {loop_dir}")
    print(f"{n_rounds} rounds, stride {stride}, phases {phases[: n_rounds - 1]}")
    print(f"Rendering every round: {full_shifts}")

    # --- The loop ---
    manifests: List[str] = []
    state: Dict[str, Any] = {
        "rounds": [],
        "loop_dir": loop_dir,
        "cameras": cameras,
        # Wall-clock per block, so a finished run can be costed without having to
        # reconstruct it from file mtimes. Each block's own .success file carries
        # the same total plus a per-sub-block breakdown.
        "durations": {"prep": prep_duration},
    }

    for r in range(n_rounds):
        round_dir = os.path.join(loop_dir, f"round_{r:03d}")
        os.makedirs(round_dir, exist_ok=True)
        is_final = r == n_rounds - 1
        print(f"\n{'=' * 60}\nROUND {r}/{n_rounds - 1}{'  (FINAL)' if is_final else ''}")

        t_round = time.perf_counter()
        train_dir, train_duration = _timed(
            os.path.join(round_dir, "train"),
            lambda: run_training(
                round_dir, ncore_json, cameras, manifests, steps_per_round[r],
                data_factor, test_every, overrides, env,
            ),
        )
        record: Dict[str, Any] = {
            "round": r,
            "train_dir": train_dir,
            "max_steps": steps_per_round[r],
            "bank": list(manifests),
            "durations": {"train": train_duration},
        }

        # Every round rasterises the ego trajectory plus the FULL target range,
        # the final one included, so the same trajectories can be compared side
        # by side across rounds to see whether quality actually improves. Only
        # this round's scheduled subset gets cleaned into training data; the ego
        # view is a visual reference only.
        render_dir, render_duration = _timed(
            os.path.join(round_dir, "render"),
            lambda: run_render(
                round_dir, train_dir, full_shifts, cameras, overrides, env
            ),
        )
        record["render_dir"] = render_dir
        record["durations"]["render"] = render_duration

        if not is_final:
            manifest, difix_duration = _timed(
                os.path.join(round_dir, "difix"),
                lambda: run_difix(
                    round_dir, render_dir, real_bank_dir, cameras, r, stride,
                    phases[r], schedule[r], overrides, env,
                ),
            )
            # Accumulate: every round's output stays in the bank, so the phase
            # rotation widens frame coverage instead of replacing it.
            manifests = manifests + [manifest]
            record.update(
                {"manifest": manifest, "difixed_shifts": schedule[r],
                 "phase": phases[r]}
            )
            record["durations"]["difix"] = difix_duration

        record["durations"]["round_total"] = _fmt_hms(time.perf_counter() - t_round)
        print(f"  [round {r}] {record['durations']}")

        state["rounds"].append(record)
        state["durations"]["total"] = _fmt_hms(time.perf_counter() - t_loop)
        with open(os.path.join(loop_dir, "loop_state.json"), "w") as fp:
            json.dump(state, fp, indent=2)

    print(f"\n{'=' * 60}\nLoop complete in {state['durations']['total']}.")
    print(f"Final model: {state['rounds'][-1]['train_dir']}")
    print(f"State: {os.path.join(loop_dir, 'loop_state.json')}")


if __name__ == "__main__":
    main()
