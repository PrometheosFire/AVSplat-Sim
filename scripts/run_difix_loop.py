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

import json
import os
import time
from typing import Any, Dict, List

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from difix_common import (
    GSPLAT_PYTHON,
    _done,
    _fmt_hms,
    _latest_ply,
    _run,
    _timed,
    derive_step_lists,
    ensure_fused_masks,
    ensure_masks,
    ensure_ncore_dataset,
    ensure_real_bank,
    generate_config_hash,
    resolve_cameras,
    resolve_schedule,
    run_difix,
    shift_arg,
    union_shifts,
)


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
            f"++rendering.trajectory.shifts={shift_arg(all_shifts)}",
            *overrides,
        ],
        env,
    )
    return render_dir


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
