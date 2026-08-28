"""Difix pseudo-view loop WITH dynamic rigid objects.

Joins the static pseudo-view loop (``run_difix_loop.py``) with the 4DGS pipeline
(``run_4dgs.py``): track and refine 3D vehicle boxes, then run the shifted-view
curriculum on a scene that carries those vehicles as rigid Gaussian nodes.

Two modes, set by ``loop4d_task.dynamic_mode``:

``all_rounds``
    Vehicles are modelled from round 0. The renders contain them, so the cleaned
    pseudo-views contain them, so pseudo-GT supervises the rigid nodes too.

``final_round``
    Background-only rounds (SAM-masked, so no vehicle ghosts bake in) build the
    static scene, then one dynamic round (ego-masked) fits vehicles on top.

What makes both correct is that the manifest records whether the renders it
cleaned had vehicles in them, and ``PseudoViewDataset`` emits ``timestamp_us``
only for those entries -- which is what decides whether the trainer composites
rigid nodes into a pseudo-view's render. ``final_round``'s dynamic round trains
on a bank of static pseudo-views, and those correctly render background-only.

Everything lives under ``results/extended_4dgs/<dataset>/<scene>/``. Nothing is
read from ``results/4dgs/`` or ``results/<dataset>/<scene>/``: tracking and
refinement re-run the first time a scene passes through here, which is the
deliberate price of having a single place to look when a cache misses.

    STEP 10   3D tracking                     10_tracking_<h>/
    STEP 15   track refinement (+ BEV viz)    15_refine_<h>/
    STEP 03   ncore, ego masks only           03_ncore_ego_<h>/
    STEP 01   SAM masks          MODE 2 ONLY  01_masks_<h>/
    STEP 02   fuse + dilate      MODE 2 ONLY  02_fused_masks_<h>/
    STEP 03b  ncore, SAM-masked  MODE 2 ONLY  03_ncore_masked_<h>/
    STEP 035  real frames (Difix references)  035_real_frames_<h>/
    STEP 07   the round loop                  07_difix_4dgs_<h>/
                round_000/{train,render,difix}
                loop_state.json

Run::

    envs/envs/env_gsplat/bin/python scripts/run_difix_4dgs.py \\
        dataset.scene=scene_021 \\
        ++loop4d_task.cameras=[camera1] \\
        ++loop4d_task.max_steps_per_round=[7000,7000]
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from difix_common import (
    CC3DT_PYTHON,
    GSPLAT_PYTHON,
    _done,
    _fmt_hms,
    _latest_checkpoint,
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

# Namespaces every cache key so a folder produced here can never be confused
# with run_4dgs.py's, even where the hash input (a path under data/) carries no
# results root of its own.
HASH_PREFIX = "extended_4dgs"

DYNAMIC_MODES = ("all_rounds", "final_round")


# --------------------------------------------------------------------------- #
# Steps 10-15: tracking -> refinement                                          #
# Ports of run_4dgs.py, running in the tracker environment.                    #
# --------------------------------------------------------------------------- #


def ensure_tracking(cfg: DictConfig, base_dir: str, overrides: List[str], env) -> str:
    """STEP 10: 3D object tracks from the camera rig."""
    step10_params = {
        "pipeline": HASH_PREFIX,
        "dataset": OmegaConf.to_container(cfg.dataset, resolve=True),
        "tracker": OmegaConf.to_container(cfg.tracker, resolve=True),
        "track_task": OmegaConf.to_container(cfg.track_task, resolve=True),
    }
    tracking_dir = os.path.join(
        base_dir, f"10_tracking_{generate_config_hash(step10_params)}"
    )
    if _done(tracking_dir):
        print(f"[10] cache hit: {tracking_dir}")
        return tracking_dir

    tracker_name = cfg.tracker._target_.split(".")[-1]
    print(f"[10] tracking ({tracker_name}) -> {tracking_dir}")
    _run(
        [
            CC3DT_PYTHON,
            os.path.abspath("src/tracking/run_tracking.py"),
            f"hydra.run.dir={tracking_dir}",
            *overrides,
        ],
        env,
    )
    return tracking_dir


def ensure_refine(
    cfg: DictConfig, base_dir: str, tracking_dir: str, overrides: List[str], env
) -> str:
    """STEP 15: fuse, filter and bicycle-fit the raw tracks. Returns the JSON.

    ``user_refinement`` is excluded from the hash on purpose (mirroring
    ``run_4dgs.py``): the automatic output is identical regardless of it, and
    keeping the directory stable across that toggle is what lets a later
    ``enabled=false`` run reuse manual edits from an earlier interactive one.
    """
    refine_task_cfg = OmegaConf.to_container(cfg.refine_task, resolve=True)
    if not refine_task_cfg.get("enabled", True):
        raise ValueError(
            "refine_task.enabled=false, but the dynamic loop needs refined "
            "tracks to place rigid objects. Enable it, or use "
            "scripts/run_difix_loop.py for a static-only loop."
        )

    refine_hash_cfg = {
        k: v for k, v in refine_task_cfg.items() if k != "user_refinement"
    }
    step15_params = {
        "pipeline": HASH_PREFIX,
        "dataset": OmegaConf.to_container(cfg.dataset, resolve=True),
        "tracker": OmegaConf.to_container(cfg.tracker, resolve=True),
        "track_task": OmegaConf.to_container(cfg.track_task, resolve=True),
        "refine_task": refine_hash_cfg,
    }
    refine_dir = os.path.join(
        base_dir, f"15_refine_{generate_config_hash(step15_params)}"
    )
    refined_json = os.path.join(refine_dir, "track_3d_refined_colmap.json")

    user_refine = refine_task_cfg.get("user_refinement", {}) or {}
    user_refine_enabled = bool(user_refine.get("enabled", False))
    base_hit = _done(refine_dir)
    if base_hit and not user_refine_enabled:
        print(f"[15] cache hit: {refine_dir}")
        return refined_json

    if base_hit and user_refine_enabled:
        print("[15] cache exists but user_refinement.enabled=true -- rerunning")
    print(f"[15] refining tracks -> {refine_dir}")

    viz_cfg = refine_task_cfg.get("visualize", {}) or {}
    viz_dir = os.path.join(refine_dir, "vis")
    raw_json = os.path.join(tracking_dir, "eval", "track_3d_predictions_colmap.json")
    viz_script = os.path.abspath("src/tracking/visualize_tracks.py")

    if viz_cfg.get("raw", True):
        _run(
            [
                CC3DT_PYTHON, viz_script,
                f"refine_task.viz_input_json={raw_json}",
                f"refine_task.viz_output_path={os.path.join(viz_dir, 'bev_raw.png')}",
                *overrides,
            ],
            env,
        )

    _run(
        [
            CC3DT_PYTHON,
            os.path.abspath("src/tracking/refine_tracks.py"),
            f"hydra.run.dir={refine_dir}",
            f"refine_task.input_dir={tracking_dir}",
            *overrides,
        ],
        env,
    )

    if viz_cfg.get("refined", True):
        _run(
            [
                CC3DT_PYTHON, viz_script,
                f"refine_task.viz_input_json={refined_json}",
                f"refine_task.viz_output_path={os.path.join(viz_dir, 'bev_refined.png')}",
                *overrides,
            ],
            env,
        )
    return refined_json


# --------------------------------------------------------------------------- #
# Preconditions                                                                #
# --------------------------------------------------------------------------- #


def check_frames_balanced(real_bank_dir: str) -> None:
    """Refuse scenes where the renderer and trainer would disagree on rigid frames.

    The renderer carries no timestamps -- it walks a trajectory positionally and
    places vehicles with ``frame i -> rigid i`` -- while the trainer resolves
    them from a capture timestamp. Those agree only when every camera has the
    same frame count and they share one timestamp set. A ragged scene would put
    one camera's vehicles a frame off, silently, in every pseudo-view it
    produces.
    """
    index_path = os.path.join(real_bank_dir, "index.json")
    with open(index_path, "r") as fp:
        index = json.load(fp)

    balanced = index.get("frames_balanced")
    if balanced is None:
        raise ValueError(
            f"{index_path} predates the frames_balanced check, so whether this "
            "scene is safe for dynamic rendering is unknown. Delete the bank "
            "and let it re-export."
        )
    if not balanced:
        raise ValueError(
            f"{index_path} reports frames NOT balanced across cameras "
            f"({index.get('num_frames')} frames, {len(index.get('cameras', {}))} "
            f"cameras, {index.get('num_unique_timestamps')} unique timestamps). "
            "The renderer places rigid objects positionally and would disagree "
            "with the trainer's timestamp lookup. Static training is unaffected "
            "-- use scripts/run_difix_loop.py for this scene."
        )
    print(
        f"[035] frames balanced: {index['num_frames']} frames = "
        f"{len(index['cameras'])} cameras x {index['num_unique_timestamps']} timestamps"
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
    dynamic: bool,
    tracks_json: str,
    scene_root: str,
    overrides: List[str],
    env,
) -> str:
    """Train a FRESH model on the real frames plus the accumulated bank.

    ``dynamic`` decides whether rigid nodes exist at all this round. When true
    the model also fits vehicles, and its checkpoint -- not its PLY -- is what
    the render step must load.
    """
    train_dir = os.path.join(round_dir, "train")
    if _done(train_dir):
        print(f"  [train] cache hit: {train_dir}")
        return train_dir

    manifest_arg = "[" + ",".join(manifests) + "]"
    print(
        f"  [train] {max_steps} steps, {len(manifests)} manifest(s) in the bank, "
        f"dynamic={dynamic}"
    )
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
        f"++gaussian_splatting.enable_dynamic={str(bool(dynamic)).lower()}",
        f"++gaussian_splatting.pseudo_manifests={manifest_arg}",
    ]
    if dynamic:
        cmd += [
            f"++gaussian_splatting.dynamic_tracks_json={tracks_json}",
            f"++gaussian_splatting.dynamic_scene_root={scene_root}",
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
    dynamic: bool,
    overrides: List[str],
    env,
) -> str:
    """Render the ego trajectory plus every shift level, for this round's model.

    Dynamic rounds render from the CHECKPOINT and must not pass ``ply_path``:
    rigid nodes live only in the checkpoint, and render_standalone prefers the
    PLY whenever both are given, which would silently drop the vehicles.
    """
    render_dir = os.path.join(round_dir, "render")
    if _done(render_dir):
        print(f"  [render] cache hit: {render_dir}")
        return render_dir

    all_shifts = [[0.0, 0.0, 0.0]] + shifts
    cmd = [
        GSPLAT_PYTHON,
        os.path.abspath("src/gsplat_training/render_standalone.py"),
        f"hydra.run.dir={render_dir}",
        f"++rendering.camera_paths_dir={os.path.join(train_dir, 'camera_paths')}",
        f"++rendering.output_dir={render_dir}",
        f"++rendering.cameras_to_render=[{','.join(cameras)}]",
        "++rendering.render_modes=[full]",
        f"++rendering.trajectory.shifts={shift_arg(all_shifts)}",
    ]
    if dynamic:
        ckpt = _latest_checkpoint(os.path.join(train_dir, "ckpts"))
        if not ckpt:
            raise RuntimeError(
                f"no checkpoint under {train_dir}/ckpts; dynamic rendering "
                "needs one because rigid nodes are not written to the PLY. "
                "Check that save_steps includes max_steps."
            )
        print(f"  [render] shifts={all_shifts} dynamic (ckpt)")
        cmd += [
            f"++rendering.checkpoint_path={ckpt}",
            "++rendering.render_dynamic=true",
        ]
    else:
        print(f"  [render] shifts={all_shifts} static (ply)")
        cmd += [
            f"++rendering.ply_path={_latest_ply(train_dir)}",
            "++rendering.render_dynamic=false",
        ]
    cmd += overrides
    _run(cmd, env)
    return render_dir


# --------------------------------------------------------------------------- #
# Mode dispatch                                                                #
# --------------------------------------------------------------------------- #


def round_is_dynamic(mode: str, round_idx: int, n_rounds: int) -> bool:
    """Whether round ``round_idx`` models vehicles, per the configured mode."""
    if mode == "all_rounds":
        return True
    return round_idx == n_rounds - 1


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Starting the Difix pseudo-view loop WITH dynamic objects")
    overrides = list(HydraConfig.get().overrides.task)
    loop_cfg = cfg.loop4d_task

    mode = str(loop_cfg.dynamic_mode).lower()
    if mode not in DYNAMIC_MODES:
        raise ValueError(
            f"loop4d_task.dynamic_mode must be one of {DYNAMIC_MODES}, got {mode!r}"
        )

    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    base_dir = os.path.abspath(
        f"results/extended_4dgs/{dataset_cfg['name']}/{dataset_cfg['scene']}"
    )
    os.makedirs(base_dir, exist_ok=True)
    scene_root = os.path.abspath(dataset_cfg["base_dir"])

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".")

    t_loop = time.perf_counter()
    t_prep = time.perf_counter()

    # --- Tracks ---
    tracking_dir = ensure_tracking(cfg, base_dir, overrides, env)
    tracks_json = ensure_refine(cfg, base_dir, tracking_dir, overrides, env)

    # --- Datasets. The ego-masked one leaves vehicles VISIBLE, which is what
    #     the rigid nodes learn their appearance from. Mode 2 additionally
    #     builds a SAM-masked one so its background-only rounds do not bake
    #     vehicles in as ghosts.
    ego_masks_dir = os.path.abspath(os.path.join(dataset_cfg["base_dir"], "masks"))
    ncore_ego = ensure_ncore_dataset(
        base_dir, ego_masks_dir, overrides, env,
        hash_prefix=HASH_PREFIX, dir_name="03_ncore_ego", hash_tag="ncore_ego",
    )
    ncore_masked: Optional[str] = None
    if mode == "final_round":
        masks_dir = ensure_masks(cfg, base_dir, overrides, env, hash_prefix=HASH_PREFIX)
        fused_dir = ensure_fused_masks(
            cfg, base_dir, masks_dir, overrides, env, hash_prefix=HASH_PREFIX
        )
        ncore_masked = ensure_ncore_dataset(
            base_dir, fused_dir, overrides, env,
            hash_prefix=HASH_PREFIX, dir_name="03_ncore_masked",
        )

    def ncore_json_of(ncore_dir: str) -> str:
        return os.path.join(ncore_dir, "ncore_dataset", "staging_symlinks.json")

    cameras = resolve_cameras(ncore_ego, list(loop_cfg.cameras))
    data_factor = int(loop_cfg.data_factor)
    test_every = int(cfg.gaussian_splatting.test_every)
    print(f"Cameras: {cameras} | data_factor={data_factor} test_every={test_every}")

    # One bank serves both ncore variants: masks do not affect the exported
    # pixels, and frame ordering, intrinsics and test_every are identical
    # (verified: camera order matches and max |dK| is exactly 0).
    real_bank_dir = ensure_real_bank(
        base_dir, ncore_json_of(ncore_ego), cameras, data_factor, test_every,
        overrides, env, hash_prefix=HASH_PREFIX,
    )
    check_frames_balanced(real_bank_dir)
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
            "pipeline": HASH_PREFIX,
            "mode": mode,
            "ncore_ego": ncore_ego,
            "ncore_masked": ncore_masked,
            "tracks": tracks_json,
            "scene_root": scene_root,
            "cameras": cameras,
            "schedule": schedule,
            "steps": steps_per_round,
            "phases": phases[: n_rounds - 1],
            "stride": stride,
            "data_factor": data_factor,
            "gsplat": OmegaConf.to_yaml(cfg.gaussian_splatting),
        }
    )
    full_shifts = union_shifts(schedule)

    loop_dir = os.path.join(base_dir, f"07_difix_4dgs_{loop_hash}")
    os.makedirs(loop_dir, exist_ok=True)
    print(f"\nLoop dir: {loop_dir}")
    print(f"Mode: {mode} | {n_rounds} rounds, stride {stride}, phases {phases[: n_rounds - 1]}")
    print(f"Rendering every round: {full_shifts}")
    print(f"Tracks: {tracks_json}")

    # --- The loop ---
    manifests: List[str] = []
    state: Dict[str, Any] = {
        "rounds": [],
        "loop_dir": loop_dir,
        "cameras": cameras,
        "dynamic_mode": mode,
        "tracks_json": tracks_json,
        "ncore_ego": ncore_ego,
        "ncore_masked": ncore_masked,
        "durations": {"prep": prep_duration},
    }

    for r in range(n_rounds):
        round_dir = os.path.join(loop_dir, f"round_{r:03d}")
        os.makedirs(round_dir, exist_ok=True)
        is_final = r == n_rounds - 1
        dynamic = round_is_dynamic(mode, r, n_rounds)
        ncore_dir = ncore_ego if dynamic else (ncore_masked or ncore_ego)
        print(
            f"\n{'=' * 60}\nROUND {r}/{n_rounds - 1}"
            f"{'  (FINAL)' if is_final else ''}  dynamic={dynamic}"
        )

        t_round = time.perf_counter()
        train_dir, train_duration = _timed(
            os.path.join(round_dir, "train"),
            lambda: run_training(
                round_dir, ncore_json_of(ncore_dir), cameras, manifests,
                steps_per_round[r], data_factor, test_every, dynamic,
                tracks_json, scene_root, overrides, env,
            ),
        )
        record: Dict[str, Any] = {
            "round": r,
            "train_dir": train_dir,
            "max_steps": steps_per_round[r],
            "dynamic": dynamic,
            "ncore": ncore_dir,
            "bank": list(manifests),
            "durations": {"train": train_duration},
        }

        render_dir, render_duration = _timed(
            os.path.join(round_dir, "render"),
            lambda: run_render(
                round_dir, train_dir, full_shifts, cameras, dynamic, overrides, env
            ),
        )
        record["render_dir"] = render_dir
        record["durations"]["render"] = render_duration

        if not is_final:
            # The manifest must describe the renders that were just cleaned, not
            # the round that will train on them. In final_round mode those are
            # background-only, and the next round being dynamic does not change
            # what is in the pixels.
            manifest, difix_duration = _timed(
                os.path.join(round_dir, "difix"),
                lambda: run_difix(
                    round_dir, render_dir, real_bank_dir, cameras, r, stride,
                    phases[r], schedule[r], overrides, env, dynamic=dynamic,
                ),
            )
            manifests = manifests + [manifest]
            record.update(
                {"manifest": manifest, "difixed_shifts": schedule[r],
                 "phase": phases[r], "manifest_dynamic": dynamic}
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
