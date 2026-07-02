import hashlib
import json
import os
import subprocess

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def generate_config_hash(config_subset: dict) -> str:
    """Generates a unique 8-character hash from a dictionary."""
    config_str = json.dumps(config_subset, sort_keys=True)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()[:8]


def ensure_ncore_dataset(cfg, dataset_cfg: dict, overrides, explicit: str = "") -> str:
    """Create (or reuse) the ncore dataset for the 4DGS pipeline.

    Unlike run_pipeline.py, this pipeline skips SAM segmentation and mask
    fusion: it feeds the dataset's own ego masks
    (``<dataset.base_dir>/masks``) straight into the ncore converter. Output is
    written to the shared static results dir (``results/<name>/<scene>/``) using
    a ``03_ncore_dataset_*`` cache folder. When ``explicit`` is set, that
    directory is used directly and no conversion runs. Returns the ncore dataset
    directory (the folder containing ``ncore_dataset/``).
    """
    if explicit:
        return os.path.abspath(explicit)

    static_results_dir = os.path.abspath(
        f"results/{dataset_cfg['name']}/{dataset_cfg['scene']}"
    )
    os.makedirs(static_results_dir, exist_ok=True)

    gsplat_python = os.path.abspath("envs/envs/env_gsplat/bin/python")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".")

    # Ego masks shipped with the dataset (objects=black on white background,
    # exactly the convention the ncore converter consumes). No SAM / fusion.
    ego_masks_dir = os.path.abspath(os.path.join(dataset_cfg["base_dir"], "masks"))

    # --- STEP 03: ncore dataset conversion (ego masks only) ---
    print("\n" + "=" * 50)
    print("▶️ [STEP 03] Checking ncore Dataset Conversion (ego masks only)...")
    ncore_config_str = f"ncore_ego_{ego_masks_dir}"
    ncore_hash = hashlib.md5(ncore_config_str.encode()).hexdigest()[:8]
    ncore_dir = os.path.join(
        static_results_dir, f"03_ncore_dataset_{ncore_hash}"
    )
    if os.path.exists(os.path.join(ncore_dir, ".success")):
        print(f"✅ Cache Hit! Reusing ncore dataset from: {ncore_dir}")
    else:
        print(f"🔄 New ncore config (Hash: {ncore_hash}). Running conversion...")
        print(f"🎭 ego masks: {ego_masks_dir}")
        subprocess.run(
            [
                gsplat_python,
                os.path.abspath("src/pre_training/convert_ncore.py"),
                f"hydra.run.dir={ncore_dir}",
                f"+input_masks_dir={ego_masks_dir}",
                *overrides,
            ],
            env=env,
            check=True,
        )
        print(f"✅ ncore dataset saved to {ncore_dir}")

    return ncore_dir


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("🚀 Starting 4DGS Orchestrator...")

    # Capture CLI overrides so each step subprocess inherits them
    overrides = HydraConfig.get().overrides.task

    # Resolve config subsets used for hashing and path building
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    tracker_cfg = OmegaConf.to_container(cfg.tracker, resolve=True)
    track_task_cfg = OmegaConf.to_container(cfg.track_task, resolve=True)
    refine_task_cfg = OmegaConf.to_container(cfg.refine_task, resolve=True)
    train_task_cfg = OmegaConf.to_container(cfg.train_task, resolve=True)

    base_results_dir = os.path.abspath(
        f"results/4dgs/{dataset_cfg['name']}/{dataset_cfg['scene']}"
    )
    os.makedirs(base_results_dir, exist_ok=True)

    python_exec = os.path.abspath("envs/env_cc3dt/bin/python")
    script_path = os.path.abspath("src/tracking/run_tracking.py")
    refine_script = os.path.abspath("src/tracking/refine_tracks.py")
    viz_script = os.path.abspath("src/tracking/visualize_tracks.py")
    project_script = os.path.abspath("src/tracking/project_tracks.py")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".")

    # ==========================================
    # STEP 10: TRACKING
    # ==========================================
    print("\n" + "=" * 50)
    tracker_name = cfg.tracker._target_.split(".")[-1]
    print(f"▶️ [STEP 10] Checking Tracking ({tracker_name})...")

    step10_params = {
        "dataset": dataset_cfg,
        "tracker": tracker_cfg,
        "track_task": track_task_cfg,
    }
    step10_hash = generate_config_hash(step10_params)

    tracking_dir = os.path.join(base_results_dir, f"10_tracking_{step10_hash}")
    success_marker = os.path.join(tracking_dir, ".success")

    if os.path.exists(success_marker):
        print("✅ Cache Hit! Exact configuration already run.")
        print(f"📂 Reusing tracking results from: {tracking_dir}")
    else:
        print(f"🔄 New configuration detected (Hash: {step10_hash}). Running tracking...")

        subprocess.run(
            [
                python_exec,
                script_path,
                f"hydra.run.dir={tracking_dir}",
                *overrides,
            ],
            env=env,
            check=True,
        )

        print(f"✅ Tracking saved to {tracking_dir}")

    # ==========================================
    # STEP 15: TRACK REFINEMENT (fuse + filter) + BEV VISUALIZATION
    # ==========================================
    print("\n" + "=" * 50)
    print("▶️ [STEP 15] Checking Track Refinement...")

    refined_tracks_json = None
    if not refine_task_cfg.get("enabled", True):
        print("⏭️ Refinement disabled via refine_task.enabled=false")
    else:
        step15_params = {
            "dataset": dataset_cfg,
            "tracker": tracker_cfg,
            "track_task": track_task_cfg,
            "refine_task": refine_task_cfg,
        }
        step15_hash = generate_config_hash(step15_params)
        refine_dir = os.path.join(base_results_dir, f"15_refine_{step15_hash}")
        refine_success = os.path.join(refine_dir, ".success")

        raw_json = os.path.join(
            tracking_dir, "eval", "track_3d_predictions_colmap.json"
        )
        viz_cfg = refine_task_cfg.get("visualize", {})
        viz_dir = os.path.join(refine_dir, "vis")
        refined_tracks_json = os.path.join(
            refine_dir, "track_3d_refined_colmap.json"
        )

        if os.path.exists(refine_success):
            print(f"✅ Cache Hit! Reusing refinement from: {refine_dir}")
        else:
            print(f"🔄 New refinement config (Hash: {step15_hash}). Running...")

            # Optional BEV of the raw (pre-refinement) tracks.
            if viz_cfg.get("raw", True):
                subprocess.run(
                    [
                        python_exec, viz_script,
                        f"refine_task.viz_input_json={raw_json}",
                        f"refine_task.viz_output_path={os.path.join(viz_dir, 'bev_raw.png')}",
                        *overrides,
                    ],
                    env=env, check=True,
                )

            # Step 1.5: fuse + filter.
            subprocess.run(
                [
                    python_exec, refine_script,
                    f"hydra.run.dir={refine_dir}",
                    f"refine_task.input_dir={tracking_dir}",
                    *overrides,
                ],
                env=env, check=True,
            )

            # Optional BEV of the refined (post-refinement) tracks.
            if viz_cfg.get("refined", True):
                refined_json = os.path.join(
                    refine_dir, "track_3d_refined_colmap.json"
                )
                subprocess.run(
                    [
                        python_exec, viz_script,
                        f"refine_task.viz_input_json={refined_json}",
                        f"refine_task.viz_output_path={os.path.join(viz_dir, 'bev_refined.png')}",
                        *overrides,
                    ],
                    env=env, check=True,
                )

            # Optional: project refined boxes back onto the camera frames.
            if viz_cfg.get("project", True):
                refined_json = os.path.join(
                    refine_dir, "track_3d_refined_colmap.json"
                )
                subprocess.run(
                    [
                        python_exec, project_script,
                        f"refine_task.project_input_json={refined_json}",
                        f"refine_task.project_output_dir={os.path.join(refine_dir, 'projected')}",
                        *overrides,
                    ],
                    env=env, check=True,
                )

            print(f"✅ Refinement saved to {refine_dir}")

    # ==========================================
    # STEP 20: 4D GAUSSIAN SPLATTING TRAINING (with dynamic rigid annotations)
    #   Prepares the ncore dataset (ego masks only) then trains the 4DGS model.
    # ==========================================
    print("\n" + "=" * 50)
    print("▶️ [STEP 20] Preparing 4DGS Training (dynamic rigid objects)...")

    if not train_task_cfg.get("enabled", True):
        print("⏭️ 4DGS training disabled via train_task.enabled=false")
    elif refined_tracks_json is None:
        print(
            "⏭️ Skipping 4DGS training: refinement is disabled, so there are no "
            "annotations to feed. Enable refine_task to produce the tracks JSON."
        )
    else:
        # Create (or reuse) the ncore dataset the same way run_pipeline.py does.
        ncore_dir = ensure_ncore_dataset(
            cfg, dataset_cfg, overrides, train_task_cfg.get("ncore_dataset_dir", "")
        )
        ncore_json_path = os.path.join(
            ncore_dir, "ncore_dataset", "staging_symlinks.json"
        )

        print("\n" + "=" * 50)
        print("▶️ [STEP 20] Checking 4DGS Training (dynamic rigid objects)...")

        # Camera list auto-extracted by the ncore conversion step.
        cam_list_path = os.path.join(ncore_dir, "ncore_dataset", "camera_list.yaml")
        if os.path.exists(cam_list_path):
            cam_config = OmegaConf.load(cam_list_path)
            cam_list_str = ",".join(cam_config.ncore_camera_ids)
            cam_override = f"++gaussian_splatting.ncore_camera_ids=[{cam_list_str}]"
        else:
            cam_override = "++gaussian_splatting.ncore_camera_ids=[]"

        # COLMAP scene root (contains colmap_sparse/rig) used to align the tracks
        # from the COLMAP world into the trainer's normalized frame.
        scene_root = os.path.abspath(dataset_cfg["base_dir"])

        # Hash over the training config + its dynamic inputs so re-runs cache.
        gsplat_config_yaml = OmegaConf.to_yaml(cfg.gaussian_splatting)
        step20_hash = generate_config_hash(
            {
                "gsplat": gsplat_config_yaml,
                "ncore": ncore_dir,
                "tracks": refined_tracks_json,
                "scene_root": scene_root,
            }
        )
        training_dir = os.path.join(
            base_results_dir, f"20_gsplat_dynamic_{step20_hash}"
        )
        train_success = os.path.join(training_dir, ".success")

        if os.path.exists(train_success):
            print(f"✅ Cache Hit! Reusing 4DGS training from: {training_dir}")
        else:
            print(f"🔄 New 4DGS training config (Hash: {step20_hash}). Running...")
            print(f"📂 ncore dataset: {ncore_dir}")
            print(f"🚗 rigid annotations: {refined_tracks_json}")

            # Training runs in the gsplat environment (separate from the tracker).
            gsplat_python_exec = os.path.abspath("envs/envs/env_gsplat/bin/python")
            train_script = os.path.abspath("src/gsplat_training/train_splats.py")

            subprocess.run(
                [
                    gsplat_python_exec,
                    train_script,
                    f"hydra.run.dir={training_dir}",
                    f"++gaussian_splatting.data_dir={ncore_json_path}",
                    f"++gaussian_splatting.result_dir={training_dir}",
                    cam_override,
                    "++gaussian_splatting.enable_dynamic=true",
                    f"++gaussian_splatting.dynamic_tracks_json={refined_tracks_json}",
                    f"++gaussian_splatting.dynamic_scene_root={scene_root}",
                    *overrides,
                ],
                env=env,
                check=True,
            )

            print(f"✅ 4DGS training saved to {training_dir}")


if __name__ == "__main__":
    main()
