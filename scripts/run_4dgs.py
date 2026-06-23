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


if __name__ == "__main__":
    main()
