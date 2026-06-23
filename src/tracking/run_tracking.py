import os

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== AVSplat-Sim: Tracking ===")

    if not cfg.track_task.enabled:
        print("⏭️ Tracking disabled via cfg.track_task.enabled=false")
        return

    # ==========================================
    # 1. Path Resolution
    # ==========================================
    base_output_dir = HydraConfig.get().runtime.output_dir

    base_input_dir = to_absolute_path(
        cfg.track_task.data_root or cfg.dataset.base_dir
    )

    print(f"\n📁 Resolved Input Directory: {base_input_dir}")
    print(f"📁 Resolved Output Directory: {base_output_dir}")

    cameras = list(cfg.track_task.cameras) if cfg.track_task.cameras else list(cfg.dataset.cameras)

    if not os.path.exists(base_input_dir):
        raise FileNotFoundError(f"Tracking input directory does not exist: {base_input_dir}")

    # ==========================================
    # 2. Build task config passed to wrapper
    # ==========================================
    task_cfg = OmegaConf.to_container(cfg.track_task, resolve=True)
    assert isinstance(task_cfg, dict)

    run_name = f"{cfg.tracker.name}_{cfg.tracker.backbone}_{cfg.dataset.scene}"
    if task_cfg.get("experiment_suffix"):
        run_name = f"{run_name}_{task_cfg['experiment_suffix']}"
    task_cfg["experiment_name"] = run_name

    # ==========================================
    # 3. Instantiate the Tracker
    # ==========================================
    print(f"\n🤖 Loading Tracker: {cfg.tracker.name}...")
    tracker = instantiate(cfg.tracker)

    # ==========================================
    # 4. Execute the Contract
    # ==========================================
    output_dir = base_output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📸 Cameras: {cameras}")
    print(f"🏃 Run Name: {run_name}")

    try:
        tracker.run_tracking(
            data_root=base_input_dir,
            output_dir=output_dir,
            cameras=cameras,
            task_cfg=task_cfg,
        )
    except Exception as exc:
        print(f"\n❌ FATAL ERROR in tracking: {exc}")
        raise

    success_marker = os.path.join(output_dir, ".success")
    with open(success_marker, "w", encoding="utf-8") as f:
        f.write("Tracking finished successfully.")
    print(f"✅ .success marker written.")

    print(f"\n✅ Worker finished. Exiting cleanly.")


if __name__ == "__main__":
    main()
