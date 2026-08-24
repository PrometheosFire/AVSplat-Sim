import os
import time
import hydra
from hydra.utils import instantiate, to_absolute_path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def _fmt_hms(seconds):
    """Format a duration as HH:MM:SS. Hours accumulate past 24 rather than wrap."""
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _stamp_success(path, message, elapsed, breakdown=None):
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


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    print("=== AVSplat-Sim: Pre-Training (Mask Extraction) ===")
    
    # Optional: Print the config to verify Hydra stitched it together correctly
    # print(OmegaConf.to_yaml(cfg))

    # ==========================================
    # 1. Path Resolution
    # ==========================================
    # 2. Ask Hydra exactly where the output directory is mapped to
    base_output_dir = HydraConfig.get().runtime.output_dir 
    
    base_input_dir = to_absolute_path(cfg.dataset.base_dir)
    base_input_dir = os.path.join(base_input_dir, "images")
    
    print(f"\n📁 Resolved Input Directory: {base_input_dir}")
    print(f"📁 Resolved Output Directory: {base_output_dir}")
    
    cameras = cfg.dataset.cameras
    prompt = cfg.seg_task.prompt

    # ==========================================
    # 2. Instantiate the Segmenter
    # ==========================================
    print(f"\n🤖 Loading Segmenter: {cfg.segmenter.name}...")
    
    # Hydra reads cfg.segmenter._target_, imports your wrapper class, 
    # and passes it the checkpoint path and chunk size automatically!
    segmenter = instantiate(cfg.segmenter)

    # ==========================================
    # 3. Multi-Camera Orchestration Loop
    # ==========================================
    t_start = time.perf_counter()
    per_camera = []

    for camera_name in cameras:
        print(f"\n{'='*50}")
        print(f"🎥 Processing Camera: {camera_name}")
        print(f"{'='*50}")
        
        cam_input_dir = os.path.join(base_input_dir, camera_name)
        cam_output_dir = os.path.join(base_output_dir, camera_name)
        print(f"📁 Camera Input Directory: {cam_input_dir}")
        print(f"📁 Camera Output Directory: {cam_output_dir}")
        cam_success_marker = os.path.join(cam_output_dir, ".success")
        
        if not os.path.exists(cam_input_dir):
            print(f"⚠️ WARNING: Directory not found -> {cam_input_dir}")
            print("Skipping to the next camera...")
            continue
            
            
        
        # ==========================================
        # 4. Execute the Contract
        # ==========================================
        try:
            t_cam = time.perf_counter()
            # We just hand the folder to the wrapper. It does the heavy lifting.
            segmenter.extract_masks(
                input_dir=cam_input_dir,
                output_dir=cam_output_dir,
                prompt=prompt
            )
            cam_elapsed = time.perf_counter() - t_cam
            per_camera.append((camera_name, cam_elapsed))
            print(f"✅ Finished {camera_name} in {_fmt_hms(cam_elapsed)}")

            _stamp_success(
                cam_success_marker,
                "Extraction finished flawlessly for this camera.",
                cam_elapsed,
            )
            print(f"✅ .success marker written for {camera_name}")

        except Exception as e:
            print(f"\n❌ FATAL ERROR processing {camera_name}: {e}")
            segmenter.predictor.shutdown()
            exit(1)
        
    segmenter.predictor.shutdown()
    
    master_success_marker = os.path.join(base_output_dir, ".success")
    _stamp_success(
        master_success_marker,
        "All cameras processed successfully.",
        time.perf_counter() - t_start,
        per_camera,
    )

    print(f"\n✅ Worker finished assigned cameras. Exiting cleanly.")

if __name__ == "__main__":
    main()