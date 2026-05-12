import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    print("🚀 Booting up Post-Processing Diffusion...")

    # 1. Instantiate the Model Dynamically
    # Hydra reads configs/model/difix.yaml and builds the class automatically!
    print("📦 Loading Diffusion Model...")
    diffusion_wrapper = instantiate(cfg.diffusion)

    # 2. Determine Paths
    # We pull the gsplat result directory directly from the config
    gsplat_dir = cfg.gaussian_splatting.result_dir
    input_frames_root = os.path.join(gsplat_dir, "videos", "frames")
    
    # Define where the cleaned output should go (reading from the Orchestrator's override)
    # If not provided via CLI, fallback to a default folder
    output_base_dir = cfg.diff_task.get("output_dir", os.path.join(gsplat_dir, "difix_cleaned"))
    
    cleaned_frames_root = os.path.join(output_base_dir, "frames")
    cleaned_videos_root = os.path.join(output_base_dir, "videos")

    if not os.path.exists(input_frames_root):
        print(f"❌ Error: Cannot find GSplat rendered frames at {input_frames_root}")
        print("Did you enable video rendering in the GSplat training step?")
        return

    # 3. Get the Exact Camera List
    # We use the exact same list that was passed to GSplat!
    target_cameras = cfg.gaussian_splatting.ncore_camera_ids

    print(f"\n🎯 Found {len(target_cameras)} cameras to process: {target_cameras}")

    # 4. Processing Loop
    for cam_name in target_cameras:
        print(f"\n==========================================")
        print(f"📷 Processing Camera: {cam_name}")
        print(f"==========================================")
        
        # We need to handle the fact that gsplat replaces slashes with underscores
        # (e.g., "camera_front-forward" might be saved as "camera_front-forward")
        safe_cam_name = str(cam_name).replace("/", "_")
        
        input_dir = os.path.join(input_frames_root, safe_cam_name)
        output_dir = os.path.join(cleaned_frames_root, safe_cam_name)
        
        # Set up video path if requested in the pipeline config
        video_path = None
        if cfg.diff_task.generate_videos:
            video_path = os.path.join(cleaned_videos_root, f"{safe_cam_name}_cleaned.mp4")

        if not os.path.exists(input_dir):
            print(f"⚠️ Warning: No input folder found for {safe_cam_name}. Skipping.")
            continue

        # Execute the wrapper!
        diffusion_wrapper.process_frames(
            input_dir=input_dir,
            output_dir=output_dir,
            prompt=cfg.diff_task.prompt,
            video_path=video_path
        )

    # 5. Success Stamp for Standalone Runs
    success_stamp = os.path.join(output_base_dir, ".success")
    os.makedirs(output_base_dir, exist_ok=True)
    with open(success_stamp, "w") as f:
        f.write("Diffusion post-processing completed successfully.")

    print(f"\n🎉 Diffusion Post-Processing Complete!")
    print(f"📂 Cleaned data saved to: {output_base_dir}")

if __name__ == "__main__":
    main()