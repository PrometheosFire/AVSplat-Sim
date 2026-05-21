import os
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    print("🚀 Booting up Post-Processing Diffusion...")

    print("📦 Loading Diffusion Model...")
    diffusion_wrapper = instantiate(cfg.diffusion)

    gsplat_dir = cfg.gaussian_splatting.result_dir
    input_frames_root = os.path.join(gsplat_dir, "frames")

    output_base_dir = cfg.diff_task.get("output_dir", os.path.join(gsplat_dir, "difix_cleaned"))
    cleaned_frames_root = os.path.join(output_base_dir, "frames")
    cleaned_videos_root = os.path.join(output_base_dir, "videos")

    if not os.path.exists(input_frames_root):
        print(f"❌ Error: Cannot find GSplat rendered frames at {input_frames_root}")
        return

    # Discover shifts from directory structure: frames/{shift_name}/{camera_name}/
    shift_names = sorted([
        d for d in os.listdir(input_frames_root)
        if os.path.isdir(os.path.join(input_frames_root, d))
    ])
    print(f"\n🎯 Found {len(shift_names)} trajectory shifts: {shift_names}")

    for shift_name in shift_names:
        shift_input_dir = os.path.join(input_frames_root, shift_name)

        camera_names = sorted([
            d for d in os.listdir(shift_input_dir)
            if os.path.isdir(os.path.join(shift_input_dir, d))
        ])

        # Filter cameras if specified in config
        cameras_filter = list(cfg.diff_task.get("cameras_to_process", []))
        if cameras_filter:
            camera_names = [c for c in camera_names if c in cameras_filter]
            print(f"\n{'='*50}")
            print(f"📍 Shift: {shift_name} — processing {len(camera_names)}/{len(sorted(os.listdir(shift_input_dir)))} cameras (filtered)")
        else:
            print(f"\n{'='*50}")
            print(f"📍 Shift: {shift_name} — {len(camera_names)} cameras")

        for cam_name in camera_names:
            print(f"\n  📷 Camera: {cam_name}")

            input_dir = os.path.join(shift_input_dir, cam_name)
            output_dir = os.path.join(cleaned_frames_root, shift_name, cam_name)

            video_path = None
            if cfg.diff_task.generate_videos:
                os.makedirs(cleaned_videos_root, exist_ok=True)
                video_path = os.path.join(cleaned_videos_root, f"{shift_name}_{cam_name}_cleaned.mp4")

            diffusion_wrapper.process_frames(
                input_dir=input_dir,
                output_dir=output_dir,
                prompt=cfg.diff_task.prompt,
                video_path=video_path
            )

    success_stamp = os.path.join(output_base_dir, ".success")
    os.makedirs(output_base_dir, exist_ok=True)
    with open(success_stamp, "w") as f:
        f.write("Diffusion post-processing completed successfully.")

    print(f"\n🎉 Diffusion Post-Processing Complete!")
    print(f"📂 Cleaned data saved to: {output_base_dir}")

if __name__ == "__main__":
    main()
