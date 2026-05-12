import os
import cv2
import numpy as np
import scipy.ndimage as ndimage
import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    print("=== AVSplat-Sim: Pre-Training (Mask Fusion & Dilation) ===")
    
    # 1. Path Resolution
    
    # Input 1: Read from the CLI argument passed by the Orchestrator
    sam_masks_base_dir = to_absolute_path(cfg.input_masks_dir) 
    
    # Input 2: Strictly assumed to be in the dataset's base 'masks' folder
    ego_masks_base_dir = to_absolute_path(os.path.join(cfg.dataset.base_dir, "masks"))
    
    # Output: The Orchestrator mapped this to the new hashed directory
    output_base_dir = HydraConfig.get().runtime.output_dir
    
    dilation_pct = cfg.mask_processing.dilation_percentage

    for camera_name in cfg.dataset.cameras:
        sam_camera_dir = os.path.join(sam_masks_base_dir, camera_name)
        ego_camera_dir = os.path.join(ego_masks_base_dir, camera_name)
        output_camera_dir = os.path.join(output_base_dir, camera_name)
        
        os.makedirs(output_camera_dir, exist_ok=True)
        camera_success_marker = os.path.join(output_camera_dir, ".success")

        # Idempotent Checkpoint
        if os.path.exists(camera_success_marker):
            print(f"⏭️ Skipping {camera_name}: .success marker found. Already fused!")
            continue

        print(f"\n--- Fusing and Dilating Camera: {camera_name} ---")
        
        try:
            # Grab all SAM mask filenames
            valid_exts = ('.png', '.jpg', '.jpeg')
            sam_files = sorted([f for f in os.listdir(sam_camera_dir) if f.lower().endswith(valid_exts)])
            
            for filename in sam_files:
                sam_path = os.path.join(sam_camera_dir, filename)
                ego_path = os.path.join(ego_camera_dir, filename)
                save_path = os.path.join(output_camera_dir, filename)

                # 1. Load SAM Mask (Grayscale)
                sam_mask = cv2.imread(sam_path, cv2.IMREAD_GRAYSCALE)
                if sam_mask is None:
                    raise FileNotFoundError(f"Could not read SAM mask: {sam_path}")
                    
                # 🔄 INVERT SAM MASK: Make Objects=255 (White), Background=0 (Black)
                sam_mask_standard = cv2.bitwise_not(sam_mask)

                # 2. Load Ego Mask 
                if os.path.exists(ego_path) and cfg.mask_processing.use_ego_masks:
                    ego_mask_raw = cv2.imread(ego_path, cv2.IMREAD_GRAYSCALE)
                    # 🔄 INVERT EGO MASK: Make Ego=255 (White), Background=0 (Black)
                    ego_mask = cv2.bitwise_not(ego_mask_raw)
                else:
                    # If no ego mask exists for this frame, default to a safe, empty black background
                    ego_mask = np.zeros_like(sam_mask)

                # Ensure dimensions perfectly match before logic operations
                if sam_mask.shape != ego_mask.shape:
                    ego_mask = cv2.resize(ego_mask, (sam_mask.shape[1], sam_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

                # 3. Convert to Boolean for SciPy
                sam_mask_bool = sam_mask_standard != 0
                ego_mask_bool = ego_mask != 0

                # 4. Fuse masks using Bitwise OR on booleans
                fused_mask_bool = sam_mask_bool | ego_mask_bool

                # 5. Dynamic Dilation Calculation
                img_width = fused_mask_bool.shape[1]
                n_dilation = max(1, int(img_width * (dilation_pct / 100.0)))

                # 6. Apply SciPy Dilation (Safely expands the white objects!)
                #dilated_mask_bool = ndimage.binary_dilation(fused_mask_bool, iterations=n_dilation)
                if cfg.mask_processing.dilate_ego:
                    # Old Behavior: Fuse first, then dilate everything together
                    fused_mask_bool = sam_mask_bool | ego_mask_bool
                    final_mask_bool = ndimage.binary_dilation(fused_mask_bool, iterations=n_dilation)
                else:
                    # New Behavior: Dilate SAM objects only, THEN fuse with the exact ego mask
                    dilated_sam_bool = ndimage.binary_dilation(sam_mask_bool, iterations=n_dilation)
                    final_mask_bool = dilated_sam_bool | ego_mask_bool

                # 7. Convert back to OpenCV format (0 and 255)
                final_mask_img = (final_mask_bool.astype(np.uint8) * 255)
                
                # RE-INVERT: Put it back to Black Objects on White Background for 3D Splatting
                final_saved_mask = cv2.bitwise_not(final_mask_img)
                cv2.imwrite(save_path, final_saved_mask)
                
            # Write the granular success marker for this specific camera
            with open(camera_success_marker, "w") as f:
                f.write(f"Fused and dilated with {n_dilation} iterations ({dilation_pct}% of width)")
            print(f"✅ Fused & Dilated {len(sam_files)} frames for {camera_name}")

        except Exception as e:
            print(f"\n❌ ERROR processing {camera_name}: {e}")
            print("Crash detected! The .success marker was NOT written.")
            exit(1)
            
    master_success_marker = os.path.join(output_base_dir, ".success")
    with open(master_success_marker, "w") as f:
        f.write("All cameras fused and dilated successfully.")

    print(f"\n🎉 Mask Fusion Complete! All data ready for 3D Splatting.")

if __name__ == "__main__":
    main()