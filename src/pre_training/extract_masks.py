import os
import hydra
from hydra.utils import instantiate, to_absolute_path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

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
    
    cameras = cfg.dataset.cameras
    prompt = cfg.pipeline.prompt

    # ==========================================
    # 2. Instantiate the Segmenter
    # ==========================================
    print(f"\n🤖 Loading Segmenter: {cfg.model.name}...")
    
    # Hydra reads cfg.model._target_, imports your wrapper class, 
    # and passes it the checkpoint path and chunk size automatically!
    segmenter = instantiate(cfg.model)

    # ==========================================
    # 3. Multi-Camera Orchestration Loop
    # ==========================================
    for camera_name in cameras:
        print(f"\n{'='*50}")
        print(f"🎥 Processing Camera: {camera_name}")
        print(f"{'='*50}")
        
        cam_input_dir = os.path.join(base_input_dir, camera_name)
        cam_output_dir = os.path.join(base_output_dir, camera_name)
        
        if not os.path.exists(cam_input_dir):
            print(f"⚠️ WARNING: Directory not found -> {cam_input_dir}")
            print("Skipping to the next camera...")
            continue
            
        # ==========================================
        # 4. Execute the Contract
        # ==========================================
        try:
            # We just hand the folder to the wrapper. It does the heavy lifting.
            segmenter.extract_masks(
                input_dir=cam_input_dir,
                output_dir=cam_output_dir,
                prompt=prompt
            )
            print(f"✅ Finished {camera_name}")
            
        except Exception as e:
            print(f"\n❌ FATAL ERROR processing {camera_name}: {e}")
            pipeline_failed = True 
            break  # 2. Stop immediately! Don't try the next camera.

    # 3. Only write the success marker if the flag is still False
    if not pipeline_failed:
        print(f"\n🎉 Mask Extraction Complete! All data saved to:\n{base_output_dir}")
        with open(os.path.join(base_output_dir, ".success"), "w") as f:
            f.write("Extraction finished flawlessly.")
    else:
        print(f"\n💥 Pipeline aborted due to errors. The .success marker was NOT written.")
        # 4. Exit with an error code so the orchestrator knows it failed
        exit(1)

if __name__ == "__main__":
    main()