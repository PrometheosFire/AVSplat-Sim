import os
import shutil
import subprocess
import sys
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

def create_symlink(src: str, dst: str):
    """Creates a symlink securely. Removes existing link if necessary."""
    if os.path.exists(dst) or os.path.islink(dst):
        os.remove(dst)
    os.symlink(src, dst)

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    print("=== AVSplat-Sim: Pre-Training (ncore Conversion) ===")

    # 1. Path Resolution
    dataset_base = to_absolute_path(cfg.dataset.base_dir)
    input_masks_dir = to_absolute_path(cfg.input_masks_dir)
    workspace_dir = HydraConfig.get().runtime.output_dir
    
    # 2. Setup Directories
    staging_dir = os.path.join(workspace_dir, "staging_symlinks")
    final_output_dir = os.path.join(workspace_dir, "ncore_dataset")

    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)

    print(f"\n--- 🏗️ Building Virtual Symlink Farm ---")
    
    create_symlink(os.path.join(dataset_base, "images"), os.path.join(staging_dir, "images"))
    create_symlink(input_masks_dir, os.path.join(staging_dir, "masks"))
    
    os.makedirs(os.path.join(staging_dir, "sparse"), exist_ok=True)
    
    wayve_sparse = os.path.join(dataset_base, "colmap_sparse", "rig")
    standard_sparse = os.path.join(dataset_base, "sparse", "0")
    
    if os.path.exists(wayve_sparse):
        print("  -> Detected Wayve101 format. Mapping colmap_sparse/rig to sparse/0")
        create_symlink(wayve_sparse, os.path.join(staging_dir, "sparse", "0"))
    elif os.path.exists(standard_sparse):
        print("  -> Detected standard COLMAP format. Mapping sparse/0 to sparse/0")
        create_symlink(standard_sparse, os.path.join(staging_dir, "sparse", "0"))
    else:
        raise FileNotFoundError(f"Could not find valid COLMAP data in {dataset_base}")

    print("✅ Symlink farm built.")

    # 3. Execute the External ncore Converter
    print(f"\n--- 🚀 Running ncore Converter ---")
    
    # Resolve the absolute path to the ncore repository
    ncore_root = to_absolute_path("external/ncore")
    
    cmd = [
        sys.executable,
        "-m", "tools.data_converter.colmap.converter",
        "--root-dir", staging_dir,        
        "--output-dir", final_output_dir, 
        "colmap-v4", 
        "--masks-dir", "masks"            
    ]

    try:
        subprocess.run(cmd, env=os.environ.copy(), cwd=ncore_root, check=True)
        
        # ==========================================
        # 🧹 The Cleanup: Flatten the nested directory
        # ==========================================
        generated_nested_dir = os.path.join(final_output_dir, "staging_symlinks")
        
        if os.path.exists(generated_nested_dir):
            # Move all files/folders up one level
            for item in os.listdir(generated_nested_dir):
                shutil.move(os.path.join(generated_nested_dir, item), final_output_dir)
            # Delete the now-empty 'staging_symlinks' folder
            os.rmdir(generated_nested_dir)
            
        # Note: No .success marker written here anymore! The drone just finishes its job.
        master_ncore_success = os.path.join(workspace_dir, ".success")
        with open(master_ncore_success, "w") as f:
                f.write("Dataset successfully converted to ncore format.")

        print(f"\n🎉 Conversion Worker Finished! Final ncore dataset ready at:\n{final_output_dir}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FATAL ERROR running converter: {e}")
        exit(1)

if __name__ == "__main__":
    main()