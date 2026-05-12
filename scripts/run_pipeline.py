import os
import json
import hashlib
import subprocess
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

def generate_config_hash(config_subset: dict) -> str:
    """Generates a unique 8-character hash from a dictionary."""
    config_str = json.dumps(config_subset, sort_keys=True)
    return hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("🚀 Starting Smart AVSplat-Sim Pipeline...")
    
    # 1. Grab any custom overrides the user typed in the CLI (e.g., ['model.chunk_size=1'])
    overrides = HydraConfig.get().overrides.task
    
    # Resolve the config to standard dictionaries for hashing
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    segmenter_cfg = OmegaConf.to_container(cfg.segmenter, resolve=True) # 👈 Updated
    seg_task_cfg = OmegaConf.to_container(cfg.seg_task, resolve=True)
    
    # Base directory for this specific dataset and scene
    base_results_dir = os.path.abspath(f"results/{dataset_cfg['name']}/{dataset_cfg['scene']}")
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Point to the isolated segmentation environment
    python_exec = os.path.abspath("envs/env_segmentation/bin/python")
    
    # Point to our new, model-agnostic script location
    script_path = os.path.abspath("src/pre_training/extract_masks.py")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".")

    # ==========================================
    # STEP 1: PRE-TRAINING (MASK EXTRACTION)
    # ==========================================
    print("\n" + "="*50)
    # Dynamically print the name of the wrapper being used
    wrapper_name = cfg.segmenter._target_.split('.')[-1]
    print(f"▶️ [STEP 1] Checking Pre-Training Extraction ({wrapper_name})...")
    
    # Create a fingerprint based on dataset, model, and pipeline configs
    step1_params = {"dataset": dataset_cfg, "model": segmenter_cfg, "pipeline": seg_task_cfg}
    step1_hash = generate_config_hash(step1_params)
    
    masks_dir = os.path.join(base_results_dir, f"01_masks_{step1_hash}")
    success_marker = os.path.join(masks_dir, ".success")
    
    if os.path.exists(success_marker):
        print(f"✅ Cache Hit! Exact configuration already run.")
        print(f"📂 Reusing masks from: {masks_dir}")
    else:
        print(f"🔄 New configuration detected (Hash: {step1_hash}). Running extraction...")
        
        
        
        subprocess.run([
            python_exec, script_path, 
            f"hydra.run.dir={masks_dir}",
            *overrides
        ], env=env, check=True)
        print(f"✅ Extraction saved to {masks_dir}")
        
    # ==========================================
    # STEP 2: PRE-TRAINING (MASK FUSION & DILATION)
    # ==========================================
    
    # 1. Create a unique hash for the Fusion step
    fuse_config_str = f"{masks_dir}_{cfg.mask_processing.dilation_percentage}_{cfg.mask_processing.dilate_ego}_{cfg.mask_processing.use_ego_masks}"
    fuse_hash = hashlib.md5(fuse_config_str.encode()).hexdigest()[:8]
    
    # Define the new, isolated output directory
    fused_masks_dir = os.path.abspath(os.path.join(base_results_dir, f"02_fused_masks_{fuse_hash}"))
    master_fuse_success = os.path.join(fused_masks_dir, ".success")

    # 2. Top-Tier Check: Skip if already done
    if os.path.exists(master_fuse_success):
        print(f"⏭️ Skipping Step 2: Fused masks already exist at {fused_masks_dir}")
    else:
        print(f"\n🚀 Running Step 2: Fusing Ego Masks & Dilating...")
        
        fuse_script_path = os.path.abspath("src/pre_training/fuse_masks.py")
        
        # 3. The Clean Subprocess Call
        subprocess.run([
            python_exec, 
            fuse_script_path, 
            f"hydra.run.dir={fused_masks_dir}", # Sets the output to the new hash folder
            f"+input_masks_dir={masks_dir}",    # Passes the Step 1 folder as input
            *overrides                          # Your clean unpacking syntax!
        ], env=env, check=True)
        
        print(f"✅ Fusion and Dilation complete inside {fused_masks_dir}")


    # ==========================================
    # STEP 3: Convert to ncore Dataset
    # ==========================================
    
    # Point to the isolated segmentation environment
    python_exec = os.path.abspath("envs/envs/env_gsplat/bin/python")
    
    # Point to our new, model-agnostic script location
    #script_path = os.path.abspath("src/pre_training/extract_masks.py")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".")
    
    ncore_config_str = f"ncore_{fused_masks_dir}"
    ncore_hash = hashlib.md5(ncore_config_str.encode()).hexdigest()[:8]
    
    ncore_workspace_dir = os.path.abspath(os.path.join(base_results_dir, f"03_ncore_dataset_{ncore_hash}"))
    master_ncore_success = os.path.join(ncore_workspace_dir, ".success")

    if os.path.exists(master_ncore_success):
            print(f"⏭️ Skipping Step 3: ncore dataset already exists at {ncore_workspace_dir}")
    else:
            print(f"\n🚀 Running Step 3: Converting to ncore format...")
            
            script_path = os.path.abspath("src/pre_training/convert_ncore.py")
            
            # 1. Launch the drone
            subprocess.run([
                python_exec, 
                script_path, 
                f"hydra.run.dir={ncore_workspace_dir}", 
                f"+input_masks_dir={fused_masks_dir}", 
                *overrides
            ], env=env, check=True)            
                
            print(f"✅ Conversion complete. Master marker written. Pipeline ready for GSplat Training!")
            
    # Dynamically load the auto-extracted camera list!
    cam_list_path = os.path.join(ncore_workspace_dir, "ncore_dataset", "camera_list.yaml")
    if os.path.exists(cam_list_path):
        cam_config = OmegaConf.load(cam_list_path)
        cam_list_str = ",".join(cam_config.ncore_camera_ids)
        # Creates a string like: gaussian_splatting.ncore_camera_ids=[camera1,camera8,camera10]
        cam_override = f"++gaussian_splatting.ncore_camera_ids=[{cam_list_str}]"
    else:
        cam_override = "++gaussian_splatting.ncore_camera_ids=[]"
            
    # ==========================================
    # STEP 4: Gaussian Splatting Training
    # ==========================================
    gsplat_config_yaml = OmegaConf.to_yaml(cfg.gaussian_splatting)
    gsplat_config_str = f"gsplat_{ncore_workspace_dir}_{gsplat_config_yaml}"
    gsplat_hash = hashlib.md5(gsplat_config_str.encode()).hexdigest()[:8]
    
    # Define the isolated output directory for this specific training run
    gsplat_workspace_dir = os.path.abspath(os.path.join(base_results_dir, f"04_gsplat_training_{gsplat_hash}"))
    master_gsplat_success = os.path.join(gsplat_workspace_dir, ".success")

    if os.path.exists(master_gsplat_success):
             print(f"⏭️ Skipping Step 4: GSplat training already completed for this config at {gsplat_workspace_dir}")
    else:
            print(f"\n🚀 Running Step 4: Gaussian Splatting Training...")
            
            gsplat_script_path = os.path.abspath("src/gsplat_training/train_splats.py")
            ncore_json_path = os.path.join(ncore_workspace_dir, "ncore_dataset", "staging_symlinks.json")
            #gsplat_python_exec = os.path.abspath("envs/envs/env_gsplat/bin/python")
            
            
            # Launch the training!
            subprocess.run([
                python_exec, 
                gsplat_script_path, 
                f"hydra.run.dir={gsplat_workspace_dir}",                  
                f"++gaussian_splatting.data_dir={ncore_json_path}",         
                f"++gaussian_splatting.result_dir={gsplat_workspace_dir}",  
                cam_override,  
                *overrides
            ], env=env, check=True)
                
            print(f"\n Gaussian Splatting Training complete.")
            print(f"📂 Results saved to: {gsplat_workspace_dir}")
            
            
    # ==========================================
    # STEP 5: Diffusion Post-Processing (Difix3D+)
    # ==========================================
    
    diffusion_cfg = OmegaConf.to_container(cfg.diffusion, resolve=True)
    diff_task_cfg = OmegaConf.to_container(cfg.diff_task, resolve=True)
    
    diff_config_str = f"diff_{gsplat_workspace_dir}_{diffusion_cfg}_{diff_task_cfg}"
    diff_hash = hashlib.md5(diff_config_str.encode()).hexdigest()[:8]
    
    # 2. Define the isolated output directory for this specific cleanup run
    difix_workspace_dir = os.path.abspath(os.path.join(base_results_dir, f"05_difix_cleaned_{diff_hash}"))
    master_difix_success = os.path.join(difix_workspace_dir, ".success")

    if os.path.exists(master_difix_success):
            print(f"⏭️ Skipping Step 5: Diffusion already completed for this config at {difix_workspace_dir}")
    else:
            print(f"\n🚀 Running Step 5: Diffusion Post-Processing...")
            
            difix_script_path = os.path.abspath("src/post_processing/apply_diffusion.py")
            
            # Note: We reuse the gsplat python environment because Difix is installed there!
            subprocess.run([
                python_exec, 
                difix_script_path, 
                f"++diff_task.output_dir={difix_workspace_dir}",          # Force the output directory
                f"++gaussian_splatting.result_dir={gsplat_workspace_dir}",  # Tell it where the renders are
                cam_override,
                *overrides
            ], env=env, check=True)
            
            # The Orchestrator stamps the run as successful
            with open(master_difix_success, "w") as f:
                f.write("Diffusion post-processing completed successfully.")
                
            print(f"\n🎉 Veni, Vidi, Vici! Diffusion Post-Processing complete.")
            print(f"📂 Cleaned renders saved to: {difix_workspace_dir}")
    

if __name__ == "__main__":
    main()