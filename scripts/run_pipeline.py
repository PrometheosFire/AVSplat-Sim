import os
import json
import hashlib
import subprocess
import hydra
from omegaconf import DictConfig, OmegaConf

def generate_config_hash(config_subset: dict) -> str:
    """Generates a unique 8-character hash from a dictionary."""
    config_str = json.dumps(config_subset, sort_keys=True)
    return hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("🚀 Starting Smart AVSplat-Sim Pipeline...")
    
    # Resolve the config to standard dictionaries for hashing
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    pipeline_cfg = OmegaConf.to_container(cfg.pipeline, resolve=True)
    
    # Base directory for this specific dataset and scene
    base_results_dir = os.path.abspath(f"results/{dataset_cfg['name']}/{dataset_cfg['scene']}")
    os.makedirs(base_results_dir, exist_ok=True)

    # ==========================================
    # STEP 1: PRE-TRAINING (MASK EXTRACTION)
    # ==========================================
    print("\n" + "="*50)
    # Dynamically print the name of the wrapper being used
    wrapper_name = cfg.model._target_.split('.')[-1]
    print(f"▶️ [STEP 1] Checking Pre-Training Extraction ({wrapper_name})...")
    
    # Create a fingerprint based on dataset, model, and pipeline configs
    step1_params = {"dataset": dataset_cfg, "model": model_cfg, "pipeline": pipeline_cfg}
    step1_hash = generate_config_hash(step1_params)
    
    masks_dir = os.path.join(base_results_dir, f"01_masks_{step1_hash}")
    success_marker = os.path.join(masks_dir, "_SUCCESS")
    
    if os.path.exists(success_marker):
        print(f"✅ Cache Hit! Exact configuration already run.")
        print(f"📂 Reusing masks from: {masks_dir}")
    else:
        print(f"🔄 New configuration detected (Hash: {step1_hash}). Running extraction...")
        
        # Point to the isolated segmentation environment
        python_exec = os.path.abspath("envs/env_segmentation/bin/python")
        
        # Point to our new, model-agnostic script location
        script_path = os.path.abspath("src/pre_training/extract_masks.py")
        
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.abspath(".")
        
        subprocess.run([
            python_exec, script_path, 
            f"hydra.run.dir={masks_dir}"
        ], env=env, check=True)
        print(f"✅ Extraction saved to {masks_dir}")

if __name__ == "__main__":
    main()