import os

import hydra
from omegaconf import DictConfig, OmegaConf
from gsplat.distributed import cli

# Import your existing classes from wherever you saved them
from src.gsplat_training.config_training import Config 
from src.gsplat_training.runner import main as runner_main
from gsplat.strategy import DefaultStrategy, MCMCStrategy

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    print("🚀 Booting up GSplat Training...")

    # 1. Convert the Hydra config segment into a standard Python dictionary
    #    resolve=True evaluates any interpolations (like ${dataset.name})
    gsplat_dict = OmegaConf.to_container(cfg.gaussian_splatting, resolve=True)

    # 2. Handle the Strategy Instantiation securely
    strategy_type = gsplat_dict.pop("strategy_type", "default")
    
    # Safely extract the strategy parameters from Hydra (defaults to empty dict if missing)
    strategy_kwargs = gsplat_dict.pop("strategy", {})
    
    # We still want verbose on for the logs!
    strategy_kwargs["verbose"] = True 

    if strategy_type == "mcmc":
        # Unpack (**kwargs) all the parameters from Hydra straight into the class!
        gsplat_dict["strategy"] = MCMCStrategy(**strategy_kwargs)
    else:
        gsplat_dict["strategy"] = DefaultStrategy(**strategy_kwargs)

    # 3. The Magic Step: Instantiate your original dataclass!
    #    By unpacking the dict, the dataclass handles all defaults and type checking.
    cfg_obj = Config(**gsplat_dict)

    # 4. Safely call your adjust_steps function (since it's a real class object now)
    cfg_obj.adjust_steps(cfg_obj.steps_scaler)

    print(f"📊 Training initialized with {cfg_obj.max_steps} max steps.")
    print(f"💾 Saving results to: {cfg_obj.result_dir}")

    # 5. Launch the distributed training exactly like the original tyro script did
    #    This passes control over to gsplat's multi-GPU wrapper, which will 
    #    ultimately call `runner_main(local_rank, world_rank, world_size, cfg_obj)`
    cli(runner_main, cfg_obj, verbose=True)
    
    standalone_success_path = os.path.join(cfg_obj.result_dir, ".success")
    with open(standalone_success_path, "w") as f:
        f.write("GSplat training completed successfully.")
        
    print(f"✅ Standalone success marker written to {standalone_success_path}")

if __name__ == "__main__":
    main()