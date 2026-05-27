import sys
import gc
import torch
import imageio
from pathlib import Path
from tqdm import tqdm
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

from src.model_wrappers.base import BaseDiffusionModel

# 1. Dynamically find the root of your AVSplat-Sim workspace
# (Resolves from: src/model_wrappers/diffusion/difix_wrapper.py -> up 3 levels)
workspace_root = Path(__file__).resolve().parents[3]

# 2. Point to the specific submodule directory containing pipeline_difix.py
difix_submodule_path = workspace_root / "external" / "Difix3D+" / "src"

# 3. Add it to the system path so Python can discover it
if str(difix_submodule_path) not in sys.path:
    sys.path.insert(0, str(difix_submodule_path))

# 4. Safely import directly from the submodule!
from pipeline_difix import DifixPipeline

class DifixWrapper(BaseDiffusionModel):
    def __init__(self, model_id: str = "nvidia/difix", name: str = "difix", torch_dtype=torch.float16, default_timestep: int = 199, max_width: int = 1280, max_height: int = 720):
        """
        Initializes the Difix3D+ model once when the class is instantiated.
        """
        self.model_id = model_id
        self.name = name
        self.default_timestep = default_timestep
        self.max_width = max_width
        self.max_height = max_height
        print(f"Initializing {self.name} Wrapper from {self.model_id} (Default Timestep/Noise: {self.default_timestep})...")
        
        self.pipe = DifixPipeline.from_pretrained(
            self.model_id, 
            trust_remote_code=True, 
            torch_dtype=torch_dtype
        )
        
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.vae.enable_slicing()
        #self.pipe.vae.enable_tiling() # Breaks model
        
        print(f"✅ {self.name} loaded successfully with memory optimizations.")

    # 👇 Added timestep parameter here as well
    def process_frames(self, input_dir: str, output_dir: str, prompt: str = "remove degradation", timestep: int = None, video_path: str = None):
        """
        Processes a single camera directory frame-by-frame.
        """
        # Fallback to the class default if no specific timestep is passed
        current_timestep = timestep if timestep is not None else self.default_timestep
        
        input_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        extensions = ("*.png", "*.jpg", "*.jpeg")
        image_files = []
        for ext in extensions:
            image_files.extend(list(input_path.glob(f"**/{ext}")))
        image_files.sort()

        if not image_files:
            print(f"⚠️ No images found in {input_dir}")
            return

        print(f"\n🎬 Processing {len(image_files)} frames for camera: {input_path.name} | Timestep: {current_timestep}")
        processed_frames = []

        try:
            with torch.inference_mode():
                for img_path in tqdm(image_files, desc=f"Cleaning {input_path.name}"):
                    rel_path = img_path.relative_to(input_path)
                    save_path = out_path / rel_path
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    input_image = load_image(str(img_path))
                    w, h = input_image.size
                    scale = min(self.max_width / w, self.max_height / h)
                    if scale < 1.0:
                        # Snap to nearest multiple of 8 (VAE requirement)
                        new_w = (int(w * scale) // 8) * 8
                        new_h = (int(h * scale) // 8) * 8
                        input_image = input_image.resize((new_w, new_h))
                    elif (w % 8 != 0) or (h % 8 != 0):
                        input_image = input_image.resize((w // 8 * 8, h // 8 * 8))
                    
                    output = self.pipe(
                        prompt, 
                        image=input_image, 
                        num_inference_steps=1, 
                        timesteps=[current_timestep],  
                        guidance_scale=0.0
                    ).images[0]
                    
                    output.save(str(save_path))
                    processed_frames.append(save_path)
                    
                    # aggressive cleanup block!
                    del output
                    del input_image
                    gc.collect()
                    torch.cuda.empty_cache()
                    
        except KeyboardInterrupt:
            print(f"\n⚠️ Interrupted while processing {input_path.name}.")
            
        if video_path and processed_frames:
            self._generate_video(processed_frames, video_path)

    def _generate_video(self, frames, video_path):
        """Helper method to stitch frames into an mp4."""
        vid_path = Path(video_path)
        vid_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"🎞️ Generating video: {vid_path}")
        
        frames.sort()
        with imageio.get_writer(str(vid_path), fps=30, quality=8, macro_block_size=1) as writer:
            for frame_path in tqdm(frames, desc="Encoding Video", leave=False):
                img = imageio.imread(frame_path)
                writer.append_data(img)
        print(f"✅ Video saved to {vid_path}")