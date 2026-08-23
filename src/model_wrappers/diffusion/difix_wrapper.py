import sys
import gc
import torch
import imageio
from pathlib import Path
from PIL import Image
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
    def __init__(self, model_id: str = "nvidia/difix", name: str = "difix", torch_dtype=torch.float16, default_timestep: int = 199, max_width: int = 1280, max_height: int = 720, use_ref: bool = False, enable_vae_slicing: bool = True):
        """
        Initializes the Difix3D+ model once when the class is instantiated.
        """
        self.model_id = model_id
        self.name = name
        self.default_timestep = default_timestep
        self.max_width = max_width
        self.max_height = max_height
        self.use_ref = use_ref
        print(f"Initializing {self.name} Wrapper from {self.model_id} (Default Timestep/Noise: {self.default_timestep})...")
        
        self.pipe = DifixPipeline.from_pretrained(
            self.model_id, 
            trust_remote_code=True, 
            torch_dtype=torch_dtype
        )
        
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()
        # VAE slicing MUST stay off when a reference image is used. With a ref the
        # pipeline encodes torch.cat([image, ref_image], dim=0) as a batch of 2;
        # diffusers' slicing path encodes one sample at a time, and Difix's encoder
        # keeps only the LAST call's skip activations
        # (external/Difix3D+/src/model.py: `self.current_down_blocks = l_blocks`).
        # The decoder would then reconstruct using the REFERENCE image's skips.
        # Silent corruption, no exception raised.
        if enable_vae_slicing and not use_ref:
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

    def _snap_size(self, w: int, h: int):
        """Fit under the max_* caps, then snap to a multiple of 8 (VAE requirement)."""
        scale = min(self.max_width / w, self.max_height / h)
        if scale < 1.0:
            w, h = int(w * scale), int(h * scale)
        return (w // 8) * 8, (h // 8) * 8

    def process_pairs(self, jobs, prompt: str = "remove degradation",
                      timestep: int = None, skip_existing: bool = True):
        """Reference-conditioned cleaning of explicit (input, ref, output) triples.

        Unlike process_frames, this does not walk a directory -- the caller decides
        exactly which frames to process (subsampling, val exclusion).

        Args:
            jobs: iterable of dicts with keys input_path, ref_path (may be None),
                output_path.
            prompt: diffusion prompt.
            timestep: noise level; falls back to self.default_timestep.
            skip_existing: skip a job whose output is newer than its input.

        Returns:
            List of per-job dicts: {input_path, ref_path, output_path,
            original_size, model_size, skipped}.
        """
        current_timestep = timestep if timestep is not None else self.default_timestep
        jobs = list(jobs)
        results = []

        try:
            with torch.inference_mode():
                for job in tqdm(jobs, desc=f"Difix (ref={self.use_ref})"):
                    in_path, out_path = Path(job["input_path"]), Path(job["output_path"])
                    ref_path = job.get("ref_path")

                    if (skip_existing and out_path.exists()
                            and out_path.stat().st_mtime >= in_path.stat().st_mtime):
                        results.append({"input_path": str(in_path), "ref_path": ref_path,
                                        "output_path": str(out_path), "skipped": True})
                        continue

                    out_path.parent.mkdir(parents=True, exist_ok=True)

                    image = load_image(str(in_path))
                    original_size = image.size            # (w, h) before snapping
                    model_size = self._snap_size(*original_size)
                    if model_size != original_size:
                        image = image.resize(model_size)

                    ref_image = None
                    if self.use_ref and ref_path is not None:
                        ref_image = load_image(str(ref_path))
                        # The pipeline concatenates image and ref along the batch
                        # dim, so the ref must match the input's post-snap size
                        # exactly. Real frames are 1920x1080, renders 960x540 --
                        # without this the cat raises.
                        if ref_image.size != model_size:
                            ref_image = ref_image.resize(model_size)

                    output = self.pipe(
                        prompt, image=image, ref_image=ref_image,
                        num_inference_steps=1, timesteps=[current_timestep],
                        guidance_scale=0.0,
                    ).images[0]

                    # Restore the render's exact resolution so the cleaned frame
                    # stays consistent with the camera intrinsics: 960x540 snaps
                    # down to 960x536 and would otherwise silently mismatch K.
                    if output.size != original_size:
                        output = output.resize(original_size, Image.LANCZOS)

                    output.save(str(out_path))
                    results.append({"input_path": str(in_path), "ref_path": ref_path,
                                    "output_path": str(out_path), "skipped": False,
                                    "original_size": original_size, "model_size": model_size})

                    del output, image, ref_image
                    gc.collect()
                    torch.cuda.empty_cache()
        except KeyboardInterrupt:
            print(f"\n⚠️ Interrupted after {len(results)}/{len(jobs)} jobs. Re-run to resume.")

        return results

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