import glob
import os
import shutil
import torch
import cv2
import numpy as np

# Apply memory optimizations before loading SAM 3
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import prepare_masks_for_visualization

# Import your strict interface
from src.model_wrappers.base import BaseSegmenter


class SAM3Wrapper(BaseSegmenter):
    def __init__(self, checkpoint_path: str, chunk_size: int = 2, name: str = "sam3"):
        """
        Initializes the SAM 3 model once when the class is instantiated.
        This prevents reloading the heavy weights into the 8GB VRAM 
        for every new camera feed.
        """
        self.checkpoint_path = checkpoint_path
        self.chunk_size = chunk_size
        self.name = name
        print(f"Initializing SAM 3 Wrapper from {self.checkpoint_path}...")

        # Limit to the single RTX 4060 Max-Q
        gpus_to_use = [torch.cuda.current_device()]
        
        self.predictor = build_sam3_video_predictor(
            gpus_to_use=gpus_to_use, 
            checkpoint_path=self.checkpoint_path
        )

    def extract_masks(self, input_dir: str, output_dir: str, prompt: str):
        """
        Fulfills the BaseSegmenter contract. Processes a single directory of images.
        """
        print(f"\n🎥 Extracting masks from: {input_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Grab frames for this specific camera directory
        all_frames = sorted(glob.glob(os.path.join(input_dir, "*.jpeg")))
        
        if not all_frames:
            print(f"⚠️ WARNING: No .jpeg frames found in {input_dir}. Skipping.")
            return
            
        print(f"Found {len(all_frames)} frames. Saving masks to {output_dir}")

        # Temporary chunking directory specific to this run
        temp_dir = os.path.join(output_dir, "temp_sam_chunk")

        # ==========================================
        # File-System Chunking Loop
        # ==========================================
        for chunk_start in range(0, len(all_frames), self.chunk_size):
            chunk_frames = all_frames[chunk_start : chunk_start + self.chunk_size]
            
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            
            for frame_path in chunk_frames:
                shutil.copy(frame_path, temp_dir)
                
            print(f"  -> Chunk: Frames {chunk_start} to {chunk_start + len(chunk_frames) - 1}")
            
            # Start session
            response = self.predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=temp_dir,
                    offload_video_to_cpu=True, 
                    offload_state_to_cpu=True, 
                )
            )
            session_id = response["session_id"]

            # Inject the prompt from the contract argument
            _ = self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,  
                    text=prompt,
                )
            )

            # Inference Block
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for stream_response in self.predictor.handle_stream_request(
                    request=dict(type="propagate_in_video", session_id=session_id)
                ):
                    temp_frame_index = stream_response["frame_index"]
                    absolute_frame_index = chunk_start + temp_frame_index
                    
                    outputs = stream_response["outputs"] 
                    dict_for_meta = {temp_frame_index: outputs}
                    formatted_frame_data = prepare_masks_for_visualization(dict_for_meta)
                    clean_outputs = formatted_frame_data[temp_frame_index]
                    
                    combined_mask = None 
                    
                    for obj_id, mask_np in clean_outputs.items():
                        mask_np = np.squeeze(mask_np) 
                        binary_mask = np.where(mask_np > 0, 255, 0).astype(np.uint8)
                        
                        if combined_mask is None:
                            combined_mask = np.zeros_like(binary_mask)
                            
                        if combined_mask.shape != binary_mask.shape:
                            binary_mask = cv2.resize(
                                binary_mask, 
                                (combined_mask.shape[1], combined_mask.shape[0]), 
                                interpolation=cv2.INTER_NEAREST
                            )
                            
                        combined_mask = cv2.bitwise_or(combined_mask, binary_mask)
                        
                    if combined_mask is not None:
                        inverted_mask = cv2.bitwise_not(combined_mask)
                        
                        # 1. Grab the original filepath from the chunk list
                        original_frame_path = chunk_frames[temp_frame_index]
                        
                        # 2. Extract just the filename (e.g., '1711356121983309')
                        base_name = os.path.splitext(os.path.basename(original_frame_path))[0]
                        
                        # 3. Save as .png with the exact original name
                        save_path = os.path.join(output_dir, f"{base_name}.png")
                        cv2.imwrite(save_path, inverted_mask)

                    del outputs, dict_for_meta, formatted_frame_data, clean_outputs
                    torch.cuda.empty_cache()

            # Close session to free memory for the next chunk
            _ = self.predictor.handle_request(request=dict(type="close_session", session_id=session_id))
            torch.cuda.empty_cache()

        # Clean up temporary folder after all chunks finish
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print(f"✅ Extraction complete for {input_dir}")
        
        
if __name__ == "__main__":
    print("=== Commencing Isolated SAM3Wrapper Test ===")
    
    # We define paths relative to the root of AVSplat-Sim
    checkpoint = "external_weights/sam3/sam3.pt"
    test_input = "test_dummy_frames"
    test_output = "test_dummy_masks"

    if not os.path.exists(checkpoint):
        print(f"❌ Error: Weights not found at {checkpoint}")
        print("Please download them via Hugging Face CLI first!")
        exit(1)

    print("🖼️ Generating dummy images for testing...")
    os.makedirs(test_input, exist_ok=True)
    
    # Create a blank black image
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Draw a green square in the middle
    cv2.rectangle(dummy_img, (500, 300), (700, 500), (0, 255, 0), -1) 
    
    # Save two frames to test the chunking logic
    cv2.imwrite(os.path.join(test_input, "frame_0001.jpeg"), dummy_img)
    cv2.imwrite(os.path.join(test_input, "frame_0002.jpeg"), dummy_img)

    try:
        # 1. Test Initialization
        wrapper = SAM3Wrapper(checkpoint_path=checkpoint, chunk_size=2)

        # 2. Test Inference
        wrapper.extract_masks(
            input_dir=test_input, 
            output_dir=test_output, 
            prompt="green square"
        )

        print(f"\n✅ Isolated test passed successfully! Check '{test_output}' for masks.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")