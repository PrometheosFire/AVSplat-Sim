import os
import shutil
import subprocess
import sys
import time
import hydra
import numpy as np
import pycolmap
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


def _fmt_hms(seconds):
    """Format a duration as HH:MM:SS. Hours accumulate past 24 rather than wrap."""
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _stamp_success(path, message, elapsed, breakdown=None):
    """Write a .success marker carrying its runtime and optional sub-block times.

    Nothing in the pipeline reads these files -- only their existence is checked
    -- so the extra lines are free to grow.
    """
    lines = [message, f"duration: {_fmt_hms(elapsed)}"]
    if breakdown:
        width = max(len(label) for label, _ in breakdown)
        for label, value in breakdown:
            shown = value if isinstance(value, str) else _fmt_hms(value)
            lines.append(f"  {label:<{width}}  {shown}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


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
    masks_cfg = OmegaConf.select(cfg, "input_masks_dir")
    input_masks_dir = to_absolute_path(masks_cfg) if masks_cfg else None
    workspace_dir = HydraConfig.get().runtime.output_dir
    
    # 2. Setup Directories
    staging_dir = os.path.join(workspace_dir, "staging_symlinks")
    final_output_dir = os.path.join(workspace_dir, "ncore_dataset")

    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)

    print(f"\n--- 🏗️ Building Virtual Symlink Farm ---")

    t_start = time.perf_counter()
    t_phase = t_start
    phases = []

    create_symlink(os.path.join(dataset_base, "images"), os.path.join(staging_dir, "images"))
    if input_masks_dir:
        create_symlink(input_masks_dir, os.path.join(staging_dir, "masks"))
    else:
        print("  -> No input_masks_dir provided; converting without segmentation masks.")
    
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
    phases.append(("symlink farm", time.perf_counter() - t_phase))
    t_phase = time.perf_counter()

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
    ]
    if input_masks_dir:
        cmd += ["--masks-dir", "masks"]

    try:
        subprocess.run(cmd, env=os.environ.copy(), cwd=ncore_root, check=True)
        
        # Cleanup: Flatten the nested directory
        generated_nested_dir = os.path.join(final_output_dir, "staging_symlinks")
        
        if os.path.exists(generated_nested_dir):
            # Move all files/folders up one level
            for item in os.listdir(generated_nested_dir):
                shutil.move(os.path.join(generated_nested_dir, item), final_output_dir)
            # Delete the now-empty 'staging_symlinks' folder
            os.rmdir(generated_nested_dir)
        
        phases.append(("ncore converter", time.perf_counter() - t_phase))
        t_phase = time.perf_counter()

        # ==========================================
        # 📷 Extract, Map, and Sort Camera IDs
        # ==========================================
        print("\n--- 📷 Extracting and Sorting Camera IDs ---")
        
        sparse_path_to_read = wayve_sparse if os.path.exists(wayve_sparse) else standard_sparse
        # Defensible Loading: Handle pycolmap version discrepancies gracefully
        try:
            recon = pycolmap.Reconstruction(sparse_path_to_read)
            images_dict = recon.images
        except AttributeError:
            # Fallback for older pycolmap versions (which gsplat/ncore often pin)
            recon = pycolmap.SceneManager(str(sparse_path_to_read))
            recon.load_cameras()
            recon.load_images()
            images_dict = recon.images
        
        # 1. Map ncore name -> semantic name
        ncore_to_semantic = {}
        for _, image in images_dict.items():
            cam_id = image.camera_id
            ncore_name = f"camera{cam_id}"
            
            if ncore_name not in ncore_to_semantic:
                semantic_name = os.path.dirname(image.name)
                if not semantic_name:
                    semantic_name = image.name.split('_')[0]
                ncore_to_semantic[ncore_name] = semantic_name

        print("Detected Mapping:")
        for ncore_cam, semantic_cam in ncore_to_semantic.items():
            print(f"  {ncore_cam: <10} --> {semantic_cam}")

        # 2. Sort based on the target order defined in Hydra
        target_order = cfg.dataset.cameras
        semantic_to_ncore = {v: k for k, v in ncore_to_semantic.items()}
        
        sorted_camera_names = []
        for semantic_name in target_order:
            if semantic_name in semantic_to_ncore:
                sorted_camera_names.append(semantic_to_ncore[semantic_name])
            else:
                print(f"⚠️ Warning: Expected camera '{semantic_name}' not found in COLMAP data.")

        # 3. Fallback: Append any extra cameras that weren't in the config list
        for ncore_name in ncore_to_semantic.keys():
            if ncore_name not in sorted_camera_names:
                sorted_camera_names.append(ncore_name)

        # 4. Save the perfectly ordered list for the Orchestrator
        cam_list_path = os.path.join(final_output_dir, "camera_list.yaml")
        OmegaConf.save({"ncore_camera_ids": sorted_camera_names}, cam_list_path)
        
        print(f"\n✅ Saved {len(sorted_camera_names)} cameras in optimal order: {sorted_camera_names}")

        phases.append(("camera ids", time.perf_counter() - t_phase))
        t_phase = time.perf_counter()

        # ==========================================
        # 🎯 Extract Point Visibility for Depth Loss
        # ==========================================
        print("\n--- 🎯 Extracting COLMAP Point Visibility ---")

        try:
            sm = pycolmap.SceneManager(str(sparse_path_to_read))
            sm.load_cameras()
            sm.load_images()
            sm.load_points3D()
        except Exception:
            sm = None

        if sm is not None and sm.points3D is not None and len(sm.points3D) > 0:
            INVALID_ID = np.iinfo(np.uint64).max
            points_3d = sm.points3D.astype(np.float32)

            # Group images by ncore camera name, sorted by filename (= timestamp order)
            images_by_cam = {}
            for _, image in sm.images.items():
                ncore_cam = f"camera{image.camera_id}"
                images_by_cam.setdefault(ncore_cam, []).append(image)

            for cam in images_by_cam:
                images_by_cam[cam].sort(key=lambda img: img.name)

            # Build CSR-style visibility per camera:
            # offsets[i] = start index for frame i, offsets[i+1] = end
            # indices = concatenated point indices for all frames
            save_dict = {"points_3d": points_3d}

            for ncore_cam, images in images_by_cam.items():
                offsets = [0]
                all_indices = []

                for image in images:
                    point3d_ids = np.array(image.point3D_ids, dtype=np.uint64)
                    valid_mask = point3d_ids != INVALID_ID
                    valid_ids = point3d_ids[valid_mask]

                    frame_indices = []
                    for pid in valid_ids:
                        pid_int = int(pid)
                        if pid_int in sm.point3D_id_to_point3D_idx:
                            frame_indices.append(sm.point3D_id_to_point3D_idx[pid_int])

                    all_indices.extend(frame_indices)
                    offsets.append(len(all_indices))

                save_dict[f"{ncore_cam}_offsets"] = np.array(offsets, dtype=np.int32)
                save_dict[f"{ncore_cam}_indices"] = np.array(all_indices, dtype=np.int32)
                print(f"  {ncore_cam}: {len(images)} frames, {len(all_indices)} total visible points")

            vis_path = os.path.join(final_output_dir, "point_visibility.npz")
            np.savez_compressed(vis_path, **save_dict)
            print(f"✅ Point visibility saved to {vis_path}")
        else:
            print("⚠️ Could not extract point visibility (no points3D found)")

        print(f"🎉 Conversion Worker Finished! Final ncore dataset ready at:\n{final_output_dir}")
        
        
        phases.append(("point visibility", time.perf_counter() - t_phase))

        # Note: No .success marker written here anymore! The drone just finishes its job.
        master_ncore_success = os.path.join(workspace_dir, ".success")
        _stamp_success(
            master_ncore_success,
            "Dataset successfully converted to ncore format.",
            time.perf_counter() - t_start,
            phases,
        )

        print(f"\n🎉 Conversion Worker Finished! Final ncore dataset ready at:\n{final_output_dir}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FATAL ERROR running converter: {e}")
        exit(1)

if __name__ == "__main__":
    main()