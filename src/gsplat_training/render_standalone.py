"""Standalone rendering script for trained Gaussian Splatting models.

This script renders frames from a trained GSplat model without requiring
the full training infrastructure. It loads Gaussian parameters from .ply
or .ckpt files and camera metadata from camera_data.json files.

Usage:
    python render_standalone.py \
        ply_path=/path/to/point_cloud.ply \
        camera_paths_dir=/path/to/camera_paths \
        output_dir=/path/to/output

    Or with checkpoint (for app_opt/post_processing):
    python render_standalone.py \
        checkpoint_path=/path/to/ckpt.pt \
        camera_paths_dir=/path/to/camera_paths \
        output_dir=/path/to/output
"""

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import imageio
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf, MISSING
from plyfile import PlyData

from gsplat.rendering import rasterization
from gsplat.cuda._wrapper import FThetaCameraDistortionParameters, FThetaPolynomialType

from trajectory import TrajectoryManipulator, TrajectoryShift


@dataclass
class CameraMetadata:
    """Camera metadata loaded from camera_data.json."""

    camera_id: str
    camera_index: int
    camtoworlds: np.ndarray  # [N, 4, 4]
    K: np.ndarray  # [3, 3]
    width: int
    height: int
    camera_model: str = "pinhole"
    radial_coeffs: Optional[np.ndarray] = None
    tangential_coeffs: Optional[np.ndarray] = None
    thin_prism_coeffs: Optional[np.ndarray] = None
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None
    world_to_normalized_scale: Optional[float] = None

    @classmethod
    def from_json(cls, json_path: str) -> "CameraMetadata":
        """Load camera metadata from camera_data.json file."""
        with open(json_path, "r") as f:
            data = json.load(f)

        camtoworlds = np.array(data["camtoworlds"], dtype=np.float32)
        # Ensure 4x4 format
        if camtoworlds.shape[-2:] == (3, 4):
            n = camtoworlds.shape[0]
            c2ws_4x4 = np.zeros((n, 4, 4), dtype=np.float32)
            c2ws_4x4[:, :3, :] = camtoworlds
            c2ws_4x4[:, 3, 3] = 1.0
            camtoworlds = c2ws_4x4

        # Parse intrinsics
        intrinsics = data.get("intrinsics", {})
        K = np.array(intrinsics.get("K", np.eye(3)), dtype=np.float32)
        width = intrinsics.get("width", 1920)
        height = intrinsics.get("height", 1080)

        # Parse distortion
        distortion = data.get("distortion", {})
        camera_model = distortion.get("camera_model", "pinhole")

        radial_coeffs = None
        if distortion.get("radial_coeffs") is not None:
            radial_coeffs = np.array(distortion["radial_coeffs"], dtype=np.float32)

        tangential_coeffs = None
        if distortion.get("tangential_coeffs") is not None:
            tangential_coeffs = np.array(
                distortion["tangential_coeffs"], dtype=np.float32
            )

        thin_prism_coeffs = None
        if distortion.get("thin_prism_coeffs") is not None:
            thin_prism_coeffs = np.array(
                distortion["thin_prism_coeffs"], dtype=np.float32
            )

        ftheta_coeffs = None
        if distortion.get("ftheta_coeffs") is not None:
            fc = distortion["ftheta_coeffs"]
            ref_poly = FThetaPolynomialType[fc["reference_poly"]]
            ftheta_coeffs = FThetaCameraDistortionParameters(
                reference_poly=ref_poly,
                pixeldist_to_angle_poly=tuple(fc["pixeldist_to_angle_poly"]),
                angle_to_pixeldist_poly=tuple(fc["angle_to_pixeldist_poly"]),
                max_angle=fc["max_angle"],
                linear_cde=tuple(fc["linear_cde"]),
            )

        return cls(
            camera_id=data["camera_id"],
            camera_index=data["camera_index"],
            camtoworlds=camtoworlds,
            K=K,
            width=width,
            height=height,
            camera_model=camera_model,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            ftheta_coeffs=ftheta_coeffs,
            world_to_normalized_scale=data.get("world_to_normalized_scale"),
        )


def load_splats_from_ply(ply_path: str, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Load Gaussian parameters from a PLY file.

    Args:
        ply_path: Path to the .ply file exported by gsplat
        device: Target device for tensors

    Returns:
        Dictionary containing Gaussian parameters:
        - means: [N, 3]
        - scales: [N, 3] (log-space)
        - quats: [N, 4]
        - opacities: [N] (logit-space)
        - sh0: [N, 1, 3]
        - shN: [N, K, 3]
    """
    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]

    # Extract positions
    x = vertex["x"]
    y = vertex["y"]
    z = vertex["z"]
    means = np.stack([x, y, z], axis=-1).astype(np.float32)

    # Extract scales (stored as log-space values)
    scale_0 = vertex["scale_0"]
    scale_1 = vertex["scale_1"]
    scale_2 = vertex["scale_2"]
    scales = np.stack([scale_0, scale_1, scale_2], axis=-1).astype(np.float32)

    # Extract quaternions
    rot_0 = vertex["rot_0"]
    rot_1 = vertex["rot_1"]
    rot_2 = vertex["rot_2"]
    rot_3 = vertex["rot_3"]
    quats = np.stack([rot_0, rot_1, rot_2, rot_3], axis=-1).astype(np.float32)

    # Extract opacity (stored as logit-space value)
    opacities = np.array(vertex["opacity"]).astype(np.float32)

    # Extract spherical harmonics coefficients
    # DC component (band 0)
    f_dc_0 = vertex["f_dc_0"]
    f_dc_1 = vertex["f_dc_1"]
    f_dc_2 = vertex["f_dc_2"]
    sh0 = np.stack([f_dc_0, f_dc_1, f_dc_2], axis=-1).astype(np.float32)
    sh0 = sh0[:, np.newaxis, :]  # [N, 1, 3]

    # Higher order SH (bands 1+)
    sh_rest = []
    idx = 0
    while True:
        try:
            sh_rest.append(vertex[f"f_rest_{idx}"])
            idx += 1
        except ValueError:
            break

    if sh_rest:
        shN = np.stack(sh_rest, axis=-1).astype(np.float32)
        # PLY stores SH as [N, 3*K] with color-channel-first ordering
        # (export does permute(0,2,1) before flattening: [N,K,3] -> [N,3,K] -> [N,3*K])
        # Reverse that: reshape to [N, 3, K] then transpose to [N, K, 3]
        num_coeffs = shN.shape[1] // 3
        shN = shN.reshape(shN.shape[0], 3, num_coeffs).transpose(0, 2, 1)
    else:
        shN = np.zeros((means.shape[0], 0, 3), dtype=np.float32)

    return {
        "means": torch.from_numpy(means).to(device),
        "scales": torch.from_numpy(scales).to(device),
        "quats": torch.from_numpy(quats).to(device),
        "opacities": torch.from_numpy(opacities).to(device),
        "sh0": torch.from_numpy(sh0).to(device),
        "shN": torch.from_numpy(shN).to(device),
    }


def load_splats_from_checkpoint(
    ckpt_path: str, device: str = "cuda"
) -> Tuple[Dict[str, torch.Tensor], Optional[dict]]:
    """Load Gaussian parameters and optional post-processing from a checkpoint.

    Args:
        ckpt_path: Path to the .pt checkpoint file
        device: Target device for tensors

    Returns:
        Tuple of (splats dict, post_processing state dict or None)
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    splats = ckpt["splats"]
    splats = {k: v.to(device) for k, v in splats.items()}
    pp_state = ckpt.get("post_processing", None)
    return splats, pp_state


class StandaloneRenderer:
    """Renders frames from a trained GSplat model."""

    def __init__(
        self,
        splats: Dict[str, torch.Tensor],
        sh_degree: int = 3,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        with_ut: bool = True,
        with_eval3d: bool = True,
        ppisp_state: Optional[dict] = None,
        device: str = "cuda",
    ):
        self.splats = splats
        self.sh_degree = sh_degree
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.with_ut = with_ut
        self.with_eval3d = with_eval3d
        self.device = device

        self.ppisp_module = None
        if ppisp_state is not None:
            try:
                from ppisp import PPISP
                self.ppisp_module = PPISP.from_state_dict(ppisp_state)
                self.ppisp_module.to(device)
                self.ppisp_module.eval()
                print(f"PPISP loaded: {self.ppisp_module.num_cameras} cameras, "
                      f"{self.ppisp_module.num_frames} frames")
            except ImportError:
                print("WARNING: ppisp package not available — rendering without post-processing")

    @torch.no_grad()
    def render_frame(
        self,
        camtoworld: np.ndarray,  # [4, 4]
        K: np.ndarray,  # [3, 3]
        width: int,
        height: int,
        camera_model: str = "pinhole",
        radial_coeffs: Optional[np.ndarray] = None,
        tangential_coeffs: Optional[np.ndarray] = None,
        thin_prism_coeffs: Optional[np.ndarray] = None,
        ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
        camera_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Render a single frame.

        Args:
            camtoworld: Camera-to-world transformation matrix [4, 4]
            K: Camera intrinsic matrix [3, 3]
            width: Image width
            height: Image height
            camera_model: Camera model type
            radial_coeffs: Radial distortion coefficients
            tangential_coeffs: Tangential distortion coefficients
            thin_prism_coeffs: Thin prism distortion coefficients
            ftheta_coeffs: F-theta camera parameters
            camera_idx: Integer camera index for PPISP (None disables per-camera effects)

        Returns:
            Rendered RGB image as uint8 numpy array [H, W, 3]
        """
        # Prepare camera tensors
        c2w = torch.from_numpy(camtoworld).float().to(self.device).unsqueeze(0)
        K_tensor = torch.from_numpy(K).float().to(self.device).unsqueeze(0)

        # Prepare distortion tensors
        radial = None
        if radial_coeffs is not None:
            radial = torch.from_numpy(radial_coeffs).float().to(self.device).unsqueeze(0)
        tangential = None
        if tangential_coeffs is not None:
            tangential = (
                torch.from_numpy(tangential_coeffs).float().to(self.device).unsqueeze(0)
            )
        thin_prism = None
        if thin_prism_coeffs is not None:
            thin_prism = (
                torch.from_numpy(thin_prism_coeffs).float().to(self.device).unsqueeze(0)
            )

        # Extract Gaussian parameters
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)

        # Rasterize
        render_colors, render_alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(c2w),
            Ks=K_tensor,
            width=width,
            height=height,
            sh_degree=self.sh_degree,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            packed=False,
            camera_model=camera_model,
            with_ut=self.with_ut,
            with_eval3d=self.with_eval3d,
            ftheta_coeffs=ftheta_coeffs,
            radial_coeffs=radial,
            tangential_coeffs=tangential,
            thin_prism_coeffs=thin_prism,
        )

        # Apply PPISP post-processing if available
        if self.ppisp_module is not None and camera_idx is not None:
            rgb = render_colors[0, ..., :3]  # [H, W, 3]
            pixel_x = torch.arange(width, device=self.device).float()
            pixel_y = torch.arange(height, device=self.device).float()
            pixel_coords = torch.stack(
                torch.meshgrid(pixel_x, pixel_y, indexing="xy"), dim=-1
            )  # [H, W, 2]
            rgb = self.ppisp_module(
                rgb=rgb,
                pixel_coords=pixel_coords,
                resolution=(width, height),
                camera_idx=camera_idx,
                frame_idx=None,
                exposure_prior=None,
            )
            colors_np = rgb.clamp(0, 1).cpu().numpy()
        else:
            colors_np = render_colors[0, ..., :3].clamp(0, 1).cpu().numpy()

        colors_np = (colors_np * 255).astype(np.uint8)
        return colors_np

    def render_trajectory(
        self,
        camera: CameraMetadata,
        camtoworlds: np.ndarray,  # [N, 4, 4] - possibly shifted
        output_dir: str,
        shift_name: str = "original",
        save_video: bool = True,
        fps: int = 30,
        camera_idx: Optional[int] = None,
    ) -> List[str]:
        """Render a full camera trajectory.

        Args:
            camera: Camera metadata with intrinsics and distortion
            camtoworlds: Trajectory poses (possibly shifted) [N, 4, 4]
            output_dir: Directory to save frames
            shift_name: Name of the shift configuration
            save_video: Whether to save an MP4 video
            fps: Video frame rate
            camera_idx: Integer camera index for PPISP post-processing

        Returns:
            List of saved frame paths
        """
        safe_cam_name = str(camera.camera_id).replace("/", "_")
        frames_dir = os.path.join(output_dir, "frames", shift_name, safe_cam_name)
        os.makedirs(frames_dir, exist_ok=True)

        frame_paths = []
        video_frames = []

        desc = f"Rendering {shift_name}/{safe_cam_name}"
        for i in tqdm.trange(len(camtoworlds), desc=desc):
            frame = self.render_frame(
                camtoworld=camtoworlds[i],
                K=camera.K,
                width=camera.width,
                height=camera.height,
                camera_model=camera.camera_model,
                radial_coeffs=camera.radial_coeffs,
                tangential_coeffs=camera.tangential_coeffs,
                thin_prism_coeffs=camera.thin_prism_coeffs,
                ftheta_coeffs=camera.ftheta_coeffs,
                camera_idx=camera_idx,
            )

            frame_path = os.path.join(frames_dir, f"frame_{i:05d}.png")
            imageio.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            video_frames.append(frame)

        # Save video
        if save_video:
            video_dir = os.path.join(output_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"{shift_name}_{safe_cam_name}.mp4")
            imageio.mimwrite(video_path, video_frames, fps=fps)
            print(f"Video saved to {video_path}")

        return frame_paths


def load_all_cameras(camera_paths_dir: str) -> List[CameraMetadata]:
    """Load all camera metadata from a directory.

    Args:
        camera_paths_dir: Directory containing per-camera subdirectories

    Returns:
        List of CameraMetadata objects
    """
    cameras = []
    for subdir in sorted(os.listdir(camera_paths_dir)):
        json_path = os.path.join(camera_paths_dir, subdir, "camera_data.json")
        if os.path.exists(json_path):
            cameras.append(CameraMetadata.from_json(json_path))
    return cameras


def _shift_name(x: float, y: float, z: float) -> str:
    """Auto-generate a filename-safe name from shift values.

    Examples: (0,0,0) -> "original", (-0.5,0,0) -> "X_-0.5", (0.5,0,-1) -> "X_0.5_Z_-1"
    Dots in decimals are kept (valid on Linux/macOS/Windows file systems).
    """
    if abs(x) < 1e-9 and abs(y) < 1e-9 and abs(z) < 1e-9:
        return "original"
    parts = []
    for axis, val in (("X", x), ("Y", y), ("Z", z)):
        if abs(val) > 1e-9:
            # Format: strip trailing zeros, e.g. 0.50 -> 0.5, 1.0 -> 1
            formatted = f"{val:g}"
            parts.append(f"{axis}_{formatted}")
    return "_".join(parts)


def parse_shifts_from_config(cfg: DictConfig) -> List[TrajectoryShift]:
    """Parse trajectory shifts from Hydra config.

    Accepts either:
    - List of [x, y, z] vectors (names auto-generated)
    - Legacy list of {name, x_m, y_m, z_m} dicts

    Args:
        cfg: Hydra config with trajectory.shifts list

    Returns:
        List of TrajectoryShift objects
    """
    shifts = []
    for entry in cfg.trajectory.shifts:
        if isinstance(entry, (list, tuple)) or hasattr(entry, "__iter__") and not hasattr(entry, "keys"):
            x, y, z = float(entry[0]), float(entry[1]), float(entry[2])
            shifts.append(TrajectoryShift(name=_shift_name(x, y, z), x_m=x, y_m=y, z_m=z))
        else:
            # Legacy dict format
            shifts.append(
                TrajectoryShift(
                    name=entry.name,
                    x_m=float(entry.get("x_m", 0.0)),
                    y_m=float(entry.get("y_m", 0.0)),
                    z_m=float(entry.get("z_m", 0.0)),
                )
            )
    return shifts


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for standalone rendering."""
    # Get rendering config
    render_cfg = cfg.get("rendering", {})

    # Determine model source
    ply_path = render_cfg.get("ply_path")
    checkpoint_path = render_cfg.get("checkpoint_path")
    camera_paths_dir = render_cfg.get("camera_paths_dir")
    output_dir = render_cfg.get("output_dir")

    if not camera_paths_dir:
        raise ValueError("camera_paths_dir must be specified")
    if not output_dir:
        raise ValueError("output_dir must be specified")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Gaussians
    ppisp_state = None
    if ply_path and os.path.exists(ply_path):
        print(f"Loading Gaussians from PLY: {ply_path}")
        splats = load_splats_from_ply(ply_path, device)
    elif checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading Gaussians from checkpoint: {checkpoint_path}")
        splats, ppisp_state = load_splats_from_checkpoint(checkpoint_path, device)
    else:
        raise ValueError(
            "Either ply_path or checkpoint_path must be specified and exist"
        )

    print(f"Loaded {splats['means'].shape[0]} Gaussians")

    # Load camera metadata
    print(f"Loading cameras from: {camera_paths_dir}")
    cameras = load_all_cameras(camera_paths_dir)
    print(f"Found {len(cameras)} cameras")

    # Filter cameras if specified
    cameras_to_render = render_cfg.get("cameras_to_render", [])
    if cameras_to_render:
        cameras = [c for c in cameras if c.camera_id in cameras_to_render]
        print(f"Rendering {len(cameras)} selected cameras")

    # Parse trajectory shifts
    shifts = parse_shifts_from_config(render_cfg)
    print(f"Trajectory shifts: {[s.name for s in shifts]}")

    # Create renderer
    renderer = StandaloneRenderer(
        splats=splats,
        sh_degree=render_cfg.get("render", {}).get("sh_degree", 3),
        near_plane=render_cfg.get("render", {}).get("near_plane", 0.01),
        far_plane=render_cfg.get("render", {}).get("far_plane", 1e10),
        with_ut=render_cfg.get("render", {}).get("with_ut", True),
        with_eval3d=render_cfg.get("render", {}).get("with_eval3d", True),
        ppisp_state=ppisp_state,
        device=device,
    )

    # Create trajectory manipulator
    manipulator = TrajectoryManipulator()

    # Render each camera with each shift
    os.makedirs(output_dir, exist_ok=True)

    for camera in cameras:
        # Get normalization scale (consistent across cameras from same scene)
        norm_scale = camera.world_to_normalized_scale
        if norm_scale is not None:
            print(f"Using world_to_normalized_scale={norm_scale:.6f} (1m real = {norm_scale:.4f} normalized units)")
        else:
            print("WARNING: No normalization scale found — shift values will be in raw scene units, not meters")

        for shift in shifts:
            # Apply shift to trajectory (scale converts real meters to normalized units)
            shifted_poses = manipulator.apply_shift(camera.camtoworlds, shift, world_to_normalized_scale=norm_scale)

            # Render
            renderer.render_trajectory(
                camera=camera,
                camtoworlds=shifted_poses,
                output_dir=output_dir,
                shift_name=shift.name,
                save_video=render_cfg.get("render", {}).get("save_video", True),
                fps=render_cfg.get("render", {}).get("fps", 30),
                camera_idx=camera.camera_index,
            )

    # Write success marker
    success_path = os.path.join(output_dir, ".success")
    with open(success_path, "w") as f:
        f.write("Rendering completed successfully.")

    print(f"\nRendering complete. Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
