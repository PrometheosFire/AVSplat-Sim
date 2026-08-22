import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import Literal, assert_never

from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.cuda._wrapper import CameraModel

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = True
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # If True, load the checkpoint state and continue training instead of eval-only.
    resume: bool = False
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path: "interp", "ellipse", "spiral", or "raw" (use captured poses as-is)
    render_traj_path: str = "interp"

    # Dataset backend: "colmap" or "ncore"
    data_type: str = "ncore"
    # Path to the Mip-NeRF 360 dataset (colmap) or NCore v4 meta-JSON file (ncore)
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: CameraModel = "pinhole"
    # Load EXIF exposure metadata from images (if available)
    load_exposure: bool = True

    # --- NCore-specific options (only used when data_type="ncore") ---
    # Camera sensor IDs to load (auto-detected from sequence if empty)
    ncore_camera_ids: List[str] = field(default_factory=list)
    # Lidar sensor IDs to load (auto-detected from sequence if empty)
    ncore_lidar_ids: List[str] = field(default_factory=list)
    # Temporal seek offset in seconds
    ncore_seek_offset_sec: Optional[float] = None
    # Clip duration in seconds (None = full sequence)
    ncore_duration_sec: Optional[float] = None
    # Maximum number of lidar init points
    ncore_max_lidar_points: int = 500_000
    # Generic-data key for lidar point RGB colors (fallback to gray if unavailable)
    ncore_lidar_color_generic_data_name: str = "rgb"
    # NCore component group names
    ncore_poses_component_group: str = "default"
    ncore_intrinsics_component_group: str = "default"
    ncore_masks_component_group: str = "default"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = True
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Post-processing method for appearance correction (experimental)
    post_processing: Optional[Literal["bilateral_grid", "ppisp"]] = None
    # Use fused implementation for bilateral grid (only applies when post_processing="bilateral_grid")
    bilateral_grid_fused: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
    # Enable PPISP controller
    ppisp_use_controller: bool = False
    # Use controller distillation in PPISP (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_distillation: bool = False
    # Controller activation ratio for PPISP (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_activation_num_steps: int = 25_000
    # Color correction method for cc_* metrics (only applies when post_processing is set)
    color_correct_method: Literal["affine", "quadratic"] = "affine"
    # Compute color-corrected metrics (cc_psnr, cc_ssim, cc_lpips) during evaluation
    use_color_correction_metric: bool = False

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = True
    with_eval3d: bool = True

    # ----------------------------------------------------------------- #
    # Dynamic rigid objects (vehicles) — 4D extension                    #
    # ----------------------------------------------------------------- #
    # Master switch. When False, training is background-only (unchanged).
    enable_dynamic: bool = False
    # Path to refined 3D tracks (``track_3d_refined_colmap.json``).
    dynamic_tracks_json: Optional[str] = None
    # Scene root holding the COLMAP reconstruction (``<root>/colmap_sparse/rig``
    # or ``<root>/sparse/0``), used to align boxes to the training frame.
    dynamic_scene_root: Optional[str] = None
    # Rigid class names to keep (None -> vehicle group: car/truck/bus/...).
    dynamic_rigid_classes: Optional[List[str]] = None
    # Minimum tracking score to keep a box.
    dynamic_min_track_score: float = 0.0
    # Initial number of Gaussians sampled inside each instance box.
    rigid_init_points_per_instance: int = 5000
    # Percentage to widen each instance's bounding box FOOTPRINT by before
    # training. The tracker's boxes hug the vehicle body, leaving mirrors,
    # overhang and the silhouette edges with no Gaussians initialised there --
    # and since Gaussians are only seeded inside the box (they are not clipped
    # afterwards), densification has to grow outward from nothing to cover them.
    # Only the ground-plane extents (local x = length, y = width) are scaled;
    # HEIGHT (local z) is left untouched so boxes do not sink into the road or
    # reach above the roof. 0 disables.
    rigid_bbox_expand_pct: float = 10.0
    # Optimize the per-frame rigid poses during training. The poses are part of
    # the render path (``get_world_gaussians`` builds ``means_world`` from them),
    # so when enabled the photometric loss moves every box independently, with no
    # kinematic constraint unless ``rigid_smooth_w`` > 0. Measured on scene_099
    # that cost ~0.30 m median drift from the Step 1.5 fitted trajectory and made
    # it ~9x rougher frame-to-frame, smearing the dynamic objects. Default False:
    # the refinement's bicycle-model fit is treated as the trajectory of record.
    rigid_pose_optimize: bool = False
    # Per-frame pose learning rates (translation moves faster than rotation).
    # Ignored unless ``rigid_pose_optimize`` is True.
    rigid_pose_trans_lr: float = 5e-4
    rigid_pose_quats_lr: float = 1e-5
    # Final pose learning rates for linear LR decay (OmniRe schedule).
    # Set to the same value as the initial LR to disable decay.
    rigid_pose_trans_lr_final: float = 1e-4
    rigid_pose_quats_lr_final: float = 5e-6
    # Temporal smoothness (2nd-order on translation) weight + window.
    rigid_smooth_w: float = 0.01
    rigid_smooth_range: int = 5
    # Opacity reset for rigid Gaussians: every N steps clamp opacity to max 0.01
    # (same as OmniRe). Forces dead/saturated Gaussians to re-compete. 0 = off.
    rigid_reset_opacity_every: int = 3000
    # Sharp-shape regularization: penalise Gaussians with aspect ratio > ratio.
    # Weight 1.0 and every-10-steps matches OmniRe. 0.0 weight disables.
    rigid_sharp_shape_w: float = 1.0
    rigid_sharp_shape_ratio: float = 10.0
    rigid_sharp_shape_every: int = 10
    # Rigid densification (Default-3DGS-style on the 3D positional gradient).
    rigid_grow_grad_thresh: float = 5e-5
    rigid_grow_scale3d: float = 0.01
    rigid_prune_opacity: float = 0.02
    rigid_prune_scale3d: float = 0.5
    # Hard cap on the TOTAL number of rigid Gaussians across all instances.
    rigid_cap_max: int = 1_000_000
    # Debug: also save eval/trajectory renders with per-instance 3D boxes drawn
    # (semi-transparent, per-instance color) to inspect dynamic-object alignment.
    debug_render_boxes: bool = False
    # Export per-instance rigid PLYs with DC-only color (drop higher-order SH) so
    # external viewers show flat, view-independent color instead of the rainbow
    # SH-overfit artifact when free-orbiting an object seen from few angles.
    rigid_ply_dc_only: bool = True
    rigid_refine_start_iter: int = 500
    rigid_refine_stop_iter: int = 25_000
    rigid_refine_every: int = 100
    rigid_warmup_for_big_prune: int = 3_000
    rigid_cull_out_of_bound: bool = True

    # --- Pose smoothing strategy (selects temporal regularizer) ---
    # "finite_diff": second-order translation smoothness (OmniRe-style).
    # "unicycle":    HUGS-style kinematic regularizer (smooth accel/yaw + anchoring).
    # "bicycle":     RETIRED — superseded by the Step 1.5 refinement bicycle-model
    #                fit (src/tracking/bicycle_fit.py); selecting it raises.
    rigid_pose_smoothing: str = "finite_diff"

    # Unicycle smoother hyperparameters (used when rigid_pose_smoothing="unicycle").
    # Whether to also optimize planar centers X/Z (more correction, more drift risk).
    unicycle_opt_pos: bool = True
    # Standalone pre-fit loop before 4DGS training (0 = skip).
    unicycle_prefit_iters: int = 100
    unicycle_prefit_reg_w: float = 5e-3
    unicycle_prefit_pos_w: float = 1e-3
    # Iteration window for joint unicycle loss during 4DGS training.
    unicycle_joint_start_iter: int = 1000
    unicycle_joint_end_iter: int = 15000
    unicycle_joint_reg_w: float = 1e-3
    unicycle_joint_pos_w: float = 1e-4
    # Per-parameter learning rates for the unicycle optimizer.
    unicycle_lr_speed: float = 1e-3
    unicycle_lr_heading: float = 1e-4
    unicycle_lr_center: float = 1e-3

    # Bicycle smoother hyperparameters — RETIRED (rigid_pose_smoothing="bicycle"
    # now raises). Kept inert only for backward-compatible config loading.
    # Whether to also optimize planar centers X/Z.
    bicycle_opt_pos: bool = True
    # Standalone pre-fit loop before 4DGS training (0 = skip).
    bicycle_prefit_iters: int = 100
    bicycle_prefit_reg_w: float = 5e-3
    bicycle_prefit_pos_w: float = 1e-3
    # Iteration window for joint bicycle loss during 4DGS training.
    bicycle_joint_start_iter: int = 1000
    bicycle_joint_end_iter: int = 15000
    bicycle_joint_reg_w: float = 1e-3
    bicycle_joint_pos_w: float = 1e-4
    # Coupling weights that pull rigid poses toward bicycle-smoothed center/yaw.
    bicycle_joint_couple_pos_w: float = 1e-2
    bicycle_joint_couple_yaw_w: float = 1e-2
    # Weight for anchoring rolled-out bicycle yaw to observed yaw.
    bicycle_yaw_anchor_w: float = 5e-3
    # Per-parameter learning rates for the bicycle optimizer.
    bicycle_lr_speed: float = 1e-3
    bicycle_lr_steer: float = 1e-4
    bicycle_lr_center: float = 1e-3
    # Fixed wheelbase policy from bbox long edge.
    bicycle_wheelbase_mode: str = "fixed_from_bbox_long_edge"
    bicycle_wheelbase_alpha: float = 0.60

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
            if strategy.noise_injection_stop_iter >= 0:
                strategy.noise_injection_stop_iter = int(
                    strategy.noise_injection_stop_iter * factor
                )
        else:
            assert_never(strategy)

        # Keep rigid densification schedule aligned with the global step scaler.
        self.rigid_refine_start_iter = int(self.rigid_refine_start_iter * factor)
        self.rigid_refine_stop_iter = int(self.rigid_refine_stop_iter * factor)
        self.rigid_refine_every = max(1, int(self.rigid_refine_every * factor))
        self.rigid_warmup_for_big_prune = int(self.rigid_warmup_for_big_prune * factor)
        if self.rigid_reset_opacity_every > 0:
            self.rigid_reset_opacity_every = max(1, int(self.rigid_reset_opacity_every * factor))
        if self.rigid_sharp_shape_every > 0:
            self.rigid_sharp_shape_every = max(1, int(self.rigid_sharp_shape_every * factor))
        if self.rigid_pose_smoothing == "unicycle":
            self.unicycle_prefit_iters = int(self.unicycle_prefit_iters * factor)
            self.unicycle_joint_start_iter = int(self.unicycle_joint_start_iter * factor)
            self.unicycle_joint_end_iter = int(self.unicycle_joint_end_iter * factor)
        elif self.rigid_pose_smoothing == "bicycle":
            self.bicycle_prefit_iters = int(self.bicycle_prefit_iters * factor)
            self.bicycle_joint_start_iter = int(self.bicycle_joint_start_iter * factor)
            self.bicycle_joint_end_iter = int(self.bicycle_joint_end_iter * factor)