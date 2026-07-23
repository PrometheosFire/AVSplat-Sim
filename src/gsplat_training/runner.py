# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from gsplat.color_correct import color_correct_affine, color_correct_quadratic
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization, RasterizeMode
from gsplat.cuda._wrapper import CameraModel
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap


# My imports
from config_training import Config

# Dynamic rigid-object (vehicle) support for the 4D extension.
RIGID_GAUSS_KEYS = ["means", "scales", "quats", "opacities", "sh0", "shN"]


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    """Initialize 3D Gaussian Splatting parameters and their optimizers.
    
    This function creates the parameters for all Gaussians (positions, scales, rotations,
    opacities, and colors/features) and sets up separate optimizers for each parameter
    type with their respective learning rates. It handles both SfM initialization (from
    COLMAP 3D points) and random initialization.
    
    Args:
        parser (Parser): Dataset parser containing 3D points and RGB colors from COLMAP.
        init_type (str): How to initialize Gaussian positions. Options:
            - "sfm": Use 3D points from COLMAP structure-from-motion (recommended)
            - "lidar": Use LIDAR point cloud initialization
            - "random": Randomly initialize within a bounded volume
        init_num_pts (int): Only used when init_type="random". Number of random Gaussians.
        init_extent (float): Only used when init_type="random". Size of initialization volume.
        init_opacity (float): Initial opacity value for all Gaussians (in [0, 1]). Lower
            values (e.g., 0.1) start with more transparent Gaussians.
        init_scale (float): Scale multiplier for initial Gaussian sizes. Computed as the
            average distance to 3 nearest neighbors.
        means_lr (float): Learning rate for Gaussian position (means) optimization.
        scales_lr (float): Learning rate for Gaussian scale (size) optimization.
        opacities_lr (float): Learning rate for Gaussian opacity optimization.
        quats_lr (float): Learning rate for Gaussian rotation (quaternions) optimization.
        sh0_lr (float): Learning rate for spherical harmonics band 0 (brightness/DC component).
        shN_lr (float): Learning rate for higher-order spherical harmonics (detail/view-dependence).
        scene_scale (float): Global scene scale multiplier for position learning rate adjustment.
        sh_degree (int): Maximum spherical harmonics degree (0-3). Higher = more view-dependent colors.
        sparse_grad (bool): If True, use sparse gradients (memory efficient but slower). Requires packed=True.
        visible_adam (bool): If True, use Selective Adam optimizer (only update visible Gaussians).
        batch_size (int): Training batch size (used to scale learning rates by sqrt(batch_size)).
        feature_dim (Optional[int]): If provided, adds learnable feature embeddings for appearance
            variation (e.g., 32D features for per-camera appearance). If None, uses SH colors only.
        device (str): PyTorch device (e.g., "cuda:0", "cpu").
        world_rank (int): Rank in distributed training (0 if single GPU). Used to distribute
            Gaussians across GPUs (each GPU gets every world_size-th Gaussian).
        world_size (int): Total number of GPUs in distributed training.
    
    Returns:
        Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
            - splats: ParameterDict containing all learnable Gaussian parameters:
                - "means": [N, 3] - Gaussian 3D positions
                - "scales": [N, 3] - Log-space scale parameters (exp(scales) = actual sizes)
                - "quats": [N, 4] - Rotation quaternions (normalized during rasterization)
                - "opacities": [N] - Log-odds opacity values (sigmoid applied during rendering)
                - "sh0": [N, 1, 3] - SH band 0 (RGB brightness)
                - "shN": [N, (sh_degree+1)^2-1, 3] - Higher SH bands (if feature_dim is None)
                - "features": [N, feature_dim] - Appearance features (if feature_dim is not None)
                - "colors": [N, 3] - Additional bias colors (if feature_dim is not None)
            - optimizers: Dict mapping parameter names to their optimizers:
                - Keys: "means", "scales", "quats", "opacities", "sh0", "shN"/"features"/"colors"
                - Each optimizer uses learning rates scaled by sqrt(batch_size * world_size)
    """
    
    if init_type == "sfm" or init_type == "lidar":
        points = torch.from_numpy(parser.points).float() #Load points from COLMAP SfM or LIDAR to tensors
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm, random, or lidar")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    # Logit is the inverse of sigmoid, maps (0, 1) to (-inf, +inf). 
    # This way we can optimize in unconstrained space and apply sigmoid during rendering to get valid opacity values.
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        # torch.nn.Parameter treats it as a learnable parameter that requires gradients and can be optimized by an optimizer.
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam #SparseAdam: efficient for sparse gradient tensors
    elif visible_adam:
        optimizer_class = SelectiveAdam #SelectiveAdam: efficient when only visible Gaussians should be updated

    else:
        optimizer_class = torch.optim.Adam #Adam: simplest, updates everything
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            fused=True,
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        """Initialize the training runner for 3D Gaussian Splatting.
        
        Args:
            local_rank (int): Rank of the current GPU/process on the local machine (0-indexed).
                In single-GPU training, this is 0. In multi-GPU training on one machine,
                if you have 4 GPUs, local_rank ranges from 0-3.
                Used to select which GPU this process will use (e.g., cuda:local_rank).
            world_rank (int): Rank of the current process across ALL machines/GPUs in the
                distributed training setup (0-indexed). For example, in distributed training
                across 2 machines with 4 GPUs each, world_rank ranges from 0-7.
                Used to synchronize and coordinate training across multiple machines.
            world_size (int): Total number of GPUs/processes across all machines.
                For distributed training on 2 machines with 4 GPUs each, world_size = 8.
                Used to divide batch sizes and determine when to synchronize.
            cfg (Config): Configuration dataclass containing all hyperparameters and settings
                for training (learning rates, strategy, dataset paths, etc.).
        """
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # === Tensorboard Setup ===
        # TensorBoard is a visualization tool that tracks training metrics over time.
        # It creates interactive graphs/dashboards showing:
        #   - Loss curves (how loss decreases during training)
        #   - Learning rates (how they change over time)
        #   - Memory usage, number of Gaussians, evaluation metrics (PSNR, SSIM, LPIPS)
        #   - Training images (rendered outputs vs ground truth)
        # Later in training, we'll log values using: self.writer.add_scalar("name", value, step)
        # View results with: tensorboard --logdir results/garden/tb
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        if cfg.data_type == "ncore":
            from datasets.ncore import NCoreDataset, NCoreParser

            self.parser = NCoreParser(
                meta_json_path=cfg.data_dir,
                factor=1.0 / cfg.data_factor if cfg.data_factor > 1 else 1.0,
                test_every=cfg.test_every,
                camera_ids=cfg.ncore_camera_ids or None,
                lidar_ids=cfg.ncore_lidar_ids or None,
                seek_offset_sec=cfg.ncore_seek_offset_sec,
                duration_sec=cfg.ncore_duration_sec,
                max_lidar_points=cfg.ncore_max_lidar_points,
                lidar_color_generic_data_name=cfg.ncore_lidar_color_generic_data_name,
                poses_component_group=cfg.ncore_poses_component_group,
                intrinsics_component_group=cfg.ncore_intrinsics_component_group,
                masks_component_group=cfg.ncore_masks_component_group,
                normalize_world_space=cfg.normalize_world_space,
            )
            self.trainset = NCoreDataset(self.parser, split="train", load_depths=cfg.depth_loss)
            self.valset = NCoreDataset(self.parser, split="val")
            self.ncore_camera_data = [
                self.parser.camera_render_data[cam_id]
                for cam_id in self.parser.camera_ids
            ]
            if (
                any(d.camera_model == "ftheta" for d in self.ncore_camera_data)
                and not cfg.with_eval3d
            ):
                print(
                    "[NCore] Warning: FTheta cameras detected; pass --with-eval3d True for correct results."
                )
        else:
            # read the scene once and build all metadata needed by training:
            # camera poses, intrinsics, image paths, points/colors camera indexing info, masks, exif exposure.
            self.parser = Parser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=cfg.test_every,
                load_exposure=cfg.load_exposure,
            )
            # create the training sample iterator on top of parser metadata
            self.trainset = Dataset(
                self.parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
            )
            # validation dataset used every test_every-th image
            self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        if self.parser.num_cameras > 1 and cfg.batch_size != 1:
            raise ValueError(
                f"When using multiple cameras ({self.parser.num_cameras} found), batch_size must be 1, "
                f"but got batch_size={cfg.batch_size}."
            )
        if cfg.post_processing == "ppisp" and cfg.batch_size != 1:
            raise ValueError(
                f"PPISP post-processing requires batch_size=1, got batch_size={cfg.batch_size}"
            )
        if cfg.post_processing is not None and world_size > 1:
            raise ValueError(
                f"Post-processing ({cfg.post_processing}) requires single-GPU training, "
                f"but world_size={world_size}."
            )
        if cfg.post_processing == "ppisp" and isinstance(cfg.strategy, DefaultStrategy):
            raise ValueError(
                f"PPISP post-processing requires MCMCStrategy at the moment."
            )

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Dynamic rigid objects (vehicles). Sets up self.rigid_nodes /
        # self.rigid_tracks / self.rigid_densifier and their optimizers, or
        # leaves them None when cfg.enable_dynamic is False.
        self._setup_dynamic_rigid()

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.post_processing_module = None
        if cfg.post_processing == "bilateral_grid":
            self.post_processing_module = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
        elif cfg.post_processing == "ppisp":
            ppisp_config = PPISPConfig(
                use_controller=cfg.ppisp_use_controller,
                controller_distillation=cfg.ppisp_controller_distillation,
                controller_activation_ratio=cfg.ppisp_controller_activation_num_steps
                / cfg.max_steps,
            )
            self.post_processing_module = PPISP(
                num_cameras=self.parser.num_cameras,
                num_frames=len(self.trainset),
                config=ppisp_config,
            ).to(self.device)

        self.post_processing_optimizers = []
        if cfg.post_processing == "bilateral_grid":
            self.post_processing_optimizers = [
                torch.optim.Adam(
                    self.post_processing_module.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]
        elif cfg.post_processing == "ppisp":
            self.post_processing_optimizers = (
                self.post_processing_module.create_optimizers()
            )

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer 
        # TODO looki into viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

        # Track if Gaussians are frozen (for controller distillation)
        self._gaussians_frozen = False

    def freeze_gaussians(self):
        """Freeze all Gaussian parameters for controller distillation.

        This prevents Gaussians from being updated by any loss (including regularization)
        while the controller learns to predict per-frame corrections.
        """
        if self._gaussians_frozen:
            return

        for name, param in self.splats.items():
            param.requires_grad = False

        self._gaussians_frozen = True
        print("[Distillation] Gaussian parameters frozen")

    def _setup_dynamic_rigid(self) -> None:
        """Build rigid-object nodes, poses, optimizers and densifier.

        Loads refined 3D vehicle tracks, aligns them from the COLMAP world into
        the trainer's normalized frame, initializes a :class:`RigidNodes` module
        (local Gaussians + per-frame SE(3) poses), and sets up its Gaussian /
        pose optimizers plus a :class:`RigidDensifier`. When ``enable_dynamic``
        is False, all rigid attributes are set to ``None`` so the rest of the
        pipeline runs background-only.
        """
        self.rigid_nodes = None
        self.rigid_tracks = None
        self.rigid_densifier = None
        self.rigid_gauss_optimizers = {}
        self.rigid_pose_optimizers = {}

        cfg = self.cfg
        if not getattr(cfg, "enable_dynamic", False):
            return

        from dynamic import (
            RigidDensifier,
            RigidNodes,
            UnicycleSmoother,
            compute_colmap_to_training,
            load_rigid_tracks,
            resolve_colmap_sparse_dir,
        )

        if cfg.data_type != "ncore":
            raise ValueError(
                "Dynamic rigid objects currently require data_type='ncore'."
            )
        if not cfg.dynamic_tracks_json or not cfg.dynamic_scene_root:
            raise ValueError(
                "enable_dynamic=True requires dynamic_tracks_json and "
                "dynamic_scene_root to be set."
            )

        # COLMAP world -> training frame similarity (validated by fit residual).
        colmap_dir = resolve_colmap_sparse_dir(cfg.dynamic_scene_root)
        transform = compute_colmap_to_training(self.parser, colmap_dir)

        # Load and align rigid vehicle tracks to per-frame SE(3) poses.
        self.rigid_tracks = load_rigid_tracks(
            cfg.dynamic_tracks_json,
            transform,
            self.parser.frame_timestamps_us,
            rigid_classes=cfg.dynamic_rigid_classes,
            min_score=cfg.dynamic_min_track_score,
        )
        self.rigid_nodes = RigidNodes(
            self.rigid_tracks,
            init_points_per_instance=cfg.rigid_init_points_per_instance,
            sh_degree=cfg.sh_degree,
            device=self.device,
        )

        # Split optimizers into Gaussian (densified) and pose (fixed-size) groups.
        all_optimizers = self.rigid_nodes.create_optimizers(
            batch_size=cfg.batch_size,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            quats_lr=cfg.quats_lr,
            opacities_lr=cfg.opacities_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            pose_trans_lr=cfg.rigid_pose_trans_lr,
            pose_quats_lr=cfg.rigid_pose_quats_lr,
        )
        self.rigid_gauss_optimizers = {k: all_optimizers[k] for k in RIGID_GAUSS_KEYS}
        self.rigid_pose_optimizers = {
            k: all_optimizers[k] for k in ("pose_trans", "pose_quats")
        }

        # Unicycle smoother — built and pre-fitted when strategy="unicycle".
        self.unicycle_optimizers: Dict[str, torch.optim.Optimizer] = {}
        if getattr(cfg, "rigid_pose_smoothing", "finite_diff") == "unicycle":
            uc = UnicycleSmoother(
                self.rigid_tracks,
                opt_pos=bool(getattr(cfg, "unicycle_opt_pos", True)),
                device=self.device,
            )
            self.rigid_nodes.unicycle = uc
            self.unicycle_optimizers = self.rigid_nodes.create_unicycle_optimizers(
                lr_speed=float(getattr(cfg, "unicycle_lr_speed", 1e-3)),
                lr_heading=float(getattr(cfg, "unicycle_lr_heading", 1e-4)),
                lr_center=float(getattr(cfg, "unicycle_lr_center", 1e-3)),
            )
            uc.prefit(
                n_iters=int(getattr(cfg, "unicycle_prefit_iters", 100)),
                reg_w=float(getattr(cfg, "unicycle_prefit_reg_w", 5e-3)),
                pos_w=float(getattr(cfg, "unicycle_prefit_pos_w", 1e-3)),
                lr_speed=float(getattr(cfg, "unicycle_lr_speed", 1e-3)),
                lr_heading=float(getattr(cfg, "unicycle_lr_heading", 1e-4)),
                lr_center=float(getattr(cfg, "unicycle_lr_center", 1e-3)),
            )
            print(
                f"[Unicycle] Smoother attached: {self.rigid_nodes.num_instances} instances, "
                f"opt_pos={uc.opt_pos}, prefit={getattr(cfg, 'unicycle_prefit_iters', 100)} iters."
            )

        self.rigid_densifier = RigidDensifier(
            grow_grad_thresh=cfg.rigid_grow_grad_thresh,
            grow_scale3d=cfg.rigid_grow_scale3d,
            prune_opacity=cfg.rigid_prune_opacity,
            prune_scale3d=cfg.rigid_prune_scale3d,
            cap_max=cfg.rigid_cap_max,
            refine_start_iter=cfg.rigid_refine_start_iter,
            refine_stop_iter=cfg.rigid_refine_stop_iter,
            refine_every=cfg.rigid_refine_every,
            warmup_for_big_prune=cfg.rigid_warmup_for_big_prune,
            cull_out_of_bound=cfg.rigid_cull_out_of_bound,
            verbose=True,
        )
        self.rigid_densifier.initialize_state()
        print(
            f"[Dynamic] Rigid objects enabled: {self.rigid_nodes.num_instances} "
            f"instances, {self.rigid_nodes.num_points} initial Gaussians across "
            f"{self.rigid_nodes.num_frames} frames."
        )

    def _resolve_rigid_frame_idx(self, data: Dict) -> Optional[int]:
        """Map a sample's capture timestamp to a global rigid-track frame index.

        Returns ``None`` when dynamic rigid objects are not configured or the
        sample carries no timestamp, in which case rendering stays background-only.
        """
        tracks = getattr(self, "rigid_tracks", None)
        if tracks is None or "timestamp_us" not in data:
            return None
        ts = data["timestamp_us"]
        if torch.is_tensor(ts):
            ts = int(ts.flatten()[0].item())
        else:
            ts = int(ts)
        return tracks.frame_index_from_timestamp(ts)

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[RasterizeMode] = None,
        camera_model: Optional[CameraModel] = None,
        frame_idcs: Optional[Tensor] = None,
        camera_idcs: Optional[Tensor] = None,
        exposure: Optional[Tensor] = None,
        rigid_frame_idx: Optional[int] = None,
        draw_boxes: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """Rasterize 3D Gaussians into 2D image tensors using differentiable splatting.

        This is the core rendering function that:
        1. Extracts learnable Gaussian parameters (means, scales, rotations, opacities).
        2. Computes per-Gaussian colors from either spherical harmonics (SH) or
           appearance module features depending on training mode.
        3. Optionally applies camera-specific distortion coefficients (radial, tangential,
           thin-prism for OpenCV pinhole/fisheye; polynomial for f-theta).
        4. Rasterizes Gaussians to image space, producing RGB + optional depth/alpha.
        5. Applies pixel-level validity masks (ego vehicle, dynamic regions).
        6. Optionally applies post-processing (bilateral grid or PPISP).

        Args:
            camtoworlds: Camera-to-world transformation matrices [B, 4, 4].
            Ks: Camera intrinsic matrices [B, 3, 3].
            width: Rendered image width in pixels.
            height: Rendered image height in pixels.
            masks: Optional pixel validity mask [B, H, W]; True = valid pixel.
            rasterize_mode: Rasterization variant ("antialiased" or "classic").
                If None, uses config default.
            camera_model: Camera model type ("pinhole", "fisheye", "ftheta", etc.).
                If None, uses config default.
            frame_idcs: Frame indices for per-frame parameters (e.g., pose/appearance
                adjustments). Used by post-processing modules.
            camera_idcs: Camera indices for multi-camera setups. Used to select
                camera-specific distortion coefficients from NCore data.
            exposure: Optional per-frame EXIF exposure values [B,] for tonemapping.
            **kwargs: Additional arguments passed to rasterization (sh_degree, near_plane,
                far_plane, render_mode, etc.).

        Returns:
            Tuple[Tensor, Tensor, Dict]:
                - render_colors: Rendered image [B, H, W, C] where C depends on render_mode
                  (typically C=3 for RGB, C=4 for RGB+depth).
                - render_alphas: Per-pixel alpha/opacity [B, H, W, 1].
                - info: Dictionary with rasterization metadata including:
                  - "radii": Gaussian projection radii for visibility checking.
                  - "gaussian_ids": Visible Gaussian indices (for sparse gradient updates).
        """
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        ftheta_coeffs = None
        radial_coeffs = None
        tangential_coeffs = None
        thin_prism_coeffs = None
        with_ut = self.cfg.with_ut

        if camera_idcs is not None and hasattr(self, "ncore_camera_data"):
            cam = self.ncore_camera_data[camera_idcs.item()]
            camera_model = cam.camera_model
            ftheta_coeffs = cam.ftheta_coeffs
            if cam.radial_coeffs is not None:
                radial_coeffs = (
                    torch.from_numpy(cam.radial_coeffs).to(means.device).unsqueeze(0)
                )
            if cam.tangential_coeffs is not None:
                tangential_coeffs = (
                    torch.from_numpy(cam.tangential_coeffs)
                    .to(means.device)
                    .unsqueeze(0)
                )
            if cam.thin_prism_coeffs is not None:
                thin_prism_coeffs = (
                    torch.from_numpy(cam.thin_prism_coeffs)
                    .to(means.device)
                    .unsqueeze(0)
                )

        # --- Dynamic rigid objects: append world-space Gaussians for this frame ---
        # Background Gaussians live in ``self.splats`` (densified by MCMC). Rigid
        # instances live in a separate ``self.rigid_nodes`` module and are placed
        # into world space per frame via their learnable SE(3) poses, then simply
        # concatenated so a single rasterization pass composites both. The split
        # index ``n_background`` is recorded on ``info`` so the rigid densifier can
        # slice the rigid range out of the per-Gaussian rasterization outputs.
        n_background = means.shape[0]
        n_rigid = 0
        rigid_nodes = getattr(self, "rigid_nodes", None)
        if rigid_nodes is not None and rigid_frame_idx is not None:
            if self.cfg.app_opt:
                raise NotImplementedError(
                    "Dynamic rigid nodes are incompatible with app_opt: backgrounds "
                    "use precomputed per-camera RGB while rigid nodes use SH "
                    "coefficients, which cannot share one rasterization call."
                )
            rigid = rigid_nodes.get_world_gaussians(int(rigid_frame_idx))
            if rigid is not None:
                means = torch.cat([means, rigid["means"]], dim=0)
                quats = torch.cat([quats, rigid["quats"]], dim=0)
                scales = torch.cat([scales, rigid["scales"]], dim=0)
                opacities = torch.cat([opacities, rigid["opacities"]], dim=0)
                colors = torch.cat([colors, rigid["colors"]], dim=0)
                n_rigid = rigid["means"].shape[0]

        # --- Debug overlay: draw per-instance 3D boxes as semi-transparent
        # Gaussians so they line up exactly with the render (distortion included).
        # Visualization only (detached); typically enabled in eval / trajectory.
        if draw_boxes and rigid_nodes is not None and rigid_frame_idx is not None:
            boxes = rigid_nodes.get_box_edge_gaussians(int(rigid_frame_idx))
            if boxes is not None:
                means = torch.cat([means, boxes["means"]], dim=0)
                quats = torch.cat([quats, boxes["quats"]], dim=0)
                scales = torch.cat([scales, boxes["scales"]], dim=0)
                opacities = torch.cat([opacities, boxes["opacities"]], dim=0)
                colors = torch.cat([colors, boxes["colors"]], dim=0)

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=camera_model,
            with_ut=with_ut,
            with_eval3d=self.cfg.with_eval3d,
            ftheta_coeffs=ftheta_coeffs,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            **kwargs,
        )
        # Record the background/rigid split (useful for logging / future use).
        info["n_background"] = n_background
        info["n_rigid"] = n_rigid
        if masks is not None:
            render_colors[~masks] = 0 # Mask out invalid pixels (e.g., ego vehicle, dynamic objects) by setting their color to black.

        if self.cfg.post_processing is not None:
            # Create pixel coordinates [H, W, 2] with +0.5 center offset
            pixel_y, pixel_x = torch.meshgrid(
                torch.arange(height, device=self.device) + 0.5,
                torch.arange(width, device=self.device) + 0.5,
                indexing="ij",
            )
            pixel_coords = torch.stack([pixel_x, pixel_y], dim=-1)  # [H, W, 2]

            # Split RGB from extra channels (e.g. depth) for post-processing
            rgb = render_colors[..., :3]
            extra = render_colors[..., 3:] if render_colors.shape[-1] > 3 else None

            # Apply post-processing to RGB channels only, keep extra channels (e.g. depth) unchanged.
            if self.cfg.post_processing == "bilateral_grid":
                if frame_idcs is not None:
                    grid_xy = (
                        pixel_coords / torch.tensor([width, height], device=self.device)
                    ).unsqueeze(0)
                    rgb = slice(
                        self.post_processing_module,
                        grid_xy.expand(rgb.shape[0], -1, -1, -1),
                        rgb,
                        frame_idcs.unsqueeze(-1),
                    )["rgb"]
            elif self.cfg.post_processing == "ppisp":
                camera_idx = camera_idcs.item() if camera_idcs is not None else None
                frame_idx = frame_idcs.item() if frame_idcs is not None else None
                rgb = self.post_processing_module(
                    rgb=rgb,
                    pixel_coords=pixel_coords,
                    resolution=(width, height),
                    camera_idx=camera_idx,
                    frame_idx=frame_idx,
                    exposure_prior=exposure,
                )

            # RGB [B, H, W, 3], extra (e.g. depth) [B, H, W, C-3] -> combined [B, H, W, C]
            render_colors = (
                torch.cat([rgb, extra], dim=-1) if extra is not None else rgb
            )

        return render_colors, render_alphas, info

    def train(self):
        """Run the full optimization loop for Gaussian Splatting.

        This method performs end-to-end training, including:
        1. Scheduler and dataloader setup.
        2. Per-step forward rendering via ``rasterize_splats``.
        3. Loss computation (L1 + SSIM, optional depth, optional post-processing regularization).
        4. Backward pass and optimizer/scheduler updates.
        5. Strategy hooks for densification/pruning before and after optimization.
        6. Periodic logging, checkpointing, evaluation, trajectory rendering, and optional compression.

        Training data fields consumed from each batch:
            - ``camtoworld``: [B, 4, 4] camera-to-world poses.
            - ``K``: [B, 3, 3] intrinsics.
            - ``image``: [B, H, W, 3] uint8 ground-truth image (normalized to [0, 1]).
            - ``image_id``: [B] frame id used by pose/app/post-processing modules.
            - ``camera_idx``: [B] camera index (used for multi-camera distortion and PPISP).
            - Optional ``mask``: [B, H, W] validity mask.
            - Optional ``exposure``: [B] exposure prior.
            - Optional depth supervision tensors when ``cfg.depth_loss`` is enabled.

        Notes:
            - Viewer integration (if enabled) can pause/resume training and receives step-rate stats.
            - Sparse gradients and visible-only Adam paths are handled before optimizer steps.
            - The rendered tensor can be RGB ([B, H, W, 3]) or RGB+depth ([B, H, W, 4])
              depending on ``render_mode``.
            - Post-processing modules (bilateral grid or PPISP) are optimized jointly when active.

        Returns:
            None. Side effects include model updates, TensorBoard logs, checkpoints/stat files,
            rendered evaluation outputs, and optional exported PLY/compression artifacts.
        """
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        # Rigid pose learning-rate decay (OmniRe schedule: linear, trans 5e-4→1e-4,
        # quats 1e-5→5e-6).  Guards against a zero initial LR to avoid div-by-zero.
        if self.rigid_nodes is not None:
            for key, opt in self.rigid_pose_optimizers.items():
                init_lr = opt.param_groups[0]["lr"]
                if key == "pose_trans":
                    final_lr = cfg.rigid_pose_trans_lr_final
                else:
                    final_lr = cfg.rigid_pose_quats_lr_final
                end_factor = (final_lr / init_lr) if init_lr > 0 else 1.0
                schedulers.append(
                    torch.optim.lr_scheduler.LinearLR(
                        opt,
                        start_factor=1.0,
                        end_factor=end_factor,
                        total_iters=max_steps,
                    )
                )
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        # Post-processing module has a learning rate schedule
        if cfg.post_processing == "bilateral_grid":
            # Linear warmup + exponential decay
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.post_processing_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.post_processing_optimizers[0],
                            gamma=0.01 ** (1.0 / max_steps),
                        ),
                    ]
                )
            )
        elif cfg.post_processing == "ppisp":
            ppisp_schedulers = self.post_processing_module.create_schedulers(
                self.post_processing_optimizers,
                max_optimization_iters=max_steps,
            )
            schedulers.extend(ppisp_schedulers)

        if cfg.ckpt is not None and cfg.resume:
            if len(cfg.ckpt) != 1:
                raise ValueError("Resume mode expects exactly one checkpoint path.")
            ckpt = torch.load(cfg.ckpt[0], map_location=device)
            for name in self.splats.keys():
                self.splats[name].data = ckpt["splats"][name].to(device)
            if cfg.pose_opt and "pose_adjust" in ckpt:
                if world_size > 1:
                    self.pose_adjust.module.load_state_dict(ckpt["pose_adjust"])
                else:
                    self.pose_adjust.load_state_dict(ckpt["pose_adjust"])
            if cfg.app_opt and "app_module" in ckpt:
                if world_size > 1:
                    self.app_module.module.load_state_dict(ckpt["app_module"])
                else:
                    self.app_module.load_state_dict(ckpt["app_module"])
            if self.post_processing_module is not None and "post_processing" in ckpt:
                self.post_processing_module.load_state_dict(ckpt["post_processing"])
            if self.rigid_nodes is not None and "rigid_nodes" in ckpt:
                # Rigid Gaussian count changes with densification: the module
                # resizes its tensors to the checkpoint before loading, then we
                # rebuild the rigid optimizers (fresh Adam state) to match.
                self.rigid_nodes.load_state_dict(ckpt["rigid_nodes"])
                if self.rigid_nodes.unicycle is not None and "unicycle_smoother" in ckpt:
                    self.rigid_nodes.unicycle.load_state_dict(ckpt["unicycle_smoother"])
                all_optimizers = self.rigid_nodes.create_optimizers(
                    batch_size=cfg.batch_size,
                    means_lr=cfg.means_lr,
                    scales_lr=cfg.scales_lr,
                    quats_lr=cfg.quats_lr,
                    opacities_lr=cfg.opacities_lr,
                    sh0_lr=cfg.sh0_lr,
                    shN_lr=cfg.shN_lr,
                    pose_trans_lr=cfg.rigid_pose_trans_lr,
                    pose_quats_lr=cfg.rigid_pose_quats_lr,
                )
                self.rigid_gauss_optimizers = {
                    k: all_optimizers[k] for k in RIGID_GAUSS_KEYS
                }
                self.rigid_pose_optimizers = {
                    k: all_optimizers[k] for k in ("pose_trans", "pose_quats")
                }
            if "optimizers" in ckpt:
                for name, optimizer in self.optimizers.items():
                    optimizer.load_state_dict(ckpt["optimizers"][name])
            if cfg.pose_opt and "pose_optimizers" in ckpt:
                for optimizer, state in zip(self.pose_optimizers, ckpt["pose_optimizers"]):
                    optimizer.load_state_dict(state)
            if cfg.app_opt and "app_optimizers" in ckpt:
                for optimizer, state in zip(self.app_optimizers, ckpt["app_optimizers"]):
                    optimizer.load_state_dict(state)
            if "post_processing_optimizers" in ckpt:
                for optimizer, state in zip(self.post_processing_optimizers, ckpt["post_processing_optimizers"]):
                    optimizer.load_state_dict(state)
            if "schedulers" in ckpt:
                for scheduler, state in zip(schedulers, ckpt["schedulers"]):
                    scheduler.load_state_dict(state)
            if "strategy_state" in ckpt:
                self.strategy_state = ckpt["strategy_state"]
            init_step = int(ckpt["step"]) + 1

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,              # Affects CPU data loading speed. Can maybe increase if lots of RAM/CPU core
            persistent_workers=True,
            pin_memory=True,
            #prefetch_factor=2,           # Refetch factor for preloading data. Can maybe increase if data loading is bottleneck
        )
        trainloader_iter = iter(trainloader) 

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps)) # Progress bar for training steps. Ranges from init_step to max_steps.
        for step in pbar: #Training loop! Each iteration corresponds to one optimization step (forward + backward + update).
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            # Validate box sizes haven't changed (periodic check every 100 steps).
            if self.rigid_nodes is not None and step % 100 == 0:
                if not self.rigid_nodes.validate_sizes_unchanged():
                    print(f"[ERROR] Box sizes changed at step {step}! Check rigid_nodes.py.")

            # Freeze Gaussians when PPISP controller distillation starts
            if (
                cfg.post_processing == "ppisp"
                and cfg.ppisp_use_controller
                and cfg.ppisp_controller_distillation
                and step >= cfg.ppisp_controller_activation_num_steps
            ):
                self.freeze_gaussians()
            
            # One frame per data dict (for batch_size = 1)
            try:
                data = next(trainloader_iter) # Fetch next batch of training data. 
            except StopIteration:             # If the iterator is exhausted, it raises StopIteration, which we catch to reset the iterator for the next epoch.
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3] Normalize pixel values to [0, 1] range for loss computation.
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2] #Number of pixels in the batch (B*H*W), used by viewer
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            exposure = (
                data["exposure"].to(device) if "exposure" in data else None
            )  # [B,]
            if cfg.depth_loss:
                if "points" in data and "depths" in data:
                    points = data["points"].to(device)  # [1, M, 2]
                    depths_gt = data["depths"].to(device)  # [1, M]
                    points_px = points.clone()  # preserve pixel coords for debug viz
                else:
                    points = None
                    depths_gt = None
                    points_px = None

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # Resolve the dynamic frame index so rigid objects are placed with the
            # pose that matches this training image's capture time.
            rigid_frame_idx = self._resolve_rigid_frame_idx(data)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
                frame_idcs=image_ids,
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
                rigid_frame_idx=rigid_frame_idx,
            )
            if renders.shape[-1] == 4:  # If render includes depth (RGB+ED), split into colors and depths. Otherwise, treat all channels as colors.
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            # Prep for backward pass
            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            if masks is not None:
                # Exclude masked pixels (e.g. ego vehicle) from L1.
                # For SSIM (patch-based), zero out both sides at masked locations
                # so masked patches don't pull colors toward an arbitrary value.
                l1loss = F.l1_loss(colors[masks], pixels[masks])
                colors_ssim = colors * masks[..., None]
                pixels_ssim = pixels * masks[..., None]
            else:
                l1loss = F.l1_loss(colors, pixels)
                colors_ssim = colors
                pixels_ssim = pixels
                
            ssimloss = 1.0 - fused_ssim(
                colors_ssim.permute(0, 3, 1, 2),
                pixels_ssim.permute(0, 3, 1, 2),
                padding="valid",
            )
            # Loss = (1 - ssim_lambda) * L1 + ssim_lambda * (1 - SSIM)
            loss = torch.lerp(l1loss, ssimloss, cfg.ssim_lambda)
            if cfg.depth_loss and points is not None and depths_gt is not None:
                # query rendered depths at COLMAP point locations
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # only supervise where Gaussians actually render (depth > 0)
                # and GT depth is not too close (prevents extreme disparities)
                valid_depth_mask = (depths > 0.0) & (depths_gt > 0.1)
                if valid_depth_mask.sum() > 0:
                    disp = 1.0 / depths[valid_depth_mask]
                    disp_gt = 1.0 / depths_gt[valid_depth_mask]
                    depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                    loss += depthloss * cfg.depth_lambda
                else:
                    depthloss = torch.tensor(0.0, device=device)
            if cfg.post_processing == "bilateral_grid":
                post_processing_reg_loss = 10 * total_variation_loss(
                    self.post_processing_module.grids
                )
                loss += post_processing_reg_loss
            elif cfg.post_processing == "ppisp":
                post_processing_reg_loss = (
                    self.post_processing_module.get_regularization_loss()
                )
                loss += post_processing_reg_loss

            # regularizations
            if cfg.opacity_reg > 0.0: # penalizes Gaussians that stay too opaque
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg > 0.0: # penalizes Gaussians that grow too large
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            # Rigid regularization: temporal smoothness + sharp-shape + unicycle.
            # Strategy-dispatch and step-gating are fully inside compute_regularization_losses.
            if self.rigid_nodes is not None:
                loss = loss + self.rigid_nodes.compute_regularization_losses(
                    cfg, rigid_frame_idx, step
                )

            loss.backward()

            # Accumulate the rigid densification signal (3D positional gradient)
            # after backward and before the rigid optimizers zero their grads.
            if self.rigid_densifier is not None:
                self.rigid_densifier.update_state(self.rigid_nodes)

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss and points is not None:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            # log to TensorBoard
            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                # Log unicycle loss components separately for tuning.
                if (
                    self.rigid_nodes is not None
                    and getattr(cfg, "rigid_pose_smoothing", "finite_diff") == "unicycle"
                    and self.rigid_nodes.unicycle is not None
                ):
                    with torch.no_grad():
                        _uc = self.rigid_nodes.unicycle
                        _reg = _uc.reg_loss()
                        _pos = _uc.pos_loss()
                        self.writer.add_scalar("train/unicycle_reg_loss", _reg.item(), step)
                        self.writer.add_scalar("train/unicycle_pos_loss", _pos.item(), step)
                if cfg.depth_loss and points is not None:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.post_processing is not None:
                    self.writer.add_scalar(
                        "train/post_processing_reg_loss",
                        post_processing_reg_loss.item(),
                        step,
                    )
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                if cfg.depth_loss and points_px is not None:
                    depth_viz = (pixels[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
                    pts = points_px[0].detach().cpu().numpy().astype(np.int32)
                    for x, y in pts[:500]:
                        if 0 <= x < width and 0 <= y < height:
                            depth_viz[max(0,y-1):y+2, max(0,x-1):x+2] = [255, 0, 0]
                    self.writer.add_image(
                        "train/depth_points",
                        depth_viz,
                        step,
                        dataformats="HWC",
                    )
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3 # Log GPU memory usage in GB for this checkpoint step.
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                if self.post_processing_module is not None:
                    data["post_processing"] = self.post_processing_module.state_dict()
                data["optimizers"] = {
                    name: optimizer.state_dict()
                    for name, optimizer in self.optimizers.items()
                }
                if cfg.pose_opt:
                    data["pose_optimizers"] = [
                        optimizer.state_dict() for optimizer in self.pose_optimizers
                    ]
                if cfg.app_opt:
                    data["app_optimizers"] = [
                        optimizer.state_dict() for optimizer in self.app_optimizers
                    ]
                if self.post_processing_optimizers:
                    data["post_processing_optimizers"] = [
                        optimizer.state_dict()
                        for optimizer in self.post_processing_optimizers
                    ]
                data["schedulers"] = [scheduler.state_dict() for scheduler in schedulers]
                data["strategy_state"] = self.strategy_state
                # Persist rigid-object nodes (local Gaussians, per-frame poses,
                # and instance buffers) for the 4D extension.
                if self.rigid_nodes is not None:
                    data["rigid_nodes"] = self.rigid_nodes.state_dict()
                    if self.rigid_nodes.unicycle is not None:
                        data["unicycle_smoother"] = self.rigid_nodes.unicycle.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
                
            # checkpoint.pt saves step, splats, and optionally pose/app/post-processing modules. 
            # If i want to resume training from a chekpoint, still missing optimizer states and schedulers states, which can cause issues.
            # Done? need testing
            
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:

                if self.cfg.app_opt:
                    # eval at origin to bake the appeareance into the colors
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    sh0 = self.splats["sh0"]
                    shN = self.splats["shN"]

                # save splats to .ply
                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

                # Also dump each rigid vehicle instance to its own PLY for
                # per-object reconstruction analysis.
                if self.rigid_nodes is not None:
                    self.export_rigid_plys(step)
                    self.export_trajectory_bev(step)

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)   #Clears gradients after each update so next iteration starts clean
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.post_processing_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Rigid-object optimizers (Gaussian attributes + per-frame poses).
            if self.rigid_nodes is not None:
                for optimizer in self.rigid_gauss_optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for optimizer in self.rigid_pose_optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for optimizer in self.unicycle_optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            # Densification strategy!
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # Rigid-object densification (grow/prune on the 3D positional grad).
            # Runs independently of the background MCMC strategy above.
            if self.rigid_densifier is not None:
                self.rigid_densifier.step(
                    self.rigid_nodes, self.rigid_gauss_optimizers, step
                )

            # Rigid opacity reset (OmniRe: every 3000 steps, clamp to max 0.01).
            # Forces saturated / dead Gaussians to re-compete via densification.
            if (
                self.rigid_nodes is not None
                and cfg.rigid_reset_opacity_every > 0
                and step > 0
                and step % cfg.rigid_reset_opacity_every == 0
            ):
                _max_logit = torch.logit(torch.tensor(0.01))
                self.rigid_nodes.gauss["opacities"].data.clamp_(max=_max_logit.item())

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

        # Save camera metadata at the end of training for standalone rendering
        self.save_camera_metadata()

    def save_camera_metadata(self) -> None:
        """Save per-camera pose sequences and metadata for standalone rendering.

        This method groups frames by camera and saves complete metadata including
        intrinsics and distortion coefficients. Called at the end of training to
        enable standalone rendering without re-parsing the dataset.
        """
        print("Saving camera metadata for standalone rendering...")
        frames_by_camera = self._group_frames_by_camera()
        self._save_camera_paths(frames_by_camera)
        print(f"Camera metadata saved to {self.cfg.result_dir}/camera_paths/")

    @torch.no_grad()
    def export_rigid_plys(self, step: int) -> None:
        """Export each rigid instance's Gaussians to its own PLY (local frame).

        Writes one PLY per vehicle instance under ``result_dir/ply`` (alongside the
        background ``point_cloud_*.ply``), in the object's canonical *local* frame
        (before the per-frame SE(3) pose), so the reconstruction of each object can
        be inspected on its own — independent of motion. No-op when dynamic rigid
        objects are disabled.

        When ``cfg.rigid_ply_dc_only`` is set, the higher-order SH bands are
        dropped so external viewers show a flat, view-independent color. Rigid
        vehicles are only seen from a narrow range of angles in training, so their
        view-dependent SH overfits and "explodes" into rainbow colors when
        free-orbited in a viewer (SuperSplat etc.) — dropping it makes the base
        geometry/color inspectable.
        """
        if self.rigid_nodes is None:
            return

        rigid_dir = self.ply_dir
        os.makedirs(rigid_dir, exist_ok=True)
        dc_only = bool(getattr(self.cfg, "rigid_ply_dc_only", True))

        gauss = self.rigid_nodes.gauss
        point_ids = self.rigid_nodes.point_ids
        n_written = 0
        for m in range(self.rigid_nodes.num_instances):
            sel = (point_ids == m).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            tid = self.rigid_nodes.instance_ids[m]
            cname = str(self.rigid_nodes.class_names[m]).replace("/", "_")
            if dc_only:
                # Empty higher-order SH -> viewer renders view-independent DC color.
                shN = torch.empty(
                    (sel.numel(), 0, 3), device=gauss["shN"].device
                )
            else:
                shN = gauss["shN"][sel]
            # export_splats consumes raw params (log scales, logit opacities),
            # exactly as stored — same convention as the background PLY export.
            export_splats(
                means=gauss["means"][sel],
                scales=gauss["scales"][sel],
                quats=gauss["quats"][sel],
                opacities=gauss["opacities"][sel],
                sh0=gauss["sh0"][sel],
                shN=shN,
                format="ply",
                save_to=os.path.join(
                    rigid_dir, f"instance{m:03d}_id{tid}_{cname}_step{step}.ply"
                ),
            )
            n_written += 1
        print(
            f"[Dynamic] Exported {n_written} rigid instance PLYs to {rigid_dir}"
            f"{' (DC-only color)' if dc_only else ''}"
        )

    @torch.no_grad()
    def export_trajectory_bev(self, step: int) -> None:
        """Export per-object BEV trajectory plots with ego context.

        For each tracked vehicle instance, creates a PNG showing:
        - Individual vehicle's trajectory on the X-Z ground plane.
        - Ego trajectory as a reference.
        - Start (triangle) and end (square) markers.

        Saves to ``result_dir/trajectories/{class_name}_{instance_id}_step{step}.png``.
        No-op when rigid nodes are not enabled or matplotlib is unavailable.
        """
        if self.rigid_nodes is None:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
        except ImportError:
            print("[BEV] matplotlib not available — skipping trajectory export.")
            return

        traj_dir = os.path.join(self.cfg.result_dir, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)

        # Compute scene bounds for consistent axis scaling across all object plots.
        poses_trans = self.rigid_nodes.poses["trans"].detach().cpu().numpy()  # (T, M, 3)
        fv = self.rigid_nodes.instances_fv.cpu().numpy()  # (T, M)

        all_valid = (fv.sum(axis=1) > 0)  # frames with any active instance
        if not all_valid.any():
            print("[BEV] No valid frames; skipping.")
            return

        # Scene bounds: include all object trajectories + ego.
        xs_all = poses_trans[all_valid, :, 0].flatten()
        zs_all = poses_trans[all_valid, :, 2].flatten()
        if hasattr(self.parser, "camtoworlds"):
            c2ws = np.array(self.parser.camtoworlds)
            xs_all = np.concatenate([xs_all, c2ws[:, 0, 3]])
            zs_all = np.concatenate([zs_all, c2ws[:, 2, 3]])
        x_min, x_max = xs_all.min(), xs_all.max()
        z_min, z_max = zs_all.min(), zs_all.max()
        x_margin = (x_max - x_min) * 0.1
        z_margin = (z_max - z_min) * 0.1
        x_lims = (x_min - x_margin, x_max + x_margin)
        z_lims = (z_min - z_margin, z_max + z_margin)

        # Get ego trajectory.
        ego_x, ego_z = None, None
        if hasattr(self.parser, "camtoworlds"):
            c2ws = np.array(self.parser.camtoworlds)
            ego_x = c2ws[:, 0, 3]
            ego_z = c2ws[:, 2, 3]

        # Create one PNG per instance.
        n_created = 0
        for m in range(self.rigid_nodes.num_instances):
            valid_frames = np.where(fv[:, m])[0]
            if len(valid_frames) < 2:
                continue

            xs = poses_trans[valid_frames, m, 0]
            zs = poses_trans[valid_frames, m, 2]
            tid = self.rigid_nodes.instance_ids[m]
            cname = self.rigid_nodes.class_names[m]

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_aspect("equal")
            ax.set_xlim(*x_lims)
            ax.set_ylim(*z_lims)
            ax.set_title(f"{cname} id={tid} — BEV Trajectory (step {step})", fontsize=12)
            ax.set_xlabel("X (training units)", fontsize=10)
            ax.set_ylabel("Z (training units)", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Instance trajectory in bright color.
            color_obj = (0.2, 0.7, 0.2)  # green
            ax.plot(xs, zs, "-o", markersize=4, linewidth=2.0, color=color_obj,
                    label=f"{cname} id={tid}")
            ax.plot(xs[0], zs[0], "^", markersize=12, color=color_obj, label="start")
            ax.plot(xs[-1], zs[-1], "s", markersize=12, color=color_obj, label="end")

            # Ego trajectory as context (faint).
            if ego_x is not None:
                ax.plot(ego_x, ego_z, "k--", linewidth=1.0, alpha=0.4, label="ego")
                ax.plot(ego_x[0], ego_z[0], "kv", markersize=8)

            ax.legend(loc="upper right", fontsize=10)

            # Safe filename: replace spaces/special chars.
            safe_cname = cname.replace(" ", "_").replace("/", "_")
            out_path = os.path.join(traj_dir, f"{safe_cname}_{tid:03d}_step{step:05d}.png")
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            n_created += 1
            print(f"[BEV] Saved {safe_cname} id={tid}: {out_path}")

        if n_created == 0:
            print(f"[BEV] No instances with ≥2 frames; no trajectories exported.")
        else:
            print(f"[BEV] Created {n_created} per-object trajectory PNGs at step {step}")

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Run full-image evaluation on the validation split (or a named stage).

        This method renders each sample in ``self.valset`` with the current Gaussian
        parameters, compares predictions against ground-truth images, writes side-by-side
        render outputs, and aggregates quality metrics.

        Behavior summary:
            1. Builds a deterministic dataloader over ``self.valset`` (batch size 1).
            2. For each view, renders RGB with ``rasterize_splats``.
            3. Saves comparison images (ground truth | prediction) to ``self.render_dir``.
            4. Computes PSNR, SSIM, and LPIPS on rank 0.
            5. Optionally computes color-corrected metrics if enabled by config.
            6. Writes aggregated metrics to JSON and TensorBoard.

        Args:
            step: Global training step used for filenames/logging.
            stage: Label used in output filenames and TensorBoard namespaces
                (for example ``"val"`` or ``"compress"``).

        Returns:
            None. Side effects include writing rendered PNGs, stats JSON files,
            and TensorBoard scalars.

        Notes:
            - ``@torch.no_grad()`` disables autograd tracking during this method,
              reducing memory usage and avoiding accidental gradient updates.
            - In distributed runs, metric aggregation/logging is performed on rank 0.
        """
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            # Exposure metadata is available for any image with EXIF data (train or val)
            exposure = data["exposure"].to(device) if "exposure" in data else None

            torch.cuda.synchronize()
            tic = time.time()
            rigid_idx = self._resolve_rigid_frame_idx(data)
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
                frame_idcs=None,  # For novel views, pass None (no per-frame parameters available)
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
                rigid_frame_idx=rigid_idx,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            pixels_masked = pixels * masks[..., None]
            
            colors = torch.clamp(colors, 0.0, 1.0) # Clamp render colors to [0, 1] range for fair metric computation and visualization.
            canvas_list = [pixels_masked, colors]

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                # Debug: also save a render with per-instance 3D boxes overlaid.
                if (
                    getattr(cfg, "debug_render_boxes", False)
                    and self.rigid_nodes is not None
                    and rigid_idx is not None
                ):
                    boxed, _, _ = self.rasterize_splats(
                        camtoworlds=camtoworlds,
                        Ks=Ks,
                        width=width,
                        height=height,
                        sh_degree=cfg.sh_degree,
                        near_plane=cfg.near_plane,
                        far_plane=cfg.far_plane,
                        masks=masks,
                        frame_idcs=None,
                        camera_idcs=data["camera_idx"].to(device),
                        exposure=exposure,
                        rigid_frame_idx=rigid_idx,
                        draw_boxes=True,
                    )
                    boxed = torch.clamp(boxed[..., :3], 0.0, 1.0)
                    boxed_canvas = torch.cat([pixels_masked, boxed], dim=2)
                    boxed_canvas = (boxed_canvas.squeeze(0).cpu().numpy() * 255).astype(
                        np.uint8
                    )
                    imageio.imwrite(
                        f"{self.render_dir}/{stage}_step{step}_{i:04d}_boxes.png",
                        boxed_canvas,
                    )

                pixels_p = pixels_masked.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                # Compute color-corrected metrics for fair comparison across methods
                if cfg.use_color_correction_metric:
                    if cfg.color_correct_method == "affine":
                        cc_colors = color_correct_affine(colors, pixels)
                    else:
                        cc_colors = color_correct_quadratic(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            if cfg.use_color_correction_metric:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            else:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj_old(self, step: int):
        """Render a trajectory video from the current Gaussian scene state.

        This method synthesizes a camera path, renders each pose with the current
        model, and writes an MP4 showing RGB and normalized expected-depth side by
        side for quick qualitative inspection.

        Behavior summary:
            1. Exits early when ``cfg.disable_video`` is enabled.
            2. Builds camera poses from parser poses using the configured path mode:
               ``raw``, ``interp``, ``ellipse``, or ``spiral``.
            3. Converts poses to homogeneous 4x4 transforms and moves them to device.
            4. Renders each pose with ``render_mode="RGB+ED"``.
            5. Normalizes depth per frame to [0, 1], concatenates RGB|depth, and
               appends frames to an MP4 writer.

        Args:
            step: Global training step used to name the output file
                (for example ``videos/traj_{step}.mp4``).

        Returns:
            None. Side effects include writing a trajectory video under
            ``{cfg.result_dir}/videos``.
        """
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        # TODO Prob good idea to refract this
        
        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "raw":
            # Use captured poses as-is
            camtoworlds_all = camtoworlds_all[:, :3, :]  # [N, 3, 4]
        elif cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")
        
        
    @torch.no_grad()
    def render_traj(self, step: int):
        """Render a trajectory video from the current Gaussian scene state."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        frames_root_dir = f"{video_dir}/frames"
        os.makedirs(frames_root_dir, exist_ok=True)

        print("Grouping original poses by camera...")
        frames_by_camera = self._group_frames_by_camera()

        # Persist the original grouped camera paths for debugging/reuse.
        self._save_camera_paths(frames_by_camera)

        # Map internal indices back to COLMAP/NCore camera IDs
        idx_to_camera_id = None
        if hasattr(self.parser, "camera_id_to_idx"):
            idx_to_camera_id = {v: k for k, v in self.parser.camera_id_to_idx.items()}

        processed_cameras = []

        # Render each camera independently.
        for cam_idx, frames in frames_by_camera.items():
            safe_cam_name, cam_frames_dir = self._render_single_camera_trajectory(
                cam_idx=cam_idx,
                frames=frames,
                idx_to_camera_id=idx_to_camera_id,
                video_dir=video_dir,
                frames_root_dir=frames_root_dir,
                step=step,
                cfg=cfg,
                device=device,
            )
            processed_cameras.append((safe_cam_name, cam_frames_dir))

        # 6. Build a single side-by-side video from all camera frame folders.
        self._compose_side_by_side_video(processed_cameras, video_dir, step)

    def _group_frames_by_camera(self) -> Dict[int, List[np.ndarray]]:
        """Group parser camtoworld poses by contiguous camera index."""
        frames_by_camera: Dict[int, List[np.ndarray]] = defaultdict(list)

        if hasattr(self.parser, "camera_indices"):
            camera_idx_per_frame = self.parser.camera_indices
        elif hasattr(self.parser, "camera_idx_per_frame"):
            camera_idx_per_frame = self.parser.camera_idx_per_frame
        else:
            raise ValueError(
                "Parser must provide per-frame camera indices via "
                "`camera_indices` or `camera_idx_per_frame`."
            )

        num_frames = len(self.parser.camtoworlds)
        for i in range(num_frames):
            cam_idx = camera_idx_per_frame[i]
            if torch.is_tensor(cam_idx):
                cam_idx = cam_idx.item()
            frames_by_camera[cam_idx].append(self.parser.camtoworlds[i])

        return frames_by_camera

    def _save_camera_paths(
        self,
        frames_by_camera: Dict[int, List[np.ndarray]],
    ) -> None:
        """Save per-camera pose sequences and metadata under result_dir/camera_paths.

        For each camera, saves:
        - camtoworlds.npy: [N, 4, 4] pose matrices
        - camtoworlds.json: same as JSON (legacy format)
        - camera_data.json: complete metadata including intrinsics and distortion
        """
        camera_paths_root = os.path.join(self.cfg.result_dir, "camera_paths")
        os.makedirs(camera_paths_root, exist_ok=True)

        # Build mapping from internal index to camera ID
        idx_to_camera_id = None
        if hasattr(self.parser, "camera_id_to_idx"):
            idx_to_camera_id = {v: k for k, v in self.parser.camera_id_to_idx.items()}

        for cam_idx, frames in frames_by_camera.items():
            # Determine camera ID and directory name
            if hasattr(self.parser, "camera_ids") and 0 <= cam_idx < len(self.parser.camera_ids):
                camera_id = self.parser.camera_ids[cam_idx]
            elif idx_to_camera_id is not None:
                camera_id = idx_to_camera_id[cam_idx]
            else:
                camera_id = str(cam_idx)

            safe_cam_name = str(camera_id).replace("/", "_")
            cam_dir = os.path.join(camera_paths_root, safe_cam_name)
            os.makedirs(cam_dir, exist_ok=True)

            # Save poses
            c2ws = np.stack(frames, axis=0)
            np.save(os.path.join(cam_dir, "camtoworlds.npy"), c2ws)

            # Legacy JSON format (poses only)
            with open(os.path.join(cam_dir, "camtoworlds.json"), "w") as f:
                json.dump({"camtoworlds": c2ws.tolist()}, f)

            # Build complete camera metadata
            camera_data = {
                "camera_id": camera_id,
                "camera_index": cam_idx,
                "camtoworlds": c2ws.tolist(),
            }

            # Save the normalization scale so the renderer can convert real-meter shifts
            if hasattr(self.parser, "transform") and self.parser.transform is not None:
                norm_scale = float(np.linalg.norm(self.parser.transform[:3, 0]))
                camera_data["world_to_normalized_scale"] = norm_scale

            # Add intrinsics if available
            camera_key = camera_id if camera_id in self.parser.Ks_dict else cam_idx
            if hasattr(self.parser, "Ks_dict") and camera_key in self.parser.Ks_dict:
                K = self.parser.Ks_dict[camera_key]
                width, height = self.parser.imsize_dict[camera_key]
                camera_data["intrinsics"] = {
                    "K": K.tolist() if isinstance(K, np.ndarray) else K,
                    "width": int(width),
                    "height": int(height),
                }

            # Add distortion coefficients if available (NCore datasets)
            if hasattr(self.parser, "camera_render_data") and camera_id in self.parser.camera_render_data:
                render_data = self.parser.camera_render_data[camera_id]
                distortion = {
                    "camera_model": render_data.camera_model,
                }

                # Serialize radial coefficients
                if render_data.radial_coeffs is not None:
                    distortion["radial_coeffs"] = render_data.radial_coeffs.tolist()
                else:
                    distortion["radial_coeffs"] = None

                # Serialize tangential coefficients
                if render_data.tangential_coeffs is not None:
                    distortion["tangential_coeffs"] = render_data.tangential_coeffs.tolist()
                else:
                    distortion["tangential_coeffs"] = None

                # Serialize thin prism coefficients
                if render_data.thin_prism_coeffs is not None:
                    distortion["thin_prism_coeffs"] = render_data.thin_prism_coeffs.tolist()
                else:
                    distortion["thin_prism_coeffs"] = None

                # Serialize ftheta coefficients (complex structure)
                if render_data.ftheta_coeffs is not None:
                    ftheta = render_data.ftheta_coeffs
                    distortion["ftheta_coeffs"] = {
                        "reference_poly": ftheta.reference_poly.name if hasattr(ftheta.reference_poly, "name") else str(ftheta.reference_poly),
                        "pixeldist_to_angle_poly": list(ftheta.pixeldist_to_angle_poly),
                        "angle_to_pixeldist_poly": list(ftheta.angle_to_pixeldist_poly),
                        "max_angle": ftheta.max_angle,
                        "linear_cde": list(ftheta.linear_cde),
                    }
                else:
                    distortion["ftheta_coeffs"] = None

                camera_data["distortion"] = distortion

            # Save complete camera metadata
            with open(os.path.join(cam_dir, "camera_data.json"), "w") as f:
                json.dump(camera_data, f, indent=2)

    def _render_single_camera_trajectory(
        self,
        cam_idx: int,
        frames: List[np.ndarray],
        idx_to_camera_id: Optional[Dict[int, Union[int, str]]],
        video_dir: str,
        frames_root_dir: str,
        step: int,
        cfg: Config,
        device: str,
    ) -> Tuple[str, str]:
        """Render one camera trajectory and save its video and frame images."""
        if hasattr(self.parser, "camera_ids") and 0 <= cam_idx < len(self.parser.camera_ids):
            cam_name = str(self.parser.camera_ids[cam_idx])
        elif idx_to_camera_id is not None:
            cam_name = f"camera_{idx_to_camera_id[cam_idx]}"
        else:
            cam_name = f"camera_{cam_idx}"

        print(f"Generating trajectory for {cam_name}...")
        safe_cam_name = str(cam_name).replace("/", "_")
        cam_frames_dir = f"{frames_root_dir}/{safe_cam_name}"
        os.makedirs(cam_frames_dir, exist_ok=True)

        c2ws = np.stack(frames, axis=0)  # [N, 3, 4] or [N, 4, 4]
        if c2ws.shape[-2] == 4:
            c2ws_3x4 = c2ws[:, :3, :]
        else:
            c2ws_3x4 = c2ws

        if cfg.render_traj_path == "raw":
            pass
        elif cfg.render_traj_path == "interp":
            c2ws_3x4 = generate_interpolated_path(c2ws_3x4, 1)
        elif cfg.render_traj_path == "ellipse":
            height = c2ws_3x4[:, 2, 3].mean()
            c2ws_3x4 = generate_ellipse_path_z(c2ws_3x4, height=height)
        elif cfg.render_traj_path == "spiral":
            c2ws_3x4 = generate_spiral_path(
                c2ws_3x4,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(f"Render trajectory type not supported: {cfg.render_traj_path}")

        c2ws_4x4 = np.concatenate(
            [c2ws_3x4, np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(c2ws_3x4), axis=0)],
            axis=1,
        )
        c2ws_tensor = torch.from_numpy(c2ws_4x4).float().to(device)

        if idx_to_camera_id is not None:
            camera_key = idx_to_camera_id[cam_idx]
        elif hasattr(self.parser, "camera_ids") and 0 <= cam_idx < len(self.parser.camera_ids):
            camera_key = self.parser.camera_ids[cam_idx]
        else:
            camera_key = cam_idx

        K_np = self.parser.Ks_dict[camera_key]
        w, h = self.parser.imsize_dict[camera_key]

        K_tensor = torch.from_numpy(K_np).float().to(device).unsqueeze(0)
        camera_idx_tensor = torch.tensor([cam_idx]).to(device)

        writer = imageio.get_writer(f"{video_dir}/traj_{step}_{safe_cam_name}.mp4", fps=30)
        # Dynamic rigid objects animate with the trajectory: captured frame i shows
        # the vehicles at their frame-i pose. Only valid for the 1:1 "raw" path
        # (interp/ellipse/spiral resample the poses, breaking the frame mapping).
        rigid_animatable = (
            self.rigid_nodes is not None and cfg.render_traj_path == "raw"
        )
        num_rigid_frames = (
            self.rigid_nodes.num_frames if self.rigid_nodes is not None else 0
        )
        for i in tqdm.trange(len(c2ws_tensor), desc=f"Rendering {cam_name}"):
            c2w = c2ws_tensor[i : i + 1]
            rigid_frame_idx = (
                i if (rigid_animatable and 0 <= i < num_rigid_frames) else None
            )
            renders, _, _ = self.rasterize_splats(
                camtoworlds=c2w,
                Ks=K_tensor,
                width=w,
                height=h,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB",
                camera_idcs=camera_idx_tensor,
                frame_idcs=None,
                image_ids=None,
                masks=None,
                exposure=None,
                rigid_frame_idx=rigid_frame_idx,
                draw_boxes=getattr(cfg, "debug_render_boxes", False),
            )

            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)
            canvas = colors.squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
            imageio.imwrite(f"{cam_frames_dir}/frame_{i:05d}.png", canvas)

        writer.close()
        print(f"Saved {video_dir}/traj_{step}_{safe_cam_name}.mp4")
        print(f"Saved frames to {cam_frames_dir}")
        return safe_cam_name, cam_frames_dir

    def _compose_side_by_side_video(
        self,
        processed_cameras: List[Tuple[str, str]],
        video_dir: str,
        step: int,
    ) -> None:
        """Compose all per-camera frame folders into one grid video."""
        if len(processed_cameras) <= 1:
            return

        print("Composing multi-camera grid video...")
        frame_file_lists = []
        for _, cam_frames_dir in processed_cameras:
            frame_files = sorted(
                [
                    f
                    for f in os.listdir(cam_frames_dir)
                    if f.startswith("frame_") and f.endswith(".png")
                ]
            )
            frame_file_lists.append(frame_files)

        min_frames = min(len(files) for files in frame_file_lists)
        if min_frames == 0:
            print("Skipping combined video: one or more cameras have no saved frames.")
            return

        num_cameras = len(processed_cameras)
        grid_cols = int(math.ceil(math.sqrt(num_cameras)))
        grid_rows = int(math.ceil(num_cameras / grid_cols))

        # Determine a fixed cell size so all grid frames have consistent dimensions.
        cell_h = 0
        cell_w = 0
        for (_, cam_frames_dir), frame_files in zip(processed_cameras, frame_file_lists):
            sample_path = os.path.join(cam_frames_dir, frame_files[0])
            sample = imageio.imread(sample_path)
            if sample.ndim == 2:
                sample = np.repeat(sample[..., None], 3, axis=2)
            elif sample.shape[-1] == 4:
                sample = sample[..., :3]
            cell_h = max(cell_h, sample.shape[0])
            cell_w = max(cell_w, sample.shape[1])

        combined_path = f"{video_dir}/traj_{step}_all_cams.mp4"
        combined_writer = imageio.get_writer(combined_path, fps=30)

        for frame_i in tqdm.trange(min_frames, desc="Composing all cameras"):
            panels = []

            for (_, cam_frames_dir), frame_files in zip(processed_cameras, frame_file_lists):
                frame_path = os.path.join(cam_frames_dir, frame_files[frame_i])
                panel = imageio.imread(frame_path)
                if panel.ndim == 2:
                    panel = np.repeat(panel[..., None], 3, axis=2)
                elif panel.shape[-1] == 4:
                    panel = panel[..., :3]
                pad_h = cell_h - panel.shape[0]
                pad_w = cell_w - panel.shape[1]
                if pad_h > 0 or pad_w > 0:
                    panel = np.pad(
                        panel,
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode="constant",
                        constant_values=0,
                    )
                panels.append(panel.astype(np.uint8))

            grid_rows_list = []
            for row in range(grid_rows):
                row_panels = []
                for col in range(grid_cols):
                    idx = row * grid_cols + col
                    if idx < len(panels):
                        row_panels.append(panels[idx])
                if row_panels:
                    row_strip = np.concatenate(row_panels, axis=1)
                else:
                    row_strip = np.zeros((cell_h, 0, 3), dtype=np.uint8)

                # Center each row in the full grid width when the row has fewer cameras.
                full_row_w = grid_cols * cell_w
                centered_row = np.zeros((cell_h, full_row_w, 3), dtype=np.uint8)
                x0 = (full_row_w - row_strip.shape[1]) // 2
                if row_strip.shape[1] > 0:
                    centered_row[:, x0 : x0 + row_strip.shape[1], :] = row_strip
                grid_rows_list.append(centered_row)

            merged = np.concatenate(grid_rows_list, axis=0)
            combined_writer.append_data(merged)

        combined_writer.close()
        print(f"Saved combined video to {combined_path}")

    @torch.no_grad()
    def export_ppisp_reports(self) -> None:
        """Export PPISP visualization reports (PDF) and parameter JSON."""
        if self.cfg.post_processing != "ppisp":
            return
        print("Exporting PPISP reports...")

        # Resolve per-frame camera indices (COLMAP vs NCore naming)
        if hasattr(self.parser, "camera_indices"):
            camera_idx_per_frame = self.parser.camera_indices
        elif hasattr(self.parser, "camera_idx_per_frame"):
            camera_idx_per_frame = self.parser.camera_idx_per_frame
        else:
            print("WARNING: Cannot export PPISP reports — no camera index mapping found")
            return

        # Compute frames per camera from training dataset
        num_cameras = self.parser.num_cameras
        frames_per_camera = [0] * num_cameras
        for idx in self.trainset.indices:
            cam_idx = camera_idx_per_frame[idx]
            frames_per_camera[cam_idx] += 1

        # Generate camera names
        if hasattr(self.parser, "camera_id_to_idx"):
            idx_to_camera_id = {v: k for k, v in self.parser.camera_id_to_idx.items()}
            camera_names = [f"camera_{idx_to_camera_id[i]}" for i in range(num_cameras)]
        elif hasattr(self.parser, "camera_ids"):
            camera_names = list(self.parser.camera_ids[:num_cameras])
        else:
            camera_names = [f"camera_{i}" for i in range(num_cameras)]

        # Export reports
        output_dir = Path(self.cfg.result_dir) / "ppisp_reports"
        pdf_paths = export_ppisp_report(
            self.post_processing_module,
            frames_per_camera,
            output_dir,
            camera_names=camera_names,
        )
        print(f"PPISP reports saved to {output_dir}")
        for path in pdf_paths:
            print(f"  - {path.name}")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        """Render one interactive viewer frame for the current camera state.

        This callback is consumed by the viewer UI and is called repeatedly as the
        user moves the camera or changes render settings. It converts UI state into
        rasterization inputs, renders one frame, applies mode-specific visualization,
        and returns an image array suitable for display.

        Behavior summary:
            1. Chooses preview or viewer resolution from ``render_tab_state``.
            2. Converts camera pose/intrinsics from numpy to torch tensors on the
               active device.
            3. Maps UI render mode labels to rasterizer render modes
               (RGB, accumulated depth, expected depth, alpha).
            4. Calls ``rasterize_splats`` and updates viewer statistics
               (total/rendered Gaussian counts).
            5. Post-processes output for visualization:
               - RGB: clamp to [0, 1]
               - Depth modes: normalize depth and apply selected colormap
               - Alpha: colormap over alpha channel

        Args:
            camera_state: Current viewer camera pose and helper methods for deriving
                intrinsics at a target resolution.
            render_tab_state: Viewer render controls (resolution, near/far planes,
                mode, colormap, clipping, inversion, and related options).

        Returns:
            A numpy array representing the rendered frame for the viewer. Shape is
            ``[H, W, 3]`` with values in [0, 1] for RGB/colormapped outputs.

        Notes:
            - ``@torch.no_grad()`` avoids autograd graph creation for interactive
              rendering, which reduces memory pressure and improves responsiveness.
            - For depth visualization, normalization can use either user-provided
              near/far or per-frame min/max depending on UI settings.
        """
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",  # Mixes depth and alpha to visualize where Gaussians contribute to the render, but depth values are not accurate. Useful for debugging.
            "depth(expected)": "ED",    # Visualizes expected depth, which is the weighted average of Gaussian depths with alpha as weights. More accurate than accumulated depth, but can be blurry when many Gaussians overlap.
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    # Import post-processing modules based on configuration
    # These imports must be here (not in __main__) for distributed workers
    if cfg.post_processing == "bilateral_grid":
        global BilateralGrid, slice, total_variation_loss
        if cfg.bilateral_grid_fused:
            from fused_bilagrid import (
                BilateralGrid,
                slice,
                total_variation_loss,
            )
        else:
            from lib_bilagrid import (
                BilateralGrid,
                slice,
                total_variation_loss,
            )
    elif cfg.post_processing == "ppisp":
        global PPISP, PPISPConfig, export_ppisp_report
        from ppisp import PPISP, PPISPConfig
        from ppisp.report import export_ppisp_report

    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None and not cfg.resume:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        if runner.post_processing_module is not None:
            pp_state = ckpts[0].get("post_processing")
            if pp_state is not None:
                runner.post_processing_module.load_state_dict(pp_state)
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.save_camera_metadata()  # Save metadata for standalone rendering
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()
        runner.export_ppisp_reports()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut and cfg.with_eval3d:
        print(
            "[Trainer] Note: with_ut=True + with_eval3d=True (full 3DGUT mode). "
            "DefaultStrategy is incompatible with eval3d; use MCMCStrategy (the `mcmc` subcommand)."
        )

    cli(main, cfg, verbose=True)
