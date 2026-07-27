"""Rigid-object Gaussian nodes (vehicles) with per-frame SE(3) poses.

Each rigid instance owns a set of Gaussians defined in its *local* (object) frame.
A per-frame pose ``(quaternion, translation)`` maps those local Gaussians into
the world (training) frame: ``x_world = R[f] @ x_local + t[f]``. The local
Gaussian attributes are shared across all frames (time-invariant geometry), while
the per-frame poses carry the motion. This mirrors the OmniRe rigid-node design.

The model holds:
  * ``gauss``  - local-frame Gaussian attributes (means/scales/quats/opacities/SH).
  * ``poses``  - per-frame ``trans`` ``(T, M, 3)`` and ``quats`` ``(T, M, 4)``.
  * buffers    - ``point_ids`` (Gaussian -> instance), ``instances_size`` ``(M, 3)``
                 and ``instances_fv`` ``(T, M)`` frame-validity.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rigid_tracks import RigidTracks

# Local Gaussian parameter keys (order-independent); used for checkpoint resizing.
RIGID_GAUSS_PARAM_KEYS = ("means", "scales", "quats", "opacities", "sh0", "shN")


# ---------------------------------------------------------------------------
# Quaternion ops (wxyz convention), batched in torch
# ---------------------------------------------------------------------------
def quat_normalize(q: Tensor) -> Tensor:
    return F.normalize(q, dim=-1)


def quat_to_rotmat(q: Tensor) -> Tensor:
    """Convert ``(..., 4)`` wxyz quaternions to ``(..., 3, 3)`` rotation matrices."""
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    two = 2.0
    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    o = torch.stack(
        [
            1 - two * (yy + zz), two * (xy - wz), two * (xz + wy),
            two * (xy + wz), 1 - two * (xx + zz), two * (yz - wx),
            two * (xz - wy), two * (yz + wx), 1 - two * (xx + yy),
        ],
        dim=-1,
    )
    return o.reshape(q.shape[:-1] + (3, 3))


def quat_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """Hamilton product of two ``(..., 4)`` wxyz quaternions."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _random_quats(n: int) -> Tensor:
    q = torch.rand(n, 4)
    return F.normalize(q, dim=-1)


def _instance_palette(num_instances: int, device) -> Tensor:
    """Return ``(num_instances, 3)`` visually distinct RGB colors in [0, 1].

    Uses golden-ratio hue spacing so adjacent instance ids get well-separated
    hues (good for telling neighboring boxes apart).
    """
    import colorsys

    golden = 0.61803398875
    colors = [
        colorsys.hsv_to_rgb((i * golden) % 1.0, 0.65, 0.95)
        for i in range(max(1, num_instances))
    ]
    return torch.tensor(colors, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# RigidNodes model
# ---------------------------------------------------------------------------
class RigidNodes(nn.Module):
    """A collection of rigid Gaussian instances with learnable per-frame poses.

    World-frame Gaussians are produced on demand for a given frame index via
    :meth:`get_world_gaussians`, ready to be concatenated with the static
    background and rasterized in a single pass.
    """

    def __init__(
        self,
        tracks: RigidTracks,
        init_points_per_instance: int = 2000,
        init_opacity: float = 0.1,
        sh_degree: int = 3,
        scale_clamp: tuple = (1e-4, 1.0),
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.sh_degree = sh_degree
        self.num_sh_bases = (sh_degree + 1) ** 2
        self.scale_clamp = scale_clamp
        self.instance_ids: List[int] = list(tracks.instance_ids)
        self.class_names: List[str] = list(tracks.class_names)

        num_frames = tracks.num_frames
        num_inst = tracks.num_instances

        # --- Per-frame poses (learnable) ---
        self.poses = nn.ParameterDict(
            {
                "trans": nn.Parameter(torch.from_numpy(tracks.trans).float()),
                "quats": nn.Parameter(torch.from_numpy(tracks.quats).float()),
            }
        )

        # --- Per-instance / per-frame buffers (not learnable) ---
        self.register_buffer(
            "instances_size", torch.from_numpy(tracks.sizes).float()
        )  # (M, 3)
        self.register_buffer(
            "instances_fv", torch.from_numpy(tracks.valid).bool()
        )  # (T, M)

        # --- Local-frame Gaussians, built per instance ---
        means, scales, quats, opacities, sh0, shN, point_ids = self._init_gaussians(
            tracks, init_points_per_instance, init_opacity
        )
        self.gauss = nn.ParameterDict(
            {
                "means": nn.Parameter(means),
                "scales": nn.Parameter(scales),
                "quats": nn.Parameter(quats),
                "opacities": nn.Parameter(opacities),
                "sh0": nn.Parameter(sh0),
                "shN": nn.Parameter(shN),
            }
        )
        self.register_buffer("point_ids", point_ids)  # (N,) long, instance column

        self.num_frames = num_frames
        self.num_instances = num_inst
        # Optional kinematic smoothers — attached after construction in runner
        # according to rigid_pose_smoothing strategy.
        self.unicycle = None
        self.bicycle = None
        
        # Store reference sizes for validation (sizes must never change per-frame).
        # Register as buffer so it moves to device with .to(device).
        self.register_buffer("_sizes_reference", self.instances_size.clone().detach())
        
        self.to(device)

    # ------------------------------------------------------------------
    # Validation & Safety
    # ------------------------------------------------------------------
    def validate_sizes_unchanged(self) -> bool:
        """Check that per-instance box sizes have not been modified.
        
        Box sizes should be constant per instance (computed as median from tracker).
        If this check fails, it indicates a bug allowing per-frame size variation.
        Returns True if sizes match reference, False if changed.
        """
        if not torch.allclose(self.instances_size, self._sizes_reference, atol=1e-6):
            changed = (self.instances_size - self._sizes_reference).abs()
            print(f"[WARNING] BBox sizes have changed!\n"
                  f"  Max delta: {changed.max().item():.6f}\n"
                  f"  Per-instance max deltas: {changed.max(dim=-1).values}")
            return False
        return True

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_gaussians(
        self, tracks: RigidTracks, n_per: int, init_opacity: float
    ):
        from utils import knn, rgb_to_sh  # local import; lives in gsplat_training

        means_list, ids_list, scales_list = [], [], []
        for m in range(tracks.num_instances):
            size = torch.from_numpy(tracks.sizes[m]).float().clamp_min(1e-3)  # (3,)
            # Uniform random points within the box [-size/2, +size/2].
            pts = (torch.rand(n_per, 3) - 0.5) * size[None, :]
            means_list.append(pts)
            ids_list.append(torch.full((n_per,), m, dtype=torch.long))
            # Per-instance isotropic init scale from local kNN (instances are all
            # origin-centered in local space, so a GLOBAL kNN would mix them).
            dist2 = knn(pts, 4)[:, 1:] ** 2  # (n, 3) squared dists to 3 NN
            dist = torch.sqrt(dist2.mean(dim=-1)).clamp(*self.scale_clamp)  # (n,)
            scales_list.append(torch.log(dist).unsqueeze(-1).repeat(1, 3))

        means = torch.cat(means_list, dim=0)
        point_ids = torch.cat(ids_list, dim=0)
        scales = torch.cat(scales_list, dim=0)
        n = means.shape[0]

        quats = _random_quats(n)
        opacities = torch.logit(torch.full((n,), init_opacity))
        # Random colors -> SH DC; higher-order bands zero.
        rgbs = torch.rand(n, 3)
        colors = torch.zeros(n, self.num_sh_bases, 3)
        colors[:, 0, :] = rgb_to_sh(rgbs)
        sh0 = colors[:, :1, :].contiguous()
        shN = colors[:, 1:, :].contiguous()
        return means, scales, quats, opacities, sh0, shN, point_ids

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    @property
    def num_points(self) -> int:
        return self.gauss["means"].shape[0]

    def get_world_gaussians(
        self, frame_idx: int
    ) -> Optional[Dict[str, Tensor]]:
        """Transform active instances' local Gaussians into the world frame.

        Args:
            frame_idx: Global frame index in ``[0, num_frames)``.

        Returns:
            Dict with ``means`` ``(P, 3)``, ``quats`` ``(P, 4)``, ``scales``
            ``(P, 3)``, ``opacities`` ``(P,)``, ``colors`` ``(P, K, 3)`` and
            ``gaussian_idx`` ``(P,)`` (indices into the full local Gaussian set,
            so densification can scatter per-Gaussian render stats back) for the
            active Gaussians, or ``None`` if no instance is valid this frame.
        """
        inst = self.point_ids  # (N,)
        fv_f = self.instances_fv[frame_idx]  # (M,) bool
        active = fv_f[inst]  # (N,)
        if not bool(active.any()):
            return None

        idx = active.nonzero(as_tuple=True)[0]
        inst_a = inst[idx]  # (P,)
        local_means = self.gauss["means"][idx]  # (P, 3)
        local_quats = quat_normalize(self.gauss["quats"][idx])  # (P, 4)
        scales = torch.exp(self.gauss["scales"][idx])  # (P, 3)
        opacities = torch.sigmoid(self.gauss["opacities"][idx])  # (P,)
        colors = torch.cat([self.gauss["sh0"][idx], self.gauss["shN"][idx]], dim=1)

        # Per-frame pose for each active Gaussian's instance.
        pose_t = self.poses["trans"][frame_idx][inst_a]  # (P, 3)
        pose_q = quat_normalize(self.poses["quats"][frame_idx][inst_a])  # (P, 4)
        R = quat_to_rotmat(pose_q)  # (P, 3, 3)

        means_world = torch.bmm(R, local_means.unsqueeze(-1)).squeeze(-1) + pose_t
        quats_world = quat_multiply(pose_q, local_quats)
        return {
            "means": means_world,
            "quats": quats_world,
            "scales": scales,
            "opacities": opacities,
            "colors": colors,
            "gaussian_idx": idx,
        }

    @torch.no_grad()
    def get_box_edge_gaussians(
        self,
        frame_idx: int,
        opacity: float = 0.4,
        points_per_edge: int = 24,
        scale_frac: float = 0.03,
    ) -> Optional[Dict[str, Tensor]]:
        """Build semi-transparent, per-instance-colored box wireframes as Gaussians.

        For every instance valid at ``frame_idx``, samples points along the 12
        edges of its oriented 3D box (posed into the world/training frame) and
        returns them as ready-to-composite Gaussians (activated params: actual
        ``scales``, ``opacities`` in [0, 1], ``colors`` as SH coefficients). Drawing
        these through the normal rasterization pass makes the boxes line up exactly
        with the render, distortion included. Returns ``None`` if no instance is
        present this frame. Debug/visualization only — not optimized.
        """
        from utils import rgb_to_sh  # local import; lives in gsplat_training

        device = self.poses["trans"].device
        fv_f = self.instances_fv[frame_idx]  # (M,) bool
        active = fv_f.nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            return None

        # 8 unit-cube corners and the 12 edges connecting them.
        signs = torch.tensor(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
            dtype=torch.float32,
            device=device,
        )  # (8, 3)
        edges = torch.tensor(
            [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7),
             (0, 4), (1, 5), (2, 6), (3, 7)],
            dtype=torch.long,
            device=device,
        )  # (12, 2)
        line_t = torch.linspace(0.0, 1.0, points_per_edge, device=device)  # (P,)

        palette = _instance_palette(self.num_instances, device)  # (M, 3) rgb in [0,1]
        means_list, colors_list, scales_list = [], [], []
        for m in active.tolist():
            size = self.instances_size[m]  # (3,)
            corners_local = signs * (size[None, :] / 2.0)  # (8, 3)
            R = quat_to_rotmat(quat_normalize(self.poses["quats"][frame_idx][m]))
            t = self.poses["trans"][frame_idx][m]
            corners_world = (R @ corners_local.T).T + t  # (8, 3)
            c0 = corners_world[edges[:, 0]]  # (12, 3)
            c1 = corners_world[edges[:, 1]]  # (12, 3)
            pts = c0[:, None, :] + (c1 - c0)[:, None, :] * line_t[None, :, None]
            pts = pts.reshape(-1, 3)  # (12 * P, 3)
            means_list.append(pts)

            col = torch.zeros(pts.shape[0], self.num_sh_bases, 3, device=device)
            col[:, 0, :] = rgb_to_sh(palette[m][None, :]).expand(pts.shape[0], 3)
            colors_list.append(col)

            s = float((scale_frac * size.min()).clamp_min(1e-3))
            scales_list.append(torch.full((pts.shape[0], 3), s, device=device))

        means = torch.cat(means_list, dim=0)
        colors = torch.cat(colors_list, dim=0)
        scales = torch.cat(scales_list, dim=0)
        n = means.shape[0]
        quats = torch.zeros(n, 4, device=device)
        quats[:, 0] = 1.0
        opacities = torch.full((n,), float(opacity), device=device)
        return {
            "means": means,
            "quats": quats,
            "scales": scales,
            "opacities": opacities,
            "colors": colors,
        }

    # ------------------------------------------------------------------
    # Regularization
    # ------------------------------------------------------------------
    def sharp_shape_loss(self, max_ratio: float = 10.0) -> Tensor:
        """Penalise Gaussians whose aspect ratio exceeds ``max_ratio``.

        Mirrors OmniRe's "sharp shape" regulariser:
        ``mean(max(scale_max / scale_min, max_ratio) - max_ratio)``

        Zero when all Gaussians are rounder than ``max_ratio``; grows linearly
        for more elongated splats.  Prevents thin planar sheets from forming on
        vehicle surfaces, which confuse pose optimization from other viewpoints.
        """
        scales = torch.exp(self.gauss["scales"])          # (N, 3) actual scale
        scale_max = scales.max(dim=-1).values             # (N,)
        scale_min = scales.min(dim=-1).values.clamp_min(1e-6)
        ratio = scale_max / scale_min                     # (N,)
        return (ratio - max_ratio).clamp_min(0.0).mean()

    def temporal_smoothness_loss(
        self, frame_idx: int, smooth_range: int = 5
    ) -> Tensor:
        """Second-order finite-difference smoothness on per-frame translation.

        Penalizes acceleration ``|t[f+k] + t[f-k] - 2 * t[f]|`` for a random
        ``k`` in ``[1, smooth_range]``, averaged over instances valid at all
        three frames. Neighbor frames are detached, so the gradient only nudges
        the current frame's translation toward the midpoint of its neighbors.
        Returns a zero scalar when no valid triple exists (OmniRe convention:
        translation only, no rotation smoothing).
        """
        device = self.poses["trans"].device
        zero = torch.zeros((), device=device)
        fv = self.instances_fv
        if int(fv[frame_idx].sum()) == 0:
            return zero
        k = random.randint(1, max(1, smooth_range))
        if frame_idx - k < 0 or frame_idx + k >= self.num_frames:
            return zero
        valid = fv[frame_idx - k] & fv[frame_idx] & fv[frame_idx + k]
        if int(valid.sum()) == 0:
            return zero
        cur = self.poses["trans"][frame_idx][valid]
        prev = self.poses["trans"][frame_idx - k][valid].detach()
        nxt = self.poses["trans"][frame_idx + k][valid].detach()
        return (nxt + prev - 2.0 * cur).abs().mean()

    def compute_regularization_losses(
        self,
        cfg: object,
        frame_idx: Optional[int],
        step: int,
    ) -> Tensor:
        """Return a single weighted scalar regularization loss for this step.

        Dispatches on ``cfg.rigid_pose_smoothing``:
        - ``"finite_diff"``: second-order temporal smoothness on translation.
        - ``"unicycle"``: joint unicycle reg + pos loss (if smoother attached
          and within the configured iteration window).
                - ``"bicycle"``: joint bicycle reg + optional position anchoring +
                    pose-coupling losses (XZ center + yaw).

        Also applies sharp-shape regularization every
        ``cfg.rigid_sharp_shape_every`` steps regardless of smoothing strategy.

        Returns zero when nothing is applicable (frame_idx is None, all weights
        are zero, etc.).
        """
        device = self.poses["trans"].device
        loss = torch.zeros((), device=device)

        strategy = getattr(cfg, "rigid_pose_smoothing", "finite_diff")

        # --- Temporal / kinematic smoothness ---
        if frame_idx is not None:
            if strategy == "finite_diff":
                w = float(getattr(cfg, "rigid_smooth_w", 0.0))
                if w > 0.0:
                    loss = loss + w * self._temporal_smoothness_loss(
                        frame_idx, int(getattr(cfg, "rigid_smooth_range", 5))
                    )
            elif strategy == "unicycle" and self.unicycle is not None:
                start = int(getattr(cfg, "unicycle_joint_start_iter", 1000))
                end = int(getattr(cfg, "unicycle_joint_end_iter", 15000))
                if start <= step < end:
                    reg_w = float(getattr(cfg, "unicycle_joint_reg_w", 1e-3))
                    pos_w = float(getattr(cfg, "unicycle_joint_pos_w", 1e-4))
                    loss = loss + reg_w * self.unicycle.reg_loss()
                    if bool(getattr(cfg, "unicycle_opt_pos", True)):
                        loss = loss + pos_w * self.unicycle.pos_loss()
            elif strategy == "bicycle" and self.bicycle is not None:
                start = int(getattr(cfg, "bicycle_joint_start_iter", 1000))
                end = int(getattr(cfg, "bicycle_joint_end_iter", 15000))
                if start <= step < end:
                    reg_w = float(getattr(cfg, "bicycle_joint_reg_w", 1e-3))
                    pos_w = float(getattr(cfg, "bicycle_joint_pos_w", 1e-4))
                    couple_pos_w = float(getattr(cfg, "bicycle_joint_couple_pos_w", 1e-2))
                    couple_yaw_w = float(getattr(cfg, "bicycle_joint_couple_yaw_w", 1e-2))
                    loss = loss + reg_w * self.bicycle.reg_loss()
                    if bool(getattr(cfg, "bicycle_opt_pos", True)):
                        loss = loss + pos_w * self.bicycle.pos_loss()
                    c_pos, c_yaw = self.bicycle.pose_coupling_losses(
                        self.poses["trans"], self.poses["quats"]
                    )
                    loss = loss + couple_pos_w * c_pos + couple_yaw_w * c_yaw

        # --- Sharp-shape regularization (strategy-independent) ---
        every = int(getattr(cfg, "rigid_sharp_shape_every", 10))
        w_shape = float(getattr(cfg, "rigid_sharp_shape_w", 0.0))
        if w_shape > 0.0 and every > 0 and step % every == 0:
            loss = loss + w_shape * self._sharp_shape_loss(
                float(getattr(cfg, "rigid_sharp_shape_ratio", 10.0))
            )

        return loss

    def _sharp_shape_loss(self, max_ratio: float = 10.0) -> Tensor:
        """Penalise Gaussians whose aspect ratio exceeds ``max_ratio``."""
        return self.sharp_shape_loss(max_ratio)

    def _temporal_smoothness_loss(
        self, frame_idx: int, smooth_range: int = 5
    ) -> Tensor:
        """Second-order finite-difference smoothness on per-frame translation."""
        return self.temporal_smoothness_loss(frame_idx, smooth_range)

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    def create_unicycle_optimizers(
        self,
        lr_speed: float = 1e-3,
        lr_heading: float = 1e-4,
        lr_center: float = 1e-3,
    ):
        """Passthrough to ``self.unicycle.create_optimizers``.  Returns ``{}`` when
        no unicycle smoother is attached."""
        if self.unicycle is None:
            return {}
        return self.unicycle.create_optimizers(lr_speed, lr_heading, lr_center)

    def create_bicycle_optimizers(
        self,
        lr_speed: float = 1e-3,
        lr_steer: float = 1e-4,
        lr_center: float = 1e-3,
    ):
        """Passthrough to ``self.bicycle.create_optimizers``.  Returns ``{}`` when
        no bicycle smoother is attached."""
        if self.bicycle is None:
            return {}
        return self.bicycle.create_optimizers(lr_speed, lr_steer, lr_center)

    def create_optimizers(
        self,
        batch_size: int = 1,
        means_lr: float = 1.6e-4,
        scales_lr: float = 5e-3,
        quats_lr: float = 1e-3,
        opacities_lr: float = 5e-2,
        sh0_lr: float = 2.5e-3,
        shN_lr: float = 2.5e-3 / 20,
        pose_trans_lr: float = 5e-4,
        pose_quats_lr: float = 1e-5,
    ) -> Dict[str, torch.optim.Optimizer]:
        """Create per-parameter Adam optimizers (lrs scaled by sqrt(batch_size))."""
        bs = batch_size
        specs = {
            "means": (self.gauss["means"], means_lr),
            "scales": (self.gauss["scales"], scales_lr),
            "quats": (self.gauss["quats"], quats_lr),
            "opacities": (self.gauss["opacities"], opacities_lr),
            "sh0": (self.gauss["sh0"], sh0_lr),
            "shN": (self.gauss["shN"], shN_lr),
            "pose_trans": (self.poses["trans"], pose_trans_lr),
            "pose_quats": (self.poses["quats"], pose_quats_lr),
        }
        optimizers = {
            name: torch.optim.Adam(
                [{"params": param, "lr": lr * np.sqrt(bs), "name": name}],
                eps=1e-15 / np.sqrt(bs),
                betas=(1 - bs * (1 - 0.9), 1 - bs * (1 - 0.999)),
            )
            for name, (param, lr) in specs.items()
        }
        return optimizers

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True):
        """Load a checkpoint, resizing Gaussian tensors to the saved count.

        Densification changes the number of rigid Gaussians, so the local
        Gaussian parameters and ``point_ids`` buffer are reallocated to match the
        checkpoint's sizes before the values are copied in. Per-frame poses and
        instance buffers are fixed-size and load directly.
        """
        device = self.gauss["means"].device
        for key in RIGID_GAUSS_PARAM_KEYS:
            ckpt_key = f"gauss.{key}"
            if ckpt_key in state_dict:
                self.gauss[key] = nn.Parameter(
                    torch.empty_like(state_dict[ckpt_key], device=device),
                    requires_grad=self.gauss[key].requires_grad,
                )
        if "point_ids" in state_dict:
            self.point_ids = torch.empty_like(
                state_dict["point_ids"], device=device
            )
        return super().load_state_dict(state_dict, strict=strict)

