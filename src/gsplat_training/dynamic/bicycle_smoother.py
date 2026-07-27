"""Bicycle-model-based temporal smoother for rigid vehicle pose sequences.

This module mirrors the UnicycleSmoother integration style but uses kinematic
bicycle dynamics with a fixed (non-learned) wheelbase per instance.

State optimized per frame/instance:
- speed ``v`` (T, M)
- steering angle ``delta`` (T, M)
- optional planar centers ``center_x, center_z`` (T, M)

Wheelbase policy:
- fixed from bbox long edge in training units:
  L = wheelbase_alpha * max(size_x, size_y)
- no min/max clamping (as requested)
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .rigid_tracks import RigidTracks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yaw_from_quat_wxyz_np(q: np.ndarray) -> float:
    """Extract ground-plane yaw from a numpy wxyz quaternion."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    fx = 1 - 2 * (y * y + z * z)
    fz = 2 * (x * z + w * y)
    return math.atan2(fz, fx)


def _yaw_from_quat_wxyz_torch(q: Tensor) -> Tensor:
    """Extract ground-plane yaw from batched torch wxyz quaternions."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    fx = 1 - 2 * (y * y + z * z)
    fz = 2 * (x * z + w * y)
    return torch.atan2(fz, fx)


def _wrap_angle_torch(a: Tensor) -> Tensor:
    """Wrap angles to [-pi, pi] in a differentiable way."""
    return torch.atan2(torch.sin(a), torch.cos(a))


class BicycleSmoother(nn.Module):
    """Per-instance bicycle kinematic state + regularization losses."""

    def __init__(
        self,
        tracks: RigidTracks,
        opt_pos: bool = True,
        wheelbase_mode: str = "fixed_from_bbox_long_edge",
        wheelbase_alpha: float = 0.60,
        yaw_anchor_w: float = 5e-3,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        if wheelbase_mode != "fixed_from_bbox_long_edge":
            raise ValueError(
                f"Unsupported wheelbase_mode='{wheelbase_mode}'. "
                "Expected 'fixed_from_bbox_long_edge'."
            )

        T, M = tracks.num_frames, tracks.num_instances
        trans = tracks.trans  # (T, M, 3)
        quats = tracks.quats  # (T, M, 4)

        ts_us = tracks.frame_timestamps_us.astype(np.float64)
        ts_s = (ts_us - ts_us[0]) / 1e6
        self.register_buffer("timestamps_s", torch.from_numpy(ts_s).float())  # (T,)
        self.register_buffer("valid", torch.from_numpy(tracks.valid).bool())  # (T, M)

        # Fixed wheelbase per instance from bbox long edge in training units.
        # tracks.sizes stores [length, width, height] in training units.
        long_edge = np.maximum(tracks.sizes[:, 0], tracks.sizes[:, 1]).astype(np.float32)
        wheelbase = (float(wheelbase_alpha) * long_edge).astype(np.float32)
        self.register_buffer("wheelbase", torch.from_numpy(wheelbase))  # (M,)

        # Heading anchors from observed quaternions (used for init and yaw anchor loss).
        heading_anchor = np.zeros((T, M), dtype=np.float32)
        for t in range(T):
            for m in range(M):
                if tracks.valid[t, m]:
                    heading_anchor[t, m] = _yaw_from_quat_wxyz_np(quats[t, m])
        self.register_buffer("heading_anchor", torch.from_numpy(heading_anchor))

        # Initialize speed from consecutive translation differences.
        speed_np = np.zeros((T, M), dtype=np.float32)
        dt = np.diff(ts_s)
        for m in range(M):
            for t in range(1, T):
                if tracks.valid[t, m] and tracks.valid[t - 1, m] and dt[t - 1] > 1e-6:
                    dx = trans[t, m, 0] - trans[t - 1, m, 0]
                    dz = trans[t, m, 2] - trans[t - 1, m, 2]
                    speed_np[t, m] = math.sqrt(dx * dx + dz * dz) / dt[t - 1]
            if T > 1:
                speed_np[0, m] = speed_np[1, m]

        # Initialize steering from observed yaw-rate and speed.
        steer_np = np.zeros((T, M), dtype=np.float32)
        for m in range(M):
            L = max(float(wheelbase[m]), 1e-6)
            for t in range(T - 1):
                if tracks.valid[t, m] and tracks.valid[t + 1, m] and dt[t] > 1e-6:
                    dy = float(heading_anchor[t + 1, m] - heading_anchor[t, m])
                    dy = math.atan2(math.sin(dy), math.cos(dy))
                    yaw_rate = dy / dt[t]
                    v = max(float(speed_np[t, m]), 1e-4)
                    # tan(delta) = yaw_rate * L / v
                    steer_np[t, m] = math.atan(yaw_rate * L / v)
            if T > 1:
                steer_np[T - 1, m] = steer_np[T - 2, m]

        self.speed = nn.Parameter(torch.from_numpy(speed_np).float())
        self.steer = nn.Parameter(torch.from_numpy(steer_np).float())

        anchor_x = torch.from_numpy(trans[:, :, 0].copy()).float()
        anchor_z = torch.from_numpy(trans[:, :, 2].copy()).float()
        self.register_buffer("anchor_x", anchor_x)
        self.register_buffer("anchor_z", anchor_z)

        self.opt_pos = opt_pos
        self.yaw_anchor_w = float(yaw_anchor_w)
        if opt_pos:
            self.center_x = nn.Parameter(anchor_x.clone())
            self.center_z = nn.Parameter(anchor_z.clone())

        self.to(device)

    def _rollout_heading(self) -> Tensor:
        """Roll out heading from bicycle yaw-rate and initial observed heading (vectorized via cumsum)."""
        T, M = self.speed.shape
        device, dtype = self.speed.device, self.speed.dtype
        dt = (self.timestamps_s[1:] - self.timestamps_s[:-1]).clamp_min(1e-6)  # (T-1,)

        # Compute yaw-rate: dψ/dt = v/L * tan(δ)
        L = self.wheelbase.clamp_min(1e-6).unsqueeze(0).expand(T, -1)  # (T, M)
        yaw_rate = self.speed / L * torch.tan(self.steer)  # (T, M)

        # Compute yaw increments: Δψ[t] = dt[t-1] * yaw_rate[t-1]
        # Shape: dt is (T-1,), expand to (T-1, M), prepend 0 for frame 0
        dt_exp = dt.unsqueeze(1).expand(-1, M)  # (T-1, M)
        yaw_delta = torch.cat(
            [torch.zeros((1, M), device=device, dtype=dtype),
             dt_exp * yaw_rate[:-1]]
        )  # (T, M)

        # Integrate via cumsum to get continuous yaw profile.
        yaw_integrated = torch.cumsum(yaw_delta, dim=0)  # (T, M)

        # Anchor to first observed heading per instance.
        yaw = self.heading_anchor.clone()  # (T, M)
        yaw[:, :] = self.heading_anchor + yaw_integrated  # broadcast anchor + integrated delta

        # Handle discontinuities by resetting yaw to anchor where valid becomes False->True.
        # A discontinuity occurs at frame t if valid[t-1,m] is False and valid[t,m] is True.
        is_region_start = torch.cat([
            torch.ones((1, M), dtype=torch.bool, device=device),
            (self.valid[:-1] == False) & (self.valid[1:] == True)
        ])  # (T, M): True at frame 0 and after valid discontinuities

        # At region starts, reset yaw_integrated to 0 so yaw = heading_anchor.
        # For frames after region start, yaw_integrated grows from 0.
        # We do this by zeroing yaw_integrated at each region start and recomputing within regions.
        for m in range(M):
            starts = is_region_start[:, m].nonzero(as_tuple=True)[0].tolist()
            if not starts:
                continue
            # At each region start, reset the cumsum.
            prev_start = starts[0]
            yaw_integrated[prev_start, m] = 0.0
            for start in starts[1:]:
                yaw_integrated[prev_start:start, m] = torch.cumsum(
                    yaw_delta[prev_start:start, m], dim=0
                )
                yaw_integrated[start, m] = 0.0
                prev_start = start
            yaw_integrated[prev_start:, m] = torch.cumsum(
                yaw_delta[prev_start:, m], dim=0
            )

        # Final yaw with headings anchored at region starts.
        yaw = self.heading_anchor + yaw_integrated
        return yaw

    def reg_loss(self) -> Tensor:
        """Kinematic regularization: accel smooth + steer-rate smooth + consistency."""
        dt = (self.timestamps_s[1:] - self.timestamps_s[:-1]).clamp_min(1e-6)

        v = self.speed
        delta = self.steer
        yaw = self._rollout_heading()

        # Smooth acceleration (second-order on accel).
        dv = v[1:] - v[:-1]
        accel = dv / dt[:, None]
        d_accel = (accel[1:] - accel[:-1]).abs()
        valid_triple = (self.valid[:-2] & self.valid[1:-1] & self.valid[2:]).float()
        n_triple = valid_triple.sum().clamp_min(1.0)
        loss_accel = (d_accel * valid_triple).sum() / n_triple

        # Smooth steer-rate (second-order on delta rate).
        ddelta = delta[1:] - delta[:-1]
        steer_rate = ddelta / dt[:, None]
        d_steer_rate = (steer_rate[1:] - steer_rate[:-1]).abs()
        loss_steer = (d_steer_rate * valid_triple).sum() / n_triple

        # One-step position consistency from bicycle dynamics.
        if self.opt_pos:
            cx = self.center_x
            cz = self.center_z
        else:
            cx = self.anchor_x
            cz = self.anchor_z

        v_cur = v[:-1]
        yaw_cur = yaw[:-1]
        dt_exp = dt[:, None].expand_as(v_cur)
        pred_x = cx[:-1] + v_cur * dt_exp * torch.cos(yaw_cur)
        pred_z = cz[:-1] + v_cur * dt_exp * torch.sin(yaw_cur)
        actual_x = cx[1:].detach()
        actual_z = cz[1:].detach()

        consist_err = (pred_x - actual_x) ** 2 + (pred_z - actual_z) ** 2
        valid_pair = (self.valid[:-1] & self.valid[1:]).float()
        n_pair = valid_pair.sum().clamp_min(1.0)
        loss_consist = (consist_err * valid_pair).sum() / n_pair

        # Keep rolled-out heading close to observed heading where valid.
        yaw_err = _wrap_angle_torch(yaw - self.heading_anchor) ** 2
        valid = self.valid.float()
        n_valid = valid.sum().clamp_min(1.0)
        loss_yaw_anchor = (yaw_err * valid).sum() / n_valid

        return (
            0.01 * loss_accel
            + 0.1 * loss_steer
            + 1.0 * loss_consist
            + self.yaw_anchor_w * loss_yaw_anchor
        )

    def pos_loss(self) -> Tensor:
        """Anchor optional optimized centers to observed track centers."""
        if not self.opt_pos:
            return torch.zeros((), device=self.speed.device)

        valid = self.valid.float()
        n_valid = valid.sum().clamp_min(1.0)
        err_x = ((self.center_x - self.anchor_x) ** 2) * valid
        err_z = ((self.center_z - self.anchor_z) ** 2) * valid
        return 10.0 * (err_x.sum() + err_z.sum()) / n_valid

    def pose_coupling_losses(self, poses_trans: Tensor, poses_quats: Tensor) -> Tuple[Tensor, Tensor]:
        """Losses coupling rigid-node poses to bicycle-smoothed center/yaw."""
        if self.opt_pos:
            cx = self.center_x
            cz = self.center_z
        else:
            cx = self.anchor_x
            cz = self.anchor_z

        yaw_pred = self._rollout_heading()
        yaw_pose = _yaw_from_quat_wxyz_torch(poses_quats)

        valid = self.valid.float()
        n_valid = valid.sum().clamp_min(1.0)

        # Position coupling on XZ.
        err_pos = ((poses_trans[:, :, 0] - cx) ** 2 + (poses_trans[:, :, 2] - cz) ** 2) * valid
        loss_pos = err_pos.sum() / n_valid

        # Yaw coupling via wrapped angle error.
        err_yaw = (_wrap_angle_torch(yaw_pose - yaw_pred) ** 2) * valid
        loss_yaw = err_yaw.sum() / n_valid
        return loss_pos, loss_yaw

    def prefit(
        self,
        n_iters: int,
        reg_w: float = 5e-3,
        pos_w: float = 1e-3,
        lr_speed: float = 1e-3,
        lr_steer: float = 1e-4,
        lr_center: float = 1e-3,
    ) -> None:
        """Standalone optimization before main 4DGS training."""
        if n_iters <= 0:
            return

        param_groups = [
            {"params": [self.speed], "lr": lr_speed, "name": "bc_speed"},
            {"params": [self.steer], "lr": lr_steer, "name": "bc_steer"},
        ]
        if self.opt_pos:
            param_groups += [
                {"params": [self.center_x], "lr": lr_center, "name": "bc_cx"},
                {"params": [self.center_z], "lr": lr_center, "name": "bc_cz"},
            ]
        opt = torch.optim.Adam(param_groups, eps=1e-15)

        loss_before: Optional[float] = None
        print(
            f"[Bicycle prefit] Starting {n_iters} iterations: "
            f"reg_w={reg_w}, pos_w={pos_w}, lr_speed={lr_speed}, "
            f"lr_steer={lr_steer}, lr_center={lr_center}"
        )
        for i in range(n_iters):
            opt.zero_grad()
            loss = reg_w * self.reg_loss() + pos_w * self.pos_loss()
            loss.backward()
            opt.step()
            if i == 0:
                loss_before = loss.item()

        print(
            f"[Bicycle prefit] {n_iters} iters: "
            f"loss {loss_before:.6f} -> {loss.item():.6f}"
        )

    def create_optimizers(
        self,
        lr_speed: float = 1e-3,
        lr_steer: float = 1e-4,
        lr_center: float = 1e-3,
    ) -> Dict[str, torch.optim.Optimizer]:
        """Return per-parameter Adam optimizers for joint 4DGS training."""
        make = lambda p, lr, name: torch.optim.Adam(
            [{"params": p, "lr": lr, "name": name}], eps=1e-15
        )
        opts = {
            "bc_speed": make(self.speed, lr_speed, "bc_speed"),
            "bc_steer": make(self.steer, lr_steer, "bc_steer"),
        }
        if self.opt_pos:
            opts["bc_cx"] = make(self.center_x, lr_center, "bc_cx")
            opts["bc_cz"] = make(self.center_z, lr_center, "bc_cz")
        return opts
