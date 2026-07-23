"""Unicycle-model-based temporal smoother for rigid vehicle pose sequences.

Mirrors the HUGS / HUGSIM unicycle model (NeurIPS 2024) adapted to work on
the per-instance, per-frame pose tensors produced by :class:`RigidTracks`.

Key design choices vs. HUGSIM:
- Speed and heading are ``nn.Parameter`` tensors, directly optimizable by Adam.
- Center X/Z are optionally trainable (guarded by ``unicycle_opt_pos`` config).
- Pitch/roll and height (Y) are held fixed; height is already handled by the
  per-instance plane fitting step in the tracking pipeline.
- The ``prefit`` loop runs *before* 4DGS training starts, standalone, without
  any Gaussian gradients, to put the kinematic state in a sensible basin.
- During joint training, only the regularization + anchoring loss is applied
  (no render-time pose replacement in this version).

Coordinate convention: training-frame (ncore-normalized), Y is up, (X, Z) is
the ground plane.  The unicycle model operates entirely in the X-Z plane.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .rigid_tracks import RigidTracks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yaw_from_quat_wxyz(q: np.ndarray) -> float:
    """Extract ground-plane yaw from a wxyz quaternion.

    Computes the angle of the local +X axis projected onto the X-Z plane,
    matching the convention used by the tracking pipeline.
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # Forward direction = R @ [1, 0, 0]
    fx = 1 - 2 * (y * y + z * z)
    fz = 2 * (x * z + w * y)
    return math.atan2(fz, fx)


# ---------------------------------------------------------------------------
# UnicycleSmoother
# ---------------------------------------------------------------------------

class UnicycleSmoother(nn.Module):
    """Per-instance unicycle kinematic state + regularization losses.

    Parameters (learnable):
        speed   ``(T, M)``  — planar speed [training-units / second]
        heading ``(T, M)``  — heading angle [radians]
        center_x ``(T, M)`` — X position (only if ``opt_pos=True``)
        center_z ``(T, M)`` — Z position (only if ``opt_pos=True``)

    Buffers (fixed):
        anchor_x  ``(T, M)`` — observed X centers from track (for pos_loss)
        anchor_z  ``(T, M)`` — observed Z centers from track (for pos_loss)
        timestamps_s ``(T,)`` — per-frame real time in seconds
        valid       ``(T, M)`` — frame-validity mask (from RigidTracks.valid)
    """

    def __init__(
        self,
        tracks: RigidTracks,
        opt_pos: bool = True,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        T, M = tracks.num_frames, tracks.num_instances
        trans = tracks.trans   # (T, M, 3)  training frame
        quats = tracks.quats   # (T, M, 4)  wxyz

        # ---- Timestamps (real seconds) ----
        ts_us = tracks.frame_timestamps_us.astype(np.float64)
        ts_s = (ts_us - ts_us[0]) / 1e6  # zero-referenced seconds
        self.register_buffer(
            "timestamps_s", torch.from_numpy(ts_s).float()
        )

        # ---- Validity mask ----
        self.register_buffer(
            "valid", torch.from_numpy(tracks.valid).bool()
        )  # (T, M)

        # ---- Initialize heading from quaternions ----
        heading_np = np.zeros((T, M), dtype=np.float32)
        for t in range(T):
            for m in range(M):
                if tracks.valid[t, m]:
                    heading_np[t, m] = _yaw_from_quat_wxyz(quats[t, m])

        # ---- Initialize speed from consecutive translations ----
        speed_np = np.zeros((T, M), dtype=np.float32)
        dt = np.diff(ts_s)  # (T-1,)
        for m in range(M):
            for t in range(1, T):
                if tracks.valid[t, m] and tracks.valid[t - 1, m] and dt[t - 1] > 1e-6:
                    dx = trans[t, m, 0] - trans[t - 1, m, 0]
                    dz = trans[t, m, 2] - trans[t - 1, m, 2]
                    speed_np[t, m] = math.sqrt(dx * dx + dz * dz) / dt[t - 1]
            # Frame 0: copy frame 1
            if T > 1:
                speed_np[0, m] = speed_np[1, m]

        # ---- Learnable parameters ----
        self.speed = nn.Parameter(
            torch.from_numpy(speed_np).float()
        )  # (T, M)
        self.heading = nn.Parameter(
            torch.from_numpy(heading_np).float()
        )  # (T, M)

        # ---- Observed centers (anchor for pos_loss) ----
        anchor_x = torch.from_numpy(trans[:, :, 0].copy()).float()
        anchor_z = torch.from_numpy(trans[:, :, 2].copy()).float()
        self.register_buffer("anchor_x", anchor_x)   # (T, M)
        self.register_buffer("anchor_z", anchor_z)   # (T, M)

        # ---- Optional learnable centers ----
        self.opt_pos = opt_pos
        if opt_pos:
            self.center_x = nn.Parameter(anchor_x.clone())  # (T, M)
            self.center_z = nn.Parameter(anchor_z.clone())  # (T, M)

        self.to(device)

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def reg_loss(self) -> Tensor:
        """Kinematic regularization: smooth accel + smooth yaw-rate + consistency.

        Three components (HUGS weights):
        1. Smooth acceleration:    ``mean(|Δv / Δt|)``  × 0.01
        2. Smooth yaw-rate:        ``mean(|Δyaw / Δt|)`` × 0.1
        3. Kinematic consistency:  one-step position prediction error × 1.0

        All terms operate only over valid consecutive frame pairs.
        """
        device = self.speed.device
        ts = self.timestamps_s  # (T,)
        dt = (ts[1:] - ts[:-1]).clamp_min(1e-6)  # (T-1,)

        v = self.speed      # (T, M)
        yaw = self.heading  # (T, M)

        # --- Smooth acceleration ---
        dv = v[1:] - v[:-1]            # (T-1, M)
        accel = dv / dt[:, None]       # (T-1, M)
        d_accel = (accel[1:] - accel[:-1]).abs()  # (T-2, M)
        # mask: three consecutive valid frames
        valid_triple = (
            self.valid[:-2] & self.valid[1:-1] & self.valid[2:]
        ).float()  # (T-2, M)
        n_triple = valid_triple.sum().clamp_min(1.0)
        loss_accel = (d_accel * valid_triple).sum() / n_triple

        # --- Smooth yaw-rate ---
        dyaw = yaw[1:] - yaw[:-1]
        yaw_rate = dyaw / dt[:, None]  # (T-1, M)
        d_yaw_rate = (yaw_rate[1:] - yaw_rate[:-1]).abs()  # (T-2, M)
        loss_yaw = (d_yaw_rate * valid_triple).sum() / n_triple

        # --- Kinematic consistency (one-step position prediction) ---
        # Predict next center from current using unicycle motion equation.
        if self.opt_pos:
            cx = self.center_x  # (T, M)
            cz = self.center_z  # (T, M)
        else:
            cx = self.anchor_x
            cz = self.anchor_z

        eps = 1e-6
        v_cur = v[:-1]             # (T-1, M)
        yaw_cur = yaw[:-1]         # (T-1, M)
        yaw_nxt = yaw[1:]          # (T-1, M)
        delta_yaw = yaw_nxt - yaw_cur
        dt_exp = dt[:, None].expand_as(v_cur)

        # Standard unicycle closed-form for constant yaw-rate:
        # Δx = v * Δt * sinc(Δyaw/2) * cos(yaw + Δyaw/2)
        # Δz = v * Δt * sinc(Δyaw/2) * sin(yaw + Δyaw/2)
        sinc_val = torch.where(
            delta_yaw.abs() < eps,
            torch.ones_like(delta_yaw),
            torch.sin(delta_yaw / 2.0) / (delta_yaw / 2.0 + eps),
        )
        mid_yaw = yaw_cur + delta_yaw / 2.0
        pred_x = cx[:-1] + v_cur * dt_exp * sinc_val * torch.cos(mid_yaw)
        pred_z = cz[:-1] + v_cur * dt_exp * sinc_val * torch.sin(mid_yaw)

        actual_x = cx[1:].detach()
        actual_z = cz[1:].detach()

        consist_err = (pred_x - actual_x) ** 2 + (pred_z - actual_z) ** 2
        valid_pair = (self.valid[:-1] & self.valid[1:]).float()
        n_pair = valid_pair.sum().clamp_min(1.0)
        loss_consist = (consist_err * valid_pair).sum() / n_pair

        return 0.01 * loss_accel + 0.1 * loss_yaw + 1.0 * loss_consist

    def pos_loss(self) -> Tensor:
        """Anchoring loss: keep optimized centers near observed track centers.

        ``10 * mean((opt_center - anchor)²)`` over valid frames, for both X and Z.
        Returns zero if ``opt_pos`` is False.
        """
        if not self.opt_pos:
            return torch.zeros((), device=self.speed.device)

        valid = self.valid.float()  # (T, M)
        n_valid = valid.sum().clamp_min(1.0)

        err_x = ((self.center_x - self.anchor_x) ** 2) * valid
        err_z = ((self.center_z - self.anchor_z) ** 2) * valid

        return 10.0 * (err_x.sum() + err_z.sum()) / n_valid

    # ------------------------------------------------------------------
    # Prefit
    # ------------------------------------------------------------------

    def prefit(
        self,
        n_iters: int,
        reg_w: float = 5e-3,
        pos_w: float = 1e-3,
        lr_speed: float = 1e-3,
        lr_heading: float = 1e-4,
        lr_center: float = 1e-3,
    ) -> None:
        """Run a standalone optimization loop before main 4DGS training.

        Minimizes ``reg_w * reg_loss() + pos_w * pos_loss()`` using its own
        internal Adam optimizer — no Gaussian gradients involved.
        Exits immediately when ``n_iters == 0``.
        """
        if n_iters <= 0:
            return

        param_groups = [
            {"params": [self.speed],   "lr": lr_speed,   "name": "uc_speed"},
            {"params": [self.heading], "lr": lr_heading, "name": "uc_heading"},
        ]
        if self.opt_pos:
            param_groups += [
                {"params": [self.center_x], "lr": lr_center, "name": "uc_cx"},
                {"params": [self.center_z], "lr": lr_center, "name": "uc_cz"},
            ]
        opt = torch.optim.Adam(param_groups, eps=1e-15)

        loss_before: Optional[float] = None
        
        print(f"[Unicycle prefit] Starting {n_iters} iterations: "
              f"reg_w={reg_w}, pos_w={pos_w}, lr_speed={lr_speed}, "
              f"lr_heading={lr_heading}, lr_center={lr_center}")
        for i in range(n_iters):
            opt.zero_grad()
            loss = reg_w * self.reg_loss() + pos_w * self.pos_loss()
            loss.backward()
            opt.step()
            if i == 0:
                loss_before = loss.item()

        loss_after = loss.item()
        print(
            f"[Unicycle prefit] {n_iters} iters: "
            f"loss {loss_before:.6f} → {loss_after:.6f}"
        )

    # ------------------------------------------------------------------
    # Optimizer factory (for joint training)
    # ------------------------------------------------------------------

    def create_optimizers(
        self,
        lr_speed: float = 1e-3,
        lr_heading: float = 1e-4,
        lr_center: float = 1e-3,
    ) -> Dict[str, torch.optim.Optimizer]:
        """Return per-parameter Adam optimizers for joint 4DGS training."""
        make = lambda p, lr, name: torch.optim.Adam(
            [{"params": p, "lr": lr, "name": name}], eps=1e-15
        )
        opts = {
            "uc_speed":   make(self.speed,   lr_speed,   "uc_speed"),
            "uc_heading": make(self.heading, lr_heading, "uc_heading"),
        }
        if self.opt_pos:
            opts["uc_cx"] = make(self.center_x, lr_center, "uc_cx")
            opts["uc_cz"] = make(self.center_z, lr_center, "uc_cz")
        return opts
