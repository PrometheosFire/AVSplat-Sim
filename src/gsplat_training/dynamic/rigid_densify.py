"""Manual densification for rigid-object Gaussians (OmniRe-style, Option 1).

Background Gaussians are densified by gsplat's MCMC strategy on ``self.splats``.
Rigid Gaussians live in a separate :class:`RigidNodes` module and are grown /
pruned here with a Default-3DGS-style adaptive control (clone / split on the
positional gradient + 3D size, cull on low opacity) plus an **out-of-bound cull**
that drops Gaussians drifting outside their instance box.

Densification signal
--------------------
This pipeline rasterizes with ``with_eval3d=True`` (3D-evaluation kernel), under
which gsplat does **not** put gradients on the projected 2D means
(``info["means2d"].grad`` is ``None``) — that is precisely why the background is
densified by MCMC rather than the gradient-based Default strategy. For rigid
objects we therefore use the **3D positional gradient** on the local means
(``rigid_nodes.gauss["means"].grad``) as the grow signal. Because the local means
map to world via a rotation (``x_world = R @ x_local + t``), this gradient's norm
equals the world-space positional gradient norm — the standard densification cue.

Both the per-Gaussian ``point_ids`` (Gaussian -> instance) and the running grow
stats (``grad3d`` / ``count``) are carried through every grow/prune op by placing
them in the ``state`` dict handed to gsplat's tensor-surgery helpers, which resize
params + Adam moments consistently.

This class is deliberately a small, swappable seam: a future variant could back
rigid densification with gsplat's ``DefaultStrategy`` while keeping the same
``update_state`` / ``step`` interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from gsplat.strategy.ops import duplicate, remove, split


@dataclass
class RigidDensifier:
    """Default-3DGS-style adaptive densification for rigid Gaussians.

    Thresholds are in **training units** (rigid objects are small, so unlike the
    background we do not scale by the global scene scale). ``grow_grad_thresh``
    is on the per-Gaussian 3D positional-gradient norm accumulated between
    refinements.
    """

    grow_grad_thresh: float = 4e-4
    grow_scale3d: float = 0.01  # clone if max scale <= this, else split
    prune_opacity: float = 0.05
    prune_scale3d: float = 0.5  # cull Gaussians larger than this (after warmup)
    cap_max: int = 500_000  # hard cap on TOTAL rigid Gaussians (all instances)
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 100
    warmup_for_big_prune: int = 3_000  # only cull "too big" after this step
    cull_out_of_bound: bool = True
    verbose: bool = False

    def __post_init__(self) -> None:
        self.state: Dict[str, Any] = {"grad3d": None, "count": None}

    # ------------------------------------------------------------------
    def initialize_state(self) -> Dict[str, Any]:
        self.state = {"grad3d": None, "count": None}
        return self.state

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_state(self, rigid_nodes) -> None:
        """Accumulate the local-means positional-gradient norm after backward.

        Must be called after ``loss.backward()`` and before the rigid optimizers
        zero their gradients. Gaussians not rendered this step have zero gradient
        and simply do not contribute (their ``count`` does not advance).
        """
        grad = rigid_nodes.gauss["means"].grad
        if grad is None:
            return
        gnorm = grad.norm(dim=-1)  # (N,)
        if self.state["grad3d"] is None:
            self.state["grad3d"] = torch.zeros(gnorm.shape[0], device=gnorm.device)
            self.state["count"] = torch.zeros(gnorm.shape[0], device=gnorm.device)
        self.state["grad3d"] += gnorm
        self.state["count"] += (gnorm > 0).float()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(
        self,
        rigid_nodes,
        optimizers: Dict[str, torch.optim.Optimizer],
        step: int,
    ) -> Optional[Tuple[int, int, int]]:
        """Grow (clone/split) then prune (opacity + out-of-bound) the rigid set.

        Args:
            rigid_nodes: The :class:`RigidNodes` module (its ``gauss`` params and
                ``point_ids`` buffer are mutated in place).
            optimizers: The rigid **Gaussian** optimizers (keys must match
                ``rigid_nodes.gauss``: means/scales/quats/opacities/sh0/shN).
                Pose optimizers must NOT be included.
            step: Current global training step.

        Returns:
            ``(n_dupli, n_split, n_prune)`` when a refinement ran, else ``None``.
        """
        if not (
            self.refine_start_iter < step < self.refine_stop_iter
            and step % self.refine_every == 0
        ):
            return None
        if self.state["grad3d"] is None:
            return None

        params = rigid_nodes.gauss
        # Carry point_ids + running stats through the tensor surgery.
        state = self.state
        state["point_ids"] = rigid_nodes.point_ids

        grads = state["grad3d"] / state["count"].clamp_min(1)
        device = grads.device
        scales_max = torch.exp(params["scales"]).max(dim=-1).values

        is_grad_high = grads > self.grow_grad_thresh
        is_small = scales_max <= self.grow_scale3d
        is_dupli = is_grad_high & is_small
        is_split = is_grad_high & (~is_small)
        n_dupli = int(is_dupli.sum())
        n_split = int(is_split.sum())

        # Cap the TOTAL rigid Gaussian count: clone adds 1 and split adds 1 net
        # per selected, so trim the growth set to the remaining budget, keeping
        # the highest-gradient candidates. Pruning below is unaffected.
        n_current = int(params["means"].shape[0])
        budget = max(0, self.cap_max - n_current)
        if n_dupli + n_split > budget:
            grow_mask = is_dupli | is_split
            grow_idx = grow_mask.nonzero(as_tuple=True)[0]
            keep_idx = grow_idx[
                torch.argsort(grads[grow_idx], descending=True)[:budget]
            ]
            keep_mask = torch.zeros_like(grow_mask)
            keep_mask[keep_idx] = True
            is_dupli = is_dupli & keep_mask
            is_split = is_split & keep_mask
            n_dupli = int(is_dupli.sum())
            n_split = int(is_split.sum())

        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)
        # Newly duplicated Gaussians are appended and must not be split this round.
        if n_dupli > 0:
            is_split = torch.cat(
                [is_split, torch.zeros(n_dupli, dtype=torch.bool, device=device)]
            )
        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=False,
            )

        # --- Prune: low opacity, (optionally) too-big, and out-of-bound ---
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opacity
        if step > self.warmup_for_big_prune:
            too_big = torch.exp(params["scales"]).max(dim=-1).values > self.prune_scale3d
            is_prune = is_prune | too_big
        if self.cull_out_of_bound:
            oob = self._out_of_bound_mask(rigid_nodes, state["point_ids"])
            is_prune = is_prune | oob
        n_prune = int(is_prune.sum())
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        # Write back the carried buffers and reset running stats.
        rigid_nodes.point_ids = state.pop("point_ids")
        self.state["grad3d"] = None
        self.state["count"] = None
        torch.cuda.empty_cache()

        if self.verbose:
            print(
                f"[RigidDensifier] step {step}: +{n_dupli} dup, +{n_split} split, "
                f"-{n_prune} prune -> {rigid_nodes.num_points} rigid GS"
            )
        return n_dupli, n_split, n_prune

    # ------------------------------------------------------------------
    @staticmethod
    def _out_of_bound_mask(rigid_nodes, point_ids: Tensor) -> Tensor:
        """Local Gaussians whose center escapes their instance box half-size."""
        half = rigid_nodes.instances_size[point_ids] / 2.0  # (N, 3)
        return (rigid_nodes.gauss["means"].abs() > half).any(dim=-1)
