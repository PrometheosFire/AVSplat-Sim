"""Drive a tracked vehicle's trajectory with a different 3DGS car model.

Substituting the Gaussians of a tracked instance -- while keeping its baked pose
sequence -- multiplies the scenarios one capture can yield: the same traffic,
with different vehicles in it.

Asset format (``data/3DRealCar/<name>/``)
-----------------------------------------
``gs.pth`` is a torch pickle of ``([...], iters)`` whose list is::

    [sh_degree, xyz (N,3), features_dc (N,1,3), features_rest (N,15,3),
     objects (N,20), scales (N,3) log, quats (N,4) wxyz, opacity (N,1) logit, None]

``wlh.json`` is ``[width, length, height]`` in **metres**, and is the size source
of truth: the Gaussian cloud measures ~1.10x the box on every axis because of
capture halo and ground residue, so cropping to the box is both a size fix and a
quality fix.

Coordinate frames
-----------------
Asset canonical frame is ``x = length`` (**+x is the front**), ``y = height
pointing DOWN with the origin at ground level`` (so the body spans ``y in
[-h, 0]``), ``z = width``.

The rigid-node local frame is box-centred and axis-aligned with ``x = length``
(**+x forward**), ``y = width``, ``z = height`` (**+z up**); see
``dynamic/rigid_tracks.py``. All four of those orientations were confirmed
against a trained checkpoint rather than assumed.

Mapping asset -> node is therefore a proper rotation (``R_x(-90 deg)``, det +1,
no reflection) plus a drop onto the box floor::

    node_x =  asset_x * s
    node_y =  asset_z * s
    node_z = -asset_y * s - box_h / 2

where ``s`` is ``world_to_normalized_scale``. Because the asset's origin sits at
ground level, subtracting half the tracked box height lands its wheels exactly on
the box floor. That is what keeps a compact car from floating when it replaces a
truck, whose box may be a metre taller than the car itself.
"""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


# Asset -> node local frame: (x, y, z)_asset -> (x, z, -y)_node. Equivalent to a
# -90 degree rotation about x, whose wxyz quaternion is below.
_Q_ASSET_TO_NODE = (math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0)
# 180 degree yaw about the node's up axis (+z), for the flip_forward escape hatch.
_Q_FLIP_YAW = (0.0, 0.0, 0.0, 1.0)


@dataclass
class AssetInfo:
    """One car model on disk."""

    name: str
    path: str
    width_m: float
    length_m: float
    height_m: float

    @property
    def wlh_m(self) -> Tuple[float, float, float]:
        return (self.width_m, self.length_m, self.height_m)

    def __str__(self) -> str:
        return (
            f"{self.name} ({self.length_m:.2f}x{self.width_m:.2f}x"
            f"{self.height_m:.2f} m)"
        )


@dataclass
class Asset:
    """Pruned Gaussians in the asset's own canonical frame, sizes in metres."""

    info: AssetInfo
    means: torch.Tensor  # (N, 3)
    scales: torch.Tensor  # (N, 3) log-space, metres
    quats: torch.Tensor  # (N, 4) wxyz
    opacities: torch.Tensor  # (N,) logit-space
    sh0: torch.Tensor  # (N, 1, 3)
    shN: torch.Tensor  # (N, K, 3)

    @property
    def num_points(self) -> int:
        return int(self.means.shape[0])


def list_assets(library_dir: str) -> List[AssetInfo]:
    """Every usable car model under ``library_dir``, sorted by name."""
    if not os.path.isdir(library_dir):
        return []
    out: List[AssetInfo] = []
    for name in sorted(os.listdir(library_dir)):
        d = os.path.join(library_dir, name)
        gs, wlh = os.path.join(d, "gs.pth"), os.path.join(d, "wlh.json")
        if not (os.path.exists(gs) and os.path.exists(wlh)):
            continue
        try:
            with open(wlh) as fp:
                w, l, h = (float(v) for v in json.load(fp))
        except (OSError, ValueError, TypeError):
            continue
        out.append(AssetInfo(name=name, path=d, width_m=w, length_m=l, height_m=h))
    return out


# Decoded assets, cached on the HOST. Never cache on the GPU: an entry there is a
# live reference the allocator cannot reclaim, so every model the user auditions
# in the picker would stay pinned for the life of the process (~167 MB each --
# four cars cost 621 MB that no gc or empty_cache could release). Assets move to
# the device only as part of the rigid bank, which is dropped when the
# substitution is removed. Bounded LRU so host RAM stays finite too.
_CACHE: "OrderedDict[tuple, Asset]" = OrderedDict()
_CACHE_MAX = 8


def clear_asset_cache() -> None:
    """Drop all decoded assets (host memory)."""
    _CACHE.clear()


def load_asset(
    library_dir: str,
    name: str,
    opacity_threshold: float = 0.0,
    crop_margin: Optional[float] = None,
    max_points: int = 750_000,
    dc_only: bool = True,
) -> Asset:
    """Load one car model, whole by default. Cached -- files are ~160 MB.

    Nothing is discarded unless asked for. Both filters below default to off,
    because every geometric trim tried here removed more car than junk:

    * ``opacity_threshold`` -- a 3DGS car is mostly *low*-opacity Gaussians (only
      ~15% exceed 0.5) and they carry much of its appearance, so raising this
      thins the car visibly. Use it to buy frame rate, not quality.
    * ``crop_margin`` -- bounds Gaussians to ``wlh`` times this factor. NOT a
      halo trim: 99.4% of an asset already lies within 1.2x its ``wlh`` box,
      because ``wlh.json`` under-states the true extent rather than the cloud
      having a halo. Cropping at 1.02 discarded ~61% of a car -- roof, front,
      rear and sides. Its floor at ``y = 0`` additionally trims below-ground
      content. Set it only for a pathological asset.
    * ``max_points`` -- hard ceiling, most opaque kept first.

    There is deliberately no unconditional below-ground trim. A flat ``y <= 0``
    cut does remove the patch of road some captures include, but ``y = 0`` is a
    *fitted* ground plane, and on low-chassis cars the cut takes the air dam,
    side skirts and tyre contact patches with it. The residue it would remove is
    0.01-0.6% of an asset, so leaving it in is the cheaper mistake; enable
    ``crop_margin`` if a particular asset drags visible road along with it.

    Raises:
        FileNotFoundError: when the asset is not in the library.
    """
    key = (os.path.abspath(library_dir), name, opacity_threshold, crop_margin,
           max_points, dc_only)
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit

    info = next((a for a in list_assets(library_dir) if a.name == name), None)
    if info is None:
        available = [a.name for a in list_assets(library_dir)]
        raise FileNotFoundError(
            f"asset {name!r} not found in {library_dir} "
            f"({len(available)} available, e.g. {available[:3]})"
        )

    payload = torch.load(
        os.path.join(info.path, "gs.pth"), map_location="cpu", weights_only=False
    )
    fields = payload[0] if isinstance(payload, (tuple, list)) else payload
    _, xyz, dc, rest, _obj, scales, quats, opacity = fields[:8]

    means = xyz.float()
    opacities = opacity.float().reshape(-1)
    keep = torch.ones(means.shape[0], dtype=torch.bool)

    # Optional box crop -- off by default; see the docstring for why. Its floor
    # at y=0 also trims below-ground content, since that is where the box ends.
    if crop_margin and crop_margin > 0:
        keep &= means[:, 0].abs() <= 0.5 * info.length_m * crop_margin
        keep &= means[:, 2].abs() <= 0.5 * info.width_m * crop_margin
        keep &= (means[:, 1] <= 0.0) & (means[:, 1] >= -info.height_m * crop_margin)

    # Opacity (default 0.0 -> no-op, since sigmoid is always positive).
    if opacity_threshold and opacity_threshold > 0:
        keep &= torch.sigmoid(opacities) > float(opacity_threshold)

    idx = keep.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:  # pathological thresholds -- keep the most opaque few
        idx = torch.argsort(opacities, descending=True)[: min(1000, means.shape[0])]

    # 3. Cap, most opaque first.
    if max_points and idx.numel() > int(max_points):
        order = torch.argsort(opacities[idx], descending=True)[: int(max_points)]
        idx = idx[order]

    # Under dc_only the higher-order SH is zeroed at conversion time, and it is
    # 76% of an asset's bytes -- so keep only its shape, not 127 MB of values we
    # are going to throw away. Expanded to real zeros in to_local_gaussians.
    shN = rest.float()[idx]
    if dc_only:
        shN = torch.zeros((0,) + tuple(shN.shape[1:]), dtype=shN.dtype)

    asset = Asset(
        info=info,
        means=means[idx],
        scales=scales.float()[idx],
        quats=quats.float()[idx],
        opacities=opacities[idx],
        sh0=dc.float()[idx],
        shN=shN,
    )
    _CACHE[key] = asset
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return asset


def _quat_mul(a: Tuple[float, ...], b: torch.Tensor) -> torch.Tensor:
    """Left-multiply every wxyz quaternion in ``b`` (N,4) by the constant ``a``."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=1,
    )


def to_local_gaussians(
    asset: Asset,
    box_size: torch.Tensor,
    scale: float,
    dc_only: bool = True,
    flip_forward: bool = False,
) -> Dict[str, torch.Tensor]:
    """Convert an asset into rigid-node local Gaussians for one instance.

    Args:
        asset: A loaded :class:`Asset` (canonical frame, metres).
        box_size: The instance's ``instances_size`` row ``[l, w, h]``, in
            normalized units. Only the height is used -- to drop the asset onto
            the box floor. Length and width are deliberately ignored: the asset
            keeps its true metric size, per the sizing decision.
        scale: ``world_to_normalized_scale`` (metres -> normalized units).
        dc_only: Zero the higher-order SH. The asset's ``features_rest`` is
            expressed in its own canonical orientation and is not rotated into
            world space, so keeping it produces view-dependent lighting that
            swims as the object turns. The training pipeline makes the same
            choice for rigid PLY export (``rigid_ply_dc_only``).
        flip_forward: Yaw the asset 180 degrees. Insurance for an asset whose
            +x faces the rear; not needed for the 3DRealCar library.

    Returns:
        Dict of ``means/scales/quats/opacities/sh0/shN`` ready to concatenate
        into a rigid bank. Scales stay log-space and opacities logit-space, to
        match the checkpoint's ``gauss.*`` convention.
    """
    s = float(scale)
    a = asset.means
    # (x, y, z)_asset -> (x, z, -y)_node, scaled to normalized units.
    means = torch.stack([a[:, 0] * s, a[:, 2] * s, -a[:, 1] * s], dim=1)
    quats = _quat_mul(_Q_ASSET_TO_NODE, asset.quats)

    if flip_forward:
        means = torch.stack([-means[:, 0], -means[:, 1], means[:, 2]], dim=1)
        quats = _quat_mul(_Q_FLIP_YAW, quats)

    # Drop the asset's ground plane (its y=0) onto the tracked box's floor.
    means[:, 2] -= 0.5 * float(box_size[2])

    # A cache entry loaded with dc_only carries an empty shN (shape only); every
    # Gaussian in one rasterization pass must share the SH basis count, so the
    # zeros are materialised here, at bank-build time, rather than cached.
    if dc_only or asset.shN.shape[0] != means.shape[0]:
        shN = torch.zeros(
            (means.shape[0],) + tuple(asset.shN.shape[1:]),
            dtype=asset.sh0.dtype,
            device=asset.sh0.device,
        )
    else:
        shN = asset.shN
    return {
        "means": means,
        "scales": asset.scales + math.log(s),  # log-space
        "quats": quats,
        "opacities": asset.opacities,  # logit-space
        "sh0": asset.sh0,
        "shN": shN,
    }


def build_rigid_bank(
    rigid_state: Dict[str, torch.Tensor],
    substitutions: Dict[int, Asset],
    scale: float,
    dc_only: bool = True,
    flip_forward: bool = False,
) -> Dict[str, torch.Tensor]:
    """A ``rigid_state``-shaped dict with some instances' Gaussians replaced.

    Keeping the same keys is what lets every existing helper
    (``rigid_world_gaussians_from_pose``, ``rigid_poses_at_frame``, ...) operate
    on the result untouched. Poses, sizes, validity and bicycle params are
    carried over by reference; only the ``gauss.*`` tensors and ``point_ids``
    are rebuilt.

    With no substitutions the input is returned unchanged, so the untouched
    checkpoint path stays exactly as it was.
    """
    if not substitutions:
        return rigid_state

    device = rigid_state["gauss.means"].device
    point_ids = rigid_state["point_ids"]
    sizes = rigid_state["instances_size"]
    keys = ("means", "scales", "quats", "opacities", "sh0", "shN")

    # Original Gaussians for every column that is NOT being replaced.
    keep = torch.ones_like(point_ids, dtype=torch.bool)
    for col in substitutions:
        keep &= point_ids != int(col)
    idx = keep.nonzero(as_tuple=True)[0]

    parts = {k: [rigid_state[f"gauss.{k}"][idx]] for k in keys}
    ids = [point_ids[idx]]

    for col, asset in sorted(substitutions.items()):
        local = to_local_gaussians(
            asset, sizes[int(col)], scale, dc_only=dc_only, flip_forward=flip_forward
        )
        for k in keys:
            parts[k].append(local[k].to(device))
        ids.append(
            torch.full((local["means"].shape[0],), int(col),
                       dtype=point_ids.dtype, device=device)
        )

    bank = dict(rigid_state)
    for k in keys:
        bank[f"gauss.{k}"] = torch.cat(parts[k], dim=0)
    bank["point_ids"] = torch.cat(ids, dim=0)
    return bank
