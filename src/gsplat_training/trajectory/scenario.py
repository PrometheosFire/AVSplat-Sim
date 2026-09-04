"""Scenario transforms: perturb a baked trajectory to author a new situation.

The simulator can only ever replay what was captured. These transforms let a
captured ego path -- or a single tracked vehicle's path -- be displaced sideways
by a constant offset, or made to weave along a sinusoid, so one capture yields
many synthetic scenarios.

Everything here is pure geometry over poses: no rendering, no torch tensors on
the ego side, and no knowledge of how the poses are later composited.

Axis convention
---------------
Shifts are ``[x_m, y_m, z_m]`` in real-world metres, matching
``configs/rendering/render.yaml`` and :class:`~.manipulator.TrajectoryShift`::

    x: lateral      (+ right,   - left)
    y: vertical     (+ down,    - up)
    z: longitudinal (+ forward, - backward)

For the **ego** these map directly onto the OpenCV camera axes
(``R[:,0] / R[:,1] / R[:,2]``), so a constant shift goes through the existing
:meth:`~.manipulator.TrajectoryManipulator.apply_translation` unchanged -- which
is what makes a simulator shift of ``[-2, 0, 0]`` reproduce the ``[-2, 0, 0]``
the offline render pipeline already produces.

For a **rigid object** the local frame is different (``x=length``, ``y=width``,
``z=height``; see ``dynamic/rigid_tracks.py``), so the same user-facing
``[x, y, z]`` is remapped onto it -- lateral onto local y, vertical onto local
-z, longitudinal onto local x -- keeping one convention for the user regardless
of what is being moved.

Metres are converted to the trainer's normalized units with
``world_to_normalized_scale`` (from ``camera_data.json``; ~0.053 for a typical
scene), exactly as the manipulator does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .manipulator import TrajectoryManipulator, TrajectoryShift


_MANIPULATOR = TrajectoryManipulator()


@dataclass
class SinusoidSpec:
    """A lateral sine weave applied on top of a baked trajectory.

    Attributes:
        enabled: When false every query returns zero offset.
        amplitude_m: Peak lateral displacement in metres.
        period_frames: Frames per full cycle. Values <= 0 disable the weave.
        phase_deg: Phase offset in degrees, so several objects can weave out of
            step with each other.
        follow_yaw: Rotate the heading to the path tangent. When false the pose
            crabs sideways keeping its original heading, which isolates lateral
            parallax from rotation.
    """

    enabled: bool = False
    amplitude_m: float = 0.0
    period_frames: float = 40.0
    phase_deg: float = 0.0
    follow_yaw: bool = True

    @classmethod
    def from_cfg(cls, node) -> "SinusoidSpec":
        """Build from an OmegaConf node, plain dict, or None (-> disabled)."""
        if node is None:
            return cls()

        def pick(key, default):
            try:
                value = node.get(key, default)
            except AttributeError:
                return default
            return default if value is None else value

        return cls(
            enabled=bool(pick("enabled", False)),
            amplitude_m=float(pick("amplitude_m", 0.0)),
            period_frames=float(pick("period_frames", 40.0)),
            phase_deg=float(pick("phase_deg", 0.0)),
            follow_yaw=bool(pick("follow_yaw", True)),
        )

    @property
    def usable(self) -> bool:
        """Geometry is meaningful, regardless of whether it is switched on.

        Kept separate from :attr:`active` so a caller can force the weave on
        without rewriting ``enabled`` -- the simulator's ``sinusoid`` mode does
        exactly that.
        """
        return abs(self.amplitude_m) > 1e-9 and self.period_frames > 1e-9

    @property
    def active(self) -> bool:
        return self.enabled and self.usable


def sinusoid_offset(
    spec: Optional[SinusoidSpec], frame: float, scale: float
) -> tuple[float, float]:
    """Lateral offset and its per-frame derivative, in normalized units.

    Gates on :attr:`SinusoidSpec.usable`, not :attr:`~SinusoidSpec.active`:
    passing a spec at all *is* the caller's decision to apply it, so the
    ``enabled`` flag stays a config/UI concern. Pass ``None`` to mean "off" --
    which is what every caller here does when a weave should not apply.

    Args:
        spec: The weave, or None.
        frame: Frame index (may be fractional).
        scale: ``world_to_normalized_scale`` (metres -> normalized units).

    Returns:
        ``(offset, d_offset_per_frame)``; ``(0.0, 0.0)`` when there is no usable
        weave. The derivative is what the tangent-yaw calculation needs.
    """
    if spec is None or not spec.usable:
        return 0.0, 0.0
    amp = spec.amplitude_m * scale
    omega = 2.0 * math.pi / spec.period_frames
    phase = math.radians(spec.phase_deg)
    angle = omega * float(frame) + phase
    return amp * math.sin(angle), amp * omega * math.cos(angle)


def tangent_yaw(d_offset: float, step_len: float) -> float:
    """Heading change that points along a laterally-displaced path.

    ``step_len`` is the along-track distance covered in one frame (normalized
    units). A stationary pose has no meaningful tangent, so it yields no yaw.
    """
    if step_len <= 1e-9:
        return 0.0
    return math.atan2(d_offset, step_len)


def step_length(positions: np.ndarray, frame: int) -> float:
    """Along-track distance per frame at ``frame``, from a position sequence.

    Uses a centred difference where possible, falling back to a one-sided
    difference at the ends. ``positions`` is ``(N, 3)`` in normalized units.
    """
    n = int(positions.shape[0])
    if n < 2:
        return 0.0
    i = int(np.clip(frame, 0, n - 1))
    lo = max(i - 1, 0)
    hi = min(i + 1, n - 1)
    span = max(hi - lo, 1)
    return float(np.linalg.norm(positions[hi] - positions[lo]) / span)


def _rot_about_y(angle: float) -> np.ndarray:
    """Rotation about the camera's local Y (down) axis.

    Positive angle turns the forward axis (+Z) toward the right (+X), so a
    heading derived from a rightward lateral velocity turns right. Mirrors the
    ``Ry`` used by the simulator's free-drive pose builder.
    """
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def apply_ego_scenario(
    camtoworld: np.ndarray,
    frame: float,
    scale: float,
    shift_m: Sequence[float] = (0.0, 0.0, 0.0),
    sinusoid: Optional[SinusoidSpec] = None,
    step_len: float = 0.0,
    plane_normal: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Displace one ego camera pose by a constant shift and/or a sine weave.

    Args:
        camtoworld: Base ``(4, 4)`` camera-to-world pose for this frame.
        frame: Frame index driving the sinusoid phase.
        scale: ``world_to_normalized_scale``.
        shift_m: Constant ``[x, y, z]`` shift in metres (see module docstring).
        sinusoid: Optional weave, applied laterally on top of the shift.
        step_len: Along-track distance per frame, for tangent yaw. See
            :func:`step_length`.
        plane_normal: Fitted ground-plane normal. When given, the weave runs
            along the right vector with its out-of-plane component removed, so a
            pitched camera does not drift vertically as it weaves. The constant
            shift deliberately does *not* use this -- it goes through
            ``apply_translation`` verbatim to stay bit-comparable with the
            offline render pipeline.

    Returns:
        A new ``(4, 4)`` pose. The input is not modified.
    """
    pose = np.asarray(camtoworld, dtype=np.float32).copy()

    shift = TrajectoryShift(
        name="sim",
        x_m=float(shift_m[0]),
        y_m=float(shift_m[1]),
        z_m=float(shift_m[2]),
    )
    if not shift.is_identity():
        pose = _MANIPULATOR.apply_shift(
            pose[None, ...], shift, world_to_normalized_scale=scale
        )[0]

    offset, d_offset = sinusoid_offset(sinusoid, frame, scale)
    if abs(offset) > 1e-12:
        right = pose[:3, 0].astype(np.float32)
        if plane_normal is not None:
            n = np.asarray(plane_normal, dtype=np.float32)
            flat = right - float(np.dot(right, n)) * n
            norm = float(np.linalg.norm(flat))
            if norm > 1e-6:
                right = flat / norm
        pose[:3, 3] += right * offset

    if sinusoid is not None and sinusoid.follow_yaw:
        yaw = tangent_yaw(d_offset, step_len)
        if abs(yaw) > 1e-12:
            pose[:3, :3] = pose[:3, :3] @ _rot_about_y(yaw)

    return pose


def apply_rigid_scenario(
    trans,
    quat,
    frame: float,
    scale: float,
    shift_m: Sequence[float] = (0.0, 0.0, 0.0),
    sinusoid: Optional[SinusoidSpec] = None,
    step_len: float = 0.0,
):
    """Displace one rigid instance's pose for a single frame.

    Operates on torch tensors, since rigid poses live in the checkpoint. The
    object-local frame is ``x=length`` (heading), ``y=width`` (lateral),
    ``z=height`` (up), so the user-facing ``[x, y, z]`` shift is remapped:
    lateral -> local y, vertical (+down) -> local -z, longitudinal -> local x.

    Args:
        trans: ``(3,)`` translation in the training frame.
        quat: ``(4,)`` wxyz box-to-world rotation.
        frame: Frame index driving the sinusoid phase.
        scale: ``world_to_normalized_scale``.
        shift_m: Constant ``[x, y, z]`` shift in metres.
        sinusoid: Optional lateral weave.
        step_len: Along-track distance per frame, for tangent yaw.

    Returns:
        ``(trans, quat)`` -- new tensors; the inputs are not modified.
    """
    import torch

    from dynamic.rigid_nodes import quat_multiply, quat_normalize, quat_to_rotmat

    q = quat_normalize(quat.reshape(1, 4)).reshape(4)
    R = quat_to_rotmat(q.reshape(1, 4)).reshape(3, 3)

    lateral_m, vertical_m, longitudinal_m = (float(v) for v in shift_m)
    offset, d_offset = sinusoid_offset(sinusoid, frame, scale)

    # Local-frame displacement, then rotate into the training frame.
    local = torch.tensor(
        [
            longitudinal_m * scale,             # local x: length / heading
            lateral_m * scale + offset,         # local y: width  / lateral
            -vertical_m * scale,                # local z: height (+shift = down)
        ],
        dtype=trans.dtype,
        device=trans.device,
    )
    new_trans = trans + R @ local

    new_quat = q
    if sinusoid is not None and sinusoid.follow_yaw:
        yaw = tangent_yaw(d_offset, step_len)
        if abs(yaw) > 1e-12:
            # Rotation about local z (up): turns +x (heading) toward +y.
            half = 0.5 * yaw
            q_yaw = torch.tensor(
                [math.cos(half), 0.0, 0.0, math.sin(half)],
                dtype=q.dtype,
                device=q.device,
            )
            new_quat = quat_multiply(q.reshape(1, 4), q_yaw.reshape(1, 4)).reshape(4)

    return new_trans, new_quat
