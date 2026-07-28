"""Pure-numpy kinematic bicycle model shared across the pipeline.

Single source of truth for the bicycle equations so that:

* the offline **track fitter** (:mod:`src.tracking.bicycle_fit`),
* the **simulator / renderer** rollout (extrapolation past the observed span),

use *identical* kinematics. Fitting a model and then extrapolating it only
makes sense if the rollout used for the fit is the same one used later.

No torch, no scipy — just numpy — so it imports cleanly in the refinement env
(``env_cc3dt``) and the training/sim env (``env_gsplat``) alike.

Frame convention (COLMAP world, matching :mod:`src.tracking.track_geometry`):

* ``x`` (axis 0) and ``z`` (axis 2) span the ground plane; ``y`` (axis 1) is up.
* Heading ``theta`` is measured in the ground plane as ``atan2(dz, dx)`` so a
  vehicle with ``theta = 0`` moves along ``+x`` and ``theta = pi/2`` along ``+z``.

State vector is ``[x, z, theta, v]``; controls are ``[a, delta]`` (longitudinal
acceleration and front-wheel steering angle). The center-of-gravity variant is
used because the tracked box center is (approximately) the vehicle CoG rather
than the rear axle, so a slip angle ``beta`` is included.
"""
from __future__ import annotations

import math

import numpy as np

# COLMAP world: index 1 (y) is up; the ground/BEV plane is (x, z).
GROUND_AXES = (0, 2)
UP_AXIS = 1

# State / control layout (kept explicit so callers never index by magic number).
STATE_X = 0
STATE_Z = 1
STATE_THETA = 2
STATE_V = 3
STATE_DIM = 4


def wheelbase_from_size(size, alpha: float) -> float:
    """Estimate wheelbase ``L = alpha * max(size_x, size_y)``.

    The box ``size`` is ``[width, length, height]``; there is no direct
    wheelbase measurement, so we scale the larger of the two footprint extents.
    Using ``max`` sidesteps any width/length axis-order ambiguity. This is the
    exact formula used by the (now retired) training-time smoother.
    """
    s = np.asarray(size, dtype=np.float64)
    return float(alpha) * float(max(s[0], s[1]))


def slip_angle(steer, lr_ratio: float) -> np.ndarray:
    """Center-of-gravity slip angle ``beta = atan(lr_ratio * tan(delta))``.

    ``lr_ratio = lr / L`` is the CoG-to-rear-axle distance as a fraction of the
    wheelbase (``0.5`` places the CoG at the geometric center).
    """
    return np.arctan(lr_ratio * np.tan(np.asarray(steer, dtype=np.float64)))


def rollout(
    state0: np.ndarray,
    accel: np.ndarray,
    steer: np.ndarray,
    dt,
    wheelbase: float,
    lr_ratio: float = 0.5,
) -> np.ndarray:
    """Forward-Euler integrate the CoG bicycle model.

    Args:
        state0: Initial state ``[x, z, theta, v]``. Shape ``(4,)``.
        accel: Per-step longitudinal acceleration ``a_k``. Shape ``(K,)``.
        steer: Per-step steering angle ``delta_k``. Shape ``(K,)``.
        dt: Scalar seconds-per-step, or an array of shape ``(K,)``.
        wheelbase: Wheelbase ``L`` in meters.
        lr_ratio: CoG-to-rear-axle fraction ``lr / L``.

    Returns:
        States at every frame, shape ``(K + 1, 4)`` (includes ``state0``).
    """
    accel = np.asarray(accel, dtype=np.float64)
    steer = np.asarray(steer, dtype=np.float64)
    k = accel.shape[0]
    dt_arr = np.full(k, float(dt)) if np.isscalar(dt) else np.asarray(dt, dtype=np.float64)

    lr = max(lr_ratio * float(wheelbase), 1e-6)
    beta = slip_angle(steer, lr_ratio)  # (K,)

    states = np.empty((k + 1, STATE_DIM), dtype=np.float64)
    x, z, th, v = (float(s) for s in state0)
    states[0] = (x, z, th, v)

    # Sequential Euler integration. The per-step values are pulled into Python
    # lists and the trig uses the ``math`` module: for scalar work this is an
    # order of magnitude faster than numpy scalar ops, which matters because the
    # fitter evaluates this rollout thousands of times.
    beta_l = beta.tolist()
    sinb_l = np.sin(beta).tolist()
    accel_l = accel.tolist()
    dt_l = dt_arr.tolist()
    for i in range(k):
        h = dt_l[i]
        thb = th + beta_l[i]
        x += v * math.cos(thb) * h
        z += v * math.sin(thb) * h
        th += (v * sinb_l[i] / lr) * h
        v += accel_l[i] * h
        states[i + 1] = (x, z, th, v)
    return states


def heading_to_forward(theta: float) -> np.ndarray:
    """Ground-plane unit forward vector ``[cos(theta), 0, sin(theta)]`` (3D)."""
    return np.array([np.cos(theta), 0.0, np.sin(theta)], dtype=np.float64)


def wrap_angle(a):
    """Wrap angle(s) to ``(-pi, pi]``."""
    return (np.asarray(a, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi
