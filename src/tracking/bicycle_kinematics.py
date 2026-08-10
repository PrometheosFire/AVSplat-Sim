"""Pure-numpy kinematic bicycle model shared across the pipeline.

Single source of truth for the bicycle equations so that:

* the offline **track fitter** (:mod:`src.tracking.bicycle_fit`),
* the **simulator / renderer** rollout (extrapolation past the observed span),

use *identical* kinematics. Fitting a model and then extrapolating it only
makes sense if the rollout used for the fit is the same one used later.

No torch, no scipy — just numpy — so it imports cleanly in the refinement env
(``env_cc3dt``) and the training/sim env (``env_gsplat``) alike. ``numba`` is
used when importable purely as an accelerator; every jitted routine has a pure
numpy fallback producing the same numbers, so a missing install costs speed and
nothing else.

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

try:  # Optional accelerator; every jitted routine below has a numpy fallback.
    from numba import njit as _njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - numba is an optional speed dependency
    HAVE_NUMBA = False

    def _njit(*args, **kwargs):
        """No-op stand-in so the jitted functions stay importable and callable."""

        def _decorate(fn):
            return fn

        if args and callable(args[0]) and not kwargs:
            return args[0]
        return _decorate


# ``fastmath`` is deliberately NOT enabled: it would permit reassociation of the
# floating-point accumulation and change results relative to the numpy path.
_JIT = dict(cache=True, fastmath=False, nogil=True)

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
    accel = np.ascontiguousarray(accel, dtype=np.float64)
    steer = np.ascontiguousarray(steer, dtype=np.float64)
    k = accel.shape[0]
    dt_arr = np.full(k, float(dt)) if np.isscalar(dt) else np.ascontiguousarray(dt, dtype=np.float64)

    lr = max(lr_ratio * float(wheelbase), 1e-6)
    s0 = np.ascontiguousarray(state0, dtype=np.float64).reshape(STATE_DIM)

    if HAVE_NUMBA:
        states = np.empty((k + 1, STATE_DIM), dtype=np.float64)
        _rollout_core(s0, accel, steer, dt_arr, lr, float(lr_ratio), states)
        return states
    return _rollout_numpy(s0, accel, steer, dt_arr, lr, float(lr_ratio))


@_njit(**_JIT)
def _rollout_core(state0, accel, steer, dt, lr, lr_ratio, out):  # pragma: no cover
    """Sequential Euler integration, jitted. Writes ``out`` of shape (K+1, 4).

    Kept as an explicit scalar loop (rather than the vectorized form below)
    because that is what compiles to tight machine code; the arithmetic and its
    ordering match :func:`_rollout_numpy` exactly.
    """
    x = state0[0]
    z = state0[1]
    th = state0[2]
    v = state0[3]
    out[0, 0] = x
    out[0, 1] = z
    out[0, 2] = th
    out[0, 3] = v
    for i in range(accel.shape[0]):
        h = dt[i]
        b = math.atan(lr_ratio * math.tan(steer[i]))
        thb = th + b
        x += v * math.cos(thb) * h
        z += v * math.sin(thb) * h
        th += (v * math.sin(b) / lr) * h
        v += accel[i] * h
        out[i + 1, 0] = x
        out[i + 1, 1] = z
        out[i + 1, 2] = th
        out[i + 1, 3] = v


def _rollout_numpy(state0, accel, steer, dt, lr, lr_ratio) -> np.ndarray:
    """Vectorized Euler integration — the same recursion as three prefix sums.

    Forward Euler makes each state variable depend only on quantities already
    determined earlier in the chain, so the sequential loop has a closed form::

        v  = v0  + cumsum(a·h)
        th = th0 + cumsum(v·sin(beta)/lr · h)      # v fully known by now
        x  = x0  + cumsum(v·cos(th + beta) · h)    # th fully known by now
        z  = z0  + cumsum(v·sin(th + beta) · h)

    ``np.cumsum`` is a sequential accumulation (``np.add.accumulate``), not the
    pairwise summation ``np.sum`` uses, so the addition order matches the scalar
    loop; only the vectorized transcendentals may differ by an ulp.
    """
    k = accel.shape[0]
    beta = np.arctan(lr_ratio * np.tan(steer))

    states = np.empty((k + 1, STATE_DIM), dtype=np.float64)
    if k == 0:
        states[0] = state0
        return states

    v = states[:, STATE_V]
    v[0] = state0[3]
    v[1:] = state0[3] + np.cumsum(accel * dt)

    th = states[:, STATE_THETA]
    th[0] = state0[2]
    th[1:] = state0[2] + np.cumsum(v[:k] * np.sin(beta) / lr * dt)

    thb = th[:k] + beta
    states[0, 0] = state0[0]
    states[1:, 0] = state0[0] + np.cumsum(v[:k] * np.cos(thb) * dt)
    states[0, 1] = state0[1]
    states[1:, 1] = state0[1] + np.cumsum(v[:k] * np.sin(thb) * dt)
    return states


@_njit(**_JIT)
def _rollout_segments_core(  # pragma: no cover
    nodes, accel, steer, dt, lr, lr_ratio, node_frames, frame_states, defects
):
    """Roll every shooting segment from its own node state, in one call.

    Writes ``frame_states`` (T, 4) and ``defects`` (n_nodes-1, 4). Frame ``e`` at
    a segment boundary is owned by the NEXT node, so each segment fills only
    ``[s, e)`` and reports its rolled end state as a defect against ``nodes[i+1]``.
    """
    n_nodes = node_frames.shape[0]
    for i in range(n_nodes - 1):
        s = node_frames[i]
        e = node_frames[i + 1]
        x = nodes[i, 0]
        z = nodes[i, 1]
        th = nodes[i, 2]
        v = nodes[i, 3]
        frame_states[s, 0] = x
        frame_states[s, 1] = z
        frame_states[s, 2] = th
        frame_states[s, 3] = v
        for j in range(s, e):
            h = dt[j]
            b = math.atan(lr_ratio * math.tan(steer[j]))
            thb = th + b
            x += v * math.cos(thb) * h
            z += v * math.sin(thb) * h
            th += (v * math.sin(b) / lr) * h
            v += accel[j] * h
            if j + 1 < e:
                frame_states[j + 1, 0] = x
                frame_states[j + 1, 1] = z
                frame_states[j + 1, 2] = th
                frame_states[j + 1, 3] = v
        defects[i, 0] = x - nodes[i + 1, 0]
        defects[i, 1] = z - nodes[i + 1, 1]
        defects[i, 2] = th - nodes[i + 1, 2]
        defects[i, 3] = v - nodes[i + 1, 3]
    last = node_frames[n_nodes - 1]
    frame_states[last, 0] = nodes[n_nodes - 1, 0]
    frame_states[last, 1] = nodes[n_nodes - 1, 1]
    frame_states[last, 2] = nodes[n_nodes - 1, 2]
    frame_states[last, 3] = nodes[n_nodes - 1, 3]


def rollout_segments(
    nodes: np.ndarray,
    accel: np.ndarray,
    steer: np.ndarray,
    node_frames: np.ndarray,
    dt,
    wheelbase: float,
    lr_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Multiple-shooting sweep: per-frame states plus per-segment defects.

    Equivalent to rolling each segment separately from its own shooting node,
    but done in a single call so the fitter pays no per-segment Python overhead
    — which dominates at short ``segment_len``.
    """
    nodes = np.ascontiguousarray(nodes, dtype=np.float64)
    accel = np.ascontiguousarray(accel, dtype=np.float64)
    steer = np.ascontiguousarray(steer, dtype=np.float64)
    node_frames = np.ascontiguousarray(node_frames, dtype=np.int64)
    n_steps = accel.shape[0]
    dt_arr = (
        np.full(n_steps, float(dt))
        if np.isscalar(dt)
        else np.ascontiguousarray(dt, dtype=np.float64)
    )
    lr = max(lr_ratio * float(wheelbase), 1e-6)

    T = int(node_frames[-1]) + 1
    frame_states = np.empty((T, STATE_DIM), dtype=np.float64)
    defects = np.empty((len(node_frames) - 1, STATE_DIM), dtype=np.float64)

    if HAVE_NUMBA:
        _rollout_segments_core(
            nodes, accel, steer, dt_arr, lr, float(lr_ratio),
            node_frames, frame_states, defects,
        )
        return frame_states, defects

    for i in range(len(node_frames) - 1):
        s, e = int(node_frames[i]), int(node_frames[i + 1])
        seg = _rollout_numpy(
            nodes[i], accel[s:e], steer[s:e], dt_arr[s:e], lr, float(lr_ratio)
        )
        frame_states[s:e] = seg[:-1]
        defects[i] = seg[-1] - nodes[i + 1]
    frame_states[node_frames] = nodes
    return frame_states, defects


@_njit(**_JIT)
def _sensitivity_core(  # pragma: no cover
    state0, accel, steer, dt, lr, lr_ratio, sens
):
    """Forward sensitivities of a rollout w.r.t. its initial state and controls.

    Writes ``sens`` of shape ``(K + 1, 4, 4 + 2K)``: ``sens[m]`` is
    ``d state_m / d [state0 (4), (a_0, delta_0), ..., (a_{K-1}, delta_{K-1})]``.

    Propagated forward with ``S_{m+1} = A_m @ S_m`` plus the direct control term
    ``B_m`` in its own columns, where ``A_m = df/dstate`` and ``B_m = df/du``
    evaluated along the trajectory. This is the exact derivative of the same
    Euler step the rollout takes, so it stays consistent with the residual by
    construction rather than by finite differencing it.
    """
    k = accel.shape[0]
    ncol = 4 + 2 * k
    x = state0[0]
    z = state0[1]
    th = state0[2]
    v = state0[3]
    for r in range(4):
        for c in range(ncol):
            sens[0, r, c] = 0.0
        sens[0, r, r] = 1.0

    for i in range(k):
        h = dt[i]
        td = math.tan(steer[i])
        b = math.atan(lr_ratio * td)
        # d beta / d delta = lr_ratio * sec^2(delta) / (1 + (lr_ratio*tan delta)^2)
        sec2 = 1.0 + td * td
        dbeta = lr_ratio * sec2 / (1.0 + (lr_ratio * td) * (lr_ratio * td))
        thb = th + b
        c_thb = math.cos(thb)
        s_thb = math.sin(thb)
        s_b = math.sin(b)
        c_b = math.cos(b)

        # A_i rows (only the non-trivial partials are non-zero).
        a_x_th = -v * s_thb * h
        a_x_v = c_thb * h
        a_z_th = v * c_thb * h
        a_z_v = s_thb * h
        a_th_v = s_b * h / lr

        # S_{m+1} = A_i @ S_m, done in place over the propagated columns.
        for c in range(ncol):
            s_x = sens[i, 0, c]
            s_z = sens[i, 1, c]
            s_th = sens[i, 2, c]
            s_v = sens[i, 3, c]
            sens[i + 1, 0, c] = s_x + a_x_th * s_th + a_x_v * s_v
            sens[i + 1, 1, c] = s_z + a_z_th * s_th + a_z_v * s_v
            sens[i + 1, 2, c] = s_th + a_th_v * s_v
            sens[i + 1, 3, c] = s_v

        # B_i: direct dependence on this step's own controls.
        ca = 4 + 2 * i
        cd = ca + 1
        sens[i + 1, 3, ca] += h  # dv / da
        sens[i + 1, 0, cd] += -v * s_thb * h * dbeta
        sens[i + 1, 1, cd] += v * c_thb * h * dbeta
        sens[i + 1, 2, cd] += (v * c_b * h / lr) * dbeta

        x += v * c_thb * h
        z += v * s_thb * h
        th += (v * s_b / lr) * h
        v += accel[i] * h


def rollout_sensitivity(
    state0: np.ndarray,
    accel: np.ndarray,
    steer: np.ndarray,
    dt,
    wheelbase: float,
    lr_ratio: float = 0.5,
) -> np.ndarray:
    """``d state_m / d [state0, controls]`` for every frame of a rollout.

    Returns shape ``(K + 1, 4, 4 + 2K)``. See :func:`_sensitivity_core`.
    """
    accel = np.ascontiguousarray(accel, dtype=np.float64)
    steer = np.ascontiguousarray(steer, dtype=np.float64)
    k = accel.shape[0]
    dt_arr = (
        np.full(k, float(dt)) if np.isscalar(dt) else np.ascontiguousarray(dt, dtype=np.float64)
    )
    lr = max(lr_ratio * float(wheelbase), 1e-6)
    s0 = np.ascontiguousarray(state0, dtype=np.float64).reshape(STATE_DIM)
    sens = np.zeros((k + 1, STATE_DIM, STATE_DIM + 2 * k), dtype=np.float64)
    _sensitivity_core(s0, accel, steer, dt_arr, lr, float(lr_ratio), sens)
    return sens


def heading_to_forward(theta: float) -> np.ndarray:
    """Ground-plane unit forward vector ``[cos(theta), 0, sin(theta)]`` (3D)."""
    return np.array([np.cos(theta), 0.0, np.sin(theta)], dtype=np.float64)


def wrap_angle(a):
    """Wrap angle(s) to ``(-pi, pi]``."""
    return (np.asarray(a, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi
