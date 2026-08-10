"""Per-track kinematic bicycle-model fitting for track refinement (Step 1.5).

Fits a single CoG kinematic bicycle model (see
:mod:`src.tracking.bicycle_kinematics`) to one object's noisy per-frame ground
positions and headings. The fit is an offline batch MAP smoother solved by
**multiple shooting**: the frame span is split into short segments, each with a
free shooting node (full state), and the segments are stitched with continuity
(defect) residuals. This is far more stable than single shooting, whose rollout
error compounds over long tracks.

The nonlinear least-squares problem is solved with
:func:`scipy.optimize.least_squares` (``method='trf'``). Measurement outliers
are handled with **iteratively re-weighted least squares (IRLS)** so that
position residuals get a Huber influence (mild rejection) while wrapped
orientation residuals get a heavier Cauchy influence — scipy's single global
``loss`` cannot express that mix, so the reweighting is done explicitly.

Fitting is planar: only ``x``, ``z`` and yaw are modelled. The up axis (``y``,
height) is left to the caller — height is currently kept from the tracker's
predictions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # scipy is available in env_cc3dt (the refinement env).
    from scipy.optimize import least_squares
    from scipy.sparse import coo_matrix, lil_matrix
except Exception as exc:  # pragma: no cover - surfaced only if env is wrong
    least_squares = None
    lil_matrix = None
    coo_matrix = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None

from src.tracking.bicycle_kinematics import (
    STATE_DIM,
    STATE_THETA,
    STATE_V,
    rollout,
    rollout_segments,
    rollout_sensitivity,
    slip_angle,
    wrap_angle,
)


# --------------------------------------------------------------------------- #
# Config / result containers
# --------------------------------------------------------------------------- #
@dataclass
class BicycleFitConfig:
    """Hyper-parameters for :func:`fit_track`."""

    lr_ratio: float = 0.5  # CoG-to-rear-axle fraction lr / L
    segment_len: int = 10  # frames per multiple-shooting segment
    # Measurement residual normalizers (meters / radians). Residuals are divided
    # by these before the robust weighting, so they set the outlier transition.
    pos_scale: float = 0.5
    yaw_scale: float = 0.15
    # Robust influence transition points, in *normalized* residual units (~1).
    huber_delta: float = 1.345  # position (mild)
    cauchy_c: float = 2.3849  # yaw (heavier tails)
    irls_iters: int = 3  # outer re-weighting iterations
    # Control-smoothness priors (larger = smoother controls / trajectory). These
    # dominate the regularization: with per-step controls the trajectory is
    # heavily over-parameterized, so smoothness is what rejects position noise.
    lambda_accel: float = 4.0  # penalize accel changes (jerk)
    lambda_steer: float = 6.0  # penalize steer changes
    lambda_accel_mag: float = 0.02  # small pull of accel toward 0
    lambda_steer_mag: float = 0.05  # small pull of steer toward 0
    # Continuity (defect) weight for stitching segments. Large = hard constraint.
    defect_weight: float = 100.0
    max_steer: float = 0.7  # |delta| bound (~40 deg) for stability
    # Acceleration box bounds (m/s^2) enforced by the trf solver, so one bad
    # observation cannot drive a non-physical speed runaway.
    accel_min: float = -6.0
    accel_max: float = 4.0
    # Per-solve function-eval cap. ``None`` lets scipy use ``100 * n_params``;
    # a finite-difference Jacobian needs ~``n_params`` evals per iteration, so a
    # small cap silently under-converges.
    solver_max_nfev: int | None = None
    # Final single-shooting polish: after the multiple-shooting solve, refine
    # ``(state0, controls)`` with a *pure single-rollout* forward model so the
    # baked trajectory (the one continuous rollout the simulator reproduces from
    # the persisted params) directly fits the observations, instead of drifting
    # from accumulated segment-continuity defects. Warm-started from the
    # multiple-shooting solution, so it converges in a few iterations. Disable to
    # return the raw multiple-shooting states.
    polish_single_shooting: bool = True
    polish_max_nfev: int = 400
    # Use the analytic Jacobian (exact derivatives of the same Euler step the
    # residual integrates) instead of scipy's finite differences. The FD path
    # costs one extra residual evaluation per Jacobian column group on every
    # solver iteration; the analytic one costs roughly a single evaluation and
    # is more accurate. Set False to fall back to finite differences (useful
    # for A/B-ing fit quality).
    analytic_jacobian: bool = True


@dataclass
class BicycleFitResult:
    """Output of :func:`fit_track`.

    ``states`` are the fitted per-frame states ``[x, z, theta, v]`` over the full
    span (length ``n_steps + 1``); gaps are filled implicitly by the rollout. The
    fitted heading is used on every frame — there is no speed threshold below
    which the caller reverts to the measured box yaw.
    """

    success: bool
    states: np.ndarray  # (T, 4)
    accel: np.ndarray  # (T-1,)
    steer: np.ndarray  # (T-1,)
    node_frames: np.ndarray  # shooting-node offsets
    cost: float
    pos_rmse: float
    yaw_rmse: float
    n_obs: int
    message: str = ""

    @property
    def positions(self) -> np.ndarray:
        """Fitted ground positions ``(T, 2)`` = ``[x, z]``."""
        return self.states[:, :2]

    @property
    def yaws(self) -> np.ndarray:
        """Fitted headings ``(T,)`` in radians."""
        return self.states[:, STATE_THETA]

    @property
    def speeds(self) -> np.ndarray:
        """Fitted speeds ``(T,)`` in m/s."""
        return self.states[:, STATE_V]


# --------------------------------------------------------------------------- #
# Robust weights (IRLS influence functions)
# --------------------------------------------------------------------------- #
def _huber_weight(r: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-9))


def _cauchy_weight(r: np.ndarray, c: float) -> np.ndarray:
    return 1.0 / (1.0 + (r / c) ** 2)


# --------------------------------------------------------------------------- #
# Initialization
# --------------------------------------------------------------------------- #
def _initial_track_states(
    frames: np.ndarray,
    positions: np.ndarray,
    yaws: np.ndarray,
    n_steps: int,
    dt: float,
) -> np.ndarray:
    """Dense per-frame state initialization from sparse observations.

    Interpolates observed positions across the full span, estimates speed and
    heading from finite differences, and prefers the measured yaw where motion
    is too small to trust the tangent.
    """
    T = n_steps + 1
    span = np.arange(T)

    # Interpolate x, z across the full span.
    px = np.interp(span, frames, positions[:, 0])
    pz = np.interp(span, frames, positions[:, 1])

    # Speed from position finite differences (central where possible).
    dx = np.gradient(px)
    dz = np.gradient(pz)
    step_dist = np.hypot(dx, dz)
    v = step_dist / max(dt, 1e-6)

    # Heading: unwrap measured yaw and interpolate; fall back to tangent when the
    # object is clearly moving.
    yaw_unwrapped = np.unwrap(yaws) if yaws.size else np.zeros_like(frames, dtype=float)
    th = np.interp(span, frames, yaw_unwrapped) if yaws.size else np.zeros(T)
    tangent = np.arctan2(dz, dx)
    moving = step_dist > (0.2 * max(dt, 1e-6))
    # Align tangent to the current heading branch before blending.
    tangent = th + wrap_angle(tangent - th)
    th = np.where(moving, tangent, th)

    states = np.empty((T, STATE_DIM), dtype=np.float64)
    states[:, 0] = px
    states[:, 1] = pz
    states[:, STATE_THETA] = th
    states[:, STATE_V] = v
    return states


def _initial_controls(
    states: np.ndarray,
    dt: float,
    wheelbase: float,
    lr_ratio: float,
    max_steer: float,
) -> np.ndarray:
    """Seed controls ``[a, delta]`` from the initialized state sequence."""
    n_steps = states.shape[0] - 1
    v = states[:, STATE_V]
    th = states[:, STATE_THETA]

    accel = (v[1:] - v[:-1]) / max(dt, 1e-6)

    lr = max(lr_ratio * wheelbase, 1e-6)
    yaw_rate = wrap_angle(th[1:] - th[:-1]) / max(dt, 1e-6)
    v_mid = np.maximum(0.5 * (v[1:] + v[:-1]), 0.5)
    sin_beta = np.clip(yaw_rate * lr / v_mid, -0.99, 0.99)
    beta = np.arcsin(sin_beta)
    steer = np.arctan(np.tan(beta) / max(lr_ratio, 1e-6))
    steer = np.clip(steer, -max_steer, max_steer)

    return np.stack([accel, steer], axis=1)  # (n_steps, 2)


# --------------------------------------------------------------------------- #
# Multiple-shooting residual assembly
# --------------------------------------------------------------------------- #
def _segment_boundaries(n_steps: int, segment_len: int) -> np.ndarray:
    """Shooting-node frame offsets ``[0, S, 2S, ..., n_steps]``."""
    seg = max(int(segment_len), 1)
    nodes = list(range(0, n_steps, seg))
    if nodes[-1] != n_steps:
        nodes.append(n_steps)
    return np.asarray(nodes, dtype=np.int64)


def _rollout_all_segments(
    nodes: np.ndarray,
    accel: np.ndarray,
    steer: np.ndarray,
    node_frames: np.ndarray,
    dt: float,
    wheelbase: float,
    lr_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll each segment from its node; return per-frame states and defects.

    Returns:
        frame_states: ``(T, 4)`` state at every frame. Node frames use the node
            value directly; interior frames use the owning segment's rollout.
        defects: ``(n_interior_nodes, 4)`` mismatch between a segment's rolled
            end state and the next node's state.
    """
    return rollout_segments(nodes, accel, steer, node_frames, dt, wheelbase, lr_ratio)


def _make_residual_fn(
    obs_frames: np.ndarray,
    obs_pos: np.ndarray,
    obs_yaw: np.ndarray,
    n_steps: int,
    node_frames: np.ndarray,
    dt: float,
    wheelbase: float,
    cfg: BicycleFitConfig,
    yaw_w: np.ndarray,
):
    """Build the residual closure and helpers for a fixed problem structure."""
    n_nodes = len(node_frames)
    n_state = n_nodes * STATE_DIM

    def unpack(p: np.ndarray):
        nodes = p[:n_state].reshape(n_nodes, STATE_DIM)
        ctrl = p[n_state:].reshape(n_steps, 2)
        return nodes, ctrl[:, 0], ctrl[:, 1]

    def measurement_residuals(p: np.ndarray):
        """Raw (unweighted) normalized measurement residuals."""
        nodes, accel, steer = unpack(p)
        frame_states, _ = _rollout_all_segments(
            nodes, accel, steer, node_frames, dt, wheelbase, cfg.lr_ratio
        )
        pred = frame_states[obs_frames]
        pos_res = (pred[:, :2] - obs_pos) / cfg.pos_scale  # (N, 2)
        yaw_res = wrap_angle(pred[:, STATE_THETA] - obs_yaw) / cfg.yaw_scale  # (N,)
        return pos_res, yaw_res

    def residuals(p: np.ndarray, w_pos: np.ndarray, w_yaw: np.ndarray):
        nodes, accel, steer = unpack(p)
        frame_states, defects = _rollout_all_segments(
            nodes, accel, steer, node_frames, dt, wheelbase, cfg.lr_ratio
        )

        pred = frame_states[obs_frames]
        pos_res = (pred[:, :2] - obs_pos) / cfg.pos_scale
        yaw_res = wrap_angle(pred[:, STATE_THETA] - obs_yaw) / cfg.yaw_scale
        pos_res = pos_res * np.sqrt(w_pos)[:, None]
        yaw_res = yaw_res * np.sqrt(w_yaw) * yaw_w

        # Continuity: wrap the heading component of each defect.
        defects = defects.copy()
        defects[:, STATE_THETA] = wrap_angle(defects[:, STATE_THETA])
        defect_res = (cfg.defect_weight * defects).ravel()

        # Control smoothness + magnitude priors.
        d_accel = cfg.lambda_accel * np.diff(accel)
        d_steer = cfg.lambda_steer * np.diff(steer)
        mag_accel = cfg.lambda_accel_mag * accel
        mag_steer = cfg.lambda_steer_mag * steer

        return np.concatenate(
            [
                pos_res.ravel(),
                yaw_res,
                defect_res,
                d_accel,
                d_steer,
                mag_accel,
                mag_steer,
            ]
        )

    return residuals, measurement_residuals, unpack


def _make_jac_fn(
    obs_frames: np.ndarray,
    n_steps: int,
    node_frames: np.ndarray,
    dt: float,
    wheelbase: float,
    cfg: BicycleFitConfig,
    yaw_w: np.ndarray,
):
    """Analytic Jacobian of the multiple-shooting residual vector.

    Replaces the finite-difference Jacobian, which costs one extra residual
    evaluation per Jacobian *column group* (tens of them) on every solver
    iteration. Derivatives come from :func:`rollout_sensitivity`, the exact
    derivative of the same Euler step the residual integrates, so the two cannot
    drift apart.

    Row layout matches ``residuals``: position (2 per obs, interleaved x/z), yaw
    (1 per obs), defects (4 per interior node), then the control-smoothness and
    magnitude priors. Columns are ``[nodes..., (accel, steer) * n_steps]``.

    Within one segment every frame shares the same column set — the owning node
    plus that segment's controls — because the sensitivity of a frame to controls
    that come *after* it is structurally zero. Those zeros are stored explicitly,
    which keeps the assembly rectangular and fully vectorized at a modest cost in
    stored entries.
    """
    n_nodes = len(node_frames)
    n_state = n_nodes * STATE_DIM
    n_params = n_state + n_steps * 2
    n_obs = len(obs_frames)
    yaw_base = 2 * n_obs
    defect_base = 3 * n_obs
    smooth_base = defect_base + STATE_DIM * (n_nodes - 1)
    n_res = smooth_base + 2 * (n_steps - 1) + 2 * n_steps

    last_node_frame = int(node_frames[-1])
    # Map each observation to its owning segment; the final node frame is owned
    # by no segment (its state IS the last node), so it is flagged with -1.
    seg_of_obs = np.empty(n_obs, dtype=np.int64)
    for n, f in enumerate(obs_frames):
        f = int(f)
        if f == last_node_frame:
            seg_of_obs[n] = -1
        else:
            seg_of_obs[n] = max(0, int(np.searchsorted(node_frames, f, side="right")) - 1)

    # Constant blocks: control smoothness and magnitude priors.
    c_rows, c_cols, c_data = [], [], []

    def _accel_col(k):
        return n_state + 2 * k

    def _steer_col(k):
        return n_state + 2 * k + 1

    ks = np.arange(n_steps - 1)
    for base, col_fn, lam in (
        (smooth_base, _accel_col, cfg.lambda_accel),
        (smooth_base + (n_steps - 1), _steer_col, cfg.lambda_steer),
    ):
        c_rows.append(base + ks); c_cols.append(col_fn(ks)); c_data.append(np.full(n_steps - 1, -lam))
        c_rows.append(base + ks); c_cols.append(col_fn(ks + 1)); c_data.append(np.full(n_steps - 1, lam))
    ka = np.arange(n_steps)
    mag_base = smooth_base + 2 * (n_steps - 1)
    for base, col_fn, lam in (
        (mag_base, _accel_col, cfg.lambda_accel_mag),
        (mag_base + n_steps, _steer_col, cfg.lambda_steer_mag),
    ):
        c_rows.append(base + ka); c_cols.append(col_fn(ka)); c_data.append(np.full(n_steps, lam))
    const_rows = np.concatenate(c_rows); const_cols = np.concatenate(c_cols)
    const_data = np.concatenate(c_data)

    def jac(p: np.ndarray, w_pos: np.ndarray, w_yaw: np.ndarray):
        nodes = p[:n_state].reshape(n_nodes, STATE_DIM)
        ctrl = p[n_state:].reshape(n_steps, 2)
        accel, steer = ctrl[:, 0], ctrl[:, 1]
        sq_pos = np.sqrt(w_pos)
        sq_yaw = np.sqrt(w_yaw) * yaw_w

        rows, cols, data = [const_rows], [const_cols], [const_data]

        for i in range(n_nodes - 1):
            s, e = int(node_frames[i]), int(node_frames[i + 1])
            L = e - s
            sens = rollout_sensitivity(
                nodes[i], accel[s:e], steer[s:e], dt, wheelbase, cfg.lr_ratio
            )  # (L+1, 4, 4+2L)
            seg_cols = np.concatenate([
                np.arange(STATE_DIM * i, STATE_DIM * i + STATE_DIM),
                n_state + 2 * s + np.arange(2 * L),
            ])

            sel = np.flatnonzero(seg_of_obs == i)
            if sel.size:
                m = obs_frames[sel] - s                       # local frame index
                blk = sens[m]                                  # (M, 4, 4+2L)
                px = blk[:, 0, :] * (sq_pos[sel] / cfg.pos_scale)[:, None]
                pz = blk[:, 1, :] * (sq_pos[sel] / cfg.pos_scale)[:, None]
                py = blk[:, 2, :] * (sq_yaw[sel] / cfg.yaw_scale)[:, None]
                cc = np.broadcast_to(seg_cols, (sel.size, seg_cols.size))
                for r_off, vals in ((0, px), (1, pz), (None, py)):
                    r = (yaw_base + sel) if r_off is None else (2 * sel + r_off)
                    rows.append(np.repeat(r, seg_cols.size))
                    cols.append(cc.ravel())
                    data.append(vals.ravel())

            # Defect: rolled end state minus the next node.
            d_rows = defect_base + STATE_DIM * i + np.arange(STATE_DIM)
            end = sens[L] * cfg.defect_weight                  # (4, 4+2L)
            rows.append(np.repeat(d_rows, seg_cols.size))
            cols.append(np.broadcast_to(seg_cols, (STATE_DIM, seg_cols.size)).ravel())
            data.append(end.ravel())
            nxt = np.arange(STATE_DIM * (i + 1), STATE_DIM * (i + 1) + STATE_DIM)
            rows.append(d_rows); cols.append(nxt)
            data.append(np.full(STATE_DIM, -cfg.defect_weight))

        # Observations landing on the final node frame read that node directly.
        sel = np.flatnonzero(seg_of_obs == -1)
        if sel.size:
            base_col = STATE_DIM * (n_nodes - 1)
            for n in sel:
                w_p = sq_pos[n] / cfg.pos_scale
                rows.append(np.array([2 * n, 2 * n + 1, yaw_base + n]))
                cols.append(np.array([base_col, base_col + 1, base_col + STATE_THETA]))
                data.append(np.array([w_p, w_p, sq_yaw[n] / cfg.yaw_scale]))

        return coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_res, n_params),
        ).tocsr()

    return jac


def _build_jac_sparsity(
    obs_frames: np.ndarray, n_steps: int, node_frames: np.ndarray
):
    """Boolean Jacobian sparsity for the multiple-shooting residual vector.

    Multiple shooting decouples segments (each frame's predicted state depends
    only on its owning shooting node and that segment's controls; segments are
    linked solely through the defect residuals). The resulting Jacobian is very
    sparse, so handing this pattern to ``least_squares`` lets it compute the
    finite-difference Jacobian with far fewer function evaluations.

    The pattern must be a *superset* of the true nonzeros — it is, by
    construction, exactly the dependency structure of :func:`residuals`.
    """
    n_nodes = len(node_frames)
    n_state = n_nodes * STATE_DIM
    n_params = n_state + n_steps * 2
    n_obs = len(obs_frames)
    n_res = (
        2 * n_obs  # position
        + n_obs  # yaw
        + STATE_DIM * (n_nodes - 1)  # defects
        + (n_steps - 1)  # d_accel
        + (n_steps - 1)  # d_steer
        + n_steps  # accel magnitude
        + n_steps  # steer magnitude
    )
    S = lil_matrix((n_res, n_params), dtype=bool)

    node_set = {int(nf): i for i, nf in enumerate(node_frames)}

    def node_cols(i: int):
        return [STATE_DIM * i + k for k in range(STATE_DIM)]

    def accel_col(k: int):
        return n_state + 2 * k

    def steer_col(k: int):
        return n_state + 2 * k + 1

    def frame_deps(f: int):
        """Params a frame's predicted state depends on: (node_index, steps)."""
        f = int(f)
        if f in node_set:
            return node_set[f], range(0)
        i = int(np.searchsorted(node_frames, f)) - 1
        i = max(0, min(i, n_nodes - 2))
        return i, range(int(node_frames[i]), f)

    # Measurement residuals (position rows interleaved [x, z], then yaw block).
    yaw_base = 2 * n_obs
    for n, f in enumerate(obs_frames):
        ni, steps = frame_deps(f)
        cols = set(node_cols(ni))
        for k in steps:
            cols.add(accel_col(k))
            cols.add(steer_col(k))
        for c in cols:
            S[2 * n, c] = True
            S[2 * n + 1, c] = True
            S[yaw_base + n, c] = True

    # Defect residuals: node i, node i+1, and all controls in segment i.
    base = 3 * n_obs
    for i in range(n_nodes - 1):
        s, e = int(node_frames[i]), int(node_frames[i + 1])
        cols = set(node_cols(i) + node_cols(i + 1))
        for k in range(s, e):
            cols.add(accel_col(k))
            cols.add(steer_col(k))
        for r in range(STATE_DIM):
            for c in cols:
                S[base + STATE_DIM * i + r, c] = True

    # Control smoothness / magnitude priors.
    base += STATE_DIM * (n_nodes - 1)
    for k in range(n_steps - 1):
        S[base + k, accel_col(k)] = True
        S[base + k, accel_col(k + 1)] = True
    base += n_steps - 1
    for k in range(n_steps - 1):
        S[base + k, steer_col(k)] = True
        S[base + k, steer_col(k + 1)] = True
    base += n_steps - 1
    for k in range(n_steps):
        S[base + k, accel_col(k)] = True
    base += n_steps
    for k in range(n_steps):
        S[base + k, steer_col(k)] = True

    return S.tocsr()


# --------------------------------------------------------------------------- #
# Single-shooting polish
# --------------------------------------------------------------------------- #
def _make_ss_residual_fn(
    obs_frames: np.ndarray,
    obs_pos: np.ndarray,
    obs_yaw: np.ndarray,
    n_steps: int,
    dt: float,
    wheelbase: float,
    cfg: BicycleFitConfig,
    yaw_w: np.ndarray,
):
    """Residual closure for the *single-shooting* polish.

    Unlike the multiple-shooting residual, the whole span is one continuous
    rollout from ``state0``; there are no shooting nodes and no defect residuals.
    The measurement and control-smoothness terms are identical, so the polish
    minimizes exactly the trajectory that gets baked/persisted.
    """
    lr = cfg.lr_ratio

    def unpack(p: np.ndarray):
        state0 = p[:STATE_DIM]
        ctrl = p[STATE_DIM:].reshape(n_steps, 2)
        return state0, ctrl[:, 0], ctrl[:, 1]

    def measurement_residuals(p: np.ndarray):
        state0, accel, steer = unpack(p)
        states = rollout(state0, accel, steer, dt, wheelbase, lr)
        pred = states[obs_frames]
        pos_res = (pred[:, :2] - obs_pos) / cfg.pos_scale
        yaw_res = wrap_angle(pred[:, STATE_THETA] - obs_yaw) / cfg.yaw_scale
        return pos_res, yaw_res

    def residuals(p: np.ndarray, w_pos: np.ndarray, w_yaw: np.ndarray):
        state0, accel, steer = unpack(p)
        states = rollout(state0, accel, steer, dt, wheelbase, lr)
        pred = states[obs_frames]
        pos_res = (pred[:, :2] - obs_pos) / cfg.pos_scale
        yaw_res = wrap_angle(pred[:, STATE_THETA] - obs_yaw) / cfg.yaw_scale
        pos_res = pos_res * np.sqrt(w_pos)[:, None]
        yaw_res = yaw_res * np.sqrt(w_yaw) * yaw_w

        d_accel = cfg.lambda_accel * np.diff(accel)
        d_steer = cfg.lambda_steer * np.diff(steer)
        mag_accel = cfg.lambda_accel_mag * accel
        mag_steer = cfg.lambda_steer_mag * steer

        return np.concatenate(
            [pos_res.ravel(), yaw_res, d_accel, d_steer, mag_accel, mag_steer]
        )

    return residuals, measurement_residuals, unpack


def _make_ss_jac_fn(
    obs_frames: np.ndarray,
    n_steps: int,
    dt: float,
    wheelbase: float,
    cfg: BicycleFitConfig,
    yaw_w: np.ndarray,
):
    """Analytic Jacobian of the single-shooting polish residual.

    Simpler than the multiple-shooting case: the whole span is one rollout, so
    :func:`rollout_sensitivity` returns derivatives whose column layout already
    matches the parameter vector ``[state0, (accel, steer) * n_steps]``.
    """
    n_obs = len(obs_frames)
    n_params = STATE_DIM + 2 * n_steps
    yaw_base = 2 * n_obs
    smooth_base = 3 * n_obs
    n_res = smooth_base + 2 * (n_steps - 1) + 2 * n_steps

    def _accel_col(k):
        return STATE_DIM + 2 * k

    def _steer_col(k):
        return STATE_DIM + 2 * k + 1

    c_rows, c_cols, c_data = [], [], []
    ks = np.arange(n_steps - 1)
    for base, col_fn, lam in (
        (smooth_base, _accel_col, cfg.lambda_accel),
        (smooth_base + (n_steps - 1), _steer_col, cfg.lambda_steer),
    ):
        c_rows.append(base + ks); c_cols.append(col_fn(ks)); c_data.append(np.full(n_steps - 1, -lam))
        c_rows.append(base + ks); c_cols.append(col_fn(ks + 1)); c_data.append(np.full(n_steps - 1, lam))
    ka = np.arange(n_steps)
    mag_base = smooth_base + 2 * (n_steps - 1)
    for base, col_fn, lam in (
        (mag_base, _accel_col, cfg.lambda_accel_mag),
        (mag_base + n_steps, _steer_col, cfg.lambda_steer_mag),
    ):
        c_rows.append(base + ka); c_cols.append(col_fn(ka)); c_data.append(np.full(n_steps, lam))
    const_rows = np.concatenate(c_rows); const_cols = np.concatenate(c_cols)
    const_data = np.concatenate(c_data)
    all_cols = np.arange(n_params)

    def jac(p: np.ndarray, w_pos: np.ndarray, w_yaw: np.ndarray):
        state0 = p[:STATE_DIM]
        ctrl = p[STATE_DIM:].reshape(n_steps, 2)
        sens = rollout_sensitivity(
            state0, ctrl[:, 0], ctrl[:, 1], dt, wheelbase, cfg.lr_ratio
        )  # (T, 4, n_params)
        blk = sens[obs_frames]
        sq_pos = (np.sqrt(w_pos) / cfg.pos_scale)[:, None]
        sq_yaw = (np.sqrt(w_yaw) * yaw_w / cfg.yaw_scale)[:, None]

        cc = np.broadcast_to(all_cols, (n_obs, n_params)).ravel()
        rows = [const_rows]; cols = [const_cols]; data = [const_data]
        n = np.arange(n_obs)
        for r, vals in (
            (2 * n, blk[:, 0, :] * sq_pos),
            (2 * n + 1, blk[:, 1, :] * sq_pos),
            (yaw_base + n, blk[:, 2, :] * sq_yaw),
        ):
            rows.append(np.repeat(r, n_params)); cols.append(cc); data.append(vals.ravel())

        return coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_res, n_params),
        ).tocsr()

    return jac


def _build_ss_jac_sparsity(obs_frames: np.ndarray, n_steps: int):
    """Boolean Jacobian sparsity for the single-shooting residual vector.

    In a single rollout the predicted state at frame ``f`` depends on ``state0``
    and every control step ``0 .. f-1`` (lower-triangular in the control block).
    The control-smoothness / magnitude priors are banded. Supplying this pattern
    lets ``least_squares`` build the finite-difference Jacobian with far fewer
    function evaluations on long tracks.
    """
    n_params = STATE_DIM + n_steps * 2
    n_obs = len(obs_frames)
    n_res = (
        2 * n_obs  # position
        + n_obs  # yaw
        + (n_steps - 1)  # d_accel
        + (n_steps - 1)  # d_steer
        + n_steps  # accel magnitude
        + n_steps  # steer magnitude
    )
    S = lil_matrix((n_res, n_params), dtype=bool)

    def accel_col(k: int):
        return STATE_DIM + 2 * k

    def steer_col(k: int):
        return STATE_DIM + 2 * k + 1

    yaw_base = 2 * n_obs
    for n, f in enumerate(obs_frames):
        f = int(f)
        cols = list(range(STATE_DIM))  # state0 (all components)
        for k in range(f):
            cols.append(accel_col(k))
            cols.append(steer_col(k))
        for c in cols:
            S[2 * n, c] = True
            S[2 * n + 1, c] = True
            S[yaw_base + n, c] = True

    base = 3 * n_obs
    for k in range(n_steps - 1):
        S[base + k, accel_col(k)] = True
        S[base + k, accel_col(k + 1)] = True
    base += n_steps - 1
    for k in range(n_steps - 1):
        S[base + k, steer_col(k)] = True
        S[base + k, steer_col(k + 1)] = True
    base += n_steps - 1
    for k in range(n_steps):
        S[base + k, accel_col(k)] = True
    base += n_steps
    for k in range(n_steps):
        S[base + k, steer_col(k)] = True

    return S.tocsr()


def _param_bounds(n_state: int, n_steps: int, cfg: BicycleFitConfig):
    """Box bounds for a ``[state..., (accel, steer) * n_steps]`` parameter vector.

    States are left unbounded; controls are bounded so the trf solver keeps
    acceleration and steering physical without any post-hoc clipping.
    """
    n = n_state + 2 * n_steps
    lb = np.full(n, -np.inf)
    ub = np.full(n, np.inf)
    lb[n_state::2] = cfg.accel_min
    ub[n_state::2] = cfg.accel_max
    lb[n_state + 1::2] = -cfg.max_steer
    ub[n_state + 1::2] = cfg.max_steer
    return lb, ub


def _single_shooting_polish(
    state0: np.ndarray,
    accel: np.ndarray,
    steer: np.ndarray,
    frames: np.ndarray,
    positions: np.ndarray,
    yaws: np.ndarray,
    n_steps: int,
    dt: float,
    wheelbase: float,
    cfg: BicycleFitConfig,
    yaw_w: np.ndarray,
):
    """Refine ``(state0, accel, steer)`` with an IRLS single-shooting solve.

    Warm-started from the converged multiple-shooting controls, this directly
    minimizes the single-rollout fit to the observations (the trajectory that is
    actually baked and persisted). Returns the polished ``(state0, accel, steer,
    result)``; ``result`` is the last scipy result (``None`` if the solve
    raised).
    """
    residuals, measurement_residuals, unpack = _make_ss_residual_fn(
        frames, positions, yaws, n_steps, dt, wheelbase, cfg, yaw_w
    )
    if cfg.analytic_jacobian:
        jac_kwargs = {"jac": _make_ss_jac_fn(frames, n_steps, dt, wheelbase, cfg, yaw_w)}
    else:
        jac_kwargs = {"jac_sparsity": _build_ss_jac_sparsity(frames, n_steps)}
    lb, ub = _param_bounds(STATE_DIM, n_steps, cfg)

    n_obs = int(len(frames))
    w_pos = np.ones(n_obs)
    w_yaw = np.ones(n_obs)
    p = np.clip(
        np.concatenate(
            [np.asarray(state0, dtype=np.float64), np.stack([accel, steer], axis=1).ravel()]
        ),
        lb, ub,
    )
    result = None
    for _ in range(max(cfg.irls_iters, 1)):
        try:
            result = least_squares(
                residuals,
                p,
                args=(w_pos, w_yaw),
                method="trf",
                loss="linear",
                tr_solver="lsmr",
                bounds=(lb, ub),
                max_nfev=cfg.polish_max_nfev,
                **jac_kwargs,
            )
        except Exception:  # pragma: no cover - solver robustness
            break
        p = result.x
        pos_res, yaw_res = measurement_residuals(p)
        pos_mag = np.linalg.norm(pos_res, axis=1)
        w_pos = _huber_weight(pos_mag, cfg.huber_delta)
        w_yaw = _cauchy_weight(yaw_res, cfg.cauchy_c)

    s0, a, d = unpack(p)
    return s0.copy(), a.copy(), d.copy(), result


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def fit_track(
    frames: np.ndarray,
    positions: np.ndarray,
    yaws: np.ndarray,
    n_steps: int,
    dt: float,
    wheelbase: float,
    cfg: BicycleFitConfig | None = None,
    yaw_weights: np.ndarray | None = None,
) -> BicycleFitResult:
    """Fit a CoG bicycle model to one track's observations.

    Args:
        frames: Observed frame offsets within the span, sorted, in ``[0, n_steps]``.
            Shape ``(N,)``.
        positions: Observed ground positions ``[x, z]``. Shape ``(N, 2)``.
        yaws: Observed headings (radians). Shape ``(N,)``.
        n_steps: Number of integration steps; the span covers ``n_steps + 1``
            frames (offset ``0`` .. ``n_steps``).
        dt: Seconds per step (constant).
        wheelbase: Wheelbase ``L`` in meters (``alpha * max(size_x, size_y)``).
        cfg: Fitting hyper-parameters.

    Returns:
        A :class:`BicycleFitResult`. On failure ``success`` is ``False`` and the
        state falls back to the interpolated initialization.
    """
    if least_squares is None:  # pragma: no cover
        raise RuntimeError(f"scipy is required for bicycle fitting: {_SCIPY_IMPORT_ERROR}")

    cfg = cfg or BicycleFitConfig()
    frames = np.asarray(frames, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    yaws = np.asarray(yaws, dtype=np.float64).reshape(-1)
    n_obs = int(frames.size)
    # Per-observation orientation weight; 0 drops an untrustworthy heading.
    if yaw_weights is None:
        yaw_weights = np.ones(n_obs)
    else:
        yaw_weights = np.asarray(yaw_weights, dtype=np.float64).reshape(-1)

    init_states = _initial_track_states(frames, positions, yaws, n_steps, dt)

    # Degenerate spans: nothing to optimize, return the initialization.
    if n_steps < 1 or n_obs < 2:
        return BicycleFitResult(
            success=False,
            states=init_states,
            accel=np.zeros(max(n_steps, 0)),
            steer=np.zeros(max(n_steps, 0)),
            node_frames=np.array([0], dtype=np.int64),
            cost=float("nan"),
            pos_rmse=float("nan"),
            yaw_rmse=float("nan"),
            n_obs=n_obs,
            message="span too short to fit",
        )

    node_frames = _segment_boundaries(n_steps, cfg.segment_len)
    init_ctrl = _initial_controls(
        init_states, dt, wheelbase, cfg.lr_ratio, cfg.max_steer
    )
    p0 = np.concatenate([init_states[node_frames].ravel(), init_ctrl.ravel()])

    residuals, measurement_residuals, unpack = _make_residual_fn(
        frames, positions, yaws, n_steps, node_frames, dt, wheelbase, cfg, yaw_weights
    )
    if cfg.analytic_jacobian:
        jac_fn = _make_jac_fn(
            frames, n_steps, node_frames, dt, wheelbase, cfg, yaw_weights
        )
        jac_kwargs = {"jac": jac_fn}
    else:
        jac_kwargs = {"jac_sparsity": _build_jac_sparsity(frames, n_steps, node_frames)}
    lb, ub = _param_bounds(len(node_frames) * STATE_DIM, n_steps, cfg)
    p0 = np.clip(p0, lb, ub)

    # IRLS: alternate a least-squares solve with a robust re-weighting of the
    # measurement residuals (Huber for position, Cauchy for yaw).
    w_pos = np.ones(n_obs)
    w_yaw = np.ones(n_obs)
    p = p0
    result = None
    message = ""
    for _ in range(max(cfg.irls_iters, 1)):
        try:
            result = least_squares(
                residuals,
                p,
                args=(w_pos, w_yaw),
                method="trf",
                loss="linear",
                tr_solver="lsmr",
                bounds=(lb, ub),
                max_nfev=cfg.solver_max_nfev,
                **jac_kwargs,
            )
        except Exception as exc:  # pragma: no cover - solver robustness
            message = f"least_squares failed: {exc}"
            break
        p = result.x
        pos_res, yaw_res = measurement_residuals(p)
        pos_mag = np.linalg.norm(pos_res, axis=1)
        w_pos = _huber_weight(pos_mag, cfg.huber_delta)
        w_yaw = _cauchy_weight(yaw_res, cfg.cauchy_c)

    success = result is not None and bool(getattr(result, "success", False))
    if result is not None:
        message = message or str(result.message)

    nodes, accel, steer = unpack(p)
    frame_states, _ = _rollout_all_segments(
        nodes, accel, steer, node_frames, dt, wheelbase, cfg.lr_ratio
    )

    # Single-shooting polish: multiple shooting stabilizes the optimization but
    # its per-segment states carry continuity-defect jumps that make the SINGLE
    # rollout from (state0, controls) — what actually gets baked and persisted —
    # drift off the observations. Refine the controls with a pure single-rollout
    # model, warm-started from the multiple-shooting solution, so the baked
    # trajectory directly fits the data. The polish minimizes a robust
    # (IRLS-weighted) objective, so on already-good tracks it can occasionally
    # land at a marginally worse *raw* RMSE; keep it only when it actually
    # improves the baked single-rollout error, making the polish a strict
    # per-track improvement.
    if cfg.polish_single_shooting and result is not None:
        # Baseline: the single rollout the caller would bake from the
        # multiple-shooting controls (not the segmented ``frame_states``).
        ms_states = rollout(nodes[0], accel, steer, dt, wheelbase, cfg.lr_ratio)

        def _baked_pos_rmse(states: np.ndarray) -> float:
            err = states[frames][:, :2] - positions
            return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))

        s0_p, accel_p, steer_p, ss_result = _single_shooting_polish(
            frame_states[0], accel, steer, frames, positions, yaws,
            n_steps, dt, wheelbase, cfg, yaw_weights,
        )
        if ss_result is not None:
            pol_states = rollout(s0_p, accel_p, steer_p, dt, wheelbase, cfg.lr_ratio)
            if _baked_pos_rmse(pol_states) <= _baked_pos_rmse(ms_states):
                accel, steer, frame_states = accel_p, steer_p, pol_states
                success = bool(getattr(ss_result, "success", False))
                message = str(ss_result.message)
            else:
                # Polish did not help: bake the multiple-shooting single rollout.
                frame_states = ms_states
        else:
            frame_states = ms_states
        # A single continuous rollout has no interior shooting nodes.
        node_frames = np.array([0, n_steps], dtype=np.int64)

    # Report RMSE against the returned (possibly polished) per-frame states, so
    # ``pos_rmse`` reflects exactly the baked trajectory when polishing is on.
    pred = frame_states[frames]
    pos_err = pred[:, :2] - positions
    pos_rmse = float(np.sqrt(np.mean(np.sum(pos_err**2, axis=1)))) if n_obs else float("nan")
    yaw_err = wrap_angle(pred[:, STATE_THETA] - yaws)
    yaw_rmse = float(np.sqrt(np.mean(yaw_err**2))) if n_obs else float("nan")

    return BicycleFitResult(
        success=success,
        states=frame_states,
        accel=accel,
        steer=steer,
        node_frames=node_frames,
        cost=float(result.cost) if result is not None else float("nan"),
        pos_rmse=pos_rmse,
        yaw_rmse=yaw_rmse,
        n_obs=n_obs,
        message=message,
    )
