from __future__ import annotations

import hashlib
import json
import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import torch
import viser
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
GSPLAT_TRAINING_DIR = REPO_ROOT / "src" / "gsplat_training"
if str(GSPLAT_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(GSPLAT_TRAINING_DIR))
# Rigid extrapolation reaches back out to ``src.tracking.bicycle_kinematics``
# (via dynamic.rigid_tracks.bicycle_pose_at_frame), which resolves only with the
# repo root importable. Running as a script puts scripts/ on sys.path, not the
# root, so without this any frame past the baked span raises ModuleNotFoundError.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from render_standalone import (  # noqa: E402
    StandaloneRenderer,
    load_all_cameras,
    load_rigid_state_from_checkpoint,
    load_splats_from_checkpoint,
    load_splats_from_ply,
    rigid_poses_at_frame,
    rigid_world_gaussians_from_pose,
)
from run_discovery import (  # noqa: E402
    RunNotFoundError,
    find_latest_checkpoint,
    load_class_names,
    load_instance_ids,
    resolve_run,
)
from trajectory.scenario import (  # noqa: E402
    SinusoidSpec,
    apply_ego_scenario,
    apply_rigid_scenario,
    step_length,
)
from dynamic.asset_library import (  # noqa: E402
    build_rigid_bank,
    list_assets,
    load_asset,
)

MODES = ("replay", "sinusoid", "user")


@dataclass
class TargetScenario:
    """Trajectory edits applied to one target (the ego, or one rigid instance)."""

    shift_m: list = None  # [x, y, z] metres; project convention, see scenario.py
    sinusoid: SinusoidSpec = None
    asset: str = None  # 3DRealCar model driving this instance's track (objects only)

    def __post_init__(self):
        if self.shift_m is None:
            self.shift_m = [0.0, 0.0, 0.0]
        if self.sinusoid is None:
            self.sinusoid = SinusoidSpec()

    @property
    def shifted(self) -> bool:
        return any(abs(float(v)) > 1e-9 for v in self.shift_m)

    def sin_active(self, force_on: bool = False) -> bool:
        """Weave is on when configured enabled, or when a mode forces it on."""
        return (force_on or self.sinusoid.enabled) and self.sinusoid.usable

    def active(self, force_sin: bool = False) -> bool:
        return self.shifted or self.sin_active(force_sin)

    @classmethod
    def from_cfg(cls, node) -> "TargetScenario":
        if node is None:
            return cls()
        try:
            shift = node.get("shift_m", None)
            sin_node = node.get("sinusoid", None)
            asset = node.get("asset", None)
        except AttributeError:
            return cls()
        shift = [0.0, 0.0, 0.0] if shift is None else [float(v) for v in shift]
        return cls(
            shift_m=shift,
            sinusoid=SinusoidSpec.from_cfg(sin_node),
            asset=str(asset) if asset else None,
        )


@dataclass
class BicycleState:
    x_m: float = 0.0
    z_m: float = 0.0
    yaw_rad: float = 0.0
    speed_mps: float = 0.0


def generate_config_hash(config_subset: dict) -> str:
    config_str = json.dumps(config_subset, sort_keys=True)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()[:8]


def _resolve_cached_inputs(cfg: DictConfig) -> tuple[str, str] | None:
    """Resolve simulator inputs from the same cache hashes used by the orchestrator.

    This is the exact-hash path: it reproduces the v2 (``results/4dgs``) naming
    recipe so that "run this specific configuration" stays reproducible. Returns
    ``None`` when the hashed directories do not exist, letting the caller fall
    back to :func:`run_discovery.resolve_run`.
    """
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    tracker_cfg = OmegaConf.to_container(cfg.tracker, resolve=True)
    track_task_cfg = OmegaConf.to_container(cfg.track_task, resolve=True)
    refine_task_cfg = OmegaConf.to_container(cfg.refine_task, resolve=True)
    gsplat_cfg = OmegaConf.to_container(cfg.gaussian_splatting, resolve=True)

    # The hash recipe below is v2's. Honouring the configured root means a v2
    # lookup still works when the root is switched back to results/4dgs; under
    # results/extended_4dgs it simply misses (v3 hashes different inputs), and
    # the caller falls through to the newest-run search.
    results_root = str(cfg.get("simulator", {}).get("results_root", "results/4dgs"))
    base_results_dir = os.path.abspath(
        os.path.join(results_root, dataset_cfg["name"], dataset_cfg["scene"])
    )

    ego_masks_dir = os.path.abspath(os.path.join(dataset_cfg["base_dir"], "masks"))
    ncore_hash = hashlib.md5(f"ncore_ego_{ego_masks_dir}".encode()).hexdigest()[:8]
    ncore_dir = os.path.join(base_results_dir, f"03_ncore_dataset_{ncore_hash}")

    # user_refinement settings must NOT fork the refine cache dir (same rule as
    # run_4dgs.py): the automatic refinement output is identical regardless of
    # them, and keeping the hash stable is what lets the simulator locate the
    # refine dir created during training even when user_refinement differs
    # (e.g. enabled=false for the sim run).
    refine_hash_cfg = {
        k: v for k, v in refine_task_cfg.items() if k != "user_refinement"
    }
    step15_hash = generate_config_hash(
        {
            "dataset": dataset_cfg,
            "tracker": tracker_cfg,
            "track_task": track_task_cfg,
            "refine_task": refine_hash_cfg,
        }
    )
    refine_dir = os.path.join(base_results_dir, f"15_refine_{step15_hash}")
    # Downstream always reads the refine-dir root: the refinement step mirrors the
    # latest (fused/filtered/fit + any interactive user edits) tracks and the
    # bicycle_params.json sidecar there, so there is no need to dig into
    # user_refinement/NNN/ snapshots.
    refined_tracks_json = os.path.join(refine_dir, "track_3d_refined_colmap.json")

    scene_root = os.path.abspath(dataset_cfg["base_dir"])
    step20_hash = generate_config_hash(
        {
            "gsplat": OmegaConf.to_yaml(gsplat_cfg),
            "ncore": ncore_dir,
            "tracks": refined_tracks_json,
            "scene_root": scene_root,
        }
    )
    training_dir = os.path.join(base_results_dir, f"20_gsplat_dynamic_{step20_hash}")
    camera_paths_dir = os.path.join(training_dir, "camera_paths")
    checkpoint_path = find_latest_checkpoint(os.path.join(training_dir, "ckpts"))

    if not os.path.isdir(camera_paths_dir) or not checkpoint_path:
        print(f"[sim] exact-hash 20_gsplat_dynamic_{step20_hash}: MISS")
        return None

    print(f"[sim] exact-hash 20_gsplat_dynamic_{step20_hash}: HIT")
    return camera_paths_dir, checkpoint_path


class TerminalKeyboard:
    """Non-blocking keyboard reader for Linux terminals (cbreak mode)."""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = None

    def __enter__(self):
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def read_keys(self) -> set[str]:
        keys: set[str] = set()
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                break
            ch = sys.stdin.read(1)
            if not ch:
                break
            keys.add(ch)
        return keys


class SimulatorRuntime:
    def __init__(self, cfg: DictConfig):
        render_cfg = cfg.get("rendering", {})
        sim_cfg = cfg.get("simulator", {})

        self.dt = float(sim_cfg.get("dt_sec", 0.1))
        self.max_fps = float(sim_cfg.get("max_fps", 10.0))
        self.loop_trajectory = bool(sim_cfg.get("loop_trajectory", True))
        self.advance_rigid_in_replay = bool(sim_cfg.get("advance_rigid_in_replay", True))

        self.mode = str(sim_cfg.get("mode", "replay"))
        if self.mode not in MODES:
            raise ValueError(f"simulator.mode must be one of {MODES}.")

        self.rigid_paused = bool(sim_cfg.get("rigid_paused", False))

        self.wheelbase_m = float(sim_cfg.get("wheelbase_m", 2.7))
        self.max_steer_rad = float(sim_cfg.get("max_steer_rad", 0.45))
        self.max_accel_mps2 = float(sim_cfg.get("max_accel_mps2", 3.0))
        self.max_brake_mps2 = float(sim_cfg.get("max_brake_mps2", 6.0))
        self.drag_per_sec = float(sim_cfg.get("drag_per_sec", 0.6))
        self.max_speed_mps = float(sim_cfg.get("max_speed_mps", 25.0))

        self.server_port = int(sim_cfg.get("port", 8090))

        camera_paths_dir = render_cfg.get("camera_paths_dir")
        checkpoint_path = render_cfg.get("checkpoint_path")
        self.run_paths = None
        if not camera_paths_dir or not checkpoint_path:
            auto_camera_paths_dir, auto_checkpoint_path = self._auto_resolve(cfg)
            camera_paths_dir = camera_paths_dir or auto_camera_paths_dir
            checkpoint_path = checkpoint_path or auto_checkpoint_path

        self.cameras = load_all_cameras(str(camera_paths_dir))
        if not self.cameras:
            raise SystemExit(
                f"[sim] no camera_data.json under {camera_paths_dir} -- this run "
                f"has no camera metadata to render from."
            )

        selected_camera_id = str(sim_cfg.get("start_camera_id", ""))
        if selected_camera_id:
            picked = [c for c in self.cameras if str(c.camera_id) == selected_camera_id]
            if not picked:
                available = ", ".join(str(c.camera_id) for c in self.cameras)
                raise ValueError(
                    f"start_camera_id='{selected_camera_id}' not found. Available: {available}"
                )
            self.camera = picked[0]
        else:
            self.camera = self.cameras[0]

        self.num_ego_frames = int(self.camera.camtoworlds.shape[0])
        if self.num_ego_frames <= 0:
            raise ValueError("Selected camera has empty camtoworld trajectory.")

        self.ego_frame_idx = max(
            0, min(int(sim_cfg.get("start_frame", 0)), self.num_ego_frames - 1)
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ply_path = render_cfg.get("ply_path")

        ppisp_state = None
        if ply_path and os.path.exists(str(ply_path)):
            self.splats = load_splats_from_ply(str(ply_path), device=device)
        elif checkpoint_path and os.path.exists(str(checkpoint_path)):
            self.splats, ppisp_state = load_splats_from_checkpoint(
                str(checkpoint_path), device=device
            )
        else:
            raise SystemExit(
                f"[sim] no model to load: rendering.ply_path={ply_path!r} and "
                f"checkpoint {checkpoint_path!r} are both missing."
            )

        render_dynamic = bool(render_cfg.get("render_dynamic", True))
        self.rigid_state = None
        if render_dynamic and checkpoint_path and os.path.exists(str(checkpoint_path)):
            self.rigid_state = load_rigid_state_from_checkpoint(
                str(checkpoint_path), device=device
            )

        self.num_rigid_frames = (
            int(self.rigid_state["poses.trans"].shape[0])
            if self.rigid_state is not None
            else 0
        )
        self.rigid_frame_idx = max(
            0,
            min(
                int(sim_cfg.get("start_rigid_frame", self.ego_frame_idx)),
                max(self.num_rigid_frames - 1, 0),
            ),
        )

        # On-demand rigid extrapolation past the baked span via the fitted
        # kinematic bicycle model (Step 1.5). Only active when the checkpoint
        # actually carries fitted params (``rigid_bicycle``). ``rigid_extrap_frames``
        # extends the rigid timeline by that many synthesized frames before it
        # loops; 0 disables extrapolation while keeping baked playback.
        self.rigid_extrapolate = bool(sim_cfg.get("extrapolate_rigid", True)) and (
            self.rigid_state is not None
            and self.rigid_state.get("bicycle") is not None
        )
        self.rigid_extrap_frames = max(0, int(sim_cfg.get("rigid_extrap_frames", 0)))
        # Effective rigid loop length: baked frames + extrapolation horizon.
        self.rigid_loop_frames = self.num_rigid_frames + (
            self.rigid_extrap_frames if self.rigid_extrapolate else 0
        )

        self.renderer = StandaloneRenderer(
            splats=self.splats,
            sh_degree=render_cfg.get("render", {}).get("sh_degree", 3),
            near_plane=render_cfg.get("render", {}).get("near_plane", 0.01),
            far_plane=render_cfg.get("render", {}).get("far_plane", 1e10),
            with_ut=render_cfg.get("render", {}).get("with_ut", True),
            with_eval3d=render_cfg.get("render", {}).get("with_eval3d", True),
            ppisp_state=ppisp_state,
            rigid_state=self.rigid_state,
            device=device,
        )

        self._setup_scenarios(sim_cfg, checkpoint_path)

        self.base_pose = self.camera.camtoworlds[self.ego_frame_idx].copy()
        self.bicycle = BicycleState()

        # Fit a ground plane to the ego trajectory camera positions using SVD.
        # The direction of smallest variance across all positions is the plane normal
        # (world "up"), which is robust to any PCA axis orientation and handles
        # gently inclined roads: the camera always stays on the fitted surface.
        positions = self.camera.camtoworlds[:, :3, 3]  # [N, 3]
        self.plane_centroid = positions.mean(axis=0).astype(np.float32)
        _, _, Vt = np.linalg.svd(positions - self.plane_centroid, full_matrices=False)
        plane_normal = Vt[-1].astype(np.float32)  # smallest-variance direction
        # Orient normal toward camera "up" (not down).
        avg_cam_up = (-self.camera.camtoworlds[:, :3, 1]).mean(axis=0)
        if float(np.dot(plane_normal, avg_cam_up)) < 0:
            plane_normal = -plane_normal
        self.plane_normal = plane_normal
        print(f"[Simulator] Ground plane normal = {self.plane_normal.round(3)}, centroid = {self.plane_centroid.round(3)}")

        self.server = viser.ViserServer(port=self.server_port, verbose=False)
        self.server.gui.set_panel_label("AVSplat Simulator")
        self._build_gui()
        # Display renders as full-viewport background so the image fills the 3D canvas.
        init_img = np.zeros((self.camera.height, self.camera.width, 3), dtype=np.uint8)
        self.server.scene.set_background_image(init_img)

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------
    def _build_gui(self) -> None:
        """Live scenario controls, mirroring the config blocks.

        Handler threading: viser callbacks run off the render thread. Scalar
        edits (shifts, sinusoid params, mode) are plain attribute writes and are
        safe. An asset swap is NOT -- it reallocates the whole rigid bank, which
        the render thread reads -- so those only set ``_pending_bank_rebuild``
        and the render loop performs the rebuild between frames.
        """
        gui = self.server.gui
        self._pending_bank_rebuild = False
        self._gui_syncing = False  # re-entrancy guard for programmatic .value sets

        with gui.add_folder("Playback"):
            self.g_mode = gui.add_dropdown("mode", MODES, initial_value=self.mode)
            self.g_paused = gui.add_checkbox("rigid paused", self.rigid_paused)
            b_ego = gui.add_button("reset ego")
            b_rigid = gui.add_button("reset rigid")
            b_all = gui.add_button("reset all")

        @self.g_mode.on_update
        def _(_) -> None:
            if not self._gui_syncing:
                self.mode = str(self.g_mode.value)

        @self.g_paused.on_update
        def _(_) -> None:
            if not self._gui_syncing:
                self.rigid_paused = bool(self.g_paused.value)

        b_ego.on_click(lambda _: self._reset_ego())
        b_rigid.on_click(lambda _: self._reset_rigid())

        @b_all.on_click
        def _(_) -> None:
            self._reset_ego()
            self._reset_rigid()

        with gui.add_folder("Ego"):
            self.g_ego = self._scenario_controls(gui, self.ego_scn, lambda: self.ego_scn)

        # --- Objects -------------------------------------------------------
        if not self.rigid_state is None and self.num_rigid_frames > 0:
            num_inst = int(self.rigid_state["instances_size"].shape[0])
            labels = [self._label_for_column(c) for c in range(num_inst)]
            self._gui_labels = labels
            with gui.add_folder("Objects"):
                self.g_target = gui.add_dropdown(
                    "target", labels, initial_value=labels[0]
                )
                self.g_asset = gui.add_dropdown(
                    "asset", ["(none)"] + self.asset_names, initial_value="(none)"
                )
                self.g_obj = self._scenario_controls(
                    gui, self._selected_scn(), lambda: self._selected_scn(create=True)
                )
                b_reset = gui.add_button("reset this object")

            @self.g_target.on_update
            def _(_) -> None:
                self._sync_object_controls()

            @self.g_asset.on_update
            def _(_) -> None:
                if self._gui_syncing:
                    return
                name = str(self.g_asset.value)
                scn = self._selected_scn(create=True)
                new = None if name == "(none)" else name
                if (scn.asset or None) != new:
                    scn.asset = new
                    # Deferred: rebuilding here would race the render thread.
                    self._pending_bank_rebuild = True

            @b_reset.on_click
            def _(_) -> None:
                col = self._selected_col()
                self.obj_scn[col] = TargetScenario()
                self._pending_bank_rebuild = True
                self._sync_object_controls()

            self._sync_object_controls()

        gui.add_markdown(
            """
### Keyboard (terminal)
- `w/s`: accelerate / brake &nbsp;&nbsp; `a/d`: steer left / right
- `m`: cycle mode &nbsp;&nbsp; `p`: rigid pause
- `r`: reset ego &nbsp;&nbsp; `t`: reset rigid &nbsp;&nbsp; `g`: reset both
- `q`: quit

Shifts are metres: **x** lateral (+right), **y** vertical (+down),
**z** longitudinal (+forward).
"""
        )
        self.status_handle = gui.add_markdown("Waiting for first frame...")

    def _scenario_controls(self, gui, scn: "TargetScenario", getter):
        """Shift + sinusoid widgets bound to whatever ``getter()`` returns.

        ``getter`` is indirect so the Objects panel can retarget the same widgets
        at a different instance column without rebuilding them -- a checkpoint
        can carry 60 instances, and a folder each would be unusable.
        """
        h = {}
        h["shift"] = gui.add_vector3(
            "shift (m)", tuple(float(v) for v in scn.shift_m), step=0.1
        )
        h["enabled"] = gui.add_checkbox("weave", scn.sinusoid.enabled)
        h["amp"] = gui.add_slider(
            "amplitude (m)", 0.0, 6.0, 0.1, float(scn.sinusoid.amplitude_m)
        )
        h["period"] = gui.add_slider(
            "period (frames)", 4.0, 200.0, 1.0, float(scn.sinusoid.period_frames)
        )
        h["phase"] = gui.add_slider(
            "phase (deg)", 0.0, 360.0, 5.0, float(scn.sinusoid.phase_deg)
        )
        h["yaw"] = gui.add_checkbox("follow yaw", scn.sinusoid.follow_yaw)

        def apply(_=None) -> None:
            if self._gui_syncing:
                return
            s = getter()
            s.shift_m = [float(v) for v in h["shift"].value]
            s.sinusoid.enabled = bool(h["enabled"].value)
            s.sinusoid.amplitude_m = float(h["amp"].value)
            s.sinusoid.period_frames = float(h["period"].value)
            s.sinusoid.phase_deg = float(h["phase"].value)
            s.sinusoid.follow_yaw = bool(h["yaw"].value)

        for handle in h.values():
            handle.on_update(apply)
        return h

    def _selected_col(self) -> int:
        try:
            return self._gui_labels.index(str(self.g_target.value))
        except (AttributeError, ValueError):
            return 0

    def _selected_scn(self, create: bool = False) -> "TargetScenario":
        """Scenario for the selected column.

        Reads must not create: merely *looking* at an object in the panel should
        not add an entry to ``obj_scn``, or the dict fills with inert scenarios
        and no longer answers "what has actually been configured?". Only an edit
        passes ``create=True``.
        """
        col = self._selected_col()
        if create:
            return self.obj_scn.setdefault(col, TargetScenario())
        return self.obj_scn.get(col) or TargetScenario()

    def _sync_object_controls(self) -> None:
        """Repopulate the object widgets from the currently selected column."""
        scn = self._selected_scn()
        self._gui_syncing = True
        try:
            self.g_asset.value = scn.asset or "(none)"
            h = self.g_obj
            h["shift"].value = tuple(float(v) for v in scn.shift_m)
            h["enabled"].value = bool(scn.sinusoid.enabled)
            h["amp"].value = float(scn.sinusoid.amplitude_m)
            h["period"].value = float(scn.sinusoid.period_frames)
            h["phase"].value = float(scn.sinusoid.phase_deg)
            h["yaw"].value = bool(scn.sinusoid.follow_yaw)
        finally:
            self._gui_syncing = False

    def _sync_playback_controls(self) -> None:
        """Push keyboard-driven state back into the panel."""
        if str(self.g_mode.value) == self.mode and bool(
            self.g_paused.value
        ) == self.rigid_paused:
            return
        self._gui_syncing = True
        try:
            self.g_mode.value = self.mode
            self.g_paused.value = self.rigid_paused
        finally:
            self._gui_syncing = False

    def _auto_resolve(self, cfg: DictConfig) -> tuple[str, str]:
        """Locate the run to load: exact config hash first, newest run second.

        The hash path keeps "run this specific configuration" reproducible. It
        stops resolving as configs grow (and never matches the v3 layout at all),
        so a miss falls back to the newest usable run *for the same
        dataset/scene* -- announced loudly, because it is not what was asked for.
        """
        sim_cfg = cfg.get("simulator", {})
        dataset_cfg = cfg.dataset
        results_root = str(sim_cfg.get("results_root", "results/extended_4dgs"))
        pin = sim_cfg.get("run_dir")

        if not pin:
            hashed = _resolve_cached_inputs(cfg)
            if hashed is not None:
                return hashed

        if not pin and not bool(sim_cfg.get("allow_latest_fallback", True)):
            raise SystemExit(
                "[sim] exact-hash lookup missed and simulator.allow_latest_fallback "
                "is false. Set it true, or pin simulator.run_dir=<train or loop dir>."
            )

        try:
            run = resolve_run(
                results_root,
                str(dataset_cfg.name),
                str(dataset_cfg.scene),
                pin=str(pin) if pin else None,
            )
        except RunNotFoundError as exc:
            raise SystemExit(f"[sim] {exc}") from exc

        self.run_paths = run
        root_abs = os.path.abspath(results_root)
        if run.source == "pin":
            print(f"[sim] using pinned run: {run.describe(root_abs)}")
        else:
            print(
                f"[sim] WARNING: falling back to newest run under "
                f"{os.path.join(results_root, str(dataset_cfg.name), str(dataset_cfg.scene))}\n"
                f"[sim]   -> {run.describe(root_abs)}\n"
                f"[sim]   this is NOT a config-hash match"
            )
        return run.camera_paths_dir, run.checkpoint_path

    def _setup_scenarios(self, sim_cfg, checkpoint_path) -> None:
        """Load ego/object scenarios and everything they need to be evaluated.

        Object scenarios are keyed by ORIGINAL TRACK ID in the config, because
        that is what is meaningful to a person reading a scene. Track IDs are not
        stored in the checkpoint, so they are recovered from the training
        ``cfg.yml`` (see ``run_discovery.load_instance_ids``). Different tracking
        or refinement runs renumber tracks, so a config written for one run may
        name IDs this one does not have: those entries are reported and skipped
        rather than silently landing on the wrong vehicle.
        """
        # Metres -> normalized units. The camera metadata and the checkpoint's
        # fitted bicycle transform carry the same value; either will do.
        scale = self.camera.world_to_normalized_scale
        if scale is None and self.rigid_state is not None:
            extrap = self.rigid_state.get("bicycle")
            if extrap is not None:
                scale = float(extrap["transform_scale"])
        if scale is None:
            print(
                "[sim] WARNING: no world_to_normalized_scale found -- shift and "
                "amplitude values will be in raw scene units, not metres."
            )
            scale = 1.0
        self.scale = float(scale)

        assets_node = sim_cfg.get("assets", None) or {}

        def _acfg(key, default):
            try:
                value = assets_node.get(key, default)
            except AttributeError:
                return default
            return default if value is None else value

        crop = assets_node.get("crop_to_box_margin", None) if assets_node else None
        self.assets_cfg = {
            "library_dir": str(_acfg("library_dir", "data/3DRealCar")),
            "opacity_threshold": float(_acfg("opacity_threshold", 0.0)),
            # None -> no box crop (the default). See asset_library.load_asset.
            "crop_to_box_margin": None if crop is None else float(crop),
            "max_points": int(_acfg("max_points", 750000)),
            "dc_only": bool(_acfg("dc_only", True)),
            "flip_forward": bool(_acfg("flip_forward", False)),
        }
        self.asset_names = [a.name for a in list_assets(self.assets_cfg["library_dir"])]
        # Always a valid bank: identity until a substitution replaces something,
        # so the early returns below still leave the untouched checkpoint path.
        self.rigid_bank = self.rigid_state
        # Set here rather than only in _build_gui, so the render loop can check it
        # regardless of how the runtime was constructed.
        self._pending_bank_rebuild = False

        self.ego_scn = TargetScenario.from_cfg(sim_cfg.get("ego", None))

        # Along-track distance per frame, for tangent yaw. Precomputed: tiny
        # (T=200-ish) and it keeps the render loop free of trajectory scans.
        ego_pos = self.camera.camtoworlds[:, :3, 3]
        self.ego_step = np.array(
            [step_length(ego_pos, f) for f in range(self.num_ego_frames)],
            dtype=np.float32,
        )

        self.instance_ids = None
        self.class_names = []
        self.obj_scn: dict[int, TargetScenario] = {}
        self.rigid_step = None
        self.rigid_first_valid = None

        num_inst = (
            int(self.rigid_state["instances_size"].shape[0])
            if self.rigid_state is not None
            else 0
        )
        if num_inst == 0:
            return

        trans = self.rigid_state["poses.trans"].detach().cpu().numpy()  # (T, M, 3)
        fv = self.rigid_state["instances_fv"].detach().cpu().numpy()  # (T, M)
        self.rigid_step = np.stack(
            [
                [step_length(trans[:, m, :], f) for m in range(num_inst)]
                for f in range(self.num_rigid_frames)
            ]
        ).astype(np.float32)
        # First frame each instance is present. The weave is phase-anchored here
        # so an object entering mid-sequence starts exactly on its real track
        # instead of popping in already displaced sideways.
        self.rigid_first_valid = np.array(
            [
                int(np.argmax(fv[:, m])) if bool(fv[:, m].any()) else 0
                for m in range(num_inst)
            ],
            dtype=np.int64,
        )

        cfg_yml = self.run_paths.cfg_yml if self.run_paths is not None else None
        if cfg_yml is None and checkpoint_path:
            candidate = os.path.join(
                os.path.dirname(os.path.dirname(str(checkpoint_path))), "cfg.yml"
            )
            cfg_yml = candidate if os.path.exists(candidate) else None
        self.instance_ids = load_instance_ids(cfg_yml, num_inst)
        if self.instance_ids:
            self.class_names = load_class_names(cfg_yml, self.instance_ids)

        objects_cfg = sim_cfg.get("objects", None)
        if not objects_cfg:
            return
        if not self.instance_ids:
            print(
                f"[sim] WARNING: simulator.objects is configured but the track-ID "
                f"mapping could not be recovered for this run; skipping all "
                f"{len(objects_cfg)} object scenario(s)."
            )
            return

        id_to_col = {tid: m for m, tid in enumerate(self.instance_ids)}
        for key, node in objects_cfg.items():
            try:
                track_id = int(key)
            except (TypeError, ValueError):
                print(f"[sim] WARNING: object key {key!r} is not a track ID; skipped.")
                continue
            col = id_to_col.get(track_id)
            if col is None:
                print(
                    f"[sim] WARNING: track ID {track_id} is not in this run "
                    f"(available: {self.instance_ids}); scenario skipped."
                )
                continue
            self.obj_scn[col] = TargetScenario.from_cfg(node)
            print(f"[sim] object scenario: track {track_id} -> instance column {col}")

        self._rebuild_bank()

    def _rebuild_bank(self) -> None:
        """Rebuild the rigid Gaussian bank from the current asset assignment.

        Only called when the assignment changes -- never per frame. With no
        substitutions the bank IS ``rigid_state``, so the untouched checkpoint
        path stays exactly as it was.
        """
        # Drop the previous bank's device tensors before building the next one,
        # so a swap does not hold two banks at once.
        had_bank = self.rigid_bank is not None and self.rigid_bank is not self.rigid_state
        self.rigid_bank = self.rigid_state
        if self.rigid_state is None:
            return

        assets = {}
        for col, scn in self.obj_scn.items():
            if not scn.asset:
                continue
            try:
                # Loads to host memory; build_rigid_bank moves it to the device.
                # Caching on the GPU would pin every model auditioned in the
                # picker for the life of the process.
                asset = load_asset(
                    self.assets_cfg["library_dir"],
                    scn.asset,
                    opacity_threshold=self.assets_cfg["opacity_threshold"],
                    crop_margin=self.assets_cfg["crop_to_box_margin"],
                    max_points=self.assets_cfg["max_points"],
                    dc_only=self.assets_cfg["dc_only"],
                )
            except (FileNotFoundError, OSError, ValueError, IndexError) as exc:
                print(f"[sim] WARNING: could not load asset {scn.asset!r}: {exc}")
                scn.asset = None
                continue
            assets[col] = asset
            box_h_m = float(self.rigid_state["instances_size"][col][2]) / self.scale
            print(
                f"[sim] substitute {self._label_for_column(col)} <- {asset.info} "
                f"[{asset.num_points} gaussians, box height {box_h_m:.2f} m]"
            )

        if not assets:
            # Back to the original vehicles. The old bank is already unreferenced;
            # hand its blocks back so the drop shows up in nvidia-smi rather than
            # sitting in the caching allocator.
            if had_bank:
                self._release_device_cache()
            return
        before = int(self.rigid_state["point_ids"].shape[0])
        self.rigid_bank = build_rigid_bank(
            self.rigid_state,
            assets,
            self.scale,
            dc_only=self.assets_cfg["dc_only"],
            flip_forward=self.assets_cfg["flip_forward"],
        )
        after = int(self.rigid_bank["point_ids"].shape[0])
        if had_bank:
            self._release_device_cache()
        print(
            f"[sim] rigid gaussians: {before} -> {after} ({after - before:+d})"
            f"{self._vram_note()}"
        )

    @staticmethod
    def _release_device_cache() -> None:
        """Return freed blocks to the driver so VRAM changes are observable."""
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _vram_note() -> str:
        if not torch.cuda.is_available():
            return ""
        return f"  |  VRAM {torch.cuda.memory_allocated() / 2**20:.0f} MB"

    def set_asset(self, col: int, name) -> None:
        """Assign (or clear, with None) an asset for one instance column."""
        scn = self.obj_scn.setdefault(col, TargetScenario())
        if (scn.asset or None) == (name or None):
            return
        scn.asset = str(name) if name else None
        self._rebuild_bank()

    def _substitutions_active(self) -> bool:
        return any(s.asset for s in self.obj_scn.values())

    def _label_for_column(self, col: int) -> str:
        """Human label for an instance column, e.g. 'id 53 (car)'."""
        if not self.instance_ids or col >= len(self.instance_ids):
            return f"col {col}"
        cls = (
            self.class_names[col]
            if col < len(self.class_names) and self.class_names[col]
            else ""
        )
        return f"id {self.instance_ids[col]}{f' ({cls})' if cls else ''}"

    @property
    def replay_like(self) -> bool:
        """Modes where the baked ego trajectory drives the camera.

        'sinusoid' is 'replay' with a lateral displacement layered on, so every
        playback decision (advancing frames, honouring advance_rigid_in_replay,
        applying ego scenarios) must treat the two together. Only 'user' hands
        control to the bicycle model instead.
        """
        return self.mode in ("replay", "sinusoid")

    @property
    def ego_sinusoid_on(self) -> bool:
        """Ego weave is on in 'sinusoid' mode, or whenever configured enabled."""
        return self.ego_scn.sin_active(self.mode == "sinusoid")

    def _ego_scenario_active(self) -> bool:
        return self.replay_like and self.ego_scn.active(self.mode == "sinusoid")

    def _rigid_scenarios_active(self) -> bool:
        return any(s.active() for s in self.obj_scn.values())

    def _scenario_rigid_gaussians(self):
        """World-space rigid Gaussians with per-object scenarios applied.

        Returns ``None`` when nothing needs perturbing, which lets the caller
        fall back to the untouched checkpoint path.
        """
        if self.rigid_state is None:
            return None
        if not (self._rigid_scenarios_active() or self._substitutions_active()):
            return None
        bank = self.rigid_bank if self.rigid_bank is not None else self.rigid_state
        poses = rigid_poses_at_frame(bank, int(self.rigid_frame_idx))
        if poses is None:
            return None

        # rigid_poses_at_frame hands back the checkpoint tensors themselves for
        # baked frames -- clone before touching them.
        trans, quats, fv = poses[0].clone(), poses[1].clone(), poses[2]
        baked = 0 <= self.rigid_frame_idx < self.num_rigid_frames

        for col, scn in self.obj_scn.items():
            if col >= trans.shape[0] or not bool(fv[col]):
                continue
            # Weave only where the object has a real trajectory to weave around:
            # never on bicycle-extrapolated frames, and phase-anchored to the
            # object's first valid frame.
            use_sin = baked and scn.sinusoid.active
            local_frame = (
                float(self.rigid_frame_idx - int(self.rigid_first_valid[col]))
                if use_sin
                else 0.0
            )
            step = (
                float(self.rigid_step[int(self.rigid_frame_idx), col])
                if (use_sin and self.rigid_step is not None)
                else 0.0
            )
            trans[col], quats[col] = apply_rigid_scenario(
                trans[col],
                quats[col],
                local_frame,
                self.scale,
                shift_m=scn.shift_m,
                sinusoid=scn.sinusoid if use_sin else None,
                step_len=step,
            )

        return rigid_world_gaussians_from_pose(bank, trans, quats, fv)

    def _build_user_pose(self) -> np.ndarray:
        pose = self.base_pose.copy()
        R0 = self.base_pose[:3, :3]

        # Project right (col 0) and forward (col 2) onto the plane perpendicular to
        # the trajectory-estimated world up. Using the trajectory average (not the
        # single-frame cam-down) cancels out camera pitch so the offset is truly
        # Fitted ground-plane normal (unit vector pointing "up").
        n = self.plane_normal

        # Directions in the ground plane: remove the normal component, renormalize.
        tx_flat = R0[:, 0] - np.dot(R0[:, 0], n) * n
        tz_flat = R0[:, 2] - np.dot(R0[:, 2], n) * n
        tx_n = np.linalg.norm(tx_flat)
        tz_n = np.linalg.norm(tz_flat)
        tx_flat = tx_flat / tx_n if tx_n > 1e-6 else R0[:, 0]
        tz_flat = tz_flat / tz_n if tz_n > 1e-6 else R0[:, 2]

        # Snap the base position onto the fitted plane.  Because tx_flat and
        # tz_flat are already in the plane, the resulting position is guaranteed
        # to lie on the ground plane for any (x_m, z_m) value.
        base_on_plane = self.base_pose[:3, 3] - (
            np.dot(self.base_pose[:3, 3] - self.plane_centroid, n) * n
        )
        pose[:3, 3] = base_on_plane + tx_flat * self.bicycle.x_m + tz_flat * self.bicycle.z_m

        cy = np.cos(self.bicycle.yaw_rad)
        sy = np.sin(self.bicycle.yaw_rad)
        Ry = np.array(
            [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32
        )
        pose[:3, :3] = R0 @ Ry
        return pose

    def _apply_user_controls(self, keys: set[str]) -> None:
        accel = 0.0
        if "w" in keys:
            accel += self.max_accel_mps2
        if "s" in keys:
            accel -= self.max_brake_mps2

        steer = 0.0
        if "a" in keys:
            steer -= self.max_steer_rad  # steer left
        if "d" in keys:
            steer += self.max_steer_rad  # steer right

        self.bicycle.speed_mps += accel * self.dt
        self.bicycle.speed_mps *= max(0.0, 1.0 - self.drag_per_sec * self.dt)
        self.bicycle.speed_mps = float(
            np.clip(self.bicycle.speed_mps, -self.max_speed_mps, self.max_speed_mps)
        )

        if abs(self.wheelbase_m) > 1e-6:
            self.bicycle.yaw_rad += (
                self.bicycle.speed_mps / self.wheelbase_m * np.tan(steer) * self.dt
            )

        self.bicycle.x_m += self.bicycle.speed_mps * np.sin(self.bicycle.yaw_rad) * self.dt
        self.bicycle.z_m += self.bicycle.speed_mps * np.cos(self.bicycle.yaw_rad) * self.dt

    def _advance_indices(self) -> None:
        # 'sinusoid' is a replay variant: the baked trajectory still drives the
        # camera, it is just displaced. Testing for "replay" alone here pinned the
        # ego at its start frame, which froze both the drive-through and the weave
        # (whose phase is that very index).
        if self.replay_like:
            if self.loop_trajectory:
                self.ego_frame_idx = (self.ego_frame_idx + 1) % self.num_ego_frames
            else:
                self.ego_frame_idx = min(self.ego_frame_idx + 1, self.num_ego_frames - 1)

        if self.num_rigid_frames > 0 and not self.rigid_paused:
            if not self.replay_like or self.advance_rigid_in_replay:
                self.rigid_frame_idx = (self.rigid_frame_idx + 1) % max(
                    self.rigid_loop_frames, 1
                )

    def _reset_ego(self) -> None:
        self.ego_frame_idx = 0
        self.base_pose = self.camera.camtoworlds[0].copy()
        self.bicycle = BicycleState()

    def _reset_rigid(self) -> None:
        self.rigid_frame_idx = 0

    def _current_pose(self) -> np.ndarray:
        if self.mode == "user":
            return self._build_user_pose()

        pose = self.camera.camtoworlds[self.ego_frame_idx]
        if not self._ego_scenario_active():
            return pose
        return apply_ego_scenario(
            pose,
            float(self.ego_frame_idx),
            self.scale,
            shift_m=self.ego_scn.shift_m,
            sinusoid=self.ego_scn.sinusoid if self.ego_sinusoid_on else None,
            step_len=float(self.ego_step[self.ego_frame_idx]),
            plane_normal=self.plane_normal,
        )

    def _render_frame(self) -> np.ndarray:
        rigid_idx = self.rigid_frame_idx if self.num_rigid_frames > 0 else None
        return self.renderer.render_frame(
            rigid_gaussians=self._scenario_rigid_gaussians(),
            camtoworld=self._current_pose(),
            K=self.camera.K,
            width=self.camera.width,
            height=self.camera.height,
            camera_model=self.camera.camera_model,
            radial_coeffs=self.camera.radial_coeffs,
            tangential_coeffs=self.camera.tangential_coeffs,
            thin_prism_coeffs=self.camera.thin_prism_coeffs,
            ftheta_coeffs=self.camera.ftheta_coeffs,
            camera_idx=self.camera.camera_index,
            rigid_frame_idx=rigid_idx,
        )

    def _scenario_summary(self) -> str:
        """Which scenarios are currently shaping the render."""
        parts = []
        if self._ego_scenario_active():
            bits = []
            if self.ego_scn.shifted:
                bits.append("shift " + ",".join(f"{v:g}" for v in self.ego_scn.shift_m))
            if self.ego_sinusoid_on:
                s = self.ego_scn.sinusoid
                bits.append(f"sin {s.amplitude_m:g}m/{s.period_frames:g}f")
            parts.append("ego: " + " + ".join(bits))

        baked = 0 <= self.rigid_frame_idx < self.num_rigid_frames
        for col, scn in sorted(self.obj_scn.items()):
            if not (scn.active() or scn.asset):
                continue
            bits = []
            if scn.asset:
                bits.append(f"asset {scn.asset}")
            if scn.shifted:
                bits.append("shift " + ",".join(f"{v:g}" for v in scn.shift_m))
            if scn.sinusoid.active:
                bits.append(
                    f"sin {scn.sinusoid.amplitude_m:g}m/{scn.sinusoid.period_frames:g}f"
                    + ("" if baked else " (held: extrapolated)")
                )
            parts.append(f"{self._label_for_column(col)}: " + " + ".join(bits))

        return "  \n".join(f"- {p}" for p in parts) if parts else "_none_"

    def _update_status(self) -> None:
        extrapolating = self.rigid_frame_idx >= self.num_rigid_frames
        self.status_handle.content = (
            f"**mode**: {self.mode}  \n"
            f"**ego_frame**: {self.ego_frame_idx}/{self.num_ego_frames - 1}  \n"
            f"**rigid_frame**: {self.rigid_frame_idx}/{max(self.rigid_loop_frames - 1, 0)}"
            f"{' (extrapolated)' if extrapolating else ''}  \n"
            f"**rigid_paused**: {self.rigid_paused}  \n"
            f"**speed_mps**: {self.bicycle.speed_mps:.2f}  \n"
            f"**offset_xz_m**: ({self.bicycle.x_m:.2f}, {self.bicycle.z_m:.2f})  \n"
            f"**scenarios**:  \n{self._scenario_summary()}"
        )

    def run(self) -> None:
        print("Simulator started.")
        print(
            "Controls: w/s/a/d move, m cycle mode (replay/sinusoid/user), p rigid pause, "
            "r ego reset, t rigid reset, g reset all, q quit"
        )
        target_period = 1.0 / max(self.max_fps, 1.0)

        with TerminalKeyboard() as kb:
            while True:
                loop_t0 = time.time()
                keys = kb.read_keys()

                if "q" in keys:
                    break
                if "m" in keys:
                    self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
                if "p" in keys:
                    self.rigid_paused = not self.rigid_paused
                if "r" in keys:
                    self._reset_ego()
                if "t" in keys:
                    self._reset_rigid()
                if "g" in keys:
                    self._reset_ego()
                    self._reset_rigid()

                if self.mode == "user":
                    self._apply_user_controls(keys)

                # Asset swaps are queued by the GUI thread and applied here, so
                # the rigid bank is only ever reallocated between renders.
                if self._pending_bank_rebuild:
                    self._pending_bank_rebuild = False
                    self._rebuild_bank()

                self._sync_playback_controls()
                self.server.scene.set_background_image(self._render_frame())
                self._update_status()
                self._advance_indices()

                sleep_s = max(target_period - (time.time() - loop_t0), 0.0)
                if sleep_s > 0:
                    time.sleep(sleep_s)

        self.server.stop()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("Starting standalone simulator...")
    print("rendering config:\n" + OmegaConf.to_yaml(cfg.get("rendering", {})))
    if cfg.get("simulator") is not None:
        print("simulator config:\n" + OmegaConf.to_yaml(cfg.get("simulator")))

    runtime = SimulatorRuntime(cfg)
    runtime.run()


if __name__ == "__main__":
    main()