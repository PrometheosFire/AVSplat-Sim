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

from render_standalone import (  # noqa: E402
    StandaloneRenderer,
    load_all_cameras,
    load_rigid_state_from_checkpoint,
    load_splats_from_checkpoint,
    load_splats_from_ply,
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


def _find_latest_checkpoint(ckpt_dir: str) -> str:
    if not os.path.isdir(ckpt_dir):
        return ""

    def step_of(fname: str) -> int:
        try:
            return int(fname.split("_")[1])
        except (IndexError, ValueError):
            return -1

    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")], key=step_of)
    return os.path.join(ckpt_dir, ckpts[-1]) if ckpts else ""


def _resolve_cached_inputs(cfg: DictConfig) -> tuple[str, str]:
    """Resolve simulator inputs from the same cache hashes used by the orchestrator."""
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    tracker_cfg = OmegaConf.to_container(cfg.tracker, resolve=True)
    track_task_cfg = OmegaConf.to_container(cfg.track_task, resolve=True)
    refine_task_cfg = OmegaConf.to_container(cfg.refine_task, resolve=True)
    gsplat_cfg = OmegaConf.to_container(cfg.gaussian_splatting, resolve=True)

    base_results_dir = os.path.abspath(
        f"results/4dgs/{dataset_cfg['name']}/{dataset_cfg['scene']}"
    )
    static_results_dir = os.path.abspath(
        f"results/{dataset_cfg['name']}/{dataset_cfg['scene']}"
    )

    ego_masks_dir = os.path.abspath(os.path.join(dataset_cfg["base_dir"], "masks"))
    ncore_hash = hashlib.md5(f"ncore_ego_{ego_masks_dir}".encode()).hexdigest()[:8]
    ncore_dir = os.path.join(static_results_dir, f"03_ncore_dataset_{ncore_hash}")

    step15_hash = generate_config_hash(
        {
            "dataset": dataset_cfg,
            "tracker": tracker_cfg,
            "track_task": track_task_cfg,
            "refine_task": refine_task_cfg,
        }
    )
    refine_dir = os.path.join(base_results_dir, f"15_refine_{step15_hash}")
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
    checkpoint_path = _find_latest_checkpoint(os.path.join(training_dir, "ckpts"))

    if not os.path.isdir(camera_paths_dir) or not checkpoint_path:
        print("could not find configuration")
        raise SystemExit(1)

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
        if self.mode not in ("replay", "user"):
            raise ValueError("simulator.mode must be 'replay' or 'user'.")

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
        if not camera_paths_dir or not checkpoint_path:
            auto_camera_paths_dir, auto_checkpoint_path = _resolve_cached_inputs(cfg)
            camera_paths_dir = camera_paths_dir or auto_camera_paths_dir
            checkpoint_path = checkpoint_path or auto_checkpoint_path

        self.cameras = load_all_cameras(str(camera_paths_dir))
        if not self.cameras:
            print("could not find configuration")
            raise SystemExit(1)

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
            print("could not find configuration")
            raise SystemExit(1)

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

        self.server.gui.add_markdown(
            """
### Keyboard (terminal)
- `w/s`: accelerate / brake
- `a/d`: steer left / right
- `m`: toggle mode (`replay` <-> `user`)
- `p`: toggle rigid timeline pause
- `r`: reset ego pose
- `t`: reset rigid timestep
- `g`: reset both
- `q`: quit
"""
        )
        self.status_handle = self.server.gui.add_markdown("Waiting for first frame...")
        # Display renders as full-viewport background so the image fills the 3D canvas.
        init_img = np.zeros((self.camera.height, self.camera.width, 3), dtype=np.uint8)
        self.server.scene.set_background_image(init_img)

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
        if self.mode == "replay":
            if self.loop_trajectory:
                self.ego_frame_idx = (self.ego_frame_idx + 1) % self.num_ego_frames
            else:
                self.ego_frame_idx = min(self.ego_frame_idx + 1, self.num_ego_frames - 1)

        if self.num_rigid_frames > 0 and not self.rigid_paused:
            if self.mode != "replay" or self.advance_rigid_in_replay:
                self.rigid_frame_idx = (self.rigid_frame_idx + 1) % self.num_rigid_frames

    def _reset_ego(self) -> None:
        self.ego_frame_idx = 0
        self.base_pose = self.camera.camtoworlds[0].copy()
        self.bicycle = BicycleState()

    def _reset_rigid(self) -> None:
        self.rigid_frame_idx = 0

    def _current_pose(self) -> np.ndarray:
        return (
            self.camera.camtoworlds[self.ego_frame_idx]
            if self.mode == "replay"
            else self._build_user_pose()
        )

    def _render_frame(self) -> np.ndarray:
        rigid_idx = self.rigid_frame_idx if self.num_rigid_frames > 0 else None
        return self.renderer.render_frame(
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

    def _update_status(self) -> None:
        self.status_handle.content = (
            f"**mode**: {self.mode}  \n"
            f"**ego_frame**: {self.ego_frame_idx}/{self.num_ego_frames - 1}  \n"
            f"**rigid_frame**: {self.rigid_frame_idx}/{max(self.num_rigid_frames - 1, 0)}  \n"
            f"**rigid_paused**: {self.rigid_paused}  \n"
            f"**speed_mps**: {self.bicycle.speed_mps:.2f}  \n"
            f"**offset_xz_m**: ({self.bicycle.x_m:.2f}, {self.bicycle.z_m:.2f})"
        )

    def run(self) -> None:
        print("Simulator started.")
        print(
            "Controls: w/s/a/d move, m mode, p rigid pause, r ego reset, t rigid reset, g reset all, q quit"
        )
        target_period = 1.0 / max(self.max_fps, 1.0)

        with TerminalKeyboard() as kb:
            while True:
                loop_t0 = time.time()
                keys = kb.read_keys()

                if "q" in keys:
                    break
                if "m" in keys:
                    self.mode = "user" if self.mode == "replay" else "replay"
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