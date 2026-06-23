"""Inference-only Wayve101 dataset adapter for CC-3DT."""
from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation as R

from vis4d.common.typing import DictStrAny
from vis4d.data.const import AxisMode
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.base import VideoDataset, VideoMapping
from vis4d.data.datasets.util import im_decode
from vis4d.data.typing import DictData


def _resolve_sparse_dir(data_root: str) -> str:
    """Locate the COLMAP sparse reconstruction under a scene root."""
    for candidate in (
        os.path.join(data_root, "colmap_sparse", "rig"),
        os.path.join(data_root, "sparse", "0"),
    ):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find COLMAP sparse data under {data_root}."
    )


def _load_colmap_scene(sparse_dir: str):
    """Load COLMAP cameras/images across pycolmap API variants."""
    if hasattr(pycolmap, "Reconstruction"):
        reconstruction = pycolmap.Reconstruction(sparse_dir)
        return reconstruction.cameras, reconstruction.images
    if hasattr(pycolmap, "SceneManager"):
        manager = pycolmap.SceneManager(sparse_dir)
        manager.load_cameras()
        manager.load_images()
        return manager.cameras, manager.images
    raise ImportError(
        "Neither pycolmap.Reconstruction nor pycolmap.SceneManager is available."
    )


def _extract_cam_to_world(image) -> np.ndarray:
    """Return the 4x4 cam->world transform for a COLMAP image."""
    bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    if hasattr(image, "cam_from_world"):
        world_to_cam = np.concatenate(
            [
                np.asarray(image.cam_from_world().matrix(), dtype=np.float32),
                bottom,
            ],
            axis=0,
        )
    else:
        rotation = image.R()
        translation = image.tvec.reshape(3, 1)
        world_to_cam = np.concatenate(
            [np.concatenate([rotation, translation], axis=1), bottom], axis=0
        ).astype(np.float32)
    return np.linalg.inv(world_to_cam).astype(np.float32)


def _world_alignment_from_ref(ref_cam_to_world: np.ndarray) -> np.ndarray:
    """Rigid rotation mapping the COLMAP world frame into the ROS frame.

    The reconstruction world frame for this rig is x-left, y-up, z-forward.
    CC-3DT and the visualizers expect a ROS global frame (x-forward, y-left,
    z-up) and read each object's yaw about world-z. Route world -> reference
    camera (OpenCV) -> ROS to get one rotation applied to every cam->world
    transform. For this scene it reduces to relabeling (x-left, y-up,
    z-forward) -> (x-forward, y-left, z-up).
    """
    ref_cam_to_ros = np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    r_cam_to_world = ref_cam_to_world[:3, :3].astype(np.float64)
    r_align = ref_cam_to_ros @ r_cam_to_world.T
    align = np.eye(4, dtype=np.float32)
    align[:3, :3] = r_align.astype(np.float32)
    return align


def compute_scene_world_alignment(
    data_root: str,
    cameras: Sequence[str],
    reference_camera: str = "front-forward",
) -> np.ndarray:
    """Compute the COLMAP->ROS world alignment used for a scene.

    Mirrors the dataset's internal computation so consumers (e.g. converting
    saved predictions back to COLMAP coordinates) can reproduce the exact same
    rigid rotation without instantiating the dataset.
    """
    cams = list(cameras)
    sparse_dir = _resolve_sparse_dir(data_root)
    _, images = _load_colmap_scene(sparse_dir)
    frames: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    for image in images.values():
        camera_name = Path(image.name).parent.as_posix()
        if camera_name not in cams:
            continue
        timestamp = int(Path(image.name).stem)
        frames[timestamp][camera_name] = _extract_cam_to_world(image)
    ordered = sorted(
        ts for ts, fr in frames.items() if all(c in fr for c in cams)
    )
    if not ordered:
        raise ValueError(
            f"No synchronized timestamps found across cameras {cams} "
            f"in {data_root}."
        )
    return _world_alignment_from_ref(frames[ordered[0]][reference_camera])


@dataclass(frozen=True)
class _FrameEntry:
    """Metadata for a single camera frame."""

    sample_name: str
    image_path: str
    image_hw: tuple[int, int]
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    timestamp: int


class Wayve101CC3DTDataset(VideoDataset):
    """Wayve101 multi-camera dataset adapted to the CC-3DT inference format.

    This adapter is intentionally inference-only and keeps the first version
    small: it uses strict timestamp intersection across the configured cameras
    and synthesizes the NuScenes-style ``can_bus`` vector from the pose of a
    reference camera.
    """

    KEYS = [
        K.images,
        K.input_hw,
        K.original_images,
        K.original_hw,
        K.intrinsics,
        K.extrinsics,
        K.axis_mode,
        K.timestamp,
    ]

    def __init__(
        self,
        data_root: str,
        cameras: Sequence[str],
        reference_camera: str = "front-forward",
        keys_to_load: Sequence[str] = (
            K.images,
            K.original_images,
        ),
        image_channel_mode: str = "RGB",
        data_backend=None,
        undistort: bool = True,
        undistort_balance: float = 0.0,
        undistort_mode: str = "balance",
        virtual_focal_px: float = 1266.0,
        image_size: Sequence[int] = (900, 1600),
    ) -> None:
        super().__init__(
            image_channel_mode=image_channel_mode,
            data_backend=data_backend,
        )
        self.data_root = data_root
        self.cameras = list(cameras)
        self.reference_camera = reference_camera
        self.keys_to_load = keys_to_load
        self.validate_keys(keys_to_load)

        if self.reference_camera not in self.cameras:
            raise ValueError(
                f"Reference camera '{self.reference_camera}' is not part of {self.cameras}."
            )

        # Fisheye rectification: COLMAP stores OPENCV_FISHEYE intrinsics with
        # k1..k4, but CC-3DT assumes a pinhole camera. Precompute per-camera
        # undistort maps and feed the rectified pinhole intrinsics instead.
        self.undistort = bool(undistort)
        self.undistort_balance = float(undistort_balance)
        self.undistort_mode = str(undistort_mode)
        self.virtual_focal_px = float(virtual_focal_px)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self._undistort_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._undistort_new_k: dict[str, np.ndarray] = {}

        self.samples = self._generate_data_mapping()
        self.video_mapping = self._generate_video_mapping()

    def __len__(self) -> int:
        return len(self.samples)

    def _generate_video_mapping(self) -> VideoMapping:
        video_to_indices: dict[str, list[int]] = defaultdict(list)
        video_to_frame_ids: dict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(self.samples):
            seq = sample["scene_name"]
            video_to_indices[seq].append(index)
            video_to_frame_ids[seq].append(sample["frame_ids"])
        return self._sort_video_mapping(
            {
                "video_to_indices": dict(video_to_indices),
                "video_to_frame_ids": dict(video_to_frame_ids),
            }
        )

    def _generate_data_mapping(self) -> list[DictStrAny]:
        sparse_dir_candidates = [
            os.path.join(self.data_root, "colmap_sparse", "rig"),
            os.path.join(self.data_root, "sparse", "0"),
        ]
        sparse_dir = next((p for p in sparse_dir_candidates if os.path.exists(p)), None)
        if sparse_dir is None:
            raise FileNotFoundError(
                f"Could not find COLMAP sparse data under {self.data_root}."
            )

        cameras, images = self._load_colmap_scene(sparse_dir)

        frames_by_timestamp: dict[int, dict[str, _FrameEntry]] = defaultdict(dict)
        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        for image_id, image in images.items():
            _ = image_id
            image_name = image.name
            camera_name = Path(image_name).parent.as_posix()
            if camera_name not in self.cameras:
                continue

            timestamp_str = Path(image_name).stem
            try:
                timestamp = int(timestamp_str)
            except ValueError as exc:
                raise ValueError(
                    f"Expected integer timestamp stem in image name '{image_name}'."
                ) from exc

            camera = cameras[image.camera_id]
            fx = getattr(camera, "fx", getattr(camera, "focal_length_x"))
            fy = getattr(camera, "fy", getattr(camera, "focal_length_y"))
            cx = getattr(camera, "cx", getattr(camera, "principal_point_x"))
            cy = getattr(camera, "cy", getattr(camera, "principal_point_y"))
            intrinsics = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )

            # Swap fisheye intrinsics for rectified pinhole intrinsics and
            # cache the remap once per physical camera.
            intrinsics = self._maybe_prepare_undistortion(
                camera_name, camera, intrinsics
            )

            if hasattr(image, "cam_from_world"):
                world_to_cam = np.concatenate(
                    [
                        np.asarray(image.cam_from_world().matrix(), dtype=np.float32),
                        bottom,
                    ],
                    axis=0,
                )
            else:
                rotation = image.R()
                translation = image.tvec.reshape(3, 1)
                world_to_cam = np.concatenate(
                    [np.concatenate([rotation, translation], axis=1), bottom], axis=0
                ).astype(np.float32)
            cam_to_world = np.linalg.inv(world_to_cam).astype(np.float32)

            image_path = os.path.join(self.data_root, "images", image_name)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image path does not exist: {image_path}")

            frames_by_timestamp[timestamp][camera_name] = _FrameEntry(
                sample_name=timestamp_str,
                image_path=image_path,
                image_hw=(camera.height, camera.width),
                intrinsics=intrinsics,
                extrinsics=cam_to_world,
                timestamp=timestamp,
            )

        ordered_timestamps = sorted(
            ts for ts, frames in frames_by_timestamp.items() if all(cam in frames for cam in self.cameras)
        )
        if not ordered_timestamps:
            raise ValueError(
                f"No synchronized timestamps found across cameras {self.cameras} in {self.data_root}."
            )

        scene_name = Path(self.data_root).name

        # Each camera's own pose is OpenCV (x-right, y-down, z-forward), but
        # the COLMAP world frame for this rig is x-left, y-up, z-forward (the
        # reference cam->world is ~diag(-1, -1, 1): camera-right -> world -x,
        # camera-down -> world -y, camera-forward -> world +z). CC-3DT and the
        # visualizers expect a ROS global frame (x-forward, y-left, z-up) and
        # read each object's yaw as a rotation about world-z. Rotate the whole
        # world into ROS with a single fixed alignment derived from the first
        # reference-camera pose, applied to every extrinsic.
        world_align = self._compute_world_alignment(
            frames_by_timestamp[ordered_timestamps[0]][
                self.reference_camera
            ].extrinsics
        )

        samples: list[DictStrAny] = []
        for frame_id, timestamp in enumerate(ordered_timestamps):
            per_camera = frames_by_timestamp[timestamp]
            reference_pose = (
                world_align @ per_camera[self.reference_camera].extrinsics
            ).astype(np.float32)
            sample: DictStrAny = {
                "scene_name": scene_name,
                "token": f"{scene_name}_{timestamp}",
                "frame_ids": frame_id,
                "can_bus": self._build_can_bus(reference_pose),
            }

            for camera_name in self.cameras:
                entry = per_camera[camera_name]
                sample[camera_name] = {
                    "sample_name": entry.sample_name,
                    "image_path": entry.image_path,
                    "image_hw": entry.image_hw,
                    "intrinsics": entry.intrinsics,
                    "extrinsics": (
                        world_align @ entry.extrinsics
                    ).astype(np.float32),
                    "timestamp": entry.timestamp,
                }
                if camera_name == self.reference_camera:
                    # Ego-centric BEV anchor: a leveled LIDAR-convention pose
                    # (x-right, y-forward, z-up) synthesized from the reference
                    # camera, emulating the nuScenes LIDAR_TOP extrinsics.
                    sample[camera_name]["bev_extrinsics"] = (
                        self._build_ego_bev_extrinsics(reference_pose)
                    )

            samples.append(sample)

        return samples

    def _load_colmap_scene(self, sparse_dir: str):
        """Load COLMAP cameras/images across pycolmap API variants."""
        if hasattr(pycolmap, "Reconstruction"):
            reconstruction = pycolmap.Reconstruction(sparse_dir)
            return reconstruction.cameras, reconstruction.images

        if hasattr(pycolmap, "SceneManager"):
            manager = pycolmap.SceneManager(sparse_dir)
            manager.load_cameras()
            manager.load_images()
            return manager.cameras, manager.images

        raise ImportError("Neither pycolmap.Reconstruction nor pycolmap.SceneManager is available.")

    def _maybe_prepare_undistortion(
        self, camera_name: str, camera, intrinsics: np.ndarray
    ) -> np.ndarray:
        """Return rectified pinhole intrinsics, caching the remap per camera.

        For OPENCV_FISHEYE cameras this estimates a new pinhole camera matrix
        and precomputes the undistort/rectify maps (with R=identity so the
        camera orientation, and therefore the extrinsics, stay valid). For any
        other model it returns the intrinsics unchanged.
        """
        if not self.undistort:
            return intrinsics

        model_name = getattr(getattr(camera, "model", None), "name", None)
        if model_name is None:
            model_name = str(getattr(camera, "model", ""))
        params = np.asarray(camera.params, dtype=np.float64)
        if "FISHEYE" not in model_name.upper() or params.size < 8:
            return intrinsics

        if camera_name not in self._undistort_maps:
            distortion = params[4:8]
            size = (int(camera.width), int(camera.height))
            resize_scale = min(
                self.image_size[0] / float(camera.height),
                self.image_size[1] / float(camera.width),
            )
            target_focal_native = self.virtual_focal_px / resize_scale
            new_k, map1, map2 = self._build_undistort_map(
                intrinsics,
                distortion,
                size,
                self.undistort_balance,
                self.undistort_mode,
                target_focal_native,
            )
            self._undistort_maps[camera_name] = (map1, map2)
            self._undistort_new_k[camera_name] = new_k
        return self._undistort_new_k[camera_name]

    @staticmethod
    def _build_undistort_map(
        intrinsics: np.ndarray,
        distortion: np.ndarray,
        size: tuple[int, int],
        balance: float,
        mode: str = "balance",
        target_focal_native: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the fisheye rectification map and the new pinhole matrix.

        Args:
            intrinsics: Original 3x3 fisheye camera matrix.
            distortion: Kannala-Brandt coefficients (k1, k2, k3, k4).
            size: (width, height) of the image.
            balance: Used in "balance" mode. 0 keeps only valid pixels (max
                edge crop, no black borders); 1 keeps the full source FOV
                (introduces black borders). Values in between trade FOV for
                cropping.
            mode: "balance" lets OpenCV pick the rectified focal from balance;
                "virtual" fixes the focal to ``target_focal_native`` (a virtual
                pinhole camera, e.g. matching nuScenes scale) with a centered
                principal point, cropping FOV symmetrically.
            target_focal_native: Native-resolution focal length (pixels) for
                "virtual" mode.
        """
        k = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        d = np.asarray(distortion, dtype=np.float64).reshape(4, 1)
        if mode == "virtual":
            if target_focal_native is None:
                raise ValueError(
                    "virtual undistort mode requires target_focal_native."
                )
            width, height = int(size[0]), int(size[1])
            f = float(target_focal_native)
            new_k = np.array(
                [
                    [f, 0.0, width / 2.0],
                    [0.0, f, height / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        else:
            new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                k, d, size, np.eye(3), balance=float(balance)
            )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            k, d, np.eye(3), new_k, size, cv2.CV_16SC2
        )
        return new_k.astype(np.float32), map1, map2

    def _load_image(self, image_path: str, camera_name: str) -> np.ndarray:
        """Decode an image, rectifying it if the camera is fisheye."""
        image_bytes = self.data_backend.get(image_path)
        image = np.ascontiguousarray(
            im_decode(image_bytes, mode=self.image_channel_mode),
            dtype=np.float32,
        )
        if self.undistort and camera_name in self._undistort_maps:
            map1, map2 = self._undistort_maps[camera_name]
            image = cv2.remap(
                image, map1, map2, interpolation=cv2.INTER_LINEAR
            )
        return np.ascontiguousarray(image)[None]

    def _build_can_bus(self, cam_to_world: np.ndarray) -> list[float]:
        can_bus = [0.0] * 18
        translation = cam_to_world[:3, 3].astype(np.float32)
        quat_xyzw = R.from_matrix(cam_to_world[:3, :3]).as_quat().astype(np.float32)
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float32,
        )
        yaw_rad = float(np.arctan2(cam_to_world[1, 2], cam_to_world[0, 2]))
        yaw_deg = np.degrees(yaw_rad)

        can_bus[:3] = translation.tolist()
        can_bus[3:7] = quat_wxyz.tolist()
        can_bus[-2] = yaw_rad
        can_bus[-1] = float(yaw_deg)
        return can_bus

    @staticmethod
    def _compute_world_alignment(ref_cam_to_world: np.ndarray) -> np.ndarray:
        """Rotate the COLMAP world frame into the ROS frame CC-3DT expects.

        Each camera's own pose is in OpenCV convention (x-right, y-down,
        z-forward). The reconstruction's world frame for this rig is a
        different frame: x-left, y-up, z-forward (empirically the reference
        cam->world rotation is ~diag(-1, -1, 1), i.e. a 180deg turn about the
        shared forward axis). CC-3DT and the visualizers assume a ROS global
        frame (x-forward, y-left, z-up) and extract yaw about world-z.

        Route world -> reference-camera-frame -> ROS. Because the camera's own
        frame is OpenCV, the camera -> ROS step is the standard mapping::

            cam +z (forward) -> ROS +x
            cam +x (right)   -> ROS -y
            cam +y (down)    -> ROS -z

        Composing with (world -> ref camera) = ref_cam_to_world.T yields one
        rigid rotation applied to every camera's cam->world transform. For
        this scene it reduces to relabeling (x-left, y-up, z-forward) ->
        (x-forward, y-left, z-up).
        """
        ref_cam_to_ros = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        r_cam_to_world = ref_cam_to_world[:3, :3].astype(np.float64)
        r_align = ref_cam_to_ros @ r_cam_to_world.T
        align = np.eye(4, dtype=np.float32)
        align[:3, :3] = r_align.astype(np.float32)
        return align

    @staticmethod
    def _build_ego_bev_extrinsics(ref_cam_to_world_ros: np.ndarray) -> np.ndarray:
        """Build a leveled ego pose for the BEV visualizer.

        The BEV visualizer is ego-centric: it inverts this sensor->global
        transform, rotates boxes into the sensor frame, and draws the x-y
        footprint with a fixed ego icon pointing up. It assumes a LIDAR-
        convention sensor frame (x-right, y-forward, z-up).

        We have no LIDAR, so synthesize an ego pose from the (already ROS-
        aligned) reference camera pose:

        * position = reference camera position in ROS global,
        * forward  = camera viewing direction projected onto the ground plane
          (heading only, pitch/roll dropped so boxes stay flat in BEV),
        * up       = world +z,
        * right    = forward x up.

        Columns are laid out [right, forward, up] to match LIDAR convention.
        """
        ref = ref_cam_to_world_ros.astype(np.float64)
        position = ref[:3, 3]

        # Camera +z (viewing direction) expressed in ROS world.
        cam_forward = ref[:3, :3] @ np.array([0.0, 0.0, 1.0])

        up = np.array([0.0, 0.0, 1.0])
        forward = cam_forward - np.dot(cam_forward, up) * up
        norm = np.linalg.norm(forward)
        forward = forward / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, up)

        bev = np.eye(4, dtype=np.float32)
        bev[:3, 0] = right.astype(np.float32)
        bev[:3, 1] = forward.astype(np.float32)
        bev[:3, 2] = up.astype(np.float32)
        bev[:3, 3] = position.astype(np.float32)
        return bev

    def __getitem__(self, idx: int) -> DictData:
        sample = self.samples[idx]
        data_dict: DictData = {
            "token": sample["token"],
            K.frame_ids: sample["frame_ids"],
            K.sequence_names: sample["scene_name"],
            "can_bus": sample["can_bus"],
        }

        for camera_name in self.cameras:
            cam_data = sample[camera_name]
            data_dict[camera_name] = {K.timestamp: cam_data["timestamp"]}

            if "bev_extrinsics" in cam_data:
                data_dict[camera_name]["bev_extrinsics"] = cam_data[
                    "bev_extrinsics"
                ]

            if K.images in self.keys_to_load:
                image = self._load_image(
                    cam_data["image_path"], camera_name
                )
                data_dict[camera_name][K.images] = image
                data_dict[camera_name][K.input_hw] = cam_data["image_hw"]
                data_dict[camera_name][K.sample_names] = cam_data["sample_name"]
                data_dict[camera_name][K.intrinsics] = cam_data["intrinsics"]
                data_dict[camera_name][K.extrinsics] = cam_data["extrinsics"]
                data_dict[camera_name][K.axis_mode] = AxisMode.OPENCV

            if K.original_images in self.keys_to_load:
                if K.images not in data_dict[camera_name]:
                    data_dict[camera_name][K.images] = self._load_image(
                        cam_data["image_path"], camera_name
                    )
                data_dict[camera_name][K.original_images] = data_dict[camera_name][K.images]
                data_dict[camera_name][K.original_hw] = cam_data["image_hw"]

        return data_dict