# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.utils.data
from scipy import ndimage

import ncore.data
import ncore.data.v4
import ncore.sensors

from gsplat.rendering import FThetaCameraDistortionParameters, FThetaPolynomialType

from .ncore_utils import FrameConversion
from .normalize import (
    similarity_from_cameras,
    align_principal_axes,
    transform_cameras,
    transform_points,
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CameraRenderData:
    """Per-camera rendering parameters for gsplat rasterization."""

    camera_model: str  # "pinhole" | "fisheye" | "ftheta"
    ftheta_coeffs: Optional[
        FThetaCameraDistortionParameters
    ]  # non-None only for ftheta
    radial_coeffs: Optional[np.ndarray]  # (4,) fisheye or (4|6,) pinhole; float32
    tangential_coeffs: Optional[np.ndarray]  # (2,) pinhole only; float32 or None
    thin_prism_coeffs: Optional[np.ndarray]  # (4,) pinhole only; float32 or None


def _build_pinhole_K(
    model_params: ncore.data.OpenCVPinholeCameraModelParameters
    | ncore.data.OpenCVFisheyeCameraModelParameters,
) -> np.ndarray:
    """Build a 3x3 pinhole-style intrinsic matrix from NCore camera parameters.

    This helper is used for both OpenCV pinhole and OpenCV fisheye models,
    because the trainer expects a standard ``K`` matrix even when the actual
    camera has additional distortion coefficients. The returned matrix uses
    the model's focal length and principal point, and assumes zero skew with
    the principal point stored in pixel coordinates.

    Args:
        model_params: NCore camera model parameters that expose
            ``focal_length`` and ``principal_point``.

    Returns:
        A ``(3, 3)`` float32 intrinsic matrix in the usual pinhole form.
    """
    fl = model_params.focal_length
    pp = model_params.principal_point
    fx = float(fl[0]) if hasattr(fl, "__getitem__") else float(fl)
    fy = float(fl[1]) if hasattr(fl, "__getitem__") else float(fl)
    cx, cy = float(pp[0]), float(pp[1])
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _load_ego_mask(
    sensor: ncore.data.CameraSensorProtocol, n_dilation: int
) -> Optional[np.ndarray]:
    """Load and dilate the camera's ego-vehicle mask, if one exists.

    The NCore camera sensor may expose named mask images through
    ``sensor.get_mask_images()``. This helper looks for the ``"ego"`` mask,
    which marks pixels belonging to the host vehicle. When present, the mask
    is converted to a single-channel grayscale image, thresholded into a
    boolean array, and expanded with morphological dilation so the final mask
    is slightly more conservative.

    Args:
        sensor: NCore camera sensor that provides mask images via the
            ``CameraSensorProtocol`` interface.
        n_dilation: Number of binary dilation iterations to apply to the ego
            mask. Larger values remove a wider region around the vehicle.

    Returns:
        A boolean array with ``True`` for ego-vehicle pixels, or ``None`` if
        the sensor does not provide an ``"ego"`` mask.
    """
    mask_images = sensor.get_mask_images()
    if "ego" not in mask_images:
        return None
    mask = np.asarray(mask_images["ego"].convert("L")) != 0
    return ndimage.binary_dilation(mask, iterations=n_dilation).astype(bool)


def _parse_optional_coeffs(coeffs: Optional[Any]) -> Optional[np.ndarray]:
    """Convert optional distortion coefficients to float32 array, mapping all zero-arrays to None.

    NCore camera parameter objects expose different distortion coefficient
    fields depending on the camera model. This helper turns any provided
    coefficient sequence into a ``float32`` NumPy array so the rest of the
    parser can treat the values uniformly. If the input is ``None`` or every
    coefficient is zero, the function returns ``None`` to signal that the
    camera should be treated as having no usable distortion parameters.

    Args:
        coeffs: Optional coefficient sequence read from an NCore camera
            model object.

    Returns:
        A ``float32`` NumPy array with the coefficients, or ``None`` if the
        coefficients are missing or all zero.
    """
    if coeffs is None:
        return None
    coeffs_array = np.array(coeffs, dtype=np.float32)
    if (coeffs_array == 0).all():
        return None
    return coeffs_array


# ---------------------------------------------------------------------------
# NCoreParser
# ---------------------------------------------------------------------------


class NCoreParser:
    """NCore v4 data parser.

    Loads all frame metadata (poses, K matrices, frame lists) eagerly at init
    time. Images are loaded lazily by NCoreDataset.__getitem__.

    Coordinate frames
    -----------------
    NCore world frame  - raw sequence frame from the SLAM/pose graph.
                         Origin is typically the start of the drive;
                         units are real-world metres.

    world_global frame - globally-consistent reference frame obtained
                         from the pose graph edge "world" -> "world_global".
                         T_world_to_scene_world rotates/aligns into this frame.

    scene frame        - world_global translated so that the mean camera
                         position is at the origin.  This is the frame stored
                         in self.camtoworlds and consumed by the trainer.
                         Keeping poses near the origin improves numerical
                         stability during 3DGS optimisation.
                         When normalize_world_space=True, an additional
                         similarity + PCA transform is applied on top.
                         world_global_to_scene (FrameConversion) applies only
                         this translation; all rotation is already handled by
                         T_world_to_scene_world.
    """

    def __init__(
        self,
        meta_json_path: str,
        factor: float = 1.0,
        test_every: int = 8,
        camera_ids: Optional[List[str]] = None,
        lidar_ids: Optional[List[str]] = None,
        seek_offset_sec: Optional[float] = None,
        duration_sec: Optional[float] = None,
        max_lidar_points: int = 500_000,
        lidar_step_frame: int = 1,
        poses_component_group: str = "default",
        intrinsics_component_group: str = "default",
        masks_component_group: str = "default",
        open_consolidated: bool = True,
        n_camera_mask_dilation_iterations: int = 30,
        lidar_color_generic_data_name: str = "rgb",
        normalize_world_space: bool = False,
    ) -> None:
        """Parse an NCore sequence and precompute the metadata needed for training.

        This initializer eagerly opens the NCore sequence file, resolves the
        requested camera and lidar sensors, loads per-camera intrinsics and
        masks, batches the frame poses into scene coordinates, and optionally
        loads lidar points for Gaussian initialization. The resulting parser
        stores the scene in a stable coordinate frame that the dataset can
        query later without reopening the sequence metadata.

        The NCore-specific arguments control which sensors and time span are
        used:

        - ``camera_ids`` and ``lidar_ids`` select which sensors to include.
        - ``seek_offset_sec`` and ``duration_sec`` restrict the time window.
        - ``poses_component_group``, ``intrinsics_component_group``, and
          ``masks_component_group`` choose which metadata component groups are
          read from the sequence file.
        - ``n_camera_mask_dilation_iterations`` expands ego masks to make the
          invalid region slightly more conservative.
        - ``lidar_color_generic_data_name`` selects the per-point color field
          used when lidar points provide RGB data.
        - ``normalize_world_space`` applies the same camera-based
          recentering, PCA alignment, and upside-down correction used by the
          COLMAP parser.

        Args:
            meta_json_path: Path to the NCore single-sequence metadata JSON.
            factor: Image downscaling factor applied to camera intrinsics and
                loaded images.
            test_every: Hold out every Nth frame for the test split.
            camera_ids: Optional list of camera sensor IDs to load.
            lidar_ids: Optional list of lidar sensor IDs to load.
            seek_offset_sec: Optional offset, in seconds, applied to the start
                of the loaded time window.
            duration_sec: Optional maximum duration, in seconds, to load.
            max_lidar_points: Maximum number of lidar points kept for
                Gaussian initialization.
            lidar_step_frame: Load every Nth lidar frame when building the
                initial point cloud.
            poses_component_group: Name of the NCore component group that
                stores pose data.
            intrinsics_component_group: Name of the NCore component group that
                stores camera intrinsics.
            masks_component_group: Name of the NCore component group that
                stores camera masks.
            open_consolidated: Whether to open consolidated component stores.
            n_camera_mask_dilation_iterations: Number of dilation iterations
                applied to ego masks.
            lidar_color_generic_data_name: Generic-data key used to read RGB
                colors from lidar frames.
            normalize_world_space: Whether to normalize poses and points into
                a centered, axis-aligned scene frame.
        """
        self.test_every = test_every
        self.factor = factor
        self.normalize_world_space = normalize_world_space
        self.poses_component_group = poses_component_group
        self.intrinsics_component_group = intrinsics_component_group
        self.masks_component_group = masks_component_group
        self.open_consolidated = open_consolidated
        self.lidar_color_generic_data_name = lidar_color_generic_data_name

        self.sequence_meta_file_path: Path = Path(meta_json_path)
        sequence_loader = self._open_sequence_loader(self.sequence_meta_file_path) # load the sequence metadata and prepare to query cameras, poses, timestamps, etc.

        self.sequence_id: str = sequence_loader.sequence_id

        time_range = sequence_loader.sequence_timestamp_interval_us
        start_us = time_range.start
        stop_us = time_range.stop
        if seek_offset_sec is not None:
            start_us += int(seek_offset_sec * 1e6)
        if duration_sec is not None and duration_sec > 0:
            stop_us = min(start_us + int(duration_sec * 1e6), stop_us)
        self.time_range_us = dataclasses.replace(
            time_range, start=start_us, stop=stop_us
        )

        self._resolve_sensor_ids(sequence_loader, camera_ids, lidar_ids) # determine which camera and lidar sensors to use, either from user input or by auto-detection when only one sensor is available for a modality
        self._compute_world_global_transform(sequence_loader) # compute the transform from the NCore world frame to the world_global frame, which is a globally-aligned reference frame provided by the pose graph when available; this transform is used to express subsequent poses in the global frame
        
        # Load camera sensors and derive per-camera metadata (intrinsics, ego masks, render parameters) for all selected cameras; store the results in parser attributes for later use by the dataset
        camera_sensors = self._load_camera_data(
            sequence_loader, factor, n_camera_mask_dilation_iterations
        )
        # For each camera, determine which frames fall within the loaded time range and store the resulting frame lists in a dictionary
        camera_frame_ranges = {
            cid: self._get_sensor_frame_range(camera_sensors[cid].frames_timestamps_us)
            for cid in self.camera_ids
        }
        self._compute_scene_origin(camera_sensors, camera_frame_ranges)
        self._load_poses(camera_sensors, camera_frame_ranges) # Load camera poses in world frame

        # Stub attrs for render_traj compatibility
        self.bounds = np.array([0.01, 1.0])
        self.extconf = {"spiral_radius_scale": 1.0, "no_factor_suffix": False}

        self.points, self.points_rgb = self._load_lidar_points(
            sequence_loader, max_lidar_points, lidar_step_frame
        )

        # Normalize the world space (orient, centre, and rescale).
        if self.normalize_world_space:
            self._normalize_world_space()

        # Scene scale: max distance of each camera from the mean camera position.
        # This matches the COLMAP convention (colmap.py:396-400).
        camera_locations = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = float(np.max(dists))

        print(
            f"[NCoreParser] Loaded sequence '{self.sequence_id}': "
            f"{len(self.frame_list)} frames across {self.num_cameras} cameras, "
            f"{len(self.points)} lidar init points, scene_scale={self.scene_scale:.3f}"
        )

    # ------------------------------------------------------------------
    # Private init helpers
    # ------------------------------------------------------------------

    def _open_sequence_loader(self, path: Path) -> ncore.data.SequenceLoaderProtocol:
        """Create an NCore ``SequenceLoaderV4`` for the selected component groups.

        The input ``path`` must point to a single-sequence NCore v4 metadata
        JSON. This method performs a lightweight schema check on the JSON and
        then constructs:

        1. ``SequenceComponentGroupsReader`` over that metadata file, and
        2. ``SequenceLoaderV4`` configured with this parser's poses,
           intrinsics, and masks component-group names.

        In NCore terms, *component groups* select which synchronized data
        stores are used for each modality (e.g., different pose sources or
        calibration variants) while exposing a single loader interface to the
        rest of the parser.

        Args:
            path: Path to a single-sequence NCore v4 metadata JSON file.

        Returns:
            A ``SequenceLoaderProtocol`` instance (backed by
            ``SequenceLoaderV4``) ready to query cameras, lidars, timestamps,
            and pose-graph data.
        """

        assert path.is_file(), f"NCoreParser: path {path} is not a file"
        with open(path, "r") as fp:
            dataset_meta = json.load(fp)
        assert all(
            key in dataset_meta
            for key in (
                "sequence_id",
                "sequence_timestamp_interval_us",
                "version",
                "component_stores",
            )
        ), f"NCoreParser: {path} is not a NCore v4 single-sequence meta-file"

        return ncore.data.v4.SequenceLoaderV4(
            ncore.data.v4.SequenceComponentGroupsReader(
                [path], open_consolidated=self.open_consolidated
            ),
            poses_component_group_name=self.poses_component_group,
            intrinsics_component_group_name=self.intrinsics_component_group,
            masks_component_group_name=self.masks_component_group,
        )

    def _resolve_sensor_ids(
        self,
        sequence_loader: ncore.data.SequenceLoaderProtocol,
        camera_ids: Optional[List[str]],
        lidar_ids: Optional[List[str]],
    ) -> None:
        """Resolve and validate which camera and lidar sensors to use.

        This helper accepts optional user-provided sensor ID lists and falls
        back to NCore auto-detection when those lists are omitted. To avoid
        ambiguous training setups, auto-detection is only allowed when exactly
        one sensor exists for that modality; otherwise the caller must provide
        explicit IDs.

        The selected IDs are validated against the sensors exposed by the
        ``SequenceLoaderProtocol`` and then stored on the parser as
        ``self.camera_ids``, ``self.lidar_ids``, and ``self.num_cameras``.

        Args:
            sequence_loader: Open NCore sequence loader used to query available
                camera and lidar sensor IDs.
            camera_ids: Optional list of camera sensor IDs requested by the
                user/config.
            lidar_ids: Optional list of lidar sensor IDs requested by the
                user/config.

        Raises:
            ValueError: If auto-detection is requested but multiple sensors are
                available for a modality.
            AssertionError: If any requested ID is not present in the dataset.
        """

        # Auto-detect _single_ sensors if not specified - sensors need to be specified explicitly
        # to avoid ambiguity (e.g., in case of multiple downscaled sensors)
        if not camera_ids:
            camera_ids = sequence_loader.camera_ids

            if len(camera_ids) > 1:
                raise ValueError(
                    "NCoreParser: Multiple camera sensors in dataset, explicit"
                    f" specification of a (subset) of camera sensors required to avoid ambiguity: {camera_ids}"
                )

            print(f"[NCoreParser] Auto-detected cameras: {camera_ids}")
        if not lidar_ids:
            lidar_ids = sequence_loader.lidar_ids

            if len(lidar_ids) > 1:
                raise ValueError(
                    "NCoreParser: Multiple lidar sensors in dataset, explicit"
                    f" specification of a (subset) of lidar sensors required to avoid ambiguity: {lidar_ids}"
                )

            print(f"[NCoreParser] Auto-detected lidars: {lidar_ids}")

        assert all(
            cid in sequence_loader.camera_ids for cid in camera_ids
        ), f"NCoreParser: some specified camera_ids {camera_ids} not found in dataset cameras {sequence_loader.camera_ids}"
        assert all(
            lid in sequence_loader.lidar_ids for lid in lidar_ids
        ), f"NCoreParser: some specified lidar_ids {lidar_ids} not found in dataset lidars {sequence_loader.lidar_ids}"

        self.camera_ids: List[str] = list(camera_ids)
        self.lidar_ids: List[str] = list(lidar_ids)
        self.num_cameras: int = len(self.camera_ids)

        print(f"[NCoreParser] Using cameras: {self.camera_ids}")
        print(f"[NCoreParser] Using lidars: {self.lidar_ids}")

    def _compute_world_global_transform(
        self, sequence_loader: ncore.data.SequenceLoaderProtocol
    ) -> None:
        """Compute the transform from NCore ``world`` into ``world_global``.

        NCore pose graphs may provide an explicit edge between the local
        ``world`` frame and a globally-aligned ``world_global`` frame. When
        that edge exists, this method derives ``self.T_world_to_scene_world``
        from it (using the inverse of the stored source-target transform) so
        subsequent poses can be expressed in the global frame. If the edge is
        missing, identity is used as a safe fallback, meaning local and global
        frames are treated as equivalent.

        Args:
            sequence_loader: NCore sequence loader whose pose graph is queried
                for the ``("world", "world_global")`` edge.
        """
        if (
            edge := sequence_loader.pose_graph.get_edge("world", "world_global")
        ) is not None:
            self.T_world_to_scene_world: np.ndarray = np.linalg.inv(
                edge.T_source_target
            ).astype(np.float32)
        else:
            self.T_world_to_scene_world = np.eye(4, dtype=np.float32)

    def _load_camera_data(
        self,
        sequence_loader: ncore.data.SequenceLoaderProtocol,
        factor: float,
        n_dilation: int,
    ) -> Dict[str, ncore.data.CameraSensorProtocol]:
        """Load camera sensors and derive per-camera render metadata.

        For each selected camera ID, this method:

        1. Fetches the NCore camera sensor object from ``sequence_loader``.
        2. Optionally rescales model parameters using ``factor``.
        3. Builds a camera model object for resolution access.
        4. Populates parser dictionaries used downstream by training:
                ``Ks_dict``, ``imsize_dict``, ``mask_dict``, and
                ``camera_render_data``.

        Camera-model-specific handling:

        - ``FThetaCameraModelParameters``:
            Stores a minimal intrinsic matrix with principal point and exports
            full f-theta polynomial coefficients in
            ``CameraRenderData.ftheta_coeffs``.
        - ``OpenCVFisheyeCameraModelParameters``:
            Builds pinhole-style ``K`` via :func:`_build_pinhole_K` and stores
            fisheye radial coefficients.
        - ``OpenCVPinholeCameraModelParameters``:
            Builds ``K`` and stores optional radial, tangential, and thin-prism
            coefficients (cleaned by :func:`_parse_optional_coeffs`).
        - Unknown camera types:
            Falls back to a synthesized pinhole ``K`` from image resolution and
            no distortion coefficients.

        Ego masks are loaded via :func:`_load_ego_mask` and dilated by
        ``n_dilation`` iterations.

        Args:
                sequence_loader: Open NCore sequence loader used to access camera
                        sensors and their model parameters.
                factor: Image-domain scale factor applied to camera model
                        parameters before intrinsics extraction.
                n_dilation: Number of dilation iterations for ego-mask loading.

        Returns:
                A mapping ``camera_id -> CameraSensorProtocol`` for all selected
                cameras.
        """
        camera_sensors = {
            cid: sequence_loader.get_camera_sensor(cid) for cid in self.camera_ids
        }
        self.Ks_dict: Dict[str, np.ndarray] = {}
        self.imsize_dict: Dict[str, Tuple[int, int]] = {}
        self.mask_dict: Dict[str, Optional[np.ndarray]] = {}
        self.camera_models: Dict[str, ncore.sensors.CameraModel] = {}
        self.camera_render_data: Dict[str, CameraRenderData] = {}

        for camera_id in self.camera_ids:
            sensor = camera_sensors[camera_id]
            model_params = sensor.model_parameters
            if factor != 1.0:
                try:
                    model_params = model_params.transform(image_domain_scale=factor)
                except (AssertionError, ValueError) as e:
                    print(
                        f"[NCoreParser] Error: factor={factor} produces non-integer "
                        f"resolution for {camera_id}; using factor=1.0 (full resolution). "
                        "Pass --data-factor 1 to suppress this error."
                    )
                    raise e

            camera_model = ncore.sensors.CameraModel.from_parameters(
                model_params, device="cpu", dtype=torch.float32
            )
            self.camera_models[camera_id] = camera_model

            width = int(camera_model.resolution[0].item())
            height = int(camera_model.resolution[1].item())
            self.imsize_dict[camera_id] = (width, height)

            if isinstance(model_params, ncore.data.FThetaCameraModelParameters):
                cx = float(model_params.principal_point[0].item())
                cy = float(model_params.principal_point[1].item())
                self.Ks_dict[camera_id] = np.array(
                    [[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], dtype=np.float32
                )
                ref_poly = FThetaPolynomialType[model_params.reference_poly.name]
                ftheta_coeffs = FThetaCameraDistortionParameters(
                    reference_poly=ref_poly,
                    pixeldist_to_angle_poly=tuple(
                        float(x) for x in model_params.pixeldist_to_angle_poly
                    ),
                    angle_to_pixeldist_poly=tuple(
                        float(x) for x in model_params.angle_to_pixeldist_poly
                    ),
                    max_angle=float(model_params.max_angle),
                    linear_cde=tuple(float(x) for x in model_params.linear_cde),
                )
                self.camera_render_data[camera_id] = CameraRenderData(
                    camera_model="ftheta",
                    ftheta_coeffs=ftheta_coeffs,
                    radial_coeffs=None,
                    tangential_coeffs=None,
                    thin_prism_coeffs=None,
                )
                print(f"[NCoreParser] {camera_id}: {width}x{height} (ftheta)")
            elif isinstance(
                model_params, ncore.data.OpenCVFisheyeCameraModelParameters
            ):
                self.Ks_dict[camera_id] = _build_pinhole_K(model_params)
                self.camera_render_data[camera_id] = CameraRenderData(
                    camera_model="fisheye",
                    ftheta_coeffs=None,
                    radial_coeffs=np.array(
                        model_params.radial_coeffs, dtype=np.float32
                    ),
                    tangential_coeffs=None,
                    thin_prism_coeffs=None,
                )
                print(f"[NCoreParser] {camera_id}: {width}x{height} (opencv_fisheye)")
            elif isinstance(
                model_params, ncore.data.OpenCVPinholeCameraModelParameters
            ):
                self.Ks_dict[camera_id] = _build_pinhole_K(model_params)
                self.camera_render_data[camera_id] = CameraRenderData(
                    camera_model="pinhole",
                    ftheta_coeffs=None,
                    radial_coeffs=_parse_optional_coeffs(
                        getattr(model_params, "radial_coeffs", None)
                    ),
                    tangential_coeffs=_parse_optional_coeffs(
                        getattr(model_params, "tangential_coeffs", None)
                    ),
                    thin_prism_coeffs=_parse_optional_coeffs(
                        getattr(model_params, "thin_prism_coeffs", None)
                    ),
                )
                print(f"[NCoreParser] {camera_id}: {width}x{height} (opencv_pinhole)")
            else:
                # Unknown camera type: synthesize K from resolution, treat as perfect pinhole.
                print(
                    f"[NCoreParser] {camera_id}: {width}x{height} (unknown, synthesizing K from resolution)"
                )
                self.Ks_dict[camera_id] = np.array(
                    [
                        [float(width), 0.0, width / 2.0],
                        [0.0, float(width), height / 2.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                )
                self.camera_render_data[camera_id] = CameraRenderData(
                    camera_model="pinhole",
                    ftheta_coeffs=None,
                    radial_coeffs=None,
                    tangential_coeffs=None,
                    thin_prism_coeffs=None,
                )

            self.mask_dict[camera_id] = _load_ego_mask(sensor, n_dilation)

        return camera_sensors

    def _get_sensor_frame_range(self, frames_timestamps_us: np.ndarray) -> range:
        """Compute the contiguous frame-index range that overlaps the parser time window.

        NCore sensors store per-frame timestamps for both ``START`` and
        ``END`` timepoints. This helper first asks ``self.time_range_us`` for a
        coarse coverage range using END timestamps, then refines the left
        boundary with START timestamps so returned frames begin at or after the
        requested start time.

        This produces a Python ``range`` over frame indices that can be used
        directly with NumPy slicing and NCore frame-access APIs.

        Args:
            frames_timestamps_us: Array of shape ``(N, 2)`` containing START
                and END timestamps in microseconds for each frame.

        Returns:
            A contiguous ``range`` of valid frame indices inside
            ``self.time_range_us``. May be empty when no frames overlap.
        """
        cover = self.time_range_us.cover_range(
            frames_timestamps_us[:, ncore.data.FrameTimepoint.END]
        )
        if not len(cover):
            return cover
        start_ts = frames_timestamps_us[
            cover.start : cover.stop, ncore.data.FrameTimepoint.START
        ]
        first_valid = int(
            np.searchsorted(start_ts, self.time_range_us.start, side="left")
        )
        return range(cover.start + first_valid, cover.stop)

    def _compute_scene_origin(
        self,
        camera_sensors: Dict[str, ncore.data.CameraSensorProtocol],
        camera_frame_ranges: Dict[str, range],
    ) -> None:
        """Estimate and store a scene-centering transform from camera trajectories.

        This method gathers START-time camera positions for all selected
        cameras within their valid frame ranges, maps those positions from
        NCore ``world`` into ``world_global`` using
        ``self.T_world_to_scene_world``, and computes the mean 3D position
        across all samples.

        That mean position is then used as the target origin for
        ``self.world_global_to_scene`` (a :class:`FrameConversion`), which
        recenters the scene so camera poses are numerically close to the
        origin during training.

        Args:
            camera_sensors: Mapping from camera ID to NCore camera sensor
                object used to query pose trajectories.
            camera_frame_ranges: Mapping from camera ID to the frame-index
                range selected for that camera.
        """
        positions: List[np.ndarray] = []
        for camera_id in self.camera_ids:
            frame_range = camera_frame_ranges[camera_id]
            if not len(frame_range):
                continue
            T_cam_world = camera_sensors[camera_id].get_frames_T_source_target(
                source_node=camera_id,
                target_node="world",
                frame_indices=np.arange(frame_range.start, frame_range.stop),
                frame_timepoint=ncore.data.FrameTimepoint.START,
            )  # [N, 4, 4]
            cam_positions = T_cam_world[:, :3, 3]
            positions.append(
                (
                    self.T_world_to_scene_world[:3, :3] @ cam_positions.T
                    + self.T_world_to_scene_world[:3, 3:4]
                ).T
            )

        mean_position = np.vstack(positions).mean(axis=0).astype(np.float32)
        self.world_global_to_scene = FrameConversion.from_origin_scale_axis(
            target_origin=mean_position,
            target_scale=1.0,
            target_axis=[0, 1, 2],
        )

    def _load_poses(
        self,
        camera_sensors: Dict[str, ncore.data.CameraSensorProtocol],
        camera_frame_ranges: Dict[str, range],
    ) -> None:
        """Batch-load per-frame camera poses and build parser frame indexing.

        For each selected camera, this method loads both START and END
        timepoint poses from NCore (camera -> world), converts them into the
        parser's scene frame via :meth:`_ncore_world_to_scene_poses`, and
        appends the results into flat, multi-camera arrays.

        It also builds two indexing structures used by :class:`NCoreDataset`:

        - ``self.frame_list``: ``[(camera_id, frame_idx), ...]``
        - ``self.camera_idx_per_frame``: integer camera index per flattened
          frame.

        Finally, START and END poses are stacked into:

        - ``self.camtoworlds`` with shape ``(N, 4, 4)``
        - ``self.camtoworlds_end`` with shape ``(N, 4, 4)``

        where ``N`` is the total number of retained frames across cameras.

        Args:
            camera_sensors: Mapping from camera ID to NCore camera sensor
                objects used to query pose trajectories.
            camera_frame_ranges: Mapping from camera ID to selected frame
                index ranges.
        """
        self.frame_list: List[Tuple[str, int]] = []
        self.camera_idx_per_frame: List[int] = []
        starts: List[np.ndarray] = []
        ends: List[np.ndarray] = []

        for cam_idx, camera_id in enumerate(self.camera_ids):
            frame_range = camera_frame_ranges[camera_id]
            if not len(frame_range):
                continue

            sensor = camera_sensors[camera_id]
            indices = np.arange(frame_range.start, frame_range.stop)
            T_start = self._ncore_world_to_scene_poses(
                sensor.get_frames_T_source_target(
                    source_node=camera_id,
                    target_node="world",
                    frame_indices=indices,
                    frame_timepoint=ncore.data.FrameTimepoint.START,
                ).reshape(-1, 4, 4)
            )  # [N, 4, 4]
            T_end = self._ncore_world_to_scene_poses(
                sensor.get_frames_T_source_target(
                    source_node=camera_id,
                    target_node="world",
                    frame_indices=indices,
                    frame_timepoint=ncore.data.FrameTimepoint.END,
                ).reshape(-1, 4, 4)
            )  # [N, 4, 4]

            # squeeze() in transform_poses may drop the batch dim when N=1
            if T_start.ndim == 2:
                T_start = T_start[np.newaxis]
            if T_end.ndim == 2:
                T_end = T_end[np.newaxis]

            for local_idx, frame_idx in enumerate(frame_range):
                self.frame_list.append((camera_id, frame_idx))
                self.camera_idx_per_frame.append(cam_idx)
                starts.append(T_start[local_idx])
                ends.append(T_end[local_idx])

        self.camtoworlds = np.stack(starts, axis=0)  # (N, 4, 4)
        self.camtoworlds_end = np.stack(ends, axis=0)  # (N, 4, 4)

    def _normalize_world_space(self) -> None:
        """Normalize world-space coordinates for poses and points.

        Three successive transforms are applied:
        1. ``similarity_from_cameras`` - rotate so z+ is the up axis, recenter
           at the camera focus point, and rescale by 1/median camera distance.
        2. ``align_principal_axes`` - PCA rotation that aligns the point-cloud
           principal axes to the coordinate axes.
        3. Upside-down fix - if the point cloud is inverted (median z > mean z),
           apply a 180° rotation around the x-axis.

        Operates on ``self.camtoworlds``, ``self.camtoworlds_end``, and
        ``self.points`` in-place and stores the composed transform in
        ``self.transform``.
        """
        # Ensure float64 for numerical precision during normalization.
        camtoworlds = self.camtoworlds.astype(np.float64)
        camtoworlds_end = self.camtoworlds_end.astype(np.float64)
        points = self.points.astype(np.float64) if len(self.points) else self.points

        T1 = similarity_from_cameras(camtoworlds)
        camtoworlds = transform_cameras(T1, camtoworlds)
        camtoworlds_end = transform_cameras(T1, camtoworlds_end)
        if len(points):
            points = transform_points(T1, points)

        if len(points):
            T2 = align_principal_axes(points)
        else:
            T2 = np.eye(4)
        camtoworlds = transform_cameras(T2, camtoworlds)
        camtoworlds_end = transform_cameras(T2, camtoworlds_end)
        if len(points):
            points = transform_points(T2, points)

        transform = T2 @ T1

        # Upside-down fix: if median z > mean z, flip around x-axis.
        if len(points) and np.median(points[:, 2]) > np.mean(points[:, 2]):
            T3 = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            camtoworlds = transform_cameras(T3, camtoworlds)
            camtoworlds_end = transform_cameras(T3, camtoworlds_end)
            points = transform_points(T3, points)
            transform = T3 @ transform

        self.camtoworlds = camtoworlds
        self.camtoworlds_end = camtoworlds_end
        if len(self.points):
            self.points = points.astype(np.float32)
        self.transform = transform

    # ------------------------------------------------------------------
    # Private runtime helpers
    # ------------------------------------------------------------------

    def _ncore_world_to_scene_poses(self, T_poses_world: np.ndarray) -> np.ndarray:
        """Transform poses from NCore world frame to scene frame."""
        T_poses_common = self.T_world_to_scene_world @ T_poses_world.reshape(-1, 4, 4)
        return self.world_global_to_scene.transform_poses(T_poses_common)

    def _load_lidar_points(
        self,
        sequence_loader: ncore.data.SequenceLoaderProtocol,
        max_points: int,
        step_frame: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load and transform lidar points to scene frame for Gaussian initialisation."""
        if not self.lidar_ids:
            print(
                "[NCoreParser] No lidar sensors available; using empty init point cloud"
            )
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        lidar_id = self.lidar_ids[0]
        lidar_sensor = sequence_loader.get_lidar_sensor(lidar_id)
        lidar_frame_range = self.time_range_us.cover_range(
            lidar_sensor.get_frames_timestamps_us()
        )

        all_points: List[np.ndarray] = []
        all_colors: List[np.ndarray] = []
        for lidar_frame_idx in lidar_frame_range[::step_frame]:
            try:
                pc = lidar_sensor.get_frame_point_cloud(
                    frame_index=lidar_frame_idx,
                    motion_compensation=True,
                    with_start_points=True,
                    return_index=0,
                )
            except Exception as exc:
                print(
                    f"[NCoreParser] Warning: failed to load lidar frame "
                    f"{lidar_frame_idx}: {exc}"
                )
                continue

            xyz = pc.xyz_m_end
            color: Optional[np.ndarray] = None
            if lidar_sensor.has_frame_generic_data(
                lidar_frame_idx, self.lidar_color_generic_data_name
            ):
                color = lidar_sensor.get_frame_generic_data(
                    lidar_frame_idx, self.lidar_color_generic_data_name
                )
                if color.shape != xyz.shape:
                    raise ValueError(
                        "Color data length does not match point cloud length "
                        "(expecting 3-channel RGB color per point)"
                    )
                if color.dtype != np.uint8:
                    raise ValueError("Expected color data in uint8 format")

            point_filter = ...
            if lidar_sensor.has_frame_generic_data(lidar_frame_idx, "dynamic_flag"):
                point_filter = (
                    lidar_sensor.get_frame_generic_data(lidar_frame_idx, "dynamic_flag")
                    != 1
                )
            xyz = xyz[point_filter]
            if color is not None:
                color = color[point_filter]
            if not len(xyz):
                continue

            T_sensor_scene = self._ncore_world_to_scene_poses(
                lidar_sensor.get_frames_T_sensor_target("world", lidar_frame_idx)
            )
            xyz_scene = (
                (self.world_global_to_scene.target_scale * T_sensor_scene[:3, :3])
                @ xyz.T
                + T_sensor_scene[:3, 3:4]
            ).T
            all_points.append(xyz_scene.astype(np.float32))
            if color is not None:
                all_colors.append(color)
            else:
                all_colors.append(np.full((len(xyz_scene), 3), 128, dtype=np.uint8))

        if not all_points:
            print("[NCoreParser] Warning: no lidar points loaded")
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        points = np.vstack(all_points)
        points_rgb = np.vstack(all_colors)
        if len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]
            points_rgb = points_rgb[idx]

        print(f"[NCoreParser] Loaded {len(points)} lidar points from '{lidar_id}'")
        return points, points_rgb


# ---------------------------------------------------------------------------
# NCoreDataset
# ---------------------------------------------------------------------------


class NCoreDataset(torch.utils.data.Dataset):
    """Image-based dataset for NCore v4 sequences.

    Returns batches compatible with gsplat trainers:
      {"K", "camtoworld", "image", "image_id", "camera_idx"}
    plus optional "camtoworld_end" (END pose for rolling shutter; camtoworld would be the START pose) and "mask".

    Images are loaded lazily per __getitem__. The underlying NCore sequence
    loader is (re-)opened per DataLoader worker to avoid sharing file handles.
    """

    def __init__(
        self,
        parser: NCoreParser,
        split: str = "train",
    ) -> None:
        self.parser = parser
        self.split = split

        # Build train/val split indices over the flat frame list.
        all_indices = np.arange(len(parser.frame_list))
        if split == "train":
            self.indices = all_indices[all_indices % parser.test_every != 0]
        else:
            self.indices = all_indices[all_indices % parser.test_every == 0]

        # Per-worker sequence loader (lazily initialised).
        self._sequence_loader: Optional[ncore.data.SequenceLoaderProtocol] = None
        self._camera_sensors: Optional[
            Dict[str, ncore.data.CameraSensorProtocol]
        ] = None
        self._current_worker_id: Optional[int] = None

    def _init_worker(self) -> None:
        """Ensure this DataLoader worker has its own valid NCore loader resources.

        PyTorch DataLoader workers run in separate processes, and NCore file
        handles/readers should not be shared across workers. This method
        lazily initializes a worker-local ``SequenceLoaderV4`` on first use,
        caches the current worker ID, and reuses the loader when the worker is
        unchanged.

        If a loader already exists but the worker ID differs (e.g., after
        worker context changes), it refreshes underlying resources via
        ``reload_resources()``.

        Side effects:
            - Sets ``self._current_worker_id``.
            - Initializes or refreshes ``self._sequence_loader``.
            - Populates ``self._camera_sensors`` when first created.
        """
        worker_info = torch.utils.data.get_worker_info() # get worker info to determine if we're in a worker process
        current_worker_id: Optional[int] = (
            None if worker_info is None else worker_info.id
        )

        if self._sequence_loader is not None:
            if self._current_worker_id == current_worker_id:
                return  # already initialised for this worker
            # Worker ID changed: reload file handles.
            self._current_worker_id = current_worker_id
            self._sequence_loader.reload_resources()
            return

        # First-time initialisation for this process.
        self._current_worker_id = current_worker_id
        self._sequence_loader = ncore.data.v4.SequenceLoaderV4(
            ncore.data.v4.SequenceComponentGroupsReader(
                [self.parser.sequence_meta_file_path],
                open_consolidated=self.parser.open_consolidated,
            ),
            poses_component_group_name=self.parser.poses_component_group,
            intrinsics_component_group_name=self.parser.intrinsics_component_group,
            masks_component_group_name=self.parser.masks_component_group,
        )
        self._camera_sensors = {
            cid: self._sequence_loader.get_camera_sensor(cid)
            for cid in self.parser.camera_ids
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Build one training/evaluation sample from the flattened frame index.

        The input ``item`` indexes this split-local list (train or test).
        It is first mapped to the parser-global frame index via
        ``self.indices``, then resolved to ``(camera_id, frame_idx)`` in
        ``parser.frame_list``.

        For that frame, this method loads:

        - image pixels from the NCore camera sensor,
        - intrinsic matrix ``K`` from parser metadata,
        - START and END camera poses (``camtoworld`` and
            ``camtoworld_end``) for rolling-shutter-aware rendering,
        - camera/sample identifiers.

        Optional masks are merged into a single boolean validity mask:

        - static ego mask from parser calibration metadata (inverted so
            ``True`` means valid pixel),
        - per-frame generic ``"mask"`` from NCore, when present.

        If both are available, the final mask is their logical AND.

        Args:
                item: Split-local sample index in ``[0, len(self))``.

        Returns:
                Dictionary with tensor-valued camera/image fields compatible with
                gsplat training:
                ``{"K", "camtoworld", "camtoworld_end", "image", "image_id", "camera_idx"}``
                and optional ``"mask"`` when any valid-mask source exists.
        """
        self._init_worker()
        assert self._camera_sensors is not None

        index = self.indices[item]
        camera_id, frame_idx = self.parser.frame_list[index]
        camera_idx = self.parser.camera_idx_per_frame[index]

        sensor = self._camera_sensors[camera_id]
        width, height = self.parser.imsize_dict[camera_id]
        K = self.parser.Ks_dict[camera_id].copy()

        image = sensor.get_frame_image_array(frame_idx)  # HxWx3 uint8
        if self.parser.factor != 1.0:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

        data: Dict[str, Any] = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(self.parser.camtoworlds[index]).float(),
            "camtoworld_end": torch.from_numpy(
                self.parser.camtoworlds_end[index]
            ).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,
            "camera_idx": camera_idx,
        }

        valid_mask: Optional[np.ndarray] = None

        # static ego mask, if present
        ego_mask = self.parser.mask_dict.get(camera_id)
        if ego_mask is not None:
            valid_mask = (~ego_mask).astype(bool)  # True = valid pixel
            if valid_mask.shape != (height, width):
                valid_mask = cv2.resize(
                    valid_mask.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

        # per-frame mask, if present
        if sensor.has_frame_generic_data(frame_idx, "mask"):
            frame_mask_raw = sensor.get_frame_generic_data(frame_idx, "mask")
            frame_mask = np.asarray(frame_mask_raw)
            if frame_mask.ndim == 3 and frame_mask.shape[-1] == 1:
                frame_mask = frame_mask[..., 0]
            # assumption: True/non-zero values in generic "mask" indicate valid pixels
            frame_mask = np.squeeze(frame_mask).astype(bool)
            if frame_mask.shape != (height, width):
                frame_mask = cv2.resize(
                    frame_mask.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            # merge with ego mask if present
            valid_mask = frame_mask if valid_mask is None else (valid_mask & frame_mask)

        if valid_mask is not None:
            data["mask"] = torch.from_numpy(valid_mask).bool()

        return data
