# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from pycolmap import SceneManager
from tqdm import tqdm
from typing_extensions import assert_never

from exif import compute_exposure_from_exif
from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Return all file paths under a directory, relative to that directory.

    The directory is walked recursively, and every file discovered anywhere
    under ``path_dir`` is returned as a path relative to ``path_dir``.

    Args:
        path_dir: Root directory to scan.

    Returns:
        A list of relative file paths using the platform path separator.
    """
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    """Downscale all images in a folder and save them to a new directory.

    The function scans ``image_dir`` recursively, resizes every image by the
    given factor, and writes the result as PNG files into ``resized_dir`` while
    preserving the relative directory structure.

    Existing resized files are skipped so the operation can be resumed without
    recomputing already processed images.

    Args:
        image_dir: Source directory containing the original images.
        resized_dir: Target directory for the resized PNG images.
        factor: Downscaling factor applied to width and height.

    Returns:
        The path to ``resized_dir``.
    """
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png" #save as PNG to preserve quality (lossless compression)
        )
        if os.path.isfile(resized_path):
            continue
        image = imageio.imread(image_path)[..., :3]
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor)),
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
        imageio.imwrite(resized_path, resized_image)
    return resized_dir


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        load_exposure: bool = False,
    ):
        """Load a COLMAP scene, image metadata, and optional exposure values.

        The initializer reads the COLMAP reconstruction, camera intrinsics,
        image list, and 3D points, then prepares the data needed by the
        training dataset:

        - per-image camera poses in camera-to-world form,
        - per-camera intrinsics and distortion parameters,
        - image paths aligned with COLMAP image names,
        - optional world-space normalization,
        - optional EXIF exposure values.

        The ``factor`` argument controls the image scale used by the dataset.
        When greater than 1, the parser looks for a matching downsampled image
        directory and adjusts intrinsics accordingly. The ``normalize`` flag
        recenters and reorients the reconstruction into a more stable world
        frame. The ``test_every`` value defines the train/test split pattern
        used by :class:`Dataset`. If ``load_exposure`` is enabled, exposure is
        read from the original JPEG images and stored relative to the dataset
        mean.

        Args:
            data_dir: Root directory of the COLMAP dataset.
            factor: Image downscaling factor.
            normalize: Whether to normalize camera poses and 3D points.
            test_every: Hold out every Nth image for the test split.
            load_exposure: Whether to extract EXIF exposure metadata.
        """
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.load_exposure = load_exposure

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        # pycolmap is a Python wrapper for reading COLMAP reconstructions. 
        # It provides convenient access to cameras, images, and 3D points stored in COLMAP's binary format. 
        # We use it to load the scene metadata needed for training.
        manager = SceneManager(colmap_dir) 
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = [] 
        Ks_dict = dict() # camera_id -> intrinsic matrix K
        params_dict = dict() # camera_id -> distortion parameters (empty if no distortion)
        imsize_dict = dict()  # camera_id -> width, height
        mask_dict = dict() # camera_id -> optional binary mask (I think this mask is ROI Mask, used for to undistort images, not to mask out parts of the image during training)
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c) # Build world-to-camera matrices 

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]) # Build intrinsic matrix K
            K[:2, :] /= factor # Adjust intrinsics if images are downsampled by factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor) # Adjust image size if downsampled by factor
            mask_dict[camera_id] = None     # TODO support proper Masks
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0) # Stack world-to-camera matrices into a single array of shape (num_images, 4, 4)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}" # images_2, images_4, etc. 
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        if factor > 1 and os.path.splitext(image_files[0])[1].lower() in {".jpg", ".jpeg", ".png",}:
            image_dir = _resize_image_folder(
                colmap_image_dir, image_dir + "_png", factor=factor
            )
            image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()} # invert COLMAP's image name to ID mapping for easy lookup
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        } # convert lists of point indices to numpy arrays for efficient indexing

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principal_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1

            # Fix for up side down. We assume more points towards
            # the bottom of the scene which is true when ground floor is
            # present in the images.
            # PCA-like alignment, axis sign is still ambiguous, 
            # so we check the depth distribution to see if flipping is needed. 
            if np.median(points[:, 2]) > np.mean(points[:, 2]):
                # rotate 180 degrees around x axis such that z is flipped
                T3 = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                camtoworlds = transform_cameras(T3, camtoworlds)
                points = transform_points(T3, points)
                transform = T3 @ transform
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # Create 0-based contiguous camera indices from COLMAP camera_ids.
        # This is useful for camera-based embeddings/modules. (0, 1, 2, ... instead of arbitrary camera IDs like 3, 5, 7)
        unique_camera_ids = sorted(set(camera_ids))
        self.camera_id_to_idx = {cid: idx for idx, cid in enumerate(unique_camera_ids)}
        self.camera_indices = [self.camera_id_to_idx[cid] for cid in camera_ids]
        self.num_cameras = len(unique_camera_ids)

        # Load EXIF exposure data if requested.
        # Always read from original (non-downscaled) images since PNG doesn't support EXIF.
        if load_exposure:
            exposure_values: List[Optional[float]] = []
            for image_name in tqdm(image_names, desc="Loading EXIF exposure"):
                original_path = Path(colmap_image_dir) / image_name
                exposure_values.append(compute_exposure_from_exif(original_path))

            # Compute mean across all valid exposures and subtract
            # valid if every exposure value is positive and finite, otherwise invalid.
            valid_exposures = [e for e in exposure_values if e is not None]
            if valid_exposures:
                exposure_mean = sum(valid_exposures) / len(valid_exposures)
                self.exposure_values: List[Optional[float]] = [
                    (e - exposure_mean) if e is not None else None
                    for e in exposure_values
                ]
                print(
                    f"[Parser] Loaded exposure for {len(valid_exposures)}/{len(exposure_values)} images "
                    f"(mean={exposure_mean:.3f} EV)"
                )
            else:
                self.exposure_values = [None] * len(exposure_values)
                print("[Parser] No valid EXIF exposure data found in any image.")
        else:
            self.exposure_values = [None] * len(image_paths)

        # Specific case, prob won't be needed in general
        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict() #camera_id -> mapx
        self.mapy_dict = dict() #camera_id -> mapy
        self.roi_undist_dict = dict() #camera_id -> roi of the undistorted image (x_min, y_min, width, height)
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":        
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(   # Computes a new camera matrix after undistortion.
                    K, params, (width, height), 0                       # alpha=0 means we want all pixels in the undistorted image to be valid (no black borders),
                                                                        # which may result in some cropping (but keep rectangular image)
                )
                mapx, mapy = cv2.initUndistortRectifyMap(                       # Computes the undistortion and rectification transformation map.
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None # no need for mask since getOptimalNewCameraMatrix already crops the image to valid region
                
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(               # Create a grid of pixel coordinates (x, y) for the original image size.
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx         # Normalize pixel coordinates to camera space (x1, y1) 
                y1 = (grid_y - cy) / fy         # where the principal point is at the origin and focal lengths are 1.
                theta = np.sqrt(x1**2 + y1**2)  # radius from the optical axis, used by the fisheye distortion model
                r = (                           # Fisheye polynomial model: r(theta) = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)    # Map the normalized distorted coordinates back to pixel coordinates in the undistorted image, 
                mapy = (fy * y1 * r + height // 2).astype(np.float32)   # centering the principal point at (width/2, height/2).

                # Use mask to define ROI
                mask = np.logical_and(                                      # Validity mask for the undistorted image: 
                    np.logical_and(mapx > 0, mapy > 0),                     # only pixels that map back to valid coordinates in the original image are kept.
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)                 # Finds bounding box of the valid region in the undistorted image
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]                   # Crop the mask to the bounding box to save memory and align with the cropped undistorted image.
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min] # ROI = Region of Interest, defined as the bounding box of valid pixels in the undistorted image. 
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists) # Scene sclae is the smalest radius of a sphere centered at the scene center that can enclose all cameras. 
                                         # This is used to normalize the scale of the scene for better training stability.


class Dataset:
    """Dataset wrapper that serves COLMAP images and optional supervision signals.

    The dataset uses the parsed COLMAP metadata to load per-image camera
    parameters, the corresponding image tensors, and optional extras such as
    masks, EXIF exposure offsets, and sparse depth points.
    
    Use case: Call two dataset instances with the same parser but different splits ("train" vs "test") to get separate training and test sets.
    """

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        """Create a dataset view over the parsed COLMAP scene.

        The ``split`` argument selects either the training subset or the held-
        out test subset using the parser's ``test_every`` pattern. When
        ``patch_size`` is set, samples are randomly cropped to that square
        size and the intrinsics are updated accordingly. If ``load_depths`` is
        enabled, each sample also includes sparse 2D points and depths
        projected from COLMAP's reconstructed 3D points.

        Args:
            parser: Parsed COLMAP scene metadata.
            split: Either ``"train"`` or ``"test"``.
            patch_size: Optional square crop size in pixels.
            load_depths: Whether to include sparse depth supervision.
        """
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Build one training sample from the selected image index.

        The method loads the image, applies undistortion and optional random
        cropping, then returns a dictionary with the camera intrinsics,
        camera-to-world pose, image tensor, compact camera index, optional ROI
        mask, optional exposure value, and optional sparse depth supervision.

        Args:
            item: Index within the current split returned by ``__len__``.

        Returns:
            A dictionary containing the sample tensors and metadata needed by
            the training loop.
        """
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3] # Load the image and discard alpha channel if present.
        camera_id = self.parser.camera_ids[index]                       # Get the camera ID for this image to look up intrinsics and other camera-specific data.
        K = self.parser.Ks_dict[camera_id].copy()                       # undistorted K
        params = self.parser.params_dict[camera_id]                     # distortion parameters, empty if no distortion
        camtoworlds = self.parser.camtoworlds[index]                    # camera-to-world pose for this image
        mask = self.parser.mask_dict[camera_id]                         # optional binary mask for this camera, used to crop the undistorted image to valid pixels (only for fisheye cameras in current implementation)

        if len(params) > 0:                             # TODO this does not support 3dGUT !! always applies undistortion if any distortion parameters are present, but 3dGUT uses distortion parameters to represent a non-linear image crop without actual distortion. 
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
            "camera_idx": self.parser.camera_indices[
                index
            ],  # 0-based contiguous camera index
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        # Add exposure if available for this image
        exposure = self.parser.exposure_values[index]
        if exposure is not None:
            data["exposure"] = torch.tensor(exposure, dtype=torch.float32)

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data


if __name__ == "__main__":
    # Small debug/demo entrypoint:
    # 1) load a COLMAP scene,
    # 2) build a training split dataset with sparse depth enabled,
    # 3) render projected sparse points onto each frame and save a video.
    import argparse

    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP metadata (poses, intrinsics, points, image paths).
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    # Build the training subset and request sparse depth projections.
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    # Visualize sparse COLMAP correspondences by drawing point projections
    # on each image and writing the result to a video file.
    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()  # loaded for completeness/debugging
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    # Finalize and flush video to disk.
    writer.close()
