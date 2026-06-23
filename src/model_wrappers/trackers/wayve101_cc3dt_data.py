"""Wayve101-specific CC-3DT data helpers."""
from __future__ import annotations

from collections.abc import Sequence

from vis4d.config import class_config
from vis4d.config.typing import DataConfig
from vis4d.data.const import CommonKeys as K
from vis4d.data.data_pipe import DataPipe
from vis4d.data.loader import multi_sensor_collate
from vis4d.data.transforms import compose
from vis4d.data.transforms.normalize import NormalizeImages
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.resize import (
    GenResizeParameters,
    ResizeImages,
    ResizeIntrinsics,
)
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.engine.connectors import data_key, pred_key
from vis4d.zoo.base import get_inference_dataloaders_cfg

from src.model_wrappers.trackers.wayve101_cc3dt_dataset import Wayve101CC3DTDataset

WAYVE101_CAMERAS = [
    "front-forward",
    "left-backward",
    "left-forward",
    "right-backward",
    "right-forward",
]

CONN_WAYVE101_BBOX_3D_TEST = {
    "images": data_key(K.images, sensors=WAYVE101_CAMERAS),
    "images_hw": data_key(K.input_hw, sensors=WAYVE101_CAMERAS),
    "intrinsics": data_key(K.intrinsics, sensors=WAYVE101_CAMERAS),
    "extrinsics": data_key(K.extrinsics, sensors=WAYVE101_CAMERAS),
    "frame_ids": K.frame_ids,
}

CONN_WAYVE101_BBOX_3D_VIS = {
    "images": data_key(K.images, sensors=WAYVE101_CAMERAS),
    "image_names": data_key(K.sample_names, sensors=WAYVE101_CAMERAS),
    "boxes3d": pred_key("boxes_3d"),
    "intrinsics": data_key(K.intrinsics, sensors=WAYVE101_CAMERAS),
    "extrinsics": data_key(K.extrinsics, sensors=WAYVE101_CAMERAS),
    "scores": pred_key("scores_3d"),
    "class_ids": pred_key("class_ids"),
    "track_ids": pred_key("track_ids"),
    "sequence_names": data_key(K.sequence_names),
}

def get_wayve101_bev_vis_conn(reference_camera: str) -> dict[str, object]:
    """Build the BEV visualizer connector for the reference camera."""
    return {
        "sample_names": data_key(K.sample_names, sensors=[reference_camera]),
        "boxes3d": pred_key("boxes_3d"),
        "extrinsics": data_key("bev_extrinsics", sensors=[reference_camera]),
        "track_ids": pred_key("track_ids"),
        "sequence_names": data_key(K.sequence_names),
    }


def get_wayve101_test_dataloader(
    data_root: str,
    cameras: Sequence[str] = WAYVE101_CAMERAS,
    reference_camera: str = "front-forward",
    image_size: tuple[int, int] = (900, 1600),
    samples_per_gpu: int = 1,
    workers_per_gpu: int = 1,
    data_backend=None,
    undistort: bool = True,
    undistort_balance: float = 0.0,
    undistort_mode: str = "balance",
    virtual_focal_px: float = 1266.0,
):
    """Build the inference dataloader used by CC-3DT on Wayve101."""
    test_transforms = [
        class_config(
            GenResizeParameters,
            shape=image_size,
            keep_ratio=True,
            sensors=list(cameras),
        ),
        class_config(ResizeImages, sensors=list(cameras)),
        class_config(ResizeIntrinsics, sensors=list(cameras)),
    ]

    test_preprocess_cfg = class_config(compose, transforms=test_transforms)
    test_batchprocess_cfg = class_config(
        compose,
        transforms=[
            class_config(PadImages, sensors=list(cameras)),
            class_config(NormalizeImages, sensors=list(cameras)),
            class_config(ToTensor, sensors=list(cameras)),
        ],
    )

    test_dataset_cfg = class_config(
        DataPipe,
        datasets=class_config(
            Wayve101CC3DTDataset,
            data_root=data_root,
            cameras=list(cameras),
            reference_camera=reference_camera,
            keys_to_load=(K.images, K.original_images),
            data_backend=data_backend,
            undistort=undistort,
            undistort_balance=undistort_balance,
            undistort_mode=undistort_mode,
            virtual_focal_px=virtual_focal_px,
            image_size=image_size,
        ),
        preprocess_fn=test_preprocess_cfg,
    )

    return get_inference_dataloaders_cfg(
        datasets_cfg=test_dataset_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        video_based_inference=True,
        batchprocess_cfg=test_batchprocess_cfg,
        collate_fn=multi_sensor_collate,
        sensors=list(cameras),
    )


def get_wayve101_data_cfg(
    data_root: str,
    cameras: Sequence[str] = WAYVE101_CAMERAS,
    reference_camera: str = "front-forward",
    image_size: tuple[int, int] = (900, 1600),
    samples_per_gpu: int = 1,
    workers_per_gpu: int = 1,
    data_backend=None,
    undistort: bool = True,
    undistort_balance: float = 0.0,
    undistort_mode: str = "balance",
    virtual_focal_px: float = 1266.0,
) -> DataConfig:
    """Build the minimal DataConfig needed for Wayve101 inference."""
    data = DataConfig()
    data.test_dataloader = get_wayve101_test_dataloader(
        data_root=data_root,
        cameras=cameras,
        reference_camera=reference_camera,
        image_size=image_size,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        data_backend=data_backend,
        undistort=undistort,
        undistort_balance=undistort_balance,
        undistort_mode=undistort_mode,
        virtual_focal_px=virtual_focal_px,
    )
    return data