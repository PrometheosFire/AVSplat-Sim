"""Local CC-3DT R101 visualization config for Wayve101 scene inference."""
from __future__ import annotations

from vis4d.config import class_config
from vis4d.data.io import FileBackend
from vis4d.engine.callbacks import VisualizerCallback
from vis4d.engine.connectors import MultiSensorCallbackConnector
from vis4d.vis.image.bbox3d_visualizer import MultiCameraBBox3DVisualizer
from vis4d.zoo.cc_3dt.cc_3dt_frcnn_r101_fpn_kf3d_24e_nusc import (
    get_config as _base_get_config,
)

from src.model_wrappers.trackers.wayve101_cc3dt_data import (
    CONN_WAYVE101_BBOX_3D_TEST,
    CONN_WAYVE101_BBOX_3D_VIS,
    WAYVE101_CAMERAS,
    get_wayve101_data_cfg,
)


def get_config():
    """Return a minimal R101 CC-3DT visualization config for Wayve101."""
    config = _base_get_config()
    config.experiment_name = "cc_3dt_r101_wayve101_vis"
    config.params.samples_per_gpu = 1
    config.params.workers_per_gpu = 1

    data_root = "/home/dinis-matos/workspace/AVSplat-Sim/data/wayve101/scene_001"
    config.data = get_wayve101_data_cfg(
        data_root=data_root,
        cameras=WAYVE101_CAMERAS,
        reference_camera="front-forward",
        samples_per_gpu=1,
        workers_per_gpu=1,
        data_backend=class_config(FileBackend),
    )

    config.test_data_connector = class_config(
        MultiSensorCallbackConnector,
        key_mapping=CONN_WAYVE101_BBOX_3D_TEST,
    )

    config.callbacks = [
        class_config(
            VisualizerCallback,
            visualizer=class_config(
                MultiCameraBBox3DVisualizer,
                cat_mapping={
                    "bicycle": 0,
                    "motorcycle": 1,
                    "pedestrian": 2,
                    "bus": 3,
                    "car": 4,
                    "trailer": 5,
                    "truck": 6,
                    "construction_vehicle": 7,
                    "traffic_cone": 8,
                    "barrier": 9,
                },
                width=2,
                camera_near_clip=0.15,
                cameras=WAYVE101_CAMERAS,
                vis_freq=1,
            ),
            output_dir=config.output_dir,
            save_prefix="boxes3d",
            test_connector=class_config(
                MultiSensorCallbackConnector,
                key_mapping=CONN_WAYVE101_BBOX_3D_VIS,
            ),
        )
    ]

    return config