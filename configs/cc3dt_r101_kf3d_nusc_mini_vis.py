"""Local CC-3DT R101 + KF3D visualization config for nuScenes mini."""
from __future__ import annotations

from vis4d.config import class_config
from vis4d.data.io import FileBackend
from vis4d.data.datasets.nuscenes import NuScenes, nuscenes_class_map
from vis4d.engine.callbacks import VisualizerCallback
from vis4d.engine.connectors import MultiSensorCallbackConnector
from vis4d.vis.image.bbox3d_visualizer import MultiCameraBBox3DVisualizer
from vis4d.vis.image.bev_visualizer import BEVBBox3DVisualizer
from vis4d.zoo.cc_3dt.cc_3dt_frcnn_r101_fpn_kf3d_24e_nusc import (
    get_config as _base_get_config,
)
from vis4d.zoo.cc_3dt.data import (
    CONN_NUSC_BBOX_3D_VIS,
    CONN_NUSC_BEV_BBOX_3D_VIS,
    get_nusc_cfg,
)


def get_config():
    """Return a mini visualization config for R101 inference."""
    config = _base_get_config()
    data_root = "/home/dinis-matos/workspace/AVSplat-Sim/data/nuscenes/v1.0-mini"

    config.experiment_name = "cc_3dt_r101_kf3d_nusc_mini_vis"
    config.params.samples_per_gpu = 1
    config.params.workers_per_gpu = 1
    config.data = get_nusc_cfg(
        data_root=data_root,
        version="v1.0-mini",
        train_split="mini_train",
        test_split="mini_val",
        data_backend=class_config(FileBackend),
        samples_per_gpu=1,
        workers_per_gpu=1,
    )

    for callback in config.callbacks:
        evaluator = getattr(callback, "evaluator", None)
        if evaluator is not None and hasattr(evaluator, "data_root"):
            evaluator.data_root = data_root
        if evaluator is not None and hasattr(evaluator, "version"):
            evaluator.version = "v1.0-mini"
        if evaluator is not None and hasattr(evaluator, "split"):
            evaluator.split = "mini_val"

    config.callbacks.append(
        class_config(
            VisualizerCallback,
            visualizer=class_config(
                MultiCameraBBox3DVisualizer,
                cat_mapping=nuscenes_class_map,
                width=2,
                camera_near_clip=0.15,
                cameras=NuScenes.CAMERAS,
                vis_freq=1,
            ),
            output_dir=config.output_dir,
            save_prefix="boxes3d",
            test_connector=class_config(
                MultiSensorCallbackConnector,
                key_mapping=CONN_NUSC_BBOX_3D_VIS,
            ),
        )
    )

    config.callbacks.append(
        class_config(
            VisualizerCallback,
            visualizer=class_config(BEVBBox3DVisualizer, width=2, vis_freq=1),
            output_dir=config.output_dir,
            save_prefix="bev",
            test_connector=class_config(
                MultiSensorCallbackConnector,
                key_mapping=CONN_NUSC_BEV_BBOX_3D_VIS,
            ),
        )
    )

    return config
