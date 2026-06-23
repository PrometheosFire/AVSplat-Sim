"""Local CC-3DT R50 + KF3D visualization config for nuScenes mini."""
from __future__ import annotations

from vis4d.config import class_config
from vis4d.data.io import FileBackend
from vis4d.zoo.cc_3dt.cc_3dt_nusc_vis import get_config as _base_vis_get_config
from vis4d.zoo.cc_3dt.data import get_nusc_cfg


def get_config():
    """Return a local mini vis config that renders projected 3D boxes."""
    config = _base_vis_get_config()

    data_root = "/home/dinis-matos/workspace/AVSplat-Sim/data/nuscenes/v1.0-mini"

    config.experiment_name = "cc_3dt_r50_kf3d_nusc_mini_vis"
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

    return config
