"""Local CC-3DT R50 + KF3D smoke-test config for nuScenes mini."""
from __future__ import annotations

from vis4d.config import class_config
from vis4d.data.io import FileBackend
from vis4d.zoo.cc_3dt.cc_3dt_frcnn_r50_fpn_kf3d_12e_nusc import get_config as _base_get_config
from vis4d.zoo.cc_3dt.data import get_nusc_cfg


def get_config():
    """Return a low-memory config for local nuScenes mini inference."""
    config = _base_get_config()

    data_root = "/home/dinis-matos/workspace/AVSplat-Sim/data/nuscenes/v1.0-mini"
    version = "v1.0-mini"
    train_split = "mini_train"
    test_split = "mini_val"

    config.experiment_name = "cc_3dt_r50_kf3d_nusc_mini_smoke"
    config.params.samples_per_gpu = 1
    config.params.workers_per_gpu = 1
    config.data = get_nusc_cfg(
        data_root=data_root,
        version=version,
        train_split=train_split,
        test_split=test_split,
        data_backend=class_config(FileBackend),
        samples_per_gpu=1,
        workers_per_gpu=1,
    )

    for callback in config.callbacks:
        evaluator = getattr(callback, "evaluator", None)
        if evaluator is not None and hasattr(evaluator, "data_root"):
            evaluator.data_root = data_root
        if evaluator is not None and hasattr(evaluator, "version"):
            evaluator.version = version
        if evaluator is not None and hasattr(evaluator, "split"):
            evaluator.split = test_split

    return config