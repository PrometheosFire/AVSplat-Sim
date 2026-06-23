"""Parametric CC-3DT visualization config builder for Wayve101 scenes."""
from __future__ import annotations

from vis4d.config import class_config
from vis4d.data.io import FileBackend
from vis4d.engine.callbacks import VisualizerCallback
from vis4d.engine.connectors import MultiSensorCallbackConnector
from vis4d.engine.connectors import MultiSensorDataConnector
from vis4d.op.track3d.cc_3dt import CC3DTrackAssociation
from vis4d.vis.image.bbox3d_visualizer import MultiCameraBBox3DVisualizer
from vis4d.vis.image.bev_visualizer import BEVBBox3DVisualizer
from vis4d.zoo.cc_3dt.cc_3dt_frcnn_r101_fpn_kf3d_24e_nusc import (
    get_config as _r101_get_config,
)
from vis4d.zoo.cc_3dt.cc_3dt_frcnn_r50_fpn_kf3d_12e_nusc import (
    get_config as _r50_get_config,
)

from src.model_wrappers.trackers.wayve101_cc3dt_data import (
    CONN_WAYVE101_BBOX_3D_TEST,
    CONN_WAYVE101_BBOX_3D_VIS,
    WAYVE101_CAMERAS,
    get_wayve101_bev_vis_conn,
    get_wayve101_data_cfg,
)


def get_config(
    backbone: str = "r101",
    data_root: str = "data/wayve101/scene_001",
    reference_camera: str = "front-forward",
    image_size: tuple[int, int] = (900, 1600),
    samples_per_gpu: int = 1,
    workers_per_gpu: int = 1,
    cameras: list[str] | None = None,
    experiment_name: str = "cc_3dt_wayve101_vis",
    pure_det: bool = False,
    motion_model: str = "KF3D",
    fps: int = 2,
    memory_size: int = 10,
    memory_momentum: float = 0.8,
    backdrop_memory_size: int = 1,
    nms_backdrop_iou_thr: float = 0.3,
    update_3d_score: bool = True,
    add_backdrops: bool = True,
    use_velocities: bool = False,
    assoc_init_score_thr: float = 0.8,
    assoc_obj_score_thr: float = 0.5,
    assoc_match_score_thr: float = 0.5,
    assoc_nms_backdrop_iou_thr: float = 0.3,
    assoc_nms_class_iou_thr: float = 0.7,
    assoc_nms_conf_thr: float = 0.5,
    assoc_with_cats: bool = True,
    assoc_with_velocities: bool = False,
    bbox_affinity_weight: float = 0.5,
    cat_mapping: dict[str, int] | None = None,
    undistort: bool = True,
    undistort_balance: float = 0.0,
    undistort_mode: str = "balance",
    virtual_focal_px: float = 1266.0,
):
    """Build a Vis4D ExperimentConfig for CC-3DT test/vis on Wayve101."""
    backbone = backbone.lower()
    cameras = cameras or WAYVE101_CAMERAS
    image_h, image_w = int(image_size[0]), int(image_size[1])

    base_get = _r101_get_config if backbone == "r101" else _r50_get_config
    config = base_get()

    config.experiment_name = experiment_name
    config.params.samples_per_gpu = int(samples_per_gpu)
    config.params.workers_per_gpu = int(workers_per_gpu)

    # Model-level tunables exposed through configs/model/cc3dt.yaml.
    config.model.init_args.pure_det = bool(pure_det)

    track_assoc_cfg = class_config(
        CC3DTrackAssociation,
        init_score_thr=float(assoc_init_score_thr),
        obj_score_thr=float(assoc_obj_score_thr),
        match_score_thr=float(assoc_match_score_thr),
        nms_backdrop_iou_thr=float(assoc_nms_backdrop_iou_thr),
        nms_class_iou_thr=float(assoc_nms_class_iou_thr),
        nms_conf_thr=float(assoc_nms_conf_thr),
        with_cats=bool(assoc_with_cats),
        with_velocities=bool(assoc_with_velocities),
        bbox_affinity_weight=float(bbox_affinity_weight),
    )

    track_graph_init_args = config.model.init_args.track_graph.init_args
    track_graph_init_args.motion_model = motion_model
    track_graph_init_args.fps = int(fps)
    track_graph_init_args.track = track_assoc_cfg
    track_graph_init_args.memory_size = int(memory_size)
    track_graph_init_args.memory_momentum = float(memory_momentum)
    track_graph_init_args.backdrop_memory_size = int(backdrop_memory_size)
    track_graph_init_args.nms_backdrop_iou_thr = float(nms_backdrop_iou_thr)
    track_graph_init_args.update_3d_score = bool(update_3d_score)
    track_graph_init_args.add_backdrops = bool(add_backdrops)
    track_graph_init_args.use_velocities = bool(use_velocities)

    config.data = get_wayve101_data_cfg(
        data_root=data_root,
        cameras=cameras,
        reference_camera=reference_camera,
        image_size=(image_h, image_w),
        samples_per_gpu=int(samples_per_gpu),
        workers_per_gpu=int(workers_per_gpu),
        data_backend=class_config(FileBackend),
        undistort=bool(undistort),
        undistort_balance=float(undistort_balance),
        undistort_mode=str(undistort_mode),
        virtual_focal_px=float(virtual_focal_px),
    )

    config.test_data_connector = class_config(
        MultiSensorDataConnector,
        key_mapping=CONN_WAYVE101_BBOX_3D_TEST,
    )

    for callback in config.callbacks:
        # Redirect evaluator output_dir; mark save_only=True so the evaluator
        # writes JSON predictions without trying to load nuscenes groundtruth.
        ia = getattr(callback, "init_args", None)
        if ia is None:
            continue
        if hasattr(ia, "output_dir"):
            ia.output_dir = config.output_dir
        evaluator_cfg = getattr(ia, "evaluator", None)
        if evaluator_cfg is not None:
            eval_ia = getattr(evaluator_cfg, "init_args", None)
            if eval_ia is not None and hasattr(eval_ia, "save_only"):  # NuScenesDet3DEvaluator
                eval_ia.save_only = True

    # Append visualizer callback; keep the evaluator callbacks from base config
    # so that track_3d_predictions.json is written.
    vis_callback = class_config(
        VisualizerCallback,
        visualizer=class_config(
            MultiCameraBBox3DVisualizer,
            cat_mapping=cat_mapping
            or {
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
            cameras=cameras,
            vis_freq=1,
        ),
        output_dir=config.output_dir,
        save_prefix="boxes3d",
        test_connector=class_config(
            MultiSensorCallbackConnector,
            key_mapping=CONN_WAYVE101_BBOX_3D_VIS,
        ),
    )

    bev_callback = class_config(
        VisualizerCallback,
        visualizer=class_config(BEVBBox3DVisualizer, width=2, vis_freq=1),
        output_dir=config.output_dir,
        save_prefix="bev",
        test_connector=class_config(
            MultiSensorCallbackConnector,
            key_mapping=get_wayve101_bev_vis_conn(reference_camera),
        ),
    )

    config.callbacks = list(config.callbacks) + [vis_callback, bev_callback]

    return config