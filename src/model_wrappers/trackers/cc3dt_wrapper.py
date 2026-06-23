import glob
import logging
import os
import os.path as osp
import shutil

import torch
from vis4d.common.logging import dump_config, rank_zero_info, setup_logger
from vis4d.common.util import set_tf32
from vis4d.config import instantiate_classes
from vis4d.config.registry import get_config_by_name
from vis4d.engine.callbacks import Callback, LRSchedulerCallback, VisualizerCallback
from vis4d.engine.data_module import DataModule
from vis4d.engine.trainer import PLTrainer
from vis4d.engine.training_module import TrainingModule

from src.model_wrappers.base import BaseTracker


class CC3DTWrapper(BaseTracker):
    """Wrapper around Vis4D CC-3DT inference for AVSplat-Sim."""

    def __init__(
        self,
        checkpoint_path: str = "",
        backbone: str = "r101",
        vis4d_config_path: str = "src/model_wrappers/trackers/cc3dt_vis_config.py",
        name: str = "cc3dt",
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
    ) -> None:
        self.name = name
        self.backbone = backbone.lower()
        self.vis4d_config_path = vis4d_config_path
        self.checkpoint_path = checkpoint_path
        self.pure_det = pure_det
        self.motion_model = motion_model
        self.fps = fps
        self.memory_size = memory_size
        self.memory_momentum = memory_momentum
        self.backdrop_memory_size = backdrop_memory_size
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr
        self.update_3d_score = update_3d_score
        self.add_backdrops = add_backdrops
        self.use_velocities = use_velocities
        self.assoc_init_score_thr = assoc_init_score_thr
        self.assoc_obj_score_thr = assoc_obj_score_thr
        self.assoc_match_score_thr = assoc_match_score_thr
        self.assoc_nms_backdrop_iou_thr = assoc_nms_backdrop_iou_thr
        self.assoc_nms_class_iou_thr = assoc_nms_class_iou_thr
        self.assoc_nms_conf_thr = assoc_nms_conf_thr
        self.assoc_with_cats = assoc_with_cats
        self.assoc_with_velocities = assoc_with_velocities
        self.bbox_affinity_weight = bbox_affinity_weight
        self.cat_mapping = cat_mapping

    def _load_config(self, task_cfg: dict, data_root: str, cameras: list[str]):
        image_size = task_cfg.get("image_size", [900, 1600])
        samples_per_gpu = int(task_cfg.get("samples_per_gpu", 1))
        workers_per_gpu = int(task_cfg.get("workers_per_gpu", 1))
        reference_camera = str(task_cfg.get("reference_camera", "front-forward"))
        experiment_name = str(task_cfg.get("experiment_name", "cc3dt_tracking"))
        undistort = bool(task_cfg.get("undistort", True))
        undistort_balance = float(task_cfg.get("undistort_balance", 0.0))
        undistort_mode = str(task_cfg.get("undistort_mode", "balance"))
        virtual_focal_px = float(task_cfg.get("virtual_focal_px", 1266.0))

        return get_config_by_name(
            os.path.abspath(self.vis4d_config_path),
            self.backbone,
            os.path.abspath(data_root),
            reference_camera,
            (int(image_size[0]), int(image_size[1])),
            int(samples_per_gpu),
            int(workers_per_gpu),
            list(cameras),
            experiment_name,
            bool(self.pure_det),
            str(self.motion_model),
            int(self.fps),
            int(self.memory_size),
            float(self.memory_momentum),
            int(self.backdrop_memory_size),
            float(self.nms_backdrop_iou_thr),
            bool(self.update_3d_score),
            bool(self.add_backdrops),
            bool(self.use_velocities),
            float(self.assoc_init_score_thr),
            float(self.assoc_obj_score_thr),
            float(self.assoc_match_score_thr),
            float(self.assoc_nms_backdrop_iou_thr),
            float(self.assoc_nms_class_iou_thr),
            float(self.assoc_nms_conf_thr),
            bool(self.assoc_with_cats),
            bool(self.assoc_with_velocities),
            float(self.bbox_affinity_weight),
            self.cat_mapping,
            undistort,
            undistort_balance,
            undistort_mode,
            virtual_focal_px,
        )

    def _run_test(self, config, ckpt_path: str, vis_enabled: bool, num_gpus: int) -> None:
        logger_vis4d = logging.getLogger("vis4d")
        logger_pl = logging.getLogger("pytorch_lightning")
        log_file = osp.join(config.output_dir, f"log_{config.timestamp}.txt")
        setup_logger(logger_vis4d, log_file)
        setup_logger(logger_pl, log_file)

        dump_config(config, osp.join(config.output_dir, f"config_{config.timestamp}.yaml"))

        set_tf32(config.use_tf32, config.tf32_matmul_precision)
        # Cache hub downloads in a stable shared location, not the per-run
        # output dir (which is a temp work dir that gets cleaned up).
        torch.hub.set_dir(osp.expanduser("~/.cache/torch/hub"))

        if num_gpus > 0:
            config.pl_trainer.accelerator = "gpu"
            config.pl_trainer.devices = num_gpus
        else:
            config.pl_trainer.accelerator = "cpu"
            config.pl_trainer.devices = 1

        trainer_args = instantiate_classes(config.pl_trainer).to_dict()

        test_data_connector = None
        if config.test_data_connector is not None:
            test_data_connector = instantiate_classes(config.test_data_connector)

        callbacks: list[Callback] = []
        for cb in config.callbacks:
            callback = instantiate_classes(cb)
            assert isinstance(callback, Callback), (
                "Callback must be a subclass of Callback. "
                f"Provided callback: {cb} is not!"
            )

            if not vis_enabled and isinstance(callback, VisualizerCallback):
                continue

            callbacks.append(callback)

        callbacks.append(LRSchedulerCallback())

        trainer = PLTrainer(callbacks=callbacks, **trainer_args)

        hyper_params = trainer_args
        if config.get("params", None) is not None:
            hyper_params.update(config.params.to_dict())

        training_module = TrainingModule(
            config.model,
            config.optimizers,
            None,
            None,
            test_data_connector,
            hyper_params,
            config.seed,
            ckpt_path,
            config.compute_flops,
            config.check_unused_parameters,
        )
        data_module = DataModule(config.data)
        rank_zero_info("Running CC-3DT tracking through Vis4D engine API.")
        trainer.test(training_module, datamodule=data_module, verbose=False)

    def _default_ckpt_for_backbone(self) -> str:
        if self.backbone == "r101":
            return "external_weights/cc3dt/cc_3dt_frcnn_r101_fpn_24e_nusc_f24f84.pt"
        return "external_weights/cc3dt/cc_3dt_frcnn_r50_fpn_12e_nusc_d98509.pt"

    def _resolve_checkpoint(self) -> str:
        ckpt = self.checkpoint_path or self._default_ckpt_for_backbone()
        ckpt = os.path.abspath(ckpt)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt}. Set the tracker model checkpoint_path."
            )
        return ckpt

    def run_tracking(
        self,
        data_root: str,
        output_dir: str,
        cameras: list[str],
        task_cfg: dict,
    ) -> None:
        mode = str(task_cfg.get("mode", "vis"))
        if mode != "vis":
            raise NotImplementedError("Only mode='vis' is implemented for CC-3DT wrapper.")

        image_size = task_cfg.get("image_size", [900, 1600])
        if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
            raise ValueError("task_cfg.image_size must be [height, width].")

        vis_enabled = bool(task_cfg.get("vis", True))
        experiment_name = str(task_cfg.get("experiment_name", "cc3dt_tracking"))
        num_gpus = int(task_cfg.get("gpus", 1))

        ckpt = self._resolve_checkpoint()
        config_path = os.path.abspath(self.vis4d_config_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Vis4D config not found: {config_path}.")

        os.makedirs(output_dir, exist_ok=True)

        # Run Vis4D into a temp work dir so its nested layout stays contained,
        # then reorganize the wanted artifacts into a clean logs/eval/vis tree.
        work_dir = osp.join(output_dir, ".vis4d_tmp")
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)

        config = self._load_config(task_cfg, data_root, cameras)
        config.experiment_name = experiment_name
        config.output_dir = work_dir
        config.work_dir = work_dir

        for callback in config.callbacks:
            init_args = getattr(callback, "init_args", None)
            if init_args is not None and hasattr(init_args, "output_dir"):
                init_args.output_dir = work_dir

        self._run_test(config, ckpt, vis_enabled=vis_enabled, num_gpus=num_gpus)
        self._reorganize_outputs(work_dir, output_dir)
        reference_camera = str(task_cfg.get("reference_camera", "front-forward"))
        self._write_colmap_predictions(
            output_dir, data_root, list(cameras), reference_camera
        )
        shutil.rmtree(work_dir, ignore_errors=True)

    def _write_colmap_predictions(
        self,
        output_dir: str,
        data_root: str,
        cameras: list[str],
        reference_camera: str,
    ) -> None:
        """Write COLMAP-frame copies of the prediction JSONs.

        The tracker runs in a ROS-aligned global frame (required for correct
        yaw), so the saved predictions are in that frame. 4DGS uses the raw
        COLMAP frame, so we reproduce the exact world alignment and emit
        ``*_colmap.json`` siblings with boxes rotated back into COLMAP
        coordinates (the alignment is a pure rotation, so this is lossless).
        """
        import json

        import numpy as np
        from scipy.spatial.transform import Rotation as R

        from src.model_wrappers.trackers.wayve101_cc3dt_dataset import (
            compute_scene_world_alignment,
        )

        try:
            align = compute_scene_world_alignment(
                data_root, cameras, reference_camera
            )
        except Exception as exc:  # pragma: no cover - best-effort export
            rank_zero_info(f"Skipping COLMAP-frame export: {exc}")
            return

        rot = np.asarray(align[:3, :3], dtype=np.float64)
        rot_inv = rot.T  # ROS -> COLMAP (orthonormal, so transpose == inverse)

        eval_dir = osp.join(output_dir, "eval")
        for fname in (
            "track_3d_predictions.json",
            "detect_3d_predictions.json",
        ):
            src = osp.join(eval_dir, fname)
            if not os.path.exists(src):
                continue
            with open(src, encoding="utf-8") as f:
                data = json.load(f)

            for boxes in data.get("results", {}).values():
                for box in boxes:
                    t = np.asarray(box["translation"], dtype=np.float64)
                    box["translation"] = (rot_inv @ t).tolist()

                    # rotation is [w, x, y, z]; scipy wants [x, y, z, w].
                    w, x, y, z = box["rotation"]
                    r_box = R.from_quat([x, y, z, w]).as_matrix()
                    qx, qy, qz, qw = R.from_matrix(rot_inv @ r_box).as_quat()
                    box["rotation"] = [
                        float(qw),
                        float(qx),
                        float(qy),
                        float(qz),
                    ]

                    vel = box.get("velocity")
                    if vel is not None and len(vel) >= 2:
                        v3 = np.array(
                            [vel[0], vel[1], 0.0], dtype=np.float64
                        )
                        v3c = rot_inv @ v3
                        box["velocity"] = [float(v3c[0]), float(v3c[1])]

            dst = osp.join(eval_dir, fname.replace(".json", "_colmap.json"))
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(data, f)

    def _reorganize_outputs(self, work_dir: str, output_dir: str) -> None:
        """Flatten Vis4D's nested output tree into logs/, eval/, vis/.

        Vis4D writes a deeply nested layout::

            <work>/<exp>/<timestamp>/{events..., hparams.yaml}
            <work>/{log_<ts>.txt, config_<ts>.yaml}
            <work>/eval/<metric>/<metric>_predictions.json
            <work>/vis/test/bev/<scene>/BEV/<frames>
            <work>/vis/test/boxes3d/<scene>/<camera>/<frames>

        which this reshapes into::

            <out>/logs/{events..., hparams.yaml, log_<ts>.txt, config_<ts>.yaml}
            <out>/eval/{detect_3d_predictions.json, track_3d_predictions.json}
            <out>/vis/bev/<frames>
            <out>/vis/boxes3d/<camera>/<frames>
        """
        # logs: collect tensorboard + run logs into one flat folder.
        logs_dir = osp.join(output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_patterns = [
            "**/events.out.tfevents.*",
            "**/hparams.yaml",
            "log_*.txt",
            "config_*.yaml",
        ]
        for pattern in log_patterns:
            for src in glob.glob(osp.join(work_dir, pattern), recursive=True):
                if os.path.isfile(src):
                    shutil.move(src, osp.join(logs_dir, os.path.basename(src)))

        # eval: flatten the per-metric subfolders (detect_3d/, track_3d/).
        src_eval = osp.join(work_dir, "eval")
        if os.path.isdir(src_eval):
            dst_eval = osp.join(output_dir, "eval")
            os.makedirs(dst_eval, exist_ok=True)
            for src in glob.glob(
                osp.join(src_eval, "**", "*.json"), recursive=True
            ):
                shutil.move(src, osp.join(dst_eval, os.path.basename(src)))

        # vis/bev: flatten <scene>/BEV/ into a single folder of frames.
        src_bev = osp.join(work_dir, "vis", "test", "bev")
        if os.path.isdir(src_bev):
            dst_bev = osp.join(output_dir, "vis", "bev")
            os.makedirs(dst_bev, exist_ok=True)
            for src in glob.glob(osp.join(src_bev, "**", "*"), recursive=True):
                if os.path.isfile(src):
                    shutil.move(src, osp.join(dst_bev, os.path.basename(src)))

        # vis/boxes3d: keep one subfolder per camera, dropping the scene level.
        src_box = osp.join(work_dir, "vis", "test", "boxes3d")
        if os.path.isdir(src_box):
            dst_box = osp.join(output_dir, "vis", "boxes3d")
            for scene_dir in sorted(glob.glob(osp.join(src_box, "*"))):
                if not os.path.isdir(scene_dir):
                    continue
                for cam_dir in sorted(glob.glob(osp.join(scene_dir, "*"))):
                    if not os.path.isdir(cam_dir):
                        continue
                    dst_cam = osp.join(dst_box, os.path.basename(cam_dir))
                    os.makedirs(dst_cam, exist_ok=True)
                    for src in glob.glob(osp.join(cam_dir, "*")):
                        if os.path.isfile(src):
                            shutil.move(
                                src, osp.join(dst_cam, os.path.basename(src))
                            )
