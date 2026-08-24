"""Export the real frames the trainer actually sees, for use as Difix references.

The pseudo-view loop needs, for every rendered frame, the matching REAL image at
the same camera and timestep to condition Difix on. The raw JPEGs under
``data/<dataset>/<scene>/images/`` are not usable for that: they are 1920x1080
and go through a different decode path than training, which reads 960x540 arrays
straight from the NCore sensor. Feeding a differently-resampled image as the
reference would inject a resolution and sharpness mismatch into the conditioning
signal.

This script writes exactly the training pixels, once per scene, plus the index
that ties a camera-local frame position to its flat parser index. That index is
what makes validation exclusion correct: the parser's frame list is CAMERA-MAJOR
(``datasets/ncore.py`` ``_load_poses``), so with 200 frames per camera and
``test_every=16`` the validation phase rotates per camera (``200 % 16 == 8``) and
a per-camera ``i % test_every`` test would silently leak held-out frames.

Frame numbering matches the renderer: ``frame_{local:05d}.png`` uses the
camera-local position, which is the same index as
``camera_paths/<cam>/camtoworlds.npy`` and the renderer's own ``frame_%05d.png``.

Output layout (under the Hydra run dir)::

    camera1/frame_00000.png ... frame_00199.png
    index.json
    .success

Run standalone::

    PYTHONPATH=. envs/envs/env_gsplat/bin/python \\
        src/gsplat_training/export_real_frames.py \\
        hydra.run.dir=<out_dir> \\
        ++gaussian_splatting.data_dir=<ncore>/ncore_dataset/staging_symlinks.json \\
        ++gaussian_splatting.ncore_camera_ids=[camera1]
"""

import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import cv2
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from config_training import Config
from datasets.ncore import NCoreParser

SCHEMA_VERSION = 1


def _fmt_hms(seconds: float) -> str:
    """Format a duration as HH:MM:SS. Hours accumulate past 24 rather than wrap."""
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _stamp_success(path: str, message: str, elapsed: float, breakdown=None) -> None:
    """Write a .success marker carrying its runtime and optional sub-block times.

    Nothing in the pipeline reads these files -- only their existence is checked
    -- so the extra lines are free to grow.
    """
    lines = [message, f"duration: {_fmt_hms(elapsed)}"]
    if breakdown:
        width = max(len(label) for label, _ in breakdown)
        for label, value in breakdown:
            shown = value if isinstance(value, str) else _fmt_hms(value)
            lines.append(f"  {label:<{width}}  {shown}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def load_gsplat_config(cfg: DictConfig) -> Config:
    """Materialise the training Config from the ``gaussian_splatting`` subtree.

    Mirrors ``train_splats.py``: the YAML only defines a subset of the fields,
    so the rest must come from the dataclass defaults. Reading the raw YAML
    instead would miss keys such as ``ncore_lidar_ids`` and, worse, could drift
    from whatever the trainer actually used. The strategy keys are popped
    because they are not ``Config`` fields; this script never rasterises, so the
    strategy itself is irrelevant here.
    """
    gsplat_dict = OmegaConf.to_container(cfg.gaussian_splatting, resolve=True)
    gsplat_dict.pop("strategy_type", None)
    gsplat_dict.pop("strategy", None)
    return Config(**gsplat_dict)


def build_parser(gs_cfg: Config) -> NCoreParser:
    """Construct the NCoreParser exactly as the trainer does.

    The argument list mirrors ``Runner.__init__`` so that frame ordering, the
    train/val split and the intrinsics all match the training run bit for bit.
    Any divergence here would desynchronise ``global_index`` from the split the
    trainer actually uses.
    """
    return NCoreParser(
        meta_json_path=gs_cfg.data_dir,
        factor=1.0 / gs_cfg.data_factor if gs_cfg.data_factor > 1 else 1.0,
        test_every=gs_cfg.test_every,
        camera_ids=list(gs_cfg.ncore_camera_ids) or None,
        lidar_ids=list(gs_cfg.ncore_lidar_ids) or None,
        seek_offset_sec=gs_cfg.ncore_seek_offset_sec,
        duration_sec=gs_cfg.ncore_duration_sec,
        max_lidar_points=gs_cfg.ncore_max_lidar_points,
        lidar_color_generic_data_name=gs_cfg.ncore_lidar_color_generic_data_name,
        poses_component_group=gs_cfg.ncore_poses_component_group,
        intrinsics_component_group=gs_cfg.ncore_intrinsics_component_group,
        masks_component_group=gs_cfg.ncore_masks_component_group,
        normalize_world_space=gs_cfg.normalize_world_space,
    )


def export_frames(
    parser: NCoreParser, out_dir: str
) -> Tuple[Dict[str, Any], List[Tuple[str, float]]]:
    """Write every frame in the parser to ``out_dir`` and return the index.

    Walks the flat (camera-major) frame list once, keeping a per-camera counter
    so each image lands at its camera-local position. Pixels are read through
    the same call the dataset uses (``get_frame_image_array`` plus the optional
    ``factor`` resize), so the exported PNG is byte-identical to the training
    tensor before normalisation.

    Returns:
        ``(index, per_camera_seconds)``. ``index`` is the ``index.json`` payload:
        per camera, its ``camera_index``, size, intrinsics and a list of
        ``{frame_idx, global_index, timestamp_us, is_val, image}`` records.
        ``per_camera_seconds`` is the decode+write time attributed to each
        camera, accumulated per frame so it stays correct whatever order the
        flat frame list happens to be in.
    """
    sequence_loader = parser._open_sequence_loader(parser.sequence_meta_file_path)
    sensors = {cid: sequence_loader.get_camera_sensor(cid) for cid in parser.camera_ids}

    cameras: Dict[str, Any] = {}
    for cam_idx, camera_id in enumerate(parser.camera_ids):
        width, height = parser.imsize_dict[camera_id]
        os.makedirs(os.path.join(out_dir, camera_id), exist_ok=True)
        cameras[camera_id] = {
            "camera_index": cam_idx,
            "width": int(width),
            "height": int(height),
            "K": parser.Ks_dict[camera_id].tolist(),
            "frames": [],
        }

    local_counter: Dict[str, int] = defaultdict(int)
    per_camera_seconds: Dict[str, float] = defaultdict(float)
    for global_index in range(len(parser.frame_list)):
        t_frame = time.perf_counter()
        camera_id, sensor_frame_idx = parser.frame_list[global_index]
        local_idx = local_counter[camera_id]
        local_counter[camera_id] += 1

        width, height = parser.imsize_dict[camera_id]
        image = sensors[camera_id].get_frame_image_array(sensor_frame_idx)
        if parser.factor != 1.0:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

        rel_path = os.path.join(camera_id, f"frame_{local_idx:05d}.png")
        # get_frame_image_array returns RGB; cv2 writes BGR.
        cv2.imwrite(os.path.join(out_dir, rel_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        cameras[camera_id]["frames"].append(
            {
                "frame_idx": local_idx,
                "global_index": global_index,
                "timestamp_us": int(parser.frame_timestamps_us[global_index]),
                # Mirrors NCoreDataset's split so held-out frames can be kept
                # out of the pseudo-view bank.
                "is_val": bool(global_index % parser.test_every == 0),
                "image": rel_path,
            }
        )
        per_camera_seconds[camera_id] += time.perf_counter() - t_frame

    index = {
        "schema_version": SCHEMA_VERSION,
        "test_every": int(parser.test_every),
        "num_frames": int(len(parser.frame_list)),
        "cameras": cameras,
    }
    per_camera = [(cid, per_camera_seconds[cid]) for cid in parser.camera_ids]
    return index, per_camera


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = HydraConfig.get().runtime.output_dir
    success_marker = os.path.join(out_dir, ".success")
    if os.path.exists(success_marker):
        print(f"Real-frame bank already exported at {out_dir}")
        return

    t_start = time.perf_counter()
    parser = build_parser(load_gsplat_config(cfg))
    t_parser = time.perf_counter() - t_start
    index, per_camera = export_frames(parser, out_dir)

    with open(os.path.join(out_dir, "index.json"), "w") as fp:
        json.dump(index, fp, indent=2)

    n_val = sum(
        1 for cam in index["cameras"].values() for f in cam["frames"] if f["is_val"]
    )
    print(
        f"Exported {index['num_frames']} frames across "
        f"{len(index['cameras'])} cameras to {out_dir} "
        f"({n_val} marked is_val and excluded from pseudo-view generation)"
    )

    _stamp_success(
        success_marker,
        "Real frame export completed successfully.",
        time.perf_counter() - t_start,
        [("parser build", t_parser), *per_camera],
    )


if __name__ == "__main__":
    main()
