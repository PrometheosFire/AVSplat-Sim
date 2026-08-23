"""Difix-cleaned pseudo-views as a second training source.

The pseudo-view loop renders laterally shifted trajectories, cleans the artifacts
with Difix3D+, and feeds the result back as extra supervision so the 3DGS model
sees parallax the ego trajectory never provided. This module is the read side of
that: it turns one or more round manifests into batches shaped exactly like
``NCoreDataset.__getitem__``'s, so the trainer can draw from it without any
change to the loss.

The split mirrors ``ncore.py`` and ``colmap.py``: :class:`PseudoViewParser` reads
the manifests once and exposes the flat per-view metadata, while
:class:`PseudoViewDataset` indexes into it and materialises images lazily.

Two things this parser deliberately does NOT own, because the real scene already
does. Distortion coefficients are resolved by the trainer from
``ncore_camera_data[camera_idx]`` (``runner.py`` line 769), since lens geometry
is a property of the camera and not of the pose a view was rendered from. Scene
normalisation (``transform``, ``scene_scale``) likewise stays with the real
parser; the manifest's poses are already in the trainer's normalized frame.

Pseudo-views live in a SEPARATE dataset from the real frames, never a joined
pool. The mixing ratio is then a fixed setting (``pseudo_sample_prob``) rather
than an emergent property of how big the bank has grown. That matters because
the bank accumulates across rounds: with a joined pool, pseudo-GT's share of the
gradient would climb round over round on its own (~14% -> ~32% -> ~49% for a
single camera), diluting the real-data anchor without anyone choosing it.

Batches deliberately omit two keys the NCore dataset provides:

* ``mask`` -- pseudo-views are unmasked. ``runner.py`` reads it as
  ``if "mask" in data``, so the loss falls back to a plain full-image L1/SSIM.
* ``timestamp_us`` -- its absence makes ``_resolve_rigid_frame_idx`` return
  ``None``, which is what a static-only bake wants.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.utils.data

SCHEMA_VERSION = 1


class PseudoViewParser:
    """Reads pseudo-view round manifests into flat per-view metadata.

    Mirrors :class:`~datasets.ncore.NCoreParser`'s role: parse once, expose the
    arrays the dataset indexes into. Concatenating several manifests is what
    implements the accumulating bank -- round *r* is handed rounds 1..*r*'s
    manifests.

    Attributes:
        entries: Raw manifest records, in load order.
        image_paths: Absolute path to each view's difixed PNG.
        camtoworlds: ``[N, 4, 4]`` SHIFTED poses, already in the trainer's
            normalized frame.
        Ks_dict: Per-camera ``[3, 3]`` intrinsics.
        imsize_dict: Per-camera ``(width, height)``.
        camera_ids: Cameras present, in the trainer's ordering.
        camera_idx_per_frame: Trainer camera index per view.
        frame_list: ``[(camera_id, frame_idx), ...]`` per view.
        frame_timestamps_us: Capture timestamp of the REAL frame each view was
            rendered at (carried for provenance; not emitted in batches).
        shift_names: Shift label per view, e.g. ``"X_-1"``.
        test_every: Validation stride recorded by the manifest builder.
        num_cameras: Number of distinct cameras present.

    Args:
        manifest_paths: One manifest per round.
        expected_camera_ids: Trainer camera order (``parser.camera_ids``). When
            given, each entry's ``camera_idx`` is checked against the position
            of its ``camera_id`` in this list, and ``camera_ids`` is emitted in
            this order.
        expected_Ks: Trainer intrinsics (``parser.Ks_dict``). When given, each
            entry's ``K`` is checked against it.
        k_atol: Absolute tolerance for the intrinsics check.

    Raises:
        ValueError: On a schema mismatch, a camera-index or intrinsics
            disagreement, or a validation frame appearing in the bank.
    """

    def __init__(
        self,
        manifest_paths: List[str],
        expected_camera_ids: Optional[List[str]] = None,
        expected_Ks: Optional[Dict[str, np.ndarray]] = None,
        k_atol: float = 1e-3,
    ) -> None:
        self.manifest_paths = list(manifest_paths)
        self.entries: List[Dict[str, Any]] = []
        self.image_paths: List[str] = []
        self.frame_list: List[Tuple[str, int]] = []
        self.camera_idx_per_frame: List[int] = []
        self.frame_timestamps_us: List[int] = []
        self.shift_names: List[str] = []
        self.Ks_dict: Dict[str, np.ndarray] = {}
        self.imsize_dict: Dict[str, Tuple[int, int]] = {}
        self.test_every: Optional[int] = None

        seen_cameras: List[str] = []

        for manifest_path in self.manifest_paths:
            with open(manifest_path, "r") as fp:
                manifest = json.load(fp)

            version = manifest.get("schema_version")
            if version != SCHEMA_VERSION:
                raise ValueError(
                    f"{manifest_path}: schema_version {version} != {SCHEMA_VERSION}"
                )

            test_every = manifest.get("test_every")
            if test_every:
                self.test_every = int(test_every)

            for entry in manifest["entries"]:
                self._validate_entry(
                    entry,
                    manifest_path,
                    test_every,
                    expected_camera_ids,
                    expected_Ks,
                    k_atol,
                )

                camera_id = entry["camera_id"]
                if camera_id not in seen_cameras:
                    seen_cameras.append(camera_id)
                    self.Ks_dict[camera_id] = np.asarray(
                        entry["K"], dtype=np.float32
                    )
                    self.imsize_dict[camera_id] = (
                        int(entry["width"]),
                        int(entry["height"]),
                    )

                self.entries.append(entry)
                self.image_paths.append(entry["image_path"])
                self.frame_list.append((camera_id, int(entry["frame_idx"])))
                self.camera_idx_per_frame.append(int(entry["camera_idx"]))
                self.frame_timestamps_us.append(int(entry["timestamp_us"]))
                self.shift_names.append(entry["shift_name"])

        # Preserve the trainer's camera ordering when we know it, so that
        # camera_ids.index(cid) agrees with camera_idx everywhere.
        if expected_camera_ids is not None:
            self.camera_ids = [c for c in expected_camera_ids if c in seen_cameras]
        else:
            self.camera_ids = seen_cameras
        self.num_cameras = len(self.camera_ids)

        self.camtoworlds = (
            np.stack(
                [np.asarray(e["camtoworld"], dtype=np.float32) for e in self.entries]
            )
            if self.entries
            else np.zeros((0, 4, 4), dtype=np.float32)
        )

        self._print_summary()

    @staticmethod
    def _validate_entry(
        entry: Dict[str, Any],
        manifest_path: str,
        test_every: Optional[int],
        expected_camera_ids: Optional[List[str]],
        expected_Ks: Optional[Dict[str, np.ndarray]],
        k_atol: float,
    ) -> None:
        """Fail loudly on the mismatches that would otherwise corrupt silently."""
        camera_id = entry["camera_id"]

        # camera_idx selects the distortion model at runner.py:769-775. A stale
        # index renders a fisheye pseudo-view with another camera's coefficients
        # -- camera1 is a narrow forward lens, camera9 a wide side one -- and
        # still produces a plausible-looking image, so check it up front.
        if expected_camera_ids is not None:
            if camera_id not in expected_camera_ids:
                raise ValueError(
                    f"{manifest_path}: camera_id {camera_id!r} is not in the "
                    f"trainer's camera list {expected_camera_ids}"
                )
            expected_idx = expected_camera_ids.index(camera_id)
            if entry["camera_idx"] != expected_idx:
                raise ValueError(
                    f"{manifest_path}: {camera_id} has camera_idx "
                    f"{entry['camera_idx']} but the trainer has it at "
                    f"{expected_idx}; the manifest is stale."
                )

        # K is the one calibration value the manifest actually supplies (the
        # trainer takes distortion from its own ncore_camera_data), so a
        # data_factor change between rendering and training must not slip past.
        if expected_Ks is not None and camera_id in expected_Ks:
            manifest_K = np.asarray(entry["K"], dtype=np.float64)
            if not np.allclose(manifest_K, expected_Ks[camera_id], atol=k_atol):
                raise ValueError(
                    f"{manifest_path}: K for {camera_id} disagrees with the "
                    f"trainer's intrinsics (atol={k_atol}). Was the bank "
                    f"rendered with a different data_factor?"
                )

        # Second line of defence on validation leakage, independent of the
        # manifest builder having filtered correctly.
        if test_every and entry["global_index"] % test_every == 0:
            raise ValueError(
                f"{manifest_path}: entry {entry['image_path']} has "
                f"global_index {entry['global_index']}, which is a validation "
                f"frame (test_every={test_every})."
            )

    def _print_summary(self) -> None:
        by_shift: Dict[str, int] = {}
        for name in self.shift_names:
            by_shift[name] = by_shift.get(name, 0) + 1
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(by_shift.items()))
        print(
            f"[PseudoViewParser] Loaded {len(self.entries)} views from "
            f"{len(self.manifest_paths)} manifest(s) across "
            f"{self.num_cameras} cameras [{breakdown}]"
        )

    def __len__(self) -> int:
        return len(self.entries)


class PseudoViewDataset(torch.utils.data.Dataset):
    """Image-based dataset over :class:`PseudoViewParser` metadata.

    Returns batches compatible with the gsplat trainer:
    ``{"K", "camtoworld", "image", "image_id", "camera_idx"}``.

    Images are loaded lazily per ``__getitem__`` from plain PNGs on disk, so
    unlike :class:`~datasets.ncore.NCoreDataset` there is no per-worker sequence
    loader to manage.

    Args:
        parser: Parsed manifest metadata.
        split: Only ``"train"`` is valid. Pseudo-views must never reach
            validation -- ``eval()`` scores the model against real held-out
            frames, and admitting Difix output there would make the metric
            measure agreement with the diffusion model instead.

    Raises:
        ValueError: If ``split`` is anything other than ``"train"``.
    """

    def __init__(self, parser: PseudoViewParser, split: str = "train") -> None:
        if split != "train":
            raise ValueError(
                f"PseudoViewDataset supports split='train' only, got {split!r}. "
                "Pseudo-views are training-only by design; validation must stay "
                "on real held-out frames."
            )
        self.parser = parser
        self.split = split
        self.indices = np.arange(len(parser))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Build one pseudo-view training sample.

        Returns:
            Dictionary with the keys ``runner.py`` reads at lines 1119-1168.
            ``image`` is float 0-255 (the trainer divides by 255) and
            ``camtoworld`` is already the SHIFTED pose in the trainer's
            normalized frame.
        """
        index = self.indices[item]
        camera_id, _ = self.parser.frame_list[index]
        width, height = self.parser.imsize_dict[camera_id]
        K = self.parser.Ks_dict[camera_id].copy()

        image_path = self.parser.image_paths[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # A Difix output that was not resized back to the render's size would
        # silently mismatch K (960x540 snaps to 960x536 through the VAE), so
        # make that a hard failure rather than a subtle geometric error.
        if image.shape[:2] != (height, width):
            raise ValueError(
                f"{image_path}: image is {image.shape[:2]}, manifest says "
                f"{(height, width)}"
            )

        return {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(self.parser.camtoworlds[index]).float(),
            "image": torch.from_numpy(image).float(),
            # Constant on purpose: image_id indexes per-image embedding tables
            # sized len(trainset) (pose_opt, app_opt, bilateral grid, PPISP).
            # Pseudo-views own no row in those, which is why the Runner guard
            # forbids enabling them alongside a pseudo bank.
            "image_id": 0,
            "camera_idx": int(self.parser.camera_idx_per_frame[index]),
        }
