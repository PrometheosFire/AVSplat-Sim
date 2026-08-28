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

Batches omit ``mask`` unconditionally -- pseudo-views are unmasked, and
``runner.py`` reads it as ``if "mask" in data``, so the loss falls back to a
plain full-image L1/SSIM.

``timestamp_us`` is conditional, and it is the one key that decides whether
vehicles appear on a pseudo step. Its presence makes ``_resolve_rigid_frame_idx``
(``runner.py:721``) return a rigid frame index, so the loss renders the scene's
rigid nodes into the pseudo-view's pose; its absence leaves rendering
background-only. Which is correct depends entirely on what the cleaned image
actually contains, so the MANIFEST decides rather than a trainer flag: it
records whether the renders it cleaned had dynamic objects on, and this module
emits ``timestamp_us`` per entry to match.

Getting that backwards is silent in both directions. A dynamic image trained
without the key teaches the background to paint vehicles into itself; a static
image trained with it asks the loss to erase vehicles the render just added.
The runner refuses the first combination outright (see ``has_dynamic``).

Schema versions: 1 predates the flag and always means non-dynamic, so banks
produced before this existed keep loading unchanged.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.utils.data

SCHEMA_VERSION = 2
# Version 1 lacked the "dynamic" field; absent means static, never unknown, so
# older banks stay readable rather than needing regeneration.
SUPPORTED_SCHEMA_VERSIONS = (1, 2)


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
            rendered at. Emitted in batches only for dynamic entries, where it
            selects the rigid pose the vehicles were rendered with.
        image_ids: Split-local training index of the REAL frame each view was
            rendered from, or 0 throughout when ``train_indices`` was not given.
            This is what lets per-image modules (appearance embeddings, the
            bilateral grid) work alongside a pseudo bank: a pseudo-view of
            camera *c* at timestep *t* shares that frame's exposure and white
            balance, so sharing its row is correct rather than merely safe.
        is_dynamic: Per view, whether the render it was cleaned from contained
            rigid objects. Tracked per entry rather than per parser because the
            bank concatenates one manifest per round, and a mixed bank is a real
            case -- the dynamic loop's ``final_round`` mode trains a dynamic
            round on a bank of static rounds.
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
        train_indices: The trainer's ``NCoreDataset.indices`` -- the parser-global
            frame indices making up the train split. When given, each entry's
            ``global_index`` is mapped through it to the split-local ``image_id``
            that ``NCoreDataset`` would assign the same frame, and an entry whose
            frame is NOT in the train split is rejected. Leave unset (the
            default) to emit ``image_id`` 0 as before.

    Raises:
        ValueError: On a schema mismatch, a camera-index or intrinsics
            disagreement, or a validation frame appearing in the bank.
    """

    def __init__(
        self,
        manifest_paths: List[str],
        expected_camera_ids: Optional[List[str]] = None,
        expected_Ks: Optional[Dict[str, np.ndarray]] = None,
        train_indices: Optional[np.ndarray] = None,
        k_atol: float = 1e-3,
    ) -> None:
        self.manifest_paths = list(manifest_paths)
        self.entries: List[Dict[str, Any]] = []
        self.image_paths: List[str] = []
        self.frame_list: List[Tuple[str, int]] = []
        self.camera_idx_per_frame: List[int] = []
        self.frame_timestamps_us: List[int] = []
        self.is_dynamic: List[bool] = []
        self.image_ids: List[int] = []
        self.shift_names: List[str] = []
        self._train_indices = (
            np.asarray(train_indices) if train_indices is not None else None
        )
        self.Ks_dict: Dict[str, np.ndarray] = {}
        self.imsize_dict: Dict[str, Tuple[int, int]] = {}
        self.test_every: Optional[int] = None

        seen_cameras: List[str] = []

        for manifest_path in self.manifest_paths:
            with open(manifest_path, "r") as fp:
                manifest = json.load(fp)

            version = manifest.get("schema_version")
            if version not in SUPPORTED_SCHEMA_VERSIONS:
                raise ValueError(
                    f"{manifest_path}: schema_version {version} is not one of "
                    f"{SUPPORTED_SCHEMA_VERSIONS}"
                )

            test_every = manifest.get("test_every")
            if test_every:
                self.test_every = int(test_every)

            # Absent on schema 1, which predates dynamic support and can only
            # have been produced by a static bake.
            manifest_dynamic = bool(manifest.get("dynamic", False))

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
                self.is_dynamic.append(manifest_dynamic)
                self.image_ids.append(
                    self._resolve_image_id(int(entry["global_index"]), manifest_path)
                )
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

    def _resolve_image_id(self, global_index: int, manifest_path: str) -> int:
        """Split-local training index of the real frame this view came from.

        ``NCoreDataset`` assigns ``image_id = item``, the position within
        ``indices = all[all % test_every != 0]``. Mapping through that same array
        with ``searchsorted`` -- rather than re-deriving the arithmetic -- means
        this cannot drift if the split rule ever changes, and it turns a frame
        that is not in the train split into a hard error instead of a silently
        wrong row.

        Returns 0 when the trainer's split was not supplied, preserving the
        original constant-``image_id`` behaviour for standalone use.
        """
        if self._train_indices is None:
            return 0
        pos = int(np.searchsorted(self._train_indices, global_index))
        if (
            pos >= len(self._train_indices)
            or self._train_indices[pos] != global_index
        ):
            raise ValueError(
                f"{manifest_path}: global_index {global_index} is not in the "
                "trainer's train split, so it owns no per-image row. A "
                "validation frame reached the bank, or the manifest was built "
                "against a different test_every."
            )
        return pos

    @property
    def has_dynamic(self) -> bool:
        """Whether any view was cleaned from a render containing rigid objects.

        The Runner reads this to reject a dynamic bank paired with
        ``enable_dynamic=false``: those images contain vehicles the model cannot
        render, so the photometric loss would drive the background Gaussians to
        paint them in permanently.
        """
        return any(self.is_dynamic)

    def _print_summary(self) -> None:
        by_shift: Dict[str, int] = {}
        for name in self.shift_names:
            by_shift[name] = by_shift.get(name, 0) + 1
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(by_shift.items()))
        n_dyn = sum(self.is_dynamic)
        # Printed even when uniform: a bank loaded in the wrong mode is
        # otherwise invisible until the vehicles look wrong many steps later.
        kind = (
            "static"
            if n_dyn == 0
            else ("dynamic" if n_dyn == len(self.entries) else
                  f"MIXED {n_dyn} dynamic / {len(self.entries) - n_dyn} static")
        )
        print(
            f"[PseudoViewParser] Loaded {len(self.entries)} views from "
            f"{len(self.manifest_paths)} manifest(s) across "
            f"{self.num_cameras} cameras [{breakdown}] ({kind})"
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
            normalized frame. ``timestamp_us`` is present only for entries whose
            manifest reported dynamic renders -- see the module docstring.
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

        sample: Dict[str, Any] = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(self.parser.camtoworlds[index]).float(),
            "image": torch.from_numpy(image).float(),
            # The split-local row of the REAL frame this view was rendered from
            # (0 when the trainer's split was not supplied). image_id indexes
            # per-image tables sized len(trainset) -- appearance embeddings, the
            # bilateral grid, PPISP, pose deltas. Sharing the source frame's row
            # is not just safe but correct: same camera, same instant, same
            # exposure and white balance.
            "image_id": int(self.parser.image_ids[index]),
            "camera_idx": int(self.parser.camera_idx_per_frame[index]),
        }

        # The render this was cleaned from had vehicles in it, so the loss must
        # render them too. The timestamp is the REAL frame's capture time, which
        # is what the rigid track index is keyed on -- verified to agree with
        # the renderer's own camera-local frame index on both 200- and
        # 201-frame scenes.
        if self.parser.is_dynamic[index]:
            sample["timestamp_us"] = int(self.parser.frame_timestamps_us[index])

        return sample
