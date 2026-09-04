"""Locate a trained 4DGS run's checkpoint and camera metadata on disk.

The pipeline has gone through three generations, each with its own results root
and its own step-directory naming::

    v1  results/<dataset>/<scene>/04_gsplat_training_<hash>/
    v2  results/4dgs/<dataset>/<scene>/20_gsplat_dynamic_<hash>/
    v3  results/extended_4dgs/<dataset>/<scene>/07_difix_4dgs_<hash>/round_NNN/train/

v2 and v3 training directories have an identical interior (``ckpts/``,
``camera_paths/``, ``cfg.yml``); v3 only adds the ``round_NNN/train/`` nesting
because it trains one checkpoint per Difix loop round, of which the last is the
one to use. v1 predates ``camera_paths/`` entirely and so cannot drive the
simulator -- it is dropped by the usability filter rather than special-cased,
which is also what keeps this module tolerant of future step renames.

Callers should try their exact config-hash path first -- that is what makes "run
this specific configuration" reproducible -- and fall back to :func:`resolve_run`,
which answers the looser question "what is the newest usable run for this
dataset/scene?". Config hashes fold in a growing set of settings, so they stop
resolving as the pipeline evolves; the fallback is what keeps older runs
reachable.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence


# Directory-name globs for each generation's training step. Order does not matter
# (candidates are ranked by checkpoint mtime); this is just the set of places a
# trained run can live.
TRAIN_STEP_GLOBS = (
    "07_difix_4dgs_*",       # v3: Difix loop, one train dir per round
    "20_gsplat_dynamic_*",   # v2
    "04_gsplat_training_*",  # v1 (no camera_paths/ -> rejected by _is_usable)
)

# Mirrors dynamic.rigid_tracks.DEFAULT_RIGID_CLASSES. Duplicated only as a
# fallback so this module stays importable without gsplat present; the real
# tuple is preferred whenever it can be imported.
_FALLBACK_RIGID_CLASSES = ("car", "truck", "bus")


class RunNotFoundError(RuntimeError):
    """No usable training run could be located."""


@dataclass
class RunPaths:
    """A resolved training run -- everything needed to load it."""

    train_dir: str
    checkpoint_path: str
    camera_paths_dir: str
    cfg_yml: Optional[str]
    source: str  # "pin" | "latest" -- how it was found, for logging

    def describe(self, root: Optional[str] = None) -> str:
        """One-line human summary, e.g. for the fallback warning."""
        shown = os.path.relpath(self.train_dir, root) if root else self.train_dir
        stamp = ""
        try:
            import time

            stamp = time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(self.checkpoint_path))
            )
        except OSError:
            pass
        ckpt = os.path.splitext(os.path.basename(self.checkpoint_path))[0]
        return f"{shown}  ({ckpt}{', ' + stamp if stamp else ''})"


def find_latest_checkpoint(ckpt_dir: str) -> str:
    """Highest-step ``ckpt_<step>_rank0.pt`` in ``ckpt_dir``, or "" if none."""
    if not os.path.isdir(ckpt_dir):
        return ""

    def step_of(fname: str) -> int:
        try:
            return int(fname.split("_")[1])
        except (IndexError, ValueError):
            return -1

    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")], key=step_of)
    return os.path.join(ckpt_dir, ckpts[-1]) if ckpts else ""


def _is_usable(train_dir: str) -> bool:
    """True when ``train_dir`` has both a checkpoint and camera metadata.

    This is what excludes v1 runs: they predate ``camera_paths/``, so the
    simulator has no trajectory or intrinsics to render from.
    """
    if not find_latest_checkpoint(os.path.join(train_dir, "ckpts")):
        return False
    return bool(
        glob.glob(os.path.join(train_dir, "camera_paths", "*", "camera_data.json"))
    )


def _rounds_from_loop_state(loop_dir: str) -> List[str]:
    """Round train dirs for a v3 loop dir, in round order.

    Prefers ``loop_state.json`` (authoritative: it records every round the loop
    actually ran) and falls back to globbing ``round_*/train`` when the file is
    missing or malformed. Recorded ``train_dir`` values are absolute, so they go
    stale if the repo moves -- the path is reconstructed from the loop dir
    whenever the recorded one is gone.
    """
    state_path = os.path.join(loop_dir, "loop_state.json")
    try:
        with open(state_path) as fp:
            rounds = json.load(fp)["rounds"]
    except (OSError, ValueError, KeyError, TypeError):
        return sorted(glob.glob(os.path.join(loop_dir, "round_*", "train")))

    out: List[str] = []
    for entry in rounds:
        if not isinstance(entry, dict):
            continue
        recorded = entry.get("train_dir")
        if recorded and os.path.isdir(recorded):
            out.append(recorded)
            continue
        idx = entry.get("round")
        if idx is not None:
            rebuilt = os.path.join(loop_dir, f"round_{int(idx):03d}", "train")
            if os.path.isdir(rebuilt):
                out.append(rebuilt)
    return out or sorted(glob.glob(os.path.join(loop_dir, "round_*", "train")))


def _train_dirs_of(candidate: str) -> List[str]:
    """Expand one step directory into the train dirs it contains.

    A v3 loop dir holds one per round; v1/v2 step dirs are themselves the train
    dir. Detection is structural (does it look like a loop dir?) rather than
    name-based, so a renamed step still resolves.
    """
    is_loop = os.path.exists(os.path.join(candidate, "loop_state.json")) or glob.glob(
        os.path.join(candidate, "round_*")
    )
    return _rounds_from_loop_state(candidate) if is_loop else [candidate]


def _scene_dir(results_root: str, dataset: str, scene: str) -> str:
    return os.path.abspath(os.path.join(results_root, dataset, scene))


def resolve_run(
    results_root: str,
    dataset: str,
    scene: str,
    pin: Optional[str] = None,
) -> RunPaths:
    """Find the newest usable training run for ``dataset``/``scene``.

    Args:
        results_root: Pipeline results root, e.g. ``results/extended_4dgs``.
        dataset: Dataset name (``dataset.name``).
        scene: Scene name (``dataset.scene``).
        pin: Optional explicit directory -- either a train dir or a v3 loop dir.
            When given, only it is considered.

    Returns:
        The resolved :class:`RunPaths`.

    Raises:
        RunNotFoundError: when nothing usable is found. Never crosses into
            another scene or another results root.
    """
    if pin:
        pinned = os.path.abspath(pin)
        if not os.path.isdir(pinned):
            raise RunNotFoundError(f"simulator.run_dir does not exist: {pinned}")
        usable = [d for d in _train_dirs_of(pinned) if _is_usable(d)]
        if not usable:
            raise RunNotFoundError(
                f"simulator.run_dir has no usable training run (needs ckpts/*.pt "
                f"and camera_paths/*/camera_data.json): {pinned}"
            )
        return _build(usable[-1], source="pin")

    scene_dir = _scene_dir(results_root, dataset, scene)
    if not os.path.isdir(scene_dir):
        raise RunNotFoundError(
            f"No results directory for this scene: {scene_dir}\n"
            f"  (results_root={results_root}, dataset={dataset}, scene={scene})"
        )

    candidates: List[str] = []
    for pattern in TRAIN_STEP_GLOBS:
        for step_dir in sorted(glob.glob(os.path.join(scene_dir, pattern))):
            candidates.extend(_train_dirs_of(step_dir))

    usable = [d for d in candidates if _is_usable(d)]
    if not usable:
        detail = (
            f"  {len(candidates)} training dir(s) found but none had both "
            f"ckpts/*.pt and camera_paths/*/camera_data.json"
            if candidates
            else "  no training step directories matched "
            + ", ".join(TRAIN_STEP_GLOBS)
        )
        raise RunNotFoundError(f"No usable training run under {scene_dir}\n{detail}")

    newest = max(usable, key=lambda d: os.path.getmtime(_ckpt_of(d)))
    return _build(newest, source="latest")


def _ckpt_of(train_dir: str) -> str:
    return find_latest_checkpoint(os.path.join(train_dir, "ckpts"))


def _build(train_dir: str, source: str) -> RunPaths:
    cfg_yml = os.path.join(train_dir, "cfg.yml")
    return RunPaths(
        train_dir=train_dir,
        checkpoint_path=_ckpt_of(train_dir),
        camera_paths_dir=os.path.join(train_dir, "camera_paths"),
        cfg_yml=cfg_yml if os.path.exists(cfg_yml) else None,
        source=source,
    )


def _load_train_cfg(cfg_yml: str) -> Optional[dict]:
    """Parse a training ``cfg.yml``.

    The trainer dumps its config dataclass with plain ``yaml.dump``, so the file
    carries ``!!python/...`` tags -- tuples, and whole objects such as
    ``gsplat.strategy.mcmc.MCMCStrategy``. Safe loaders reject those. Only three
    scalar keys are needed here, so the entire ``python/`` tag family is
    neutralised (tuples become tuples, everything else becomes ``None``) instead
    of reaching for ``UnsafeLoader``: these files are ours, but nothing here has
    any reason to be able to instantiate arbitrary Python.
    """
    try:
        import yaml

        class _Loader(yaml.SafeLoader):
            pass

        def _python_tag(loader, suffix, node):
            if suffix.startswith("tuple") and isinstance(node, yaml.SequenceNode):
                return tuple(loader.construct_sequence(node))
            return None

        _Loader.add_multi_constructor("tag:yaml.org,2002:python/", _python_tag)
        with open(cfg_yml) as fp:
            data = yaml.load(fp, Loader=_Loader)
        return data if isinstance(data, dict) else None
    except Exception as exc:  # noqa: BLE001 - discovery must never be fatal
        print(f"[run_discovery] could not read {cfg_yml}: {exc}")
        return None


def load_instance_ids(
    cfg_yml: Optional[str], num_instances: int
) -> Optional[List[int]]:
    """Recover the rigid instance column -> original track ID mapping.

    ``RigidNodes.instance_ids`` is a plain Python list, not a registered buffer,
    so it does not survive into the checkpoint -- only anonymous columns
    ``0..M-1`` do. It is recoverable because the training ``cfg.yml`` records the
    tracks JSON and the exact filter used, and columns are simply the filtered
    track IDs in sorted order (see ``dynamic.rigid_tracks.load_rigid_tracks``).

    Returns:
        ``[track_id]`` indexed by instance column, or ``None`` when it cannot be
        rebuilt or disagrees with ``num_instances`` -- callers should then fall
        back to addressing objects by column, rather than trust a wrong mapping.
    """
    if not cfg_yml or not os.path.exists(cfg_yml):
        return None

    cfg = _load_train_cfg(cfg_yml)
    if cfg is None:
        return None
    tracks_json = cfg.get("dynamic_tracks_json")
    classes = cfg.get("dynamic_rigid_classes")
    min_score = float(cfg.get("dynamic_min_track_score") or 0.0)

    if not tracks_json or not os.path.exists(str(tracks_json)):
        return None

    try:
        from dynamic.rigid_tracks import DEFAULT_RIGID_CLASSES as _default
    except Exception:  # noqa: BLE001 - keeps this module importable without gsplat
        _default = _FALLBACK_RIGID_CLASSES
    keep = set(classes) if classes else set(_default)

    try:
        with open(str(tracks_json)) as fp:
            results = json.load(fp)["results"]
    except (OSError, ValueError, KeyError) as exc:
        print(f"[run_discovery] could not read tracks JSON {tracks_json}: {exc}")
        return None

    ids = {
        int(box["tracking_id"])
        for boxes in results.values()
        for box in boxes
        if box.get("tracking_name") in keep
        and float(box.get("tracking_score", 1.0)) >= min_score
    }
    instance_ids = sorted(ids)

    if len(instance_ids) != num_instances:
        print(
            f"[run_discovery] WARNING: track-ID mapping disagrees with the "
            f"checkpoint ({len(instance_ids)} filtered tracks vs {num_instances} "
            f"instances); falling back to column indices."
        )
        return None
    return instance_ids


def load_class_names(cfg_yml: Optional[str], instance_ids: Sequence[int]) -> List[str]:
    """Class name per instance column, or "" where unknown (display only)."""
    if not cfg_yml or not os.path.exists(cfg_yml) or not instance_ids:
        return ["" for _ in instance_ids]
    cfg = _load_train_cfg(cfg_yml)
    if cfg is None:
        return ["" for _ in instance_ids]
    try:
        with open(str(cfg.get("dynamic_tracks_json"))) as fp:
            results = json.load(fp)["results"]
    except Exception:  # noqa: BLE001 - cosmetic only
        return ["" for _ in instance_ids]

    by_id: dict = {}
    for boxes in results.values():
        for box in boxes:
            by_id.setdefault(int(box["tracking_id"]), box.get("tracking_name", ""))
    return [by_id.get(int(tid), "") for tid in instance_ids]
