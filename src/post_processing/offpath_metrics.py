"""Distributional metrics for off-trajectory renders: KID (primary) and FID.

Off the recorded trajectory no photograph exists, so no paired metric (PSNR,
SSIM, LPIPS) is definable. What remains is a distributional comparison: are the
renders from N metres off-path drawn from the same distribution of images as real
driving photographs of this scene? See docs/06-evaluation/off-trajectory-metrics.md
for the full argument; the short version of why KID leads and FID accompanies it:

  * The deliverable is a CURVE over lateral offset. Every FID point carries a bias
    term that depends on the generated set's covariance structure, and that
    structure changes systematically with offset (renders degrade as the camera
    leaves observed space). So FID's bias varies along the very axis being
    plotted. KID's estimator is unbiased at every offset.
  * FID's bias is model-dependent (Chong & Forsyth, CVPR 2020), so variant A can
    beat variant B on bias alone -- fatal for an orthogonality claim across
    variants.
  * N is ~1000 per shift level here, 50x short of FID's 50k convention.
  * FID is still reported, because Difix3D+, UniSim, NeuRAD and ReconDreamer all
    report it and it is nearly free once the Inception features exist.

Definitions, both over Inception-V3 pool3 features (d = 2048):

    KID = MMD_u^2 with k(x, y) = ((1/d) x'y + 1)^3     (unbiased)
    FID = ||mu_r - mu_g||^2 + Tr(S_r + S_g - 2 (S_r S_g)^(1/2))

Reference set is the stage ``035_real_frames`` bank, which is 960x540 PNG and
byte-identical to what training consumed -- so it sits in the same pixel domain
as the renders. Scoring against the raw 1920x1080 JPEGs instead would introduce
both a resize and a JPEG-vs-PNG difference between the two sets, which is exactly
the low-level confound clean-fid (Parmar et al., CVPR 2022) documents.

Usage (standalone, over an existing round):

    PYTHONPATH=. envs/envs/env_gsplat/bin/python src/post_processing/offpath_metrics.py \\
        ++metrics_task.render_dir=<round>/render/full \\
        ++metrics_task.real_bank_dir=<...>/035_real_frames_<h> \\
        ++metrics_task.output_dir=<round>/metrics
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

# --------------------------------------------------------------------------- #
# Shift discovery                                                             #
# --------------------------------------------------------------------------- #

def parse_shift_metres(shift_name: str) -> Optional[List[float]]:
    """Invert ``render_standalone._shift_name``: ``"X_-1"`` -> ``[-1, 0, 0]``.

    Kept local rather than imported from ``difix_pseudo_views`` so scoring does
    not pull in the diffusion stack.
    """
    if shift_name == "original":
        return [0.0, 0.0, 0.0]
    axes = {"X": 0, "Y": 1, "Z": 2}
    out = [0.0, 0.0, 0.0]
    matches = re.findall(r"([XYZ])_(-?\d+(?:\.\d+)?)", shift_name)
    if not matches:
        return None
    for axis, value in matches:
        out[axes[axis]] = float(value)
    return out


def list_shifts(render_dir: str) -> List[str]:
    """Shift directories under ``<render_dir>/frames``, sorted by |offset|.

    Sorting by magnitude rather than by name puts the curve's x-axis in order,
    so ``original`` leads and the largest shift is last.
    """
    root = os.path.join(render_dir, "frames")
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"{root} not found. Point render_dir at a render mode directory, "
            "e.g. <round>/render/full"
        )
    names = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

    def key(n: str) -> Tuple[float, str]:
        m = parse_shift_metres(n)
        return (float(np.linalg.norm(m)) if m else float("inf"), n)

    return sorted(names, key=key)


# --------------------------------------------------------------------------- #
# Ego masking                                                                 #
# --------------------------------------------------------------------------- #

def load_ego_masks(
    data_dir: Optional[str],
    cameras: Sequence[str],
    n_probe: int = 8,
) -> Dict[str, np.ndarray]:
    """Per-camera invalid-pixel masks, ``True`` where the pixel is excluded.

    Training excludes these pixels from the loss, so the model never learns them
    and its renders put arbitrary content there while the real frames show real
    bodywork. Measured on ``scene_084`` the excluded fraction is **very uneven
    across the rig** -- camera7 5.11%, camera9 2.92%, camera10 0.84%, camera1
    0.05%, camera8 0.04% -- so leaving it unmasked does not merely add a constant
    offset to every score, it biases each camera differently and corrupts the
    per-camera comparison specifically.

    Two mask paths exist in the parser and only the second is populated here:

      * a static per-camera ``"ego"`` mask from ``sensor.get_mask_images()``
        (``parser.mask_dict``) -- **absent** in this dataset, every entry is None;
      * a per-frame generic ``"mask"`` component, which is what the dataset
        actually emits and what training used.

    The WayveScenes101 masks combine **two** things with different behaviour, and
    separating them is what makes a static mask defensible:

      * **ego bodywork** -- static in the camera frame, the large blob that
        touches the bottom image edge (11670 px, unchanging, on ``right-forward``);
      * **privacy redactions** -- small blobs over faces and number plates that
        MOVE frame to frame (on ``front-forward``, one ~1000 px blob tracking a
        receding plate across rows 667 -> 663 -> 657).

    So ``n_probe`` frames per camera are checked and the **INTERSECTION** is taken:
    a pixel masked in *every* probed frame is the static ego region, while moving
    privacy blobs drop out by construction. Taking the union instead would
    accumulate every position a privacy blob ever visited and mask pixels that are
    valid in almost every frame -- measured, that inflated camera1 from 0.05% to
    0.35% at ``n_probe=8``, and it gets worse the more frames are probed.

    Privacy regions therefore go unmasked. That is deliberate: they are ~0.05-0.1%
    of pixels and, like the ego region, are present at similar rates in every frame
    regardless of camera offset, so they raise the FLOOR of KID-vs-real without
    changing its SLOPE. They cost absolute comparability, not the shape of the
    curve section 6.4 claims. Exact handling would mask per frame, pairing each
    render with its own frame's mask.
    """
    if not data_dir:
        return {}
    try:
        import sys
        sys.path.insert(0, os.path.abspath("src/gsplat_training"))
        from datasets.ncore import NCoreParser, NCoreDataset  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"  [metrics] ego masks unavailable ({type(exc).__name__}: {exc})")
        return {}
    try:
        parser = NCoreParser(meta_json_path=data_dir, camera_ids=list(cameras))
        dataset = NCoreDataset(parser, split="train")
    except Exception as exc:
        print(f"  [metrics] ego masks unavailable ({type(exc).__name__}: {exc})")
        return {}

    cams = list(cameras)
    acc: Dict[str, np.ndarray] = {}
    probes: Dict[str, int] = {c: 0 for c in cams}
    varying: List[str] = []
    # Strided scan, not sequential: the frame list is camera-major, so walking it
    # in order would decode every frame of camera 1 before reaching camera 5.
    # Each __getitem__ decodes a full 1920x1080 image, so that costs minutes for
    # information a handful of frames provides. A stride visits all cameras early
    # without assuming the exact layout.
    n = len(dataset)
    stride = max(1, n // max(1, len(cams) * n_probe * 3))
    order = list(range(0, n, stride)) + [i for i in range(n) if i % stride]
    for i in order:
        if all(v >= n_probe for v in probes.values()):
            break
        sample = dataset[i]
        cam = cams[int(sample["camera_idx"])]
        if probes.get(cam, n_probe) >= n_probe:
            continue
        m = sample.get("mask")
        if m is None:
            probes[cam] = n_probe
            continue
        invalid = ~np.asarray(m).astype(bool)   # dataset gives True = VALID
        probes[cam] += 1
        if cam not in acc:
            acc[cam] = invalid
        else:
            if not np.array_equal(acc[cam], invalid) and cam not in varying:
                varying.append(cam)
            # INTERSECTION: keeps the static ego region, drops moving privacy blobs.
            acc[cam] = acc[cam] & invalid

    masks = {c: m for c, m in acc.items() if m.any()}
    if varying:
        print(f"  [metrics] mask varies between frames on {varying} "
              "(privacy redactions move); intersected to the static ego region")
    absent = [c for c in cams if c not in masks]
    if absent:
        print(f"  [metrics] no excluded pixels on {absent}; scored unmasked")
    if masks:
        frac = ", ".join(f"{c} {100 * masks[c].mean():.2f}%" for c in sorted(masks))
        print(f"  [metrics] masked fraction: {frac}")
    return masks


# --------------------------------------------------------------------------- #
# Feature extraction                                                          #
# --------------------------------------------------------------------------- #

def _read_uint8(path: str, mask: Optional[np.ndarray]) -> torch.Tensor:
    """One image as ``uint8`` CHW, with masked pixels zeroed.

    The mask is resized by nearest-neighbour when it was authored at a different
    resolution than the render, so a 1920x1080 mask still applies to a 960x540
    image without inventing intermediate values.
    """
    arr = np.asarray(Image.open(path).convert("RGB"))
    if mask is not None and mask.shape[:2] != arr.shape[:2]:
        m = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (arr.shape[1], arr.shape[0]), Image.NEAREST
        )
        mask = np.asarray(m) != 0
    if mask is not None:
        arr = arr.copy()
        arr[mask] = 0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@torch.no_grad()
def embed(
    paths: Sequence[str],
    inception,
    mask: Optional[np.ndarray],
    device: str,
    batch_size: int = 32,
) -> torch.Tensor:
    """Inception-V3 pool3 features for a list of images, as ``(N, 2048)`` float64.

    float64 because FID's covariance and matrix square root are numerically
    touchy at float32 in 2048 dimensions.
    """
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = torch.stack([_read_uint8(p, mask) for p in paths[i : i + batch_size]])
        feats.append(inception(batch.to(device)).double().cpu())
    return torch.cat(feats, dim=0) if feats else torch.empty(0, 2048, dtype=torch.float64)


# --------------------------------------------------------------------------- #
# The metrics                                                                 #
# --------------------------------------------------------------------------- #

def kid_from_features(
    f_real: torch.Tensor,
    f_fake: torch.Tensor,
    subset_size: int,
    subsets: int,
    seed: int,
    degree: int = 3,
    gamma: Optional[float] = None,
    coef: float = 1.0,
) -> Tuple[float, float, int]:
    """KID mean and standard deviation over random subsets.

    ``subset_size`` must be strictly LESS than the sample count, not merely <=.
    At ``subset_size == n`` every subset is the same set in a different order,
    and MMD^2 is permutation-invariant, so every subset returns an identical
    value and the reported std is exactly 0 -- a fabricated error bar. This
    clamps to ``n - 1`` and reports the value actually used.

    The std is a resampling spread over subsets of a fixed sample, not a
    confidence interval on the true KID; it understates uncertainty about the
    underlying distributions.
    """
    from torchmetrics.image.kid import poly_mmd

    n = min(f_real.shape[0], f_fake.shape[0])
    m = max(2, min(subset_size, n - 1))
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(subsets):
        ir = torch.randperm(f_real.shape[0], generator=g)[:m]
        if_ = torch.randperm(f_fake.shape[0], generator=g)[:m]
        vals.append(poly_mmd(f_real[ir], f_fake[if_], degree, gamma, coef))
    v = torch.stack(vals)
    return float(v.mean()), float(v.std(unbiased=False)), m


def fid_from_features(f_real: torch.Tensor, f_fake: torch.Tensor) -> float:
    """FID between two feature sets. Reported for comparability, not for claims."""
    from torchmetrics.image.fid import _compute_fid

    mu_r, mu_f = f_real.mean(0), f_fake.mean(0)
    s_r = f_real.T.cov()
    s_f = f_fake.T.cov()
    return float(_compute_fid(mu_r, s_r, mu_f, s_f))


def noise_floor(f_real: torch.Tensor, subset_size: int, subsets: int, seed: int) -> Dict:
    """KID between two DISJOINT halves of the real set.

    Both halves are real photographs, so the true value is ~0 and what comes back
    is the measurement's resolution at this sample size and feature extractor.
    A difference between two shift levels smaller than this floor is not
    resolvable, and saying so is stronger than plotting it.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(f_real.shape[0], generator=g)
    half = f_real.shape[0] // 2
    a, b = f_real[perm[:half]], f_real[perm[half : 2 * half]]
    mean, std, used = kid_from_features(a, b, subset_size, subsets, seed)
    return {"kid_mean": mean, "kid_std": std, "subset_size_used": used,
            "n_per_half": half, "fid": fid_from_features(a, b)}


# --------------------------------------------------------------------------- #
# Scoring a round                                                             #
# --------------------------------------------------------------------------- #

def real_paths_by_camera(real_bank_dir: str) -> Dict[str, List[str]]:
    """Reference images per camera, from the stage 035 bank's ``index.json``."""
    with open(os.path.join(real_bank_dir, "index.json")) as fp:
        index = json.load(fp)
    out: Dict[str, List[str]] = {}
    for cam, rec in index["cameras"].items():
        out[cam] = [os.path.join(real_bank_dir, f["image"]) for f in rec["frames"]]
    return out


def render_paths(render_dir: str, shift: str, camera: str) -> List[str]:
    d = os.path.join(render_dir, "frames", shift, camera)
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in sorted(os.listdir(d))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))]


def score(
    render_dir: str,
    real_bank_dir: str,
    cameras: Optional[Sequence[str]] = None,
    data_dir: Optional[str] = None,
    subset_size: int = 500,
    subset_size_per_camera: int = 100,
    subsets: int = 100,
    seed: int = 0,
    device: str = "cuda",
    mask_ego: bool = True,
) -> Dict:
    """KID and FID for every shift level, per camera and pooled.

    Real features are embedded ONCE and reused across every shift level, which is
    both faster and strictly more correct than re-embedding: the reference is
    identical at every point on the curve, so it should not be resampled.
    """
    from torchmetrics.image.fid import NoTrainInceptionV3

    real_by_cam = real_paths_by_camera(real_bank_dir)
    cams = list(cameras) if cameras else sorted(real_by_cam)
    cams = [c for c in cams if c in real_by_cam]
    if not cams:
        raise ValueError(f"no requested camera present in {real_bank_dir}")

    masks = load_ego_masks(data_dir, cams) if mask_ego else {}
    print(f"  [metrics] cameras={cams} ego_mask={'yes' if masks else 'NO'}")

    inception = NoTrainInceptionV3(
        name="inception-v3-compat", features_list=["2048"]
    ).to(device).eval()

    # Reference features, once.
    f_real_cam = {c: embed(real_by_cam[c], inception, masks.get(c), device) for c in cams}
    f_real_all = torch.cat([f_real_cam[c] for c in cams], dim=0)
    n_real = {c: int(f_real_cam[c].shape[0]) for c in cams}
    print(f"  [metrics] reference embedded: {int(f_real_all.shape[0])} frames")

    shifts = list_shifts(render_dir)
    results: List[Dict] = []
    for shift in shifts:
        f_fake_cam = {}
        for c in cams:
            paths = render_paths(render_dir, shift, c)
            if not paths:
                continue
            f_fake_cam[c] = embed(paths, inception, masks.get(c), device)
        if not f_fake_cam:
            print(f"  [metrics] {shift}: no renders, skipped")
            continue

        for c, ff in f_fake_cam.items():
            mean, std, used = kid_from_features(
                f_real_cam[c], ff, subset_size_per_camera, subsets, seed)
            results.append({
                "shift_name": shift, "shift_m": parse_shift_metres(shift),
                "camera": c, "kid_mean": mean, "kid_std": std,
                "fid": fid_from_features(f_real_cam[c], ff),
                "n_real": n_real[c], "n_fake": int(ff.shape[0]),
                "subset_size_used": used,
            })

        f_fake_all = torch.cat([f_fake_cam[c] for c in cams if c in f_fake_cam], dim=0)
        mean, std, used = kid_from_features(f_real_all, f_fake_all, subset_size, subsets, seed)
        results.append({
            "shift_name": shift, "shift_m": parse_shift_metres(shift),
            "camera": "pooled", "kid_mean": mean, "kid_std": std,
            "fid": fid_from_features(f_real_all, f_fake_all),
            "n_real": int(f_real_all.shape[0]), "n_fake": int(f_fake_all.shape[0]),
            "subset_size_used": used,
        })
        pooled = results[-1]
        print(f"  [metrics] {shift:12s} KID {pooled['kid_mean']:.5f} "
              f"+/- {pooled['kid_std']:.5f}   FID {pooled['fid']:7.2f}   "
              f"n={pooled['n_fake']}")

    floor = noise_floor(f_real_all, subset_size, subsets, seed)
    print(f"  [metrics] noise floor (real vs real) KID {floor['kid_mean']:.5f} "
          f"FID {floor['fid']:.2f}")

    # Provenance is part of the measurement: none of these choices is comparable
    # across papers if left implicit, and a metric whose configuration is not
    # recorded cannot be re-derived.
    return {
        "schema_version": 1,
        "config": {
            "render_dir": render_dir, "real_bank_dir": real_bank_dir,
            "reference_set": "035_real_frames (all frames, train and val)",
            "cameras": cams, "ego_masked": bool(masks),
            "subset_size_pooled": subset_size,
            "subset_size_per_camera": subset_size_per_camera,
            "subsets": subsets, "seed": seed,
            "feature_extractor": "InceptionV3 pool3 (2048)",
            "kid_kernel": "poly degree=3 gamma=1/d coef=1",
        },
        "noise_floor": floor,
        "rows": results,
    }


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #

def _fmt_hms(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import time

    task = cfg.metrics_task
    out_dir = task.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Same resume convention as the other stages: the orchestrator only checks
    # that .success exists, so a completed scoring pass is skipped on re-run.
    marker = os.path.join(out_dir, ".success")
    if os.path.exists(marker) and not bool(task.get("force", False)):
        print(f"  [metrics] cache hit: {out_dir}")
        return

    started = time.perf_counter()
    result = score(
        render_dir=task.render_dir,
        real_bank_dir=task.real_bank_dir,
        cameras=list(task.cameras) if task.get("cameras") else None,
        data_dir=task.get("data_dir"),
        subset_size=int(task.subset_size),
        subset_size_per_camera=int(task.subset_size_per_camera),
        subsets=int(task.subsets),
        seed=int(task.seed),
        device=str(task.get("device", "cuda")),
        mask_ego=bool(task.get("mask_ego", True)),
    )

    path = os.path.join(out_dir, "offpath_metrics.json")
    with open(path, "w") as fp:
        json.dump(result, fp, indent=2)
    print(f"  [metrics] wrote {path}")

    elapsed = time.perf_counter() - started
    n_levels = len({r["shift_name"] for r in result["rows"]})
    with open(marker, "w") as fp:
        fp.write(
            f"off-path metrics: {n_levels} shift levels, "
            f"{len(result['config']['cameras'])} cameras\n"
            f"duration: {_fmt_hms(elapsed)}\n"
            f"  ego_masked  {result['config']['ego_masked']}\n"
            f"  noise_floor_kid  {result['noise_floor']['kid_mean']:.6f}\n"
        )


if __name__ == "__main__":
    main()
