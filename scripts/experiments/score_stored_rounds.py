"""Score every stored loop round off-path (KID / FID), writing beside each round.

One-off backfill for rounds that predate the metrics stage. New rounds score
themselves -- ``run_metrics`` is wired into both orchestrators after the render
step -- so this exists to cover history, not as a parallel code path.

Why it is not just a shell loop over the stage: the reference set is per SCENE,
not per round, and embedding 1000 real frames through Inception is the expensive
part. Grouping rounds by scene and reusing both the real features and the ego mask
turns 22 embeddings of the reference into 4.

Usage:
    PYTHONPATH=. envs/envs/env_gsplat/bin/python \\
        scripts/experiments/score_stored_rounds.py [--dry-run] [--force]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath("src/post_processing"))

import torch  # noqa: E402

from offpath_metrics import (  # noqa: E402
    embed, fid_from_features, kid_from_features, list_shifts, load_ego_masks,
    noise_floor, parse_shift_metres, real_paths_by_camera, render_paths,
)


def _fmt(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m:d}m{s:02d}s"


def scene_of(path: str) -> Optional[str]:
    m = re.search(r"/(scene_\d+)/", path)
    return m.group(1) if m else None


def find_rounds() -> Dict[str, List[str]]:
    """Round directories with renders, grouped by scene."""
    out: Dict[str, List[str]] = defaultdict(list)
    for d in sorted(glob.glob("results/**/round_*/render/full", recursive=True)):
        rd = os.path.dirname(os.path.dirname(d))
        s = scene_of(rd)
        if s:
            out[s].append(rd)
    return out


def find_real_bank(scene: str) -> Optional[str]:
    """The stage 035 bank for a scene, preferring one with a readable index."""
    for d in sorted(glob.glob(f"results/**/{scene}/035_real_frames_*", recursive=True)):
        if os.path.exists(os.path.join(d, "index.json")):
            return d
    return None


def find_ncore_json(scene: str) -> Optional[str]:
    """Any NCore meta-json for the scene, used only to load the ego mask."""
    pats = [f"results/**/{scene}/03_ncore_ego_*/ncore_dataset/*.json",
            f"results/**/{scene}/03_ncore_dataset_*/ncore_dataset/*.json"]
    for pat in pats:
        for f in sorted(glob.glob(pat, recursive=True)):
            if not f.endswith("index.json"):
                return f
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="list work, score nothing")
    ap.add_argument("--force", action="store_true", help="re-score completed rounds")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--subset-size", type=int, default=500)
    ap.add_argument("--subset-size-per-camera", type=int, default=100)
    ap.add_argument("--subsets", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    by_scene = find_rounds()
    total = sum(len(v) for v in by_scene.values())
    print(f"{total} rounds with renders across {len(by_scene)} scenes\n")

    plan = []
    for scene in sorted(by_scene):
        bank = find_real_bank(scene)
        nj = find_ncore_json(scene)
        for rd in by_scene[scene]:
            done = os.path.exists(os.path.join(rd, "metrics", ".success"))
            plan.append((scene, rd, bank, nj, done))
        state = "OK" if bank else "NO REAL BANK -- will skip"
        print(f"  {scene}: {len(by_scene[scene])} rounds, bank={state}, "
              f"ego_mask={'yes' if nj else 'no'}")

    todo = [p for p in plan if p[2] and (args.force or not p[4])]
    skipped = [p for p in plan if p[4] and not args.force]
    nobank = [p for p in plan if not p[2]]
    print(f"\nto score: {len(todo)}   already done: {len(skipped)}   "
          f"no bank: {len(nobank)}")
    if args.dry_run:
        for _, rd, _, _, _ in todo:
            print(f"  would score {rd}")
        return 0
    if not todo:
        print("nothing to do")
        return 0

    from torchmetrics.image.fid import NoTrainInceptionV3
    inception = NoTrainInceptionV3(
        name="inception-v3-compat", features_list=["2048"]
    ).to(args.device).eval()

    ok = fail = 0
    t0 = time.perf_counter()
    for scene in sorted({p[0] for p in todo}):
        rounds = [p for p in todo if p[0] == scene]
        bank, nj = rounds[0][2], rounds[0][3]
        real_by_cam = real_paths_by_camera(bank)
        cams = sorted(real_by_cam)

        # Per scene, once: the reference features and the ego mask. This is the
        # whole reason this script exists rather than a shell loop.
        print(f"\n=== {scene}: {len(rounds)} rounds, embedding reference once")
        masks = load_ego_masks(nj, cams) if nj else {}
        ts = time.perf_counter()
        f_real_cam = {c: embed(real_by_cam[c], inception, masks.get(c), args.device)
                      for c in cams}
        f_real_all = torch.cat([f_real_cam[c] for c in cams], dim=0)
        print(f"    reference: {int(f_real_all.shape[0])} frames in {_fmt(time.perf_counter()-ts)}")
        floor = noise_floor(f_real_all, args.subset_size, args.subsets, args.seed)

        for _, rd, _, _, _ in rounds:
            render_dir = os.path.join(rd, "render", "full")
            out_dir = os.path.join(rd, "metrics")
            ts = time.perf_counter()
            try:
                rows = []
                for shift in list_shifts(render_dir):
                    f_fake_cam = {}
                    for c in cams:
                        paths = render_paths(render_dir, shift, c)
                        if paths:
                            f_fake_cam[c] = embed(paths, inception, masks.get(c), args.device)
                    if not f_fake_cam:
                        continue
                    for c, ff in f_fake_cam.items():
                        mean, std, used = kid_from_features(
                            f_real_cam[c], ff, args.subset_size_per_camera,
                            args.subsets, args.seed)
                        rows.append({
                            "shift_name": shift, "shift_m": parse_shift_metres(shift),
                            "camera": c, "kid_mean": mean, "kid_std": std,
                            "fid": fid_from_features(f_real_cam[c], ff),
                            "n_real": int(f_real_cam[c].shape[0]),
                            "n_fake": int(ff.shape[0]), "subset_size_used": used,
                        })
                    f_fake_all = torch.cat(
                        [f_fake_cam[c] for c in cams if c in f_fake_cam], dim=0)
                    mean, std, used = kid_from_features(
                        f_real_all, f_fake_all, args.subset_size, args.subsets, args.seed)
                    rows.append({
                        "shift_name": shift, "shift_m": parse_shift_metres(shift),
                        "camera": "pooled", "kid_mean": mean, "kid_std": std,
                        "fid": fid_from_features(f_real_all, f_fake_all),
                        "n_real": int(f_real_all.shape[0]),
                        "n_fake": int(f_fake_all.shape[0]), "subset_size_used": used,
                    })
                if not rows:
                    raise RuntimeError("no shift level produced any render")

                os.makedirs(out_dir, exist_ok=True)
                result = {
                    "schema_version": 1,
                    "config": {
                        "render_dir": render_dir, "real_bank_dir": bank,
                        "reference_set": "035_real_frames (all frames, train and val)",
                        "cameras": cams, "ego_masked": bool(masks),
                        "subset_size_pooled": args.subset_size,
                        "subset_size_per_camera": args.subset_size_per_camera,
                        "subsets": args.subsets, "seed": args.seed,
                        "feature_extractor": "InceptionV3 pool3 (2048)",
                        "kid_kernel": "poly degree=3 gamma=1/d coef=1",
                        "scored_by": "scripts/experiments/score_stored_rounds.py",
                    },
                    "noise_floor": floor, "rows": rows,
                }
                with open(os.path.join(out_dir, "offpath_metrics.json"), "w") as fp:
                    json.dump(result, fp, indent=2)
                el = time.perf_counter() - ts
                with open(os.path.join(out_dir, ".success"), "w") as fp:
                    levels = len({r['shift_name'] for r in rows})
                    fp.write(f"off-path metrics: {levels} shift levels, {len(cams)} cameras\n"
                             f"duration: {_fmt(el)}\n"
                             f"  ego_masked  {bool(masks)}\n"
                             f"  noise_floor_kid  {floor['kid_mean']:.6f}\n"
                             f"  backfilled  score_stored_rounds.py\n")
                p0 = [r for r in rows if r["camera"] == "pooled"]
                curve = "  ".join(f"{r['shift_name']}={r['kid_mean']:.5f}" for r in p0)
                print(f"    {os.path.relpath(rd)[-58:]}  {_fmt(el)}")
                print(f"      {curve}")
                ok += 1
            except Exception as exc:
                print(f"    FAILED {os.path.relpath(rd)[-58:]}: {type(exc).__name__}: {exc}")
                fail += 1

    print(f"\n===== {ok} scored, {fail} failed, {len(skipped)} skipped, "
          f"{len(nobank)} without a bank, in {_fmt(time.perf_counter()-t0)} =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
