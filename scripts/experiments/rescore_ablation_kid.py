"""Re-score the 4060 anchor-scene ablation with the current off-path metrics stage.

The ablation's off-trajectory KID (docs/06-evaluation/ablation-scene084.md) came from an ad-hoc
script that no longer exists, and the renders it scored were deleted, so those values cannot be
re-derived and must not share an axis with the stage's. The checkpoints survive. This script
renders each one along the recorded trajectory and the lateral shifts the loop uses, scores the
renders with the stage's own functions, and puts frame-bootstrap intervals on every KID and every
3 m penalty. The reference set and ego mask are the ones the stored loop rounds were backfilled
with (score_stored_rounds.find_real_bank / find_ncore_json), and loop dirs given with --loops
(run B, ``07_difix_4dgs_c209be42``) are scored from their stored renders the same way. Every
curve of the comparison then shares one scorer and one set of paired draws, so the capacity, step
and bank effects can be set side by side and differences between them get intervals.

Run on the machine that holds the checkpoints (the RTX 4060 laptop), from the repo root:

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \\
    envs/envs/env_gsplat/bin/python scripts/experiments/rescore_ablation_kid.py \\
        --variants multirun/ablation_scene084 multirun/ablation_scene084_push \\
        --loops results/extended_4dgs/wayve101/scene_084/07_difix_4dgs_c209be42 \\
        --out results/ablation_kid_rescored

then copy ``results/ablation_kid_rescored/`` to the batch machine. Needs ``kid_bootstrap.py``
beside this file. ``--dry-run`` lists the work, ``--limit 1`` scores one variant as a smoke test,
and a variant already scored is skipped on a re-run. Renders are deleted after scoring unless
``--keep-renders`` (about 3.6 GB per variant).

Caveat recorded in the output: render_standalone has no antialiasing option, so a variant trained
with ``antialiased: true`` is rendered with classic rasterisation here, as it was by the ad-hoc
script.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)  # score_stored_rounds globs relative to the repo root
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "post_processing"))

from offpath_metrics import (  # noqa: E402
    embed, fid_from_features, kid_from_features, list_shifts, load_ego_masks, noise_floor,
    parse_shift_metres, real_paths_by_camera, render_paths,
)
from scripts.difix_common import _latest_checkpoint, shift_arg  # noqa: E402
from scripts.experiments.kid_bootstrap import SceneBootstrap, ci  # noqa: E402
from scripts.experiments.score_stored_rounds import find_ncore_json, find_real_bank  # noqa: E402


class _CfgLoader(yaml.SafeLoader):
    """cfg.yml carries python/tuple and python/object tags, which SafeLoader refuses."""


_CfgLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/",
    lambda l, s, n: l.construct_mapping(n, deep=True) if isinstance(n, yaml.MappingNode)
    else l.construct_sequence(n, deep=True) if isinstance(n, yaml.SequenceNode) else l.construct_scalar(n))


def load_yaml(path, loader=_CfgLoader):
    with open(path) as fp:
        return yaml.load(fp, Loader=loader)


def find_variants(paths):
    out = []
    for p in paths:
        if os.path.isdir(os.path.join(p, "ckpts")):
            out.append(p)
        else:
            out += sorted(d for d in glob.glob(os.path.join(p, "*")) if os.path.isdir(os.path.join(d, "ckpts")))
    return out


def variant_info(vdir):
    cfg = load_yaml(os.path.join(vdir, "cfg.yml"))
    hyd = os.path.join(vdir, ".hydra", "config.yaml")
    scene = None
    if os.path.exists(hyd):
        scene = (load_yaml(hyd, yaml.SafeLoader).get("dataset") or {}).get("scene")
    return {"cams": list(cfg.get("ncore_camera_ids") or []), "dynamic": bool(cfg.get("enable_dynamic")),
            "antialiased": bool(cfg.get("antialiased")), "max_steps": cfg.get("max_steps"),
            "cap_max": (cfg.get("strategy") or {}).get("cap_max"), "scene": scene,
            "ckpt": _latest_checkpoint(os.path.join(vdir, "ckpts"))}


def render(vdir, info, shifts, out_dir):
    """The loop's own render call (run_difix_4dgs.run_render), from the checkpoint."""
    cmd = [sys.executable, os.path.join(ROOT, "src", "gsplat_training", "render_standalone.py"),
           f"hydra.run.dir={out_dir}",
           f"++rendering.camera_paths_dir={os.path.join(vdir, 'camera_paths')}",
           f"++rendering.output_dir={out_dir}",
           f"++rendering.cameras_to_render=[{','.join(info['cams'])}]",
           "++rendering.render_modes=[full]",
           f"++rendering.trajectory.shifts={shift_arg([[0.0, 0.0, 0.0]] + shifts)}",
           f"++rendering.checkpoint_path={info['ckpt']}",
           f"++rendering.render_dynamic={'true' if info['dynamic'] else 'false'}"]
    if info["scene"]:
        cmd.append(f"dataset.scene={info['scene']}")
    with open(os.path.join(out_dir + ".log"), "w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    return os.path.join(out_dir, "full")


class Scorer:
    """Real features, ego mask, noise floor and bootstrap draws for one scene, made once."""

    def __init__(self, scene, cams, args):
        from torchmetrics.image.fid import NoTrainInceptionV3
        self.args, self.cams = args, cams
        self.bank = args.real_bank or find_real_bank(scene)
        self.ncore = args.ncore_json or find_ncore_json(scene)
        if not self.bank:
            raise RuntimeError(f"no 035_real_frames bank found for {scene}; pass --real-bank")
        self.inception = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(args.device).eval()
        self.masks = load_ego_masks(self.ncore, cams) if self.ncore else {}
        real = real_paths_by_camera(self.bank)
        self.f_real_cam = {c: embed(real[c], self.inception, self.masks.get(c), args.device) for c in cams}
        self.f_real = torch.cat([self.f_real_cam[c] for c in cams])
        self.floor = noise_floor(self.f_real, args.subset_size, args.subsets, args.seed)
        self.boot = SceneBootstrap(self.f_real, draws=args.draws, seed=args.seed, device=args.device)

    def score(self, render_dir, label, out_dir):
        """Stage-identical rows (per camera and pooled) plus the pooled bootstrap draws."""
        a, rows, draws = self.args, [], {}
        for shift in list_shifts(render_dir):
            f_cam = {}
            for c in self.cams:
                paths = render_paths(render_dir, shift, c)
                if paths:
                    f_cam[c] = embed(paths, self.inception, self.masks.get(c), a.device)
            for c, ff in f_cam.items():
                m, s, u = kid_from_features(self.f_real_cam[c], ff, a.subset_size_per_camera, a.subsets, a.seed)
                rows.append({"shift_name": shift, "shift_m": parse_shift_metres(shift), "camera": c,
                             "kid_mean": m, "kid_std": s, "fid": fid_from_features(self.f_real_cam[c], ff),
                             "n_real": int(self.f_real_cam[c].shape[0]), "n_fake": int(ff.shape[0]),
                             "subset_size_used": u})
            f_all = torch.cat([f_cam[c] for c in self.cams if c in f_cam])
            m, s, u = kid_from_features(self.f_real, f_all, a.subset_size, a.subsets, a.seed)
            rows.append({"shift_name": shift, "shift_m": parse_shift_metres(shift), "camera": "pooled",
                         "kid_mean": m, "kid_std": s, "fid": fid_from_features(self.f_real, f_all),
                         "n_real": int(self.f_real.shape[0]), "n_fake": int(f_all.shape[0]), "subset_size_used": u})
            point, d = self.boot.score(f_all)
            draws[shift] = d
            rows[-1].update(kid_full=point)
        os.makedirs(out_dir, exist_ok=True)
        result = {"schema_version": 1,
                  "config": {"render_dir": render_dir, "real_bank_dir": self.bank,
                             "reference_set": "035_real_frames (all frames, train and val)", "cameras": self.cams,
                             "ego_masked": bool(self.masks), "ncore_json": self.ncore,
                             "subset_size_pooled": a.subset_size, "subset_size_per_camera": a.subset_size_per_camera,
                             "subsets": a.subsets, "seed": a.seed, "feature_extractor": "InceptionV3 pool3 (2048)",
                             "kid_kernel": "poly degree=3 gamma=1/d coef=1", "bootstrap_draws": a.draws,
                             "scored_by": "scripts/experiments/rescore_ablation_kid.py", "label": label},
                  "noise_floor": self.floor, "rows": rows}
        with open(os.path.join(out_dir, "offpath_metrics.json"), "w") as fp:
            json.dump(result, fp, indent=2)
        np.savez(os.path.join(out_dir, "bootstrap_draws.npz"), **draws)
        return result, draws


def load_scored(out_dir):
    p, q = os.path.join(out_dir, "offpath_metrics.json"), os.path.join(out_dir, "bootstrap_draws.npz")
    if os.path.exists(p) and os.path.exists(q):
        return json.load(open(p)), dict(np.load(q))
    return None


def far_shift(names):
    """The largest-magnitude shift (the curves here are one-sided)."""
    shifted = [n for n in names if n != "original"]
    return max(shifted, key=lambda n: float(np.linalg.norm(parse_shift_metres(n)))) if shifted else None


def summarise(sets, out):
    """curves: one row per scored set; comparisons: paired changes against each set's reference."""
    curves = []
    for s in sets:
        pooled = OrderedDict((r["shift_name"], r) for r in s["result"]["rows"] if r["camera"] == "pooled")
        far = far_shift(list(pooled))
        d0, dF = s["draws"]["original"], s["draws"][far]
        pen = (dF - d0) * 1e3
        lo, hi = ci(pen)
        row = OrderedDict(label=s["label"], group=s["group"], reference=s["reference"], max_steps=s.get("max_steps"),
                          cap_max=s.get("cap_max"), antialiased=s.get("antialiased"))
        for n, r in pooled.items():
            row[f"kid_{n}"] = r["kid_mean"] * 1e3
            row[f"std_{n}"] = r["kid_std"] * 1e3
        row.update(far_shift=far, penalty=(pooled[far]["kid_full"] - pooled["original"]["kid_full"]) * 1e3,
                   penalty_ci_lo=lo, penalty_ci_hi=hi,
                   ratio=pooled[far]["kid_mean"] / pooled["original"]["kid_mean"])
        curves.append(row)
    comps = []
    by_label = {s["label"]: s for s in sets}
    for s in sets:
        ref = by_label.get(s["reference"])
        if not ref or ref is s:
            continue
        far = far_shift(list(s["draws"]))
        if far not in ref["draws"]:
            continue
        pen_s = s["draws"][far] - s["draws"]["original"]
        pen_r = ref["draws"][far] - ref["draws"]["original"]
        rel = 100.0 * (pen_s / pen_r - 1.0)
        rel0 = 100.0 * (s["draws"]["original"] / ref["draws"]["original"] - 1.0)
        pts = {x["label"]: x for x in curves}
        comps.append(OrderedDict(
            label=s["label"], reference=ref["label"], penalty_ref=pts[ref["label"]]["penalty"],
            penalty=pts[s["label"]]["penalty"],
            penalty_change_pct=100.0 * (pts[s["label"]]["penalty"] / pts[ref["label"]]["penalty"] - 1.0),
            penalty_change_ci_lo=ci(rel)[0], penalty_change_ci_hi=ci(rel)[1],
            kid0_change_pct=100.0 * (pts[s["label"]]["kid_original"] / pts[ref["label"]]["kid_original"] - 1.0),
            kid0_change_ci_lo=ci(rel0)[0], kid0_change_ci_hi=ci(rel0)[1]))
    for name, rows in (("curves", curves), ("comparisons", comps)):
        if not rows:
            continue
        cols = list(OrderedDict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(out, f"{name}.csv"), "w") as fp:
            fp.write(",".join(cols) + "\n")
            for r in rows:
                fp.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")
        with open(os.path.join(out, f"{name}.md"), "w") as fp:
            fp.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
            for r in rows:
                fp.write("| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else ("" if v is None else str(v))
                                           for v in (r.get(c) for c in cols)) + " |\n")
    return curves, comps


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variants", nargs="*", default=[], help="sweep dirs or variant dirs with ckpts/")
    ap.add_argument("--loops", nargs="*", default=[], help="07_difix_4dgs_* dirs scored from stored renders")
    ap.add_argument("--reference", default="abl_baseline", help="variant the others are compared with")
    ap.add_argument("--shifts", nargs="*", type=float, default=[-1.0, -2.0, -3.0], help="lateral (x) shifts, m")
    ap.add_argument("--out", default="results/ablation_kid_rescored")
    ap.add_argument("--real-bank", default=None)
    ap.add_argument("--ncore-json", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--subset-size", type=int, default=500)
    ap.add_argument("--subset-size-per-camera", type=int, default=100)
    ap.add_argument("--subsets", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep-renders", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    variants = find_variants(a.variants)[: a.limit] if a.limit else find_variants(a.variants)
    infos = {v: variant_info(v) for v in variants}
    loops = []
    for L in a.loops:
        st = json.load(open(os.path.join(L, "loop_state.json")))
        for r in st["rounds"]:
            rd = os.path.join(L, f"round_{r['round']:03d}")
            rdir = os.path.join(rd, "render", "full")
            loops.append({"loop": L, "round": r["round"], "render": rdir, "max_steps": r["max_steps"],
                          "bank": len(r.get("bank", [])), "cams": st.get("cameras"),
                          "has_renders": os.path.isdir(os.path.join(rdir, "frames"))})
    scenes = {i["scene"] for i in infos.values()} | ({"scene_084"} if loops else set())
    scene = sorted(s for s in scenes if s)[0] if scenes else "scene_084"
    cams = next((i["cams"] for i in infos.values() if i["cams"]), None) or (loops[0]["cams"] if loops else None)
    print(f"scene {scene}, cameras {cams}")
    print(f"real bank: {a.real_bank or find_real_bank(scene)}   ncore json (ego mask): {a.ncore_json or find_ncore_json(scene)}")
    for v, i in infos.items():
        flag = "  [antialiased: rendered classic]" if i["antialiased"] else ""
        print(f"  variant {os.path.basename(v):22s} ckpt={os.path.basename(i['ckpt']) or 'NONE'} "
              f"steps={i['max_steps']} cap={i['cap_max']} dynamic={i['dynamic']}{flag}")
    for l in loops:
        print(f"  loop {os.path.basename(l['loop'])} round {l['round']}: steps={l['max_steps']} bank={l['bank']} "
              f"renders={'yes' if l['has_renders'] else 'MISSING'}")
    if a.dry_run:
        return 0

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    scorer = Scorer(scene, cams, a)
    sets, failed = [], []
    for v, i in infos.items():
        name = os.path.basename(v)
        sd = os.path.join(a.out, "sets", name)
        cached = load_scored(sd)
        try:
            if cached:
                res, draws = cached
                print(f"  {name}: cached")
            else:
                if not i["ckpt"]:
                    raise RuntimeError("no checkpoint")
                ts = time.time()
                rdir = os.path.join(a.out, "renders", name)
                os.makedirs(os.path.dirname(rdir), exist_ok=True)
                rfull = render(v, i, [[s, 0.0, 0.0] for s in a.shifts], rdir)
                res, draws = scorer.score(rfull, name, sd)
                if not a.keep_renders:
                    shutil.rmtree(rdir, ignore_errors=True)
                print(f"  {name}: scored in {time.time() - ts:.0f} s")
            sets.append({"label": name, "group": "ablation", "reference": a.reference, "result": res, "draws": draws,
                         "max_steps": i["max_steps"], "cap_max": i["cap_max"], "antialiased": i["antialiased"]})
        except Exception as e:  # noqa: BLE001 -- one variant's failure must not stop the others
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"  {name}: FAILED {type(e).__name__}: {e}")
    for l in loops:
        name = f"{os.path.basename(l['loop'])}_r{l['round']}"
        sd = os.path.join(a.out, "sets", name)
        cached = load_scored(sd)
        if not cached and not l["has_renders"]:
            failed.append((name, "renders missing; re-render the round to score it"))
            continue
        res, draws = cached or scorer.score(l["render"], name, sd)
        sets.append({"label": name, "group": "loop", "max_steps": l["max_steps"],
                     "reference": f"{os.path.basename(l['loop'])}_r0", "result": res, "draws": draws})
        print(f"  {name}: {'cached' if cached else 'scored'}")

    curves, comps = summarise(sets, a.out)
    git = lambda *c: subprocess.run(["git", *c], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    with open(os.path.join(a.out, "provenance.json"), "w") as fp:
        json.dump({"command": " ".join(sys.argv), "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "seconds": round(time.time() - t0), "git": git("log", "-1", "--oneline"),
                   "dirty": bool(git("status", "--short")), "scene": scene, "cameras": cams,
                   "real_bank": scorer.bank, "ncore_json": scorer.ncore, "ego_masked": bool(scorer.masks),
                   "noise_floor": scorer.floor, "shifts_m": a.shifts, "draws": a.draws, "seed": a.seed,
                   "failed": failed,
                   "notes": ["Variants with antialiased=true were rendered with classic rasterisation "
                             "(render_standalone has no antialiasing option).",
                             "penalty = KID(far shift) - KID(0 m), x1e3, from the full-sample estimate; "
                             "intervals are 95% percentile intervals over paired frame-bootstrap draws."]},
                  fp, indent=2)
    print(f"\n{len(sets)} sets scored, {len(failed)} failed, in {time.time() - t0:.0f} s -> {a.out}")
    for n, e in failed:
        print(f"  failed {n}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
