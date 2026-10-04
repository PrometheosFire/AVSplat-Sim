"""Collect every Chapter 6 table from the stored runs.

The thesis may only quote a number that was read out of an artifact in this
repository (thesis CLAUDE.md, "Fact sourcing"). This script produces those
artifacts. It reads only what the pipeline already wrote -- loop dirs, refine
dirs, real-frame banks, the raw COLMAP poses and the dataset metadata -- and
writes one CSV and one markdown table per output under ``results/chapter6/``,
plus a README naming the git commit, the command line and the sources of every
table. Nothing here trains or renders. The GPU scores the saved validation
canvases (SSIM, LPIPS) and, with ``--bootstrap``, re-embeds renders to put
confidence intervals on KID.

The plan this implements is ``docs/06-evaluation/chapter6-data-plan.md``
(section 5.1, outputs C1-C11); each table names the plan items it serves.

Usage:
    PYTHONPATH=. envs/envs/env_gsplat/bin/python scripts/experiments/collect_chapter6.py \\
        [--out results/chapter6] [--only scenes,audit,onpath,percamera,kid,loop,trajectories,curation,cost] \\
        [--scenes scene_084,scene_099] [--bootstrap scene_084] [--device cuda]
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, OrderedDict, defaultdict

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "post_processing"))

from scripts.experiments.export_results import newest_loop  # noqa: E402
from scripts.experiments.collect_ablation import VIABLE_MIN_GAUSSIANS, rigid_detail  # noqa: E402
from src.gsplat_training.dynamic.rigid_tracks import DEFAULT_RIGID_CLASSES  # noqa: E402

W = os.path.join(ROOT, "results", "extended_4dgs", "wayve101")
METADATA = os.path.join(ROOT, "data", "wayve101", "dataset_info", "scene_metadata.csv")

# Author decisions of 2026-10-03 (chapter6-data-plan.md, section 6). 071 is skipped;
# 048 and 018 collapsed in their 30k round and keep rounds 0-2 only, which the
# status column records without a hard-coded list.
SKIPPED = {"scene_071": "skipped by the author (collapsed reconstruction)"}

TRAFFIC_COLS = ("Same direction Vehicle Traffic", "Oncoming vehicle traffic", "Cross vehicle traffic")
REGISTRY = []  # (name, title, plan items, sources, rows) for the README


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #

class _CfgLoader(yaml.SafeLoader):
    """cfg.yml is a dump of the training Config: it carries python/tuple and
    python/object tags (the MCMC strategy), which SafeLoader refuses."""


def _construct_python(loader, suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_CfgLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_python)


def load_cfg(path):
    with open(path) as fp:
        return yaml.load(fp, Loader=_CfgLoader)


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def hms(s):
    """'HH:MM:SS' -> seconds. 'cached' (a resumed step) and missing -> None."""
    if not s or not re.fullmatch(r"\d+:\d\d:\d\d", str(s)):
        return None
    h, m, sec = (int(x) for x in s.split(":"))
    return h * 3600 + m * 60 + sec


def rel(p):
    return os.path.relpath(p, ROOT) if p else ""


def jload(path):
    with open(path) as fp:
        return json.load(fp)


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.4g}"
    return str(v)


def median(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def write(out_dir, name, rows, title, items, sources, md_cols=None, notes=()):
    """One CSV with every column, one markdown table with the readable subset."""
    os.makedirs(out_dir, exist_ok=True)
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(os.path.join(out_dir, f"{name}.csv"), "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    md_cols = md_cols or cols
    lines = [f"# {title}", "", f"Plan items: {items}  ", f"Source: {sources}", ""]
    lines += [n for n in notes] + ([""] if notes else [])
    if rows:
        lines.append("| " + " | ".join(md_cols) + " |")
        lines.append("|" + "---|" * len(md_cols))
        for r in rows:
            lines.append("| " + " | ".join(fmt(r.get(c)) for c in md_cols) + " |")
    else:
        lines.append("No rows.")
    with open(os.path.join(out_dir, f"{name}.md"), "w") as fp:
        fp.write("\n".join(lines) + "\n")
    REGISTRY.append((name, title, items, sources, len(rows), time.strftime("%Y-%m-%d %H:%M")))
    print(f"  wrote {name}: {len(rows)} rows")


# --------------------------------------------------------------------------- #
# Scene discovery                                                             #
# --------------------------------------------------------------------------- #

def discover(only_scenes=None):
    scenes = []
    for sd in sorted(glob.glob(os.path.join(W, "scene_*"))):
        s = os.path.basename(sd)
        if only_scenes and s not in only_scenes:
            continue
        loop = newest_loop(s)
        st = jload(os.path.join(loop, "loop_state.json")) if loop else None
        rounds = []
        if st:
            for r in st["rounds"]:
                rd = os.path.join(loop, f"round_{r['round']:03d}")
                rounds.append({
                    "round": r["round"], "max_steps": r["max_steps"], "dynamic": r.get("dynamic"),
                    "bank": len(r.get("bank", [])), "durations": r.get("durations", {}),
                    "dir": rd, "train": os.path.join(rd, "train"),
                    "difixed_shifts": r.get("difixed_shifts"),
                })
        refine = None
        if st and st.get("tracks_json"):
            refine = os.path.dirname(st["tracks_json"])
        else:
            cands = [d for d in glob.glob(os.path.join(sd, "15_refine_*"))
                     if os.path.exists(os.path.join(d, "refine_report.json"))]
            refine = max(cands, key=os.path.getmtime) if cands else None
        bank = None
        for r in rounds:
            m = os.path.join(r["dir"], "metrics", "offpath_metrics.json")
            if os.path.exists(m):
                bank = jload(m)["config"]["real_bank_dir"]
                break
        if s in SKIPPED:
            status = "skipped"
        elif not rounds:
            status = "not run"
        elif len(rounds) < 4:
            status = "no final round"
        else:
            status = "chapter"
        scenes.append({"scene": s, "dir": sd, "loop": loop, "state": st, "rounds": rounds,
                       "refine": refine, "bank": bank, "status": status})
    return scenes


def schedule(rounds):
    steps = [r["max_steps"] for r in rounds]
    if len(steps) == 4 and set(steps) == {30000}:
        return "30k x4"
    if steps[:3] == [7000] * 3:
        return "7k/7k/7k/30k"
    return "/".join(f"{s // 1000}k" for s in steps)


def fixed_step_group(rounds):
    """Rounds trained for as many steps as round 0: only the bank differs."""
    if not rounds:
        return []
    s0 = rounds[0]["max_steps"]
    out = []
    for r in rounds:
        if r["max_steps"] != s0:
            break
        out.append(r)
    return out


def val_stats(train_dir):
    v = sorted(glob.glob(os.path.join(train_dir, "stats", "val_step*.json")))
    return jload(v[-1]) if v else {}


def train_stats(train_dir):
    v = sorted(glob.glob(os.path.join(train_dir, "stats", "train_step*_rank0.json")))
    return jload(v[-1]) if v else {}


def round_cfg(train_dir):
    p = os.path.join(train_dir, "cfg.yml")
    return flatten(load_cfg(p)) if os.path.exists(p) else {}


# --------------------------------------------------------------------------- #
# C2  Scene table: conditions, ego motion, held-out split                     #
# --------------------------------------------------------------------------- #

def ego_motion(scene):
    """Camera centres from the raw COLMAP reconstruction, whose poses are metric
    (docs/01-overview/data-wayvescenes101-ncore.md, section 3), per camera folder,
    in timestamp order."""
    import contextlib
    import io
    import pycolmap
    path = os.path.join(ROOT, "data", "wayve101", scene, "colmap_sparse", "rig") + "/"
    with contextlib.redirect_stdout(io.StringIO()):
        m = pycolmap.SceneManager(path)
        m.load_cameras()
        m.load_images()
    cams = defaultdict(list)
    for im in m.images.values():
        folder, fname = im.name.split("/", 1)
        ts = int(os.path.splitext(os.path.basename(fname))[0])
        c = im.C() if callable(im.C) else im.C
        cams[folder].append((ts, np.asarray(c, dtype=np.float64), im.camera_id))
    out = {}
    for folder, lst in cams.items():
        lst.sort(key=lambda t: t[0])
        out[folder] = {"ts": np.array([t[0] for t in lst]), "C": np.stack([t[1] for t in lst]),
                       "ncore": f"camera{lst[0][2]}"}
    return out


def bank_frames(bank):
    idx = jload(os.path.join(bank, "index.json"))
    cams = sorted(idx["cameras"].items(), key=lambda kv: kv[1]["camera_index"])
    return idx, cams


def c2_scenes(scenes, meta, out):
    rows = []
    for sc in scenes:
        s = sc["scene"]
        md = meta.get(s, {})
        row = OrderedDict(scene=s, status=sc["status"], road=md.get("Road Type"), weather=md.get("Weather"),
                          time=md.get("Time of Day"),
                          traffic_k=sum(md.get(c) == "Yes" for c in TRAFFIC_COLS) if md else None)
        for key, col in (("same_dir", TRAFFIC_COLS[0]), ("oncoming", TRAFFIC_COLS[1]), ("cross", TRAFFIC_COLS[2]),
                         ("ped_crossing", "Pedestrian crossing road"), ("ped_sidewalk", "Pedestrian on sidewalk"),
                         ("cyclists", "Cyclists / motorbike present"), ("exposure_change", "Large Exposure Change")):
            row[key] = md.get(col)
        row["schedule"] = schedule(sc["rounds"]) if sc["rounds"] else None
        row["rounds_done"] = len(sc["rounds"])
        if sc["rounds"]:
            cfg = round_cfg(sc["rounds"][-1]["train"])
            row["cap_max"] = cfg.get("strategy.cap_max")
            row["dynamic_final"] = sc["rounds"][-1]["dynamic"]
            om = os.path.join(sc["rounds"][-1]["dir"], "metrics", "offpath_metrics.json")
            if os.path.exists(om):
                names = sorted({r["shift_name"] for r in jload(om)["rows"]} - {"original"})
                row["shift_levels"] = " ".join(names)
        try:
            em = ego_motion(s)
            f = em["front-forward"]
            d = np.linalg.norm(np.diff(f["C"], axis=0), axis=1)
            dur = (f["ts"][-1] - f["ts"][0]) / 1e6
            row.update(duration_s=round(float(dur), 2), path_m=float(d.sum()), mean_speed_mps=float(d.sum() / dur),
                       median_step_m=float(np.median(d)), p90_step_m=float(np.percentile(d, 90)),
                       max_step_m=float(d.max()))
        except Exception as e:  # noqa: BLE001
            em = None
            row["ego_error"] = repr(e)[:80]
        if sc["bank"] and em:
            idx, cams = bank_frames(sc["bank"])
            by_ncore = {v["ncore"]: v for v in em.values()}
            dists, nval, first_val = [], 0, []
            for cam, rec in cams:
                fr = sorted(rec["frames"], key=lambda f: f["frame_idx"])
                C = by_ncore[cam]["C"]
                if len(C) != len(fr):
                    raise RuntimeError(f"{s} {cam}: {len(C)} COLMAP images vs {len(fr)} bank frames")
                vals = [f["frame_idx"] for f in fr if f["is_val"]]
                first_val.append(f"{cam}:{vals[0] if vals else '-'}")
                nval += len(vals)
                for k in vals:
                    nb = [j for j in (k - 1, k + 1) if 0 <= j < len(C)]
                    dists.append(min(float(np.linalg.norm(C[k] - C[j])) for j in nb))
            row.update(n_val=nval, val_nearest_train_median_m=float(np.median(dists)),
                       val_nearest_train_max_m=float(np.max(dists)), first_val_frame=" ".join(first_val),
                       test_every=idx.get("test_every"), frames_per_camera=len(cams[0][1]["frames"]))
        row.update(loop=rel(sc["loop"]), refine=rel(sc["refine"]), bank=rel(sc["bank"]))
        rows.append(row)
    write(out, "scenes", rows, "Scene set: conditions, ego motion and held-out split",
          "6.1-a to 6.1-f, 6.6-g", "scene_metadata.csv; data/wayve101/<scene>/colmap_sparse/rig (metric poses, "
          "front-forward camera); 035_real_frames_*/index.json; round cfg.yml; loop_state.json",
          md_cols=["scene", "status", "road", "weather", "time", "traffic_k", "schedule", "cap_max", "shift_levels",
                   "mean_speed_mps", "median_step_m", "path_m", "n_val", "val_nearest_train_median_m",
                   "val_nearest_train_max_m"],
          notes=["traffic_k counts Yes among same-direction, oncoming and cross vehicle traffic. Speeds and steps "
                 "come from the front-forward camera centres; a held-out frame's nearest training view is the "
                 "same camera's adjacent frame."])

    # Dataset-wide availability (6.1-b).
    allrows = list(meta.values())
    grid = Counter((r["Road Type"], sum(r[c] == "Yes" for c in TRAFFIC_COLS)) for r in allrows)
    av = []
    for road in sorted({r["Road Type"] for r in allrows}):
        av.append(OrderedDict(road=road, **{f"traffic_{k}": grid.get((road, k), 0) for k in range(4)},
                              total=sum(grid.get((road, k), 0) for k in range(4))))
    notes = [f"{len(allrows)} scenes. " + "; ".join(
        f"{col}: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r[col] for r in allrows).items()))
        for col in ("Weather", "Time of Day", "Large Exposure Change"))]
    write(out, "dataset_availability", av, "WayveScenes101: road type x traffic k/3", "6.1-b",
          "data/wayve101/dataset_info/scene_metadata.csv", notes=notes)


# --------------------------------------------------------------------------- #
# C3  Config audit: every round's effective config against its peers          #
# --------------------------------------------------------------------------- #

AUDIT_SKIP = {"pseudo_manifests", "eval_steps", "save_steps", "ply_steps", "ckpt", "max_steps"}


def c3_audit(scenes, out):
    per_round = []
    for sc in scenes:
        for r in sc["rounds"]:
            cfg = round_cfg(r["train"])
            if cfg:
                per_round.append((sc, r, cfg))
    rows = []
    for steps in sorted({r["max_steps"] for _, r, _ in per_round}):
        group = [(sc, r, c) for sc, r, c in per_round if r["max_steps"] == steps]
        keys = sorted({k for _, _, c in group for k in c})
        for k in keys:
            if k in AUDIT_SKIP:
                continue
            vals = [json.dumps(c.get(k), default=str) for _, _, c in group]
            if any("/home/" in v for v in vals):
                continue
            mode, n_mode = Counter(vals).most_common(1)[0]
            for (sc, r, c), v in zip(group, vals):
                if v != mode:
                    rows.append(OrderedDict(scene=sc["scene"], status=sc["status"], round=r["round"],
                                            max_steps=steps, key=k, value=v, mode=mode,
                                            rounds_at_mode=f"{n_mode}/{len(group)}"))
    write(out, "config_audit", rows, "Training-config deviations from the batch mode (per step budget)",
          "6.1-k", "round_*/train/cfg.yml (effective config, post adjust_steps)",
          notes=["Each round is compared with every round of the same step budget. Paths, step lists and the "
                 "pseudo-view manifests are excluded."])


# --------------------------------------------------------------------------- #
# C4  On-path metrics per round                                               #
# --------------------------------------------------------------------------- #

def c4_onpath(scenes, out):
    rows = []
    for sc in scenes:
        for r in sc["rounds"]:
            v = val_stats(r["train"])
            cap = round_cfg(r["train"]).get("strategy.cap_max")
            rows.append(OrderedDict(
                scene=sc["scene"], status=sc["status"], round=r["round"], max_steps=r["max_steps"],
                bank=r["bank"], dynamic=r["dynamic"], psnr=v.get("psnr"), ssim=v.get("ssim"), lpips=v.get("lpips"),
                num_GS=v.get("num_GS"), cap_max=cap, at_cap=(v.get("num_GS") == cap) if cap else None,
                num_rigid_GS=v.get("num_rigid_GS"), rigid_instances_alive=v.get("rigid_instances_alive"),
                rigid_capacity_effective=v.get("rigid_capacity_effective"), dynamic_psnr=v.get("dynamic_psnr"),
                dynamic_frames=v.get("dynamic_frames"), dynamic_pixel_frac=v.get("dynamic_pixel_frac"),
                val_s_per_image=v.get("ellipse_time"), source=rel(r["train"])))
    write(out, "onpath_by_round", rows, "Held-out (on-path) metrics per round", "6.2-a to 6.2-d, 6.3-b",
          "round_*/train/stats/val_step*.json; cfg.yml (cap_max)",
          md_cols=["scene", "status", "round", "max_steps", "bank", "psnr", "ssim", "lpips", "num_GS", "at_cap",
                   "rigid_instances_alive", "dynamic_psnr"])

    # Summary over the chapter set: final round, and the bank's on-path cost at fixed steps.
    summ = []
    for sc in scenes:
        if not sc["rounds"]:
            continue
        g = fixed_step_group(sc["rounds"])
        v0, vg, vf = val_stats(g[0]["train"]), val_stats(g[-1]["train"]), val_stats(sc["rounds"][-1]["train"])
        summ.append(OrderedDict(
            scene=sc["scene"], status=sc["status"], schedule=schedule(sc["rounds"]),
            final_round=sc["rounds"][-1]["round"], final_psnr=vf.get("psnr"), final_ssim=vf.get("ssim"),
            final_lpips=vf.get("lpips"), fixed_steps=g[0]["max_steps"], fixed_rounds=f"{g[0]['round']}-{g[-1]['round']}",
            psnr_no_bank=v0.get("psnr"), psnr_last_fixed=vg.get("psnr"),
            d_psnr_bank=(vg["psnr"] - v0["psnr"]) if vg and v0 else None,
            d_lpips_bank=(vg["lpips"] - v0["lpips"]) if vg and v0 else None))
    ch = [r for r in summ if r["status"] == "chapter"]
    notes = []
    for k in ("final_psnr", "final_ssim", "final_lpips"):
        xs = [r[k] for r in ch if r[k] is not None]
        if xs:
            notes.append(f"{k} over the {len(xs)} chapter scenes: median {np.median(xs):.4g}, "
                         f"range {min(xs):.4g} to {max(xs):.4g}.")
    write(out, "onpath_summary", summ, "Final-round quality and the bank's on-path cost at fixed steps",
          "6.2-a, 6.2-b, 6.2-c, 6.4.2-f", "round_*/train/stats/val_step*.json", notes=notes)


# --------------------------------------------------------------------------- #
# C5  Per-camera on-path metrics and colour-corrected PSNR, from the canvases #
# --------------------------------------------------------------------------- #

def _cc_ls_affine(x, y, valid):
    """Per-channel least squares of the GT y on the render x over valid pixels.
    The identity is one candidate, so this cannot lower PSNR on those pixels."""
    import torch
    out = x.clone()
    for c in range(3):
        xs, ys = x[..., c][valid], y[..., c][valid]
        A = torch.stack([xs, torch.ones_like(xs)], 1)
        w = torch.linalg.lstsq(A, ys[:, None]).solution[:, 0]
        out[..., c] = w[0] * x[..., c] + w[1]
    return out.clamp(0, 1)


def c5_percamera(scenes, out, device):
    import importlib.util
    import torch
    from PIL import Image
    from torchmetrics.functional.image import peak_signal_noise_ratio as tm_psnr
    from torchmetrics.functional.image import structural_similarity_index_measure as tm_ssim
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    spec = importlib.util.spec_from_file_location(
        "color_correct", os.path.join(ROOT, "external", "gsplat", "gsplat", "color_correct.py"))
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)  # as runner.py

    def load(p):
        return torch.from_numpy(np.array(Image.open(p).convert("RGB"))).float().div(255).to(device)

    def psnr(a, b):
        return float(tm_psnr(a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None], data_range=1.0))

    cam_rows, check_rows, cc_rows = [], [], []
    for sc in scenes:
        if not sc["bank"]:
            continue
        idx, cams = bank_frames(sc["bank"])
        val = sorted(((cam, f) for cam, rec in cams for f in rec["frames"] if f["is_val"]),
                     key=lambda cf: cf[1]["global_index"])
        for r in sc["rounds"]:
            canv = sorted(p for p in glob.glob(os.path.join(r["train"], "renders", "val_step*.png"))
                          if "_boxes" not in p)
            if len(canv) != len(val):
                print(f"  ! {sc['scene']} r{r['round']}: {len(canv)} canvases vs {len(val)} val frames, skipped")
                continue
            t0 = time.time()
            per = defaultdict(lambda: defaultdict(list))
            for (cam, f), cp in zip(val, canv):
                im = load(cp)
                w = im.shape[1] // 2
                gt_m, pr = im[:, :w], im[:, w:]
                gt = load(os.path.join(sc["bank"], f["image"]))
                masked = (gt.sum(-1) > 0) & (gt_m.sum(-1) == 0)
                valid = ~masked
                a, b = pr.permute(2, 0, 1)[None], gt_m.permute(2, 0, 1)[None]
                vals = {
                    "psnr": psnr(pr, gt_m),
                    "ssim": float(tm_ssim(a, b, data_range=1.0)),
                    "lpips": float(lpips_fn(a.clamp(0, 1), b.clamp(0, 1))),
                    "masked_pct": 100.0 * float(masked.float().mean()),
                    "cc_affine_as_eval": psnr(cc.color_correct_affine(pr, gt), gt_m),
                }
                # A near-black frame (night) leaves too few unclipped pixels for a
                # well-posed fit; such frames are counted as failed, not averaged in.
                for key, fn in (("cc_ls_affine", lambda: _cc_ls_affine(pr, gt_m, valid)),
                                ("cc_quadratic", lambda: cc.color_correct_quadratic(pr, gt_m))):
                    try:
                        corr = fn()
                        corr[masked] = 0
                        vals[key] = psnr(corr, gt_m) if bool(torch.isfinite(corr).all()) else float("nan")
                    except (AssertionError, RuntimeError):
                        vals[key] = float("nan")
                vals["cc_failed"] = float(any(math.isnan(vals[k]) for k in ("cc_ls_affine", "cc_quadratic")))
                for k, v in vals.items():
                    per[cam][k].append(v)
                    per["pooled"][k].append(v)
            v = val_stats(r["train"])
            for cam in [c for c, _ in cams] + ["pooled"]:
                m = {k: float(np.nansum(x)) if k == "cc_failed" else float(np.nanmean(x))
                     for k, x in per[cam].items()}
                base = OrderedDict(scene=sc["scene"], status=sc["status"], round=r["round"],
                                   max_steps=r["max_steps"], camera=cam, n=len(per[cam]["psnr"]))
                cam_rows.append(OrderedDict(base, psnr=m["psnr"], ssim=m["ssim"], lpips=m["lpips"],
                                            masked_pct=m["masked_pct"]))
                cc_rows.append(OrderedDict(base, psnr=m["psnr"], cc_ls_affine=m["cc_ls_affine"],
                                           cc_quadratic=m["cc_quadratic"], cc_affine_as_eval=m["cc_affine_as_eval"],
                                           cc_failed_frames=int(m["cc_failed"]),
                                           gain_ls_affine=m["cc_ls_affine"] - m["psnr"],
                                           gain_quadratic=m["cc_quadratic"] - m["psnr"],
                                           gain_affine_as_eval=m["cc_affine_as_eval"] - m["psnr"]))
            pm = {k: float(np.mean(x)) for k, x in per["pooled"].items()}
            check_rows.append(OrderedDict(scene=sc["scene"], round=r["round"], canvas_psnr=pm["psnr"],
                                          logged_psnr=v.get("psnr"), canvas_ssim=pm["ssim"], logged_ssim=v.get("ssim"),
                                          canvas_lpips=pm["lpips"], logged_lpips=v.get("lpips"),
                                          d_psnr=(pm["psnr"] - v["psnr"]) if v else None))
            print(f"    {sc['scene']} r{r['round']}: {len(canv)} canvases {time.time() - t0:.0f}s")
    src = "round_*/train/renders/val_step*.png (GT | render canvases), 035_real_frames_*/ (unmasked GT, is_val order)"
    write(out, "percamera_onpath", cam_rows, "Held-out metrics per camera, recomputed from the validation canvases",
          "6.2.2-b, 6.2.2-e", src,
          notes=["Canvases are 8-bit, so values differ slightly from the logged float metrics; "
                 "canvas_check.csv gives the size of that difference per round. masked_pct is the share of pixels "
                 "the per-frame dataset mask (ego bodywork and privacy redactions) removes."])
    write(out, "canvas_check", check_rows, "Canvas recomputation against the logged metrics", "6.2.2-b", src)
    write(out, "color_correction", cc_rows, "Colour-corrected PSNR: correct least squares vs the eval() affine",
          "6.1-i, plan section 2.10", src,
          md_cols=["scene", "status", "round", "camera", "psnr", "gain_ls_affine", "gain_quadratic",
                   "gain_affine_as_eval"],
          notes=["cc_ls_affine regresses the GT on the render per channel over valid pixels; cc_quadratic is "
                 "gsplat's quadratic (multiNeRF) fitted against the masked GT; both re-zero masked pixels. "
                 "cc_affine_as_eval reproduces runner.eval(): gsplat's affine fitted against the UNMASKED GT and "
                 "scored against the masked GT. Means over frames, as eval() averages per-image PSNR."])


# --------------------------------------------------------------------------- #
# C6  KID against offset                                                      #
# --------------------------------------------------------------------------- #

SIDE = {("X", -1): "left", ("X", 1): "right", ("Y", -1): "up", ("Y", 1): "down"}


def offpath_rows(round_dir):
    p = os.path.join(round_dir, "metrics", "offpath_metrics.json")
    return jload(p) if os.path.exists(p) else None


def curves(om, camera="pooled"):
    """{side: [(offset_m, kid, std, fid), ...]} from one offpath_metrics.json, x10^3."""
    pts = {}
    zero = None
    for row in om["rows"]:
        if row["camera"] != camera:
            continue
        val = (row["kid_mean"] * 1e3, row["kid_std"] * 1e3, row["fid"])
        if row["shift_name"] == "original":
            zero = val
            continue
        ax, v = row["shift_name"].split("_", 1)
        v = float(v)
        side = SIDE[(ax, int(math.copysign(1, v)))]
        pts.setdefault(side, []).append((abs(v),) + val)
    return {side: [(0.0,) + zero] + sorted(p) for side, p in pts.items()} if zero else {}


def ls_slope(c):
    x = np.array([p[0] for p in c])
    y = np.array([p[1] for p in c])
    return float(np.polyfit(x, y, 1)[0])


def curve_summary(c):
    k0, s0 = c[0][1], c[0][2]
    kN = c[-1][1]
    first = None
    for x, k, s, _ in c[1:]:
        if k - k0 > 2 * math.hypot(s, s0):
            first = x
            break
    adj = ["yes" if (c[i + 1][1] - c[i][1]) > 2 * math.hypot(c[i + 1][2], c[i][2]) else "no"
           for i in range(len(c) - 1)]
    return OrderedDict(kid_0m=k0, kid_far=kN, far_m=c[-1][0], penalty=kN - k0, ratio=kN / k0 if k0 else None,
                       slope_ls=ls_slope(c), slope_endpoint=(kN - k0) / c[-1][0],
                       first_resolvable_m=first, adjacent_steps_resolvable=" ".join(adj))


def c6_kid(scenes, out):
    long_rows, summ = [], []
    for sc in scenes:
        for r in sc["rounds"]:
            om = offpath_rows(r["dir"])
            if not om:
                continue
            nf = om["noise_floor"]["kid_mean"] * 1e3
            for row in om["rows"]:
                long_rows.append(OrderedDict(
                    scene=sc["scene"], status=sc["status"], round=r["round"], max_steps=r["max_steps"],
                    bank=r["bank"], shift=row["shift_name"], camera=row["camera"], kid_x1e3=row["kid_mean"] * 1e3,
                    kid_std_x1e3=row["kid_std"] * 1e3, fid=row["fid"], n_real=row["n_real"], n_fake=row["n_fake"],
                    subset_size=row["subset_size_used"], noise_floor_x1e3=nf))
            for side, c in curves(om).items():
                summ.append(OrderedDict(scene=sc["scene"], status=sc["status"], round=r["round"],
                                        max_steps=r["max_steps"], bank=r["bank"], side=side,
                                        **curve_summary(c), noise_floor=nf,
                                        source=rel(os.path.join(r["dir"], "metrics", "offpath_metrics.json"))))
    write(out, "kid_long", long_rows, "KID x10^3 and FID per scene, round, shift and camera", "6.4.1-a, 6.4.1-e",
          "round_*/metrics/offpath_metrics.json",
          md_cols=["scene", "round", "shift", "camera", "kid_x1e3", "kid_std_x1e3", "fid"])
    write(out, "kid_curves", summ, "KID-vs-offset curve summaries (pooled, x10^3)", "6.4.1-b, 6.4.1-d",
          "round_*/metrics/offpath_metrics.json (camera = pooled)",
          md_cols=["scene", "status", "round", "max_steps", "bank", "side", "kid_0m", "kid_far", "penalty", "ratio",
                   "slope_ls", "first_resolvable_m", "adjacent_steps_resolvable"],
          notes=["penalty = KID(far) - KID(0 m), far = 3 m. slope_ls is the least-squares slope over 0-3 m; "
                 "slope_endpoint = penalty / 3. A step is resolvable when the gap exceeds "
                 "2 * sqrt(std_a^2 + std_b^2) of the subset spreads. Pooled and per-camera KID are on different "
                 "scales and are never mixed here."])


# --------------------------------------------------------------------------- #
# C7  The loop: KID by round at fixed steps, Difix statistics, step split     #
# --------------------------------------------------------------------------- #

def tb_split(train_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    f = glob.glob(os.path.join(train_dir, "tb", "events.*"))
    if not f:
        return None
    ea = EventAccumulator(f[0], size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    nr = len(ea.Scalars("train/loss_real")) if "train/loss_real" in tags else 0
    npse = len(ea.Scalars("train/loss_pseudo")) if "train/loss_pseudo" in tags else 0
    return nr, npse


def c7_loop(scenes, out):
    eff = []
    for sc in scenes:
        g = fixed_step_group(sc["rounds"])
        if len(g) < 2:
            continue
        first, last = g[0], g[-1]
        c0, c1 = curves(offpath_rows(first["dir"]) or {"rows": []}), curves(offpath_rows(last["dir"]) or {"rows": []})
        v0, v1 = val_stats(first["train"]), val_stats(last["train"])
        for side in sorted(set(c0) & set(c1)):
            a, b = curve_summary(c0[side]), curve_summary(c1[side])
            eff.append(OrderedDict(
                scene=sc["scene"], status=sc["status"], side=side, fixed_steps=first["max_steps"],
                rounds=f"{first['round']}->{last['round']}", bank=f"{first['bank']}->{last['bank']}",
                kid_0m_first=a["kid_0m"], kid_0m_last=b["kid_0m"], d_kid_0m=b["kid_0m"] - a["kid_0m"],
                kid_far_first=a["kid_far"], kid_far_last=b["kid_far"],
                d_kid_far_pct=100 * (b["kid_far"] / a["kid_far"] - 1),
                penalty_first=a["penalty"], penalty_last=b["penalty"],
                d_penalty_pct=100 * (b["penalty"] / a["penalty"] - 1) if a["penalty"] else None,
                slope_ls_first=a["slope_ls"], slope_ls_last=b["slope_ls"],
                psnr_first=v0.get("psnr"), psnr_last=v1.get("psnr"),
                d_psnr=(v1["psnr"] - v0["psnr"]) if v0 and v1 else None))
    ch = [r for r in eff if r["status"] == "chapter" and r["side"] != "up"]
    notes = []
    if ch:
        notes.append(f"Chapter scenes, lateral sides: penalty falls in {sum(r['penalty_last'] < r['penalty_first'] for r in ch)} "
                     f"of {len(ch)}; KID at 0 m rises in {sum(r['d_kid_0m'] > 0 for r in ch)} of {len(ch)}; "
                     f"KID at 3 m falls in {sum(r['d_kid_far_pct'] < 0 for r in ch)} of {len(ch)}.")
    write(out, "loop_fixed_steps", eff, "The bank's effect at a fixed step budget (first vs last fixed-step round)",
          "6.4.2-a, 6.4.2-b, 6.4.2-f", "round_*/metrics/offpath_metrics.json; round_*/train/stats/val_step*.json",
          md_cols=["scene", "status", "side", "fixed_steps", "rounds", "kid_0m_first", "kid_0m_last", "kid_far_first",
                   "kid_far_last", "penalty_first", "penalty_last", "d_penalty_pct", "d_psnr"],
          notes=notes + ["30k x4 scenes compare rounds 0 and 3; 7k scenes compare rounds 0 and 2 (round 3 adds 23k "
                         "steps). KID is pooled, x10^3."])

    by_round = []
    for sc in scenes:
        for r in sc["rounds"]:
            om = offpath_rows(r["dir"])
            if not om:
                continue
            for side, c in curves(om).items():
                s = curve_summary(c)
                by_round.append(OrderedDict(scene=sc["scene"], status=sc["status"], side=side, round=r["round"],
                                            max_steps=r["max_steps"], bank=r["bank"], kid_0m=s["kid_0m"],
                                            kid_1m=c[1][1] if len(c) > 1 else None,
                                            kid_2m=c[2][1] if len(c) > 2 else None, kid_3m=s["kid_far"],
                                            penalty=s["penalty"], slope_ls=s["slope_ls"]))
    write(out, "loop_kid_by_round", by_round, "KID x10^3 at every offset, every round", "6.4.1-f, 6.4.2-a, 6.4.2-b",
          "round_*/metrics/offpath_metrics.json (pooled)")

    dfx, ctrl, split = [], [], []
    for sc in scenes:
        for r in sc["rounds"]:
            p = os.path.join(r["dir"], "difix", "stats_summary.json")
            if os.path.exists(p):
                bs = jload(p)["by_shift"]
                for shift, v in sorted(bs.items()):
                    dfx.append(OrderedDict(scene=sc["scene"], status=sc["status"], round=r["round"], shift=shift,
                                           n=v["n"], mean_abs_delta=v["mean_abs_delta"],
                                           p95_abs_delta=v["p95_abs_delta"], frac_changed=v["frac_changed"]))
                o = bs.get("original")
                if o and o.get("lpips_render_vs_real") is not None:
                    ctrl.append(OrderedDict(scene=sc["scene"], status=sc["status"], round=r["round"], n=o["n"],
                                            psnr_render=o["psnr_render_vs_real"], psnr_difix=o["psnr_difix_vs_real"],
                                            d_psnr=o["psnr_difix_vs_real"] - o["psnr_render_vs_real"],
                                            lpips_render=o["lpips_render_vs_real"],
                                            lpips_difix=o["lpips_difix_vs_real"],
                                            d_lpips=o["lpips_difix_vs_real"] - o["lpips_render_vs_real"]))
            sp = tb_split(r["train"])
            if sp:
                nr, npse = sp
                split.append(OrderedDict(scene=sc["scene"], round=r["round"], max_steps=r["max_steps"], bank=r["bank"],
                                         logged_real=nr, logged_pseudo=npse,
                                         pseudo_share=npse / (nr + npse) if (nr + npse) and npse else 0.0))
    write(out, "difix_by_round", dfx, "Difix correction magnitude per round and shift level", "6.4.2-d",
          "round_*/difix/stats_summary.json (by_shift)")
    nctl = [r for r in ctrl if r["status"] == "chapter"]
    notes = []
    if nctl:
        notes.append(f"Chapter scenes: LPIPS lower after cleaning in {sum(r['d_lpips'] < 0 for r in nctl)} of "
                     f"{len(nctl)} control rounds; PSNR higher in {sum(r['d_psnr'] > 0 for r in nctl)} of {len(nctl)}.")
    write(out, "difix_control", ctrl, "Difix 0 m control: render vs real against cleaned vs real", "6.4.2-e",
          "round_*/difix/stats_summary.json (original)", notes=notes)
    write(out, "pseudo_step_share", split, "Share of logged training steps that drew a pseudo-view", "6.4.2-c",
          "round_*/train/tb/events.* (train/loss_real and train/loss_pseudo event counts)",
          notes=["Configured: pseudo_sample_prob 0.3, real loss x1.5 (configs/gaussian_splatting/train.yaml)."])


# --------------------------------------------------------------------------- #
# C8  Trajectory pipeline and vehicle survival                                #
# --------------------------------------------------------------------------- #

def c8_trajectories(scenes, out):
    funnel, chain, raudit_rows = [], [], []
    rcfgs = []
    for sc in scenes:
        R = sc["refine"]
        if not R:
            continue
        rep = jload(os.path.join(R, "refine_report.json"))
        bf = rep.get("bicycle_fit", {})
        tr = bf.get("tracks", {}) or {}
        rmse = [t["pos_rmse"] for t in tr.values() if t.get("pos_rmse") is not None]
        hc = os.path.join(R, ".hydra", "config.yaml")
        rcfg = yaml.safe_load(open(hc)) if os.path.exists(hc) else {}
        rt = rcfg.get("refine_task", {}) or {}
        gate = (rt.get("bicycle_fit") or {}).get("max_fit_pos_rmse") if isinstance(rt.get("bicycle_fit"), dict) else None
        rcfgs.append((sc, flatten(rt) if isinstance(rt, dict) else {}))
        funnel.append(OrderedDict(
            scene=sc["scene"], status=sc["status"], tracks_in=rep.get("tracks_in"),
            tracks_after_fusion=rep.get("tracks_after_fusion"), merges=rep.get("merges"),
            tracks_kept=rep.get("tracks_kept"), dropped_static=rep.get("dropped_static"),
            dropped_short=rep.get("dropped_short"), frames_filled=rep.get("frames_filled"),
            frames_extended=rep.get("frames_extended"), frames_smoothed=rep.get("frames_smoothed"),
            bicycle_fitted=bf.get("tracks_fitted"), bicycle_rejected=len(bf.get("rejected_ids", [])),
            gate_max_fit_pos_rmse=gate, fit_tracks_with_stats=len(tr), pos_rmse_median=median(rmse),
            pos_rmse_max=max(rmse) if rmse else None,
            resumed_from_snapshot=bf.get("resumed_from_post_snapshot"), source=rel(R)))
        tj = jload(os.path.join(R, "track_3d_refined_colmap.json"))["results"]
        ids = {}
        for fr in tj.values():
            for b in fr:
                ids.setdefault(b["tracking_id"], b["tracking_name"])
        cls = Counter(ids.values())
        row = OrderedDict(scene=sc["scene"], status=sc["status"], curated_tracks=len(ids),
                          classes=" ".join(f"{k}:{v}" for k, v in sorted(cls.items())),
                          vehicle_tracks=sum(v for k, v in cls.items() if k in DEFAULT_RIGID_CLASSES))
        if sc["rounds"] and sc["rounds"][0]["dynamic"]:
            for tag, r in (("r0", sc["rounds"][0]), ("final", sc["rounds"][-1])):
                d = rigid_detail(r["train"])
                if d:
                    row[f"{tag}_round"] = r["round"]
                    row[f"{tag}_instances_defined"] = d["total"]
                    row[f"{tag}_alive"] = d["kept"]
                    row[f"{tag}_viable"] = d.get("viable")
                    row[f"{tag}_rigid_GS"] = d["n_rigid"]
                    row[f"{tag}_effective_capacity"] = d.get("effective")
        chain.append(row)
    write(out, "trajectory_funnel", funnel, "Trajectory pipeline per scene", "6.3-d, 6.3-e, 6.3-f",
          "15_refine_*/refine_report.json; 15_refine_*/.hydra/config.yaml",
          md_cols=["scene", "status", "tracks_in", "tracks_after_fusion", "merges", "tracks_kept", "dropped_static",
                   "dropped_short", "bicycle_fitted", "bicycle_rejected", "gate_max_fit_pos_rmse", "pos_rmse_median",
                   "pos_rmse_max"])
    write(out, "vehicle_chain", chain, "Curated tracks -> vehicle tracks -> rigid instances -> survivors",
          "6.3-g, 6.3-h", "track_3d_refined_colmap.json; round_*/train/ckpts/ckpt_*_rank0.pt (rigid_nodes)",
          notes=[f"Vehicle classes are DEFAULT_RIGID_CLASSES {DEFAULT_RIGID_CLASSES}. instances_defined counts "
                 f"instances the tracks define; alive owns at least one Gaussian; viable at least "
                 f"{VIABLE_MIN_GAUSSIANS} (collect_ablation.VIABLE_MIN_GAUSSIANS)."])
    # Refinement-config drift across scenes (6.3-i).
    keys = sorted({k for _, c in rcfgs for k in c})
    for k in keys:
        vals = [json.dumps(c.get(k), default=str) for _, c in rcfgs]
        if any("/home/" in v for v in vals):
            continue
        mode, n = Counter(vals).most_common(1)[0]
        for (sc, c), v in zip(rcfgs, vals):
            if v != mode:
                raudit_rows.append(OrderedDict(scene=sc["scene"], key=k, value=v, mode=mode,
                                               scenes_at_mode=f"{n}/{len(rcfgs)}"))
    write(out, "refine_config_audit", raudit_rows, "Refinement-config deviations across scenes", "6.3-i",
          "15_refine_*/.hydra/config.yaml (refine_task)",
          notes=[f"{len(rcfgs)} refine configs compared key by key, paths excluded. "
                 + ("No deviations: every scene was refined with the same configuration."
                    if not raudit_rows else "Deviations listed below.")])


# --------------------------------------------------------------------------- #
# C9  Curation: manual edits per scene                                        #
# --------------------------------------------------------------------------- #

def parse_commands(path):
    typed, undos, net, iters = Counter(), 0, 0, 0
    for line in open(path):
        line = line.strip()
        m = re.match(r"iter \d+ \| (\w+)(?: \| (.*))?$", line)
        if m:
            iters += 1
            if m.group(1) == "apply" and m.group(2):
                for cmd in m.group(2).split(";"):
                    cmd = cmd.strip()
                    if cmd:
                        typed[cmd.split()[0]] += 1
            elif m.group(1) == "undo":
                undos += 1
        m = re.match(r"commands_applied_total:\s*(\d+)", line)
        if m:
            net += int(m.group(1))
    return typed, undos, net, iters


def c9_curation(scenes, out):
    rows = []
    for sc in scenes:
        if not sc["refine"]:
            continue
        verbs, undos, net, iters, phases = Counter(), 0, 0, 0, {}
        for f in sorted(glob.glob(os.path.join(sc["refine"], "user_refinement*", "commands_applied.txt"))):
            t, u, n, i = parse_commands(f)
            phase = "post_bicycle" if "post_bicycle" in f else "pre_bicycle"
            phases[phase] = sum(t.values())
            verbs += t
            undos, net, iters = undos + u, net + n, iters + i
        typed = sum(verbs.values())
        rows.append(OrderedDict(scene=sc["scene"], status=sc["status"], typed=typed, undos=undos, net_applied=net,
                                iterations=iters, pre_bicycle=phases.get("pre_bicycle", 0),
                                post_bicycle=phases.get("post_bicycle", 0),
                                **{f"verb_{k}": v for k, v in sorted(verbs.items())}))
    ch = [r for r in rows if r["status"] != "not run"]
    notes = []
    if ch:
        tot = Counter()
        for r in ch:
            tot.update({k: v for k, v in r.items() if k.startswith("verb_")})
        allv = sum(tot.values())
        notes.append(f"{len(ch)} curated scenes: typed commands median {np.median([r['typed'] for r in ch]):.0f}, "
                     f"range {min(r['typed'] for r in ch)}-{max(r['typed'] for r in ch)}; net applied median "
                     f"{np.median([r['net_applied'] for r in ch]):.0f}. Verb shares: "
                     + ", ".join(f"{k[5:]} {100 * v / allv:.0f}%" for k, v in tot.most_common()))
    write(out, "curation", rows, "Manual curation per scene", "6.3-j",
          "15_refine_*/user_refinement*/commands_applied.txt", notes=notes)


# --------------------------------------------------------------------------- #
# C10  Cost: time and memory                                                  #
# --------------------------------------------------------------------------- #

def stage_duration(dir_glob):
    for d in sorted(glob.glob(dir_glob)):
        p = os.path.join(d, ".success")
        if os.path.exists(p):
            m = re.search(r"duration:\s*(\d+:\d\d:\d\d)", open(p).read())
            if m:
                return hms(m.group(1))
    return None


def c10_cost(scenes, out):
    rows, prep = [], []
    for sc in scenes:
        for r in sc["rounds"]:
            d = r["durations"]
            t = train_stats(r["train"])
            rows.append(OrderedDict(
                scene=sc["scene"], status=sc["status"], round=r["round"], max_steps=r["max_steps"],
                bank=r["bank"], dynamic=r["dynamic"], train_s=hms(d.get("train")), render_s=hms(d.get("render")),
                metrics_s=hms(d.get("metrics")), difix_s=hms(d.get("difix")), round_s=hms(d.get("round_total")),
                cached=",".join(k for k, v in d.items() if v == "cached") or None,
                train_loop_s=t.get("ellipse_time"), peak_mem_alloc_gb=t.get("mem"), num_GS=t.get("num_GS")))
        if sc["state"]:
            sd = sc["dir"]
            prep.append(OrderedDict(
                scene=sc["scene"], status=sc["status"], sam_masks_s=stage_duration(os.path.join(sd, "01_masks_*")),
                fused_masks_s=stage_duration(os.path.join(sd, "02_fused_masks_*")),
                ncore_s=stage_duration(os.path.join(sd, "03_ncore_dynamic_*")),
                real_bank_s=stage_duration(os.path.join(sd, "035_real_frames_*")),
                loop_prep_s=hms(sc["state"].get("durations", {}).get("prep")),
                loop_total_s=hms(sc["state"].get("durations", {}).get("total"))))
    write(out, "cost_by_round", rows, "Wall-clock time and peak allocated GPU memory per round (RTX 5070)",
          "6.5-d, 6.5-e", "loop_state.json (durations); round_*/train/stats/train_step*_rank0.json",
          md_cols=["scene", "round", "max_steps", "bank", "train_s", "render_s", "metrics_s", "difix_s", "round_s",
                   "cached", "peak_mem_alloc_gb"],
          notes=["peak_mem_alloc_gb is torch max_memory_allocated; reserved memory was not logged. 'cached' marks "
                 "steps a resumed run skipped, whose original timings were overwritten."])
    write(out, "cost_prep", prep, "Preparation stages per scene (RTX 5070)", "6.5-g",
          "<stage>_*/.success (duration line); loop_state.json",
          notes=["Tracking and refinement do not record a duration; curation time is not logged."])


# --------------------------------------------------------------------------- #
# C11  Bootstrap confidence intervals on KID (GPU, no training)               #
# --------------------------------------------------------------------------- #

def far_by_side(shift_names):
    """{side: largest-offset shift name} for a set of shift-level names."""
    out = {}
    for n in shift_names:
        if n == "original":
            continue
        ax, v = n.split("_", 1)
        side = SIDE[(ax, int(math.copysign(1, float(v))))]
        if side not in out or abs(float(v)) > abs(float(out[side].split("_", 1)[1])):
            out[side] = n
    return out


def c11_bootstrap(scenes, names, out, device, draws=1000, seed=0):
    """Paired frame bootstrap (scripts/experiments/kid_bootstrap.py). One set of draws serves
    every round of a scene, so the change in the 3 m penalty between rounds gets an interval."""
    import torch
    from offpath_metrics import embed, load_ego_masks, real_paths_by_camera, render_paths
    from torchmetrics.image.fid import NoTrainInceptionV3
    from scripts.experiments.kid_bootstrap import SceneBootstrap, ci

    inception = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()
    rows, prow, erow = [], [], []
    for sc in scenes:
        if sc["scene"] not in names:
            continue
        rounds = [r for r in sc["rounds"] if offpath_rows(r["dir"])]
        if not rounds:
            continue
        ov = os.path.join(rounds[0]["dir"], "metrics", "hydra", ".hydra", "overrides.yaml")
        data_dir = next((l.split("=", 1)[1].strip() for l in open(ov) if "metrics_task.data_dir=" in l), None)
        cams = sc["state"]["cameras"]
        masks = load_ego_masks(data_dir, cams)
        real = real_paths_by_camera(sc["bank"])
        boot = SceneBootstrap(torch.cat([embed(real[c], inception, masks.get(c), device) for c in cams]),
                              draws=draws, seed=seed, device=device)
        res = {}
        for r in rounds:
            stage = {x["shift_name"]: x for x in offpath_rows(r["dir"])["rows"] if x["camera"] == "pooled"}
            render_dir = os.path.join(r["dir"], "render", "full")
            for s in sorted(stage, key=lambda s: (s != "original", s)):
                ff = torch.cat([embed(render_paths(render_dir, s, c), inception, masks.get(c), device) for c in cams])
                point, d = boot.score(ff)
                res[(r["round"], s)] = (point, d)
                lo, hi = ci(d * 1e3)
                rows.append(OrderedDict(scene=sc["scene"], status=sc["status"], round=r["round"], bank=r["bank"],
                                        shift=s, kid_full_x1e3=point * 1e3, ci95_lo=lo, ci95_hi=hi,
                                        stage_kid_x1e3=stage[s]["kid_mean"] * 1e3,
                                        stage_std_x1e3=stage[s]["kid_std"] * 1e3, n_fake=int(ff.shape[0])))
            for s in stage:
                if s == "original":
                    continue
                d = (res[(r["round"], s)][1] - res[(r["round"], "original")][1]) * 1e3
                lo, hi = ci(d)
                prow.append(OrderedDict(scene=sc["scene"], status=sc["status"], round=r["round"], bank=r["bank"],
                                        shift=s, penalty_x1e3=(res[(r["round"], s)][0] - res[(r["round"], "original")][0]) * 1e3,
                                        ci95_lo=lo, ci95_hi=hi, p_le_0=float((d <= 0).mean())))
            print(f"    {sc['scene']} r{r['round']}: bootstrapped {len(stage)} levels")
        g = [r for r in fixed_step_group(sc["rounds"]) if (r["round"], "original") in res]
        if len(g) >= 2:
            f, l = g[0]["round"], g[-1]["round"]
            common = {s for (rr, s) in res if rr == f} & {s for (rr, s) in res if rr == l}
            for side, far in sorted(far_by_side(common).items()):
                pf_d = res[(f, far)][1] - res[(f, "original")][1]
                pl_d = res[(l, far)][1] - res[(l, "original")][1]
                pf = res[(f, far)][0] - res[(f, "original")][0]
                pl = res[(l, far)][0] - res[(l, "original")][0]
                rel = 100.0 * (pl_d / pf_d - 1.0)
                d0 = (res[(l, "original")][1] - res[(f, "original")][1]) * 1e3
                erow.append(OrderedDict(
                    scene=sc["scene"], status=sc["status"], side=side, fixed_steps=g[0]["max_steps"],
                    rounds=f"{f}->{l}", penalty_first=pf * 1e3, penalty_last=pl * 1e3,
                    penalty_change_pct=100.0 * (pl / pf - 1.0), change_ci95_lo=ci(rel)[0], change_ci95_hi=ci(rel)[1],
                    d_kid0_x1e3=(res[(l, "original")][0] - res[(f, "original")][0]) * 1e3,
                    d_kid0_ci95_lo=ci(d0)[0], d_kid0_ci95_hi=ci(d0)[1]))
    note = [f"Full-sample unbiased MMD^2 (polynomial kernel, degree 3, gamma 1/2048, coef 1) with {draws} frame-"
            f"bootstrap draws shared by every shift level and every round of a scene, so penalties and their changes "
            f"between rounds get paired intervals. Ego pixels masked as in the metrics stage."]
    write(out, "kid_bootstrap", rows, "KID with frame-bootstrap 95% intervals (pooled, x10^3)", "6.4.1-c",
          "round_*/render/full/frames, 035_real_frames_* (re-embedded); offpath_metrics.json for the stage values",
          notes=note)
    write(out, "kid_bootstrap_penalty", prow, "Off-path penalty KID(s) - KID(0 m) with paired 95% intervals",
          "6.4.1-c", "as kid_bootstrap", notes=note)
    write(out, "kid_bootstrap_loop_effect", erow,
          "The bank's effect on the 3 m penalty at fixed steps, with paired 95% intervals", "6.4.1-c, 6.4.2-a, 6.4.2-b",
          "as kid_bootstrap", notes=note + ["30k x4 scenes compare rounds 0 and 3; 7k scenes rounds 0 and 2."])


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "chapter6"))
    ap.add_argument("--only", default="scenes,audit,onpath,percamera,kid,loop,trajectories,curation,cost")
    ap.add_argument("--scenes", default=None, help="comma-separated subset")
    ap.add_argument("--bootstrap", default="", help="comma-separated scenes for C11")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else set()
    scenes = discover(set(a.scenes.split(",")) if a.scenes else None)
    print(f"{len(scenes)} scenes: " + ", ".join(f"{s['scene']} ({s['status']})" for s in scenes))
    meta = {r["scene_id"]: r for r in csv.DictReader(open(METADATA))}
    t0 = time.time()
    steps = [("scenes", lambda: c2_scenes(scenes, meta, a.out)), ("audit", lambda: c3_audit(scenes, a.out)),
             ("onpath", lambda: c4_onpath(scenes, a.out)), ("percamera", lambda: c5_percamera(scenes, a.out, a.device)),
             ("kid", lambda: c6_kid(scenes, a.out)), ("loop", lambda: c7_loop(scenes, a.out)),
             ("trajectories", lambda: c8_trajectories(scenes, a.out)), ("curation", lambda: c9_curation(scenes, a.out)),
             ("cost", lambda: c10_cost(scenes, a.out))]
    for name, fn in steps:
        if name in only:
            print(f"[{name}]")
            fn()
    if a.bootstrap:
        print("[bootstrap]")
        c11_bootstrap(scenes, set(a.bootstrap.split(",")), a.out, a.device)

    git = lambda *c: subprocess.run(["git", *c], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    readme = os.path.join(a.out, "README.md")
    old = {}
    if os.path.exists(readme):  # keep entries of tables this invocation did not rewrite
        for line in open(readme):
            m = re.match(r"\| \[(\w+)\]", line)
            if m:
                old[m.group(1)] = line.rstrip("\n")
    lines = ["# Chapter 6 tables", "",
             "Generated by `scripts/experiments/collect_chapter6.py`; do not edit by hand.", "",
             f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')} in {time.time() - t0:.0f} s",
             f"- last command: `{' '.join(sys.argv)}`",
             f"- git: {git('log', '-1', '--oneline')} (working tree {'dirty' if git('status', '--short') else 'clean'})",
             f"- scenes: " + ", ".join(f"{s['scene']} ({s['status']})" for s in scenes), "",
             "Each table row carries the time it was generated; tables not rewritten by the last command keep "
             "their earlier time.", "",
             "| table | title | plan items | source | rows | generated |", "|---|---|---|---|---|---|"]
    cell = lambda x: str(x).replace("|", "\\|")
    new = {n: f"| [{n}]({n}.md) | {cell(t)} | {cell(i)} | {cell(s)} | {k} | {g} |" for n, t, i, s, k, g in REGISTRY}
    old.update(new)
    lines += [old[k] for k in sorted(old)]
    with open(readme, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"done in {time.time() - t0:.0f} s -> {rel(a.out)}")


if __name__ == "__main__":
    main()
