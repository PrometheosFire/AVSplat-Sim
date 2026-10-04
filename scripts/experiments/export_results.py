"""Package the extended_4dgs runs for analysis on another machine.

Tier A (one archive): every number and record needed to analyse the runs --
loop_state, per-round validation stats, off-path KID/FID, Difix stats and
manifests, training configs, logs and tensorboard events, the curated tracks
with their command logs and BEVs, plus a flat metrics CSV and the run settings.

Tier B (one archive per scene, so a Drive upload resumes per file): the
visuals -- final-round rendered frames, every round's videos, validation
renders, and the Difix-cleaned pseudo-views of each round.

Only the newest loop dir per scene is exported (older ones are failed or
superseded attempts). Files are archived in place (tar --files-from), nothing is
copied first. Archives are zstd-compressed; images barely compress, so B is
close to its raw size.

Usage:  envs/envs/env_gsplat/bin/python scripts/experiments/export_results.py OUT_DIR [--dry] [--csv-only] [--skip=scene_a,scene_b]

--csv-only writes results/export_generated/metrics_by_round.csv (and git_state.txt)
and stops before archiving.
"""
import csv, glob, hashlib, json, os, subprocess, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
W = "results/extended_4dgs/wayve101"


def newest_loop(scene: str):
    loops = [d for d in glob.glob(f"{ROOT}/{W}/{scene}/07_difix_4dgs_*")
             if os.path.exists(f"{d}/loop_state.json")]
    return max(loops, key=os.path.getmtime) if loops else None


def rel(p):
    return os.path.relpath(p, ROOT)


def files(pattern):
    return sorted(rel(p) for p in glob.glob(f"{ROOT}/{pattern}", recursive=True) if os.path.isfile(p))


def tier_a(scene, loop):
    L = rel(loop)
    out = [f"{L}/loop_state.json"]
    for pat in ("round_*/train/stats/*.json", "round_*/train/cfg.yml", "round_*/train/tb/*",
                "round_*/train/train_splats.log", "round_*/metrics/offpath_metrics.json",
                "round_*/difix/stats_summary.json", "round_*/difix/stats.csv",
                "round_*/difix/manifest.json", "round_*/render/render_standalone.log"):
        out += files(f"{L}/{pat}")
    for refine in glob.glob(f"{ROOT}/{W}/{scene}/15_refine_*"):
        R = rel(refine)
        out += files(f"{R}/*.json") + files(f"{R}/*.log") + files(f"{R}/vis/*.png")
        out += files(f"{R}/user_refinement*/commands_applied.txt")
        out += files(f"{R}/user_refinement*/[0-9]*/refine_report_user.json")
        out += files(f"{R}/user_refinement*/[0-9]*/bev_refined.png")
    out += files(f"{W}/{scene}/035_real_frames_*/index.json")
    out += files(f"results/prep_logs/train_curated/{scene}.log")
    return out


def tier_b(scene, loop):
    L = rel(loop)
    n = len(json.load(open(f"{loop}/loop_state.json"))["rounds"])
    out = files(f"{L}/round_{n - 1:03d}/render/full/frames/**/*")
    out += files(f"{L}/round_*/render/full/videos/*")
    out += files(f"{L}/round_{n - 1:03d}/train/renders/*")
    out += files(f"{L}/round_*/difix/frames/**/*")
    return out


def metrics_rows(scene, loop):
    st = json.load(open(f"{loop}/loop_state.json"))
    for r in st["rounds"]:
        rd = f"{loop}/round_{r['round']:03d}"
        val = glob.glob(f"{rd}/train/stats/val_step*.json")
        v = json.load(open(val[0])) if val else {}
        base = dict(scene=scene, loop=os.path.basename(loop), round=r["round"],
                    max_steps=r["max_steps"], dynamic=r.get("dynamic"),
                    vehicle_free=st.get("vehicle_free"), mode=st.get("dynamic_mode"),
                    train_time=r["durations"].get("train"), difix_time=r["durations"].get("difix"),
                    psnr=v.get("psnr"), ssim=v.get("ssim"), lpips=v.get("lpips"),
                    num_GS=v.get("num_GS"), num_rigid_GS=v.get("num_rigid_GS"),
                    rigid_instances=v.get("rigid_instances_alive"), dynamic_psnr=v.get("dynamic_psnr"))
        m = f"{rd}/metrics/offpath_metrics.json"
        if not os.path.exists(m):
            yield dict(base, shift="", camera="", kid=None, kid_std=None, fid=None, kid_noise_floor=None)
            continue
        d = json.load(open(m))
        for row in d["rows"]:
            yield dict(base, shift=row["shift_name"], camera=row["camera"], kid=row["kid_mean"],
                       kid_std=row["kid_std"], fid=row["fid"],
                       kid_noise_floor=d["noise_floor"]["kid_mean"])


def tar_zst(out_path, file_list, list_dir):
    lst = os.path.join(list_dir, os.path.basename(out_path) + ".files")
    with open(lst, "w") as f:
        f.write("\n".join(file_list) + "\n")
    subprocess.run(["tar", "-C", ROOT, "--files-from", lst, "-I", "zstd -T0 -3", "-cf", out_path], check=True)
    os.remove(lst)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out_dir = os.path.abspath(sys.argv[1]); dry = "--dry" in sys.argv
    os.makedirs(out_dir, exist_ok=True)
    skip = set()
    for a in sys.argv:
        if a.startswith("--skip="):
            skip |= set(a.split("=", 1)[1].split(","))
    scenes = sorted(s for s in os.listdir(f"{ROOT}/{W}") if newest_loop(s) and s not in skip)
    a_files, b_files, rows = [], {}, []
    for s in scenes:
        loop = newest_loop(s)
        a_files += tier_a(s, loop)
        b_files[s] = tier_b(s, loop)
        rows += list(metrics_rows(s, loop))
        size = lambda fl: sum(os.path.getsize(f"{ROOT}/{p}") for p in fl) / 1e9
        print(f"{s}: {os.path.basename(loop)}  A {size(tier_a(s, loop))*1e3:6.1f} MB  B {size(b_files[s]):5.2f} GB", flush=True)

    # Settings and code state alongside the data, so the numbers can be traced.
    extra = [rel(p) for p in [f"{ROOT}/scripts/experiments/run_curated_scenes.sh",
                              f"{ROOT}/scripts/experiments/export_results.py"]]
    extra += files("configs/**/*.yaml")
    a_files += extra
    print(f"total: A {sum(os.path.getsize(f'{ROOT}/{p}') for p in a_files)/1e6:.0f} MB raw, "
          f"B {sum(os.path.getsize(f'{ROOT}/{p}') for fl in b_files.values() for p in fl)/1e9:.1f} GB raw, "
          f"{len(scenes)} scenes")
    if dry:
        return

    # Generated files are staged under results/ (git-ignored) so they sit inside
    # ROOT and archive with clean relative paths.
    gen = f"{ROOT}/results/export_generated"; os.makedirs(gen, exist_ok=True)
    for p in os.environ.get("EXPORT_EXTRA", "").split(":"):
        if p and os.path.isfile(p):
            subprocess.run(["cp", p, gen], check=True)
    with open(f"{gen}/metrics_by_round.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(f"{gen}/git_state.txt", "w") as f:
        for cmd in (["git", "log", "-1", "--oneline"], ["git", "status", "--short"], ["git", "diff"]):
            f.write(f"$ {' '.join(cmd)}\n" + subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout + "\n")
    gen_rel = [os.path.relpath(p, ROOT) for p in glob.glob(f"{gen}/*")]
    if "--csv-only" in sys.argv:
        print(f"wrote {gen}/metrics_by_round.csv ({len(rows)} rows)")
        return

    t = time.time()
    tar_zst(f"{out_dir}/A_analysis.tar.zst", a_files + gen_rel, out_dir)
    print(f"A done ({time.time()-t:.0f}s)", flush=True)
    for s, fl in b_files.items():
        t = time.time()
        tar_zst(f"{out_dir}/B_visuals_{s}.tar.zst", fl, out_dir)
        print(f"B {s} done ({time.time()-t:.0f}s)", flush=True)
    with open(f"{out_dir}/SHA256SUMS", "w") as f:
        for p in sorted(glob.glob(f"{out_dir}/*.tar.zst")):
            f.write(f"{sha256(p)}  {os.path.basename(p)}\n")
    print("checksums written")


if __name__ == "__main__":
    main()
