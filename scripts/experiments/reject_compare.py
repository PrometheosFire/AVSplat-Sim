import json, copy
from src.tracking.refine_tracks import apply_bicycle_fit, _track_class, _frame_index
from scripts.experiments.alpha_sweep import _find_predictions, _fixed_fit_tracks, _load_cfg

cfg = _load_cfg()
data = json.load(open(_find_predictions("scene_099")))
results = data.get("results", data)
frame_keys, _ = _frame_index(results)
kept = _fixed_fit_tracks(results, cfg)
label = {int(t): _track_class(tr) for t, tr in kept.items()}


def run(bcfg):
    return apply_bicycle_fit(copy.deepcopy(kept), frame_keys, bcfg)[1]


off = dict(cfg)
off.update(preclean_yaw_flip=False, preclean_outlier_dist=0.0, trim_refit=False)
rep_off = run(off)
rep_on = run(dict(cfg))


def show(tag, rep):
    rej = sorted(int(x) for x in rep["rejected_ids"])
    print(f"[{tag}] fitted={rep['tracks_fitted']} rejected={len(rej)} "
          f"precleaned_frames={rep.get('tracks_precleaned_frames')} "
          f"trimmed={rep.get('tracks_trimmed')}")
    print("        rejected:", [f"{label.get(t)}_{t}" for t in rej])


show("OFF (old)", rep_off)
show("ON  (new)", rep_on)
print("bus_30 rejected OFF:", 30 in rep_off["rejected_ids"],
      "| ON:", 30 in rep_on["rejected_ids"])
