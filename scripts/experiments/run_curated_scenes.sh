#!/usr/bin/env bash
# Difix 4DGS loop on the scenes whose tracks are already hand-curated.
#
# ONE PROCESS PER SCENE (see run_rigid_reset_study.sh for why): a CUDA OOM in
# one scene costs that scene only. Re-runnable: run_difix_4dgs.py caches every
# step (masks, ncore, bank, and each round's train/render/difix/metrics), so a
# restart resumes at the first unfinished step.
#
# user_refinement.enabled=false on every run: the curated tracks are read from
# the refine dir (cache hit), and the loop never blocks on stdin at
# `user-refine>`.
#
# Memory (RTX 5070, 12 GB): every process runs with expandable_segments. The
# highway scene_050 OOMed in its 30k round at 3M and diverged at 2.5M: the OOM
# was a symptom of that divergence, which means_lr=4e-5 fixes (fits at 3M, 7.0 GB
# peak). scene_071 OOMed at 5M and again at 3M (step 24.7k of its 30k round), so
# it runs at 2.5M.
# SAM on scene_024 OOMed at chunk_size 50, 20 and 10, with and without offload:
# right-backward frames 100-109 hold 27 pedestrians, and SAM's memory attention
# over that many tracked objects needs >10 GB for a 10-frame chunk (8.4 GB peak
# at 5, 6.9 GB at 2; offload changes neither). It runs chunk_size 5.
#
# scene_041's curated tracks contain no vehicles (every car was parked), so
# run_difix_4dgs.py trains all its rounds static. Run it by name.
#
# Usage:  bash scripts/experiments/run_curated_scenes.sh [scene_050 ...]
#         (no args = the scenes in SCENES below, in that order)

set -u
cd "$(dirname "$0")/../.." || exit 1

LOGDIR=results/prep_logs/train_curated
PYBIN=envs/envs/env_gsplat/bin/python
mkdir -p "$LOGDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON=(++refine_task.user_refinement.enabled=false)

# 3 Difix rounds + final round, left or right ramp in 1 m steps (cumulative).
STEPS_7K='++loop4d_task.max_steps_per_round=[7000,7000,7000,30000]'
LEFT3=('++loop4d_task.target_shift=[-3,0,0]' '++loop4d_task.step_shift=[-1,0,0]')
RIGHT3=('++loop4d_task.target_shift=[3,0,0]' '++loop4d_task.step_shift=[1,0,0]')
# Vertical: y is +down / -up (configs/rendering/render.yaml), so "up" is -y.
UP3=('++loop4d_task.target_shift=[0,-3,0]' '++loop4d_task.step_shift=[0,-1,0]')
# -3 m and +3 m together. The generated ramp is single-direction, so a
# bilateral curriculum must be an explicit shift_schedule; this one mirrors
# schedule_mode=cumulative (round r cleans every level up to r+1, both sides).
BILATERAL3='++loop4d_task.shift_schedule=[[[-1,0,0],[1,0,0]],[[-1,0,0],[1,0,0],[-2,0,0],[2,0,0]],[[-1,0,0],[1,0,0],[-2,0,0],[2,0,0],[-3,0,0],[3,0,0]]]'

args_for() {
  case "$1" in
    # means_lr x0.25: at the default, scene_050's 30k round diverged from ~3k steps
    # (also with real frames only, also without vehicles); 4e-5 kept it stable.
    scene_050) ARGS=(++gaussian_splatting.strategy.cap_max=3000000 ++gaussian_splatting.means_lr=4e-5 "${RIGHT3[@]}" "$STEPS_7K") ;;
    scene_042) ARGS=(++gaussian_splatting.strategy.cap_max=3000000 "${LEFT3[@]}" "$STEPS_7K") ;;
    scene_041) ARGS=(++gaussian_splatting.strategy.cap_max=3000000 "$BILATERAL3") ;;
    scene_024) ARGS=(++gaussian_splatting.strategy.cap_max=3000000 "$BILATERAL3") ;;  # + SAM chunk, below
    scene_071) ARGS=(++gaussian_splatting.strategy.cap_max=2500000 "${LEFT3[@]}" "$STEPS_7K") ;;
    # --- Second batch (tiers 1-2). 3M background, default 1M rigid cap. ---
    scene_005) ARGS=(++gaussian_splatting.strategy.cap_max=3000000 "${LEFT3[@]}" "$STEPS_7K") ;;
    scene_018) ARGS=(++gaussian_splatting.strategy.cap_max=3000000 "$BILATERAL3" "$STEPS_7K") ;;
    scene_020) ARGS=(++gaussian_splatting.strategy.cap_max=3000000 "${LEFT3[@]}" "$STEPS_7K") ;;
    scene_048) ARGS=(++gaussian_splatting.strategy.cap_max=3000000 "${LEFT3[@]}" "$STEPS_7K") ;;
    scene_099) ARGS=(++gaussian_splatting.strategy.cap_max=3000000 "${LEFT3[@]}" "$STEPS_7K") ;;
    scene_096) ARGS=(++gaussian_splatting.strategy.cap_max=3000000 "${LEFT3[@]}" "$STEPS_7K") ;;
    scene_067) ARGS=(++gaussian_splatting.strategy.cap_max=3000000 "${RIGHT3[@]}" "$STEPS_7K") ;;
    scene_044) ARGS=(++gaussian_splatting.strategy.cap_max=3000000 "${LEFT3[@]}" "$STEPS_7K") ;;
    scene_077) ARGS=(++gaussian_splatting.strategy.cap_max=3000000 "${UP3[@]}") ;;   # 30k x 4
    scene_021) ARGS=(++gaussian_splatting.strategy.cap_max=3000000 "$BILATERAL3") ;; # 30k x 4
    *) return 1 ;;
  esac
}

SCENES=("$@")
# Fastest scenes last: 018 (10.1 m/s) and 048 (8.8 m/s) are above scene_050's
# 8.6 m/s, where default training collapsed in the 30k round.
[ ${#SCENES[@]} -eq 0 ] && SCENES=(scene_005 scene_020 scene_099 scene_096 scene_067 scene_044 scene_077 scene_021 scene_048 scene_018)

# SAM chunk sizes to try, in order; a later one is used only when SAM OOMs on
# the previous. Empty = the sam3.yaml default.
sam_chunks_for() {
  case "$1" in
    scene_024) CHUNKS=(5) ;;
    # Default chunk_size first, then smaller on a SAM OOM (024 needed 5).
    *) CHUNKS=("" 20 10 5) ;;
  esac
}

ok=0; fail=0; failed=()
for scene in "${SCENES[@]}"; do
  if ! args_for "$scene"; then
    echo "[FAIL] $scene -- no settings defined in this script"
    fail=$((fail + 1)); failed+=("$scene"); continue
  fi
  log="$LOGDIR/$scene.log"
  sam_chunks_for "$scene"
  for chunk in "${CHUNKS[@]}"; do
    SAM=(); [ -n "$chunk" ] && SAM=("++segmenter.chunk_size=$chunk")
    echo "[run ] $scene  ($(date '+%m-%d %H:%M:%S'))  ${ARGS[*]} ${SAM[*]}"
    start=$(wc -l < "$log" 2>/dev/null || echo 0)
    PYTHONPATH=. "$PYBIN" scripts/run_difix_4dgs.py \
        "dataset.scene=$scene" "${COMMON[@]}" "${ARGS[@]}" "${SAM[@]}" >> "$log" 2>&1
    rc=$?
    # Retry with the next chunk size only if THIS attempt died of a SAM OOM.
    [ $rc -ne 0 ] && tail -n +"$((start + 1))" "$log" | grep -aq "FATAL ERROR processing.*CUDA out of memory" \
      && { echo "[sam ] $scene  SAM OOM at chunk_size=$chunk"; continue; }
    break
  done
  if [ $rc -eq 0 ]; then
    echo "[ok  ] $scene  ($(date '+%m-%d %H:%M:%S'))"
    ok=$((ok + 1))
  else
    echo "[FAIL] $scene rc=$rc -- see $log"
    tail -n 15 "$log" | sed 's/^/         /'
    fail=$((fail + 1)); failed+=("$scene")
  fi
  # Let the driver release VRAM before the next scene starts.
  sleep 10
done

echo
echo "===== $ok ok, $fail failed ====="
[ $fail -gt 0 ] && printf 'failed: %s\n' "${failed[*]}"
exit 0
