#!/usr/bin/env bash
# Rigid reset x prune x refine study on scene_084 at 30k steps (TODO item 1).
#
# ONE PROCESS PER CELL, deliberately, instead of a single `python -m` multirun.
# The first attempt used multirun and lost all seven cells to one failure:
# Hydra's basic launcher runs every job in the same Python process, so when the
# reset=0 cell hit a CUDA OOM at step 22100 the allocator was left unusable and
# the remaining six died progressively faster (46m, 4m45, 3m18, 1m20, 1s, 1s,
# 1s). Separate processes make each cell independent -- one OOM costs one cell.
#
# Other things this fixes:
#   * stdout is captured per cell, so RigidDensifier's per-tick grow/prune counts
#     survive. It reports via print(), so those lines never reach
#     train_splats.log and are lost if stdout is a terminal.
#   * the reset=0 cell runs LAST, since it is the memory-hungry one (it is the
#     only cell that actually spends its rigid budget). Six cells are banked
#     before it is attempted.
#   * completed cells are skipped, so the script is re-runnable after a failure
#     and can adopt a cell trained elsewhere.
#
# Usage:   bash scripts/experiments/run_rigid_reset_study.sh
# Collect: envs/envs/env_gsplat/bin/python scripts/experiments/collect_ablation.py \
#            multirun/rigid_reset_study

set -u
cd "$(dirname "$0")/../.." || exit 1

SWEEP=multirun/rigid_reset_study
PYBIN=envs/envs/env_gsplat/bin/python

# reset=0 last: it is the only cell that uses its rigid budget, so it is the
# only one at risk of OOM, and a failure there must not cost the others.
CELLS=(
  abl_rigid_r3000_p020_f100
  abl_rigid_r3000_p005_f100
  abl_rigid_r3000_p010_f100
  abl_rigid_r3000_p050_f100
  abl_rigid_r3000_p020_f300
  abl_rigid_r3000_p005_f300
  abl_rigid_r0_p020_f100
)

mkdir -p "$SWEEP"
ok=0; skip=0; fail=0; failed=()

for cell in "${CELLS[@]}"; do
  out="$SWEEP/$cell"
  # A cell counts as done when it has val stats; the checkpoint alone is not
  # enough, since that is what the OOMed run left behind.
  if compgen -G "$out/stats/val_step*.json" > /dev/null; then
    echo "[skip] $cell -- already has val stats"
    skip=$((skip + 1))
    continue
  fi
  echo "[run ] $cell  ($(date +%H:%M:%S))"
  PYTHONPATH=. "$PYBIN" src/gsplat_training/train_splats.py -m \
      "+experiment=$cell" \
      "hydra.sweep.dir=$SWEEP" 'hydra.sweep.subdir=${exp_name}' \
      > "$SWEEP/$cell.stdout.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && compgen -G "$out/stats/val_step*.json" > /dev/null; then
    echo "[ok  ] $cell  ($(date +%H:%M:%S))"
    ok=$((ok + 1))
  else
    echo "[FAIL] $cell rc=$rc -- see $SWEEP/$cell.stdout.log"
    tail -n 15 "$SWEEP/$cell.stdout.log" | sed 's/^/         /'
    fail=$((fail + 1)); failed+=("$cell")
  fi
  # Let the driver release VRAM before the next cell starts.
  sleep 10
done

echo
echo "===== $ok ok, $skip skipped, $fail failed ====="
[ $fail -gt 0 ] && printf 'failed: %s\n' "${failed[*]}"
echo "collect with:"
echo "  $PYBIN scripts/experiments/collect_ablation.py $SWEEP"
exit 0
