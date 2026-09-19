#!/usr/bin/env bash
# Create the pipeline's virtual environments at the paths the scripts expect.
#
# Usage: scripts/setup_envs.sh [segmentation] [gsplat] [cc3dt]   (default: all three)
#
# Requires python3.12, git and the CUDA 12.8 toolkit (nvcc) on PATH. Existing envs
# are reused, so re-running the script only (re)installs the locked packages.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3.12}"

declare -A ENV_PATHS=(
    [segmentation]="envs/env_segmentation"
    [gsplat]="envs/envs/env_gsplat"
    [cc3dt]="envs/env_cc3dt"
)

command -v "$PYTHON" >/dev/null || { echo "error: $PYTHON not found" >&2; exit 1; }
command -v nvcc >/dev/null || { echo "error: nvcc not found (install the CUDA 12.8 toolkit)" >&2; exit 1; }

git submodule update --init --recursive

setup_env() {
    local name="$1" path="${ENV_PATHS[$1]}"
    echo "=== $name -> $path"

    [ -x "$path/bin/python" ] || "$PYTHON" -m venv "$path"

    # Every package is pinned in the lock, so --no-deps reproduces the env exactly
    # (the segmentation env knowingly runs numpy 2.x despite sam3 declaring numpy<2).
    "$path/bin/python" -m pip install --no-deps -r "requirements/$name.txt"

    # CUDA extensions build against the torch installed above, hence no build isolation.
    if [ -f "requirements/$name-cuda.txt" ]; then
        "$path/bin/python" -m pip install --no-deps --no-build-isolation -r "requirements/$name-cuda.txt"
    fi
}

targets=("$@")
[ ${#targets[@]} -gt 0 ] || targets=(segmentation gsplat cc3dt)

for name in "${targets[@]}"; do
    [ -n "${ENV_PATHS[$name]:-}" ] || { echo "error: unknown env '$name'" >&2; exit 1; }
    setup_env "$name"
done

echo "Done. Model weights are not included; see README.md."
