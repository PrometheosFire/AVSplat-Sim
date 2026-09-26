# AVSplat-Sim

## Setup

### System requirements

- Linux (developed on Ubuntu 24.04)
- Python 3.12 with the `venv` module and development headers (`Python.h`, needed to compile the CUDA extensions). On Ubuntu 24.04 these are not installed by default: `sudo apt install python3.12-venv python3.12-dev`
- NVIDIA GPU and driver supporting CUDA 12.8
- CUDA 12.8 toolkit (`nvcc` on `PATH`) and a matching host compiler (developed with gcc 14), used to compile gsplat and the other CUDA extensions
- An SSH key registered on GitHub (the gsplat and ncore submodules are cloned over SSH)

### 1. Clone with submodules

```bash
git clone --recursive git@github.com:PrometheosFire/AVSplat-Sim.git
cd AVSplat-Sim
```

For an existing clone: `git submodule sync && git submodule update --init --recursive`.

| Submodule | Source | Notes |
|---|---|---|
| `external/gsplat` | `PrometheosFire/gsplat` (fork) | Installed editable in the gsplat env |
| `external/ncore` | `PrometheosFire/ncore`, branch `avsplat` | Patched COLMAP converter, run from the submodule (see below) |
| `external/sam3` | `facebookresearch/sam3` | Installed editable in the segmentation env |
| `external/Difix3D+` | `nv-tlabs/Difix3D` | Imported from `external/Difix3D+/src` by the Difix wrapper |

### 2. Create the Python environments

The pipeline uses three separate virtual environments. The orchestration scripts call each one by path, so they must live exactly here:

| Env | Path | Used for |
|---|---|---|
| segmentation | `envs/env_segmentation` | SAM3 mask extraction, Difix post-processing |
| gsplat | `envs/envs/env_gsplat` | NCore conversion, Gaussian splatting training and rendering, Difix loop |
| cc3dt | `envs/env_cc3dt` | CC-3DT tracking, bicycle-model track refinement |

```bash
scripts/setup_envs.sh                # all three
scripts/setup_envs.sh gsplat cc3dt   # or a subset
```

Each env is installed from a frozen lock in `requirements/` (`<env>.txt`). CUDA extensions are then compiled against the installed torch from `<env>-cuda.txt`: gsplat, fused-ssim, fused-bilagrid and ppisp in the gsplat env, and vis4d_cuda_ops in the cc3dt env. Compilation targets the GPU visible at build time; set `TORCH_CUDA_ARCH_LIST` to build for another architecture, and `MAX_JOBS` to limit memory use while compiling.

Notes on the pinned versions:

- **Different torch versions**: the segmentation env runs torch 2.10 + numpy 2.5; the other two run torch 2.11 + numpy 1.26. The envs cannot simply be merged (vis4d needs `numpy<2` and `pydantic<2`; the gsplat and cc3dt code needs different `pycolmap` packages).
- **numpy in the segmentation env**: sam3 declares `numpy<2`, but the env works with numpy 2.5.1. This is why the locks are installed with `--no-deps`.
- **ncore**: the `ncore` library comes from PyPI (`nvidia-ncore==18.6.0` in the gsplat env), while the COLMAP converter (`tools.data_converter.colmap.converter`) is run from the `external/ncore` submodule, which carries the capture-order and mask-path fixes.

To refresh a lock after changing an env: `envs/<path>/bin/python -m pip freeze --all`, then drop `pip` and move editable and CUDA-extension lines into the matching files in `requirements/`.

### 3. Download model weights

Weights go in `external_weights/` (gitignored).

**SAM3** is gated and is **not** downloaded automatically: `configs/model/sam3.yaml` loads `external_weights/sam3/sam3.pt`, so it must exist before the first run.

1. Request access at https://huggingface.co/facebook/sam3.
2. Create a Read token at https://huggingface.co/settings/tokens and log in from the terminal. Being logged in on the website is not enough; the CLI needs its own token.
3. Download the weights:

```bash
envs/env_segmentation/bin/huggingface-cli login
envs/env_segmentation/bin/huggingface-cli whoami    # should print your username
envs/env_segmentation/bin/huggingface-cli download facebook/sam3 sam3.pt model.safetensors --local-dir external_weights/sam3
```

**CC-3DT** tracking models (r50 is the default; r101 is used when the tracker backbone is `r101`). The vis4d model-zoo host these were originally downloaded from (`dl.cv.ethz.ch`) no longer resolves (checked 2026-09-20), so they are fetched from the public Hugging Face repo [RoyYang0714/cc-3dt](https://huggingface.co/RoyYang0714/cc-3dt) instead. No login is needed:

```bash
mkdir -p external_weights/cc3dt
for f in cc_3dt_frcnn_r50_fpn_12e_nusc_d98509.pt cc_3dt_frcnn_r101_fpn_24e_nusc_f24f84.pt; do
    wget -P external_weights/cc3dt "https://huggingface.co/RoyYang0714/cc-3dt/resolve/main/$f"
done
sha256sum external_weights/cc3dt/*.pt
```

Expected checksums (the first six characters also match the suffix in each filename):

```
f24f844d436d1cb9dc17e37ce79f17a14eeb2b3c1792be148d5ca15b6c54243f  cc_3dt_frcnn_r101_fpn_24e_nusc_f24f84.pt
d9850985cc7f6981e352c40af38c958ac263ef44ce068c54aede7c323114a650  cc_3dt_frcnn_r50_fpn_12e_nusc_d98509.pt
```

**Difix** (`nvidia/difix_ref`) is downloaded automatically from Hugging Face on first use.

### 4. Data

Datasets live under `data/` (gitignored). Only `wayve101` and `3DRealCar` are covered here; `data/nuscenes` is not.

#### Wayve Scenes 101 -> `data/wayve101`

The dataset is distributed as split zip parts (`WayveScenes101-<timestamp>-1-NNN.zip`). Each part contains per-scene zips (`WayveScenes101/scene_NNN.zip`), so it has to be unzipped twice. Put the parts in `data/`, then extract one part at a time so the intermediate copies stay small:

```bash
tmp=$(mktemp -d)
mkdir -p data/wayve101
for part in data/WayveScenes101-*.zip; do
    unzip -oq "$part" -d "$tmp"
    for scene in "$tmp"/WayveScenes101/scene_*.zip; do
        unzip -oq "$scene" -d data/wayve101 && rm "$scene"
    done
done
rm -r "$tmp"
```

Result (about 49 GB for 101 scenes; the pipeline reads `data/wayve101/<scene>`, see `configs/dataset/wayve101.yaml`):

```
data/wayve101/
├── scene_001 ... scene_101
│   ├── colmap_sparse/rig/     cameras.bin, images.bin, points3D.bin
│   ├── images/<5 cameras>/
│   └── masks/<5 cameras>/
└── dataset_info/              scene_metadata.csv, baselines.json
```

#### 3DRealCar vehicle assets -> `data/3DRealCar`

The simulator needs the reconstructed 3DGS vehicles, not the raw 3DRealCar capture data. These come from the [HUGSIM](https://huggingface.co/datasets/XDimLab/HUGSIM) dataset (public, MIT). Download only its `3DRealCar/` folder (110 vehicles, about 19 GB; the rest of the repo is about 41 GB and is not needed):

```bash
envs/env_segmentation/bin/huggingface-cli download XDimLab/HUGSIM \
    --repo-type dataset --include "3DRealCar/*" --local-dir data
```

This produces the layout expected by `library_dir: data/3DRealCar` in `configs/rendering/simulator.yaml`:

```
data/3DRealCar/<timestamp>/
├── gs.pth      reconstructed Gaussians (torch pickle)
└── wlh.json    [width, length, height] in metres
```

The command also creates a small `data/.cache/huggingface/` folder with download bookkeeping, which is safe to ignore (it lets an interrupted download resume).
