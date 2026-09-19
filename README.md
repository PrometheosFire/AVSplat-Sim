# AVSplat-Sim

## Setup

### System requirements

- Linux (developed on Ubuntu 24.04)
- Python 3.12 (`python3.12` with the `venv` module)
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

Weights go in `external_weights/` (gitignored):

```bash
# SAM3 (gated: request access at https://huggingface.co/facebook/sam3 first,
# then `envs/env_segmentation/bin/huggingface-cli login`)
envs/env_segmentation/bin/huggingface-cli download facebook/sam3 sam3.pt model.safetensors --local-dir external_weights/sam3

# CC-3DT tracking models (vis4d model zoo)
mkdir -p external_weights/cc3dt
wget -P external_weights/cc3dt \
    https://dl.cv.ethz.ch/vis4d/cc_3dt/cc_3dt_frcnn_r50_fpn_12e_nusc_d98509.pt \
    https://dl.cv.ethz.ch/vis4d/cc_3dt/cc_3dt_frcnn_r101_fpn_24e_nusc_f24f84.pt
```

Difix (`nvidia/difix_ref`) is downloaded automatically from Hugging Face on first use.

### 4. Data

Datasets live under `data/` (gitignored), e.g. `data/wayve101`, `data/nuscenes`, `data/3DRealCar`.
