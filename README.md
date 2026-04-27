# Pix2Poly (Image-Only) – Reproducible Setup for the NY Subset

> **This is not a reimplementation.**
> This repository is a **reproducible setup and patch layer** built on top of the original [PixelsPointsPolygons (P3)](https://github.com/raphaelsulzer/PixelsPointsPolygons) codebase by Raphael Sulzer et al.
> It documents the minimal code fixes, dataset preparation steps, and configuration changes required to run **image-only Pix2Poly** training and inference on the **New York (NY) subset** of the P3 dataset, without any LiDAR or Open3D dependency.

---

## Table of Contents

1. [Scope](#scope)
2. [Motivation](#motivation)
3. [Tested Configuration](#tested-configuration)
4. [Path Convention](#path-convention)
5. [Quick Start](#quick-start)
6. [Installation](#installation)
7. [Required Fixes](#required-fixes)
8. [Dataset Preparation](#dataset-preparation)
9. [Pretrained Backbone](#pretrained-backbone)
10. [Configuration Notes](#configuration-notes)
11. [Training](#training)
12. [Training Optimizations](#training-optimizations)
13. [Monitoring & Metrics](#monitoring--metrics)
14. [Validation Predictions](#validation-predictions)
15. [Inference](#inference)
16. [Expected Results](#expected-results)
17. [Qualitative Results](#qualitative-results)
18. [Troubleshooting](#troubleshooting)
19. [Reproducibility Checklist](#reproducibility-checklist)
20. [Citation](#citation)
21. [Acknowledgements](#acknowledgements)

---

## Scope

- Modality: **image-only** (`p2p_image` experiment)
- Training scale: **full NY subset** (~43k training images), filtered from the complete P3 dataset
- Hardware: **single GPU**, SSH workstation (no sudo required)
- Excluded: LiDAR, Open3D, multimodal fusion

---

## Motivation

The original P3 implementation targets multimodal training (image + LiDAR) on a full multi-country dataset (~163 GB), which creates several barriers to image-only experimentation:

- **Broken imports** caused by unconditional LiDAR / Open3D module loading
- **Dataset inconsistencies** (missing images in some splits)
- **Hydra path issues** requiring absolute path configuration
- **Dependency conflicts** (transformers version)

This setup resolves all of the above and provides a clean, working baseline for image-only experiments on the NY subset.

---

## Tested Configuration

| Component | Version |
|-----------|---------|
| OS | Linux (Ubuntu 20.04, SSH workstation) |
| GPU | NVIDIA A100 / V100 (single GPU) |
| CUDA | 12.1 |
| Python | 3.11.11 |
| PyTorch | 2.2.2 |
| torchvision | 0.17.2 |
| transformers | 4.38.2 |
| Conda | any recent version |

> Other GPU models (RTX 3090, A6000) should work. CPU fallback is available but training will be prohibitively slow.

---

## Path Convention

This setup uses a `P3_ROOT` environment variable to avoid hardcoded paths. Set it once in your shell profile:

```bash
export P3_ROOT=/path/to/your/working/directory
```

All paths below use `$P3_ROOT` as the base. A typical layout is:

```
$P3_ROOT/
├── PixelsPointsPolygons/          # cloned repository
├── p3/                            # conda environment (prefix)
├── PixelsPointsPolygons_dataset/  # raw P3 dataset (downloaded separately)
├── p3_NY_full/                    # filtered NY-only dataset
│   └── data/224/
│       ├── images -> (symlink to dataset images)
│       └── annotations/blocks/
│           ├── annotations_NY_train.json
│           ├── annotations_NY_val.json
│           └── annotations_NY_test.json
├── PixelsPointsPolygons_output/   # checkpoints and logs
│   ├── backbones/
│   │   └── dino_deitsmall8_pretrain.pth
│   └── pix2poly/224/v4_image_vit_bs4x16/
└── train_full_ny.log
```

---

## Quick Start

These commands run a short smoke-test training (10 steps) to verify the installation is working before committing to a full run.

```bash
# 1. Set your working directory
export P3_ROOT=/path/to/your/working/directory

# 2. Clone, install, and apply fixes (see Installation and Required Fixes)

# 3. Run a minimal training pass
export CUDA_VISIBLE_DEVICES=0
cd $P3_ROOT/PixelsPointsPolygons

python scripts/train.py \
  experiment=p2p_image \
  experiment.model.num_epochs=1 \
  experiment.dataloader.max_samples=64
```

If this completes without errors, your environment is correctly set up.

---

## Installation

### 1. Clone the original repository

```bash
git clone https://github.com/raphaelsulzer/PixelsPointsPolygons
cd PixelsPointsPolygons
```

### 2. Create a Conda environment

> Install the environment on a disk with sufficient space (the environment alone takes ~5 GB).

```bash
conda create --prefix $P3_ROOT/p3 python=3.11.11 -y
conda activate $P3_ROOT/p3
```

### 3. Install dependencies

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
pip install -e .
pip install transformers==4.38.2
```

> `transformers==4.38.2` is pinned because newer versions break the tokenizer interface used by Pix2Poly.

---

## Required Fixes

The fixes below are necessary to run the image-only pipeline. The original code assumes LiDAR modules are always available; without these patches, the imports will fail at startup even if LiDAR is not used.

### Fix 1 — Disable LiDAR imports in PointPillars module

**File:** `models/pointpillars/__init__.py`

Comment out the Open3D-dependent import:

```python
# from .pointpillars_o3d import PointPillarsEncoder, PointPillars
```

**Why:** This import triggers `open3d`, which is not installed and not needed for image-only training. Commenting it out prevents an `ImportError` at startup.

---

### Fix 2 — Disable LiDAR imports in the Pix2Poly model

**File:** `models/pix2poly/model_pix2poly.py`

Replace the multimodal imports with image-only ones:

```python
# from ..pointpillars import PointPillarsViT        # remove: requires LiDAR
from ..vision_transformer import ViT, ViTDINOv2
# from ..fusion_layers import EarlyFusionViT         # remove: multimodal only
```

**Why:** `PointPillarsViT` and `EarlyFusionViT` are only used in multimodal configurations. Importing them unconditionally causes failures even when `experiment=p2p_image` is selected.

---

### Fix 3 — Patch `get_tile_names_from_dataloader` in `shared_utils.py`

**File:** `shared_utils.py` (or wherever `get_tile_names_from_dataloader` is defined)

Replace the original implementation with:

```python
def get_tile_names_from_dataloader(loader, ids):
    imgs_dict = loader.dataset.coco.imgs
    names = []

    for img_id in ids:
        img_info = imgs_dict.get(img_id)

        if img_info is None:
            names.append(f"unknown_{img_id}")
            continue

        name = img_info['file_name'].split('/')[-1].replace('.tif', '')
        names.append(name)
    return names
```

**Why:** The original function raises an `IndexError` when image IDs present in the dataloader are missing from the COCO index (which happens with filtered subsets). The patched version handles missing IDs gracefully with a placeholder name instead of crashing.

---

### Fix 4 — Use absolute paths in Hydra host config

**File:** `config/host/default.yaml`

```yaml
data_root: $P3_ROOT/p3_NY_full/data
model_root: $P3_ROOT/PixelsPointsPolygons_output
```

**Why:** Hydra's path interpolation does not reliably resolve relative paths across all launch contexts (e.g., tmux sessions, cron jobs). Absolute paths prevent silent misconfiguration.

---

## Dataset Preparation

The full P3 dataset covers multiple countries. This section extracts the NY-only subset used for training.

### Prerequisites

- The full P3 dataset downloaded at `$P3_ROOT/PixelsPointsPolygons_dataset/data/224/`
- Annotation files named `annotations_all_train.json`, `annotations_all_val.json`, `annotations_all_test.json` in `annotations/blocks/`

---

### Step 1 — Filter NY-only annotations

Create and run the following script **inside** `$P3_ROOT/PixelsPointsPolygons_dataset/data/224/annotations/blocks/`:

```python
# filter_ny_only.py
import json, os

for split in ['train', 'val', 'test']:
    with open(f'annotations_all_{split}.json') as f:
        data = json.load(f)

    # Keep only images whose file_name contains '/NY/'
    ny_imgs = [img for img in data['images'] if '/NY/' in img['file_name']]
    ny_ids  = {img['id'] for img in ny_imgs}
    ny_anns = [a for a in data['annotations'] if a['image_id'] in ny_ids]

    out = {'images': ny_imgs, 'annotations': ny_anns}
    with open(f'annotations_NY_{split}.json', 'w') as f:
        json.dump(out, f)

    print(f"{split}: {len(ny_imgs)} images, {len(ny_anns)} annotations")
```

Expected output:

```
train: 43,333 images, 142,556 annotations
val:     529 images,   2,352 annotations
test:  14,313 images,  51,757 annotations
```

---

### Step 2 — Create the `p3_NY_full` directory structure

```bash
mkdir -p $P3_ROOT/p3_NY_full/data/224/annotations/blocks

# Symlink images to avoid duplicating ~163 GB of data
ln -s $P3_ROOT/PixelsPointsPolygons_dataset/data/224/images \
      $P3_ROOT/p3_NY_full/data/224/images

# Copy the filtered annotation files
cp $P3_ROOT/PixelsPointsPolygons_dataset/data/224/annotations/blocks/annotations_NY_*.json \
   $P3_ROOT/p3_NY_full/data/224/annotations/blocks/
```

The resulting structure:

```
$P3_ROOT/p3_NY_full/
└── data/
    └── 224/
        ├── images -> (symlink to original images)
        └── annotations/
            └── blocks/
                ├── annotations_NY_train.json
                ├── annotations_NY_val.json
                └── annotations_NY_test.json
```

---

### Step 3 — Update Hydra dataset config

**File:** `config/dataset/p3.yaml`

```yaml
in_path: $P3_ROOT/p3_NY_full/data/224
annotations:
  train: $P3_ROOT/p3_NY_full/data/224/annotations/blocks/annotations_NY_train.json
  val:   $P3_ROOT/p3_NY_full/data/224/annotations/blocks/annotations_NY_val.json
  test:  $P3_ROOT/p3_NY_full/data/224/annotations/blocks/annotations_NY_test.json
```

---

## Pretrained Backbone

Download the DINOv2 small backbone used by Pix2Poly:

```bash
mkdir -p $P3_ROOT/PixelsPointsPolygons_output/backbones

wget -P $P3_ROOT/PixelsPointsPolygons_output/backbones \
  https://huggingface.co/rsi/PixelsPointsPolygons/resolve/main/backbones/dino_deitsmall8_pretrain.pth
```

Ensure `model_root` in `config/host/default.yaml` points to `$P3_ROOT/PixelsPointsPolygons_output` so Hydra can locate the checkpoint.

---

## Configuration Notes

### Timestamped output directories

By default, Hydra may overwrite the output directory on each run. To prevent this and keep all experiments isolated, add a timestamp to the run folder in `config/config.yaml`:

```yaml
hydra:
  run:
    dir: ${model_root}/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

Each training run then writes to a unique folder (e.g., `outputs/2024-11-01/14-32-07/`), making it safe to resume or compare runs without risking overwriting previous checkpoints or logs.

---

## Training

### Fresh start

```bash
export CUDA_VISIBLE_DEVICES=0
cd $P3_ROOT/PixelsPointsPolygons

python scripts/train.py experiment=p2p_image
```

### Long run with resume (recommended for full 400-epoch training)

```bash
export WANDB_MODE=offline

tmux new-session -d -s train_full_ny "bash -c '
  export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline &&
  cd $P3_ROOT/PixelsPointsPolygons &&
  $P3_ROOT/p3/bin/python scripts/train.py \
    experiment=p2p_image \
    checkpoint=latest \
    experiment.model.num_epochs=400 \
  2>&1 | tee $P3_ROOT/train_full_ny.log'"
```

> **Important:** Use the full Python path (`$P3_ROOT/p3/bin/python`) instead of `conda activate` inside tmux. Conda activation inside a non-interactive shell is unreliable and can cause the session to silently use the wrong interpreter.

Training produces:
- Checkpoints in a timestamped subfolder of `$P3_ROOT/PixelsPointsPolygons_output/outputs/`
- `metrics.csv` with per-epoch loss and IoU in the same output folder
- Live log at `$P3_ROOT/train_full_ny.log`

---

## Training Optimizations

The following modifications improve training speed and stability without changing the model architecture or affecting final accuracy. They are applied on top of the original training script and are each marked with `# === OPTIM:` in the code.

**Speed**

| Optimization | Description | Where applied |
|---|---|---|
| Mixed precision (AMP) | `GradScaler` + `autocast` to halve memory and speed up forward/backward | `scripts/train.py` |
| `torch.compile` | Compiles the model graph at first step for faster subsequent iterations | `scripts/train.py` |
| Loss accumulation | Accumulates gradients over N mini-batches to simulate a larger effective batch size | `scripts/train.py` |
| Pin memory | `pin_memory=True` in dataloaders for faster CPU→GPU transfers | dataloader config |
| Persistent workers | Keeps dataloader workers alive between epochs (`persistent_workers=True`) | dataloader config |
| Separate viz loader | Dedicated `train_viz_loader` with augmentations off for visualization (see below) | `scripts/train.py` |

**Stability**

| Optimization | Description | Where applied |
|---|---|---|
| Gradient clipping | Clips gradients to max norm 1.0, preventing divergence with long sequences | `scripts/train.py` |
| BCELoss fix | Clamps logits before BCE to avoid `log(0)` NaN loss | `scripts/train.py` |
| Memory cleanup | `torch.cuda.empty_cache()` + `gc.collect()` called after each epoch | `scripts/train.py` |
| `destroy_process_group` guard | Wrapped in try/except to prevent crash on single-GPU exit | `scripts/train.py` |
| wandb offline fallback | Catches wandb auth errors and automatically switches to offline mode | `scripts/train.py` |

### Visualization loader

During training, a separate `train_viz_loader` is used exclusively for generating prediction visualizations. This loader has **augmentations disabled** and **only includes samples that contain polygon annotations**, ensuring that visualizations are clean and meaningful rather than showing augmented or empty tiles. The main training loader is unaffected.

---

## Monitoring & Metrics

### Real-time monitoring

```bash
# GPU utilization
watch -n 1 nvidia-smi

# Live training log
tail -f $P3_ROOT/train_full_ny.log
```

### Metrics logging (`metrics.csv`)

Each epoch appends a row to `metrics.csv` in the run output directory. It records:
- `train_loss` — average training loss for the epoch
- `val_loss` — validation loss
- `val_iou` — polygon IoU on the validation set (computed every epoch)

`val_iou` is the primary metric for selecting the best checkpoint.

### Plotting training curves

```bash
python $P3_ROOT/plot_losses_auto.py
```

This script reads `metrics.csv` from the most recent timestamped run directory and saves `loss_curves.png` alongside it. The figure has two panels:
- Left: train and val loss over epochs
- Right: validation IoU over epochs

You can run this at any point during or after training to inspect progress.

---

## Validation Predictions

At the end of each 5 epochs, predicted building polygons for the validation set are saved in COCO JSON format:

```
outputs/<date>/<time>/
└── predictions_NY_val/
    ├── val_epoch_4.json
    ├── val_epoch_9.json
    ├── ...
    └── best_val_iou.json     # copy of the best-epoch predictions
```

Each file contains the model's polygon predictions in COCO format and can be used for:
- Computing additional metrics offline (e.g., with `pycocotools`)
- Visualizing specific validation samples
- Submitting to an evaluation server

`best_val_iou.json` is automatically updated whenever a new best validation IoU is achieved.

---

## Inference

Run inference on a single image using the best validation checkpoint:

```bash
export CUDA_VISIBLE_DEVICES=0
cd $P3_ROOT/PixelsPointsPolygons

python scripts/predict_demo.py \
  checkpoint=best_val_iou \
  experiment=p2p_image \
  host.device=cuda \
  +image_file=demo_data/image0_CH_val.tif
```

**What this produces:**
- A visualization image saved as `prediction_pix2poly_image.png` in the current directory (or the configured output directory)
- Predicted building polygons overlaid on the input image

> To run on your own image, replace `demo_data/image0_CH_val.tif` with the path to a `.tif` file in the same format as the P3 dataset (224×224 px, RGB).

---

## Expected Results

> These are approximate reference values. Exact numbers will vary by GPU, random seed, and dataset version.

| Metric | Approximate Value |
|--------|------------------|
| Training time (400 epochs, A100) | ~48–72 hours |
| Training time (400 epochs, V100) | ~72–96 hours |
| Best validation IoU | ~0.55–0.65 |
| GPU memory usage | ~18–24 GB |

Results on other country subsets (CH, NZ) are expected to be similar but have not been systematically evaluated with this setup.

---

## Qualitative Results

### Example Predictions (Pix2Poly – Image Only)

| Input Image | Predicted Polygons |
|-------------|-------------------|
| ![Input example](media/input_example.png) | ![Prediction example](media/prediction_example.png) |

> Replace the placeholder images above with actual outputs from your run (`prediction_pix2poly_image.png`). Outputs should show predicted building footprint polygons overlaid on the input aerial image.

---

## Troubleshooting

### Philosophy

Most failures in this setup come from one of three root causes:

1. **Wrong paths** — Hydra silently loads a default config instead of your intended one. Always verify `data_root` and `model_root` are resolved correctly (add a `print(cfg)` at the top of the train script if unsure).
2. **Missing or mismatched dependencies** — especially `transformers`. Always install the pinned version (`4.38.2`) after `pip install -e .`.
3. **Dataset mismatch** — the annotation JSON references image paths that do not exist in your symlinked directory. Verify paths with a quick sanity check:

```python
import json
with open('annotations_NY_train.json') as f:
    data = json.load(f)
print(data['images'][0]['file_name'])  # should match your images/ directory
```

### Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ImportError: open3d` | LiDAR imports not disabled | Apply Fix 1 and Fix 2 above |
| Hydra path errors | Relative paths in config | Use absolute paths (Fix 4) |
| `transformers` API error | Wrong transformers version | `pip install transformers==4.38.2` |
| `IndexError` in utils | `get_tile_names_from_dataloader` crashes on missing IDs | Apply Fix 3 above |
| tmux session dies silently | `conda activate` fails in non-interactive shell | Use full Python path (see Training section) |
| wandb authentication failure | wandb tries to connect without credentials | `export WANDB_MODE=offline` |

---

## Reproducibility Checklist

### Environment
- [ ] Python 3.11.11
- [ ] PyTorch 2.2.2, torchvision 0.17.2
- [ ] `transformers==4.38.2`
- [ ] CUDA available and detected by PyTorch

### Setup
- [ ] Original repository cloned
- [ ] Conda environment created on a disk with sufficient space
- [ ] Project installed with `pip install -e .`

### Code Fixes
- [ ] LiDAR import in `models/pointpillars/__init__.py` commented out
- [ ] LiDAR imports in `models/pix2poly/model_pix2poly.py` commented out
- [ ] `get_tile_names_from_dataloader` patched in `shared_utils.py`

### Dataset
- [ ] Full P3 dataset available
- [ ] NY-only annotations filtered using `filter_ny_only.py`
- [ ] `p3_NY_full/` directory created with symlink to images
- [ ] Filtered JSONs copied into `p3_NY_full/data/224/annotations/blocks/`

### Configuration
- [ ] `config/host/default.yaml` uses absolute paths
- [ ] `config/dataset/p3.yaml` points to `p3_NY_full` and NY annotation files
- [ ] Backbone checkpoint path is valid
- [ ] Timestamped output directories configured in `config/config.yaml`

### Training
- [ ] Quick Start smoke test passes (1 epoch, 64 samples)
- [ ] Full training runs without crash
- [ ] `metrics.csv` is being written to the timestamped output directory
- [ ] `best_val_iou.json` is updated after each best epoch

### Inference
- [ ] `predict_demo.py` runs on a sample image
- [ ] Output image (`prediction_pix2poly_image.png`) is generated
- [ ] `plot_losses_auto.py` generates `loss_curves.png` from `metrics.csv`

---

## Citation

If you use this setup in your work, please cite the original P3 paper:

```bibtex
@misc{sulzer2025p3datasetpixelspoints,
  title={The P$^3$ Dataset: Pixels, Points and Polygons},
  author={Raphael Sulzer et al.},
  year={2025}
}
```

---

## Acknowledgements

All credit for the original model, dataset, and codebase belongs to the P3 authors:

[https://github.com/raphaelsulzer/PixelsPointsPolygons](https://github.com/raphaelsulzer/PixelsPointsPolygons)

This repository only provides a reproducible configuration layer on top of their work.
