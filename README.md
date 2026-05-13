# Pix2Poly (Image-Only) – NY Subset Setup

This repository is a patch layer on top of the original
[PixelsPointsPolygons (P3)](https://github.com/raphaelsulzer/PixelsPointsPolygons)
codebase. It is **not a reimplementation** — the model, trainer, and backbone all come
from Raphael Sulzer et al. What this adds is: the minimal fixes to make image-only
training work without LiDAR, the NY subset filtering, and the training optimisations
applied to `scripts/train.py`.

---

## Why this isn't plug-and-play

The original P3 code targets multimodal training (image + LiDAR) on a full multi-country
dataset (~163 GB). Running image-only on a subset runs into several concrete problems:

| Problem | What breaks | Fix |
|---------|-------------|-----|
| LiDAR imports at module load time | `ImportError: open3d` even when `experiment=p2p_image` | Comment out in `pointpillars/__init__.py` and `model_pix2poly.py` |
| `get_tile_names_from_dataloader` assumes all IDs exist in COCO index | `IndexError` on filtered subsets where some IDs are missing | Patched to return `unknown_<id>` instead of crashing |
| Hydra relative paths | Silent misconfiguration depending on launch context (tmux, cron) | Absolute paths in `config/host/default.yaml` |
| `transformers` version drift | Tokenizer interface changed in newer releases | Pin to `4.38.2` |

---

## Tested on

| | |
|---|---|
| OS | Ubuntu 20.04, SSH workstation |
| GPU | NVIDIA RTX 3090 (single GPU, out of 4 available) |
| CUDA | 12.8 |
| Python | 3.11.11 |
| PyTorch | 2.2.2 + torchvision 0.17.2 |
| transformers | 4.38.2 |

---

## Directory layout

Set `P3_ROOT` once in your shell profile and use it everywhere:

```bash
export P3_ROOT=/path/to/your/working/directory
```

```
$P3_ROOT/
├── PixelsPointsPolygons/          # cloned repo
├── p3/                            # conda env (prefix)
├── PixelsPointsPolygons_dataset/  # raw P3 dataset
├── p3_NY_full/                    # filtered NY-only dataset
│   └── data/224/
│       ├── images -> (symlink)
│       └── annotations/blocks/
│           ├── annotations_NY_train.json
│           ├── annotations_NY_val.json
│           └── annotations_NY_test.json
├── PixelsPointsPolygons_output/
│   ├── backbones/
│   │   └── dino_deitsmall8_pretrain.pth
│   └── pix2poly/224/v4_image_vit_bs4x16/
└── train_full_ny.log
```

---

## Installation

```bash
git clone https://github.com/raphaelsulzer/PixelsPointsPolygons
cd PixelsPointsPolygons

# Conda env as a prefix (easier to manage disk location)
conda create --prefix $P3_ROOT/p3 python=3.11.11 -y
conda activate $P3_ROOT/p3

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
pip install -e .
pip install transformers==4.38.2
```

`transformers==4.38.2` must be installed *after* `pip install -e .` — the latter can
pull in a newer version that breaks the tokenizer interface.

The conda env is ~5 GB. Make sure `$P3_ROOT` has enough space.

---

## Required fixes

All four changes below are necessary. The original code will crash at import time without
fixes 1–2, and at runtime without fixes 3–4.

### Fix 1 — `models/pointpillars/__init__.py`

Comment out:

```python
# from .pointpillars_o3d import PointPillarsEncoder, PointPillars
```

This file is imported unconditionally at startup. `pointpillars_o3d` depends on
`open3d`, which isn't installed and isn't needed here.

### Fix 2 — `models/pix2poly/model_pix2poly.py`

```python
# from ..pointpillars import PointPillarsViT        # multimodal only
from ..vision_transformer import ViT, ViTDINOv2
# from ..fusion_layers import EarlyFusionViT         # multimodal only
```

Same issue: these imports run regardless of which experiment config is selected.

### Fix 3 — `shared_utils.py` — patch `get_tile_names_from_dataloader`

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

The original uses direct index lookup and raises `IndexError` on image IDs that are in
the dataloader but not in the COCO index — which happens whenever you train on a filtered
subset.

### Fix 4 — `config/host/default.yaml`

```yaml
data_root: /absolute/path/to/p3_NY_full/data
model_root: /absolute/path/to/PixelsPointsPolygons_output
```

Hydra silently falls back to its own working directory when relative paths don't resolve.
If training starts but reads from the wrong place, this is the first thing to check.

---

## Dataset preparation

The full P3 dataset includes multiple countries. This filters down to NY only.

**Prerequisites:** full P3 dataset at `$P3_ROOT/PixelsPointsPolygons_dataset/data/224/`,
with `annotations_all_{train,val,test}.json` in `annotations/blocks/`.

### Step 1 — filter NY annotations

Run this from inside `$P3_ROOT/PixelsPointsPolygons_dataset/data/224/annotations/blocks/`:

```python
# filter_ny_only.py
import json
import os

OUTPUT_DIR = "/mnt/DATA/IMANE/p3_NY_full/data/224/annotations/blocks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for split in ['train', 'val', 'test']:
    in_file = f'annotations_all_{split}.json'
    out_file = os.path.join(OUTPUT_DIR, f'annotations_NY_{split}.json')
    print(f"Processing {split}...")
    with open(in_file, 'r') as f:
        data = json.load(f)
    ny_images = [img for img in data['images'] if '/NY/' in img['file_name']]
    ny_ids = {img['id'] for img in ny_images}
    ny_annotations = [ann for ann in data['annotations'] if ann['image_id'] in ny_ids]
    data['images'] = ny_images
    data['annotations'] = ny_annotations
    with open(out_file, 'w') as f:
        json.dump(data, f)
    print(f"  -> {len(ny_images)} NY images, {len(ny_annotations)} annotations")
```

Expected output:

```
train: 43,333 images, 142,556 annotations
val:     529 images,   2,352 annotations
test:  14,313 images,  51,757 annotations
```

### Step 2 — build `p3_NY_full`

```bash
mkdir -p $P3_ROOT/p3_NY_full/data/224/annotations/blocks

# symlink instead of copying ~163 GB
ln -s $P3_ROOT/PixelsPointsPolygons_dataset/data/224/images \
      $P3_ROOT/p3_NY_full/data/224/images
```

The filter script above writes the JSON files directly to `p3_NY_full`. If you ran it
with a different output path, copy them over:

```bash
cp /wherever/annotations_NY_*.json \
   $P3_ROOT/p3_NY_full/data/224/annotations/blocks/
```

### Step 3 — point Hydra at the NY dataset

`config/dataset/p3.yaml`:

```yaml
in_path: $P3_ROOT/p3_NY_full/data/224
annotations:
  train: $P3_ROOT/p3_NY_full/data/224/annotations/blocks/annotations_NY_train.json
  val:   $P3_ROOT/p3_NY_full/data/224/annotations/blocks/annotations_NY_val.json
  test:  $P3_ROOT/p3_NY_full/data/224/annotations/blocks/annotations_NY_test.json
```

---

## Pretrained backbone

```bash
mkdir -p $P3_ROOT/PixelsPointsPolygons_output/backbones

wget -P $P3_ROOT/PixelsPointsPolygons_output/backbones \
  https://huggingface.co/rsi/PixelsPointsPolygons/resolve/main/backbones/dino_deitsmall8_pretrain.pth
```

`model_root` in `config/host/default.yaml` must point to
`$P3_ROOT/PixelsPointsPolygons_output` for Hydra to find it.

---

## One thing to fix in Hydra config

By default Hydra overwrites the output directory on each run. Add a timestamp so
experiments don't clobber each other — `config/config.yaml`:

```yaml
hydra:
  run:
    dir: ${model_root}/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

---

## Smoke test

Before starting a full run, verify the environment with 1 epoch / 64 samples:

```bash
export CUDA_VISIBLE_DEVICES=0
cd $P3_ROOT/PixelsPointsPolygons

python scripts/train.py \
  experiment=p2p_image \
  experiment.model.num_epochs=1 \
  experiment.dataloader.max_samples=64
```

---

## Training

### Short run / quick iteration

```bash
export CUDA_VISIBLE_DEVICES=0
cd $P3_ROOT/PixelsPointsPolygons

python scripts/train.py experiment=p2p_image
```

### Full run in tmux (recommended)

```bash
tmux new-session -d -s train_full_ny "bash -c '
  export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline &&
  cd $P3_ROOT/PixelsPointsPolygons &&
  $P3_ROOT/p3/bin/python scripts/train.py \
    experiment=p2p_image \
    checkpoint=latest \
    experiment.model.num_epochs=400 \
  2>&1 | tee $P3_ROOT/train_full_ny.log'"
```

Use the full Python path (`$P3_ROOT/p3/bin/python`) rather than `conda activate` inside
tmux. Conda activation in non-interactive shells is unreliable and can silently use the
wrong interpreter with no error.

Outputs:
- Checkpoints in a timestamped subfolder under `PixelsPointsPolygons_output/outputs/`
- `metrics.csv` with per-epoch train/val loss and IoU
- Live log at `$P3_ROOT/train_full_ny.log`

---

## Training optimisations

These are applied on top of the original `scripts/train.py` and tagged `# === OPTIM:`
in the code. None of them change the model architecture or affect final accuracy.

**Speed**

| | |
|---|---|
| Mixed precision (AMP) | `GradScaler` + `autocast` — roughly halves memory, faster forward/backward |
| `torch.compile` | Compiles the graph at first step; subsequent steps are faster |
| Gradient accumulation | Simulates a larger effective batch size without extra memory |
| `pin_memory=True` | Faster CPU→GPU transfers in the dataloader |
| `persistent_workers=True` | Workers stay alive between epochs |

**Stability**

| | |
|---|---|
| Gradient clipping (norm 1.0) | The autoregressive decoder generates long sequences; without clipping, loss can diverge early |
| BCELoss logit clamping | Avoids `log(0)` NaN loss |
| `empty_cache()` + `gc.collect()` per epoch | Prevents slow memory accumulation over long runs |
| `destroy_process_group` guard | Single-GPU runs don't have a process group; the original code crashes on exit without this |
| wandb offline fallback | Catches auth errors and switches to offline mode automatically |

**Visualisation loader**

A separate `train_viz_loader` is used for per-epoch prediction plots. It has
augmentations off and skips empty tiles (no polygon annotations). The main training
loader is untouched. Without this, visualisations show flipped/cropped versions of tiles
that are hard to interpret.

---

## Monitoring

```bash
watch -n 1 nvidia-smi
tail -f $P3_ROOT/train_full_ny.log
```

`metrics.csv` in the run output directory gets a row per epoch:
`train_loss`, `val_loss`, `val_iou`. `val_iou` is used for best-checkpoint selection.

```bash
python $P3_ROOT/plot_losses_auto.py
```

Reads `metrics.csv` from the most recent timestamped run and saves `loss_curves.png`
next to it — two panels, loss and IoU over epochs. Works mid-run.

---

## Validation predictions

Every 5 epochs, polygon predictions for the val set are saved as COCO JSON:

```
outputs/<date>/<time>/predictions_NY_val/
├── val_epoch_4.json
├── val_epoch_9.json
├── ...
└── best_val_iou.json
```

`best_val_iou.json` is overwritten whenever a new best `val_iou` is reached. Useful for
offline evaluation with `pycocotools` or visualising specific samples without rerunning
inference.

---

## Inference

```bash
export CUDA_VISIBLE_DEVICES=0
cd $P3_ROOT/PixelsPointsPolygons

python scripts/predict_demo.py \
  checkpoint=best_val_iou \
  experiment=p2p_image \
  host.device=cuda \
  +image_file=demo_data/image0_NY_val.tif
```

Saves `prediction_pix2poly_image.png` with predicted building polygons overlaid. Input
must be a 224×224 px RGB `.tif` in the same format as the P3 dataset.

---

## Results

Single RTX 3090, stopped at epoch 160 (configured for 200) once val IoU had clearly
converged:

| | |
|---|---|
| Best val IoU | **0.831** (epoch 139) |
| Val IoU at stop (epoch 159) | 0.815 |
| Train loss at stop | ~1.53 |

Val IoU crossed 0.80 around epoch 79 and stayed in the 0.81–0.83 range from epoch 84
onward. No meaningful improvement after epoch 139, so I stopped early.

I haven't evaluated on other country subsets (CH, NZ) with this setup.

---

## Qualitative results

Epoch 159 on the NY val set. Left: ground truth. Right: predictions.

![Val prediction example](media/val_prediction_example.png)

---

## Troubleshooting

Most failures come from one of three things: wrong paths (Hydra silently uses defaults),
wrong `transformers` version (install order matters), or the annotation JSON referencing
paths that don't exist under your symlink. Quick sanity check for the last one:

```python
import json
with open('annotations_NY_train.json') as f:
    data = json.load(f)
print(data['images'][0]['file_name'])  # should match your images/ directory
```

**`ImportError: open3d`** — Fixes 1 and 2 not applied.

**Hydra path errors / training reads wrong data** — Relative paths in
`config/host/default.yaml`. Use absolute paths.

**`transformers` API error** — `pip install -e .` pulled in a newer version. Run
`pip install transformers==4.38.2` again after.

**`IndexError` in `shared_utils.py`** — Fix 3 not applied.

**tmux session exits silently** — `conda activate` failed in the non-interactive shell.
Use the full Python path.

**wandb auth error** — `export WANDB_MODE=offline`.

---

## Citation

```bibtex
@misc{sulzer2025p3datasetpixelspoints,
  title={The P$^3$ Dataset: Pixels, Points and Polygons},
  author={Raphael Sulzer et al.},
  year={2025}
}
```

Original codebase: [raphaelsulzer/PixelsPointsPolygons](https://github.com/raphaelsulzer/PixelsPointsPolygons)
