# Pix2Poly (Image-Only) on NY Dataset – Reproducible Setup

This repository provides a minimal, reproducible pipeline to train and run inference with **Pix2Poly (image modality only)** from the P3 framework.

👉 Original repository: https://github.com/raphaelsulzer/PixelsPointsPolygons

---

## Scope

- Image-only Pix2Poly (`p2p_image`)
- Single-GPU training
- **Full NY subset** (~43k images) – filtered from the complete P3 dataset
- No LiDAR / no Open3D dependency
- GPU inference (SSH workstation)

---

## Motivation

The original implementation targets multimodal training (image + LiDAR) on a full dataset (~163GB), which introduces:

- Broken imports (LiDAR dependencies)
- Dataset inconsistencies (missing images)
- Hydra path issues
- Dependency conflicts

This repo provides a clean, minimal setup that actually works for image-only experiments on the **complete NY subset**.

---

## Qualitative Results

### Example Predictions (Pix2Poly – Image Only)

| Input Image | Predicted Polygons |
|------------|------------------|
| ![](media/input_example.png) | ![](media/prediction_example.png) |

> Replace with your own outputs from `prediction_pix2poly_image.png`

---

## Reproducibility Checklist

To fully reproduce results:

### Environment
- [ ] Python 3.11
- [ ] PyTorch 2.2.2
- [ ] transformers==4.38.2
- [ ] CUDA available (or CPU fallback)

### Setup
- [ ] Repository cloned
- [ ] Conda environment created on large disk
- [ ] Project installed with `pip install -e .`

### Code Fixes
- [ ] LiDAR imports removed
- [ ] No Open3D dependency
- [ ] `shared_utils.py` patched

### Dataset (Full NY)
- [ ] Full P3 dataset available at `/mnt/DATA/IMANE/PixelsPointsPolygons_dataset/data/224/`
- [ ] NY‑only annotations filtered using `filter_ny_only.py`
- [ ] `p3_NY_full` directory with symlink to images
- [ ] Configs point to `p3_NY_full`

### Config
- [ ] Absolute paths set (Hydra)
- [ ] Backbone checkpoint path valid
- [ ] No broken logger interpolation

### Training
- [ ] Runs without crash (tmux, wandb offline, OOM avoided)
- [ ] Visualization bug fixed
- [ ] CSV logging enabled (`metrics.csv`)

### Inference
- [ ] GPU inference works
- [ ] Output image generated

---

## Environment

- Linux (remote workstation via SSH)
- No sudo access
- Single GPU used

---

## Installation

```bash
git clone https://github.com/raphaelsulzer/PixelsPointsPolygons
cd PixelsPointsPolygons
```

```bash
conda create --prefix /mnt/DATA/IMANE/p3 python=3.11.11 -y
conda activate /mnt/DATA/IMANE/p3
```

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
pip install -e .
pip install transformers==4.38.2
```

---

##  Required Fixes

### Disable LiDAR modules

`models/pointpillars/__init__.py`

```python
# from .pointpillars_o3d import PointPillarsEncoder, PointPillars
```

`models/pix2poly/model_pix2poly.py`

```python
# from ..pointpillars import PointPillarsViT
from ..vision_transformer import ViT, ViTDINOv2
# from ..fusion_layers import EarlyFusionViT
```

---

### Use absolute paths

`config/host/default.yaml`

```yaml
data_root: /mnt/DATA/IMANE/p3_NY/data
model_root: /mnt/DATA/IMANE
```

---

##  Dataset Preparation – Full NY Subset

The complete P3 dataset contains images from multiple countries. We extract only NY images.

### 1. Filter NY‑only annotations

Create and run filter_ny_only.py inside `/mnt/DATA/IMANE/PixelsPointsPolygons_dataset/data/224/annotations/blocks/` :

```python
import json, os

for split in ['train', 'val', 'test']:
    with open(f'annotations_all_{split}.json') as f:
        data = json.load(f)

    # Keep images where file_name contains '/NY/'
    ny_imgs = [img for img in data['images'] if '/NY/' in img['file_name']]
    ny_ids = {img['id'] for img in ny_imgs}
    ny_anns = [a for a in data['annotations'] if a['image_id'] in ny_ids]

    out = {'images': ny_imgs, 'annotations': ny_anns}
    with open(f'annotations_NY_{split}.json', 'w') as f:
        json.dump(out, f)
```
Results:

* Train: 43,333 images, 142,556 annotations
* Val: 529 images, 2,352 annotations
* Test: 14,313 images, 51,757 annotations

### 2. Create p3_NY_full directory structure

```bash
mkdir -p /mnt/DATA/IMANE/p3_NY_full/data/224
ln -s /mnt/DATA/IMANE/PixelsPointsPolygons_dataset/data/224/images \
      /mnt/DATA/IMANE/p3_NY_full/data/224/images
mkdir -p /mnt/DATA/IMANE/p3_NY_full/data/224/annotations/blocks
```

Copy the filtered JSON files into `p3_NY_full/data/224/annotations/blocks/`.

### 3. Update Hydra configs

`config/host/default.yaml`

```yaml
data_root: /mnt/DATA/IMANE/p3_NY_full/data

config/dataset/p3.yaml
yaml

in_path: /mnt/DATA/IMANE/p3_NY_full/data/224
annotations:
  train: /mnt/DATA/IMANE/p3_NY_full/data/224/annotations/blocks/annotations_NY_train.json
  val:   /mnt/DATA/IMANE/p3_NY_full/data/224/annotations/blocks/annotations_NY_val.json
  test:  /mnt/DATA/IMANE/p3_NY_full/data/224/annotations/blocks/annotations_NY_test.json
```

---

##  Pretrained Backbone

```bash
mkdir -p /mnt/DATA/IMANE/PixelsPointsPolygons_output/backbones

wget https://huggingface.co/rsi/PixelsPointsPolygons/resolve/main/backbones/dino_deitsmall8_pretrain.pth
```

---

##  Training

```bash
export CUDA_VISIBLE_DEVICES=0

python scripts/train.py experiment=p2p_image
```

Resume:

```bash
tmux new-session -d -s train_full_ny "bash -c '
  export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline &&
  cd /mnt/DATA/IMANE/PixelsPointsPolygons &&
  /mnt/DATA/IMANE/p3/bin/python scripts/train.py \
    experiment=p2p_image \
    checkpoint=latest \
    experiment.model.num_epochs=400 \
  2>&1 | tee /mnt/DATA/IMANE/train_full_ny.log'"```
```

* Resume from checkpoint (epoch 199)
* Train for 400 epochs total
* Logs saved to train_full_ny.log and metrics.csv inside the output directory


---
## Monitoring

```bash

# GPU usage
watch -n 1 nvidia-smi

# Real‑time training log
tail -f /mnt/DATA/IMANE/train_full_ny.log

# Plot loss and IoU curves (run anytime)
python /mnt/DATA/IMANE/plot_losses.py
```

The plot_losses.py script reads metrics.csv and produces a two‑panel figure:
(left) train/val loss, (right) validation IoU
```

---

##  Critical Fix for get_tile_names_from_dataloader

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

---

##  Inference (GPU)

```bash
export CUDA_VISIBLE_DEVICES=0

python scripts/predict_demo.py \
  checkpoint=best_val_iou \
  experiment=p2p_image \
  host.device=cuda \
  +image_file=demo_data/image0_CH_val.tif
```

---

##  Common Issues

| Issue              | Fix                  |
| ------------------ | -------------------- |
| open3d error       | remove LiDAR imports |
| Hydra path errors       | use absolute paths   |
| transformers error | downgrade version    |
| IndexError in utils         | Patch get_tile_names_from_dataloader          |
| tmux session dies silently  | Use full Python path, not conda activate
|  wandb authentication failure|  export WANDB_MODE=offline

---

##  Notes

* Works without LiDAR
* Optimized for full NY dataset (43k images)
* Easily extendable to other countries (CH, NZ, etc.) by modifying the filter script
* All changes are tracked in git commits for reproducibility

---

##  Citation

```bibtex
@misc{sulzer2025p3datasetpixelspoints,
  title={The P$^3$ Dataset: Pixels, Points and Polygons},
  author={Raphael Sulzer et al.},
  year={2025}
}
```

---

##  Acknowledgement

All credit to the original authors:

[https://github.com/raphaelsulzer/PixelsPointsPolygons](https://github.com/raphaelsulzer/PixelsPointsPolygons)






