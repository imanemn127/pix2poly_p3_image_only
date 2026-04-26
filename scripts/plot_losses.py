"""
Read metrics.csv produced by Pix2PolyTrainer and plot training curves.

Usage:
    python scripts/plot_losses.py --csv /path/to/metrics.csv --out losses.png
"""

import argparse
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_csv(path):
    epochs, train_loss, val_loss, val_iou = [], [], [], []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(row['epoch']))
                train_loss.append(float(row['train_loss']))
                val_loss.append(float(row['val_loss']))
                iou = row.get('val_iou', 'nan')
                val_iou.append(float(iou) if iou not in ('', 'nan') else float('nan'))
            except (ValueError, KeyError):
                continue
    return epochs, train_loss, val_loss, val_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='metrics.csv', help='Path to metrics.csv')
    parser.add_argument('--out', default='losses.png', help='Output figure path')
    args = parser.parse_args()

    epochs, train_loss, val_loss, val_iou = load_csv(args.csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, label='Train loss')
    ax1.plot(epochs, val_loss, label='Val loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train / Val Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, val_iou, label='Val IoU', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.set_title('Validation IoU')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved to {args.out}")


if __name__ == '__main__':
    main()
