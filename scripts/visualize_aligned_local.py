#!/usr/bin/env python3
"""
Visualize aligned mask data on LOCAL machine.

Prerequisites:
    pip install numpy matplotlib

Usage:
    python visualize_aligned_local.py
    python visualize_aligned_local.py --file aligned_viz.npz
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def visualize(data_file):
    """Interactive visualization."""

    print(f"Loading {data_file}...")
    data = np.load(data_file, allow_pickle=True)

    image = data['image']
    skin_mask = data['skin_mask']
    wall_original = data['wall_original']
    wall_filled = data['wall_filled']
    slice_indices = data['slice_indices']
    subject_id = str(data['subject_id'])
    skin_count = int(data['skin_count'])
    wall_orig_count = int(data['wall_orig_count'])
    wall_fill_count = int(data['wall_fill_count'])
    total_pixels = int(data['total_pixels'])

    print(f"Subject: {subject_id}")
    print(f"Loaded {len(slice_indices)} slices")
    print(f"Skin: {skin_count:,} ({100*skin_count/total_pixels:.3f}%)")
    print(f"Wall original: {wall_orig_count:,} ({100*wall_orig_count/total_pixels:.4f}%)")
    print(f"Wall filled: {wall_fill_count:,} ({100*wall_fill_count/total_pixels:.3f}%)")
    print(f"Fill ratio: {wall_fill_count/max(wall_orig_count,1):.0f}x")

    # Setup figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.subplots_adjust(bottom=0.15, hspace=0.3)

    init_idx = len(slice_indices) // 2
    vmin, vmax = -1000, 1000

    # Row 1
    im0 = axes[0, 0].imshow(image[init_idx], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('CT Image')
    axes[0, 0].axis('off')

    im1 = axes[0, 1].imshow(skin_mask[init_idx], cmap='Reds', vmin=0, vmax=1)
    axes[0, 1].set_title('Skin (aligned)', color='red')
    axes[0, 1].axis('off')

    im2 = axes[0, 2].imshow(wall_original[init_idx], cmap='Greens', vmin=0, vmax=1)
    axes[0, 2].set_title('Wall Original (contour)', color='green')
    axes[0, 2].axis('off')

    # Row 2
    im3 = axes[1, 0].imshow(wall_filled[init_idx], cmap='Blues', vmin=0, vmax=1)
    axes[1, 0].set_title('Wall Filled (solid)', color='blue')
    axes[1, 0].axis('off')

    def make_overlay(idx):
        img = image[idx]
        img_norm = (img - vmin) / (vmax - vmin)
        img_norm = np.clip(img_norm, 0, 1)
        overlay = np.stack([img_norm, img_norm, img_norm], axis=-1)

        # Red for skin
        overlay[skin_mask[idx] > 0, 0] = 1.0
        overlay[skin_mask[idx] > 0, 1] = 0.2
        overlay[skin_mask[idx] > 0, 2] = 0.2

        # Blue for wall
        wall_only = (wall_filled[idx] > 0) & (skin_mask[idx] == 0)
        overlay[wall_only, 0] = 0.3
        overlay[wall_only, 1] = 0.3
        overlay[wall_only, 2] = 1.0

        return overlay

    im4 = axes[1, 1].imshow(make_overlay(init_idx))
    axes[1, 1].set_title('Overlay (Red=Skin, Blue=Wall)')
    axes[1, 1].axis('off')

    # Statistics
    axes[1, 2].axis('off')
    stats_text = f"""Subject: {subject_id}

Skin mask:
  {skin_count:,} voxels
  {100*skin_count/total_pixels:.3f}%

Wall (contour):
  {wall_orig_count:,} voxels
  {100*wall_orig_count/total_pixels:.4f}%

Wall (filled):
  {wall_fill_count:,} voxels
  {100*wall_fill_count/total_pixels:.3f}%

Fill ratio: {wall_fill_count/max(wall_orig_count,1):.0f}x
"""
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace')

    fig.suptitle(f'Aligned Mask Visualization - {subject_id}', fontsize=14)

    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, len(slice_indices) - 1,
                    valinit=init_idx, valstep=1)

    slice_label = ax_slider.text(1.02, 0.5, f'[{slice_indices[init_idx]}]',
                                  transform=ax_slider.transAxes, va='center')

    def update(val):
        idx = int(slider.val)
        im0.set_data(image[idx])
        im1.set_data(skin_mask[idx])
        im2.set_data(wall_original[idx])
        im3.set_data(wall_filled[idx])
        im4.set_data(make_overlay(idx))
        slice_label.set_text(f'[{slice_indices[idx]}]')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize aligned masks (local)')
    parser.add_argument('--file', type=str, default='aligned_viz.npz',
                        help='Input npz file')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        print("\nMake sure you:")
        print("1. Run on server: python scripts/visualize_aligned_masks.py --save")
        print("2. Copy to local: scp SERVER:~/aligned_viz.npz ./")
        print("3. Run locally:   python visualize_aligned_local.py")
        sys.exit(1)

    visualize(args.file)


if __name__ == '__main__':
    main()
