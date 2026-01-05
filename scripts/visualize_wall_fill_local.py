#!/usr/bin/env python3
"""
Visualize wall fill data on LOCAL machine.

Prerequisites:
    pip install numpy matplotlib

Usage:
    python visualize_wall_fill_local.py
    python visualize_wall_fill_local.py --file wall_fill_viz.npz
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def visualize(data_file):
    """Interactive visualization of original vs filled wall mask."""

    # Load data
    print(f"Loading {data_file}...")
    data = np.load(data_file, allow_pickle=True)

    image = data['image']
    wall_original = data['wall_original']
    wall_filled = data['wall_filled']
    slice_indices = data['slice_indices']
    subject_id = str(data['subject_id'])
    orig_count = int(data['orig_count'])
    filled_count = int(data['filled_count'])
    total_pixels = int(data['total_pixels'])

    print(f"Subject: {subject_id}")
    print(f"Loaded {len(slice_indices)} slices")
    print(f"Original: {orig_count:,} pixels ({100*orig_count/total_pixels:.4f}%)")
    print(f"Filled:   {filled_count:,} pixels ({100*filled_count/total_pixels:.3f}%)")
    print(f"Ratio:    {filled_count/max(orig_count,1):.1f}x larger")

    # Setup figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    plt.subplots_adjust(bottom=0.2)

    init_idx = len(slice_indices) // 2

    # Initial display
    vmin, vmax = -1000, 1000
    im0 = axes[0].imshow(image[init_idx], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title('CT Image')
    axes[0].axis('off')

    im1 = axes[1].imshow(wall_original[init_idx], cmap='Reds', vmin=0, vmax=1)
    axes[1].set_title('Original\n(Contour Only)', color='red')
    axes[1].axis('off')

    im2 = axes[2].imshow(wall_filled[init_idx], cmap='Blues', vmin=0, vmax=1)
    axes[2].set_title('Filled\n(Solid Region)', color='blue')
    axes[2].axis('off')

    # Overlay
    def make_overlay(idx):
        img = image[idx]
        img_norm = (img - vmin) / (vmax - vmin)
        img_norm = np.clip(img_norm, 0, 1)

        overlay = np.stack([img_norm, img_norm, img_norm], axis=-1)

        # Red for original contour
        mask_orig = wall_original[idx] > 0
        overlay[mask_orig, 0] = 1.0
        overlay[mask_orig, 1] = 0.2
        overlay[mask_orig, 2] = 0.2

        # Blue tint for filled (but not contour)
        mask_fill_only = (wall_filled[idx] > 0) & (~mask_orig)
        overlay[mask_fill_only, 0] = overlay[mask_fill_only, 0] * 0.5
        overlay[mask_fill_only, 1] = overlay[mask_fill_only, 1] * 0.5 + 0.3
        overlay[mask_fill_only, 2] = overlay[mask_fill_only, 2] * 0.5 + 0.5

        return overlay

    im3 = axes[3].imshow(make_overlay(init_idx))
    axes[3].set_title('Overlay\n(Red=Contour, Blue=Filled)')
    axes[3].axis('off')

    fig.suptitle(
        f'Subject: {subject_id}  |  '
        f'Original: {orig_count:,} ({100*orig_count/total_pixels:.4f}%)  |  '
        f'Filled: {filled_count:,} ({100*filled_count/total_pixels:.2f}%)  |  '
        f'Ratio: {filled_count/max(orig_count,1):.0f}x',
        fontsize=11
    )

    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        ax_slider, 'Slice',
        0, len(slice_indices) - 1,
        valinit=init_idx, valstep=1
    )

    # Add slice index label
    slice_label = ax_slider.text(
        1.02, 0.5, f'[{slice_indices[init_idx]}]',
        transform=ax_slider.transAxes, va='center'
    )

    def update(val):
        idx = int(slider.val)
        im0.set_data(image[idx])
        im1.set_data(wall_original[idx])
        im2.set_data(wall_filled[idx])
        im3.set_data(make_overlay(idx))
        slice_label.set_text(f'[{slice_indices[idx]}]')
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize wall fill (local)')
    parser.add_argument('--file', type=str, default='wall_fill_viz.npz',
                        help='Input npz file (default: wall_fill_viz.npz)')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        print("\nMake sure you:")
        print("1. Run on server: python scripts/export_wall_fill_sample.py")
        print("2. Copy to local: scp SERVER:~/wall_fill_viz.npz ./")
        print("3. Run locally:   python visualize_wall_fill_local.py")
        sys.exit(1)

    visualize(args.file)


if __name__ == '__main__':
    main()
