#!/usr/bin/env python3
"""
Diagnose mask alignment issues between DICOM images and NIfTI masks.
Tests different flip/transpose combinations to find the correct alignment.

Usage:
    python scripts/diagnose_alignment.py
    python scripts/diagnose_alignment.py --save  # Save result for local viewing
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import pydicom
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import get_config
from src.dataset import DICOMLoader, MaskLoader, find_data_paths
from scipy.ndimage import binary_fill_holes


def try_all_orientations(mask: np.ndarray, target_shape: tuple):
    """Try all possible transpose and flip combinations"""
    orientations = []

    # All possible transpose orders
    transpose_orders = [
        (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)
    ]

    for order in transpose_orders:
        transposed = np.transpose(mask, order)
        if transposed.shape == target_shape:
            # Try all flip combinations
            for flip_x in [False, True]:
                for flip_y in [False, True]:
                    for flip_z in [False, True]:
                        flipped = transposed.copy()
                        if flip_x:
                            flipped = np.flip(flipped, axis=0)
                        if flip_y:
                            flipped = np.flip(flipped, axis=1)
                        if flip_z:
                            flipped = np.flip(flipped, axis=2)

                        orientations.append({
                            'transpose': order,
                            'flip': (flip_x, flip_y, flip_z),
                            'mask': np.ascontiguousarray(flipped)
                        })

    return orientations


def fill_contour(mask):
    """Fill wall contour slice by slice."""
    filled = np.zeros_like(mask)
    for z in range(mask.shape[0]):
        if mask[z].max() > 0:
            filled[z] = binary_fill_holes(mask[z]).astype(np.uint8)
    return filled


def compute_alignment_score(image: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute a score for how well the mask aligns with the image.
    Good alignment: mask should be on body tissue, not air (low HU values).
    """
    masked_values = image[mask > 0]

    if len(masked_values) == 0:
        return -1000

    # Good alignment: mask should cover tissue (HU > -500), not air (HU ~ -1000)
    return np.mean(masked_values)


def main():
    parser = argparse.ArgumentParser(description='Diagnose mask alignment')
    parser.add_argument('--save', action='store_true', help='Save to ~/alignment_test.npz')
    parser.add_argument('--output', type=str, default=os.path.expanduser('~/alignment_test.npz'))
    args = parser.parse_args()

    config = get_config()

    print("Finding samples...")
    samples = find_data_paths(config.data.base_path, config)
    if not samples:
        print("No samples found!")
        return

    sample = samples[0]
    print(f"Loading: {sample['subject_id']}")

    # Load DICOM
    print("Loading DICOM...")
    image, meta = DICOMLoader.load_dicom_series(sample['image_dir'])
    print(f"  Image shape: {image.shape}")

    # Load wall mask (this is the problematic one)
    print("Loading wall mask...")
    wall_mask_raw, _ = MaskLoader.load_nifti_mask(sample['abdominal_wall_mask'])
    print(f"  Mask raw shape: {wall_mask_raw.shape}")

    # Try all orientations
    print("\nTrying all orientations...")
    orientations = try_all_orientations(wall_mask_raw, image.shape)
    print(f"Found {len(orientations)} valid orientations")

    if not orientations:
        print("ERROR: No valid orientation found!")
        return

    # Fill contours and compute scores
    print("Filling contours and computing scores...")
    for ori in orientations:
        ori['filled'] = fill_contour(ori['mask'])
        ori['score'] = compute_alignment_score(image, ori['filled'])

    # Sort by score
    sorted_oris = sorted(orientations, key=lambda x: x['score'], reverse=True)

    # Print top results
    print("\n=== TOP 5 ALIGNMENTS (by score) ===")
    for i, ori in enumerate(sorted_oris[:5]):
        print(f"{i+1}. T:{ori['transpose']} F:{ori['flip']} -> Score: {ori['score']:.1f}")

    best = sorted_oris[0]
    print(f"\n=== BEST ALIGNMENT ===")
    print(f"Transpose: {best['transpose']}")
    print(f"Flip (axis0, axis1, axis2): {best['flip']}")
    print(f"Score: {best['score']:.1f}")

    if args.save:
        # Save for local visualization
        print(f"\nSaving to {args.output}...")

        # Find slices with content
        mid = image.shape[0] // 2
        selected = np.arange(max(0, mid-30), min(image.shape[0], mid+30))

        # Save top 4 alignments
        np.savez_compressed(
            args.output,
            image=image[selected].astype(np.float32),
            ori_1_mask=sorted_oris[0]['filled'][selected].astype(np.uint8),
            ori_1_info=f"T:{sorted_oris[0]['transpose']} F:{sorted_oris[0]['flip']} Score:{sorted_oris[0]['score']:.0f}",
            ori_2_mask=sorted_oris[1]['filled'][selected].astype(np.uint8),
            ori_2_info=f"T:{sorted_oris[1]['transpose']} F:{sorted_oris[1]['flip']} Score:{sorted_oris[1]['score']:.0f}",
            ori_3_mask=sorted_oris[2]['filled'][selected].astype(np.uint8),
            ori_3_info=f"T:{sorted_oris[2]['transpose']} F:{sorted_oris[2]['flip']} Score:{sorted_oris[2]['score']:.0f}",
            ori_4_mask=sorted_oris[3]['filled'][selected].astype(np.uint8),
            ori_4_info=f"T:{sorted_oris[3]['transpose']} F:{sorted_oris[3]['flip']} Score:{sorted_oris[3]['score']:.0f}",
            slice_indices=selected,
            subject_id=sample['subject_id']
        )
        print(f"Saved! Copy to local and run visualize_alignment_local.py")
    else:
        # Interactive visualization
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        # Show top 4 alignments
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        plt.subplots_adjust(bottom=0.15)

        mid = image.shape[0] // 2
        vmin, vmax = -1000, 1000

        def update(z):
            z = int(z)
            for i, ori in enumerate(sorted_oris[:4]):
                row, col = i // 2, (i % 2) * 2

                # CT
                axes[row, col].clear()
                axes[row, col].imshow(image[z], cmap='gray', vmin=vmin, vmax=vmax)
                axes[row, col].set_title(f'CT (slice {z})')
                axes[row, col].axis('off')

                # Overlay
                axes[row, col+1].clear()
                img_norm = (image[z] + 1000) / 2000
                img_norm = np.clip(img_norm, 0, 1)
                overlay = np.stack([img_norm, img_norm, img_norm], axis=-1)
                mask = ori['filled'][z]
                overlay[mask > 0, 0] = 1.0
                overlay[mask > 0, 1] = 0.2
                overlay[mask > 0, 2] = 0.2
                axes[row, col+1].imshow(overlay)
                axes[row, col+1].set_title(f"T:{ori['transpose']} F:{ori['flip']}\nScore:{ori['score']:.0f}")
                axes[row, col+1].axis('off')

            fig.canvas.draw_idle()

        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        slider = Slider(ax_slider, 'Slice', 0, image.shape[0]-1, valinit=mid, valstep=1)
        slider.on_changed(update)

        update(mid)
        fig.suptitle(f"Subject: {sample['subject_id']} - Find the correct alignment!", fontsize=14)
        plt.show()


if __name__ == '__main__':
    main()
