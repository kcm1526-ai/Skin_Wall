#!/usr/bin/env python3
"""
Visualize the wall contour filling operation.
Shows original outline vs filled mask side by side.

Usage:
    python scripts/visualize_wall_fill.py
    python scripts/visualize_wall_fill.py --subject SUBJECT_ID
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import binary_fill_holes

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import get_config
from src.dataset import DICOMLoader, MaskLoader, find_data_paths


def fill_contour_mask(mask: np.ndarray) -> np.ndarray:
    """Fill the interior of a contour/outline mask slice by slice."""
    filled = np.zeros_like(mask)
    for z in range(mask.shape[0]):
        slice_2d = mask[z, :, :]
        if slice_2d.max() > 0:
            filled[z, :, :] = binary_fill_holes(slice_2d).astype(np.uint8)
    return filled


def align_mask_to_image(mask: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Align NIfTI mask to DICOM image coordinate system."""
    transposed = np.transpose(mask, (2, 1, 0))
    if transposed.shape == target_shape:
        aligned = np.flip(transposed, axis=0)
        return np.ascontiguousarray(aligned)

    for axes in [(2, 0, 1), (1, 2, 0), (0, 2, 1), (1, 0, 2), (0, 1, 2)]:
        transposed = np.transpose(mask, axes)
        if transposed.shape == target_shape:
            aligned = np.flip(transposed, axis=0)
            return np.ascontiguousarray(aligned)

    from scipy.ndimage import zoom
    zoom_factors = [t / s for t, s in zip(target_shape, mask.shape)]
    resampled = zoom(mask.astype(np.float32), zoom_factors, order=0)
    return resampled.astype(np.uint8)


def visualize_fill(image, wall_original, wall_filled, title="Wall Mask Fill Visualization"):
    """Interactive visualization of original vs filled wall mask."""

    # Find slices that have wall contour
    wall_slices = np.where(wall_original.sum(axis=(1, 2)) > 0)[0]
    if len(wall_slices) == 0:
        print("No wall contour found in this volume!")
        return

    # Start at middle slice with wall
    init_slice = wall_slices[len(wall_slices) // 2]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    plt.subplots_adjust(bottom=0.2)

    # Initial display
    im0 = axes[0].imshow(image[init_slice], cmap='gray', vmin=-1000, vmax=1000)
    axes[0].set_title('CT Image')
    axes[0].axis('off')

    im1 = axes[1].imshow(wall_original[init_slice], cmap='Reds', vmin=0, vmax=1)
    axes[1].set_title('Original (Contour Only)')
    axes[1].axis('off')

    im2 = axes[2].imshow(wall_filled[init_slice], cmap='Blues', vmin=0, vmax=1)
    axes[2].set_title('Filled (Solid Region)')
    axes[2].axis('off')

    # Overlay: CT with filled mask
    overlay = np.zeros((*image[init_slice].shape, 3))
    img_norm = (image[init_slice] - image[init_slice].min()) / (image[init_slice].max() - image[init_slice].min() + 1e-8)
    overlay[:, :, 0] = img_norm
    overlay[:, :, 1] = img_norm
    overlay[:, :, 2] = img_norm
    overlay[:, :, 0] = np.where(wall_filled[init_slice] > 0, 1.0, overlay[:, :, 0])
    overlay[:, :, 1] = np.where(wall_filled[init_slice] > 0, 0.3, overlay[:, :, 1])
    overlay[:, :, 2] = np.where(wall_filled[init_slice] > 0, 0.3, overlay[:, :, 2])

    im3 = axes[3].imshow(overlay)
    axes[3].set_title('CT + Filled Overlay')
    axes[3].axis('off')

    # Calculate statistics
    orig_pixels = wall_original.sum()
    filled_pixels = wall_filled.sum()
    total_pixels = wall_original.size

    fig.suptitle(f'{title}\n'
                 f'Original: {orig_pixels:,} pixels ({100*orig_pixels/total_pixels:.3f}%) | '
                 f'Filled: {filled_pixels:,} pixels ({100*filled_pixels/total_pixels:.2f}%) | '
                 f'Ratio: {filled_pixels/max(orig_pixels,1):.1f}x', fontsize=12)

    # Slider for slice selection
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, image.shape[0] - 1, valinit=init_slice, valstep=1)

    def update(val):
        z = int(slider.val)
        im0.set_data(image[z])
        im1.set_data(wall_original[z])
        im2.set_data(wall_filled[z])

        # Update overlay
        overlay = np.zeros((*image[z].shape, 3))
        img_norm = (image[z] - image[z].min()) / (image[z].max() - image[z].min() + 1e-8)
        overlay[:, :, 0] = img_norm
        overlay[:, :, 1] = img_norm
        overlay[:, :, 2] = img_norm
        overlay[:, :, 0] = np.where(wall_filled[z] > 0, 1.0, overlay[:, :, 0])
        overlay[:, :, 1] = np.where(wall_filled[z] > 0, 0.3, overlay[:, :, 1])
        overlay[:, :, 2] = np.where(wall_filled[z] > 0, 0.3, overlay[:, :, 2])
        im3.set_data(overlay)

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize wall contour filling')
    parser.add_argument('--subject', type=str, default=None, help='Specific subject ID to visualize')
    args = parser.parse_args()

    config = get_config()

    # Find samples
    print("Finding data samples...")
    samples = find_data_paths(config.data.base_path, config)

    if not samples:
        print("No samples found!")
        return

    # Select sample
    if args.subject:
        sample = next((s for s in samples if s['subject_id'] == args.subject), None)
        if sample is None:
            print(f"Subject {args.subject} not found!")
            print(f"Available subjects: {[s['subject_id'] for s in samples[:10]]}...")
            return
    else:
        sample = samples[0]

    print(f"Loading subject: {sample['subject_id']}")

    # Load DICOM
    print("Loading DICOM...")
    image, image_meta = DICOMLoader.load_dicom_series(sample['image_dir'])
    print(f"  Image shape: {image.shape}")

    # Load wall mask
    print("Loading wall mask...")
    wall_mask_raw, _ = MaskLoader.load_nifti_mask(sample['abdominal_wall_mask'])
    print(f"  Raw mask shape: {wall_mask_raw.shape}")

    # Align mask
    print("Aligning mask...")
    wall_mask_aligned = align_mask_to_image(wall_mask_raw, image.shape)
    print(f"  Aligned mask shape: {wall_mask_aligned.shape}")

    # Fill mask
    print("Filling contour...")
    wall_mask_filled = fill_contour_mask(wall_mask_aligned)

    # Statistics
    orig_nonzero = (wall_mask_aligned > 0).sum()
    filled_nonzero = (wall_mask_filled > 0).sum()
    print(f"\nStatistics:")
    print(f"  Original (contour): {orig_nonzero:,} pixels ({100*orig_nonzero/wall_mask_aligned.size:.4f}%)")
    print(f"  Filled (solid):     {filled_nonzero:,} pixels ({100*filled_nonzero/wall_mask_filled.size:.3f}%)")
    print(f"  Fill ratio:         {filled_nonzero/max(orig_nonzero,1):.1f}x larger")

    # Visualize
    print("\nOpening visualization...")
    visualize_fill(image, wall_mask_aligned, wall_mask_filled, f"Subject: {sample['subject_id']}")


if __name__ == '__main__':
    main()
