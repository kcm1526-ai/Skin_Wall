#!/usr/bin/env python3
"""
Visualize DICOM image with mask overlay after alignment and wall filling.

This script:
1. Reads DICOM and NIfTI mask files directly
2. Applies mask alignment (transpose + flip)
3. Fills wall contour (converts outline to solid region)
4. Visualizes CT with mask overlay

Usage:
    python scripts/visualize_aligned_masks.py
    python scripts/visualize_aligned_masks.py --subject SUBJECT_ID
    python scripts/visualize_aligned_masks.py --save  # Save to ~/aligned_viz.npz for local viewing
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
from scipy.ndimage import binary_fill_holes

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import get_config
from src.dataset import DICOMLoader, MaskLoader, find_data_paths


def align_mask_to_image(mask: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Align NIfTI mask to DICOM image coordinate system.

    NIfTI: (X, Y, Z) orientation
    DICOM: (Z, Y, X) orientation

    Transformation: transpose(2, 1, 0) only (no flip needed)
    """
    # Apply transpose (2, 1, 0) to convert (X, Y, Z) -> (Z, Y, X)
    transposed = np.transpose(mask, (2, 1, 0))

    if transposed.shape == target_shape:
        return np.ascontiguousarray(transposed)

    # Try other permutations if needed
    for axes in [(2, 0, 1), (1, 2, 0), (0, 2, 1), (1, 0, 2), (0, 1, 2)]:
        transposed = np.transpose(mask, axes)
        if transposed.shape == target_shape:
            return np.ascontiguousarray(transposed)

    # Fallback: resample
    from scipy.ndimage import zoom
    print(f"  Warning: Mask shape {mask.shape} -> target {target_shape}, resampling...")
    zoom_factors = [t / s for t, s in zip(target_shape, mask.shape)]
    resampled = zoom(mask.astype(np.float32), zoom_factors, order=0)
    return resampled.astype(np.uint8)


def fill_wall_contour(mask: np.ndarray) -> np.ndarray:
    """Fill the interior of wall contour mask slice by slice."""
    filled = np.zeros_like(mask)
    for z in range(mask.shape[0]):
        slice_2d = mask[z, :, :]
        if slice_2d.max() > 0:
            filled[z, :, :] = binary_fill_holes(slice_2d).astype(np.uint8)
    return filled


def visualize_interactive(image, skin_mask, wall_original, wall_filled, subject_id):
    """Interactive visualization with matplotlib."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, CheckButtons

    # Find slices with masks
    mask_slices = np.where((skin_mask.sum(axis=(1, 2)) > 0) | (wall_filled.sum(axis=(1, 2)) > 0))[0]
    if len(mask_slices) == 0:
        mask_slices = np.arange(image.shape[0])
    init_slice = mask_slices[len(mask_slices) // 2]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.subplots_adjust(bottom=0.2, hspace=0.3)

    vmin, vmax = -1000, 1000

    # Row 1: CT, Skin, Wall Original
    im_ct = axes[0, 0].imshow(image[init_slice], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('CT Image')
    axes[0, 0].axis('off')

    im_skin = axes[0, 1].imshow(skin_mask[init_slice], cmap='Reds', vmin=0, vmax=1)
    axes[0, 1].set_title('Skin Mask (aligned)')
    axes[0, 1].axis('off')

    im_wall_orig = axes[0, 2].imshow(wall_original[init_slice], cmap='Greens', vmin=0, vmax=1)
    axes[0, 2].set_title('Wall Original (contour)')
    axes[0, 2].axis('off')

    # Row 2: Wall Filled, Overlay, Statistics
    im_wall_fill = axes[1, 0].imshow(wall_filled[init_slice], cmap='Blues', vmin=0, vmax=1)
    axes[1, 0].set_title('Wall Filled (solid)')
    axes[1, 0].axis('off')

    # Overlay
    def make_overlay(z):
        img = image[z]
        img_norm = (img - vmin) / (vmax - vmin)
        img_norm = np.clip(img_norm, 0, 1)
        overlay = np.stack([img_norm, img_norm, img_norm], axis=-1)

        # Red for skin
        overlay[skin_mask[z] > 0, 0] = 1.0
        overlay[skin_mask[z] > 0, 1] = 0.2
        overlay[skin_mask[z] > 0, 2] = 0.2

        # Blue for filled wall
        overlay[wall_filled[z] > 0, 0] = 0.2
        overlay[wall_filled[z] > 0, 1] = 0.2
        overlay[wall_filled[z] > 0, 2] = 1.0

        return overlay

    im_overlay = axes[1, 1].imshow(make_overlay(init_slice))
    axes[1, 1].set_title('Overlay (Red=Skin, Blue=Wall)')
    axes[1, 1].axis('off')

    # Statistics panel
    axes[1, 2].axis('off')
    skin_count = (skin_mask > 0).sum()
    wall_orig_count = (wall_original > 0).sum()
    wall_fill_count = (wall_filled > 0).sum()
    total = image.size

    stats_text = f"""Statistics:

Image shape: {image.shape}
Total voxels: {total:,}

Skin mask:
  Count: {skin_count:,}
  Ratio: {100*skin_count/total:.3f}%

Wall (original contour):
  Count: {wall_orig_count:,}
  Ratio: {100*wall_orig_count/total:.4f}%

Wall (filled solid):
  Count: {wall_fill_count:,}
  Ratio: {100*wall_fill_count/total:.3f}%

Fill ratio: {wall_fill_count/max(wall_orig_count,1):.1f}x
"""
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')

    fig.suptitle(f'Subject: {subject_id}', fontsize=14)

    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, image.shape[0] - 1, valinit=init_slice, valstep=1)

    def update(val):
        z = int(slider.val)
        im_ct.set_data(image[z])
        im_skin.set_data(skin_mask[z])
        im_wall_orig.set_data(wall_original[z])
        im_wall_fill.set_data(wall_filled[z])
        im_overlay.set_data(make_overlay(z))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def save_for_local(image, skin_mask, wall_original, wall_filled, subject_id, output_path):
    """Save data to npz file for local visualization."""
    # Take subset of slices
    mask_slices = np.where((skin_mask.sum(axis=(1, 2)) > 0) | (wall_filled.sum(axis=(1, 2)) > 0))[0]
    if len(mask_slices) > 0:
        step = max(1, len(mask_slices) // 60)
        selected = mask_slices[::step]
    else:
        mid = image.shape[0] // 2
        selected = np.arange(max(0, mid-30), min(image.shape[0], mid+30))

    np.savez_compressed(
        output_path,
        image=image[selected].astype(np.float32),
        skin_mask=skin_mask[selected].astype(np.uint8),
        wall_original=wall_original[selected].astype(np.uint8),
        wall_filled=wall_filled[selected].astype(np.uint8),
        slice_indices=selected,
        subject_id=subject_id,
        skin_count=int((skin_mask > 0).sum()),
        wall_orig_count=int((wall_original > 0).sum()),
        wall_fill_count=int((wall_filled > 0).sum()),
        total_pixels=int(image.size)
    )
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved {len(selected)} slices to {output_path} ({file_size:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description='Visualize aligned masks')
    parser.add_argument('--subject', type=str, default=None, help='Specific subject ID')
    parser.add_argument('--save', action='store_true', help='Save to ~/aligned_viz.npz instead of displaying')
    parser.add_argument('--output', type=str, default=os.path.expanduser('~/aligned_viz.npz'),
                        help='Output file for --save mode')
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
            print(f"Available: {[s['subject_id'] for s in samples[:10]]}...")
            return
    else:
        sample = samples[0]

    print(f"\nLoading subject: {sample['subject_id']}")

    # Step 1: Load DICOM
    print("  Loading DICOM...")
    image, image_meta = DICOMLoader.load_dicom_series(sample['image_dir'])
    print(f"    Shape: {image.shape}, Spacing: {image_meta['spacing']}")

    # Step 2: Load masks
    print("  Loading masks...")
    skin_mask_raw, _ = MaskLoader.load_nifti_mask(sample['skin_mask'])
    wall_mask_raw, _ = MaskLoader.load_nifti_mask(sample['abdominal_wall_mask'])
    print(f"    Skin raw shape: {skin_mask_raw.shape}")
    print(f"    Wall raw shape: {wall_mask_raw.shape}")

    # Step 3: Align masks
    print("  Aligning masks (transpose + flip)...")
    skin_mask = align_mask_to_image(skin_mask_raw, image.shape)
    wall_original = align_mask_to_image(wall_mask_raw, image.shape)
    print(f"    Aligned shapes: {skin_mask.shape}, {wall_original.shape}")

    # Step 4: Fill wall contour
    print("  Filling wall contour...")
    wall_filled = fill_wall_contour(wall_original)

    # Statistics
    print(f"\nStatistics:")
    print(f"  Skin: {(skin_mask > 0).sum():,} voxels ({100*(skin_mask > 0).sum()/image.size:.3f}%)")
    print(f"  Wall original: {(wall_original > 0).sum():,} voxels ({100*(wall_original > 0).sum()/image.size:.4f}%)")
    print(f"  Wall filled: {(wall_filled > 0).sum():,} voxels ({100*(wall_filled > 0).sum()/image.size:.3f}%)")
    print(f"  Fill ratio: {(wall_filled > 0).sum() / max((wall_original > 0).sum(), 1):.1f}x")

    if args.save:
        print(f"\nSaving to {args.output}...")
        save_for_local(image, skin_mask, wall_original, wall_filled, sample['subject_id'], args.output)
        print("\n=== Next Steps ===")
        print(f"1. Copy to local: scp SERVER:{args.output} ./")
        print(f"2. Run locally:   python visualize_aligned_local.py")
    else:
        print("\nOpening visualization...")
        visualize_interactive(image, skin_mask, wall_original, wall_filled, sample['subject_id'])


if __name__ == '__main__':
    main()
