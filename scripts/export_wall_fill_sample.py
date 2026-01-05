#!/usr/bin/env python3
"""
Export wall fill visualization data to home directory for local viewing.

Run on SERVER:
    python scripts/export_wall_fill_sample.py
    python scripts/export_wall_fill_sample.py --subject SUBJECT_ID

Then copy files to local machine:
    scp server:~/wall_fill_viz.npz ./

Then run locally:
    python visualize_wall_fill_local.py
"""

import os
import sys
from pathlib import Path
import numpy as np
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Export wall fill visualization data')
    parser.add_argument('--subject', type=str, default=None, help='Specific subject ID')
    parser.add_argument('--output', type=str, default=os.path.expanduser('~/wall_fill_viz.npz'),
                        help='Output file path (default: ~/wall_fill_viz.npz)')
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

    print(f"Loading subject: {sample['subject_id']}")

    # Load DICOM
    print("Loading DICOM...")
    image, image_meta = DICOMLoader.load_dicom_series(sample['image_dir'])
    print(f"  Image shape: {image.shape}")

    # Load wall mask
    print("Loading wall mask...")
    wall_mask_raw, _ = MaskLoader.load_nifti_mask(sample['abdominal_wall_mask'])

    # Align mask
    print("Aligning mask...")
    wall_original = align_mask_to_image(wall_mask_raw, image.shape)

    # Fill mask
    print("Filling contour...")
    wall_filled = fill_contour_mask(wall_original)

    # Statistics
    orig_count = (wall_original > 0).sum()
    filled_count = (wall_filled > 0).sum()
    print(f"\nStatistics:")
    print(f"  Original (contour): {orig_count:,} pixels ({100*orig_count/wall_original.size:.4f}%)")
    print(f"  Filled (solid):     {filled_count:,} pixels ({100*filled_count/wall_filled.size:.3f}%)")
    print(f"  Fill ratio:         {filled_count/max(orig_count,1):.1f}x larger")

    # Find slices with wall contour for efficient storage
    wall_slices = np.where(wall_original.sum(axis=(1, 2)) > 0)[0]
    if len(wall_slices) > 0:
        # Take a subset of slices (every 5th slice with wall, max 50 slices)
        step = max(1, len(wall_slices) // 50)
        selected_slices = wall_slices[::step]

        # Extract selected slices
        image_subset = image[selected_slices].astype(np.float32)
        wall_orig_subset = wall_original[selected_slices].astype(np.uint8)
        wall_filled_subset = wall_filled[selected_slices].astype(np.uint8)
    else:
        # No wall found, take middle slices
        mid = image.shape[0] // 2
        selected_slices = np.arange(max(0, mid-25), min(image.shape[0], mid+25))
        image_subset = image[selected_slices].astype(np.float32)
        wall_orig_subset = wall_original[selected_slices].astype(np.uint8)
        wall_filled_subset = wall_filled[selected_slices].astype(np.uint8)

    # Save to file
    print(f"\nSaving to {args.output}...")
    np.savez_compressed(
        args.output,
        image=image_subset,
        wall_original=wall_orig_subset,
        wall_filled=wall_filled_subset,
        slice_indices=selected_slices,
        subject_id=sample['subject_id'],
        orig_count=orig_count,
        filled_count=filled_count,
        total_pixels=wall_original.size
    )

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"  Saved {len(selected_slices)} slices ({file_size:.1f} MB)")

    print(f"\n=== Next Steps ===")
    print(f"1. Copy to local: scp SERVER:{args.output} ./")
    print(f"2. Run locally:   python visualize_wall_fill_local.py")


if __name__ == '__main__':
    main()
