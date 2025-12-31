#!/usr/bin/env python3
"""
Pre-cache DICOM data to numpy files for faster loading during training.
Run this once before training to convert all DICOM series to .npy files.

Usage:
    python scripts/precache_data.py
    python scripts/precache_data.py --force  # Re-cache all with correct alignment
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from scipy.ndimage import binary_fill_holes

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import get_config
from src.dataset import DICOMLoader, MaskLoader, find_data_paths


def fill_contour_mask(mask: np.ndarray) -> np.ndarray:
    """
    Fill the interior of a contour/outline mask.

    The wall mask is just an outline/boundary. This function fills
    everything inside the closed contour to create a solid mask.

    Args:
        mask: 3D binary mask where 1 = contour outline

    Returns:
        3D binary mask where 1 = filled interior (including contour)
    """
    filled = np.zeros_like(mask)

    # Process slice by slice (axial slices) since contours are drawn per-slice
    for z in range(mask.shape[0]):
        slice_2d = mask[z, :, :]
        if slice_2d.max() > 0:  # Only process slices with contour
            # binary_fill_holes fills the interior of closed contours
            filled[z, :, :] = binary_fill_holes(slice_2d).astype(np.uint8)

    return filled


def align_mask_to_image(mask: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Align NIfTI mask to DICOM image coordinate system.

    NIfTI stores as (X, Y, Z), DICOM loads as (Z, Y, X).
    Based on diagnostic analysis, the correct transformation is:
    - Transpose: (2, 1, 0) to convert axes
    - Flip: axis 0 to correct orientation

    This function ALWAYS applies the transformation since NIfTI masks
    always need to be transformed to match DICOM coordinates.
    """
    # Always apply transpose (2, 1, 0) first
    transposed = np.transpose(mask, (2, 1, 0))

    # Check if transposed shape matches target
    if transposed.shape == target_shape:
        # Apply flip for correct orientation
        aligned = np.flip(transposed, axis=0)
        return np.ascontiguousarray(aligned)

    # If shapes don't match after transpose, try other permutations
    for axes in [(2, 0, 1), (1, 2, 0), (0, 2, 1), (1, 0, 2), (0, 1, 2)]:
        transposed = np.transpose(mask, axes)
        if transposed.shape == target_shape:
            # Try with and without flip
            aligned = np.flip(transposed, axis=0)
            return np.ascontiguousarray(aligned)

    # Fallback: resample (less ideal)
    print(f"  Warning: Mask shape {mask.shape} doesn't match target {target_shape} after transpose, resampling...")
    from scipy.ndimage import zoom
    zoom_factors = [t / s for t, s in zip(target_shape, mask.shape)]
    resampled = zoom(mask.astype(np.float32), zoom_factors, order=0)
    return resampled.astype(np.uint8)


def process_sample(sample: dict, cache_dir: str, force_recache: bool = False) -> dict:
    """Process a single sample and save to cache."""
    subject_id = sample['subject_id']
    # Use .npy (uncompressed) for faster loading - no decompression overhead
    cache_path = os.path.join(cache_dir, f"{subject_id}.npy")

    # Skip if already cached (unless force_recache)
    if not force_recache:
        if os.path.exists(cache_path) or os.path.exists(cache_path.replace('.npy', '.npz')):
            return {'subject_id': subject_id, 'status': 'skipped', 'path': cache_path}

    try:
        # Load DICOM
        image, image_meta = DICOMLoader.load_dicom_series(sample['image_dir'])

        # Load masks
        skin_mask, _ = MaskLoader.load_nifti_mask(sample['skin_mask'])
        abdominal_mask, _ = MaskLoader.load_nifti_mask(sample['abdominal_wall_mask'])

        # IMPORTANT: Apply alignment transformation to masks
        # NIfTI masks need transpose(2,1,0) + flip(axis=0) to align with DICOM
        skin_mask = align_mask_to_image(skin_mask, image.shape)
        abdominal_mask = align_mask_to_image(abdominal_mask, image.shape)

        # IMPORTANT: Fill the wall contour mask
        # The wall mask is just an outline/boundary - we need the filled interior
        abdominal_mask = fill_contour_mask(abdominal_mask)

        # Save as uncompressed numpy dict (much faster to load than .npz)
        cache_data = {
            'image': image.astype(np.float32),
            'skin_mask': skin_mask.astype(np.uint8),
            'abdominal_mask': abdominal_mask.astype(np.uint8),
            'spacing': np.array(image_meta['spacing'])
        }
        np.save(cache_path, cache_data, allow_pickle=True)

        return {'subject_id': subject_id, 'status': 'cached', 'path': cache_path}

    except Exception as e:
        return {'subject_id': subject_id, 'status': 'error', 'error': str(e)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Pre-cache DICOM data to numpy files')
    parser.add_argument('--force', action='store_true',
                        help='Force re-cache all samples (overwrite existing cache)')
    args = parser.parse_args()

    config = get_config()

    # Create cache directory
    cache_dir = os.path.join(config.data.base_path, '_numpy_cache')
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Cache directory: {cache_dir}")
    if args.force:
        print("FORCE MODE: Re-caching all samples with correct mask alignment")

    # Find all samples
    print("Finding data samples...")
    samples = find_data_paths(config.data.base_path, config)
    print(f"Found {len(samples)} samples to process")

    # Process samples in parallel
    num_workers = min(32, os.cpu_count() or 1)
    print(f"Using {num_workers} workers")

    results = {'cached': 0, 'skipped': 0, 'error': 0}
    errors = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_sample, sample, cache_dir, args.force): sample
            for sample in samples
        }

        for future in tqdm(as_completed(futures), total=len(samples), desc="Caching"):
            result = future.result()
            results[result['status']] += 1
            if result['status'] == 'error':
                errors.append(result)

    print(f"\nResults:")
    print(f"  Cached: {results['cached']}")
    print(f"  Skipped (already cached): {results['skipped']}")
    print(f"  Errors: {results['error']}")

    if errors:
        print("\nErrors:")
        for err in errors:
            print(f"  {err['subject_id']}: {err['error']}")

    print(f"\nCache saved to: {cache_dir}")
    if not args.force and results['skipped'] > 0:
        print("NOTE: Some samples were skipped. Use --force to re-cache all with correct alignment.")


if __name__ == '__main__':
    main()
