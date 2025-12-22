#!/usr/bin/env python3
"""
Pre-cache DICOM data to numpy files for faster loading during training.
Run this once before training to convert all DICOM series to .npy files.

Usage:
    python scripts/precache_data.py
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import get_config
from src.dataset import DICOMLoader, MaskLoader, find_data_paths


def process_sample(sample: dict, cache_dir: str) -> dict:
    """Process a single sample and save to cache."""
    subject_id = sample['subject_id']
    cache_path = os.path.join(cache_dir, f"{subject_id}.npz")

    # Skip if already cached
    if os.path.exists(cache_path):
        return {'subject_id': subject_id, 'status': 'skipped', 'path': cache_path}

    try:
        # Load DICOM
        image, image_meta = DICOMLoader.load_dicom_series(sample['image_dir'])

        # Load masks
        skin_mask, _ = MaskLoader.load_nifti_mask(sample['skin_mask'])
        abdominal_mask, _ = MaskLoader.load_nifti_mask(sample['abdominal_wall_mask'])

        # Save as compressed numpy
        np.savez_compressed(
            cache_path,
            image=image.astype(np.float32),
            skin_mask=skin_mask.astype(np.uint8),
            abdominal_mask=abdominal_mask.astype(np.uint8),
            spacing=np.array(image_meta['spacing'])
        )

        return {'subject_id': subject_id, 'status': 'cached', 'path': cache_path}

    except Exception as e:
        return {'subject_id': subject_id, 'status': 'error', 'error': str(e)}


def main():
    config = get_config()

    # Create cache directory
    cache_dir = os.path.join(config.data.base_path, '_numpy_cache')
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Cache directory: {cache_dir}")

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
            executor.submit(process_sample, sample, cache_dir): sample
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
    print("To use cached data, set use_cache=True in your dataset config.")


if __name__ == '__main__':
    main()
