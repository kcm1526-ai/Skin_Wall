#!/usr/bin/env python3
"""
Diagnose data issues - check if masks have proper content
"""

import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import get_config


def main():
    config = get_config()
    cache_dir = os.path.join(config.data.base_path, '_numpy_cache')

    print(f"Checking cache directory: {cache_dir}")

    # Get all cached files
    npy_files = list(Path(cache_dir).glob("*.npy"))
    npz_files = list(Path(cache_dir).glob("*.npz"))

    print(f"Found {len(npy_files)} .npy files, {len(npz_files)} .npz files")

    # Check a few samples
    files_to_check = npy_files[:10] if npy_files else npz_files[:10]

    skin_stats = []
    wall_stats = []

    for f in files_to_check:
        print(f"\n{'='*60}")
        print(f"Checking: {f.name}")

        try:
            if f.suffix == '.npy':
                data = np.load(f, allow_pickle=True).item()
                image = data['image']
                skin_mask = data['skin_mask']
                wall_mask = data['abdominal_mask']
            else:
                data = np.load(f)
                image = data['image']
                skin_mask = data['skin_mask']
                wall_mask = data['abdominal_mask']

            print(f"  Image shape: {image.shape}, dtype: {image.dtype}")
            print(f"  Image range: [{image.min():.1f}, {image.max():.1f}]")

            print(f"  Skin mask shape: {skin_mask.shape}, dtype: {skin_mask.dtype}")
            print(f"  Skin mask unique values: {np.unique(skin_mask)}")
            skin_voxels = (skin_mask > 0).sum()
            skin_pct = 100 * skin_voxels / skin_mask.size
            print(f"  Skin mask non-zero voxels: {skin_voxels:,} ({skin_pct:.2f}%)")
            skin_stats.append(skin_pct)

            print(f"  Wall mask shape: {wall_mask.shape}, dtype: {wall_mask.dtype}")
            print(f"  Wall mask unique values: {np.unique(wall_mask)}")
            wall_voxels = (wall_mask > 0).sum()
            wall_pct = 100 * wall_voxels / wall_mask.size
            print(f"  Wall mask non-zero voxels: {wall_voxels:,} ({wall_pct:.2f}%)")
            wall_stats.append(wall_pct)

            # Check if shapes match
            if image.shape != skin_mask.shape:
                print(f"  ⚠️  WARNING: Image and skin mask shapes don't match!")
            if image.shape != wall_mask.shape:
                print(f"  ⚠️  WARNING: Image and wall mask shapes don't match!")

        except Exception as e:
            print(f"  ERROR loading: {e}")

    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"  Skin mask coverage: {np.mean(skin_stats):.2f}% avg (range: {np.min(skin_stats):.2f}% - {np.max(skin_stats):.2f}%)")
    print(f"  Wall mask coverage: {np.mean(wall_stats):.2f}% avg (range: {np.min(wall_stats):.2f}% - {np.max(wall_stats):.2f}%)")

    if np.mean(wall_stats) == 0:
        print("\n⚠️  CRITICAL: Wall masks appear to be EMPTY!")
        print("   This explains why Wall dice is 0.0000")
        print("   Check your original mask files (Abdominal_wall.nii.gz)")
    elif np.mean(wall_stats) < 0.1:
        print("\n⚠️  WARNING: Wall masks are very sparse!")
        print("   This might cause learning difficulties")


if __name__ == '__main__':
    main()
