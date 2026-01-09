#!/usr/bin/env python3
"""
Flip skin and wall masks in Z-axis, with exclusions for specific indices.

This script:
1. Reads preprocessed NIfTI files from input directory
2. Flips skin and wall masks in Z-axis (except for excluded indices)
3. Saves to output directory

Excluded indices (no flip applied):
- 1, 90, 368, 541~741

Usage:
    python scripts/flip_masks_z_axis.py
    python scripts/flip_masks_z_axis.py --input_dir /path/to/input --output_dir /path/to/output
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
import re
import shutil


# Indices to EXCLUDE from flipping (these are already correct)
EXCLUDED_INDICES = {1, 90, 368}
EXCLUDED_RANGE = (541, 741)  # inclusive range


def is_excluded(index: int) -> bool:
    """Check if index should be excluded from flipping."""
    if index in EXCLUDED_INDICES:
        return True
    if EXCLUDED_RANGE[0] <= index <= EXCLUDED_RANGE[1]:
        return True
    return False


def extract_index(folder_name: str) -> int:
    """
    Extract index from folder name.
    Example: '01011ug_446' -> 446
    """
    match = re.search(r'_(\d+)$', folder_name)
    if match:
        return int(match.group(1))
    return None


def load_nifti(path: str):
    """Load NIfTI file and return SimpleITK image."""
    return sitk.ReadImage(path)


def save_nifti(image: sitk.Image, path: str):
    """Save SimpleITK image to NIfTI file."""
    sitk.WriteImage(image, path)


def flip_z_axis(image: sitk.Image) -> sitk.Image:
    """Flip SimpleITK image along Z-axis."""
    # Get array, flip, and create new image
    array = sitk.GetArrayFromImage(image)  # (Z, Y, X)
    flipped_array = np.flip(array, axis=0)  # Flip Z-axis
    flipped_array = np.ascontiguousarray(flipped_array)

    # Create new image with same metadata
    flipped_image = sitk.GetImageFromArray(flipped_array)
    flipped_image.SetSpacing(image.GetSpacing())
    flipped_image.SetOrigin(image.GetOrigin())
    flipped_image.SetDirection(image.GetDirection())

    return flipped_image


def process_folder(folder_path: str, output_dir: str, flip_masks: bool) -> dict:
    """
    Process a single folder.

    Args:
        folder_path: Input folder path
        output_dir: Output directory
        flip_masks: Whether to flip the masks

    Returns:
        dict with processing info
    """
    folder_name = os.path.basename(folder_path)
    subject_output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(subject_output_dir, exist_ok=True)

    result = {
        'folder': folder_name,
        'success': False,
        'flipped': flip_masks,
        'error': None
    }

    try:
        # Find files in folder
        nii_files = glob(os.path.join(folder_path, '*.nii.gz'))

        image_file = None
        skin_file = None
        wall_file = None

        for f in nii_files:
            basename = os.path.basename(f)
            if '_image.nii.gz' in basename:
                image_file = f
            elif '_skin.nii.gz' in basename:
                skin_file = f
            elif '_wall.nii.gz' in basename:
                wall_file = f

        if not all([image_file, skin_file, wall_file]):
            result['error'] = f"Missing files"
            return result

        # Output paths
        image_out = os.path.join(subject_output_dir, os.path.basename(image_file))
        skin_out = os.path.join(subject_output_dir, os.path.basename(skin_file))
        wall_out = os.path.join(subject_output_dir, os.path.basename(wall_file))

        # Copy image as-is (no flip for image)
        shutil.copy2(image_file, image_out)

        if flip_masks:
            # Load and flip masks
            skin_img = load_nifti(skin_file)
            wall_img = load_nifti(wall_file)

            skin_flipped = flip_z_axis(skin_img)
            wall_flipped = flip_z_axis(wall_img)

            save_nifti(skin_flipped, skin_out)
            save_nifti(wall_flipped, wall_out)
        else:
            # Copy masks as-is (excluded from flipping)
            shutil.copy2(skin_file, skin_out)
            shutil.copy2(wall_file, wall_out)

        result['success'] = True

    except Exception as e:
        result['error'] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description='Flip masks in Z-axis with exclusions')
    parser.add_argument('--input_dir', type=str,
                        default='/raid/users/ai_kcm_0/skin_resampled',
                        help='Input directory containing preprocessed folders')
    parser.add_argument('--output_dir', type=str,
                        default='/raid/users/ai_kcm_0/skin_wall_flipped',
                        help='Output directory for flipped masks')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be done without actually processing')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"\nExcluded indices (no flip):")
    print(f"  Individual: {sorted(EXCLUDED_INDICES)}")
    print(f"  Range: {EXCLUDED_RANGE[0]} ~ {EXCLUDED_RANGE[1]}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all subject folders
    folders = sorted([f for f in glob(os.path.join(input_dir, '*')) if os.path.isdir(f)])

    if not folders:
        print(f"No folders found in {input_dir}")
        return

    print(f"\nFound {len(folders)} folders")

    # Categorize folders
    to_flip = []
    excluded = []

    for folder in folders:
        folder_name = os.path.basename(folder)
        index = extract_index(folder_name)

        if index is None:
            print(f"  Warning: Could not extract index from {folder_name}, will flip")
            to_flip.append((folder, True))
        elif is_excluded(index):
            excluded.append((folder, index))
            to_flip.append((folder, False))
        else:
            to_flip.append((folder, True))

    # Count
    flip_count = sum(1 for _, flip in to_flip if flip)
    exclude_count = len(excluded)

    print(f"\nWill flip masks: {flip_count}")
    print(f"Excluded (no flip): {exclude_count}")

    if excluded:
        print(f"\nExcluded folders:")
        for folder, idx in excluded[:10]:
            print(f"  {os.path.basename(folder)} (index={idx})")
        if len(excluded) > 10:
            print(f"  ... and {len(excluded) - 10} more")

    if args.dry_run:
        print("\n[DRY RUN] No files were processed.")
        return

    # Process all folders
    print(f"\nProcessing {len(folders)} folders...")

    results = []
    successful = 0
    failed = 0

    for folder, should_flip in tqdm(to_flip, desc="Processing"):
        result = process_folder(folder, output_dir, should_flip)
        results.append(result)

        if result['success']:
            successful += 1
        else:
            failed += 1
            tqdm.write(f"  Error: {result['folder']}: {result['error']}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{len(folders)}")
    print(f"Failed: {failed}/{len(folders)}")
    print(f"Flipped: {sum(1 for r in results if r['success'] and r['flipped'])}")
    print(f"Not flipped (excluded): {sum(1 for r in results if r['success'] and not r['flipped'])}")
    print(f"Output directory: {output_dir}")

    # Save log
    log_path = os.path.join(output_dir, "flip_log.txt")
    with open(log_path, 'w') as f:
        f.write(f"Input: {input_dir}\n")
        f.write(f"Output: {output_dir}\n")
        f.write(f"Total folders: {len(folders)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"\nExcluded indices: {sorted(EXCLUDED_INDICES)}\n")
        f.write(f"Excluded range: {EXCLUDED_RANGE[0]} ~ {EXCLUDED_RANGE[1]}\n")
        f.write(f"\n{'='*40}\n")

        f.write("\nFLIPPED:\n")
        for r in results:
            if r['success'] and r['flipped']:
                f.write(f"  {r['folder']}\n")

        f.write("\nNOT FLIPPED (excluded):\n")
        for r in results:
            if r['success'] and not r['flipped']:
                f.write(f"  {r['folder']}\n")

        f.write("\nFAILED:\n")
        for r in results:
            if not r['success']:
                f.write(f"  {r['folder']}: {r['error']}\n")

    print(f"\nLog saved to: {log_path}")


if __name__ == '__main__':
    main()
