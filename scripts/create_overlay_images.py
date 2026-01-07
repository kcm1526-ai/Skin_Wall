#!/usr/bin/env python3
"""
Create overlay images from preprocessed NIfTI files.

This script:
1. Reads preprocessed NIfTI files from a directory
2. Overlays skin (green) and wall (red) masks on CT image
3. Saves overlay images to ./overlays folder

Usage:
    python scripts/create_overlay_images.py
    python scripts/create_overlay_images.py --input_dir /path/to/resampled --output_dir ./overlays
    python scripts/create_overlay_images.py --slice_mode middle  # or 'max_mask'
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_nifti(nifti_path: str) -> np.ndarray:
    """Load NIfTI file and return numpy array (Z, Y, X)."""
    image = sitk.ReadImage(nifti_path)
    return sitk.GetArrayFromImage(image)


def normalize_ct(image: np.ndarray, window_center: float = 40, window_width: float = 400) -> np.ndarray:
    """
    Normalize CT image using windowing.
    Default: soft tissue window (center=40, width=400)
    """
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    image = np.clip(image, min_val, max_val)
    image = (image - min_val) / (max_val - min_val)
    return image


def create_overlay(ct_slice: np.ndarray, skin_slice: np.ndarray, wall_slice: np.ndarray,
                   skin_color: tuple = (0.2, 0.8, 0.2),  # Green
                   wall_color: tuple = (0.9, 0.2, 0.2),  # Red
                   skin_alpha: float = 0.4,
                   wall_alpha: float = 0.6) -> np.ndarray:
    """
    Create RGB overlay image with masks on CT.

    Wall mask is rendered on top (most front), then skin mask.
    """
    # Normalize CT to 0-1
    ct_norm = normalize_ct(ct_slice)

    # Create RGB image from grayscale CT
    rgb = np.stack([ct_norm, ct_norm, ct_norm], axis=-1)

    # Apply skin mask (green) - rendered first (behind wall)
    skin_mask = skin_slice > 0
    if skin_mask.any():
        for c in range(3):
            rgb[skin_mask, c] = (1 - skin_alpha) * rgb[skin_mask, c] + skin_alpha * skin_color[c]

    # Apply wall mask (red) - rendered on top (most front)
    wall_mask = wall_slice > 0
    if wall_mask.any():
        for c in range(3):
            rgb[wall_mask, c] = (1 - wall_alpha) * rgb[wall_mask, c] + wall_alpha * wall_color[c]

    return np.clip(rgb, 0, 1)


def find_best_slice(skin_mask: np.ndarray, wall_mask: np.ndarray, mode: str = 'max_mask') -> int:
    """
    Find the best slice to visualize.

    Args:
        skin_mask: 3D skin mask (Z, Y, X)
        wall_mask: 3D wall mask (Z, Y, X)
        mode: 'middle' for middle slice, 'max_mask' for slice with most mask voxels

    Returns:
        Best slice index
    """
    num_slices = skin_mask.shape[0]

    if mode == 'middle':
        # Find middle slice among slices with masks
        mask_slices = np.where((skin_mask.sum(axis=(1, 2)) > 0) | (wall_mask.sum(axis=(1, 2)) > 0))[0]
        if len(mask_slices) > 0:
            return mask_slices[len(mask_slices) // 2]
        return num_slices // 2

    elif mode == 'max_mask':
        # Find slice with maximum mask coverage
        combined_mask = (skin_mask > 0).astype(int) + (wall_mask > 0).astype(int)
        slice_sums = combined_mask.sum(axis=(1, 2))
        return int(np.argmax(slice_sums))

    else:
        return num_slices // 2


def process_folder(folder_path: str, output_dir: str, slice_mode: str = 'max_mask') -> dict:
    """
    Process a single folder and create overlay image.

    Returns:
        dict with processing info
    """
    folder_name = os.path.basename(folder_path)
    result = {
        'folder': folder_name,
        'success': False,
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
            result['error'] = f"Missing files: image={image_file is not None}, skin={skin_file is not None}, wall={wall_file is not None}"
            return result

        # Load files
        image = load_nifti(image_file)
        skin_mask = load_nifti(skin_file)
        wall_mask = load_nifti(wall_file)

        # Find best slice
        best_slice = find_best_slice(skin_mask, wall_mask, mode=slice_mode)

        # Create overlay
        overlay = create_overlay(
            image[best_slice],
            skin_mask[best_slice],
            wall_mask[best_slice]
        )

        # Save overlay image
        output_path = os.path.join(output_dir, f"{folder_name}_overlay.png")

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(overlay)
        ax.set_title(f"{folder_name} (slice {best_slice})\nGreen=Skin, Red=Wall", fontsize=12)
        ax.axis('off')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=(0.2, 0.8, 0.2), alpha=0.6, label='Skin'),
            Patch(facecolor=(0.9, 0.2, 0.2), alpha=0.6, label='Wall')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        result['success'] = True
        result['output_path'] = output_path
        result['slice'] = best_slice
        result['image_shape'] = image.shape
        result['skin_voxels'] = int((skin_mask > 0).sum())
        result['wall_voxels'] = int((wall_mask > 0).sum())

    except Exception as e:
        result['error'] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description='Create overlay images from preprocessed NIfTI files')
    parser.add_argument('--input_dir', type=str,
                        default='/raid/users/ai_kcm_0/skin_resampled',
                        help='Input directory containing preprocessed folders')
    parser.add_argument('--output_dir', type=str, default='./overlays',
                        help='Output directory for overlay images')
    parser.add_argument('--slice_mode', type=str, default='max_mask',
                        choices=['middle', 'max_mask'],
                        help='How to select slice: middle or max_mask (default)')
    parser.add_argument('--subject', type=str, default=None,
                        help='Process only this subject folder')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Slice mode: {args.slice_mode}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all subject folders
    if args.subject:
        folders = [os.path.join(input_dir, args.subject)]
        if not os.path.exists(folders[0]):
            print(f"Subject folder not found: {folders[0]}")
            return
    else:
        folders = sorted([f for f in glob(os.path.join(input_dir, '*')) if os.path.isdir(f)])

    if not folders:
        print(f"No folders found in {input_dir}")
        return

    print(f"Found {len(folders)} folders to process")

    # Process all folders
    results = []
    successful = 0
    failed = 0

    for folder in tqdm(folders, desc="Creating overlays"):
        result = process_folder(folder, output_dir, args.slice_mode)
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
    print(f"Output directory: {output_dir}")

    # Show sample info
    if successful > 0:
        success_result = next(r for r in results if r['success'])
        print(f"\nSample ({success_result['folder']}):")
        print(f"  Image shape: {success_result['image_shape']}")
        print(f"  Slice used: {success_result['slice']}")
        print(f"  Skin voxels: {success_result['skin_voxels']:,}")
        print(f"  Wall voxels: {success_result['wall_voxels']:,}")

    # Save log
    log_path = os.path.join(output_dir, "overlay_log.txt")
    with open(log_path, 'w') as f:
        f.write(f"Input: {input_dir}\n")
        f.write(f"Slice mode: {args.slice_mode}\n")
        f.write(f"Total: {len(folders)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n\n")

        for r in results:
            f.write(f"\n{r['folder']}:\n")
            if r['success']:
                f.write(f"  Status: Success\n")
                f.write(f"  Output: {r['output_path']}\n")
                f.write(f"  Slice: {r['slice']}\n")
            else:
                f.write(f"  Status: Failed\n")
                f.write(f"  Error: {r['error']}\n")

    print(f"\nLog saved to: {log_path}")


if __name__ == '__main__':
    main()
