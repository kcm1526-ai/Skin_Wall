#!/usr/bin/env python3
"""
Diagnose mask alignment issues between DICOM images and NIfTI masks.
Tests different flip/transpose combinations to find the correct alignment.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pydicom


def load_dicom_series(dicom_dir: str) -> np.ndarray:
    """Load DICOM series"""
    dicom_files = []
    for f in os.listdir(dicom_dir):
        file_path = os.path.join(dicom_dir, f)
        if os.path.isfile(file_path):
            if f.endswith(('.nii', '.nii.gz', '.json', '.txt', '.xml')):
                continue
            if f == 'Mask':
                continue
            try:
                dcm = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
                if hasattr(dcm, 'SOPClassUID') or hasattr(dcm, 'Modality'):
                    dicom_files.append(file_path)
            except:
                continue

    print(f"Found {len(dicom_files)} DICOM files")

    slices = []
    for f in dicom_files:
        try:
            dcm = pydicom.dcmread(f, force=True)
            if hasattr(dcm, 'pixel_array'):
                slices.append(dcm)
        except:
            continue

    try:
        slices.sort(key=lambda x: float(x.SliceLocation))
    except:
        try:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        except:
            pass

    volume = np.stack([s.pixel_array for s in slices], axis=0)

    try:
        slope = float(slices[0].RescaleSlope)
        intercept = float(slices[0].RescaleIntercept)
        volume = volume * slope + intercept
    except:
        pass

    return volume.astype(np.float32)


def load_nifti_mask(mask_path: str) -> tuple:
    """Load NIfTI mask and return with affine"""
    nii = nib.load(mask_path)
    mask = nii.get_fdata().astype(np.uint8)
    affine = nii.affine
    return mask, affine


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
                            'mask': flipped
                        })

    return orientations


def compute_alignment_score(image: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute a score for how well the mask aligns with the image.
    Good alignment: mask should be on body tissue, not air (low HU values).
    """
    # Get masked region
    masked_values = image[mask > 0]

    if len(masked_values) == 0:
        return -1000  # No mask

    # Good alignment: mask should cover tissue (HU > -500), not air (HU ~ -1000)
    # Score = mean HU of masked region (higher is better, means more tissue)
    return np.mean(masked_values)


def visualize_orientations(image: np.ndarray, orientations: list, slice_idx: int = None):
    """Visualize different orientations to find the best one"""
    if slice_idx is None:
        slice_idx = image.shape[0] // 2

    # Compute scores for each orientation
    scores = []
    for ori in orientations:
        score = compute_alignment_score(image, ori['mask'])
        scores.append(score)
        ori['score'] = score

    # Sort by score (higher is better)
    sorted_orientations = sorted(orientations, key=lambda x: x['score'], reverse=True)

    # Show top 8 orientations
    n_show = min(8, len(sorted_orientations))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # Window the image for display
    img_slice = image[slice_idx]
    img_display = np.clip(img_slice, -200, 400)
    img_display = (img_display + 200) / 600

    for i, ori in enumerate(sorted_orientations[:n_show]):
        ax = axes[i]
        mask_slice = ori['mask'][slice_idx]

        ax.imshow(img_display, cmap='gray')
        mask_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
        ax.imshow(mask_overlay, cmap='Reds', alpha=0.5, vmin=0, vmax=1)

        title = f"T:{ori['transpose']} F:{ori['flip']}\nScore: {ori['score']:.1f}"
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('alignment_comparison.png', dpi=150)
    print(f"\nSaved comparison to alignment_comparison.png")

    # Print best orientation
    best = sorted_orientations[0]
    print(f"\n=== BEST ORIENTATION ===")
    print(f"Transpose order: {best['transpose']}")
    print(f"Flip (Z, Y, X): {best['flip']}")
    print(f"Score: {best['score']:.1f}")

    return sorted_orientations


def main():
    parser = argparse.ArgumentParser(description='Diagnose mask alignment')
    parser.add_argument('subject_path', type=str, help='Path to subject folder')
    parser.add_argument('--dicom-subpath', type=str, default='01_DICOM/PP')
    parser.add_argument('--mask-subpath', type=str, default='Mask')
    parser.add_argument('--mask-name', type=str, default='Skin.nii.gz',
                        help='Mask filename (Skin.nii.gz or Abdominal_wall.nii.gz)')
    parser.add_argument('--slice', type=int, default=None, help='Slice index to visualize')
    args = parser.parse_args()

    dicom_dir = os.path.join(args.subject_path, args.dicom_subpath)
    mask_path = os.path.join(dicom_dir, args.mask_subpath, args.mask_name)

    print(f"Loading DICOM from: {dicom_dir}")
    image = load_dicom_series(dicom_dir)
    print(f"Image shape: {image.shape}")
    print(f"Image range: [{image.min():.1f}, {image.max():.1f}]")

    print(f"\nLoading mask from: {mask_path}")
    mask, affine = load_nifti_mask(mask_path)
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {np.unique(mask)}")
    print(f"Affine:\n{affine}")

    print(f"\nTrying all orientations...")
    orientations = try_all_orientations(mask, image.shape)
    print(f"Found {len(orientations)} valid orientations")

    if len(orientations) == 0:
        print("ERROR: No valid orientation found. Dimensions don't match.")
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        return

    slice_idx = args.slice if args.slice else image.shape[0] // 2
    sorted_oris = visualize_orientations(image, orientations, slice_idx)

    plt.show()


if __name__ == '__main__':
    main()
