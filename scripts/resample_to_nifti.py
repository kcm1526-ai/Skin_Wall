#!/usr/bin/env python3
"""
Resample all DICOM images and masks to target voxel spacing and save as NIfTI files.

This script:
1. Loads DICOM images and NIfTI masks
2. Aligns masks to DICOM coordinate system
3. Fills wall contours
4. Resamples all to target voxel spacing using SimpleITK
5. Saves as NIfTI files

Usage:
    python scripts/resample_to_nifti.py
    python scripts/resample_to_nifti.py --output_dir /path/to/output
    python scripts/resample_to_nifti.py --spacing 1.0 1.0 1.0
    python scripts/resample_to_nifti.py --subject SUBJECT_ID  # Process single subject
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
from scipy.ndimage import binary_fill_holes
import SimpleITK as sitk
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import get_config
from src.dataset import DICOMLoader, MaskLoader, find_data_paths


def align_mask_to_image(mask: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Align NIfTI mask to DICOM image coordinate system.
    Transformation: transpose(2, 1, 0) only (no flip needed)
    """
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


def resample_sitk_fixed_spacing(img_sitk, interpolation, new_spacing):
    """
    Resample SimpleITK image to fixed spacing.
    """
    dimension = img_sitk.GetDimension()

    # Calculate new size based on new spacing
    original_size = np.array(img_sitk.GetSize())
    original_spacing = np.array(img_sitk.GetSpacing())
    new_spacing_arr = np.array(new_spacing)

    new_size = np.round(original_size * original_spacing / new_spacing_arr).astype(int)
    new_size = [int(s) for s in new_size]

    reference_direction = img_sitk.GetDirection()
    reference_origin = img_sitk.GetOrigin()

    # Create reference image
    reference_image = sitk.Image(new_size, img_sitk.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing([float(s) for s in new_spacing])
    reference_image.SetDirection(reference_direction)

    # Calculate centers
    reference_center = np.array(
        reference_image.TransformContinuousIndexToPhysicalPoint(
            np.array(reference_image.GetSize()) / 2.0
        )
    )

    # Create transform
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img_sitk.GetDirection())
    transform.SetTranslation(np.array(img_sitk.GetOrigin()) - np.array(reference_origin))

    # Centering transform
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(
        img_sitk.TransformContinuousIndexToPhysicalPoint(
            np.array(img_sitk.GetSize()) / 2.0
        )
    )
    centering_transform.SetOffset(
        np.array(transform.GetInverse().TransformPoint(img_center) - reference_center)
    )

    # Composite transform
    centered_transform = sitk.CompositeTransform([centering_transform, transform])

    # Resample
    new_img = sitk.Resample(img_sitk, reference_image, centered_transform, interpolation, 0.0)
    return new_img


def resample_volume(volume: np.ndarray, current_spacing: tuple,
                    target_spacing: tuple, is_mask: bool = False) -> np.ndarray:
    """
    Resample volume to target voxel spacing using SimpleITK.

    Args:
        volume: 3D numpy array (Z, Y, X)
        current_spacing: Current voxel spacing (X, Y, Z) in mm
        target_spacing: Target voxel spacing (X, Y, Z) in mm
        is_mask: If True, use nearest neighbor interpolation

    Returns:
        Resampled volume
    """
    # Check if resampling is needed
    current_spacing_zyx = (current_spacing[2], current_spacing[1], current_spacing[0])
    target_spacing_zyx = (target_spacing[2], target_spacing[1], target_spacing[0])
    zoom_factors = [c / t for c, t in zip(current_spacing_zyx, target_spacing_zyx)]

    if all(0.95 < z < 1.05 for z in zoom_factors):
        return volume

    # Convert numpy array to SimpleITK image
    img_sitk = sitk.GetImageFromArray(volume)
    img_sitk.SetSpacing((current_spacing[0], current_spacing[1], current_spacing[2]))

    # Select interpolation method
    if is_mask:
        interpolation = sitk.sitkNearestNeighbor
    else:
        interpolation = sitk.sitkBSpline

    # Resample using SimpleITK
    resampled_sitk = resample_sitk_fixed_spacing(img_sitk, interpolation, target_spacing)

    # Convert back to numpy array
    resampled = sitk.GetArrayFromImage(resampled_sitk)
    return resampled


def save_as_nifti(volume: np.ndarray, spacing: tuple, output_path: str):
    """
    Save numpy array as NIfTI file.

    Args:
        volume: 3D numpy array (Z, Y, X)
        spacing: Voxel spacing (X, Y, Z) in mm
        output_path: Output file path
    """
    img_sitk = sitk.GetImageFromArray(volume)
    img_sitk.SetSpacing((spacing[0], spacing[1], spacing[2]))
    sitk.WriteImage(img_sitk, output_path)


def process_subject(sample: dict, target_spacing: tuple, output_dir: str) -> dict:
    """
    Process a single subject: load, align, fill, resample, and save.

    Returns:
        dict with processing info
    """
    subject_id = sample['subject_id']
    subject_output_dir = os.path.join(output_dir, subject_id)
    os.makedirs(subject_output_dir, exist_ok=True)

    result = {
        'subject_id': subject_id,
        'success': False,
        'error': None
    }

    try:
        # Step 1: Load DICOM image
        image, image_meta = DICOMLoader.load_dicom_series(sample['image_dir'])
        current_spacing = image_meta['spacing']  # (X, Y, Z)
        original_shape = image.shape

        # Step 2: Load masks
        skin_mask_raw, _ = MaskLoader.load_nifti_mask(sample['skin_mask'])
        wall_mask_raw, _ = MaskLoader.load_nifti_mask(sample['abdominal_wall_mask'])

        # Step 3: Align masks to DICOM coordinate system
        skin_mask = align_mask_to_image(skin_mask_raw, image.shape)
        wall_mask = align_mask_to_image(wall_mask_raw, image.shape)

        # Step 4: Fill wall contour
        wall_mask_filled = fill_wall_contour(wall_mask)

        # Step 5: Resample to target spacing
        image_resampled = resample_volume(image, current_spacing, target_spacing, is_mask=False)
        skin_resampled = resample_volume(skin_mask, current_spacing, target_spacing, is_mask=True)
        wall_resampled = resample_volume(wall_mask_filled, current_spacing, target_spacing, is_mask=True)

        # Ensure masks are binary
        skin_resampled = (skin_resampled > 0.5).astype(np.uint8)
        wall_resampled = (wall_resampled > 0.5).astype(np.uint8)

        # Step 6: Save as NIfTI
        image_path = os.path.join(subject_output_dir, f"{subject_id}_image.nii.gz")
        skin_path = os.path.join(subject_output_dir, f"{subject_id}_skin.nii.gz")
        wall_path = os.path.join(subject_output_dir, f"{subject_id}_wall.nii.gz")

        save_as_nifti(image_resampled.astype(np.float32), target_spacing, image_path)
        save_as_nifti(skin_resampled, target_spacing, skin_path)
        save_as_nifti(wall_resampled, target_spacing, wall_path)

        result['success'] = True
        result['original_spacing'] = current_spacing
        result['original_shape'] = original_shape
        result['resampled_shape'] = image_resampled.shape
        result['image_path'] = image_path
        result['skin_path'] = skin_path
        result['wall_path'] = wall_path

    except Exception as e:
        result['error'] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description='Resample all data to target spacing and save as NIfTI')
    parser.add_argument('--output_dir', type=str, default='./resampled_nifti',
                        help='Output directory for NIfTI files')
    parser.add_argument('--spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help='Target spacing (X, Y, Z) in mm. Default: 1.0 1.0 1.0')
    parser.add_argument('--subject', type=str, default=None,
                        help='Process only this subject ID')
    args = parser.parse_args()

    target_spacing = tuple(args.spacing)
    output_dir = args.output_dir

    print(f"Target voxel spacing: {target_spacing} mm")
    print(f"Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all samples
    config = get_config()
    print(f"\nSearching for data in: {config.data.base_path}")
    samples = find_data_paths(config.data.base_path, config)

    if not samples:
        print("No samples found!")
        return

    print(f"Found {len(samples)} samples")

    # Filter if specific subject requested
    if args.subject:
        samples = [s for s in samples if s['subject_id'] == args.subject]
        if not samples:
            print(f"Subject {args.subject} not found!")
            return
        print(f"Processing only: {args.subject}")

    # Process all subjects
    results = []
    successful = 0
    failed = 0

    print(f"\nProcessing {len(samples)} subjects...")
    for sample in tqdm(samples, desc="Resampling"):
        result = process_subject(sample, target_spacing, output_dir)
        results.append(result)

        if result['success']:
            successful += 1
        else:
            failed += 1
            print(f"\n  Error processing {result['subject_id']}: {result['error']}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{len(samples)}")
    print(f"Failed: {failed}/{len(samples)}")
    print(f"Output directory: {output_dir}")
    print(f"Target spacing: {target_spacing} mm")

    # Show sample output info
    if successful > 0:
        success_result = next(r for r in results if r['success'])
        print(f"\nSample output ({success_result['subject_id']}):")
        print(f"  Original spacing: {success_result['original_spacing']}")
        print(f"  Original shape: {success_result['original_shape']}")
        print(f"  Resampled shape: {success_result['resampled_shape']}")

    # Save processing log
    log_path = os.path.join(output_dir, "processing_log.txt")
    with open(log_path, 'w') as f:
        f.write(f"Target spacing: {target_spacing} mm\n")
        f.write(f"Total samples: {len(samples)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n\n")

        for r in results:
            f.write(f"\n{r['subject_id']}:\n")
            if r['success']:
                f.write(f"  Status: Success\n")
                f.write(f"  Original spacing: {r['original_spacing']}\n")
                f.write(f"  Original shape: {r['original_shape']}\n")
                f.write(f"  Resampled shape: {r['resampled_shape']}\n")
            else:
                f.write(f"  Status: Failed\n")
                f.write(f"  Error: {r['error']}\n")

    print(f"\nProcessing log saved to: {log_path}")


if __name__ == '__main__':
    main()
