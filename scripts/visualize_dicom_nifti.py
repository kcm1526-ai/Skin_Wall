#!/usr/bin/env python3
"""
Visualize DICOM image with NIfTI label mask overlay.

Usage:
    python scripts/visualize_dicom_nifti.py --dicom /path/to/dicom_dir --mask /path/to/mask.nii.gz
    python scripts/visualize_dicom_nifti.py --dicom /path/to/dicom_dir --mask /path/to/mask.nii.gz --save output.png
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import SimpleITK as sitk
import nibabel as nib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_dicom(dicom_path: str) -> tuple:
    """
    Load DICOM series from directory or single file.

    Returns:
        (volume, spacing) - numpy array (Z, Y, X) and spacing (X, Y, Z)
    """
    if os.path.isdir(dicom_path):
        # Load DICOM series from directory
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(dicom_path)

        if series_ids:
            series_file_names = reader.GetGDCMSeriesFileNames(dicom_path, series_ids[0])
            reader.SetFileNames(series_file_names)
        else:
            # Fallback: find all files
            files = sorted([os.path.join(dicom_path, f) for f in os.listdir(dicom_path)
                          if os.path.isfile(os.path.join(dicom_path, f))])
            reader.SetFileNames(files)

        image = reader.Execute()
    else:
        # Single DICOM file
        image = sitk.ReadImage(dicom_path)

    volume = sitk.GetArrayFromImage(image)  # (Z, Y, X)
    spacing = image.GetSpacing()  # (X, Y, Z)

    return volume, spacing


def load_nifti(nifti_path: str) -> tuple:
    """
    Load NIfTI mask file.

    Returns:
        (mask, spacing) - numpy array and spacing
    """
    # Use SimpleITK for consistency
    image = sitk.ReadImage(nifti_path)
    mask = sitk.GetArrayFromImage(image)  # (Z, Y, X)
    spacing = image.GetSpacing()  # (X, Y, Z)

    return mask, spacing


def visualize(dicom_volume, dicom_spacing, mask_volume, mask_spacing, save_path=None):
    """
    Interactive visualization of DICOM with mask overlay.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    print(f"DICOM shape: {dicom_volume.shape}, spacing: {dicom_spacing}")
    print(f"Mask shape: {mask_volume.shape}, spacing: {mask_spacing}")

    # Check if shapes match
    if dicom_volume.shape != mask_volume.shape:
        print(f"\nWarning: Shapes don't match!")
        print(f"  DICOM: {dicom_volume.shape}")
        print(f"  Mask:  {mask_volume.shape}")
        print("  Attempting to resample mask to match DICOM...")

        # Resample mask to match DICOM
        from scipy.ndimage import zoom
        zoom_factors = [d / m for d, m in zip(dicom_volume.shape, mask_volume.shape)]
        mask_volume = zoom(mask_volume.astype(np.float32), zoom_factors, order=0)
        mask_volume = (mask_volume > 0.5).astype(np.uint8)
        print(f"  Resampled mask shape: {mask_volume.shape}")

    # Get unique labels in mask
    unique_labels = np.unique(mask_volume)
    print(f"Unique labels in mask: {unique_labels}")

    # Find slices with mask
    mask_slices = np.where(mask_volume.sum(axis=(1, 2)) > 0)[0]
    if len(mask_slices) > 0:
        init_slice = mask_slices[len(mask_slices) // 2]
        print(f"Slices with mask: {len(mask_slices)} (from {mask_slices[0]} to {mask_slices[-1]})")
    else:
        init_slice = dicom_volume.shape[0] // 2
        print("No mask voxels found!")

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.2)

    vmin, vmax = -1000, 1000  # CT HU range

    # DICOM image
    im_dicom = axes[0].imshow(dicom_volume[init_slice], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title('DICOM Image')
    axes[0].axis('off')

    # Mask
    im_mask = axes[1].imshow(mask_volume[init_slice], cmap='jet', vmin=0, vmax=max(1, unique_labels.max()))
    axes[1].set_title('NIfTI Mask')
    axes[1].axis('off')

    # Overlay
    def make_overlay(z):
        img = dicom_volume[z]
        img_norm = (img - vmin) / (vmax - vmin)
        img_norm = np.clip(img_norm, 0, 1)
        overlay = np.stack([img_norm, img_norm, img_norm], axis=-1)

        # Color overlay for mask
        mask_slice = mask_volume[z]

        # Red for label 1 (e.g., skin)
        overlay[mask_slice == 1, 0] = 1.0
        overlay[mask_slice == 1, 1] = 0.2
        overlay[mask_slice == 1, 2] = 0.2

        # Blue for label 2 (e.g., wall)
        overlay[mask_slice == 2, 0] = 0.2
        overlay[mask_slice == 2, 1] = 0.2
        overlay[mask_slice == 2, 2] = 1.0

        # Green for other labels
        overlay[mask_slice > 2, 0] = 0.2
        overlay[mask_slice > 2, 1] = 1.0
        overlay[mask_slice > 2, 2] = 0.2

        return overlay

    im_overlay = axes[2].imshow(make_overlay(init_slice))
    axes[2].set_title('Overlay (Red=1, Blue=2, Green=other)')
    axes[2].axis('off')

    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, dicom_volume.shape[0] - 1, valinit=init_slice, valstep=1)

    def update(val):
        z = int(slider.val)
        im_dicom.set_data(dicom_volume[z])
        im_mask.set_data(mask_volume[z])
        im_overlay.set_data(make_overlay(z))
        fig.canvas.draw_idle()

    slider.on_changed(update)

    if save_path:
        # Save current view
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize DICOM with NIfTI mask overlay')
    parser.add_argument('--dicom', type=str, required=True,
                        help='Path to DICOM directory or file')
    parser.add_argument('--mask', type=str, required=True,
                        help='Path to NIfTI mask file (.nii or .nii.gz)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save visualization to file (e.g., output.png)')
    args = parser.parse_args()

    # Check paths
    if not os.path.exists(args.dicom):
        print(f"Error: DICOM path not found: {args.dicom}")
        return

    if not os.path.exists(args.mask):
        print(f"Error: Mask file not found: {args.mask}")
        return

    print(f"Loading DICOM from: {args.dicom}")
    dicom_volume, dicom_spacing = load_dicom(args.dicom)

    print(f"Loading mask from: {args.mask}")
    mask_volume, mask_spacing = load_nifti(args.mask)

    print("\nVisualizing...")
    visualize(dicom_volume, dicom_spacing, mask_volume, mask_spacing, args.save)


if __name__ == '__main__':
    main()
