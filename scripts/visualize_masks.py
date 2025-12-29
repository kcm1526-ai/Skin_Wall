#!/usr/bin/env python3
"""
Interactive visualization tool for CT images with Skin and Abdominal Wall masks.

Usage:
    python scripts/visualize_masks.py /path/to/subject_folder

Example:
    python scripts/visualize_masks.py /data/01011ug_309

The script expects:
    - DICOM files in: {subject}/01_DICOM/PP/
    - Masks in: {subject}/01_DICOM/PP/Mask/Skin.nii.gz
                {subject}/01_DICOM/PP/Mask/Abdominal_wall.nii.gz
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons
import nibabel as nib
import pydicom
import SimpleITK as sitk
from pathlib import Path


def load_dicom_series(dicom_dir: str) -> np.ndarray:
    """Load DICOM series from directory (handles files without .dcm extension)"""

    # Find all potential DICOM files
    dicom_files = []
    for f in os.listdir(dicom_dir):
        file_path = os.path.join(dicom_dir, f)
        if os.path.isfile(file_path):
            # Skip known non-DICOM files
            if f.endswith(('.nii', '.nii.gz', '.json', '.txt', '.xml')):
                continue
            # Skip Mask directory
            if f == 'Mask':
                continue
            try:
                # Try to read as DICOM with force=True for files without extension
                dcm = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
                if hasattr(dcm, 'SOPClassUID') or hasattr(dcm, 'Modality'):
                    dicom_files.append(file_path)
            except:
                continue

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    print(f"Found {len(dicom_files)} DICOM files")

    # Load with pydicom (works with files without .dcm extension)
    slices = []
    for f in dicom_files:
        try:
            dcm = pydicom.dcmread(f, force=True)
            if hasattr(dcm, 'pixel_array'):
                slices.append(dcm)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")
            continue

    if not slices:
        raise ValueError("Could not load any DICOM slices with pixel data")

    print(f"Loaded {len(slices)} slices with pixel data")

    # Sort by slice location or instance number
    try:
        slices.sort(key=lambda x: float(x.SliceLocation))
        print("Sorted by SliceLocation")
    except:
        try:
            slices.sort(key=lambda x: int(x.InstanceNumber))
            print("Sorted by InstanceNumber")
        except:
            print("Warning: Could not sort slices")

    # Stack slices into volume
    volume = np.stack([s.pixel_array for s in slices], axis=0)

    # Apply rescale slope/intercept for HU values
    try:
        slope = float(slices[0].RescaleSlope)
        intercept = float(slices[0].RescaleIntercept)
        volume = volume * slope + intercept
        print(f"Applied rescale: slope={slope}, intercept={intercept}")
    except:
        pass

    return volume.astype(np.float32)


def load_nifti_mask(mask_path: str) -> np.ndarray:
    """Load NIfTI mask file"""
    if not os.path.exists(mask_path):
        print(f"Warning: Mask not found at {mask_path}")
        return None

    nii = nib.load(mask_path)
    mask = nii.get_fdata().astype(np.uint8)
    return mask


def align_mask_to_image(mask: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Align mask to image shape through transpose"""
    if mask is None:
        return None

    if mask.shape == target_shape:
        return mask

    # Check if dimensions match but in different order
    mask_dims = sorted(mask.shape)
    target_dims = sorted(target_shape)

    if mask_dims == target_dims:
        # Try different transpose orders
        for axes in [(2, 0, 1), (1, 2, 0), (0, 2, 1), (2, 1, 0), (1, 0, 2)]:
            transposed = np.transpose(mask, axes)
            if transposed.shape == target_shape:
                print(f"Transposed mask from {mask.shape} to {transposed.shape}")
                return transposed

    print(f"Warning: Could not align mask {mask.shape} to image {target_shape}")
    return None


class MaskVisualizer:
    """Interactive visualizer for CT images with mask overlays"""

    def __init__(self, image: np.ndarray, skin_mask: np.ndarray, wall_mask: np.ndarray, title: str = ""):
        self.image = image
        self.skin_mask = skin_mask
        self.wall_mask = wall_mask
        self.title = title

        # Current slice and view
        self.current_slice = image.shape[0] // 2
        self.current_view = 'axial'  # 'axial', 'coronal', 'sagittal'

        # Display options
        self.show_skin = True
        self.show_wall = True
        self.alpha = 0.4

        # Window/level for CT
        self.window_center = 40  # Soft tissue
        self.window_width = 400

        self._setup_figure()

    def _setup_figure(self):
        """Setup the matplotlib figure"""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 10))
        plt.subplots_adjust(left=0.1, bottom=0.25, right=0.75)

        # Initial display
        self._update_display()

        # Slice slider
        ax_slice = plt.axes([0.1, 0.15, 0.55, 0.03])
        self.slice_slider = Slider(
            ax_slice, 'Slice', 0, self._get_max_slice() - 1,
            valinit=self.current_slice, valstep=1
        )
        self.slice_slider.on_changed(self._on_slice_change)

        # Window center slider
        ax_wc = plt.axes([0.1, 0.10, 0.55, 0.03])
        self.wc_slider = Slider(ax_wc, 'W/L Center', -1000, 1000, valinit=self.window_center)
        self.wc_slider.on_changed(self._on_window_change)

        # Window width slider
        ax_ww = plt.axes([0.1, 0.05, 0.55, 0.03])
        self.ww_slider = Slider(ax_ww, 'W/L Width', 1, 2000, valinit=self.window_width)
        self.ww_slider.on_changed(self._on_window_change)

        # Checkboxes for mask visibility
        ax_check = plt.axes([0.78, 0.6, 0.2, 0.15])
        self.check = CheckButtons(ax_check, ['Show Skin', 'Show Wall'], [True, True])
        self.check.on_clicked(self._on_check)

        # Radio buttons for view selection
        ax_radio = plt.axes([0.78, 0.35, 0.2, 0.2])
        self.radio = RadioButtons(ax_radio, ['Axial', 'Coronal', 'Sagittal'])
        self.radio.on_clicked(self._on_view_change)

        # Alpha slider
        ax_alpha = plt.axes([0.78, 0.25, 0.15, 0.03])
        self.alpha_slider = Slider(ax_alpha, 'Alpha', 0, 1, valinit=self.alpha)
        self.alpha_slider.on_changed(self._on_alpha_change)

        # Keyboard navigation
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Add legend
        self._add_legend()

    def _get_max_slice(self):
        """Get maximum slice index for current view"""
        if self.current_view == 'axial':
            return self.image.shape[0]
        elif self.current_view == 'coronal':
            return self.image.shape[1]
        else:  # sagittal
            return self.image.shape[2]

    def _get_slice(self, volume, idx):
        """Get a slice from volume based on current view"""
        if volume is None:
            return None
        if self.current_view == 'axial':
            return volume[idx, :, :]
        elif self.current_view == 'coronal':
            return volume[:, idx, :]
        else:  # sagittal
            return volume[:, :, idx]

    def _apply_window(self, image):
        """Apply window/level to image"""
        min_val = self.window_center - self.window_width / 2
        max_val = self.window_center + self.window_width / 2
        windowed = np.clip(image, min_val, max_val)
        windowed = (windowed - min_val) / (max_val - min_val)
        return windowed

    def _update_display(self):
        """Update the display"""
        self.ax.clear()

        # Get current slice
        img_slice = self._get_slice(self.image, self.current_slice)
        skin_slice = self._get_slice(self.skin_mask, self.current_slice)
        wall_slice = self._get_slice(self.wall_mask, self.current_slice)

        # Apply window/level
        img_display = self._apply_window(img_slice)

        # Display image
        self.ax.imshow(img_display, cmap='gray', aspect='auto')

        # Overlay masks
        if self.show_skin and skin_slice is not None:
            skin_overlay = np.ma.masked_where(skin_slice == 0, skin_slice)
            self.ax.imshow(skin_overlay, cmap='Reds', alpha=self.alpha, aspect='auto', vmin=0, vmax=1)

        if self.show_wall and wall_slice is not None:
            wall_overlay = np.ma.masked_where(wall_slice == 0, wall_slice)
            self.ax.imshow(wall_overlay, cmap='Blues', alpha=self.alpha, aspect='auto', vmin=0, vmax=1)

        # Title
        view_name = self.current_view.capitalize()
        self.ax.set_title(f'{self.title}\n{view_name} - Slice {self.current_slice + 1}/{self._get_max_slice()}')
        self.ax.axis('off')

        self.fig.canvas.draw_idle()

    def _add_legend(self):
        """Add a legend"""
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.5, label='Skin'),
            Patch(facecolor='blue', alpha=0.5, label='Abdominal Wall')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')

    def _on_slice_change(self, val):
        self.current_slice = int(val)
        self._update_display()

    def _on_window_change(self, val):
        self.window_center = self.wc_slider.val
        self.window_width = self.ww_slider.val
        self._update_display()

    def _on_check(self, label):
        if label == 'Show Skin':
            self.show_skin = not self.show_skin
        elif label == 'Show Wall':
            self.show_wall = not self.show_wall
        self._update_display()

    def _on_view_change(self, label):
        self.current_view = label.lower()
        max_slice = self._get_max_slice()
        self.current_slice = min(self.current_slice, max_slice - 1)
        self.slice_slider.valmax = max_slice - 1
        self.slice_slider.set_val(self.current_slice)
        self._update_display()

    def _on_alpha_change(self, val):
        self.alpha = val
        self._update_display()

    def _on_scroll(self, event):
        """Handle scroll events for slice navigation"""
        if event.button == 'up':
            self.current_slice = min(self.current_slice + 1, self._get_max_slice() - 1)
        else:
            self.current_slice = max(self.current_slice - 1, 0)
        self.slice_slider.set_val(self.current_slice)

    def _on_key(self, event):
        """Handle keyboard events"""
        if event.key == 'up' or event.key == 'right':
            self.current_slice = min(self.current_slice + 1, self._get_max_slice() - 1)
            self.slice_slider.set_val(self.current_slice)
        elif event.key == 'down' or event.key == 'left':
            self.current_slice = max(self.current_slice - 1, 0)
            self.slice_slider.set_val(self.current_slice)
        elif event.key == 'pageup':
            self.current_slice = min(self.current_slice + 10, self._get_max_slice() - 1)
            self.slice_slider.set_val(self.current_slice)
        elif event.key == 'pagedown':
            self.current_slice = max(self.current_slice - 10, 0)
            self.slice_slider.set_val(self.current_slice)
        elif event.key == 's':
            self.show_skin = not self.show_skin
            self._update_display()
        elif event.key == 'w':
            self.show_wall = not self.show_wall
            self._update_display()
        elif event.key == '1':
            self._on_view_change('Axial')
        elif event.key == '2':
            self._on_view_change('Coronal')
        elif event.key == '3':
            self._on_view_change('Sagittal')

    def show(self):
        """Display the visualization"""
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize CT images with Skin and Wall masks')
    parser.add_argument('subject_path', type=str, help='Path to subject folder')
    parser.add_argument('--dicom-subpath', type=str, default='01_DICOM/PP',
                        help='Subpath to DICOM files (default: 01_DICOM/PP)')
    parser.add_argument('--mask-subpath', type=str, default='Mask',
                        help='Subpath to mask folder relative to DICOM path (default: Mask)')
    args = parser.parse_args()

    subject_path = args.subject_path
    subject_name = os.path.basename(subject_path.rstrip('/'))

    # Build paths
    dicom_dir = os.path.join(subject_path, args.dicom_subpath)
    mask_dir = os.path.join(dicom_dir, args.mask_subpath)

    skin_mask_path = os.path.join(mask_dir, 'Skin.nii.gz')
    wall_mask_path = os.path.join(mask_dir, 'Abdominal_wall.nii.gz')

    print(f"Subject: {subject_name}")
    print(f"DICOM dir: {dicom_dir}")
    print(f"Mask dir: {mask_dir}")

    # Load DICOM
    print("\nLoading DICOM series...")
    image = load_dicom_series(dicom_dir)
    print(f"Image shape: {image.shape}")
    print(f"Image range: [{image.min():.1f}, {image.max():.1f}]")

    # Load masks
    print("\nLoading masks...")
    skin_mask = load_nifti_mask(skin_mask_path)
    wall_mask = load_nifti_mask(wall_mask_path)

    if skin_mask is not None:
        print(f"Skin mask shape: {skin_mask.shape}, unique values: {np.unique(skin_mask)}")
    if wall_mask is not None:
        print(f"Wall mask shape: {wall_mask.shape}, unique values: {np.unique(wall_mask)}")

    # Align masks to image
    skin_mask = align_mask_to_image(skin_mask, image.shape)
    wall_mask = align_mask_to_image(wall_mask, image.shape)

    # Create visualizer
    print("\nStarting visualization...")
    print("Controls:")
    print("  - Scroll/Arrow keys: Navigate slices")
    print("  - PageUp/PageDown: Jump 10 slices")
    print("  - 's': Toggle skin mask")
    print("  - 'w': Toggle wall mask")
    print("  - '1/2/3': Switch to Axial/Coronal/Sagittal view")

    viz = MaskVisualizer(image, skin_mask, wall_mask, title=subject_name)
    viz.show()


if __name__ == '__main__':
    main()
