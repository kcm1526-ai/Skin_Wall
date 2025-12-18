"""
Utility functions for medical image segmentation
"""

import os
import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def get_lr(optimizer) -> float:
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


class EarlyStopping:
    """Early stopping handler"""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def compute_surface_dice(
    pred: np.ndarray,
    target: np.ndarray,
    tolerance: float = 2.0,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
) -> float:
    """
    Compute Surface Dice (Normalized Surface Distance)

    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        tolerance: Tolerance in mm for surface matching
        spacing: Voxel spacing

    Returns:
        Surface Dice score
    """
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0

    # Compute distance transforms
    pred_boundary = get_surface(pred)
    target_boundary = get_surface(target)

    # Distance from pred surface to target
    dist_pred_to_target = distance_transform_edt(~target, sampling=spacing)
    dist_target_to_pred = distance_transform_edt(~pred, sampling=spacing)

    # Get distances at boundary points
    pred_to_target_dist = dist_pred_to_target[pred_boundary]
    target_to_pred_dist = dist_target_to_pred[target_boundary]

    # Count points within tolerance
    pred_match = (pred_to_target_dist <= tolerance).sum()
    target_match = (target_to_pred_dist <= tolerance).sum()

    # Surface Dice
    total_points = pred_boundary.sum() + target_boundary.sum()
    surface_dice = (pred_match + target_match) / total_points

    return surface_dice


def get_surface(mask: np.ndarray) -> np.ndarray:
    """Extract surface voxels from binary mask"""
    eroded = ndimage.binary_erosion(mask)
    surface = mask & ~eroded
    return surface


def compute_hausdorff_distance(
    pred: np.ndarray,
    target: np.ndarray,
    percentile: float = 95.0,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
) -> float:
    """
    Compute Hausdorff Distance (or percentile variant)

    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        percentile: Percentile for robust HD (e.g., 95)
        spacing: Voxel spacing

    Returns:
        Hausdorff distance in mm
    """
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return float('inf')

    # Get surface points
    pred_surface = get_surface(pred)
    target_surface = get_surface(target)

    # Compute distance transforms
    dist_pred = distance_transform_edt(~pred, sampling=spacing)
    dist_target = distance_transform_edt(~target, sampling=spacing)

    # Get distances at surface points
    dist_pred_to_target = dist_target[pred_surface]
    dist_target_to_pred = dist_pred[target_surface]

    if percentile == 100:
        # Maximum (standard Hausdorff)
        return max(dist_pred_to_target.max(), dist_target_to_pred.max())
    else:
        # Percentile (robust Hausdorff)
        return max(
            np.percentile(dist_pred_to_target, percentile),
            np.percentile(dist_target_to_pred, percentile)
        )


def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 3,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute all segmentation metrics

    Args:
        pred: Predicted segmentation mask
        target: Ground truth mask
        num_classes: Number of classes
        spacing: Voxel spacing
        class_names: Optional class names

    Returns:
        Dictionary of metrics
    """
    if class_names is None:
        class_names = [f'class_{i}' for i in range(num_classes)]

    metrics = {}

    for c in range(1, num_classes):  # Skip background
        pred_c = (pred == c)
        target_c = (target == c)

        # Dice
        intersection = (pred_c & target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = 2.0 * intersection / (union + 1e-8)
        metrics[f'{class_names[c]}_dice'] = dice

        # IoU / Jaccard
        iou = intersection / (pred_c.sum() + target_c.sum() - intersection + 1e-8)
        metrics[f'{class_names[c]}_iou'] = iou

        # Precision and Recall
        precision = intersection / (pred_c.sum() + 1e-8)
        recall = intersection / (target_c.sum() + 1e-8)
        metrics[f'{class_names[c]}_precision'] = precision
        metrics[f'{class_names[c]}_recall'] = recall

        # F1 Score
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        metrics[f'{class_names[c]}_f1'] = f1

        # Volume difference
        vol_diff = abs(pred_c.sum() - target_c.sum()) / (target_c.sum() + 1e-8)
        metrics[f'{class_names[c]}_volume_diff'] = vol_diff

        # Surface Dice
        if pred_c.sum() > 0 or target_c.sum() > 0:
            surface_dice = compute_surface_dice(pred_c, target_c, spacing=spacing)
            metrics[f'{class_names[c]}_surface_dice'] = surface_dice

            # Hausdorff Distance (95%)
            hd95 = compute_hausdorff_distance(pred_c, target_c, percentile=95, spacing=spacing)
            metrics[f'{class_names[c]}_hd95'] = hd95

    # Average metrics
    dice_values = [v for k, v in metrics.items() if 'dice' in k and 'surface' not in k]
    metrics['mean_dice'] = np.mean(dice_values) if dice_values else 0.0

    return metrics


def visualize_prediction(
    image: np.ndarray,
    pred: np.ndarray,
    target: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Visualize prediction overlaid on image

    Args:
        image: Input image (D, H, W)
        pred: Predicted segmentation (D, H, W)
        target: Optional ground truth (D, H, W)
        slice_idx: Slice index to visualize (middle if None)
        class_names: Class names for legend
        save_path: Optional path to save figure
        figsize: Figure size
    """
    if slice_idx is None:
        slice_idx = image.shape[0] // 2

    if class_names is None:
        class_names = ['Background', 'Skin', 'Abdominal_Wall']

    # Define colors (RGBA)
    colors = [
        [0, 0, 0, 0],      # Background (transparent)
        [1, 0, 0, 0.5],    # Skin (red)
        [0, 0, 1, 0.5],    # Abdominal wall (blue)
    ]

    # Create figure
    if target is not None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes = [axes[0], axes[1]]

    # Normalize image for display
    img_slice = image[slice_idx]
    img_normalized = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

    # Plot image
    axes[0].imshow(img_normalized, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Plot prediction
    axes[1].imshow(img_normalized, cmap='gray')
    for c in range(1, len(class_names)):
        mask = (pred[slice_idx] == c)
        if mask.any():
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask] = colors[c]
            axes[1].imshow(colored_mask)
    axes[1].set_title('Prediction')
    axes[1].axis('off')

    # Plot ground truth if provided
    if target is not None:
        axes[2].imshow(img_normalized, cmap='gray')
        for c in range(1, len(class_names)):
            mask = (target[slice_idx] == c)
            if mask.any():
                colored_mask = np.zeros((*mask.shape, 4))
                colored_mask[mask] = colors[c]
                axes[2].imshow(colored_mask)
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')

    # Add legend
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, fc=colors[i][:3])
        for i in range(1, len(class_names))
    ]
    fig.legend(legend_patches, class_names[1:], loc='lower center', ncol=len(class_names) - 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_3d_visualization(
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    save_path: Optional[str] = None
):
    """
    Create 3D surface visualization using marching cubes

    Args:
        mask: Binary or multi-class segmentation mask
        spacing: Voxel spacing
        save_path: Optional path to save figure
    """
    try:
        from skimage import measure
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        colors = ['red', 'blue', 'green', 'yellow']
        unique_labels = np.unique(mask)

        for i, label in enumerate(unique_labels):
            if label == 0:
                continue

            binary_mask = (mask == label)

            # Skip if mask is empty or too small
            if binary_mask.sum() < 10:
                continue

            try:
                # Generate surface mesh
                verts, faces, _, _ = measure.marching_cubes(
                    binary_mask.astype(float),
                    level=0.5,
                    spacing=spacing
                )

                # Create mesh
                mesh = Poly3DCollection(verts[faces])
                mesh.set_facecolor(colors[i % len(colors)])
                mesh.set_alpha(0.5)
                ax.add_collection3d(mesh)
            except:
                continue

        # Set axis limits
        ax.set_xlim(0, mask.shape[0] * spacing[0])
        ax.set_ylim(0, mask.shape[1] * spacing[1])
        ax.set_zlim(0, mask.shape[2] * spacing[2])
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    except ImportError:
        print("Warning: skimage not available for 3D visualization")


def resample_volume(
    volume: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    is_mask: bool = False
) -> np.ndarray:
    """
    Resample volume to target spacing

    Args:
        volume: Input volume (D, H, W)
        original_spacing: Original voxel spacing
        target_spacing: Target voxel spacing
        is_mask: Whether the volume is a segmentation mask

    Returns:
        Resampled volume
    """
    # Calculate new shape
    resize_factor = np.array(original_spacing) / np.array(target_spacing)
    new_shape = np.round(volume.shape * resize_factor).astype(int)

    # Resample
    if is_mask:
        # Use nearest neighbor for masks
        resampled = ndimage.zoom(volume, resize_factor, order=0)
    else:
        # Use cubic interpolation for images
        resampled = ndimage.zoom(volume, resize_factor, order=3)

    return resampled


def crop_or_pad_volume(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int],
    pad_value: float = 0.0
) -> np.ndarray:
    """
    Crop or pad volume to target shape (center crop/pad)

    Args:
        volume: Input volume
        target_shape: Target shape
        pad_value: Value to use for padding

    Returns:
        Cropped/padded volume
    """
    current_shape = volume.shape

    # Calculate padding/cropping for each dimension
    result = volume

    for dim in range(3):
        diff = target_shape[dim] - current_shape[dim]

        if diff > 0:
            # Pad
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_width = [(0, 0)] * 3
            pad_width[dim] = (pad_before, pad_after)
            result = np.pad(result, pad_width, mode='constant', constant_values=pad_value)
        elif diff < 0:
            # Crop
            crop_before = abs(diff) // 2
            crop_after = crop_before + target_shape[dim]
            slices = [slice(None)] * 3
            slices[dim] = slice(crop_before, crop_after)
            result = result[tuple(slices)]

    return result


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name: str = ''):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f'{self.name}: {self.avg:.4f}'


if __name__ == '__main__':
    # Test utilities
    print("Testing utility functions...")

    # Test metrics
    pred = np.zeros((64, 64, 64), dtype=np.uint8)
    target = np.zeros((64, 64, 64), dtype=np.uint8)

    # Create some test regions
    pred[20:40, 20:40, 20:40] = 1
    pred[30:50, 30:50, 30:50] = 2

    target[22:42, 22:42, 22:42] = 1
    target[32:52, 32:52, 32:52] = 2

    metrics = compute_all_metrics(pred, target, num_classes=3)
    print("Metrics:", metrics)
