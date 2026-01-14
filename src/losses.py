"""
Loss functions for medical image segmentation
Includes Dice Loss, Focal Loss, and combined losses with deep supervision support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation

    Args:
        include_background: Whether to include background class in loss
        softmax: Apply softmax to predictions
        smooth: Smoothing factor to avoid division by zero
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        include_background: bool = True,
        softmax: bool = True,
        smooth: float = 1e-5,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.include_background = include_background
        self.softmax = softmax
        self.smooth = smooth
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions of shape (B, C, D, H, W)
            target: Ground truth of shape (B, 1, D, H, W) or (B, D, H, W)

        Returns:
            Dice loss
        """
        if self.softmax:
            pred = F.softmax(pred, dim=1)

        # Handle target shape
        if target.dim() == 4:
            target = target.unsqueeze(1)

        # One-hot encode target
        num_classes = pred.shape[1]
        target_one_hot = torch.zeros_like(pred)

        # Ensure target is within valid range
        target_clamped = target.long().clamp(0, num_classes - 1)
        target_one_hot.scatter_(1, target_clamped, 1)

        # Optionally exclude background
        if not self.include_background:
            pred = pred[:, 1:]
            target_one_hot = target_one_hot[:, 1:]

        # Flatten spatial dimensions
        pred_flat = pred.flatten(2)  # (B, C, N)
        target_flat = target_one_hot.flatten(2)  # (B, C, N)

        # Calculate Dice per class
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice

        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class GeneralizedDiceLoss(nn.Module):
    """
    Generalized Dice Loss for handling class imbalance

    Reference: Sudre, C.H., et al. "Generalised Dice overlap as a deep learning
    loss function for highly unbalanced segmentations." DLMIA 2017.
    """

    def __init__(
        self,
        include_background: bool = True,
        softmax: bool = True,
        smooth: float = 1e-5
    ):
        super().__init__()
        self.include_background = include_background
        self.softmax = softmax
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        if self.softmax:
            pred = F.softmax(pred, dim=1)

        if target.dim() == 4:
            target = target.unsqueeze(1)

        num_classes = pred.shape[1]
        target_one_hot = torch.zeros_like(pred)
        target_clamped = target.long().clamp(0, num_classes - 1)
        target_one_hot.scatter_(1, target_clamped, 1)

        if not self.include_background:
            pred = pred[:, 1:]
            target_one_hot = target_one_hot[:, 1:]

        pred_flat = pred.flatten(2)
        target_flat = target_one_hot.flatten(2)

        # Calculate class weights (inverse of class frequency)
        class_weights = target_flat.sum(dim=2) + self.smooth
        class_weights = 1.0 / (class_weights ** 2)

        intersection = (pred_flat * target_flat).sum(dim=2) * class_weights
        union = (pred_flat + target_flat).sum(dim=2) * class_weights

        gdc = (2.0 * intersection.sum(dim=1) + self.smooth) / (union.sum(dim=1) + self.smooth)

        return (1.0 - gdc).mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance

    Reference: Lin, T.Y., et al. "Focal loss for dense object detection." ICCV 2017.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, List[float]]] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions of shape (B, C, D, H, W)
            target: Ground truth of shape (B, 1, D, H, W) or (B, D, H, W)
        """
        if target.dim() == 5:
            target = target.squeeze(1)

        num_classes = pred.shape[1]
        target = target.long().clamp(0, num_classes - 1)

        # Cross-entropy
        ce_loss = F.cross_entropy(pred, target, reduction='none')

        # Get predictions for target class
        pred_softmax = F.softmax(pred, dim=1)
        pt = pred_softmax.gather(1, target.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting if specified
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha_tensor = torch.tensor(self.alpha, device=pred.device, dtype=pred.dtype)
                alpha_weight = alpha_tensor.gather(0, target.flatten()).view_as(target)
            else:
                alpha_weight = self.alpha
            focal_weight = focal_weight * alpha_weight

        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss for handling false positives and false negatives differently

    Reference: Salehi, S.S.M., et al. "Tversky loss function for image segmentation
    using 3D fully convolutional deep networks." MLMI 2017.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1e-5,
        include_background: bool = True
    ):
        super().__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.smooth = smooth
        self.include_background = include_background

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        pred = F.softmax(pred, dim=1)

        if target.dim() == 4:
            target = target.unsqueeze(1)

        num_classes = pred.shape[1]
        target_one_hot = torch.zeros_like(pred)
        target_clamped = target.long().clamp(0, num_classes - 1)
        target_one_hot.scatter_(1, target_clamped, 1)

        if not self.include_background:
            pred = pred[:, 1:]
            target_one_hot = target_one_hot[:, 1:]

        pred_flat = pred.flatten(2)
        target_flat = target_one_hot.flatten(2)

        # True positives, false positives, false negatives
        tp = (pred_flat * target_flat).sum(dim=2)
        fp = (pred_flat * (1 - target_flat)).sum(dim=2)
        fn = ((1 - pred_flat) * target_flat).sum(dim=2)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        return (1.0 - tversky).mean()


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for focusing on boundary regions

    Reference: Kervadec, H., et al. "Boundary loss for highly unbalanced segmentation."
    MIDL 2019.
    """

    def __init__(self, include_background: bool = False):
        super().__init__()
        self.include_background = include_background

    def forward(
        self,
        pred: torch.Tensor,
        dist_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions of shape (B, C, D, H, W)
            dist_map: Distance transform map of shape (B, C, D, H, W)
        """
        pred = F.softmax(pred, dim=1)

        if not self.include_background:
            pred = pred[:, 1:]
            dist_map = dist_map[:, 1:]

        loss = pred * dist_map
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function supporting multiple losses with weights
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        focal_weight: float = 0.0,
        tversky_weight: float = 0.0,
        gdc_weight: float = 0.0,
        focal_gamma: float = 2.0,
        class_weights: Optional[List[float]] = None,
        include_background: bool = True
    ):
        super().__init__()

        self.weights = {
            'dice': dice_weight,
            'ce': ce_weight,
            'focal': focal_weight,
            'tversky': tversky_weight,
            'gdc': gdc_weight
        }

        self.losses = nn.ModuleDict()

        if dice_weight > 0:
            self.losses['dice'] = DiceLoss(
                include_background=include_background,
                softmax=True
            )

        if ce_weight > 0:
            # Register class weights as buffer so they move with .to(device)
            if class_weights:
                self.register_buffer('ce_class_weights', torch.tensor(class_weights))
            else:
                self.register_buffer('ce_class_weights', None)
            # CE loss will be created with weights in forward pass
            self._ce_loss_fn = None

        if focal_weight > 0:
            self.losses['focal'] = FocalLoss(
                gamma=focal_gamma,
                alpha=class_weights
            )

        if tversky_weight > 0:
            self.losses['tversky'] = TverskyLoss(
                alpha=0.3,
                beta=0.7,  # Emphasize false negatives
                include_background=include_background
            )

        if gdc_weight > 0:
            self.losses['gdc'] = GeneralizedDiceLoss(
                include_background=include_background
            )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred: Predictions of shape (B, C, D, H, W)
            target: Ground truth of shape (B, 1, D, H, W) or (B, D, H, W)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        loss_dict = {}
        total_loss = 0.0

        # Handle target shape for CE loss
        target_ce = target.squeeze(1).long() if target.dim() == 5 else target.long()

        for name, weight in self.weights.items():
            if weight <= 0:
                continue

            if name == 'ce':
                # Use CrossEntropyLoss with weights on correct device
                loss = F.cross_entropy(pred, target_ce, weight=self.ce_class_weights)
            elif name in self.losses:
                loss_fn = self.losses[name]
                loss = loss_fn(pred, target)
            else:
                continue

            loss_dict[name] = loss.item()
            total_loss = total_loss + weight * loss

        return total_loss, loss_dict


class DeepSupervisionLoss(nn.Module):
    """
    Loss wrapper for deep supervision

    Applies loss to main output and intermediate outputs with decreasing weights.
    Automatically downsamples target to match each output's spatial resolution.
    """

    def __init__(
        self,
        loss_fn: nn.Module,
        weights: Optional[List[float]] = None,
        num_outputs: int = 4
    ):
        super().__init__()
        self.loss_fn = loss_fn

        if weights is None:
            # Exponentially decreasing weights
            weights = [0.5 ** i for i in range(num_outputs)]
            # Normalize
            total = sum(weights)
            weights = [w / total for w in weights]

        self.weights = weights

    def _downsample_target(self, target: torch.Tensor, output_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Downsample target to match output spatial dimensions.

        Args:
            target: Target tensor of shape (B, D, H, W) or (B, 1, D, H, W)
            output_shape: Output tensor shape (B, C, D', H', W')

        Returns:
            Downsampled target matching output spatial dimensions
        """
        # Get target spatial shape
        if target.dim() == 5:
            target_spatial = target.shape[2:]  # (D, H, W)
        else:
            target_spatial = target.shape[1:]  # (D, H, W)

        # Get output spatial shape
        output_spatial = output_shape[2:]  # (D', H', W')

        # Check if downsampling is needed
        if target_spatial == output_spatial:
            return target

        # Downsample using nearest neighbor interpolation (preserves label values)
        if target.dim() == 4:
            # Add channel dimension for interpolation
            target_5d = target.unsqueeze(1).float()
        else:
            target_5d = target.float()

        # Use nearest neighbor to preserve discrete labels
        downsampled = F.interpolate(
            target_5d,
            size=output_spatial,
            mode='nearest'
        )

        if target.dim() == 4:
            downsampled = downsampled.squeeze(1)

        return downsampled.long()

    def forward(
        self,
        outputs: Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]],
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            outputs: Either single output or tuple of (main_output, deep_supervision_outputs)
            target: Ground truth

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        if isinstance(outputs, tuple):
            main_output, deep_outputs = outputs
            deep_outputs = [main_output] + deep_outputs
        else:
            return self.loss_fn(outputs, target)

        loss_dict = {}
        total_loss = 0.0

        for i, (output, weight) in enumerate(zip(deep_outputs, self.weights)):
            # Downsample target to match output spatial dimensions
            target_ds = self._downsample_target(target, output.shape)

            if isinstance(self.loss_fn, CombinedLoss):
                loss, sub_loss_dict = self.loss_fn(output, target_ds)
                for k, v in sub_loss_dict.items():
                    loss_dict[f'{k}_ds{i}'] = v
            else:
                loss = self.loss_fn(output, target_ds)
                loss_dict[f'loss_ds{i}'] = loss.item()

            total_loss = total_loss + weight * loss

        return total_loss, loss_dict


def compute_class_weights(
    dataloader,
    num_classes: int,
    device: torch.device
) -> torch.Tensor:
    """
    Compute class weights based on inverse class frequency

    Args:
        dataloader: DataLoader to compute weights from
        num_classes: Number of classes
        device: Target device

    Returns:
        Class weights tensor
    """
    class_counts = torch.zeros(num_classes, device=device)

    for batch in dataloader:
        labels = batch['label'].to(device)
        for c in range(num_classes):
            class_counts[c] += (labels == c).sum()

    # Inverse frequency weighting
    total = class_counts.sum()
    weights = total / (num_classes * class_counts + 1e-6)

    # Normalize
    weights = weights / weights.sum() * num_classes

    return weights


def get_loss_function(config, num_classes: int = None) -> nn.Module:
    """
    Create loss function from config

    Args:
        config: Training configuration
        num_classes: Number of output classes (2 for binary, 3 for both mode)

    Returns:
        Loss function module
    """
    loss_type = config.train.loss_type.lower()

    # Determine component weights based on loss_type
    dice_weight = 0.0
    ce_weight = 0.0
    focal_weight = 0.0
    gdc_weight = 0.0

    if 'dice' in loss_type and 'gdc' not in loss_type:
        dice_weight = config.train.dice_weight
    if 'ce' in loss_type:
        ce_weight = config.train.ce_weight
    if 'focal' in loss_type:
        focal_weight = config.train.focal_weight
    if 'gdc' in loss_type:
        gdc_weight = getattr(config.train, 'gdc_weight', 1.0)

    # Adjust class weights for the number of classes
    class_weights = config.train.class_weights
    if class_weights and num_classes:
        if len(class_weights) != num_classes:
            # Adjust weights to match num_classes
            if num_classes == 2:
                # Binary mode: use [bg, fg] weights
                class_weights = [class_weights[0], max(class_weights[1:])]
            elif num_classes == 3 and len(class_weights) == 2:
                # Both mode but only 2 weights provided: [bg, skin, wall]
                class_weights = [class_weights[0], class_weights[1], class_weights[1]]

    # Create combined loss
    base_loss = CombinedLoss(
        dice_weight=dice_weight,
        ce_weight=ce_weight,
        focal_weight=focal_weight,
        gdc_weight=gdc_weight,
        focal_gamma=config.train.focal_gamma,
        class_weights=class_weights,
        include_background=True
    )

    # Wrap with deep supervision if enabled
    if config.model.use_deep_supervision:
        loss_fn = DeepSupervisionLoss(
            loss_fn=base_loss,
            weights=[0.5, 0.25, 0.125, 0.0625],  # Decreasing weights
            num_outputs=4
        )
    else:
        loss_fn = base_loss

    return loss_fn


# Metrics for evaluation
class DiceMetric:
    """Dice score metric for evaluation"""

    def __init__(self, num_classes: int, include_background: bool = False):
        self.num_classes = num_classes
        self.include_background = include_background
        self.reset()

    def reset(self):
        self.dice_scores = [[] for _ in range(self.num_classes)]

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metric with batch predictions

        Args:
            pred: Predictions of shape (B, C, D, H, W) or (B, D, H, W)
            target: Ground truth of shape (B, D, H, W)
        """
        if pred.dim() == 5:
            pred = pred.argmax(dim=1)

        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        start_class = 0 if self.include_background else 1

        for c in range(start_class, self.num_classes):
            pred_c = (pred == c).astype(float)
            target_c = (target == c).astype(float)

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            if union > 0:
                dice = 2.0 * intersection / union
            else:
                dice = 1.0 if intersection == 0 else 0.0

            self.dice_scores[c].append(dice)

    def compute(self) -> dict:
        """Compute mean Dice scores"""
        results = {}
        start_class = 0 if self.include_background else 1

        all_scores = []
        for c in range(start_class, self.num_classes):
            if self.dice_scores[c]:
                mean_dice = np.mean(self.dice_scores[c])
                results[f'dice_class_{c}'] = mean_dice
                all_scores.append(mean_dice)

        if all_scores:
            results['mean_dice'] = np.mean(all_scores)

        return results


if __name__ == "__main__":
    # Test loss functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy data
    pred = torch.randn(2, 3, 32, 32, 32, device=device)
    target = torch.randint(0, 3, (2, 32, 32, 32), device=device)

    print("Testing loss functions...")

    # Test Dice Loss
    dice_loss = DiceLoss()
    loss = dice_loss(pred, target)
    print(f"Dice Loss: {loss.item():.4f}")

    # Test Focal Loss
    focal_loss = FocalLoss(gamma=2.0)
    loss = focal_loss(pred, target)
    print(f"Focal Loss: {loss.item():.4f}")

    # Test Combined Loss
    combined_loss = CombinedLoss(dice_weight=1.0, ce_weight=1.0, focal_weight=0.5)
    loss, loss_dict = combined_loss(pred, target)
    print(f"Combined Loss: {loss.item():.4f}, Details: {loss_dict}")

    # Test Deep Supervision Loss
    deep_outputs = [
        torch.randn(2, 3, 32, 32, 32, device=device),
        torch.randn(2, 3, 32, 32, 32, device=device),
        torch.randn(2, 3, 32, 32, 32, device=device)
    ]
    ds_loss = DeepSupervisionLoss(combined_loss, num_outputs=4)
    loss, loss_dict = ds_loss((pred, deep_outputs), target)
    print(f"Deep Supervision Loss: {loss.item():.4f}")

    # Test Dice Metric
    print("\nTesting Dice Metric...")
    metric = DiceMetric(num_classes=3, include_background=False)
    pred_labels = pred.argmax(dim=1)
    metric.update(pred_labels, target)
    results = metric.compute()
    print(f"Dice Scores: {results}")
