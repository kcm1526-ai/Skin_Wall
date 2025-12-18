"""
Configuration settings for Skin and Abdominal Wall Segmentation Model
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class DataConfig:
    """Data configuration"""
    # Base data path
    base_path: str = "/raid/users/ai_kcm_0/skin_and_wall/01011_SEV_sale"

    # Image paths pattern (relative to each subject folder)
    image_subpath: str = "01_DICOM/PP"

    # Primary mask paths
    primary_mask_subpath: str = "01_DICOM/PP/Mask"

    # Alternative mask paths
    alt_mask_subpath: str = "03_Vein"

    # Mask filenames
    skin_mask_name: str = "Skin.nii.gz"
    abdominal_wall_mask_name: str = "Abdominal_wall.nii.gz"
    # Alternative naming (typo in original data)
    alt_abdominal_wall_mask_name: str = "Abdonimal_wall.nii.gz"

    # Number of classes (background, skin, abdominal_wall)
    num_classes: int = 3
    class_names: List[str] = field(default_factory=lambda: ["Background", "Skin", "Abdominal_Wall"])

    # Data split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Random seed for reproducibility
    seed: int = 42


@dataclass
class PreprocessConfig:
    """Preprocessing configuration"""
    # Target spacing for resampling (mm)
    target_spacing: Tuple[float, float, float] = (1.5, 1.5, 2.0)

    # Patch size for training (D, H, W)
    patch_size: Tuple[int, int, int] = (96, 96, 96)

    # Intensity normalization
    clip_values: Tuple[float, float] = (-1000, 1000)  # HU values for CT
    normalize_method: str = "zscore"  # "zscore" or "minmax"

    # Window/Level for CT (optional)
    use_windowing: bool = True
    window_center: float = 40.0
    window_width: float = 400.0


@dataclass
class AugmentConfig:
    """Data augmentation configuration"""
    # Spatial augmentations
    random_flip_prob: float = 0.5
    random_rotate_prob: float = 0.3
    rotation_range: Tuple[float, float] = (-15.0, 15.0)  # degrees
    random_scale_prob: float = 0.3
    scale_range: Tuple[float, float] = (0.85, 1.15)

    # Intensity augmentations
    random_intensity_shift_prob: float = 0.3
    intensity_shift_range: Tuple[float, float] = (-0.1, 0.1)
    random_intensity_scale_prob: float = 0.3
    intensity_scale_range: Tuple[float, float] = (0.9, 1.1)

    # Noise and blur
    gaussian_noise_prob: float = 0.2
    gaussian_noise_std: float = 0.05
    gaussian_blur_prob: float = 0.2

    # Elastic deformation
    elastic_deform_prob: float = 0.2


@dataclass
class ModelConfig:
    """Model configuration"""
    # Model type: "unet3d", "unetr", "swin_unetr", "nnunet"
    model_type: str = "swin_unetr"

    # Input channels
    in_channels: int = 1

    # Output channels (number of classes)
    out_channels: int = 3

    # UNet3D specific
    unet_features: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])

    # UNETR specific
    unetr_hidden_size: int = 768
    unetr_mlp_dim: int = 3072
    unetr_num_heads: int = 12
    unetr_num_layers: int = 12

    # Swin UNETR specific
    swin_feature_size: int = 48
    swin_depths: Tuple[int, ...] = (2, 2, 2, 2)
    swin_num_heads: Tuple[int, ...] = (3, 6, 12, 24)
    swin_drop_rate: float = 0.0
    swin_attn_drop_rate: float = 0.0

    # Deep supervision
    use_deep_supervision: bool = True

    # Dropout
    dropout_rate: float = 0.1


@dataclass
class TrainConfig:
    """Training configuration"""
    # Experiment name
    exp_name: str = "skin_wall_segmentation"

    # Output directory
    output_dir: str = "./outputs"

    # Training parameters
    batch_size: int = 2
    num_epochs: int = 500
    num_workers: int = 8

    # Optimizer
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Learning rate scheduler
    scheduler: str = "cosine_warmup"  # "cosine", "cosine_warmup", "step", "reduce_on_plateau"
    warmup_epochs: int = 10
    min_lr: float = 1e-6

    # Loss function
    loss_type: str = "dice_ce_focal"  # "dice", "ce", "dice_ce", "dice_focal", "dice_ce_focal"
    dice_weight: float = 1.0
    ce_weight: float = 1.0
    focal_weight: float = 0.5
    focal_gamma: float = 2.0

    # Class weights for imbalanced data
    use_class_weights: bool = True
    class_weights: Optional[List[float]] = None  # Auto-computed if None

    # Gradient accumulation
    accumulation_steps: int = 4

    # Mixed precision training
    use_amp: bool = True

    # Gradient clipping
    grad_clip_max_norm: float = 1.0

    # Checkpointing
    save_every_n_epochs: int = 10
    keep_n_checkpoints: int = 5

    # Early stopping
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 0.001

    # Validation
    val_every_n_epochs: int = 1

    # Sliding window inference
    sw_batch_size: int = 4
    sw_overlap: float = 0.5

    # Resume training
    resume_checkpoint: Optional[str] = None


@dataclass
class InferenceConfig:
    """Inference configuration"""
    # Model checkpoint
    checkpoint_path: str = ""

    # Sliding window parameters
    sw_batch_size: int = 4
    sw_overlap: float = 0.5

    # Post-processing
    use_postprocessing: bool = True
    min_component_size: int = 100

    # Test-time augmentation
    use_tta: bool = True
    tta_flips: List[int] = field(default_factory=lambda: [2, 3, 4])  # Flip axes

    # Output
    save_predictions: bool = True
    output_dir: str = "./predictions"


@dataclass
class Config:
    """Main configuration combining all sub-configs"""
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def __post_init__(self):
        """Ensure output directories exist"""
        os.makedirs(self.train.output_dir, exist_ok=True)
        os.makedirs(self.inference.output_dir, exist_ok=True)


def get_config(model_type: str = "swin_unetr") -> Config:
    """Get configuration with specified model type"""
    config = Config()
    config.model.model_type = model_type
    return config
