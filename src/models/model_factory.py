"""
Model Factory for creating segmentation models
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from .unet3d import UNet3D, ResUNet3D, AttentionUNet3D
from .unetr import UNETR
from .swin_unetr import SwinUNETR, get_swin_unetr_monai
from .dense_unet3d import DenseUNet3D, DenseUNet3DSmall, DenseUNet3DLarge


MODEL_REGISTRY = {
    'unet3d': UNet3D,
    'resunet3d': ResUNet3D,
    'attention_unet3d': AttentionUNet3D,
    'unetr': UNETR,
    'swin_unetr': SwinUNETR,
    'dense_unet3d': DenseUNet3D,
    'dense_unet3d_small': DenseUNet3DSmall,
    'dense_unet3d_large': DenseUNet3DLarge,
}


def get_model_names():
    """Get list of available model names"""
    return list(MODEL_REGISTRY.keys()) + ['swin_unetr_monai', 'nnunet']


def create_model(
    model_type: str,
    in_channels: int = 1,
    out_channels: int = 3,
    img_size: Tuple[int, int, int] = (96, 96, 96),
    config: Optional[Any] = None,
    pretrained: bool = False,
    **kwargs
) -> nn.Module:
    """
    Create a segmentation model

    Args:
        model_type: Type of model to create
        in_channels: Number of input channels
        out_channels: Number of output classes
        img_size: Input image size (D, H, W)
        config: Optional configuration object
        pretrained: Whether to load pretrained weights (if available)
        **kwargs: Additional model-specific arguments

    Returns:
        PyTorch model
    """
    model_type = model_type.lower()

    if model_type == 'unet3d':
        model = UNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            features=kwargs.get('features', [32, 64, 128, 256, 512]),
            dropout=kwargs.get('dropout', 0.1),
            use_deep_supervision=kwargs.get('use_deep_supervision', True)
        )

    elif model_type == 'resunet3d':
        model = ResUNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            features=kwargs.get('features', [32, 64, 128, 256, 512]),
            dropout=kwargs.get('dropout', 0.1),
            use_deep_supervision=kwargs.get('use_deep_supervision', True)
        )

    elif model_type == 'attention_unet3d':
        model = AttentionUNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            features=kwargs.get('features', [32, 64, 128, 256, 512]),
            dropout=kwargs.get('dropout', 0.1),
            use_deep_supervision=kwargs.get('use_deep_supervision', True)
        )

    elif model_type == 'unetr':
        model = UNETR(
            img_size=img_size,
            patch_size=kwargs.get('patch_size', (16, 16, 16)),
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=kwargs.get('embed_dim', 768),
            num_layers=kwargs.get('num_layers', 12),
            num_heads=kwargs.get('num_heads', 12),
            mlp_ratio=kwargs.get('mlp_ratio', 4.0),
            dropout=kwargs.get('dropout', 0.1),
            use_deep_supervision=kwargs.get('use_deep_supervision', True)
        )

    elif model_type == 'swin_unetr':
        model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=kwargs.get('feature_size', 48),
            depths=kwargs.get('depths', (2, 2, 2, 2)),
            num_heads=kwargs.get('num_heads', (3, 6, 12, 24)),
            drop_rate=kwargs.get('drop_rate', 0.0),
            attn_drop_rate=kwargs.get('attn_drop_rate', 0.0),
            drop_path_rate=kwargs.get('drop_path_rate', 0.1),
            use_deep_supervision=kwargs.get('use_deep_supervision', True)
        )

    elif model_type == 'swin_unetr_monai':
        model = get_swin_unetr_monai(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=kwargs.get('feature_size', 48),
            use_checkpoint=kwargs.get('use_checkpoint', True),
            use_deep_supervision=kwargs.get('use_deep_supervision', True)
        )

    elif model_type == 'dense_unet3d':
        model = DenseUNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=kwargs.get('init_features', 48),
            growth_rate=kwargs.get('growth_rate', 12),
            block_config=kwargs.get('block_config', (4, 4, 4, 4)),
            bn_size=kwargs.get('bn_size', 4),
            dropout=kwargs.get('dropout', 0.1),
            use_deep_supervision=kwargs.get('use_deep_supervision', True)
        )

    elif model_type == 'dense_unet3d_small':
        model = DenseUNet3DSmall(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=kwargs.get('dropout', 0.1),
            use_deep_supervision=kwargs.get('use_deep_supervision', True)
        )

    elif model_type == 'dense_unet3d_large':
        model = DenseUNet3DLarge(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=kwargs.get('dropout', 0.1),
            use_deep_supervision=kwargs.get('use_deep_supervision', True)
        )

    elif model_type == 'nnunet':
        # Use MONAI's implementation of nnUNet-style model
        try:
            from monai.networks.nets import DynUNet

            # nnUNet-style configuration
            kernels = [[3, 3, 3]] * 5
            strides = [[1, 1, 1]] + [[2, 2, 2]] * 4

            model = DynUNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernels,
                strides=strides,
                upsample_kernel_size=strides[1:],
                norm_name="INSTANCE",
                deep_supervision=kwargs.get('use_deep_supervision', True),
                deep_supr_num=3
            )
        except ImportError:
            raise ImportError("MONAI is required for nnUNet model. Install with: pip install monai")

    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: {get_model_names()}")

    # Load pretrained weights if specified
    if pretrained:
        model = load_pretrained_weights(model, model_type, **kwargs)

    return model


def load_pretrained_weights(
    model: nn.Module,
    model_type: str,
    pretrained_path: Optional[str] = None,
    **kwargs
) -> nn.Module:
    """
    Load pretrained weights

    Args:
        model: Model to load weights into
        model_type: Type of model
        pretrained_path: Path to pretrained weights

    Returns:
        Model with loaded weights
    """
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        # Remove 'module.' prefix if present (from DataParallel)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Load weights (allow missing keys for fine-tuning)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")

    else:
        # Try to load MONAI pretrained weights for Swin UNETR
        if model_type in ['swin_unetr', 'swin_unetr_monai']:
            try:
                from monai.apps import download_and_extract

                # Download pretrained weights from MONAI model zoo
                # This would need actual URLs from MONAI model zoo
                print("Attempting to load MONAI pretrained weights...")

            except Exception as e:
                print(f"Could not load pretrained weights: {e}")

    return model


def get_model_params(model: nn.Module) -> Dict[str, int]:
    """Get model parameter statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params
    }


def print_model_summary(model: nn.Module, input_size: Tuple[int, ...] = (1, 1, 96, 96, 96)):
    """Print model summary"""
    params = get_model_params(model)

    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(f"Total Parameters: {params['total_params']:,}")
    print(f"Trainable Parameters: {params['trainable_params']:,}")
    print(f"Non-trainable Parameters: {params['non_trainable_params']:,}")
    print(f"Model Size: {params['total_params'] * 4 / 1024 / 1024:.2f} MB (float32)")

    # Try to calculate FLOPs if thop is available
    try:
        from thop import profile
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_size).to(device)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        print(f"FLOPs: {flops / 1e9:.2f} G")
    except:
        pass

    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test model creation
    print("Testing model factory...")

    for model_type in ['unet3d', 'resunet3d', 'attention_unet3d', 'unetr', 'swin_unetr']:
        print(f"\nCreating {model_type}...")
        model = create_model(
            model_type=model_type,
            in_channels=1,
            out_channels=3,
            img_size=(96, 96, 96)
        )
        print_model_summary(model)
