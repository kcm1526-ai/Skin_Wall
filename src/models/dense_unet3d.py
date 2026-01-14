"""
3D Dense UNet for Medical Image Segmentation

Dense UNet combines DenseNet's dense connections with UNet's encoder-decoder structure.
Each dense block has multiple layers with dense connections (all layers connected to all subsequent layers).

Reference:
- DenseNet: "Densely Connected Convolutional Networks" (Huang et al., CVPR 2017)
- UNet: "U-Net: Convolutional Networks for Biomedical Image Segmentation" (Ronneberger et al., MICCAI 2015)
- Dense UNet for medical imaging: Various adaptations for 3D medical imaging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class DenseLayer(nn.Module):
    """
    Single dense layer: BN -> ReLU -> Conv3D

    Each layer takes all preceding feature maps as input and produces k (growth_rate) feature maps.
    """

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()

        # Bottleneck: 1x1 conv to reduce channels before 3x3 conv
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv1 = nn.Conv3d(
            in_channels, bn_size * growth_rate,
            kernel_size=1, stride=1, bias=False
        )

        # 3x3 conv
        self.bn2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.conv2 = nn.Conv3d(
            bn_size * growth_rate, growth_rate,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bottleneck
        out = self.conv1(F.relu(self.bn1(x)))
        # 3x3 conv
        out = self.conv2(F.relu(self.bn2(out)))

        if self.dropout is not None:
            out = self.dropout(out)

        return out


class DenseBlock(nn.Module):
    """
    Dense Block: Multiple dense layers with dense connections.

    Each layer receives feature maps from all preceding layers.
    Output channels = in_channels + num_layers * growth_rate
    """

    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        growth_rate: int,
        bn_size: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            layer = DenseLayer(
                in_channels=layer_in_channels,
                growth_rate=growth_rate,
                bn_size=bn_size,
                dropout=dropout
            )
            self.layers.append(layer)

        self.out_channels = in_channels + num_layers * growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]

        for layer in self.layers:
            # Concatenate all previous features
            concat_features = torch.cat(features, dim=1)
            new_features = layer(concat_features)
            features.append(new_features)

        return torch.cat(features, dim=1)


class TransitionDown(nn.Module):
    """
    Transition layer for downsampling: BN -> Conv1x1 -> AvgPool
    Reduces channels and spatial dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super().__init__()

        self.bn = nn.BatchNorm3d(in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


class TransitionUp(nn.Module):
    """
    Transition layer for upsampling: TransposedConv3D
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super().__init__()

        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=2, stride=2
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.conv_transpose(x)

        # Handle size mismatch
        if out.shape != skip.shape:
            diff_d = skip.shape[2] - out.shape[2]
            diff_h = skip.shape[3] - out.shape[3]
            diff_w = skip.shape[4] - out.shape[4]

            out = F.pad(out, [
                diff_w // 2, diff_w - diff_w // 2,
                diff_h // 2, diff_h - diff_h // 2,
                diff_d // 2, diff_d - diff_d // 2
            ])

        return torch.cat([out, skip], dim=1)


class DenseUNet3D(nn.Module):
    """
    3D Dense UNet for Medical Image Segmentation.

    Architecture:
    - Encoder: Initial conv + Dense blocks with transition down
    - Bottleneck: Dense block
    - Decoder: Dense blocks with transition up + skip connections
    - Output: Final conv to num_classes

    Args:
        in_channels: Number of input channels (1 for CT)
        out_channels: Number of output classes
        init_features: Initial number of features after first conv
        growth_rate: Number of features added by each dense layer
        block_config: Number of layers in each dense block (encoder, bottleneck, decoder)
        bn_size: Bottleneck size multiplier
        dropout: Dropout rate
        use_deep_supervision: Whether to use deep supervision
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        init_features: int = 48,
        growth_rate: int = 12,
        block_config: Tuple[int, ...] = (4, 4, 4, 4),
        bn_size: int = 4,
        dropout: float = 0.1,
        use_deep_supervision: bool = True
    ):
        super().__init__()

        self.use_deep_supervision = use_deep_supervision
        num_blocks = len(block_config)

        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, init_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(init_features),
            nn.ReLU(inplace=True)
        )

        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        self.transitions_down = nn.ModuleList()
        self.skip_channels = []

        num_features = init_features

        for i in range(num_blocks - 1):  # Last block is bottleneck
            # Dense block
            block = DenseBlock(
                in_channels=num_features,
                num_layers=block_config[i],
                growth_rate=growth_rate,
                bn_size=bn_size,
                dropout=dropout
            )
            self.encoder_blocks.append(block)
            num_features = block.out_channels
            self.skip_channels.append(num_features)

            # Transition down (reduce channels by half)
            trans_down = TransitionDown(num_features, num_features // 2)
            self.transitions_down.append(trans_down)
            num_features = num_features // 2

        # Bottleneck
        self.bottleneck = DenseBlock(
            in_channels=num_features,
            num_layers=block_config[-1],
            growth_rate=growth_rate,
            bn_size=bn_size,
            dropout=dropout
        )
        num_features = self.bottleneck.out_channels

        # Decoder path
        self.transitions_up = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for i in range(num_blocks - 1):
            skip_ch = self.skip_channels[-(i + 1)]

            # Transition up
            trans_up = TransitionUp(num_features, num_features)
            self.transitions_up.append(trans_up)

            # Dense block (input = upsampled + skip)
            block_in_channels = num_features + skip_ch
            block = DenseBlock(
                in_channels=block_in_channels,
                num_layers=block_config[-(i + 2)],
                growth_rate=growth_rate,
                bn_size=bn_size,
                dropout=dropout
            )
            self.decoder_blocks.append(block)
            num_features = block.out_channels

        # Final convolution
        self.final_conv = nn.Conv3d(num_features, out_channels, kernel_size=1)

        # Deep supervision outputs
        if use_deep_supervision:
            self.deep_supervision_outputs = nn.ModuleList()
            # Add outputs at each decoder level
            ds_features = [self.bottleneck.out_channels]
            for block in self.decoder_blocks[:-1]:
                ds_features.append(block.out_channels)

            for feat in ds_features:
                self.deep_supervision_outputs.append(
                    nn.Conv3d(feat, out_channels, kernel_size=1)
                )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        # Initial conv
        x = self.initial_conv(x)

        # Encoder
        skips = []
        for i, (block, trans) in enumerate(zip(self.encoder_blocks, self.transitions_down)):
            x = block(x)
            skips.append(x)
            x = trans(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with deep supervision
        deep_outputs = []

        if self.use_deep_supervision and self.training:
            # First deep supervision from bottleneck
            ds_out = self.deep_supervision_outputs[0](x)
            deep_outputs.append(ds_out)

        for i, (trans_up, block) in enumerate(zip(self.transitions_up, self.decoder_blocks)):
            skip = skips[-(i + 1)]
            x = trans_up(x, skip)
            x = block(x)

            # Deep supervision (except for last decoder block)
            if self.use_deep_supervision and self.training and i < len(self.decoder_blocks) - 1:
                ds_idx = i + 1
                if ds_idx < len(self.deep_supervision_outputs):
                    ds_out = self.deep_supervision_outputs[ds_idx](x)
                    deep_outputs.append(ds_out)

        # Final output
        output = self.final_conv(x)

        if self.use_deep_supervision and self.training:
            return output, deep_outputs

        return output


class DenseUNet3DSmall(DenseUNet3D):
    """
    Smaller version of Dense UNet for faster training / less GPU memory.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        dropout: float = 0.1,
        use_deep_supervision: bool = True
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=32,
            growth_rate=8,
            block_config=(3, 3, 3, 3),
            bn_size=4,
            dropout=dropout,
            use_deep_supervision=use_deep_supervision
        )


class DenseUNet3DLarge(DenseUNet3D):
    """
    Larger version of Dense UNet for better performance.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        dropout: float = 0.1,
        use_deep_supervision: bool = True
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=64,
            growth_rate=16,
            block_config=(6, 6, 6, 6),
            bn_size=4,
            dropout=dropout,
            use_deep_supervision=use_deep_supervision
        )


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test standard Dense UNet
    model = DenseUNet3D(
        in_channels=1,
        out_channels=3,
        use_deep_supervision=True
    ).to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"DenseUNet3D:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")

    # Test forward pass
    x = torch.randn(2, 1, 96, 96, 96).to(device)

    model.train()
    output, deep_outputs = model(x)
    print(f"\n  Training mode:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Deep supervision outputs: {len(deep_outputs)}")
    for i, ds in enumerate(deep_outputs):
        print(f"    DS {i}: {ds.shape}")

    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"\n  Eval mode:")
    print(f"  Output shape: {output.shape}")

    # Test small version
    print(f"\n\nDenseUNet3DSmall:")
    model_small = DenseUNet3DSmall(in_channels=1, out_channels=3).to(device)
    small_params = sum(p.numel() for p in model_small.parameters())
    print(f"  Total parameters: {small_params:,}")

    # Test large version
    print(f"\nDenseUNet3DLarge:")
    model_large = DenseUNet3DLarge(in_channels=1, out_channels=3).to(device)
    large_params = sum(p.numel() for p in model_large.parameters())
    print(f"  Total parameters: {large_params:,}")
