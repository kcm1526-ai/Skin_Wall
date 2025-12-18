"""
3D UNet architectures with various improvements
Includes: Basic UNet, Residual UNet, Attention UNet with Deep Supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ConvBlock3D(nn.Module):
    """Basic 3D convolution block with BatchNorm and activation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class ResConvBlock3D(nn.Module):
    """Residual 3D convolution block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        # Skip connection
        self.skip = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.relu(x + residual)
        return x


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for 3D"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class AttentionGate3D(nn.Module):
    """Attention gate for skip connections"""

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm3d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm3d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # g: gate signal from decoder
        # x: skip connection from encoder

        # Upsample gate if needed
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='trilinear', align_corners=False)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Encoder3D(nn.Module):
    """Encoder part of UNet"""

    def __init__(
        self,
        in_channels: int,
        features: List[int],
        use_residual: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        block_class = ResConvBlock3D if use_residual else ConvBlock3D

        # First encoder block
        self.encoders.append(block_class(in_channels, features[0], dropout=dropout))

        # Remaining encoder blocks
        for i in range(len(features) - 1):
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            self.encoders.append(block_class(features[i], features[i + 1], dropout=dropout))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skip_connections = []

        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            if i < len(self.encoders) - 1:
                skip_connections.append(x)
                x = self.pools[i](x)

        return x, skip_connections


class Decoder3D(nn.Module):
    """Decoder part of UNet"""

    def __init__(
        self,
        features: List[int],
        use_residual: bool = False,
        use_attention: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.attention_gates = nn.ModuleList() if use_attention else None

        block_class = ResConvBlock3D if use_residual else ConvBlock3D
        reversed_features = features[::-1]

        for i in range(len(reversed_features) - 1):
            # Transposed convolution for upsampling
            self.upconvs.append(
                nn.ConvTranspose3d(
                    reversed_features[i],
                    reversed_features[i + 1],
                    kernel_size=2,
                    stride=2
                )
            )

            if use_attention:
                self.attention_gates.append(
                    AttentionGate3D(
                        reversed_features[i + 1],
                        reversed_features[i + 1],
                        reversed_features[i + 1] // 2
                    )
                )

            # Decoder block (concatenated skip + upsampled)
            self.decoders.append(
                block_class(
                    reversed_features[i + 1] * 2,
                    reversed_features[i + 1],
                    dropout=dropout
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        skip_connections: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        decoder_outputs = []
        skip_connections = skip_connections[::-1]

        for i in range(len(self.decoders)):
            x = self.upconvs[i](x)

            # Ensure spatial dimensions match
            skip = skip_connections[i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)

            # Apply attention if available
            if self.attention_gates is not None:
                skip = self.attention_gates[i](x, skip)

            x = torch.cat([x, skip], dim=1)
            x = self.decoders[i](x)
            decoder_outputs.append(x)

        return x, decoder_outputs


class UNet3D(nn.Module):
    """
    Basic 3D UNet architecture

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (classes)
        features: List of feature channels at each level
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        features: List[int] = [32, 64, 128, 256, 512],
        dropout: float = 0.0,
        use_deep_supervision: bool = False
    ):
        super().__init__()
        self.use_deep_supervision = use_deep_supervision

        self.encoder = Encoder3D(in_channels, features, use_residual=False, dropout=dropout)
        self.decoder = Decoder3D(features, use_residual=False, use_attention=False, dropout=dropout)

        # Final convolution
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

        # Deep supervision heads
        if use_deep_supervision:
            self.deep_supervision_heads = nn.ModuleList([
                nn.Conv3d(features[i], out_channels, kernel_size=1)
                for i in range(1, len(features) - 1)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        bottleneck, skip_connections = self.encoder(x)

        # Decoder
        x, decoder_outputs = self.decoder(bottleneck, skip_connections)

        # Final output
        output = self.final_conv(x)

        if self.training and self.use_deep_supervision:
            # Generate deep supervision outputs
            deep_outputs = []
            for i, (head, dec_out) in enumerate(zip(self.deep_supervision_heads, decoder_outputs[:-1])):
                ds_out = head(dec_out)
                # Upsample to original size
                ds_out = F.interpolate(ds_out, size=output.shape[2:], mode='trilinear', align_corners=False)
                deep_outputs.append(ds_out)

            return output, deep_outputs

        return output


class ResUNet3D(nn.Module):
    """
    Residual 3D UNet with skip connections within blocks

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (classes)
        features: List of feature channels at each level
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        features: List[int] = [32, 64, 128, 256, 512],
        dropout: float = 0.1,
        use_deep_supervision: bool = True
    ):
        super().__init__()
        self.use_deep_supervision = use_deep_supervision

        self.encoder = Encoder3D(in_channels, features, use_residual=True, dropout=dropout)
        self.decoder = Decoder3D(features, use_residual=True, use_attention=False, dropout=dropout)

        # SE blocks for channel attention
        self.se_blocks = nn.ModuleList([SEBlock3D(f) for f in features])

        # Final convolution
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

        # Deep supervision heads
        if use_deep_supervision:
            self.deep_supervision_heads = nn.ModuleList([
                nn.Conv3d(features[i], out_channels, kernel_size=1)
                for i in range(1, len(features) - 1)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder with SE blocks
        skip_connections = []
        for i, (encoder, pool) in enumerate(zip(self.encoder.encoders[:-1], self.encoder.pools)):
            x = encoder(x)
            x = self.se_blocks[i](x)
            skip_connections.append(x)
            x = pool(x)

        # Bottleneck
        x = self.encoder.encoders[-1](x)
        x = self.se_blocks[-1](x)

        # Decoder
        x, decoder_outputs = self.decoder(x, skip_connections)

        # Final output
        output = self.final_conv(x)

        if self.training and self.use_deep_supervision:
            deep_outputs = []
            for i, (head, dec_out) in enumerate(zip(self.deep_supervision_heads, decoder_outputs[:-1])):
                ds_out = head(dec_out)
                ds_out = F.interpolate(ds_out, size=output.shape[2:], mode='trilinear', align_corners=False)
                deep_outputs.append(ds_out)

            return output, deep_outputs

        return output


class AttentionUNet3D(nn.Module):
    """
    Attention 3D UNet with attention gates in skip connections

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (classes)
        features: List of feature channels at each level
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        features: List[int] = [32, 64, 128, 256, 512],
        dropout: float = 0.1,
        use_deep_supervision: bool = True
    ):
        super().__init__()
        self.use_deep_supervision = use_deep_supervision

        self.encoder = Encoder3D(in_channels, features, use_residual=True, dropout=dropout)
        self.decoder = Decoder3D(features, use_residual=True, use_attention=True, dropout=dropout)

        # Final convolution
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

        # Deep supervision heads
        if use_deep_supervision:
            self.deep_supervision_heads = nn.ModuleList([
                nn.Conv3d(features[i], out_channels, kernel_size=1)
                for i in range(1, len(features) - 1)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        bottleneck, skip_connections = self.encoder(x)

        # Decoder with attention
        x, decoder_outputs = self.decoder(bottleneck, skip_connections)

        # Final output
        output = self.final_conv(x)

        if self.training and self.use_deep_supervision:
            deep_outputs = []
            for i, (head, dec_out) in enumerate(zip(self.deep_supervision_heads, decoder_outputs[:-1])):
                ds_out = head(dec_out)
                ds_out = F.interpolate(ds_out, size=output.shape[2:], mode='trilinear', align_corners=False)
                deep_outputs.append(ds_out)

            return output, deep_outputs

        return output


if __name__ == "__main__":
    # Test models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test input
    x = torch.randn(1, 1, 96, 96, 96).to(device)

    print("Testing UNet3D...")
    model = UNet3D(in_channels=1, out_channels=3, use_deep_supervision=True).to(device)
    model.train()
    output, deep_outputs = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Deep supervision outputs: {len(deep_outputs)}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTesting ResUNet3D...")
    model = ResUNet3D(in_channels=1, out_channels=3, use_deep_supervision=True).to(device)
    model.train()
    output, deep_outputs = model(x)
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTesting AttentionUNet3D...")
    model = AttentionUNet3D(in_channels=1, out_channels=3, use_deep_supervision=True).to(device)
    model.train()
    output, deep_outputs = model(x)
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
