"""
UNETR: UNet with Transformer encoder for 3D medical image segmentation

Reference:
Hatamizadeh, A., et al. "UNETR: Transformers for 3D Medical Image Segmentation"
WACV 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math


class PatchEmbedding3D(nn.Module):
    """3D Patch Embedding for Vision Transformer"""

    def __init__(
        self,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 1,
        embed_dim: int = 768,
        dropout: float = 0.0
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size[0]) * \
                         (img_size[1] // patch_size[1]) * \
                         (img_size[2] // patch_size[2])

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feed-forward network"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, attn_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Vision Transformer Encoder"""

    def __init__(
        self,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 1,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Calculate number of patches for each dimension
        self.patch_shape = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2]
        )
        self.n_patches = self.patch_shape[0] * self.patch_shape[1] * self.patch_shape[2]

        # Patch embedding
        self.patch_embed = PatchEmbedding3D(
            img_size, patch_size, in_channels, embed_dim, dropout
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Layers to extract intermediate features
        self.extract_layers = [3, 6, 9, 12]  # z3, z6, z9, z12

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Store intermediate features
        hidden_states = []

        # Transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            if (i + 1) in self.extract_layers:
                hidden_states.append(x)

        x = self.norm(x)

        return x, hidden_states


class ConvBlock3D(nn.Module):
    """3D Convolution block for decoder"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DeconvBlock3D(nn.Module):
    """3D Deconvolution block for upsampling"""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=scale_factor,
                stride=scale_factor
            ),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


class UNETR(nn.Module):
    """
    UNETR: UNet with Transformer encoder

    Args:
        img_size: Input image size (D, H, W)
        patch_size: Patch size for ViT
        in_channels: Number of input channels
        out_channels: Number of output channels (classes)
        embed_dim: Transformer embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate
        use_deep_supervision: Whether to use deep supervision
    """

    def __init__(
        self,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 1,
        out_channels: int = 3,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_deep_supervision: bool = True
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_deep_supervision = use_deep_supervision

        # Feature sizes at different levels
        self.feat_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2]
        )

        # Transformer encoder
        self.vit = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

        # CNN encoder for skip connections
        self.encoder1 = ConvBlock3D(in_channels, 32)
        self.encoder2 = ConvBlock3D(32, 64)
        self.encoder3 = ConvBlock3D(64, 128)
        self.encoder4 = ConvBlock3D(128, 256)

        self.pool = nn.MaxPool3d(2, 2)

        # Project transformer features to decoder dimensions
        self.proj3 = nn.Linear(embed_dim, 128)
        self.proj6 = nn.Linear(embed_dim, 256)
        self.proj9 = nn.Linear(embed_dim, 512)
        self.proj12 = nn.Linear(embed_dim, 1024)

        # Decoder
        self.decoder5 = DeconvBlock3D(1024, 512)
        self.decoder4_conv = ConvBlock3D(512 + 512, 512)

        self.decoder4 = DeconvBlock3D(512, 256)
        self.decoder3_conv = ConvBlock3D(256 + 256, 256)

        self.decoder3 = DeconvBlock3D(256, 128)
        self.decoder2_conv = ConvBlock3D(128 + 128, 128)

        self.decoder2 = DeconvBlock3D(128, 64)
        self.decoder1_conv = ConvBlock3D(64 + 64, 64)

        self.decoder1 = DeconvBlock3D(64, 32)
        self.decoder0_conv = ConvBlock3D(32 + 32, 32)

        # Final convolution
        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)

        # Deep supervision heads
        if use_deep_supervision:
            self.deep_heads = nn.ModuleList([
                nn.Conv3d(512, out_channels, kernel_size=1),
                nn.Conv3d(256, out_channels, kernel_size=1),
                nn.Conv3d(128, out_channels, kernel_size=1),
                nn.Conv3d(64, out_channels, kernel_size=1),
            ])

    def _reshape_transformer_output(
        self,
        x: torch.Tensor,
        proj_layer: nn.Linear
    ) -> torch.Tensor:
        """Reshape transformer output to 3D feature map"""
        B, N, C = x.shape
        x = proj_layer(x)
        x = x.transpose(1, 2).reshape(B, -1, *self.feat_size)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get transformer features at different layers
        _, hidden_states = self.vit(x)
        z3, z6, z9, z12 = hidden_states

        # Reshape transformer outputs
        z3 = self._reshape_transformer_output(z3, self.proj3)
        z6 = self._reshape_transformer_output(z6, self.proj6)
        z9 = self._reshape_transformer_output(z9, self.proj9)
        z12 = self._reshape_transformer_output(z12, self.proj12)

        # CNN encoder for local features
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        # Decoder
        d5 = self.decoder5(z12)
        if d5.shape[2:] != z9.shape[2:]:
            d5 = F.interpolate(d5, size=z9.shape[2:], mode='trilinear', align_corners=False)
        d5 = self.decoder4_conv(torch.cat([d5, z9], dim=1))

        d4 = self.decoder4(d5)
        if d4.shape[2:] != z6.shape[2:]:
            d4 = F.interpolate(d4, size=z6.shape[2:], mode='trilinear', align_corners=False)
        d4 = self.decoder3_conv(torch.cat([d4, z6], dim=1))

        d3 = self.decoder3(d4)
        if d3.shape[2:] != z3.shape[2:]:
            d3 = F.interpolate(d3, size=z3.shape[2:], mode='trilinear', align_corners=False)
        d3 = self.decoder2_conv(torch.cat([d3, z3], dim=1))

        d2 = self.decoder2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d2 = self.decoder1_conv(torch.cat([d2, e2], dim=1))

        d1 = self.decoder1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d1 = self.decoder0_conv(torch.cat([d1, e1], dim=1))

        # Final output
        output = self.final_conv(d1)

        if self.training and self.use_deep_supervision:
            deep_outputs = []
            for head, feat in zip(self.deep_heads, [d5, d4, d3, d2]):
                ds_out = head(feat)
                ds_out = F.interpolate(ds_out, size=output.shape[2:], mode='trilinear', align_corners=False)
                deep_outputs.append(ds_out)
            return output, deep_outputs

        return output


if __name__ == "__main__":
    # Test UNETR
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(1, 1, 96, 96, 96).to(device)

    print("Testing UNETR...")
    model = UNETR(
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        in_channels=1,
        out_channels=3,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        use_deep_supervision=True
    ).to(device)

    model.train()
    output, deep_outputs = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Deep supervision outputs: {len(deep_outputs)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test inference mode
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"Inference output shape: {output.shape}")
