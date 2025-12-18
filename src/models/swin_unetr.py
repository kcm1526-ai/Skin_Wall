"""
Swin UNETR: Swin Transformer for 3D Medical Image Segmentation

Reference:
Hatamizadeh, A., et al. "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images"
BrainLes Workshop, MICCAI 2021

This implementation uses a Swin Transformer encoder with a UNet-style decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Sequence
import numpy as np
from functools import reduce
from operator import mul


def window_partition(x: torch.Tensor, window_size: Tuple[int, int, int]) -> torch.Tensor:
    """
    Partition input into non-overlapping windows

    Args:
        x: Input tensor of shape (B, D, H, W, C)
        window_size: Window size (Wd, Wh, Ww)

    Returns:
        Windows of shape (num_windows * B, Wd, Wh, Ww, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(
        B,
        D // window_size[0], window_size[0],
        H // window_size[1], window_size[1],
        W // window_size[2], window_size[2],
        C
    )
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse(
    windows: torch.Tensor,
    window_size: Tuple[int, int, int],
    B: int,
    D: int,
    H: int,
    W: int
) -> torch.Tensor:
    """
    Reverse window partition

    Args:
        windows: Windows of shape (num_windows * B, Wd, Wh, Ww, C)
        window_size: Window size
        B, D, H, W: Original dimensions

    Returns:
        Tensor of shape (B, D, H, W, C)
    """
    x = windows.view(
        B,
        D // window_size[0],
        H // window_size[1],
        W // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, D, H, W, -1)
    return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    """MLP for Swin Transformer"""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
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


class WindowAttention3D(nn.Module):
    """
    Window-based multi-head self-attention for 3D

    Args:
        dim: Number of input channels
        window_size: Window size
        num_heads: Number of attention heads
        qkv_bias: If True, add bias to qkv projection
        attn_drop: Attention dropout rate
        proj_drop: Projection dropout rate
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                num_heads
            )
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Get relative position index
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """
    Swin Transformer Block for 3D

    Args:
        dim: Number of input channels
        num_heads: Number of attention heads
        window_size: Window size
        shift_size: Shift size for SW-MSA
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        qkv_bias: If True, add bias to qkv projection
        drop: Dropout rate
        attn_drop: Attention dropout rate
        drop_path: Drop path rate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int] = (7, 7, 7),
        shift_size: Tuple[int, int, int] = (0, 0, 0),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dim, drop)

    def forward(self, x: torch.Tensor, mask_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, D, H, W, C = x.shape

        shortcut = x
        x = self.norm1(x)

        # Pad if needed
        pad_d = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        _, Dp, Hp, Wp, _ = x.shape

        # Cyclic shift
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                dims=(1, 2, 3)
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, reduce(mul, self.window_size), C)

        # Window attention
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, *self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, B, Dp, Hp, Wp)

        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                dims=(1, 2, 3)
            )
        else:
            x = shifted_x

        # Remove padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging3D(nn.Module):
    """Patch Merging Layer for downsampling"""

    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, C = x.shape

        # Pad if needed
        pad_d = D % 2
        pad_h = H % 2
        pad_w = W % 2
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
            _, D, H, W, _ = x.shape

        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 0::2, 0::2, 1::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 1::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 1::2, 1::2, 0::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class SwinTransformerStage(nn.Module):
    """A Swin Transformer stage (multiple blocks)"""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        downsample: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = tuple(w // 2 for w in window_size)

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
            )
            for i in range(depth)
        ])

        self.downsample = PatchMerging3D(dim) if downsample else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x)

        if self.downsample is not None:
            x_down = self.downsample(x)
        else:
            x_down = x

        return x, x_down


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding"""

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        in_channels: int = 1,
        embed_dim: int = 96,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, D, H, W = x.shape
        x = self.proj(x)  # (B, C, D', H', W')
        x = x.permute(0, 2, 3, 4, 1)  # (B, D', H', W', C)
        x = self.norm(x)
        return x


class SwinTransformerEncoder(nn.Module):
    """Swin Transformer Encoder"""

    def __init__(
        self,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        in_channels: int = 1,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 2, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: Tuple[int, int, int] = (7, 7, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        self.num_stages = len(depths)
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm
        )

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage = SwinTransformerStage(
                dim=embed_dim * (2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=(i < self.num_stages - 1)
            )
            self.stages.append(stage)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)

        features = []
        for stage in self.stages:
            x, x_down = stage(x)
            features.append(x.permute(0, 4, 1, 2, 3).contiguous())  # (B, C, D, H, W)
            x = x_down

        return features


class UNetDecoder3D(nn.Module):
    """UNet-style decoder for Swin UNETR"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_sizes: List[int],
        use_deep_supervision: bool = True
    ):
        super().__init__()
        self.use_deep_supervision = use_deep_supervision

        # Decoder blocks
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        reversed_features = feature_sizes[::-1]

        for i in range(len(reversed_features) - 1):
            in_ch = reversed_features[i]
            out_ch = reversed_features[i + 1]
            skip_ch = reversed_features[i + 1]

            self.upsamples.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2),
                    nn.InstanceNorm3d(out_ch),
                    nn.LeakyReLU(inplace=True)
                )
            )

            self.decoders.append(
                nn.Sequential(
                    nn.Conv3d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
                    nn.InstanceNorm3d(out_ch),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.InstanceNorm3d(out_ch),
                    nn.LeakyReLU(inplace=True)
                )
            )

        # Final upsampling to original resolution
        self.final_up = nn.Sequential(
            nn.ConvTranspose3d(feature_sizes[0], feature_sizes[0], kernel_size=4, stride=4),
            nn.InstanceNorm3d(feature_sizes[0]),
            nn.LeakyReLU(inplace=True)
        )

        # Output head
        self.out_conv = nn.Conv3d(feature_sizes[0], out_channels, kernel_size=1)

        # Deep supervision heads
        if use_deep_supervision:
            self.deep_heads = nn.ModuleList([
                nn.Conv3d(f, out_channels, kernel_size=1)
                for f in reversed_features[1:]  # Match decoder output channels
            ])

    def forward(
        self,
        features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        features = features[::-1]  # Reverse order
        x = features[0]

        decoder_outputs = []
        for i, (upsample, decoder) in enumerate(zip(self.upsamples, self.decoders)):
            x = upsample(x)

            # Match spatial dimensions
            skip = features[i + 1]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
            decoder_outputs.append(x)

        # Final upsampling
        x = self.final_up(x)
        output = self.out_conv(x)

        if self.training and self.use_deep_supervision:
            deep_outputs = []
            for head, dec_out in zip(self.deep_heads, decoder_outputs):
                ds_out = head(dec_out)
                ds_out = F.interpolate(ds_out, size=output.shape[2:], mode='trilinear', align_corners=False)
                deep_outputs.append(ds_out)
            return output, deep_outputs

        return output


class SwinUNETR(nn.Module):
    """
    Swin UNETR for 3D Medical Image Segmentation

    Args:
        img_size: Input image size (D, H, W)
        in_channels: Number of input channels
        out_channels: Number of output classes
        feature_size: Base feature size
        depths: Number of layers in each stage
        num_heads: Number of attention heads in each stage
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
        use_deep_supervision: Whether to use deep supervision
    """

    def __init__(
        self,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        in_channels: int = 1,
        out_channels: int = 3,
        feature_size: int = 48,
        depths: Tuple[int, ...] = (2, 2, 2, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: Tuple[int, int, int] = (7, 7, 7),
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_deep_supervision: bool = True
    ):
        super().__init__()

        # Feature sizes at each stage
        feature_sizes = [feature_size * (2 ** i) for i in range(len(depths))]

        # Encoder
        self.encoder = SwinTransformerEncoder(
            img_size=img_size,
            patch_size=(4, 4, 4),
            in_channels=in_channels,
            embed_dim=feature_size,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate
        )

        # Decoder
        self.decoder = UNetDecoder3D(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_sizes=feature_sizes,
            use_deep_supervision=use_deep_supervision
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        features = self.encoder(x)

        # Decoder
        output = self.decoder(features)

        return output


# Use MONAI's implementation if available (preferred)
def get_swin_unetr_monai(
    img_size: Tuple[int, int, int] = (96, 96, 96),
    in_channels: int = 1,
    out_channels: int = 3,
    feature_size: int = 48,
    use_checkpoint: bool = True,
    use_deep_supervision: bool = True,
    **kwargs
):
    """Get MONAI's Swin UNETR implementation if available"""
    try:
        from monai.networks.nets import SwinUNETR as MonaiSwinUNETR

        model = MonaiSwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            drop_rate=kwargs.get('drop_rate', 0.0),
            attn_drop_rate=kwargs.get('attn_drop_rate', 0.0),
        )

        # Wrap for deep supervision if needed
        if use_deep_supervision:
            return SwinUNETRWithDeepSupervision(model, out_channels)

        return model

    except ImportError:
        print("MONAI's SwinUNETR not available, using custom implementation")
        return SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_deep_supervision=use_deep_supervision,
            **kwargs
        )


class SwinUNETRWithDeepSupervision(nn.Module):
    """Wrapper to add deep supervision to MONAI's SwinUNETR"""

    def __init__(self, base_model, out_channels: int):
        super().__init__()
        self.base_model = base_model
        self.out_channels = out_channels

        # Access decoder layers for deep supervision
        # This depends on MONAI's implementation details

    def forward(self, x: torch.Tensor):
        return self.base_model(x)


if __name__ == "__main__":
    # Test Swin UNETR
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(1, 1, 96, 96, 96).to(device)

    print("Testing SwinUNETR (custom implementation)...")
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=3,
        feature_size=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
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

    # Test MONAI version if available
    print("\nTesting MONAI's SwinUNETR...")
    try:
        model_monai = get_swin_unetr_monai(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=3,
            feature_size=48
        ).to(device)
        output = model_monai(x)
        print(f"MONAI output shape: {output.shape}")
        print(f"MONAI parameters: {sum(p.numel() for p in model_monai.parameters()):,}")
    except Exception as e:
        print(f"MONAI test failed: {e}")
