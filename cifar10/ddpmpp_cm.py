# https://github.com/openai/consistency_models
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def convert_module_to_f16(l: nn.Module):
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l: nn.Module):
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


def timestep_embedding(
    timesteps: torch.Tensor, dim: int, max_period: int = 10000, scale: float = 1.0
):
    assert dim % 2 == 0
    denom = -np.log(max_period) / (dim // 2)
    # [E // 2]
    freqs = torch.exp(
        denom * torch.arange(0, dim // 2, dtype=torch.float32, device=timesteps.device)
    )
    # [T, E // 2]
    args = scale * timesteps[:, None].float() * freqs
    # [T, E]
    return torch.cat([args.cos(), args.sin()], dim=-1).to(timesteps.dtype)


@torch.no_grad()
def zero_module(module: nn.Module):
    for p in module.parameters():
        p.zero_()
    return module


class GroupNorm32bit(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class TimestepBlock(nn.Module):
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TimestepBlock.forward is not implemented")


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, out_channels: int | None = None):
        super().__init__()
        self.conv = None
        if use_conv:
            self.conv = nn.Conv2d(channels, out_channels or channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, out_channels: int | None = None):
        super().__init__()
        if use_conv:
            self.op = nn.Conv2d(
                channels, out_channels or channels, 3, stride=2, padding=1
            )
        else:
            self.op = nn.AveragePool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: int | None = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        use_double_norm: bool = False,
        up: bool = False,
        down: bool = False,
    ):
        super().__init__()
        out_channels = out_channels or channels
        self.in_norm = GroupNorm32bit(32, channels)
        self.in_layer = nn.Conv2d(channels, out_channels, 3, padding=1)

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_double_norm = use_double_norm
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * out_channels if use_scale_shift_norm else out_channels,
            ),
        )

        self.out_norm = GroupNorm32bit(32, out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1)),
        )

        if out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W]
        h = F.silu(self.in_norm(x))
        # [B, C, H', W']
        if self.updown:
            h, x = self.h_upd(h), self.x_upd(x)
        h = self.in_layer(h)
        # [B, C, 1, 1]
        emb_out = self.emb_layers(emb).to(h.dtype)[..., None, None]
        # [B, C, H', W']
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            scale = scale + 1
            if self.use_double_norm:
                _pixel_norm = lambda p: p * (p.square().mean(dim=1, keepdims=True) + 1e-8).rsqrt()
                scale = _pixel_norm(scale)
                shift = _pixel_norm(shift)
            h = self.out_norm(h) * scale + shift
            h = self.out_layers(h)
        else:
            h = h + emb_out
            h = self.out_layers(self.out_norm(h))
        # [B, C, H', W']
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        attention_type: str = "flash",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.dtype = dtype
        self.norm = GroupNorm32bit(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        if attention_type == "flash":
            # assert channels % num_heads == 0 and channels // num_heads in [16, 32, 64], (channels, num_heads)
            self.attention = QKVFlashAttention(num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(num_heads)

        self.proj_out = zero_module(nn.Conv2d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, C, H, W
        b, _, *spatial = x.shape
        # [B, C x 3, H x W]
        qkv = self.qkv(self.norm(x)).view(b, -1, np.prod(spatial))
        # [B, C, H x W]
        h = self.attention.forward(qkv.to(self.dtype)).to(x.dtype)
        # [B, C, H, W]
        h = h.view(b, -1, *spatial)
        h = self.proj_out(h)
        return x + h


class QKVFlashAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        from flash_attn.modules.mha import FlashSelfAttention

        self.num_heads = num_heads
        self.inner_attn = FlashSelfAttention(
            causal=False,
            attention_dropout=attention_dropout,
        )

    def forward(self, qkv: torch.Tensor):
        b, _, t = qkv.shape
        # [B, 3 x C, H x W] > [B, 3, N, C // N, H x W] > [B, H x W, 3, N, C // N]
        qkv = qkv.view(b, 3, self.num_heads, -1, t).permute(0, 4, 1, 2, 3)
        # [B, H x W, N, C // N]
        out = self.inner_attn(qkv)
        # [B, N, C // N, H x W] > [B, C, H x W]
        return out.permute(0, 2, 3, 1).view(b, -1, t)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/output heads shaping
    """

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        # [B, 3 x C, H x W]
        b, _, t = qkv.shape
        # [B, 3 x C, H x W] > [B, 3, N, C // N, H x W] > [3, B, N, H x W, C // N] > 3 x [B, N, H x W, C // N]
        q, k, v = qkv.view(b, 3, self.n_heads, -1, t).permute(1, 0, 2, 4, 3)
        # C // N
        *_, ch = q.shape
        # [B, N, H x W, H x W]
        weight = torch.softmax(
            (q @ k.transpose(-1, -2)) * (ch**-0.5), dim=-1
        )  # 1 / match.sqrt(math.sqrt(ch))
        # [B, N, H x W, C // N] > [B, N, C // N, H x W] > [B, C, H x W]
        return (weight @ v).permute(0, 1, 3, 2).view(b, -1, t)


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: list[int],
        dropout: float = 0,
        channel_mult: list[int] = [1, 2, 4, 8],
        conv_resample: bool = True,
        num_classes: int | None = None,
        num_heads: int = 1,
        use_scale_shift_norm: bool = False,
        use_double_norm: bool = False,
        resblock_updown: bool = False,
        temb_scale: float = 1.0,
        attn_module: Literal["flash", "legacy"] = "flash",
        attn_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.model_channels = model_channels
        self.temb_scale = temb_scale

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        _ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(in_channels, _ch, 3, padding=1))]
        )

        _ds, _input_block_chans = 1, [_ch]
        for i, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        _ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_double_norm=use_double_norm,
                    )
                ]
                _ch = int(mult * model_channels)
                if _ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            _ch,
                            attention_type=attn_module,
                            num_heads=num_heads,
                            dtype=attn_dtype,
                        )
                    )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                _input_block_chans.append(_ch)

            if i < len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            _ch,
                            time_embed_dim,
                            dropout,
                            use_scale_shift_norm=use_scale_shift_norm,
                            use_double_norm=use_double_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(_ch, conv_resample)
                    )
                )
                _ds *= 2
                _input_block_chans.append(_ch)

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                _ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
                use_double_norm=use_double_norm,
            ),
            AttentionBlock(
                _ch,
                num_heads=num_heads,
                attention_type=attn_module,
                dtype=attn_dtype,
            ),
            ResBlock(
                _ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
                use_double_norm=use_double_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = _input_block_chans.pop()
                layers = [
                    ResBlock(
                        _ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_double_norm=use_double_norm,
                    )
                ]
                _ch = int(model_channels * mult)
                if _ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            _ch,
                            num_heads=num_heads,
                            attention_type=attn_module,
                            dtype=attn_dtype,
                        )
                    )

                if level and i == num_res_blocks:
                    layers.append(
                        ResBlock(
                            _ch,
                            time_embed_dim,
                            dropout,
                            use_scale_shift_norm=use_scale_shift_norm,
                            use_double_norm=use_double_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(_ch, conv_resample)
                    )
                    _ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            GroupNorm32bit(32, _ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, y: torch.Tensor | None = None
    ):
        hs = []
        # [B, E]
        emb = self.time_embed(
            timestep_embedding(
                timesteps.to(x), self.model_channels, scale=self.temb_scale
            ),
        )
        if y is not None:
            emb = emb + self.label_emb(y.to(x))
        # [B, C, H, W]
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = module(torch.cat([h, hs.pop()], dim=1), emb)
        return self.out(h)
