import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(
    timesteps: torch.Tensor, dim: int, max_positions: int = 10000
) -> torch.Tensor:
    assert dim % 2 == 0
    emb = np.log(max_positions) / (dim // 2 - 1)
    # [E // 2]
    emb = torch.exp(
        torch.arange(dim // 2, dtype=torch.float32, device=timesteps.device) * -emb
    )
    # [T, E // 2]
    emb = timesteps[:, None] * emb[None]
    # [T, E]
    return torch.cat([emb.sin(), emb.cos()], axis=1)


def default_init(
    *shape: int, scale: float = 1.0, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    out_size, in_size = shape[0], shape[1]
    field_size = math.prod(shape) / in_size / out_size
    fan_in, fan_out = in_size * field_size, out_size * field_size
    denominator = (fan_in + fan_out) / 2
    if scale == 0.0:
        scale = 1e-10
    return (torch.rand(*shape, dtype=dtype) * 2 - 1) * np.sqrt(3 * scale / denominator)


class NIN(nn.Module):
    def __init__(self, in_dim: int, num_units: int, init_scale: float = 0.1):
        super().__init__()
        self.W = nn.Parameter(default_init(num_units, in_dim, scale=init_scale))
        self.b = nn.Parameter(torch.zeros(num_units))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [..., in_dim] > [..., num_units]
        return x @ self.W.T + self.b


class AttnBlockpp(nn.Module):
    def __init__(
        self, channels: int, skip_rescale: bool = True, init_scale: float = 0.0
    ):
        super().__init__()
        self.groupnorm = nn.GroupNorm(min(channels // 4, 32), channels, eps=1e-6)

        self.proj_q = NIN(channels, channels)
        self.proj_k = NIN(channels, channels)
        self.proj_v = NIN(channels, channels)
        self.proj_out = NIN(channels, channels, init_scale=init_scale)

        self.skip_rescale = skip_rescale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # [B, C, H, W]
        h = self.groupnorm.forward(x)
        # [B, H x W, C]
        h = h.permute(0, 2, 3, 1).view(B, H * W, C)
        # [B, H x W, C]
        q, k, v = self.proj_q(h), self.proj_k(h), self.proj_v(h)
        # [B, H x W, H x W]
        w = torch.softmax(q @ k.permute(0, 2, 1) * (C**-0.5), dim=-1)
        # [B, H x W, C] > [B, H, W, C] > [B, C, H, W]
        h = self.proj_out(w @ v).view(B, H, W, C).permute(0, 3, 1, 2)
        if self.skip_rescale:
            return (x + h) * (2**-0.5)
        return x + h


class Conv3x3(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, init_scale: float = 1.0):
        super().__init__(
            in_channels,
            out_channels,
            (3, 3),
            padding=(1, 1),
        )
        with torch.no_grad():
            # [out_channels, in_channels, kernels, kernels]
            self.weight.copy_(
                default_init(out_channels, in_channels, 3, 3, scale=init_scale)
            )
            self.bias.zero_()


class Conv1x1(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, init_scale: float = 1.0):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=1,
        )
        with torch.no_grad():
            self.weight.copy_(
                default_init(out_channels, in_channels, 1, 1, scale=init_scale)
            )
            self.bias.zero_()


class ResnetBlockBigGANpp(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        up: bool = False,
        down: bool = False,
        dropout: float = 0.1,
        init_scale: float = 0.0,
        skip_rescale: bool = True,
    ):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(
            min(in_channels // 4, 32), in_channels, eps=1e-6
        )

        self.up = up
        self.down = down
        self.proj_1 = Conv3x3(in_channels, out_channels)
        self.proj_temb = nn.Linear(emb_channels, out_channels)
        with torch.no_grad():
            self.proj_temb.weight.copy_(default_init(out_channels, emb_channels))
            self.proj_temb.bias.zero_()

        self.groupnorm_2 = nn.GroupNorm(
            min(out_channels // 4, 32), out_channels, eps=1e-6
        )
        self.dropout = nn.Dropout(dropout)

        self.proj_2 = Conv3x3(out_channels, out_channels, init_scale=init_scale)
        self.proj_res = None
        if in_channels != out_channels or up or down:
            self.proj_res = Conv1x1(in_channels, out_channels)

        self.skip_rescale = skip_rescale

    def _upsample(self, x: torch.Tensor, factor: int = 2) -> torch.Tensor:
        B, C, H, W = x.shape
        # [B, C, H, 1, W, 1] > [B, C, H, F, W, F] > [B, C, H x F, W x F]
        return (
            x[..., None, :, None]
            .tile(1, 1, 1, factor, 1, factor)
            .view(B, C, H * factor, W * factor)
        )

    def _downsample(self, x: torch.Tensor, factor: int = 2) -> torch.tensor:
        B, C, H, W = x.shape
        # [B, C, H // F, F, W // F, F] > [B, C, H // F, W // F]
        return x.view(B, C, H // factor, factor, W // factor, factor).mean(dim=[3, 5])

    def forward(
        self, x: torch.Tensor, temb: torch.Tensor | None = None
    ) -> torch.Tensor:
        # [B, C, H, W]
        h = F.silu(self.groupnorm_1.forward(x))
        # [B, C, H', W']
        if self.up:
            h, x = self._upsample(h, 2), self._upsample(x, 2)
        if self.down:
            h, x = self._downsample(h, 2), self._downsample(x, 2)
        # [B, C, H', W']
        h = self.proj_1.forward(h)
        if temb is not None:
            h = h + self.proj_temb.forward(temb)[..., None, None]
        # [B, C, H', W']
        h = F.silu(self.groupnorm_2.forward(h))
        # [B, C, H', W']
        h = self.proj_2.forward(self.dropout(h))
        if self.proj_res:
            x = self.proj_res.forward(x)
        # [B, C, H', W']
        if self.skip_rescale:
            return (x + h) * (2**-0.5)
        return x + h


class DDPMpp(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        nf: int = 128,
        ch_mult: list[int] = [1, 2, 2, 2],
        attn_resolutions: list[int] = [16],
        num_res_blocks: int = 4,
        init_scale: float = 0.0,
        skip_rescale: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.nf = nf

        _temb_channels = nf * 4
        self.proj_temb = nn.Sequential(
            nn.Linear(nf, _temb_channels),
            nn.SiLU(),
            nn.Linear(_temb_channels, _temb_channels),
        )
        with torch.no_grad():
            self.proj_temb[0].weight.copy_(default_init(_temb_channels, nf))
            self.proj_temb[0].bias.zero_()
            self.proj_temb[2].weight.copy_(default_init(_temb_channels, _temb_channels))
            self.proj_temb[2].bias.zero_()

        self.proj_in = Conv3x3(in_channels, nf)

        _channels = [nf * mult for mult in ch_mult]
        _fst, *_, _bottleneck = _channels
        self.down_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ResnetBlockBigGANpp(
                            in_channels=channels if i > 0 else prev,
                            out_channels=channels,
                            emb_channels=_temb_channels,
                            dropout=dropout,
                            init_scale=init_scale,
                            skip_rescale=skip_rescale,
                        )
                        for i in range(num_res_blocks)
                    ]
                )
                for prev, channels in zip([nf] + _channels, _channels)
            ]
        )

        _resolutions = [resolution * (2**-i) for i in range(len(_channels))]
        self.down_attns = nn.ModuleDict(
            {
                str(i): nn.ModuleList(
                    [
                        AttnBlockpp(channels, skip_rescale, init_scale)
                        for _ in range(num_res_blocks)
                    ]
                )
                for i, (res, channels) in enumerate(zip(_resolutions, _channels))
                if res in attn_resolutions
            }
        )

        self.downsamplers = nn.ModuleList(
            [
                ResnetBlockBigGANpp(
                    in_channels=channels,
                    out_channels=channels,
                    emb_channels=_temb_channels,
                    down=True,
                    dropout=dropout,
                    init_scale=init_scale,
                    skip_rescale=skip_rescale,
                )
                for channels in _channels[:-1]
            ]
        )

        self.bottleneck_res = nn.ModuleList(
            [
                ResnetBlockBigGANpp(
                    in_channels=_bottleneck,
                    out_channels=_bottleneck,
                    emb_channels=_temb_channels,
                    dropout=dropout,
                    init_scale=init_scale,
                    skip_rescale=skip_rescale,
                )
                for _ in range(2)
            ]
        )
        self.bottleneck_attn = AttnBlockpp(_bottleneck, skip_rescale, init_scale)

        _backwards = _channels[::-1]
        self.up_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ResnetBlockBigGANpp(
                            in_channels=channels
                            + (
                                channels
                                if 0 < i < num_res_blocks
                                else (prev if i == 0 else next_)
                            ),
                            out_channels=channels,
                            emb_channels=_temb_channels,
                            dropout=dropout,
                            init_scale=init_scale,
                            skip_rescale=skip_rescale,
                        )
                        for i in range(num_res_blocks + 1)
                    ]
                )
                for prev, channels, next_ in zip(
                    [_bottleneck] + _backwards,
                    _backwards,
                    _backwards[1:] + [nf],
                )
            ]
        )

        _resolutions = _resolutions[::-1]
        self.up_attns = nn.ModuleDict(
            {
                str(i): AttnBlockpp(channels, skip_rescale, init_scale)
                for i, (res, channels) in enumerate(zip(_resolutions, _backwards))
                if res in attn_resolutions
            }
        )

        self.upsamplers = nn.ModuleList(
            [
                ResnetBlockBigGANpp(
                    in_channels=channels,
                    out_channels=channels,
                    emb_channels=_temb_channels,
                    up=True,
                    dropout=dropout,
                    init_scale=init_scale,
                    skip_rescale=skip_rescale,
                )
                for channels in _backwards[:-1]
            ]
        )

        self.postproc = nn.Sequential(
            nn.GroupNorm(min(_fst // 4, 32), _fst, eps=1e-6),
            nn.SiLU(),
            Conv3x3(_fst, in_channels, init_scale),
        )

    def forward(self, x: torch.Tensor, time_cond: torch.Tensor) -> torch.Tensor:
        # [B, E]
        temb = get_timestep_embedding(time_cond.to(x), self.nf)
        # [B, E x 4]
        temb = self.proj_temb(temb)

        h = self.proj_in.forward(x)
        hs = [h]
        for i, down_blocks in enumerate(self.down_blocks):
            _attns = [None for _ in down_blocks]
            if str(i) in self.down_attns:
                _attns = self.down_attns[str(i)]
            for block, attn in zip(down_blocks, _attns):
                h = block.forward(h, temb)
                if attn:
                    h = attn.forward(h)
                hs.append(h)

            if i < len(self.downsamplers):
                h = self.downsamplers[i].forward(h, temb)
                hs.append(h)

        h = self.bottleneck_res[0].forward(h, temb)
        h = self.bottleneck_attn.forward(h)
        h = self.bottleneck_res[1].forward(h, temb)

        for i, up_blocks in enumerate(self.up_blocks):
            for block in up_blocks:
                h = block.forward(torch.cat([h, hs.pop()], dim=1), temb)
            if str(i) in self.up_attns:
                h = self.up_attns[str(i)].forward(h)
            if i < len(self.upsamplers):
                h = self.upsamplers[i].forward(h, temb)

        return self.postproc(h)
