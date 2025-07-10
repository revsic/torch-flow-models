# type: ignore
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(
    timesteps: torch.Tensor, dim: int, max_positions: int = 10000, scale: float = 1.0
) -> torch.Tensor:
    assert dim % 2 == 0
    emb = np.log(max_positions) / (dim // 2 - 1)
    # [E // 2]
    emb = torch.exp(
        torch.arange(dim // 2, dtype=torch.float32, device=timesteps.device) * -emb
    )
    # [T, E // 2]
    emb = timesteps[:, None] * emb[None] * scale
    # [T, E]
    return torch.cat([emb.sin(), emb.cos()], axis=1)


def default_init(
    *shape: int,
    scale: float = 1.0,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    out_size, in_size = shape[0], shape[1]
    field_size = math.prod(shape) / in_size / out_size
    fan_in, fan_out = in_size * field_size, out_size * field_size
    denominator = (fan_in + fan_out) / 2
    if scale == 0.0:
        scale = 1e-10
    rand = torch.rand(*shape, dtype=dtype, generator=generator)
    return (rand * 2 - 1) * np.sqrt(3 * scale / denominator)


class NIN(nn.Module):
    def __init__(self, in_dim: int, num_units: int, init_scale: float = 0.1):
        super().__init__()
        self.W = nn.Parameter(default_init(num_units, in_dim, scale=init_scale))
        self.b = nn.Parameter(torch.zeros(num_units))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [..., in_dim] > [..., num_units]
        return x @ self.W.T + self.b


class GroupNorm(nn.GroupNorm):
    def __init__(self, *args, force_on_32bit: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_on_32bit = force_on_32bit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.force_on_32bit:
            return super().forward(x)

        return F.group_norm(
            x.float(),
            self.num_groups,
            self.weight.float(),
            self.bias.float(),
            self.eps,
        ).to(x.dtype)


class AttnBlockpp(nn.Module):
    def __init__(
        self,
        channels: int,
        skip_rescale: bool = True,
        init_scale: float = 0.0,
        _force_norm_32bit: bool = False,
    ):
        super().__init__()
        self.groupnorm = GroupNorm(
            min(channels // 4, 32), channels, eps=1e-6, force_on_32bit=_force_norm_32bit
        )

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
        use_shift_scale_norm: bool = False,
        use_double_norm: bool = False,
        _force_norm_32bit: bool = False,
    ):
        super().__init__()
        self.groupnorm_1 = GroupNorm(
            min(in_channels // 4, 32),
            in_channels,
            eps=1e-6,
            force_on_32bit=_force_norm_32bit,
        )

        self.up = up
        self.down = down
        self.use_shift_scale_norm = use_shift_scale_norm
        self.use_double_norm = use_double_norm

        self.proj_1 = Conv3x3(in_channels, out_channels)
        _out_channels = out_channels * 2 if use_shift_scale_norm else out_channels
        self.proj_temb = nn.Linear(emb_channels, _out_channels)
        with torch.no_grad():
            self.proj_temb.weight.copy_(default_init(_out_channels, emb_channels))
            self.proj_temb.bias.zero_()

        self.groupnorm_2 = GroupNorm(
            min(out_channels // 4, 32),
            out_channels,
            eps=1e-6,
            force_on_32bit=_force_norm_32bit,
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
        ## FYI. F.interpolate(mode="nearest") on openai/consistency_models
        return (
            x[..., None, :, None]
            .tile(1, 1, 1, factor, 1, factor)
            .view(B, C, H * factor, W * factor)
        )

    def _downsample(self, x: torch.Tensor, factor: int = 2) -> torch.tensor:
        return F.avg_pool2d(x, factor, factor)

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
            temb = self.proj_temb.forward(temb)[..., None, None]
            if not self.use_shift_scale_norm:
                # [B, C, 1, 1] > [B,, C, H', W']
                h = self.groupnorm_2.forward(h + temb)
            else:
                # [B, C, 1, 1], [B, C, 1, 1]
                scale, shift = temb.chunk(2, dim=1)
                scale = scale + 1
                if self.use_double_norm:
                    _pixel_norm = (
                        lambda p: p
                        * (p.square().mean(dim=1, keepdims=True) + 1e-8).rsqrt()
                    )
                    scale = _pixel_norm(scale)
                    shift = _pixel_norm(shift)
                # [B, C, H', W']
                h = self.groupnorm_2.forward(h) * scale + shift
        # [B, C, H', W']
        h = F.silu(h)
        # [B, C, H', W']
        h = self.proj_2.forward(self.dropout(h))
        if self.proj_res:
            x = self.proj_res.forward(x)
        # [B, C, H', W']
        if self.skip_rescale:
            return (x + h) * (2**-0.5)
        return x + h


@dataclass
class ModelConfig:
    resolution: int
    in_channels: int
    nf: int = 128
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 2, 2])
    attn_resolutions: list[int] = field(default_factory=lambda: [16])
    num_res_blocks: int = 4
    init_scale: float = 0.0
    skip_rescale: bool = True
    dropout: float = 0.1
    pe_scale: float = 1.0
    use_shift_scale_norm: bool = False
    use_double_norm: bool = False
    n_classes: int | None = None
    _force_norm_32bit: bool = False


class DDPMpp(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.nf = config.nf
        self.pe_scale = config.pe_scale

        self.proj_label = None
        if config.n_classes:
            self.proj_label = nn.Embedding(config.n_classes, config.nf)

        _temb_channels = config.nf * 4
        self.proj_temb = nn.Sequential(
            nn.Linear(config.nf, _temb_channels),
            nn.SiLU(),
            nn.Linear(_temb_channels, _temb_channels),
        )
        with torch.no_grad():
            self.proj_temb[0].weight.copy_(default_init(_temb_channels, config.nf))
            self.proj_temb[0].bias.zero_()
            self.proj_temb[2].weight.copy_(default_init(_temb_channels, _temb_channels))
            self.proj_temb[2].bias.zero_()

        self.proj_in = Conv3x3(config.in_channels, config.nf)

        _channels = [config.nf * mult for mult in config.ch_mult]
        _fst, *_, _bottleneck = _channels
        self.down_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ResnetBlockBigGANpp(
                            in_channels=channels if i > 0 else prev,
                            out_channels=channels,
                            emb_channels=_temb_channels,
                            dropout=config.dropout,
                            init_scale=config.init_scale,
                            skip_rescale=config.skip_rescale,
                            use_shift_scale_norm=config.use_shift_scale_norm,
                            use_double_norm=config.use_double_norm,
                            _force_norm_32bit=config._force_norm_32bit,
                        )
                        for i in range(config.num_res_blocks)
                    ]
                )
                for prev, channels in zip([config.nf] + _channels, _channels)
            ]
        )

        _resolutions = [config.resolution * (2**-i) for i in range(len(_channels))]
        self.down_attns = nn.ModuleDict(
            {
                str(i): nn.ModuleList(
                    [
                        AttnBlockpp(
                            channels,
                            config.skip_rescale,
                            config.init_scale,
                            _force_norm_32bit=config._force_norm_32bit,
                        )
                        for _ in range(config.num_res_blocks)
                    ]
                )
                for i, (res, channels) in enumerate(zip(_resolutions, _channels))
                if res in config.attn_resolutions
            }
        )

        self.downsamplers = nn.ModuleList(
            [
                ResnetBlockBigGANpp(
                    in_channels=channels,
                    out_channels=channels,
                    emb_channels=_temb_channels,
                    down=True,
                    dropout=config.dropout,
                    init_scale=config.init_scale,
                    skip_rescale=config.skip_rescale,
                    use_shift_scale_norm=config.use_shift_scale_norm,
                    use_double_norm=config.use_double_norm,
                    _force_norm_32bit=config._force_norm_32bit,
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
                    dropout=config.dropout,
                    init_scale=config.init_scale,
                    skip_rescale=config.skip_rescale,
                    use_shift_scale_norm=config.use_shift_scale_norm,
                    use_double_norm=config.use_double_norm,
                    _force_norm_32bit=config._force_norm_32bit,
                )
                for _ in range(2)
            ]
        )
        self.bottleneck_attn = AttnBlockpp(
            _bottleneck,
            config.skip_rescale,
            config.init_scale,
            _force_norm_32bit=config._force_norm_32bit,
        )

        _backwards = _channels[::-1]
        self.up_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ResnetBlockBigGANpp(
                            in_channels=channels
                            + (
                                channels
                                if 0 < i < config.num_res_blocks
                                else (prev if i == 0 else next_)
                            ),
                            out_channels=channels,
                            emb_channels=_temb_channels,
                            dropout=config.dropout,
                            init_scale=config.init_scale,
                            skip_rescale=config.skip_rescale,
                            use_shift_scale_norm=config.use_shift_scale_norm,
                            use_double_norm=config.use_double_norm,
                            _force_norm_32bit=config._force_norm_32bit,
                        )
                        for i in range(config.num_res_blocks + 1)
                    ]
                )
                for prev, channels, next_ in zip(
                    [_bottleneck] + _backwards,
                    _backwards,
                    _backwards[1:] + [config.nf],
                )
            ]
        )

        _resolutions = _resolutions[::-1]
        self.up_attns = nn.ModuleDict(
            {
                str(i): AttnBlockpp(
                    channels,
                    config.skip_rescale,
                    config.init_scale,
                    _force_norm_32bit=config._force_norm_32bit,
                )
                for i, (res, channels) in enumerate(zip(_resolutions, _backwards))
                if res in config.attn_resolutions
            }
        )

        self.upsamplers = nn.ModuleList(
            [
                ResnetBlockBigGANpp(
                    in_channels=channels,
                    out_channels=channels,
                    emb_channels=_temb_channels,
                    up=True,
                    dropout=config.dropout,
                    init_scale=config.init_scale,
                    skip_rescale=config.skip_rescale,
                    use_shift_scale_norm=config.use_shift_scale_norm,
                    use_double_norm=config.use_double_norm,
                    _force_norm_32bit=config._force_norm_32bit,
                )
                for channels in _backwards[:-1]
            ]
        )

        self.postproc = nn.Sequential(
            GroupNorm(
                min(_fst // 4, 32),
                _fst,
                eps=1e-6,
                force_on_32bit=config._force_norm_32bit,
            ),
            nn.SiLU(),
            Conv3x3(_fst, config.in_channels, config.init_scale),
        )

        self._debug_purpose = {}

    def forward(
        self,
        x: torch.Tensor,
        time_cond: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert (label is None) == (self.proj_label is None)
        # [B, E]
        temb = get_timestep_embedding(time_cond.to(x), self.nf, scale=self.pe_scale)
        if label is not None:
            temb = temb + self.proj_label.forward(label)
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

        output = self.postproc(h)

        with torch.no_grad():
            self._debug_purpose["ddpmpp/before_postproc"] = (
                h.square().mean().sqrt().detach().cpu().item()
            )
            self._debug_purpose["ddpmpp/after_postproc"] = (
                output.square().mean().sqrt().detach().cpu().item()
            )

        return output
