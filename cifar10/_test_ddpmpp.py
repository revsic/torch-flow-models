import sys
import contextlib

import flax.linen
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn

from ddpmpp import (
    get_timestep_embedding,
    default_init,
    NIN,
    AttnBlockpp,
    Conv1x1,
    Conv3x3,
    ResnetBlockBigGANpp,
    DDPMpp,
)

torch.manual_seed(0)


@contextlib.contextmanager
def from_score_sde():
    sys.path.append("/workspace/repos/score_sde")
    yield
    sys.path.pop()


with from_score_sde():
    from configs.subvp.cifar10_ddpmpp_continuous import get_config

    config = get_config()
    assert config.model.resblock_type == "biggan"

    from models.ncsnpp import (
        default_initializer as flax_default_initializer,
        ResnetBlockBigGAN as flax_ResnetBlockBigGAN,
        conv1x1 as flax_conv1x1,
        conv3x3 as flax_conv3x3,
        NCSNpp as flax_ncsnpp,
    )
    from models.layerspp import AttnBlockpp as flax_AttnBlockpp
    from models.layers import (
        NIN as flax_NIN,
        get_timestep_embedding as flax_get_timestep_embedding,
        ddpm_conv1x1,
        ddpm_conv3x3,
    )

    assert flax_conv1x1 == ddpm_conv1x1
    assert flax_conv3x3 == ddpm_conv3x3


def _test(a: jnp.ndarray, b: torch.tensor, threshold: float = 1e-7):
    a = torch.tensor(a)
    e = (a - b).abs().amax()
    assert e <= threshold, (e, threshold)


def _init_test(
    a: jnp.ndarray,
    b: torch.Tensor,
    test_numel: bool = True,
    threshold: float = 1e-4,
):
    assert np.prod(a.shape).item() == b.numel()
    assert not test_numel or b.numel() > 1000000
    _test(a.mean(), b.mean(), threshold)
    _test(a.std(), b.std(), threshold)


@torch.no_grad()
def _test_NIN(params, nin: NIN, test_numel: bool = True, threshold: float = 1e-4):
    _init_test(params["W"], nin.W, test_numel, threshold)
    _test(params["b"], nin.b, 0)


@torch.no_grad()
def _copy_NIN(params, nin: NIN):
    nin.W.copy_(torch.tensor(params["W"].T))
    nin.b.copy_(torch.tensor(params["b"]))


@torch.no_grad()
def _test_GroupNorm(params, groupnorm: nn.GroupNorm):
    _test(params["scale"], groupnorm.weight, 0)
    _test(params["bias"], groupnorm.bias, 0)


@torch.no_grad()
def _test_AttnBlockpp(
    params, attn: AttnBlockpp, test_numel: bool = True, threshold: float = 1e-4
):
    _test_GroupNorm(params["GroupNorm_0"], attn.groupnorm)
    _test_NIN(params["NIN_0"], attn.proj_q, test_numel, threshold)
    _test_NIN(params["NIN_1"], attn.proj_k, test_numel, threshold)
    _test_NIN(params["NIN_2"], attn.proj_v, test_numel, threshold)
    _test_NIN(params["NIN_3"], attn.proj_out, test_numel, threshold)


@torch.no_grad()
def _copy_GroupNorm(params, groupnorm: nn.GroupNorm):
    groupnorm.weight.copy_(torch.tensor(params["scale"]))
    groupnorm.bias.copy_(torch.tensor(params["bias"]))


@torch.no_grad()
def _copy_AttnBlockpp(params, attn: AttnBlockpp):
    _copy_GroupNorm(params["GroupNorm_0"], attn.groupnorm)
    _copy_NIN(params["NIN_0"], attn.proj_q)
    _copy_NIN(params["NIN_1"], attn.proj_k)
    _copy_NIN(params["NIN_2"], attn.proj_v)
    _copy_NIN(params["NIN_3"], attn.proj_out)


@torch.no_grad()
def _test_Conv(
    params,
    conv: Conv1x1 | Conv3x3,
    test_numel: bool = True,
    threshold: float = 1e-4,
):
    _init_test(params["kernel"], conv.weight, test_numel, threshold)
    _test(params["bias"], conv.bias, 0.0)


@torch.no_grad()
def _copy_Conv(params, conv: Conv1x1 | Conv3x3):
    conv.weight.copy_(torch.tensor(params["kernel"]).permute(3, 2, 0, 1))
    conv.bias.copy_(torch.tensor(params["bias"]))


@torch.no_grad()
def _test_Dense(
    params,
    proj: nn.Linear,
    test_numel: bool = True,
    threshold: float = 1e-4,
):
    _init_test(params["kernel"], proj.weight, test_numel, threshold)
    _test(params["bias"], proj.bias)


@torch.no_grad()
def _copy_Dense(params, proj: nn.Linear):
    proj.weight.copy_(torch.tensor(params["kernel"]).T)
    proj.bias.copy_(torch.tensor(params["bias"]))


@torch.no_grad()
def _test_ResnetBlockBigGANpp(
    params,
    block: ResnetBlockBigGANpp,
    test_numel: bool = True,
    threshold: float = 1e-4,
):
    _test_GroupNorm(params["GroupNorm_0"], block.groupnorm_1)
    _test_Conv(params["Conv_0"], block.proj_1, test_numel, threshold)
    _test_Dense(params["Dense_0"], block.proj_temb, False, max(5e-4, threshold))
    _test_GroupNorm(params["GroupNorm_1"], block.groupnorm_2)
    _test_Conv(params["Conv_1"], block.proj_2, test_numel, threshold)
    assert ("Conv_2" in params) == (getattr(block, "proj_res", None) is not None)
    if "Conv_2" in params:
        _test_Conv(params["Conv_2"], block.proj_res, test_numel, threshold)


@torch.no_grad()
def _copy_ResnetBlockBigGANpp(params, block: ResnetBlockBigGANpp):
    _copy_GroupNorm(params["GroupNorm_0"], block.groupnorm_1)
    _copy_Conv(params["Conv_0"], block.proj_1)
    _copy_Dense(params["Dense_0"], block.proj_temb)
    _copy_GroupNorm(params["GroupNorm_1"], block.groupnorm_2)
    _copy_Conv(params["Conv_1"], block.proj_2)
    if "Conv_2" in params:
        _copy_Conv(params["Conv_2"], block.proj_res)


def _match_attn(params, model: DDPMpp):
    attns = [attn for _, _attns in model.down_attns.items() for attn in _attns]
    attns.append(model.bottleneck_attn)
    attns.extend([attn for _, attn in model.up_attns.items()])
    params = [p for k, p in params.items() if k.startswith("AttnBlockpp_")]
    assert len(attns) == len(params)
    yield from zip(params, attns)


def _match_resblock(params, model: DDPMpp):
    blocks = []
    for i, down_blocks in enumerate(model.down_blocks):
        blocks.extend(down_blocks)
        if i < len(model.downsamplers):
            blocks.append(model.downsamplers[i])
    blocks.extend(model.bottleneck_res)
    for i, up_blocks in enumerate(model.up_blocks):
        blocks.extend(up_blocks)
        if i < len(model.upsamplers):
            blocks.append(model.upsamplers[i])
    params = [p for k, p in params.items() if k.startswith("ResnetBlockBigGANpp_")]
    assert len(blocks) == len(params), (len(blocks), len(params))
    yield from zip(params, blocks)


@torch.no_grad()
def _test_DDPMpp(params, model: DDPMpp, config):
    assert config.model.nonlinearity == "swish"
    assert config.model.conditional
    assert not config.model.fir
    assert config.model.resblock_type == "biggan"
    assert (
        config.model.progressive == "none" and config.model.progressive_input == "none"
    )
    assert config.model.embedding_type == "positional"

    _test_Dense(params["Dense_0"], model.proj_temb[0], False, 5e-4)
    _test_Dense(params["Dense_1"], model.proj_temb[2], False, 5e-4)
    _test_Conv(params["Conv_0"], model.proj_in, False, 1e-3)
    for param, attn in _match_attn(params, model):
        _test_AttnBlockpp(param, attn, False, 5e-4)
    for param, block in _match_resblock(params, model):
        _test_ResnetBlockBigGANpp(param, block, False, 1e-3)
    _test_GroupNorm(params["GroupNorm_0"], model.postproc[0])
    _test_Conv(params["Conv_1"], model.postproc[2], False, 5e-4)


@torch.no_grad()
def _copy_DDPMpp(params, model: DDPMpp):
    _copy_Dense(params["Dense_0"], model.proj_temb[0])
    _copy_Dense(params["Dense_1"], model.proj_temb[2])
    _copy_Conv(params["Conv_0"], model.proj_in)
    for param, attn in _match_attn(params, model):
        _copy_AttnBlockpp(param, attn)
    for param, block in _match_resblock(params, model):
        _copy_ResnetBlockBigGANpp(param, block)
    _copy_GroupNorm(params["GroupNorm_0"], model.postproc[0])
    _copy_Conv(params["Conv_1"], model.postproc[2])


_test(
    temb := flax_get_timestep_embedding(jnp.linspace(0, 1, 30), 128),
    get_timestep_embedding(torch.linspace(0, 1, 30), 128),
)

rng = jax.random.PRNGKey(0)

gen = torch.Generator().manual_seed(0)
t = flax_default_initializer()(rng, (3, 3, 1024, 1000))
s = default_init(1000, 1024, 3, 3, generator=gen)
_init_test(t, s)

x = jax.random.normal(rng, (3, 2048))
t = flax_NIN(1024)
s = NIN(2048, 1024)
params = t.init({"params": rng}, x)
_test_NIN(params["params"], s)
_copy_NIN(params["params"], s)
_test(t.apply(params, x), s.forward(torch.tensor(x)), 1e-5)

x = jax.random.normal(rng, (3, 32, 32, 1024))
t = flax_AttnBlockpp(skip_rescale=True)
s = AttnBlockpp(1024, skip_rescale=True)
params = t.init({"params": rng}, x)
_test_AttnBlockpp(params["params"], s)
_copy_AttnBlockpp(params["params"], s)
_test(
    t.apply(params, x),
    s.forward(torch.tensor(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1),
    1e-6,
)

x = jax.random.normal(rng, (3, 32, 32, 1024))


def _run_test_ResnetBlockBigGAN(t, s):
    params = t.init({"params": rng}, x, temb[: x.shape[0]])
    _test_ResnetBlockBigGANpp(params["params"], s)
    _copy_ResnetBlockBigGANpp(params["params"], s)
    _test(
        t.apply(params, x, temb[: x.shape[0]]),
        s.forward(
            torch.tensor(x).permute(0, 3, 1, 2), torch.tensor(temb[: x.shape[0]])
        ).permute(0, 2, 3, 1),
        5e-6,
    )


_run_test_ResnetBlockBigGAN(
    flax_ResnetBlockBigGAN(act=flax.linen.swish, dropout=0.0),
    ResnetBlockBigGANpp(1024, 1024, emb_channels=128, dropout=0.0),
)
_run_test_ResnetBlockBigGAN(
    flax_ResnetBlockBigGAN(act=flax.linen.swish, dropout=0.0, up=True),
    ResnetBlockBigGANpp(1024, 1024, emb_channels=128, dropout=0.0, up=True),
)
_run_test_ResnetBlockBigGAN(
    flax_ResnetBlockBigGAN(act=flax.linen.swish, dropout=0.0, down=True),
    ResnetBlockBigGANpp(1024, 1024, emb_channels=128, dropout=0.0, down=True),
)

x = jax.random.normal(rng, (3, 32, 32, 3))
c = jnp.linspace(0, 1, 3)
config = get_config()
config.model.dropout = 0.0
t = flax_ncsnpp(config)
s = DDPMpp(32, 3, dropout=0.0)
params = t.init({"params": rng}, x, c)
_test_DDPMpp(params["params"], s, config)
_copy_DDPMpp(params["params"], s)
_test(
    t.apply(params, x, c),
    s.forward(torch.tensor(x).permute(0, 3, 1, 2), torch.tensor(c)).permute(0, 2, 3, 1),
    1e-6,
)
