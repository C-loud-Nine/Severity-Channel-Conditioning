"""DiffIR S2 backbone and the input-level SCC surgery (function-preserving initialization).

Only the prior encoder's first projection (condition.E[0]) is widened. After pixel-unshuffle by 4 the
3 image channels give 48 inputs and the severity channel 16 more: W'[:, :48] = W, W'[:, 48:64] = 0,
so the widened network equals the DAB at initialization. The restorer G stays 3-channel.
Requires the official DiffIR code (https://github.com/Zj-BinXia/DiffIR) on PYTHONPATH.
"""
import torch
import torch.nn as nn


def make_diffir(ckpt_path=None, device="cuda"):
    from DiffIR.archs.S2_arch import DiffIRS2
    m = DiffIRS2(n_encoder_res=5, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=[3, 5, 6, 6], num_refinement_blocks=4, heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2, bias=False, LayerNorm_type="WithBias",
                 n_denoise_res=1, linear_start=0.1, linear_end=0.99, timesteps=4).to(device)
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location="cpu")
        m.load_state_dict(ck.get("params_ema", ck), strict=True)
    return m


def widen_cpen(base: nn.Module) -> nn.Module:
    old = base.condition.E[0]
    new = nn.Conv2d(64, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=old.bias is not None).to(old.weight.device)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :48] = old.weight.data          # W'[:, 0:48] = W ; W'[:, 48:64] = 0
        if old.bias is not None:
            new.bias.copy_(old.bias.data)
    base.condition.E[0] = new
    assert base.diffusion.condition is base.condition
    return base


class SevDiffIR(nn.Module):
    """Feeds [x; s] to the prior encoder (CPEN) and plain x to the restorer, via a forward pre-hook."""

    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self._x4 = None
        self.base.condition.register_forward_pre_hook(
            lambda module, args: (self._x4,) if self._x4 is not None else None)

    def forward(self, x4):
        self._x4 = x4
        try:
            out = self.base(x4[:, :3].contiguous())
            return out[0] if isinstance(out, (list, tuple)) else out
        finally:
            self._x4 = None


def block_channels(block):
    for sm in block.modules():
        if sm.__class__.__name__ in ("WithBias_LayerNorm", "BiasFree_LayerNorm") and getattr(sm, "weight", None) is not None:
            return sm.weight.shape[0]
        if isinstance(sm, nn.LayerNorm):
            return sm.normalized_shape[0]
    return None


def transformer_blocks(base):
    """All 38 TransformerBlocks of the generator (encoder, bottleneck, decoder, refinement)."""
    info = [(n, m, block_channels(m)) for n, m in base.G.named_modules()
            if m.__class__.__name__ == "TransformerBlock"]
    assert len(info) == 38, len(info)
    return info
