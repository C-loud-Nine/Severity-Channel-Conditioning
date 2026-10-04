"""Restormer cross-backbone arm: first convolution widened 3 -> 4 channels (432 added weights,
zero-initialized). A pre-hook feeds [x; s] to patch_embed while the restorer's global residual adds x.
Requires the official Restormer code (https://github.com/swz30/Restormer) on PYTHONPATH.
"""
import torch
import torch.nn as nn


def make_restormer(ckpt_path=None, device="cuda"):
    from basicsr.models.archs.restormer_arch import Restormer
    m = Restormer().to(device)
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location="cpu")
        m.load_state_dict(ck.get("params", ck.get("params_ema", ck)), strict=True)
    return m


class SevRestormer(nn.Module):
    def __init__(self, base):
        super().__init__()
        old = base.patch_embed.proj
        new = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding,
                        bias=old.bias is not None).to(old.weight.device)
        with torch.no_grad():
            new.weight.zero_(); new.weight[:, :3] = old.weight.data
            if old.bias is not None: new.bias.copy_(old.bias.data)
        base.patch_embed.proj = new
        self.base, self._x4 = base, None
        base.patch_embed.register_forward_pre_hook(lambda m, a: (self._x4,) if self._x4 is not None else None)

    def forward(self, x4):
        self._x4 = x4
        try:
            return self.base(x4[:, :3].contiguous())
        finally:
            self._x4 = None
