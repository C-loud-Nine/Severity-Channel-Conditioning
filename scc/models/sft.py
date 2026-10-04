"""Feature-level spatial variant (SFT-style), as executed.

Per block: severity map bilinearly resized to the block resolution -> conv 1->32->32->2C (3x3, SiLU)
-> spatially varying (1 + g, beta) on all 38 TransformerBlock outputs. Final conv Xavier gain 0.01,
zero bias. Trained with L1 only; oracle severity map at inference. Adds 4,053,536 parameters.
"""
import torch.nn as nn
import torch.nn.functional as F
from .diffir_scc import transformer_blocks


class SFTGen(nn.Module):
    def __init__(self, num_channels, hidden=32):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(1, hidden, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(hidden, 2 * num_channels, 3, padding=1))
        nn.init.xavier_uniform_(self.body[-1].weight, gain=0.01)
        nn.init.zeros_(self.body[-1].bias)
        self.C = num_channels

    def forward(self, sev, out_hw):
        gb = self.body(F.interpolate(sev, size=out_hw, mode="bilinear", align_corners=False))
        return 1.0 + gb[:, :self.C], gb[:, self.C:]


class SFTDiffIR(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        blocks = transformer_blocks(base_model)
        self.sft_gens = nn.ModuleList([SFTGen(c) for (_, _, c) in blocks])
        self._sev = None
        for i, (_, mod, _) in enumerate(blocks):
            mod.register_forward_hook(self._make_hook(i))

    def _make_hook(self, idx):
        def hook(module, args, output):
            if self._sev is None:
                return None
            feat, rest = (output[0], output[1:]) if isinstance(output, (list, tuple)) else (output, None)
            g, b = self.sft_gens[idx](self._sev, feat.shape[-2:])
            feat = feat * g.to(feat.dtype) + b.to(feat.dtype)
            return feat if rest is None else (feat, *rest)
        return hook

    def forward(self, blur, sev):
        self._sev = sev
        try:
            out = self.base(blur)
            return out[0] if isinstance(out, (list, tuple)) else out
        finally:
            self._sev = None
