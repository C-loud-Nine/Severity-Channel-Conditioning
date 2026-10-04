"""Feature-level FiLM arm, as executed.

Five z-scored metadata fields -> MLP 5->128->128->128 (SiLU) -> per-block linear 128 -> 2C,
(1 + g, beta) applied to the output of all 38 TransformerBlocks. Xavier gain 0.01, zero bias.
Training-only auxiliary head on the 384-channel bottleneck (latent.0): mean L1 to the z-scored
metadata, weight 0.05. Embedding + generators add 1,680,864 parameters.
"""
import torch.nn as nn
from .diffir_scc import transformer_blocks, block_channels

META_FIELDS = ["motion", "isp_diff", "blur_window", "blur_duration_ms", "noise_estimate"]


class MetaEmbed(nn.Module):
    def __init__(self, in_dim=5, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, embed_dim), nn.SiLU(),
                                 nn.Linear(embed_dim, embed_dim), nn.SiLU(),
                                 nn.Linear(embed_dim, embed_dim))

    def forward(self, m):
        return self.net(m)


class FiLMGen(nn.Module):
    def __init__(self, embed_dim, num_channels):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 2 * num_channels)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)
        nn.init.zeros_(self.linear.bias)
        self.C = num_channels

    def forward(self, e):
        gb = self.linear(e)
        return 1.0 + gb[..., :self.C], gb[..., self.C:]


class AuxHead(nn.Module):
    def __init__(self, in_channels, out_dim=5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(in_channels, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, out_dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat):
        return self.mlp(self.pool(feat).flatten(1))


class FiLMDiffIR(nn.Module):
    def __init__(self, base_model, embed_dim=128, meta_dim=5):
        super().__init__()
        self.base = base_model
        blocks = transformer_blocks(base_model)
        latent = dict(base_model.G.named_modules())["latent.0"]
        self.embed = MetaEmbed(meta_dim, embed_dim)
        self.film_gens = nn.ModuleList([FiLMGen(embed_dim, c) for (_, _, c) in blocks])
        self.aux_head = AuxHead(block_channels(latent), meta_dim)
        self._gbs, self._latent = [], None
        for i, (_, mod, c) in enumerate(blocks):
            mod.register_forward_hook(self._make_hook(i, c))
        latent.register_forward_hook(lambda m, a, o: setattr(self, "_latent", o[0] if isinstance(o, (list, tuple)) else o))

    def _make_hook(self, idx, ch):
        def hook(module, args, output):
            if not self._gbs:
                return None
            g, b = self._gbs[idx]
            feat, rest = (output[0], output[1:]) if isinstance(output, (list, tuple)) else (output, None)
            feat = feat * g.to(feat.dtype).view(-1, ch, 1, 1) + b.to(feat.dtype).view(-1, ch, 1, 1)
            return feat if rest is None else (feat, *rest)
        return hook

    def forward(self, blur, meta_vec):
        emb = self.embed(meta_vec)
        self._gbs = [gen(emb) for gen in self.film_gens]
        self._latent = None
        out = self.base(blur)
        out = out[0] if isinstance(out, (list, tuple)) else out
        pred_meta = self.aux_head(self._latent) if self._latent is not None else None
        self._gbs = []
        return out, pred_meta
