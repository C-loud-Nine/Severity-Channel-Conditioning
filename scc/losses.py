"""Training objectives."""
import torch
import torch.nn.functional as F


def spectral_mag_l1(pred, gt):
    """Spectral-loss control: mean L1 between orthonormal real-FFT magnitudes, per colour channel."""
    P = torch.fft.rfft2(pred, norm="ortho")
    G = torch.fft.rfft2(gt, norm="ortho")
    return (P.abs() - G.abs()).abs().mean()


def tier_supcon(z, tiers, tau=0.1):
    """Supervised-contrastive loss over difficulty tiers within the mini-batch.

    z: (B, d) projection of the prediction (l2-normalized here); tiers: (B,) integer tier ids.
    Positives P(i) = other same-tier images (anchor excluded); anchors with empty P(i) are masked;
    mean over the remaining anchors. With physical batch 4, each anchor sees three others.
    """
    z = F.normalize(z, dim=1)
    B = z.shape[0]
    logits = z @ z.t() / tau
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos = (tiers[:, None] == tiers[None, :]) & ~self_mask
    n_pos = pos.sum(1)
    valid = n_pos > 0
    if not valid.any():
        return z.sum() * 0.0
    per_anchor = -(log_prob.masked_fill(~pos, 0.0).sum(1)[valid] / n_pos[valid])
    return per_anchor.mean()


class ProjectionHead(torch.nn.Module):
    """Two-layer MLP projection z(y_hat) of the prediction, used only in training."""

    def __init__(self, in_ch=3, hidden=128, out_dim=128):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(8)
        self.mlp = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(in_ch * 64, hidden),
                                       torch.nn.SiLU(), torch.nn.Linear(hidden, out_dim))

    def forward(self, y):
        return self.mlp(self.pool(y))
