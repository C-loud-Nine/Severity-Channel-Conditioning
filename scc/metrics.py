"""PSNR and SSIM on 8-bit RGB (data range 255, scikit-image; SSIM over colour channels); LPIPS (AlexNet)."""
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_lpips = None


def psnr(pred, gt):
    return float(peak_signal_noise_ratio(gt, pred, data_range=255))


def ssim(pred, gt):
    return float(structural_similarity(gt, pred, channel_axis=2, data_range=255))


def lpips_alex(pred, gt, device="cuda"):
    global _lpips
    if _lpips is None:
        import lpips
        _lpips = lpips.LPIPS(net="alex").to(device).eval()
    t = lambda a: torch.from_numpy(a.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        return float(_lpips(t(pred), t(gt)).item())
