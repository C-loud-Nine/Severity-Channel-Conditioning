"""Full-resolution tiled inference: reflect-pad to a multiple of 8, 768 px tiles, 48 px overlap, averaging."""
import cv2
import numpy as np
import torch


def pad_mult(img, m=8):
    h, w = img.shape[:2]
    ph, pw = (m - h % m) % m, (m - w % m) % m
    return np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect"), (h, w)


@torch.no_grad()
def deblur_tiled(img_rgb, model_fn, sev_map=None, tile=768, overlap=48, device="cuda"):
    """model_fn(x) -> restored tile; x is [rgb] or [rgb; sev] when sev_map is given."""
    padded, (h0, w0) = pad_mult(img_rgb, 8)
    Hp, Wp = padded.shape[:2]
    inp = torch.from_numpy(padded.astype(np.float32).transpose(2, 0, 1) / 255.0)[None].to(device)
    if sev_map is not None:
        s = cv2.resize(sev_map, (Wp, Hp), interpolation=cv2.INTER_LINEAR)
        inp = torch.cat([inp, torch.from_numpy(s)[None, None].to(device)], dim=1)
    out = torch.zeros(1, 3, Hp, Wp, device=device)
    wsum = torch.zeros(1, 1, Hp, Wp, device=device)
    step = tile - overlap
    for y in range(0, max(1, Hp - overlap), step):
        for x in range(0, max(1, Wp - overlap), step):
            y2, x2 = min(y + tile, Hp), min(x + tile, Wp)
            y1, x1 = max(0, y2 - tile), max(0, x2 - tile)
            out[:, :, y1:y2, x1:x2] += model_fn(inp[:, :, y1:y2, x1:x2])
            wsum[:, :, y1:y2, x1:x2] += 1.0
    res = (out / wsum.clamp(min=1.0)).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (res * 255.0 + 0.5).astype(np.uint8)[:h0, :w0]
