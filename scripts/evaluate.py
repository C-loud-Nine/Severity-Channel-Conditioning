"""Evaluate a final-epoch checkpoint on all 1,686 test images; writes a per-image CSV.

--mode oracle: target-derived severity channel at inference (oracle diagnostic).
--mode zero:   severity channel zeroed (exploratory inference ablation).
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scc.data import load_split, image_paths, read_rgb, zscore_stats, META_FIELDS
from scc.metrics import psnr, ssim, lpips_alex
from scc.models import make_diffir, widen_cpen, SevDiffIR, FiLMDiffIR, SFTDiffIR, make_restormer, SevRestormer
from scc.tiling import deblur_tiled

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--ckpt", required=True, help="final.pth from scripts/train.py")
ap.add_argument("--mode", default="oracle", choices=["oracle", "zero"])
ap.add_argument("--out", required=True, help="per-image CSV")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config)); dev = "cuda"; arm = cfg["arm"]
torch.manual_seed(0)       # fixed diffusion latent across conditions
restormer = cfg.get("backbone", "diffir") == "restormer"
base = make_restormer(None, dev) if restormer else make_diffir(None, dev)
model = SevRestormer(base).to(dev) if (restormer and arm == "scc") else {"scc": lambda: SevDiffIR(widen_cpen(base)), "control": lambda: SevDiffIR(widen_cpen(base)),
         "film": lambda: FiLMDiffIR(base), "sft": lambda: SFTDiffIR(base)}.get(arm, lambda: base)().to(dev)
model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["params_ema"], strict=True)
model.eval()
test = load_split(cfg["manifest"], cfg["metadata"], "test")
stats = zscore_stats(load_split(cfg["manifest"], cfg["metadata"], "train")) if arm == "film" else None
rows = []
for _, r in tqdm(test.iterrows(), total=len(test)):
    blur, sharp = (read_rgb(p) for p in image_paths(cfg["root"], r))
    sev = np.load(f"{cfg['severity_dir']}/{r['img_id']}.npy").astype(np.float32)
    if a.mode == "zero":
        sev = np.zeros_like(sev)
    if arm in ("scc", "control"):
        pred = deblur_tiled(blur, model, sev_map=sev, device=dev)   # DiffIR or Restormer wrapper
    elif arm == "film":
        m = torch.tensor([[(r[f] - stats[f][0]) / stats[f][1] for f in META_FIELDS]], device=dev)
        pred = deblur_tiled(blur, lambda x: model(x, m)[0], device=dev)
    elif arm == "sft":
        pred = deblur_tiled(blur, lambda x: model(x[:, :3], x[:, 3:]), sev_map=sev, device=dev)
    else:
        pred = deblur_tiled(blur, lambda x: (lambda o: o[0] if isinstance(o, (list, tuple)) else o)(model(x)), device=dev)
    rows.append({"img_id": r["img_id"], "video": r["video"], "tier": r["difficulty"], "arm": arm,
                 "mode": a.mode, "psnr": psnr(pred, sharp), "ssim": ssim(pred, sharp),
                 "lpips": lpips_alex(pred, sharp, dev)})
df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
print(df.groupby("tier")[["psnr", "ssim", "lpips"]].mean().round(3))
print("overall", df[["psnr", "ssim", "lpips"]].mean().round(3).to_dict())
