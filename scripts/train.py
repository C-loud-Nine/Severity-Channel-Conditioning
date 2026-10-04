"""Fine-tune one configuration from the Domain-Adapted Baseline (DAB) checkpoint.

Budget (every DiffIR arm): 12 epochs; physical batch 2 with accumulation 2 (contrastive: physical 4,
no accumulation); loss divided by the accumulation factor under AMP; trailing micro-batch of each epoch
discarded, so 1,428 updates per epoch and 17,136 per arm. AdamW (0.9, 0.99), no weight decay; cosine
from 2e-5 (min 1e-6) with period 34,284 steps, stepped once per update; EMA 0.999. The final-epoch EMA
weights are saved and evaluated. No test data is used during training.
"""
import argparse, json, os, sys, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scc.data import PairPatches, load_split, zscore_stats
from scc.ema import EMA
from scc.losses import spectral_mag_l1, tier_supcon, ProjectionHead
from scc.models import make_diffir, widen_cpen, SevDiffIR, FiLMDiffIR, SFTDiffIR, make_restormer, SevRestormer
from scc.severity import isp_gate

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--seed", type=int, default=None)
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
seed = a.seed if a.seed is not None else cfg.get("seed", 42)
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
dev = "cuda"
arm = cfg["arm"]

train_df = load_split(cfg["manifest"], cfg["metadata"], "train")
stats = zscore_stats(train_df) if arm == "film" else None
ds = PairPatches(cfg["root"], train_df, cfg["severity_dir"], stats, cfg.get("patch", 256))
dl = DataLoader(ds, batch_size=cfg["batch"], shuffle=True, num_workers=2, drop_last=True, pin_memory=True)

restormer = cfg.get("backbone", "diffir") == "restormer"
base = make_restormer(cfg["init_ckpt"], dev) if restormer else make_diffir(cfg["init_ckpt"], dev)
extra = nn.ModuleDict()
if restormer and arm == "scc":
    model = SevRestormer(base)
elif arm in ("scc", "control"):
    model = SevDiffIR(widen_cpen(base))
elif arm == "film":
    model = FiLMDiffIR(base)
elif arm == "sft":
    model = SFTDiffIR(base)
else:                       # dab (schedule-matched), contrastive, spectral: plain 3-channel DAB
    model = base
if arm == "contrastive":
    extra["proj"] = ProjectionHead()
model, extra = model.to(dev), extra.to(dev)


def fourth_channel(sev, meta_isp=None):
    """Fourth-channel content for the descriptor controls; 'severity' is the full severity map."""
    kind = cfg.get("control", "severity")
    if kind == "random_constant":
        return torch.rand(sev.shape[0], 1, 1, 1, device=sev.device).expand_as(sev)
    if kind == "shuffled":
        flat = sev.flatten(1)
        return flat[:, torch.randperm(flat.shape[1], device=sev.device)].view_as(sev)
    return sev              # severity, gradient_only, scale_only: precomputed maps (severity_dir)


params = list(model.parameters()) + list(extra.parameters())
opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, 0.99), weight_decay=0)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["t_max"], eta_min=cfg["lr_min"])
scaler = torch.amp.GradScaler("cuda")
ema = EMA(model, cfg.get("ema", 0.999))
accum = cfg["accum"]
os.makedirs(cfg["out_dir"], exist_ok=True)
updates = 0
model.eval()                # DiffIR S2 fine-tuned through its inference path, as in the reported runs
for epoch in range(cfg["epochs"]):
    opt.zero_grad()
    for it, (lq, gt, sev, meta, tier) in enumerate(tqdm(dl, desc=f"epoch {epoch}")):
        lq, gt, sev, meta, tier = lq.to(dev), gt.to(dev), sev.to(dev), meta.to(dev), tier.to(dev)
        with torch.amp.autocast("cuda"):
            if arm in ("scc", "control"):
                out = model(torch.cat([lq, fourth_channel(sev)], 1)); loss = F.l1_loss(out, gt)
            elif arm == "film":
                out, pred_meta = model(lq, meta)
                loss = F.l1_loss(out, gt) + cfg["aux_weight"] * F.l1_loss(pred_meta, meta)
            elif arm == "sft":
                out = model(lq, sev); loss = F.l1_loss(out, gt)
            else:
                out = model(lq); out = out[0] if isinstance(out, (list, tuple)) else out
                loss = F.l1_loss(out, gt)
                if arm == "contrastive":
                    loss = loss + cfg["con_weight"] * tier_supcon(extra["proj"](out), tier, cfg["tau"])
                if arm == "spectral":
                    loss = loss + cfg["lambda_f"] * spectral_mag_l1(out, gt)
            loss = loss / accum
        scaler.scale(loss).backward()
        if (it + 1) % accum == 0:
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step(); ema.update(model)
            updates += 1
    torch.save({"model": model.state_dict(), "ema": ema.shadow, "epoch": epoch, "updates": updates},
               f"{cfg['out_dir']}/latest.pth")
torch.save({"params_ema": ema.shadow, "epochs": cfg["epochs"], "updates": updates, "arm": arm, "seed": seed},
           f"{cfg['out_dir']}/final.pth")
json.dump({"arm": arm, "seed": seed, "epochs": cfg["epochs"], "updates": updates,
           "final_lr": sched.get_last_lr()[0]}, open(f"{cfg['out_dir']}/run.json", "w"), indent=2)
print(f"done: {updates} optimizer updates, final LR {sched.get_last_lr()[0]:.3e}")
