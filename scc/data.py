"""iPhoneBlur pairs with the 36/15 source-video split of data/split_manifest.csv.

Expected layout (as released): {root}/{train,test}/{video_stem}/{blur,sharp}/{img_num:05d}.jpg
and the metadata table complete_metadata.csv. Severity maps are precomputed by
scripts/precompute_severity.py into {sev_dir}/{img_id}.npy (256x256, float16).
"""
import json
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TIER_ID = {"Easy": 0, "Medium": 1, "Hard": 2}
META_FIELDS = ["motion", "isp_diff", "blur_window", "blur_duration_ms", "noise_estimate"]


def load_split(manifest_csv, metadata_csv, split):
    man = pd.read_csv(manifest_csv)
    meta = pd.read_csv(metadata_csv)
    df = meta.merge(man[["img_id", "split", "cluster"]], on="img_id")
    return df[df.split == split].reset_index(drop=True)


def image_paths(root, row):
    stem = os.path.splitext(str(row["video"]))[0]
    fn = f"{int(row['img_num']):05d}.jpg"
    split = row["split"]
    return f"{root}/{split}/{stem}/blur/{fn}", f"{root}/{split}/{stem}/sharp/{fn}"


def read_rgb(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def zscore_stats(train_df):
    """FiLM normalization: statistics fitted on the training split only."""
    return {f: (float(train_df[f].mean()), float(train_df[f].std())) for f in META_FIELDS}


class PairPatches(Dataset):
    """Random 256x256 crops with horizontal flip; returns blur, sharp, severity, metadata, tier."""

    def __init__(self, root, df, sev_dir, stats=None, patch=256):
        self.root, self.df, self.sev_dir, self.stats, self.p = root, df, sev_dir, stats, patch

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        bp, sp = image_paths(self.root, r)
        lq, gt = read_rgb(bp), read_rgb(sp)
        sev = np.load(f"{self.sev_dir}/{r['img_id']}.npy").astype(np.float32)
        H, W = lq.shape[:2]
        sev = cv2.resize(sev, (W, H), interpolation=cv2.INTER_LINEAR)
        p = self.p
        if H < p or W < p:
            size = (max(W, p), max(H, p))
            lq, gt, sev = cv2.resize(lq, size), cv2.resize(gt, size), cv2.resize(sev, size)
            H, W = lq.shape[:2]
        y, x = random.randint(0, H - p), random.randint(0, W - p)
        lq, gt, sev = lq[y:y + p, x:x + p], gt[y:y + p, x:x + p], sev[y:y + p, x:x + p]
        if random.random() < 0.5:
            lq, gt, sev = lq[:, ::-1].copy(), gt[:, ::-1].copy(), sev[:, ::-1].copy()
        t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)
        meta = torch.zeros(len(META_FIELDS))
        if self.stats:
            meta = torch.tensor([(r[f] - self.stats[f][0]) / self.stats[f][1] for f in META_FIELDS],
                                dtype=torch.float32)
        return t(lq), t(gt), torch.from_numpy(sev).unsqueeze(0), meta, TIER_ID[r["difficulty"]]
