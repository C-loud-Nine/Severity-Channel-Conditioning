"""Precompute the severity map (or a control variant) for every image (256x256, float16)."""
import argparse, os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scc.severity import build_variant
from scc.data import image_paths, read_rgb

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="iPhoneBlur root ({split}/{video}/blur|sharp)")
ap.add_argument("--metadata", required=True, help="complete_metadata.csv")
ap.add_argument("--manifest", default="data/split_manifest.csv")
ap.add_argument("--out", default="cache/severity")
ap.add_argument("--variant", default="severity", choices=["severity", "gradient_only", "scale_only"])
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
df = pd.read_csv(a.metadata).merge(pd.read_csv(a.manifest)[["img_id", "split"]], on="img_id")
for _, r in tqdm(df.iterrows(), total=len(df)):
    p = f"{a.out}/{r['img_id']}.npy"
    if not os.path.exists(p):
        np.save(p, build_variant(read_rgb(image_paths(a.root, r)[0]), r["isp_diff"], a.variant).astype(np.float16))
