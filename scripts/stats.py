"""Per-image, per-video, and seed-level significance analyses.

  python scripts/stats.py --base dab.csv --method scc.csv     # per-image + per-video tests
  python scripts/stats.py --per-video results/per_video.csv
  python scripts/stats.py --seeds 31.23,31.18,31.26 31.30,31.34,31.32
"""
import argparse
from itertools import product
import numpy as np
import pandas as pd
from scipy import stats


def exact_wilcoxon(x):
    x = np.asarray(x); order = np.argsort(np.abs(x)); ranks = np.empty(len(x)); ranks[order] = np.arange(1, len(x) + 1)
    w_neg = ranks[x < 0].sum()
    n = len(x)
    cnt = sum(1 for s in product([0, 1], repeat=n) if sum(k + 1 for k in range(n) if s[k]) <= w_neg) if n <= 20 else None
    return w_neg, (2 * cnt / 2 ** n if cnt is not None else stats.wilcoxon(x).pvalue)


def video_report(delta, frames, seed=20260929, n_boot=100000):
    delta, frames = np.asarray(delta), np.asarray(frames)
    w, p = exact_wilcoxon(delta)
    rng = np.random.default_rng(seed)
    boot = rng.choice(delta, (n_boot, len(delta)), replace=True).mean(1)
    print(f"videos {len(delta)} | positive {int((delta > 0).sum())} | equal-video mean {delta.mean():+.4f} | "
          f"frame-weighted mean {(delta * frames).sum() / frames.sum():+.4f}")
    print(f"exact Wilcoxon W- = {w:.0f}, p = {p:.6f} | equal-video 95% bootstrap CI "
          f"[{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]")


ap = argparse.ArgumentParser()
ap.add_argument("--base"); ap.add_argument("--method")
ap.add_argument("--per-video")
ap.add_argument("--seeds", nargs=2)
a = ap.parse_args()
if a.base and a.method:
    b, m = pd.read_csv(a.base), pd.read_csv(a.method)
    d = m.merge(b, on=["img_id", "video", "tier"], suffixes=("_m", "_b"))
    d["delta"] = d.psnr_m - d.psnr_b
    print(f"images {len(d)} | win rate {(d.delta > 0).mean():.1%} | per-image Wilcoxon p = {stats.wilcoxon(d.delta).pvalue:.2e}")
    v = d.groupby("video").agg(delta=("delta", "mean"), n=("delta", "size"))
    video_report(v.delta, v.n)
if a.per_video:
    v = pd.read_csv(a.per_video)
    video_report(v.mean_delta_db, v.frames)
    v2 = v[v.frames > 2]; print("excluding the 2-frame video:"); video_report(v2.mean_delta_db, v2.frames)
if a.seeds:
    b, m = (np.array([float(x) for x in s.split(",")]) for s in a.seeds)
    t = stats.ttest_rel(m, b); d = m - b
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    print(f"paired seeds: mean {d.mean():+.4f}, t = {t.statistic:.3f}, df = {len(d) - 1}, p = {t.pvalue:.4f}, "
          f"95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
