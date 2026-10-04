"""Rebuild data/split_manifest.csv from complete_metadata.csv and data/test_videos.txt."""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--metadata", required=True)
ap.add_argument("--test-videos", default="data/test_videos.txt")
ap.add_argument("--out", default="data/split_manifest.csv")
a = ap.parse_args()
order = [l.split()[0] for l in open(a.test_videos) if l.strip() and not l.startswith("#")]
df = pd.read_csv(a.metadata)
df["split"] = df.video.isin(order).map({True: "test", False: "train"})
df["cluster"] = df.video.map({v: i + 1 for i, v in enumerate(order)}).astype("Int64")
assert df.groupby("video").split.nunique().max() == 1
out = df[["img_id", "video", "split", "difficulty", "cluster"]].rename(columns={"video": "source_video", "difficulty": "tier"})
out.to_csv(a.out, index=False)
print(out.groupby("split").agg(videos=("source_video", "nunique"), images=("img_id", "size")))
