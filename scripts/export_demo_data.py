"""Exports aggregate numbers for the public demo app into demo_data/.

Nothing here contains comment text or author ids: only counts per week/topic/language, the pipeline's metrics,
the alert table and the hand-check counts. Run after the pipeline:

    python scripts/export_demo_data.py
"""
import json
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from socintel.io_utils import load_config, read_jsonl   # noqa: E402

OUT = "demo_data"


def week(df):
    return pd.to_datetime(df["created_utc"], unit="s").dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")


def main():
    cfg = load_config()
    proc = cfg["paths"]["processed"]
    df = read_jsonl(os.path.join(proc, "05_misinfo.jsonl"))
    os.makedirs(OUT, exist_ok=True)
    df["week"] = week(df)

    df.groupby(["week", "sentiment"]).size().rename("n").reset_index() \
      .to_csv(f"{OUT}/weekly_sentiment.csv", index=False)

    d = df[df["topic_id"] != -1]
    big = d["topic_label"].value_counts()
    big = big[big >= 50].index
    d[d["topic_label"].isin(big)].groupby(["week", "topic_label"]).size().rename("n").reset_index() \
        .to_csv(f"{OUT}/weekly_topics.csv", index=False)

    df["language"].value_counts().rename_axis("language").rename("n").reset_index() \
      .to_csv(f"{OUT}/languages.csv", index=False)

    ti = pd.read_csv(os.path.join(proc, "topic_info.csv"))[["Topic", "Count", "Name"]]
    ti.to_csv(f"{OUT}/topics.csv", index=False)

    for name in ("alerts_neg_spikes.csv", "metrics.json"):
        shutil.copy(os.path.join(proc, name), f"{OUT}/{name}")
    rr = "data/review/review_results.json"
    if os.path.exists(rr):
        shutil.copy(rr, f"{OUT}/review_results.json")

    with open(f"{OUT}/meta.json", "w") as f:
        json.dump({"rows": int(len(df)), "videos": int(df["thread_id"].nunique()),
                   "first": str(pd.to_datetime(df["created_utc"].min(), unit="s").date()),
                   "last": str(pd.to_datetime(df["created_utc"].max(), unit="s").date())}, f)
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()