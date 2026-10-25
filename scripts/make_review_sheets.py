"""Builds the two sheets for the manual check (opens fine in Excel):

  data/review/sentiment_sample.csv   100 random comments, fill the `human_label` column
  data/review/flagged_review.csv     the flagged high-reach posts, fill the `judgement` column
  data/review/sentiment_key.csv      the model's labels, kept apart so they don't bias you; don't open it

Run:  python scripts/make_review_sheets.py
Then label, save as CSV, and run  python scripts/score_review.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from socintel import alerts                     # noqa: E402
from socintel.io_utils import load_config, read_jsonl   # noqa: E402

OUT = "data/review"


def main(n=100, seed=7):
    cfg = load_config()
    df = read_jsonl(os.path.join(cfg["paths"]["processed"], "05_misinfo.jsonl"))
    os.makedirs(OUT, exist_ok=True)

    pool = df[df["clean_text"].str.len() >= 15]
    s = pool.sample(min(n, len(pool)), random_state=seed)
    sheet = s[["post_id", "language", "text"]].copy()
    sheet["text"] = sheet["text"].str.slice(0, 400)
    sheet["human_label"] = ""          # negative / neutral / positive
    sheet.to_csv(f"{OUT}/sentiment_sample.csv", index=False, encoding="utf-8-sig")
    s[["post_id", "sentiment", "sentiment_score"]].to_csv(f"{OUT}/sentiment_key.csv", index=False)

    fl = alerts.flagged_high_reach_posts(df, cfg["alerts"]["influencer_top_n"])
    fl = fl[["post_id", "misinfo_prob", "topic_label", "clean_text"]].copy()
    fl["judgement"] = ""               # misleading / opinion / other
    fl.to_csv(f"{OUT}/flagged_review.csv", index=False, encoding="utf-8-sig")
    print(f"wrote {len(sheet)} comments to label and {len(fl)} flagged posts to judge in {OUT}/")


if __name__ == "__main__":
    main()