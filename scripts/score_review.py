"""Scores the manual check after you filled in the two sheets.

    python scripts/score_review.py

Writes data/review/review_results.json (numbers for docs/results.md).
"""
import json
import os

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

OUT = "data/review"
LABELS = ["negative", "neutral", "positive"]


def score_sentiment(sheet_path, key_path):
    sheet = pd.read_csv(sheet_path, encoding="utf-8-sig")
    key = pd.read_csv(key_path)
    d = sheet.merge(key, on="post_id")
    d["human_label"] = d["human_label"].astype(str).str.strip().str.lower()
    d = d[d["human_label"].isin(LABELS)]
    if d.empty:
        raise SystemExit("no valid human_label values yet (negative / neutral / positive)")
    res = {
        "n": len(d),
        "accuracy": round(accuracy_score(d["human_label"], d["sentiment"]), 3),
        "macro_f1": round(f1_score(d["human_label"], d["sentiment"], average="macro", labels=LABELS), 3),
        "human_share": d["human_label"].value_counts(normalize=True).round(3).to_dict(),
        "model_share": d["sentiment"].value_counts(normalize=True).round(3).to_dict(),
        "confusion_rows_human_cols_model": confusion_matrix(
            d["human_label"], d["sentiment"], labels=LABELS).tolist(),
    }
    return res


def score_flagged(path):
    f = pd.read_csv(path, encoding="utf-8-sig")
    f["judgement"] = f["judgement"].astype(str).str.strip().str.lower()
    f = f[f["judgement"].isin(["misleading", "opinion", "other"])]
    if f.empty:
        raise SystemExit("no valid judgement values yet (misleading / opinion / other)")
    return {"n": len(f), "counts": f["judgement"].value_counts().to_dict()}


if __name__ == "__main__":
    out = {"sentiment": score_sentiment(f"{OUT}/sentiment_sample.csv", f"{OUT}/sentiment_key.csv"),
           "flagged": score_flagged(f"{OUT}/flagged_review.csv")}
    json.dump(out, open(f"{OUT}/review_results.json", "w"), indent=2)
    print(json.dumps(out, indent=2))