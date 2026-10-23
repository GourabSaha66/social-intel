"""Multilingual sentiment with a pretrained XLM-R model, plus sentiment-over-time helpers.

Model: cardiffnlp/twitter-xlm-roberta-base-sentiment (negative / neutral / positive,
trained on tweets in ~8 languages). It's a reasonable choice for short social text. For
languages outside its training set expect lower accuracy - we report accuracy by language.
"""

import numpy as np
import pandas as pd

from .schema import SENTIMENT_LABELS


def load_pipeline(model_name, device=-1):
    from transformers import pipeline          # imported here so tests don't need torch
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name,
                    device=device, truncation=True)


def predict(texts, clf, batch_size=32, max_length=128):
    """returns (labels, confidence) lists, labels lowercased to match SENTIMENT_LABELS."""
    labels, scores = [], []
    for i in range(0, len(texts), batch_size):
        out = clf(texts[i:i + batch_size], batch_size=batch_size, max_length=max_length)
        for o in out:
            labels.append(o["label"].lower())
            scores.append(float(o["score"]))
    return labels, scores


def add_sentiment(df, clf, batch_size=32, max_length=128):
    df = df.copy()
    df["sentiment"], df["sentiment_score"] = predict(df["clean_text"].tolist(), clf,
                                                     batch_size, max_length)
    return df


# ---------- over time ----------

def sentiment_over_time(df, freq="D", group=None):
    """share of each label per time bucket (optionally per community/topic).
    returns long frame: bucket, [group], sentiment, n, share."""
    d = df.copy()
    d["bucket"] = pd.to_datetime(d["created_utc"], unit="s").dt.floor(freq)
    keys = ["bucket"] + ([group] if group else [])
    counts = d.groupby(keys + ["sentiment"]).size().rename("n").reset_index()
    totals = counts.groupby(keys)["n"].transform("sum")
    counts["share"] = counts["n"] / totals
    return counts


def net_sentiment(df, freq="D", group=None):
    """(pos - neg) / total per bucket. one number per bucket, easier to plot/alert on."""
    d = df.copy()
    d["bucket"] = pd.to_datetime(d["created_utc"], unit="s").dt.floor(freq)
    d["val"] = d["sentiment"].map({"positive": 1, "neutral": 0, "negative": -1})
    keys = ["bucket"] + ([group] if group else [])
    return d.groupby(keys)["val"].agg(net="mean", n="size").reset_index()


# ---------- evaluation ----------

def evaluate(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=SENTIMENT_LABELS),
        "report": classification_report(y_true, y_pred, labels=SENTIMENT_LABELS, zero_division=0),
    }


def eval_on_tweeteval(clf, n=2000, seed=42):
    """sanity check against the English TweetEval sentiment test split (0=neg,1=neu,2=pos)."""
    from datasets import load_dataset
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="test")
    df = ds.to_pandas().sample(min(n, len(ds)), random_state=seed)
    df["gold"] = df["label"].map(dict(enumerate(SENTIMENT_LABELS)))
    pred, _ = predict(df["text"].tolist(), clf)
    return evaluate(df["gold"], pred)
