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
    """returns (labels, confidence) lists in the same order as texts, labels lowercased.

    Texts are scored shortest-first so each batch holds similar lengths (the model pads every
    text in a batch to the longest one, which wastes a lot of CPU on mixed lengths); results
    are put back into the original order afterwards.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:                      # progress bar is nice-to-have only
        def tqdm(it, **kwargs):
            return it
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    labels, scores = [None] * len(texts), [None] * len(texts)
    for start in tqdm(range(0, len(order), batch_size), desc="sentiment", unit="batch",
                      disable=len(texts) < 500):
        idx = order[start:start + batch_size]
        out = clf([texts[i] for i in idx], batch_size=batch_size, max_length=max_length)
        for i, o in zip(idx, out):
            labels[i] = o["label"].lower()
            scores[i] = float(o["score"])
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
    d["bucket"] = pd.to_datetime(d["created_utc"], unit="s").dt.to_period(freq).dt.start_time
    keys = ["bucket"] + ([group] if group else [])
    counts = d.groupby(keys + ["sentiment"]).size().rename("n").reset_index()
    totals = counts.groupby(keys)["n"].transform("sum")
    counts["share"] = counts["n"] / totals
    return counts


def net_sentiment(df, freq="D", group=None):
    """(pos - neg) / total per bucket. one number per bucket, easier to plot/alert on."""
    d = df.copy()
    d["bucket"] = pd.to_datetime(d["created_utc"], unit="s").dt.to_period(freq).dt.start_time
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


STARS_TO_LABEL = {0: "negative", 1: "negative", 2: "neutral", 3: "positive", 4: "positive"}   # keyed by label = stars - 1
AMAZON_DATASET = "SetFit/amazon_reviews_multi_{lang}"       # plain jsonl files, no loading script


def amazon_stars(df):
    """1-5 stars per row. The SetFit port has `label` (0-4) and a `label_text`; if the text carries the
    star count we trust it, otherwise label + 1. Either way it is checked to be inside 1..5."""
    stars = None
    if "label_text" in df.columns:
        digits = df["label_text"].astype(str).str.extract(r"([1-5])")[0]
        if digits.notna().all():
            stars = digits.astype(int)
    if stars is None:
        stars = df["label"].astype(int) + 1
    if not stars.between(1, 5).all():
        raise ValueError("unexpected star values in the Amazon data")
    return stars


def eval_on_amazon(clf, langs=("en", "de", "es", "fr", "ja", "zh"), n=1000, seed=42):
    """accuracy / macro-F1 per language on the Multilingual Amazon Reviews Corpus test split
    (SetFit port, one dataset per language). stars 1-2 = negative, 3 = neutral, 4-5 = positive.
    Languages that fail to load are skipped."""
    from datasets import load_dataset
    out = {}
    for lang in langs:
        try:
            ds = load_dataset(AMAZON_DATASET.format(lang=lang), split="test")
        except Exception as e:
            print(f"amazon {lang}: could not load ({e})")
            continue
        df = ds.to_pandas().sample(min(n, len(ds)), random_state=seed)
        stars = amazon_stars(df)
        if not out:     # once: show the mapping so it can be eyeballed
            cols = [c for c in ("label", "label_text") if c in df.columns]
            print("amazon label check:", df[cols].drop_duplicates().sort_values(cols[0]).to_dict("records"))
        gold = (stars - 1).map(STARS_TO_LABEL)
        pred, _ = predict(df["text"].tolist(), clf)
        res = evaluate(gold, pred)
        out[lang] = {"accuracy": res["accuracy"], "macro_f1": res["macro_f1"], "n": len(df)}
    return out


def eval_on_tweeteval(clf, n=2000, seed=42):
    """sanity check against the English TweetEval sentiment test split (0=neg,1=neu,2=pos)."""
    from datasets import load_dataset
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="test")
    df = ds.to_pandas().sample(min(n, len(ds)), random_state=seed)
    df["gold"] = df["label"].map(dict(enumerate(SENTIMENT_LABELS)))
    pred, _ = predict(df["text"].tolist(), clf)
    return evaluate(df["gold"], pred)