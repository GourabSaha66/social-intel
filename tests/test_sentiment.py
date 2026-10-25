import pandas as pd
import pytest

from socintel import sentiment


def fake_clf(texts, batch_size=None, max_length=None):
    out = []
    for t in texts:
        if "love" in t:
            out.append({"label": "Positive", "score": 0.9})
        elif "hate" in t:
            out.append({"label": "Negative", "score": 0.8})
        else:
            out.append({"label": "Neutral", "score": 0.6})
    return out


def test_predict_lowercases_labels_and_batches():
    texts = ["love it", "hate it", "meh"] * 5
    labels, scores = sentiment.predict(texts, fake_clf, batch_size=4)
    assert len(labels) == len(scores) == 15
    assert set(labels) == {"positive", "negative", "neutral"}


def make_df():
    day = 86400
    base = 1_700_000_000 - (1_700_000_000 % day)       # midnight UTC
    return pd.DataFrame({
        "created_utc": [base, base + 10, base + 20, base + day, base + day + 5],
        "sentiment": ["positive", "negative", "negative", "positive", "positive"],
        "community": ["a", "a", "b", "a", "b"],
    })


def test_sentiment_share_sums_to_one_per_bucket():
    out = sentiment.sentiment_over_time(make_df())
    sums = out.groupby("bucket")["share"].sum()
    assert sums.round(6).eq(1.0).all()
    assert len(sums) == 2


def test_net_sentiment_values():
    out = sentiment.net_sentiment(make_df()).sort_values("bucket")
    # day 1: +1 -1 -1 -> -1/3 ; day 2: +1 +1 -> 1
    assert out["net"].tolist() == pytest.approx([-1 / 3, 1.0])
    assert out["n"].tolist() == [3, 2]


def test_net_sentiment_grouped():
    out = sentiment.net_sentiment(make_df(), group="community")
    assert set(out["community"]) == {"a", "b"}


def test_evaluate_perfect_and_imperfect():
    labs = ["negative", "neutral", "positive", "positive"]
    assert sentiment.evaluate(labs, labs)["accuracy"] == 1.0
    res = sentiment.evaluate(labs, ["negative", "neutral", "positive", "negative"])
    assert res["accuracy"] == 0.75


def test_predict_keeps_input_order_when_batches_are_length_sorted():
    # lengths are deliberately mixed so the internal shortest-first order differs from input order
    texts = ["love " + "x" * 40, "hate", "meh " + "y" * 90, "love", "hate " + "z" * 15, "meh"]
    labels, scores = sentiment.predict(texts, fake_clf, batch_size=2)
    assert labels == ["positive", "negative", "neutral", "positive", "negative", "neutral"]
    assert scores == [0.9, 0.8, 0.6, 0.9, 0.8, 0.6]


def test_predict_empty_input():
    assert sentiment.predict([], fake_clf) == ([], [])