import numpy as np
import pandas as pd
import pytest

from socintel import alerts, index_es, misinfo, topics


# ---------- misinformation ----------

def toy_claims():
    bad = ["shocking secret cure doctors hide", "they are lying to you about the cure",
           "this miracle cure is banned everywhere", "secret plot hidden from the public"] * 15
    good = ["the budget passed with a vote of fifty to forty", "unemployment fell by two percent last year",
            "the bill increases funding for schools", "the senate approved the measure on tuesday"] * 15
    df = pd.DataFrame({"text": bad + good, "y": [1] * len(bad) + [0] * len(good)})
    return df.sample(frac=1, random_state=0).reset_index(drop=True)


def test_baseline_learns_toy_task_and_reports_metrics():
    df = toy_claims()
    m = misinfo.baseline_model()
    m.fit(df["text"][:90], df["y"][:90])
    res = misinfo.evaluate(m, df["text"][90:], df["y"][90:])
    assert res["accuracy"] > 0.9
    assert {"precision", "recall", "f1", "roc_auc", "confusion_matrix"} <= set(res)
    feats = misinfo.top_features(m, k=5)
    assert "cure" in feats["misleading"] or "secret" in feats["misleading"]


def test_score_posts_adds_probability_column():
    df = toy_claims()
    m = misinfo.baseline_model()
    m.fit(df["text"], df["y"])
    posts = pd.DataFrame({"clean_text": ["secret cure they hide", "the senate approved the bill"]})
    out = misinfo.score_posts(posts, m)
    assert out.loc[0, "misinfo_prob"] > out.loc[1, "misinfo_prob"]
    assert out["misinfo_prob"].between(0, 1).all()


def test_label_collapse_to_binary():
    d = pd.DataFrame({"label_name": misinfo.LIAR_LABELS})
    out = misinfo._finish(d.assign(text="x"), binary=True)
    assert out["y"].tolist() == [1, 1, 1, 0, 0, 0]


# ---------- alerts ----------

def spike_df():
    day = 86400
    base = 1_700_000_000 - (1_700_000_000 % day)
    rows = []
    # topic 0: 7 normal days at ~20% negative, then one day at 90%
    for d in range(8):
        neg_share = 0.9 if d == 7 else 0.2
        for i in range(30):
            rows.append(dict(post_id=f"t0_{d}_{i}", created_utc=base + d * day + i, topic_id=0,
                             topic_label="vaccine_news", author=f"u{i}",
                             sentiment="negative" if i < neg_share * 30 else "neutral"))
    return pd.DataFrame(rows)


def test_negative_spike_detected_on_the_right_day():
    out = alerts.negative_spikes(spike_df(), z_thresh=2.0, min_posts=20)
    assert len(out) == 1
    assert out.iloc[0]["topic_label"] == "vaccine_news"
    assert out.iloc[0]["neg"] == pytest.approx(0.9)


def test_no_spike_when_below_min_posts():
    out = alerts.negative_spikes(spike_df(), z_thresh=2.0, min_posts=1000)
    assert out.empty


def test_risky_influencer_filter():
    df = pd.DataFrame({
        "post_id": ["1", "2", "3"], "author": ["big", "big", "small"],
        "author_pagerank": [0.5, 0.5, 0.001], "misinfo_prob": [0.9, 0.2, 0.95],
        "topic_label": ["a", "a", "a"], "clean_text": ["x", "y", "z"]})
    out = alerts.risky_influencers(df, top_n=1, misinfo_thresh=0.7)
    assert out["post_id"].tolist() == ["1"]


# ---------- topics (pandas helpers) ----------

def test_rising_topics_ranks_growth():
    day = 86400
    base = 1_700_000_000 - (1_700_000_000 % day)
    rows = []
    for d in range(6):
        for _ in range(10):                                  # topic 1 flat
            rows.append((base + d * day, 1, "flat"))
        for _ in range(1 if d < 3 else 15):                  # topic 2 explodes in last 3 days
            rows.append((base + d * day, 2, "rising"))
    df = pd.DataFrame(rows, columns=["created_utc", "topic_id", "topic_label"])
    vol = topics.topic_volume(df)
    out = topics.rising_topics(vol, recent_buckets=3)
    assert out.iloc[0]["topic_id"] == 2


def test_topic_volume_ignores_outliers():
    df = pd.DataFrame({"created_utc": [1_700_000_000] * 3, "topic_id": [-1, -1, 4],
                       "topic_label": ["outlier", "outlier", "x"]})
    assert topics.topic_volume(df)["topic_id"].tolist() == [4]


# ---------- elasticsearch doc prep (no server needed) ----------

def test_to_actions_handles_nan_and_dates():
    df = pd.DataFrame({"post_id": ["a"], "created_utc": [1_700_000_000], "text": ["hi"],
                       "misinfo_prob": [np.nan], "sentiment": ["positive"]})
    acts = list(index_es.to_actions(df, "idx"))
    assert len(acts) == 1
    src = acts[0]["_source"]
    assert acts[0]["_id"] == "a" and acts[0]["_index"] == "idx"
    assert src["misinfo_prob"] is None
    assert src["created"] == "2023-11-14T22:13:20Z"
    assert "created_utc" not in src
