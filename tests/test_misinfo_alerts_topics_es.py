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
    out = alerts.negative_spikes(spike_df(), z_thresh=2.0, min_posts=20, freq="D")
    assert len(out) == 1
    assert out.iloc[0]["topic_label"] == "vaccine_news"
    assert out.iloc[0]["neg"] == pytest.approx(0.9)


def weekly_df():
    """10 weeks x 30 posts per week at ~20% negative, final week 90% negative."""
    week = 7 * 86400
    base = 1_700_000_000 - (1_700_000_000 % week)
    rows = []
    for w in range(11):
        share = 0.9 if w == 10 else 0.2
        for i in range(30):
            rows.append(dict(post_id=f"w{w}_{i}", created_utc=base + w * week + 3600 + i, topic_id=0,
                             topic_label="climate_news", author=f"u{i}",
                             sentiment="negative" if i < share * 30 else "neutral"))
    return pd.DataFrame(rows)


def test_weekly_buckets_find_spike_and_count_eligible():
    df = weekly_df()
    out = alerts.negative_spikes(df, z_thresh=2.0, min_posts=15, freq="W")
    assert len(out) == 1 and out.iloc[0]["neg"] == pytest.approx(0.9)
    assert alerts.eligible_buckets(df, min_posts=15, freq="W") == 11
    assert alerts.eligible_buckets(df, min_posts=500, freq="W") == 0


def test_stop_words_cover_english_and_hinglish():
    from socintel.topics import STOP_WORDS
    assert "the" in STOP_WORDS and "hai" in STOP_WORDS


def test_no_spike_when_below_min_posts():
    out = alerts.negative_spikes(spike_df(), z_thresh=2.0, min_posts=1000)
    assert out.empty


def test_flagged_high_reach_filter():
    df = pd.DataFrame({
        "post_id": ["1", "2", "3"], "author": ["big", "big", "small"],
        "author_pagerank": [0.5, 0.5, 0.001], "misinfo_prob": [0.9, 0.2, 0.95],
        "topic_label": ["a", "a", "a"], "clean_text": ["x", "y", "z"]})
    out = alerts.flagged_high_reach_posts(df, top_n=1, misinfo_thresh=0.7)
    assert out["post_id"].tolist() == ["1"]
    assert out["alert"].iloc[0] == "flagged_high_reach_content"


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


def test_hf_label_names_are_respected():
    # HF order is NOT the order of LIAR_LABELS; mapping must come from the names we pass in
    names = ["false", "half-true", "mostly-true", "true", "barely-true", "pants-fire"]
    df = pd.DataFrame({"statement": list("abcdef"), "label": [0, 1, 2, 3, 4, 5]})
    out = misinfo._from_hf(df, names, binary=True)
    assert out["label_name"].tolist() == names
    assert out["y"].tolist() == [1, 0, 0, 0, 1, 1]
    assert out["text"].tolist() == list("abcdef")


def test_local_tsv_route(tmp_path):
    rows = ("1\tfalse\tclaim one\ts\tp\tj\tst\tparty\t0\t0\t0\t0\t0\tctx\n"
            "2\ttrue\tclaim two\ts\tp\tj\tst\tparty\t0\t0\t0\t0\t0\tctx\n")
    for f in ("train.tsv", "valid.tsv", "test.tsv"):
        (tmp_path / f).write_text(rows)
    data = misinfo.load_liar(local_dir=str(tmp_path))
    assert set(data) == {"train", "val", "test"}
    assert data["train"]["y"].tolist() == [1, 0]


def test_spike_possible_with_only_three_weeks():
    """std-of-weeks z can't pass 1.15 with 3 buckets; the binomial test must still catch a real jump."""
    week = 7 * 86400
    base = 1_700_000_000 - (1_700_000_000 % week)
    rows = []
    for w, share in enumerate([0.3, 0.3, 0.9]):
        for i in range(40):
            rows.append(dict(post_id=f"{w}_{i}", created_utc=base + w * week + 3600 + i, topic_id=1,
                             topic_label="x", author="a", sentiment="negative" if i < share * 40 else "neutral"))
    out = alerts.negative_spikes(pd.DataFrame(rows), z_thresh=2.0, min_posts=15, freq="W")
    assert len(out) == 1 and out.iloc[0]["neg"] == pytest.approx(0.9)


def test_flagged_keeps_authors_tied_at_the_cutoff():
    df = pd.DataFrame({
        "post_id": ["p1", "p2", "p3", "p4"], "author": ["a", "b", "c", "d"],
        "author_pagerank": [0.5, 0.2, 0.2, 0.01], "misinfo_prob": [0.9, 0.9, 0.9, 0.9],
        "topic_label": ["x"] * 4, "clean_text": ["t"] * 4})
    out = alerts.flagged_high_reach_posts(df, top_n=2, misinfo_thresh=0.7)
    assert set(out["author"]) == {"a", "b", "c"}