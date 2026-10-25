"""Alert rules. Computed in pandas so they can be unit-tested, then written to their own
ES index (socintel-alerts) so Kibana can show an alerts table. A Kibana rule can also be
created on top of the same query (see docs/kibana.md)."""

import numpy as np
import pandas as pd


def eligible_buckets(df, min_posts=15, freq="W"):
    """how many (topic, time bucket) cells have enough posts to be judged at all.
    If this is 0, 'no spikes' just means the buckets are too small, not that nothing happened."""
    d = df[df["topic_id"] != -1]
    b = pd.to_datetime(d["created_utc"], unit="s").dt.to_period(freq).dt.start_time
    return int((d.groupby([d["topic_id"], b]).size() >= min_posts).sum())


def negative_spikes(df, z_thresh=2.0, min_posts=15, freq="W"):
    """topic-weeks (freq) whose negative share is unusually high for that topic.

    Baseline = the topic's negative share over all its *other* posts. z is a binomial z-score:
    (share_in_bucket - share_elsewhere) / sqrt(p(1-p)/n). We do not use the std of the weekly
    shares: most topics here only have a few judgeable weeks, and with k weeks a std-based z
    can never exceed (k-1)/sqrt(k), so a threshold of 2 would be unreachable.
    """
    d = df[df["topic_id"] != -1].copy()
    d["bucket"] = pd.to_datetime(d["created_utc"], unit="s").dt.to_period(freq).dt.start_time
    d["is_neg"] = (d["sentiment"] == "negative").astype(float)
    g = d.groupby(["topic_id", "topic_label", "bucket"]).agg(
        n=("post_id", "size"), neg=("is_neg", "mean")).reset_index()
    tot = d.groupby("topic_id")["is_neg"].agg(["sum", "size"])
    g = g[g["n"] >= min_posts].join(tot, on="topic_id")
    rest_n = g["size"] - g["n"]
    g["mu"] = (g["sum"] - g["neg"] * g["n"]) / rest_n.replace(0, np.nan)
    # keep p away from 0/1 so the standard error never collapses to zero
    p = g["mu"].clip(0.05, 0.95)
    g["sd"] = np.sqrt(p * (1 - p) / g["n"])
    g["z"] = (g["neg"] - g["mu"]) / g["sd"]
    g = g[rest_n >= 30]                     # need a reasonable baseline outside the bucket
    out = g[g["z"] >= z_thresh][["topic_id", "topic_label", "bucket", "n", "neg", "mu", "sd", "z"]].copy()
    out["alert"] = "negative_spike"
    return out.sort_values("z", ascending=False)


def flagged_high_reach_posts(df, top_n=50, misinfo_thresh=0.7):
    """posts the classifier flags, written by accounts in the top-PageRank group.
    This is about the content for review, not a judgement on the account: authors are
    hashed ids and the score only says 'resembles claims rated misleading in LIAR'."""
    # keep="all": authors tied with the top_n-th PageRank stay in (the graph is sparse, ties are common)
    top = (df.drop_duplicates("author").nlargest(top_n, "author_pagerank", keep="all")["author"])
    d = df[df["author"].isin(top) & (df["misinfo_prob"] >= misinfo_thresh)].copy()
    d["alert"] = "flagged_high_reach_content"
    return d.sort_values("misinfo_prob", ascending=False)[
        ["alert", "post_id", "author", "author_pagerank", "misinfo_prob", "topic_label", "clean_text"]]