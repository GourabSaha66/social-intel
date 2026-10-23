"""Alert rules. Computed in pandas so they can be unit-tested, then written to their own
ES index (socintel-alerts) so Kibana can show an alerts table. A Kibana rule can also be
created on top of the same query (see docs/kibana.md)."""

import numpy as np
import pandas as pd


def negative_spikes(df, z_thresh=2.0, min_posts=20, freq="D"):
    """topic-days where the share of negative posts is z_thresh std above that topic's own mean."""
    d = df[df["topic_id"] != -1].copy()
    d["bucket"] = pd.to_datetime(d["created_utc"], unit="s").dt.floor(freq)
    g = d.groupby(["topic_id", "topic_label", "bucket"]).agg(
        n=("post_id", "size"), neg=("sentiment", lambda s: (s == "negative").mean())).reset_index()
    g = g[g["n"] >= min_posts]
    stats = g.groupby("topic_id")["neg"].agg(["mean", "std"]).rename(columns={"mean": "mu", "std": "sd"})
    g = g.join(stats, on="topic_id")
    g["z"] = (g["neg"] - g["mu"]) / g["sd"].replace(0, np.nan)
    out = g[g["z"] >= z_thresh].copy()
    out["alert"] = "negative_spike"
    return out.sort_values("z", ascending=False)


def flagged_high_reach_posts(df, top_n=50, misinfo_thresh=0.7):
    """posts the classifier flags, written by accounts in the top-PageRank group.
    This is about the content for review, not a judgement on the account: authors are
    hashed ids and the score only says 'resembles claims rated misleading in LIAR'."""
    top = (df.drop_duplicates("author").nlargest(top_n, "author_pagerank")["author"])
    d = df[df["author"].isin(top) & (df["misinfo_prob"] >= misinfo_thresh)].copy()
    d["alert"] = "flagged_high_reach_content"
    return d.sort_values("misinfo_prob", ascending=False)[
        ["alert", "post_id", "author", "author_pagerank", "misinfo_prob", "topic_label", "clean_text"]]
