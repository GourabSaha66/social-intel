"""Public demo: results of the Social Media Intelligence pipeline.

Shows aggregate numbers only (no comment text, no user ids). The full system (Elasticsearch, Kibana,
Neo4j) runs locally with Docker; see the repository README.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parent.parent / "demo_data"
REPO = "https://github.com/GourabSaha66/social-intel"

st.set_page_config(page_title="Social Media Intelligence Platform", layout="wide")


@st.cache_data
def load():
    m = json.loads((DATA / "metrics.json").read_text(encoding="utf-8"))
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    sent = pd.read_csv(DATA / "weekly_sentiment.csv", parse_dates=["week"])
    tw = pd.read_csv(DATA / "weekly_topics.csv", parse_dates=["week"])
    langs = pd.read_csv(DATA / "languages.csv")
    topics = pd.read_csv(DATA / "topics.csv")
    spikes = pd.read_csv(DATA / "alerts_neg_spikes.csv")
    rr = DATA / "review_results.json"
    review = json.loads(rr.read_text(encoding="utf-8")) if rr.exists() else None
    return m, meta, sent, tw, langs, topics, spikes, review


m, meta, sent, tw, langs, topics, spikes, review = load()

st.title("Social Media Intelligence Platform")
st.caption(f"DS-03 capstone. {meta['rows']:,} YouTube comments from {meta['videos']} videos, "
           f"{meta['first']} to {meta['last']}. Aggregates only: no comment text or user ids are shown. "
           f"[Code and docs]({REPO})")

tabs = st.tabs(["Overview", "Sentiment", "Topics", "Languages", "Graph", "Misinformation", "Alerts", "About"])

# ---------------- Overview ----------------
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Comments", f"{meta['rows']:,}")
    c[1].metric("Videos", meta["videos"])
    c[2].metric("Topics found", m["topics"]["n_topics"])
    share = m["sentiment"]["label_share"]
    c[3].metric("Negative share", f"{share['negative'] * 100:.1f}%")
    st.markdown(
        "**What this is.** A pipeline that collects YouTube comments, scores sentiment with a multilingual "
        "transformer, finds topics with BERTopic, builds a reply graph, scores posts against a model trained on "
        "LIAR, and indexes everything into Elasticsearch/Kibana. This page shows its measured results.")
    st.info("Read the charts with care: about 39% of the comments fall in the last week of collection, so "
            "week-to-week trends are dominated by that week.")

# ---------------- Sentiment ----------------
with tabs[1]:
    st.subheader("Sentiment share per week")
    weeks = sorted(sent["week"].unique())
    default_start = weeks[max(0, len(weeks) - 52)]
    lo, hi = st.select_slider("Weeks shown", options=weeks, value=(default_start, weeks[-1]),
                              format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"))
    s = sent[(sent["week"] >= lo) & (sent["week"] <= hi)]
    piv = s.pivot_table(index="week", columns="sentiment", values="n", aggfunc="sum", fill_value=0)
    st.area_chart(piv.div(piv.sum(axis=1), axis=0), color=["#C0392B", "#8A94A6", "#1F8A7D"][: piv.shape[1]])
    st.caption("Share of comments per week. Weeks with very few comments swing a lot.")
    st.subheader("Accuracy")
    rows = [("TweetEval (English tweets)", m["bench_tweeteval"]["accuracy"], m["bench_tweeteval"]["macro_f1"])]
    names = {"en": "English", "de": "German", "es": "Spanish", "fr": "French", "ja": "Japanese", "zh": "Chinese"}
    for k, v in m["bench_amazon_by_language"].items():
        rows.append((f"Amazon reviews, {names.get(k, k)}", v["accuracy"], v["macro_f1"]))
    if review:
        rs = review["sentiment"]
        rows.append((f"Our own comments, hand-labelled (n={rs['n']})", rs["accuracy"], rs["macro_f1"]))
    st.dataframe(pd.DataFrame(rows, columns=["Test set", "Accuracy", "Macro-F1"]).round(3),
                 hide_index=True, width="stretch")
    st.caption(f"Throughput on a laptop CPU: {m['sentiment']['docs_per_sec']} comments/second.")

# ---------------- Topics ----------------
with tabs[2]:
    st.subheader("Largest topics")
    t = topics[topics["Topic"] != -1].sort_values("Count", ascending=False)
    t = t.assign(name=t["Name"].str.replace(r"^\d+_", "", regex=True))
    top = st.slider("How many topics", 5, 30, 12)
    st.bar_chart(t.head(top).set_index("name")["Count"], horizontal=True)
    out = topics.loc[topics["Topic"] == -1, "Count"].sum()
    st.caption(f"{m['topics']['n_topics']} topics. {m['topics']['outlier_share'] * 100:.1f}% of comments "
               f"({out:,}) fit no topic, mostly very short ones. Topic names are the top words of each cluster.")
    st.subheader("Topic volume per week")
    labels = sorted(tw["topic_label"].unique())
    pick = st.multiselect("Topics", labels, default=labels[:3])
    if pick:
        p = tw[tw["topic_label"].isin(pick)].pivot_table(index="week", columns="topic_label",
                                                         values="n", aggfunc="sum", fill_value=0)
        st.line_chart(p)

# ---------------- Languages ----------------
with tabs[3]:
    st.subheader("Language mix")
    st.bar_chart(langs.head(12).set_index("language")["n"])
    st.caption("'und' = too short or ambiguous to detect. Romanized Hindi/Telugu is often mislabelled.")

# ---------------- Graph ----------------
with tabs[4]:
    st.subheader("Reply graph")
    g = [("YouTube, all replies", m["graph"]), ("YouTube, comment to comment", m["graph_replies_only"]),
         ("SNAP subreddit hyperlinks", m["bench_snap_graph"])]
    df = pd.DataFrame([{"Graph": n, "Nodes": v["nodes"], "Edges": v["edges"], "Components": v["n_components"],
                        "Largest component": f"{v['largest_component_share'] * 100:.1f}%",
                        "Modularity": round(v["modularity"], 2)} for n, v in g])
    st.dataframe(df, hide_index=True, width="stretch")
    st.markdown(
        "- On YouTube most reply threads are a single exchange, so the graph is fragmented and no account "
        "dominates (top PageRank 0.004).\n"
        "- The very high modularity of the YouTube graphs comes from many tiny clusters, not strong communities.\n"
        "- The SNAP network (54k subreddits) shows the same code on a large connected graph; top PageRank: "
        "iama, askreddit, pics, funny, videos.")

# ---------------- Misinformation ----------------
with tabs[5]:
    st.subheader("Claim classifier trained on LIAR")
    t = m["misinfo_baseline"]["test"]
    c = st.columns(3)
    c[0].metric("Accuracy (LIAR test)", f"{t['accuracy']:.3f}")
    c[1].metric("F1", f"{t['f1']:.3f}")
    c[2].metric("ROC-AUC", f"{t['roc_auc']:.3f}")
    f1, f2 = st.columns(2)
    f1.markdown("**Words pointing to 'misleading'**  \n" + ", ".join(m["misinfo_top_features"]["misleading"][:10]))
    f2.markdown("**Words pointing to 'truthful'**  \n" + ", ".join(m["misinfo_top_features"]["truthful"][:10]))
    if review:
        fl = review["flagged"]
        st.markdown(f"**Flagged posts judged by reading them (n={fl['n']}):** " +
                    ", ".join(f"{k} {v}" for k, v in fl["counts"].items()))
    st.warning("LIAR is political fact-checks, not YouTube comments. The score means 'sounds like claims "
               "fact-checkers rated misleading'. It is a ranking aid for review, not a verdict.")

# ---------------- Alerts ----------------
with tabs[6]:
    st.subheader("Negative-sentiment spikes")
    a = m["alerts"]
    st.write(f"{a['negative_spikes']} spikes among {a['eligible_buckets']} topic-weeks with at least "
             f"{a['min_posts']} posts; {a['flagged_high_reach']} high-reach posts flagged for review.")
    if len(spikes):
        show = spikes[["topic_label", "bucket", "n", "neg", "mu", "z"]].copy()
        show.columns = ["Topic", "Week", "Posts", "Negative share", "Usual share", "z"]
        st.dataframe(show.round(2), hide_index=True, width="stretch")
    st.caption("A spike = a week where the topic's negative share is well above its share in other weeks "
               "(binomial z-test). Most are the last week of collection, so they show the mechanism, not news events.")

# ---------------- About ----------------
with tabs[7]:
    st.markdown(f"""
**Stack:** YouTube Data API v3, py3langid, XLM-R sentiment, BERTopic, NetworkX, Neo4j, scikit-learn (LIAR),
Elasticsearch and Kibana. Full code, tests and documentation: [{REPO}]({REPO}).

**Data and privacy:** this page contains aggregate counts only. Reddit data was not collected, because Reddit
requires approval for data access; the SNAP Reddit hyperlink graph (subreddit links) is used for scale tests.
Authors are replaced by salted hashes in the pipeline and are not published here.

**Limits:** one collection run, 39% of comments from its last week; sentiment model trained on tweets; misinformation
model trained on political fact-checks.
""")