"""Run the pipeline stage by stage:

    python -m socintel.run_pipeline clean
    python -m socintel.run_pipeline sentiment topics graph misinfo alerts index
    python -m socintel.run_pipeline all          # everything after collection

Every stage reads/writes data/processed/*.jsonl so you can rerun just one of them.
"""

import argparse
import json
import os
import time

import pandas as pd

from . import alerts, clean, graph, index_es, misinfo, normalize, sentiment, topics
from .io_utils import load_config, read_jsonl, write_jsonl

STAGES = ["clean", "sentiment", "topics", "graph", "misinfo", "alerts", "index"]


def p(cfg, name):
    return os.path.join(cfg["paths"]["processed"], name)


def log_metric(cfg, key, value):
    """append results to one json file; the README/presentation numbers come from here."""
    path = p(cfg, "metrics.json")
    m = json.load(open(path)) if os.path.exists(path) else {}
    m[key] = value
    json.dump(m, open(path, "w"), indent=2, default=str)


def stage_clean(cfg):
    df = normalize.build_table(cfg["paths"]["raw"], anonymize_authors=cfg.get("anonymize_authors", True))
    print("raw rows:", len(df))
    write_jsonl(df, p(cfg, "00_normalized.jsonl"))      # also the input for spark_stage.py
    df = clean.clean_table(df)
    print("after cleaning:", len(df))
    print(clean.language_report(df).head(8))
    write_jsonl(df, p(cfg, "01_clean.jsonl"))
    log_metric(cfg, "clean", {"rows": len(df), "languages": df["language"].value_counts().head(10).to_dict()})


def stage_sentiment(cfg):
    df = read_jsonl(p(cfg, "01_clean.jsonl"))
    sc = cfg["sentiment"]
    clf = sentiment.load_pipeline(sc["model"])
    t0 = time.time()
    df = sentiment.add_sentiment(df, clf, sc["batch_size"], sc["max_length"])
    secs = time.time() - t0
    write_jsonl(df, p(cfg, "02_sentiment.jsonl"))
    log_metric(cfg, "sentiment", {"label_share": df["sentiment"].value_counts(normalize=True).to_dict(),
                                  "docs_per_sec": round(len(df) / secs, 1)})
    try:
        ev = sentiment.eval_on_tweeteval(clf)
        print(ev["report"])
        log_metric(cfg, "sentiment_tweeteval", {k: ev[k] for k in ("accuracy", "macro_f1")})
    except Exception as e:      # no internet / HF down - not fatal
        print("skipping TweetEval check:", e)


def stage_topics(cfg):
    df = read_jsonl(p(cfg, "02_sentiment.jsonl"))
    tc = cfg["topics"]
    docs_all = df["text"].map(clean.topic_text)
    mask = docs_all.str.len() >= 3              # emoji-only / empty comments can't have a topic
    docs = docs_all[mask].tolist()
    model, tids = topics.fit_topics(docs, tc["embedding_model"], tc["min_topic_size"])
    col = pd.Series(-1, index=df.index)
    col[mask] = tids
    df = topics.add_topics(df, model, col.tolist())
    write_jsonl(df, p(cfg, "03_topics.jsonl"))
    tot = topics.topics_over_time(model, docs, df.loc[mask, "created_utc"].tolist(), tc["nr_bins"])
    tot.to_csv(p(cfg, "topics_over_time.csv"), index=False)
    model.get_topic_info().to_csv(p(cfg, "topic_info.csv"), index=False)
    n_topics = int((model.get_topic_info()["Topic"] != -1).sum())
    log_metric(cfg, "topics", {"n_topics": n_topics, "outlier_share": float((df["topic_id"] == -1).mean())})


def _graph_summary(g, ranks):
    summ = graph.graph_summary(g)
    summ["modularity"] = graph.modularity(g, ranks) if len(ranks) and g.number_of_edges() else None
    return summ


def stage_graph(cfg):
    df = read_jsonl(p(cfg, "03_topics.jsonl"))
    creators = set(df.loc[df["kind"] == "post", "author"])

    # 1) full graph: everyone -> whoever they answered, including commenter -> video owner.
    #    Owners dominate this one by construction; it is what we load into Neo4j.
    edges = graph.build_edges(df)
    g = graph.build_graph(edges)
    ranks = graph.rank_users(g)
    ranks["role"] = ranks["user"].map(lambda u: "creator" if u in creators else "commenter")
    summ = _graph_summary(g, ranks)
    print("full graph:", summ)
    edges.to_csv(p(cfg, "edges.csv"), index=False)
    ranks.to_csv(p(cfg, "user_ranks.csv"), index=False)
    log_metric(cfg, "graph", summ)

    # 2) replies-only graph (commenter -> commenter). This one ranks "influencers".
    e2 = graph.build_edges(df, replies_only=True)
    g2 = graph.build_graph(e2)
    r2 = graph.rank_users(g2)
    summ2 = _graph_summary(g2, r2)
    print("replies-only graph:", summ2)
    e2.to_csv(p(cfg, "edges_replies.csv"), index=False)
    r2.to_csv(p(cfg, "user_ranks_replies.csv"), index=False)
    log_metric(cfg, "graph_replies_only", summ2)

    try:
        graph.load_into_neo4j(edges, ranks, graph.neo4j_driver())
    except Exception as e:
        print("Neo4j load failed (is the container up?):", e)
    r = r2.set_index("user")
    df["author_pagerank"] = df["author"].map(r["pagerank"]) if len(r2) else None
    df["author_community"] = df["author"].map(r["community"]) if len(r2) else None
    write_jsonl(df, p(cfg, "04_graph.jsonl"))


def stage_misinfo(cfg):
    df = read_jsonl(p(cfg, "04_graph.jsonl"))
    model, res, data = misinfo.fit_and_report(cfg["misinfo"]["max_features"])
    print(json.dumps(res, indent=2))
    log_metric(cfg, "misinfo_baseline", res)
    log_metric(cfg, "misinfo_top_features", misinfo.top_features(model))
    misinfo.error_analysis(model, data["test"]).to_csv(p(cfg, "misinfo_errors.csv"), index=False)
    df = misinfo.score_posts(df, model)
    write_jsonl(df, p(cfg, "05_misinfo.jsonl"))


def stage_alerts(cfg):
    df = read_jsonl(p(cfg, "05_misinfo.jsonl"))
    ac = cfg["alerts"]
    freq = ac.get("freq", "W")
    min_posts = ac.get("min_posts", ac.get("min_posts_per_day", 15))
    elig = alerts.eligible_buckets(df, min_posts, freq)
    print(f"{elig} (topic, {freq}) buckets have >= {min_posts} posts and can be judged")
    a1 = alerts.negative_spikes(df, ac["neg_spike_zscore"], min_posts, freq)
    a2 = alerts.flagged_high_reach_posts(df, ac["influencer_top_n"])
    a1.to_csv(p(cfg, "alerts_neg_spikes.csv"), index=False)
    a2.to_csv(p(cfg, "alerts_flagged_high_reach.csv"), index=False)
    log_metric(cfg, "alerts", {"negative_spikes": len(a1), "flagged_high_reach": len(a2),
                               "eligible_buckets": elig, "freq": freq, "min_posts": min_posts})
    print("alerts:", len(a1), "spikes,", len(a2), "flagged high-reach posts")


def stage_index(cfg):
    df = read_jsonl(p(cfg, "05_misinfo.jsonl"))
    es = index_es.client()
    name = cfg["elasticsearch"]["index"]
    index_es.create_index(es, name, recreate=True)
    t0 = time.time()
    ok, errs = index_es.bulk_index(es, df, name)
    secs = time.time() - t0
    es.indices.refresh(index=name)
    t1 = time.time()
    index_es.search_text(es, name, "vaccine")
    q_ms = (time.time() - t1) * 1000
    log_metric(cfg, "elasticsearch", {"indexed": ok, "errors": len(errs),
                                      "docs_per_sec": round(ok / max(secs, 1e-9), 1),
                                      "sample_query_ms": round(q_ms, 1)})
    print(f"indexed {ok} docs, {len(errs)} errors")


def stage_benchmarks(cfg):
    """accuracy numbers on public datasets (no live data needed):
    TweetEval + Multilingual Amazon Reviews for sentiment, SNAP for graph scale."""
    bc = cfg["benchmarks"]
    sc = cfg["sentiment"]
    clf = sentiment.load_pipeline(sc["model"])
    try:
        ev = sentiment.eval_on_tweeteval(clf, n=bc["tweeteval_n"])
        print("TweetEval:", {k: round(ev[k], 3) for k in ("accuracy", "macro_f1")})
        log_metric(cfg, "bench_tweeteval", {k: ev[k] for k in ("accuracy", "macro_f1")})
    except Exception as e:
        print("TweetEval skipped:", e)
    am = sentiment.eval_on_amazon(clf, bc["amazon_langs"], bc["amazon_n"])
    for lang, r in am.items():
        print(f"amazon {lang}: acc={r['accuracy']:.3f} f1={r['macro_f1']:.3f} (n={r['n']})")
    log_metric(cfg, "bench_amazon_by_language", am)

    snap = bc["snap_path"]
    if os.path.exists(snap):
        edges = graph.snap_edges(snap)
        g = graph.build_graph(edges)
        ranks = graph.rank_users(g)
        summ = graph.graph_summary(g)
        summ["modularity"] = graph.modularity(g, ranks)
        print("SNAP graph:", summ)
        print(ranks.head(10)[["user", "pagerank", "community"]])
        ranks.head(100).to_csv(p(cfg, "snap_top_subreddits.csv"), index=False)
        log_metric(cfg, "bench_snap_graph", summ)
    else:
        print(f"SNAP file not found at {snap} - download it from snap.stanford.edu/data/soc-RedditHyperlinks.html")


FUNCS = {"clean": stage_clean, "sentiment": stage_sentiment, "topics": stage_topics,
         "graph": stage_graph, "misinfo": stage_misinfo, "alerts": stage_alerts,
         "index": stage_index, "benchmarks": stage_benchmarks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=STAGES + ["benchmarks", "all"])
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    stages = STAGES if "all" in args.stages else args.stages
    for s in stages:
        print(f"\n=== {s} ===")
        FUNCS[s](cfg)


if __name__ == "__main__":
    main()