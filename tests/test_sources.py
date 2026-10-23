import pandas as pd
import pytest

from socintel import graph, normalize, sentiment


def yt_raw():
    return {"query": "vaccine",
            "video": {"id": {"videoId": "v1"},
                      "snippet": {"channelTitle": "Chan", "title": "T", "description": "D",
                                  "publishedAt": "2024-05-01T00:00:00Z"}},
            "comments": [{
                "snippet": {"topLevelComment": {"id": "k1", "snippet": {
                    "authorDisplayName": "zed", "textDisplay": "nice", "publishedAt": "2024-05-01T01:00:00Z",
                    "likeCount": 3}}},
                "replies": {"comments": [
                    {"id": "k1.r1", "snippet": {"authorDisplayName": "amy", "textDisplay": "agreed",
                                                "publishedAt": "2024-05-01T02:00:00Z", "likeCount": 1}},
                    {"id": "k1.r2", "snippet": {"authorDisplayName": "zed", "textDisplay": "thanks amy",
                                                "publishedAt": "2024-05-01T03:00:00Z", "likeCount": 0}}]}}]}


def test_youtube_replies_are_parsed_with_parent():
    rows = normalize.youtube_rows(yt_raw())
    assert len(rows) == 4                       # video + 1 comment + 2 replies
    by_id = {r["post_id"]: r for r in rows}
    assert by_id["y_k1.r1"]["parent_id"] == "y_k1"
    assert by_id["y_k1.r2"]["parent_id"] == "y_k1"


def test_youtube_reply_graph_has_user_edges():
    df = pd.DataFrame(normalize.youtube_rows(yt_raw()))
    e = graph.build_edges(df)
    pairs = {(r.src, r.dst) for r in e.itertuples()}
    assert ("amy", "zed") in pairs              # amy replied to zed
    assert ("zed", "Chan") in pairs             # zed commented on Chan's video
    assert ("zed", "zed") not in pairs          # zed's reply to his own comment is a self reply, dropped


def test_anonymize_is_stable_and_hides_names():
    df = pd.DataFrame({"author": ["alice", "bob", "alice", "[deleted]"]})
    out = normalize.anonymize(df, salt="s1")
    assert "alice" not in out["author"].tolist()
    assert out["author"].iloc[0] == out["author"].iloc[2]       # same user -> same id
    assert out["author"].iloc[0] != out["author"].iloc[1]
    assert out["author"].iloc[3] == "[deleted]"                  # kept so the graph can skip it
    other = normalize.anonymize(df, salt="s2")
    assert other["author"].iloc[0] != out["author"].iloc[0]      # salt changes the ids


def test_anonymized_graph_still_ranks():
    df = pd.DataFrame({"post_id": ["p", "a", "b"], "author": ["alice", "bob", "carl"],
                       "parent_id": [None, "p", "p"]})
    e = graph.build_edges(normalize.anonymize(df, salt="x"))
    assert len(e) == 2 and e["dst"].nunique() == 1


def test_snap_loader(tmp_path):
    f = tmp_path / "snap.tsv"
    f.write_text("SOURCE_SUBREDDIT\tTARGET_SUBREDDIT\tPOST_ID\tTIMESTAMP\tLINK_SENTIMENT\tPROPERTIES\n"
                 "a\tb\tp1\t2014-01-01 10:00:00\t1\t1,2\n"
                 "a\tb\tp2\t2014-01-02 10:00:00\t-1\t1,2\n"
                 "c\tb\tp3\t2014-01-03 10:00:00\t1\t1,2\n"
                 "b\tb\tp4\t2014-01-04 10:00:00\t1\t1,2\n")             # self link is dropped
    e = graph.snap_edges(str(f))
    row = e[(e["src"] == "a") & (e["dst"] == "b")].iloc[0]
    assert row["weight"] == 2 and row["sent"] == 0.0
    assert len(e) == 2
    ranks = graph.rank_users(graph.build_graph(e))
    assert ranks.iloc[0]["user"] == "b"


def test_amazon_star_mapping():
    assert [sentiment.STARS_TO_LABEL[i] for i in range(5)] == [
        "negative", "negative", "neutral", "positive", "positive"]
