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


def test_amazon_stars_from_label_text_or_label():
    import pandas as pd
    with_text = pd.DataFrame({"label": [0, 4], "label_text": ["1 star", "5 stars"]})
    assert amazon_list(with_text) == [1, 5]
    only_label = pd.DataFrame({"label": [0, 2, 4]})
    assert amazon_list(only_label) == [1, 3, 5]
    with pytest.raises(ValueError):
        sentiment.amazon_stars(pd.DataFrame({"label": [7]}))


def amazon_list(df):
    return sentiment.amazon_stars(df).tolist()


# ---------- youtube collector error handling ----------

def test_comments_disabled_does_not_crash(monkeypatch):
    from socintel import collect_youtube as cy

    def fake_get(endpoint, params, retries=3):
        raise cy.CommentsDisabled("v1")
    monkeypatch.setattr(cy, "_get", fake_get)
    assert cy.fetch_comments("v1", 50) == []


def test_comments_paginate_until_enough(monkeypatch):
    from socintel import collect_youtube as cy
    pages = [{"items": [1, 2, 3], "nextPageToken": "t"}, {"items": [4, 5]}]

    def fake_get(endpoint, params, retries=3):
        return pages.pop(0)
    monkeypatch.setattr(cy, "_get", fake_get)
    assert cy.fetch_comments("v1", 100) == [1, 2, 3, 4, 5]


def test_quota_exceeded_is_not_swallowed(monkeypatch):
    import requests
    from socintel import collect_youtube as cy

    class R:
        status_code = 403
        text = '{"error": {"errors": [{"reason": "quotaExceeded"}]}}'
    monkeypatch.setattr(requests, "get", lambda *a, **k: R())
    monkeypatch.setenv("YOUTUBE_API_KEY", "x")
    with pytest.raises(RuntimeError):
        cy._get("search", {})


def test_config_with_hindi_query_loads_as_utf8(tmp_path):
    from socintel.io_utils import load_config
    f = tmp_path / "c.yaml"
    f.write_text('youtube:\n  queries: ["भारत समाचार", "climate change"]\n', encoding="utf-8")
    assert load_config(str(f))["youtube"]["queries"][0] == "भारत समाचार"
    assert load_config("config.yaml")["youtube"]["queries"][-1].startswith("nachrichten")


def test_collect_stops_cleanly_when_quota_runs_out(monkeypatch, tmp_path, capsys):
    from socintel import collect_youtube as cy
    calls = {"n": 0}

    def fake_search(q, n):
        calls["n"] += 1
        if calls["n"] == 2:
            raise cy.QuotaExceeded("YouTube quota exhausted for today")
        return [{"id": {"videoId": "v%d" % calls["n"]}, "snippet": {}}]
    monkeypatch.setattr(cy, "search_videos", fake_search)
    monkeypatch.setattr(cy, "fetch_comments", lambda vid, n: [])
    cfg = {"youtube": {"queries": ["a", "b", "c"], "videos_per_query": 1, "comments_per_video": 1,
                       "sleep_between_calls": 0}, "paths": {"raw": str(tmp_path)}}
    cy.collect(cfg)
    assert len(list(tmp_path.glob("youtube_*.json"))) == 1       # first query saved, then stopped
    assert "stopped early" in capsys.readouterr().out


def _cfg(tmp_path, queries):
    return {"youtube": {"queries": queries, "videos_per_query": 5, "comments_per_video": 5,
                        "sleep_between_calls": 0}, "paths": {"raw": str(tmp_path)}}


def test_results_without_video_id_are_skipped(monkeypatch, tmp_path):
    from socintel import collect_youtube as cy
    items = [{"id": {"kind": "youtube#channel", "channelId": "c1"}, "snippet": {}},
             {"id": {"videoId": "ok1"}, "snippet": {}}, {"snippet": {}}]
    monkeypatch.setattr(cy, "search_videos", lambda q, n: items)
    monkeypatch.setattr(cy, "fetch_comments", lambda vid, n: [])
    cy.collect(_cfg(tmp_path, ["q"]))
    assert [p.name.split("_")[1] for p in tmp_path.glob("youtube_*.json")] == ["ok1"]


def test_one_bad_video_does_not_stop_the_run(monkeypatch, tmp_path):
    from socintel import collect_youtube as cy
    monkeypatch.setattr(cy, "search_videos",
                        lambda q, n: [{"id": {"videoId": v}, "snippet": {}} for v in ("a", "bad", "c")])

    def fetch(vid, n):
        if vid == "bad":
            raise RuntimeError("gave up on commentThreads")
        return []
    monkeypatch.setattr(cy, "fetch_comments", fetch)
    cy.collect(_cfg(tmp_path, ["q"]))
    assert sorted(p.name.split("_")[1] for p in tmp_path.glob("youtube_*.json")) == ["a", "c"]


def test_rerun_resumes_and_handles_ids_with_underscores(monkeypatch, tmp_path):
    from socintel import collect_youtube as cy
    (tmp_path / "youtube_ab_cd-9_1700000000000.json").write_text("{}")
    assert cy.already_collected(str(tmp_path)) == {"ab_cd-9"}
    fetched = []
    monkeypatch.setattr(cy, "search_videos", lambda q, n: [{"id": {"videoId": v}, "snippet": {}}
                                                          for v in ("ab_cd-9", "new1")])
    monkeypatch.setattr(cy, "fetch_comments", lambda vid, n: fetched.append(vid) or [])
    cy.collect(_cfg(tmp_path, ["q"]))
    assert fetched == ["new1"]                                   # already-saved video not fetched again