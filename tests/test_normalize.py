from socintel.normalize import reddit_rows, youtube_rows


def test_reddit_rows_structure():
    raw = {"id": "abc", "subreddit": "news", "author": "alice", "title": "Title",
           "selftext": "body", "created_utc": 1700000000, "score": 5,
           "comments": [
               {"id": "c1", "author": "bob", "body": "hi", "created_utc": 1700000100,
                "score": 2, "parent_id": "t3_abc"},
               {"id": "c2", "author": "carol", "body": "yo", "created_utc": 1700000200,
                "score": 1, "parent_id": "t1_c1"},
           ]}
    rows = reddit_rows(raw)
    assert len(rows) == 3
    assert rows[0]["post_id"] == "r_abc" and rows[0]["text"] == "Title\nbody"
    assert rows[1]["parent_id"] == "r_abc"       # comment on the submission
    assert rows[2]["parent_id"] == "r_c1"        # reply to a comment
    assert all(r["community"] == "news" for r in rows)


def test_youtube_rows_structure():
    raw = {"query": "vaccine",
           "video": {"id": {"videoId": "v1"},
                     "snippet": {"channelTitle": "Chan", "title": "T", "description": "D",
                                 "publishedAt": "2024-05-01T00:00:00Z"}},
           "comments": [{"snippet": {"topLevelComment": {
               "id": "k1", "snippet": {"authorDisplayName": "zed", "textDisplay": "nice",
                                       "publishedAt": "2024-05-01T01:00:00Z", "likeCount": 3}}}}]}
    rows = youtube_rows(raw)
    assert len(rows) == 2
    assert rows[0]["kind"] == "post" and rows[1]["kind"] == "comment"
    assert rows[1]["parent_id"] == "y_v1" and rows[1]["score"] == 3
    assert rows[1]["created_utc"] - rows[0]["created_utc"] == 3600
