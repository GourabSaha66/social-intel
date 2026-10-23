"""Turn the raw reddit/youtube json dumps into one flat table that follows schema.Post."""

import calendar
import hashlib
import os
import time

import pandas as pd

from .io_utils import iter_raw
from .schema import CORE_FIELDS

KEEP_AS_IS = {"[deleted]", "unknown"}     # graph.py already skips these


def _yt_ts(s):
    # "2024-05-01T12:34:56Z"
    return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))


def reddit_rows(raw):
    rows = []
    sid = raw["id"]
    text = (raw.get("title") or "") + ("\n" + raw["selftext"] if raw.get("selftext") else "")
    rows.append(dict(post_id=f"r_{sid}", source="reddit", kind="post", author=raw["author"],
                     created_utc=raw["created_utc"], text=text, parent_id=None, thread_id=sid,
                     community=raw["subreddit"], score=raw["score"]))
    for c in raw.get("comments", []):
        parent = c["parent_id"].split("_", 1)[-1]
        rows.append(dict(post_id=f"r_{c['id']}", source="reddit", kind="comment", author=c["author"],
                         created_utc=c["created_utc"], text=c["body"], parent_id=f"r_{parent}",
                         thread_id=sid, community=raw["subreddit"], score=c["score"]))
    return rows


def youtube_rows(raw):
    rows = []
    vid = raw["video"]["id"]["videoId"]
    sn = raw["video"]["snippet"]
    rows.append(dict(post_id=f"y_{vid}", source="youtube", kind="post", author=sn["channelTitle"],
                     created_utc=_yt_ts(sn["publishedAt"]),
                     text=sn["title"] + "\n" + sn.get("description", ""), parent_id=None,
                     thread_id=vid, community=raw["query"], score=0))
    for item in raw.get("comments", []):
        top = item["snippet"]["topLevelComment"]
        cs = top["snippet"]
        rows.append(dict(post_id=f"y_{top['id']}", source="youtube", kind="comment",
                         author=cs.get("authorDisplayName", "unknown"),
                         created_utc=_yt_ts(cs["publishedAt"]), text=cs["textDisplay"],
                         parent_id=f"y_{vid}", thread_id=vid, community=raw["query"],
                         score=cs.get("likeCount", 0)))
        for rep in item.get("replies", {}).get("comments", []):
            rs = rep["snippet"]
            rows.append(dict(post_id=f"y_{rep['id']}", source="youtube", kind="comment",
                             author=rs.get("authorDisplayName", "unknown"),
                             created_utc=_yt_ts(rs["publishedAt"]), text=rs["textDisplay"],
                             parent_id=f"y_{top['id']}", thread_id=vid, community=raw["query"],
                             score=rs.get("likeCount", 0)))
    return rows


def anonymize(df, salt=None):
    """replace author names with a salted hash. same person -> same id, so the graph still
    works, but nobody's handle ends up in our stored files, dashboards or slides."""
    salt = salt if salt is not None else os.environ.get("AUTHOR_SALT", "socintel")
    df = df.copy()
    df["author"] = df["author"].map(
        lambda a: a if a in KEEP_AS_IS else "u_" + hashlib.sha256((salt + str(a)).encode()).hexdigest()[:10])
    return df


def build_table(raw_dir, anonymize_authors=True, salt=None):
    rows = []
    for name, raw in iter_raw(raw_dir):
        try:
            if name.startswith("reddit"):
                rows += reddit_rows(raw)
            elif name.startswith("youtube"):
                rows += youtube_rows(raw)
        except KeyError as e:
            print(f"malformed file {name}: missing {e}")
    df = pd.DataFrame(rows, columns=CORE_FIELDS)
    df = df.drop_duplicates("post_id").reset_index(drop=True)
    if anonymize_authors:
        df = anonymize(df, salt)
    return df
