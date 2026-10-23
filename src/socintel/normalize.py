"""Turn the raw reddit/youtube json dumps into one flat table that follows schema.Post."""

import calendar
import time

import pandas as pd

from .io_utils import iter_raw
from .schema import CORE_FIELDS


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
    return rows


def build_table(raw_dir):
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
    return df
