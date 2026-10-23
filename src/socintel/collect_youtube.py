"""YouTube Data API v3 collector (plain requests, no extra client lib).

Quota: the default is 10,000 units/day. search.list costs 100 units, commentThreads.list
costs 1. So ~5 queries x 1 search page = 500 units, and comments are cheap. Don't loop
search pages carelessly or you'll burn the quota in minutes.
"""

import os
import time

import requests

from .io_utils import load_config, save_raw

BASE = "https://www.googleapis.com/youtube/v3"


def _get(endpoint, params, retries=3):
    params = dict(params, key=os.environ["YOUTUBE_API_KEY"])
    for attempt in range(retries):
        r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (403, 429, 500, 503):
            # 403 can also be quotaExceeded; no point retrying that one
            if "quotaExceeded" in r.text:
                raise RuntimeError("YouTube quota exhausted for today")
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
    raise RuntimeError(f"gave up on {endpoint}")


def search_videos(query, n):
    data = _get("search", {"part": "snippet", "q": query, "type": "video",
                           "maxResults": min(n, 50), "order": "relevance"})
    return data.get("items", [])


def fetch_comments(video_id, n):
    comments, token = [], None
    while len(comments) < n:
        params = {"part": "snippet", "videoId": video_id, "maxResults": min(100, n - len(comments)),
                  "textFormat": "plainText"}
        if token:
            params["pageToken"] = token
        try:
            data = _get("commentThreads", params)
        except requests.HTTPError:
            break                            # comments disabled on the video
        comments.extend(data.get("items", []))
        token = data.get("nextPageToken")
        if not token:
            break
    return comments


def collect(cfg=None):
    cfg = cfg or load_config()
    yc = cfg["youtube"]
    out = cfg["paths"]["raw"]
    for q in yc["queries"]:
        print("query:", q)
        for v in search_videos(q, yc["videos_per_query"]):
            vid = v["id"]["videoId"]
            raw = {"query": q, "video": v, "comments": fetch_comments(vid, yc["comments_per_video"])}
            save_raw(raw, out, f"youtube_{vid}")
            time.sleep(yc["sleep_between_calls"])


if __name__ == "__main__":
    collect()
