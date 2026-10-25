"""YouTube Data API v3 collector (plain requests, no extra client lib).

Quota: the default is 10,000 units/day. search.list costs 100 units, commentThreads.list
costs 1. So ~5 queries x 1 search page = 500 units, and comments are cheap. Don't loop
search pages carelessly or you'll burn the quota in minutes.
"""

import os
import re
import time

import requests

from .io_utils import load_config, save_raw

BASE = "https://www.googleapis.com/youtube/v3"


class QuotaExceeded(RuntimeError):
    """daily quota used up - nothing more can be fetched until it resets"""


class CommentsDisabled(Exception):
    """the video owner turned comments off (youtube answers 403 commentsDisabled)"""


def _get(endpoint, params, retries=3):
    params = dict(params, key=os.environ["YOUTUBE_API_KEY"])
    for attempt in range(retries):
        r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (403, 429, 500, 503):
            # 403 can mean several things; two of them should not be retried
            if "quotaExceeded" in r.text:
                raise QuotaExceeded("YouTube quota exhausted for today")
            if "commentsDisabled" in r.text:
                raise CommentsDisabled(params.get("videoId"))
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
        # "replies" brings up to 5 replies per thread in the same call (no extra quota),
        # and replies are what give us edges between users for the graph stage
        params = {"part": "snippet,replies", "videoId": video_id, "maxResults": min(100, n - len(comments)),
                  "textFormat": "plainText"}
        if token:
            params["pageToken"] = token
        try:
            data = _get("commentThreads", params)
        except (CommentsDisabled, requests.HTTPError):
            break                            # comments off or video unavailable, keep what we have
        comments.extend(data.get("items", []))
        token = data.get("nextPageToken")
        if not token:
            break
    return comments


_RAW_NAME = re.compile(r"^youtube_(.+)_\d+\.json$")      # video ids may contain '_' and '-'


def already_collected(folder):
    """video ids that already have a raw file, so a rerun resumes instead of starting over."""
    if not os.path.isdir(folder):
        return set()
    return {m.group(1) for m in (_RAW_NAME.match(n) for n in os.listdir(folder)) if m}


def collect(cfg=None):
    cfg = cfg or load_config()
    yc = cfg["youtube"]
    out = cfg["paths"]["raw"]
    done = already_collected(out)
    saved = skipped = 0
    try:
        for q in yc["queries"]:
            print("query:", q)
            try:
                items = search_videos(q, yc["videos_per_query"])
            except QuotaExceeded:
                raise
            except Exception as e:               # one bad search should not end the run
                print("  search failed, skipping query:", e)
                continue
            for v in items:
                vid = v.get("id", {}).get("videoId")
                if not vid:                      # channels / playlists / odd results
                    skipped += 1
                    continue
                if vid in done:
                    continue
                try:
                    comments = fetch_comments(vid, yc["comments_per_video"])
                except QuotaExceeded:
                    raise
                except Exception as e:           # skip this video, keep going
                    print("  skipping", vid, "-", e)
                    continue
                save_raw({"query": q, "video": v, "comments": comments}, out, f"youtube_{vid}")
                done.add(vid)
                saved += 1
                time.sleep(yc["sleep_between_calls"])
    except QuotaExceeded as e:                   # everything saved so far is kept
        print("stopped early:", e)
    print(f"saved {saved} new videos this run ({skipped} non-video results skipped)")


if __name__ == "__main__":
    collect()