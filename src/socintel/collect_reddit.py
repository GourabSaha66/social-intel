"""Reddit collector (PRAW). Stores the raw API payload as json, one file per submission.

Needs a Reddit "script" app: https://www.reddit.com/prefs/apps
PRAW already sleeps when it hits the rate limit; I add a small sleep between submissions
on top so we stay polite. Check Reddit's Data API terms before collecting at scale.
"""

import os
import time

import praw

from .io_utils import load_config, save_raw


def get_client():
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "socintel/0.1"),
        check_for_async=False,
    )


def submission_to_raw(sub, comments_limit):
    sub.comments.replace_more(limit=0)          # drop "load more" stubs, saves a lot of calls
    comments = []
    for c in sub.comments.list()[:comments_limit]:
        comments.append({
            "id": c.id,
            "author": str(c.author) if c.author else "[deleted]",
            "body": c.body,
            "created_utc": int(c.created_utc),
            "score": c.score,
            "parent_id": c.parent_id,           # t1_xxx (comment) or t3_xxx (submission)
        })
    return {
        "id": sub.id,
        "subreddit": str(sub.subreddit),
        "author": str(sub.author) if sub.author else "[deleted]",
        "title": sub.title,
        "selftext": sub.selftext,
        "created_utc": int(sub.created_utc),
        "score": sub.score,
        "num_comments": sub.num_comments,
        "url": sub.url,
        "comments": comments,
    }


def collect(cfg=None):
    cfg = cfg or load_config()
    rc = cfg["reddit"]
    if not rc.get("enabled", False):
        print("Reddit collection is disabled in config.yaml. Reddit requires explicit approval for "
              "data access (Responsible Builder Policy); only enable this once you have it.")
        return
    reddit = get_client()
    out = cfg["paths"]["raw"]
    n = 0
    for name in rc["subreddits"]:
        print(f"r/{name}")
        for sub in reddit.subreddit(name).hot(limit=rc["posts_per_subreddit"]):
            try:
                raw = submission_to_raw(sub, rc["comments_per_post"])
            except Exception as e:           # network blips, removed posts etc
                print("  skip", sub.id, e)
                continue
            save_raw(raw, out, f"reddit_{name}_{sub.id}")
            n += 1
            time.sleep(rc["sleep_between_calls"])
    print("saved", n, "submissions")


if __name__ == "__main__":
    collect()
