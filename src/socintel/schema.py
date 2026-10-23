"""Shared document schema. Every stage reads/writes records with these fields,
and the Elasticsearch mapping in index_es.py is built from the same list."""

from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class Post:
    post_id: str
    source: str                 # "reddit" | "youtube"
    kind: str                   # "post" | "comment"
    author: str
    created_utc: int            # unix seconds
    text: str
    parent_id: Optional[str] = None     # id of the post/comment being replied to
    thread_id: Optional[str] = None     # submission id / video id
    community: Optional[str] = None     # subreddit or youtube query
    score: int = 0
    # filled in by later stages
    clean_text: Optional[str] = None
    language: Optional[str] = None
    sentiment: Optional[str] = None     # negative | neutral | positive
    sentiment_score: Optional[float] = None
    topic_id: Optional[int] = None
    topic_label: Optional[str] = None
    misinfo_prob: Optional[float] = None
    author_pagerank: Optional[float] = None
    author_community: Optional[int] = None

    def to_dict(self):
        return asdict(self)


CORE_FIELDS = ["post_id", "source", "kind", "author", "created_utc", "text",
               "parent_id", "thread_id", "community", "score"]

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
