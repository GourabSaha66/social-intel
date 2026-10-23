# Data and API documentation

## Raw files (`data/raw/*.json`)

One file per Reddit submission or YouTube video, written exactly as collected.

**Reddit** (only if approved; disabled by default) (`reddit_<subreddit>_<id>_<ms>.json`): `id, subreddit, author, title, selftext, created_utc, score,
num_comments, url, comments[]` where each comment has `id, author, body, created_utc, score, parent_id`
(`t3_` prefix = reply to the submission, `t1_` = reply to a comment).

**YouTube** (`youtube_<videoId>_<ms>.json`): `query`, `video` (the `search.list` item), `comments`
(raw `commentThreads.list` items, including up to 5 inline replies per thread).

## Shared schema (one row per post or comment)

| field | type | notes |
|---|---|---|
| post_id | keyword | `r_` or `y_` prefix + platform id, unique |
| source | keyword | reddit / youtube |
| kind | keyword | post / comment |
| author | keyword | salted hash of the username/channel (`u_xxxxxxxxxx`); see `anonymize_authors` |
| created_utc (`created` in ES) | int / date | UTC |
| text | text | original text |
| clean_text | text | URLs and mentions stripped, emojis converted to words |
| parent_id, thread_id | keyword | reply structure, used to build the graph |
| community | keyword | subreddit or search query |
| score | int | upvotes / likes |
| language | keyword | ISO code, `und` if text is too short to tell |
| sentiment, sentiment_score | keyword, float | negative / neutral / positive and model confidence |
| topic_id, topic_label | int, keyword | BERTopic; -1 = outlier |
| misinfo_prob | float | probability from the LIAR-trained classifier |
| author_pagerank, author_community | float, int | from the interaction graph |

## Processed files (`data/processed/`)

`01_clean.jsonl` -> `02_sentiment.jsonl` -> `03_topics.jsonl` -> `04_graph.jsonl` -> `05_misinfo.jsonl`,
plus `metrics.json` (all numbers quoted in the README/slides), `topics_over_time.csv`, `topic_info.csv`,
`edges.csv`, `user_ranks.csv`, `misinfo_errors.csv`, `alerts_*.csv`.

## External APIs

- **Reddit** via PRAW, disabled by default: Reddit requires approval for data access.
- **YouTube Data API v3**: 10,000 quota units per day by default. `search.list` costs 100, `commentThreads.list` costs 1.
  The collector stops cleanly on `quotaExceeded`.

## Handy Elasticsearch queries

```
GET socintel-posts/_search {"query": {"match": {"clean_text": "vaccine"}}}
GET socintel-posts/_search {"size": 0, "aggs": {"d": {"date_histogram": {"field": "created", "calendar_interval": "1d"},
                            "aggs": {"s": {"terms": {"field": "sentiment"}}}}}}
```

## Neo4j queries

```
MATCH (u:User) RETURN u.name, u.pagerank ORDER BY u.pagerank DESC LIMIT 20
MATCH (a:User)-[r:REPLIED_TO]->(b:User) WHERE b.name = 'someone' RETURN a, r, b
```
