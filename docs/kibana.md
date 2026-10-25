# Kibana setup

1. `docker compose up -d`, then run the pipeline through the `index` stage.
2. `python scripts/setup_kibana.py` creates the `socintel-posts` data view (time field `created`).
3. Build these in Kibana (Analytics > Dashboard). Export them afterwards (Stack Management > Saved Objects)
   to `kibana/dashboards.ndjson` and re-import with `python scripts/setup_kibana.py --import kibana/dashboards.ndjson`.

**Dashboard 1 - Search and overview:** saved search on `clean_text`, filters for `source`, `language`, `community`;
metric tiles (docs, share negative); language pie.

**Dashboard 2 - Trends:** stacked area of `sentiment` over `created` (weekly); line of docs per `topic_label`
(top 10 topics); bar of avg `sentiment_score` by `community`.

**Dashboard 3 - Alerts and influencers:** table of top authors by `author_pagerank`; table filtered on
`misinfo_prob >= 0.7` and high pagerank; a Kibana rule (Stack Management > Rules > Elasticsearch query) with
the KQL `sentiment : "negative"` over the last 24h, threshold set above the normal weekly count for the topic.

The pandas alert logic in `src/socintel/alerts.py` is what the unit tests cover; the Kibana rule is the
operational version of the same idea.

## Ready-made dashboards

`kibana/dashboards.ndjson` holds the two dashboards, their charts and the saved search. After `docker compose up -d`
and the `index` stage: `python scripts/setup_kibana.py --import kibana/dashboards.ndjson`.

- **Trends:** sentiment share per week, topic volume per week, language mix.
- **Influencers and alerts:** top authors by PageRank (hashed ids) and the saved search
  `misinfo_prob >= 0.7 and author_pagerank >= <PageRank of the 50th-ranked author>`, the same rule as
  `alerts.flagged_high_reach_posts` (authors tied at the cut-off are kept).

Kibana shows numbers with 3 decimals, so small PageRank values display as 0.

## Small machines

Docker is the heavy part. On an 8 GB laptop, cap Docker's WSL memory at 4 GB (`.wslconfig`) and start only
Elasticsearch and Kibana (`docker compose up -d`); Neo4j sits behind a profile:
`docker compose --profile graph up -d neo4j`.