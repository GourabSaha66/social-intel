"""Index the enriched table into Elasticsearch (bulk) and manage the mapping."""

import os

import pandas as pd
from elasticsearch import Elasticsearch, helpers

MAPPING = {
    "properties": {
        "post_id": {"type": "keyword"},
        "source": {"type": "keyword"},
        "kind": {"type": "keyword"},
        "author": {"type": "keyword"},
        "created": {"type": "date"},                 # derived from created_utc
        "text": {"type": "text"},
        "clean_text": {"type": "text"},
        "parent_id": {"type": "keyword"},
        "thread_id": {"type": "keyword"},
        "community": {"type": "keyword"},
        "score": {"type": "integer"},
        "language": {"type": "keyword"},
        "sentiment": {"type": "keyword"},
        "sentiment_score": {"type": "float"},
        "topic_id": {"type": "integer"},
        "topic_label": {"type": "keyword"},
        "misinfo_prob": {"type": "float"},
        "author_pagerank": {"type": "float"},
        "author_community": {"type": "integer"},
    }
}


def client():
    return Elasticsearch(os.environ.get("ES_URL", "http://localhost:9200"))


def create_index(es, name, recreate=False):
    if es.indices.exists(index=name):
        if not recreate:
            return
        es.indices.delete(index=name)
    es.indices.create(index=name, mappings=MAPPING,
                      settings={"number_of_shards": 1, "number_of_replicas": 0})


def to_actions(df, index):
    d = df.copy()
    d["created"] = pd.to_datetime(d["created_utc"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    d = d.drop(columns=["created_utc"])
    d = d.astype(object).where(d.notna(), None)         # NaN -> null, ES rejects NaN
    for rec in d.to_dict("records"):
        yield {"_index": index, "_id": rec["post_id"], "_source": rec}


def bulk_index(es, df, index, chunk=1000):
    ok, errors = helpers.bulk(es, to_actions(df, index), chunk_size=chunk,
                              raise_on_error=False, stats_only=False)
    return ok, errors


# example queries used in the demo + docs
def search_text(es, index, q, size=10, lang=None):
    must = [{"match": {"clean_text": q}}]
    if lang:
        must.append({"term": {"language": lang}})
    return es.search(index=index, query={"bool": {"must": must}}, size=size)


def sentiment_trend(es, index, interval="1d"):
    return es.search(index=index, size=0, aggs={
        "by_day": {"date_histogram": {"field": "created", "calendar_interval": interval},
                   "aggs": {"s": {"terms": {"field": "sentiment"}}}}})
