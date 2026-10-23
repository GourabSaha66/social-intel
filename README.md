# Social Media Intelligence Platform

Collects Reddit and YouTube posts, then works out what people are talking about, how they feel about it,
who drives the conversation, and which posts look like known misinformation. Everything ends up in
Elasticsearch and Kibana for search, trends and alerts.

![architecture](docs/architecture.png)

## What it does

| Stage | What | Tools |
|---|---|---|
| 1. Collect | Reddit submissions + comments, YouTube videos + comments, raw JSON on disk | PRAW, YouTube Data API v3 |
| 2. Clean | strip URLs/mentions/markdown, emojis turned into words, language detection, dedupe | `emoji`, `py3langid` (Spark NLP version in `spark_stage.py`) |
| 3. Sentiment | multilingual negative/neutral/positive, tracked per day | `cardiffnlp/twitter-xlm-roberta-base-sentiment` |
| 4. Topics | topic discovery and topics over time | BERTopic + multilingual MiniLM embeddings |
| 5. Graph | who replies to whom, PageRank influencers, Louvain communities | NetworkX, Neo4j |
| 6. Misinformation | claim classifier trained on LIAR (TF-IDF + LogReg baseline, optional DistilBERT) | scikit-learn, transformers |
| 7. Index | bulk index into one schema, dashboards, alerts | Elasticsearch 8, Kibana 8 |

## Setup

Needs Python 3.10+ and Docker.

```bash
git clone <this repo> && cd social-intel
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env            # add Reddit + YouTube credentials
docker compose up -d            # Elasticsearch :9200, Kibana :5601, Neo4j :7474
```

Credentials: a Reddit "script" app at https://www.reddit.com/prefs/apps, and a YouTube Data API key from
Google Cloud Console.

LIAR: either let `datasets` fetch it, or put the original `train.tsv`, `valid.tsv`, `test.tsv` in `data/liar/`.

## Run it

```bash
export PYTHONPATH=src
python -m socintel.collect_reddit
python -m socintel.collect_youtube        # watch the daily quota, see docstring

python -m socintel.run_pipeline all        # clean -> sentiment -> topics -> graph -> misinfo -> alerts -> index
# or one at a time:
python -m socintel.run_pipeline clean sentiment

python scripts/setup_kibana.py             # creates the data view; then follow docs/kibana.md
```

Intermediate files go to `data/processed/`. All headline numbers are written to `data/processed/metrics.json`.

## Tests

```bash
python -m pytest -q
```

The tests run on small synthetic data and need no network, models or Docker: cleaning, normalisation, sentiment
aggregation, graph/PageRank/communities, misinformation classifier, alert rules and Elasticsearch document
preparation.

## Repo layout

```
src/socintel/   one module per stage + run_pipeline.py
tests/          pytest suite
docs/           architecture, data dictionary, Kibana guide, results, presentation outline
scripts/        kibana setup, diagram generator
config.yaml     subreddits, queries, model names, thresholds
```

## Design decisions and caveats

- **One schema.** Every stage works on the same columns (`docs/data_dictionary.md`), so indexing is trivial.
- **Emojis are converted to words, not dropped**, because they carry sentiment.
- **Language `und`** is used for very short texts, where language ID is unreliable.
- **Misinformation model is trained on LIAR** (political fact-check statements). Reddit and YouTube comments are a
  different kind of text, so `misinfo_prob` means "resembles claims that fact-checkers rated misleading", not
  "is false". Treat it as a ranking signal for review. LIAR labels are also collapsed to binary
  (pants-fire/false/barely-true = 1); that cut-off is a choice.
- **Reply graph only sees replies inside the collected data.** Sparse crawls give sparse graphs; the SNAP Reddit
  Hyperlink Network is a fallback for graph-only experiments.
- **Spark NLP stage** (`spark_stage.py`) is written but was not executed in the development environment.
  Verify the pretrained model name before using it.
- Respect Reddit's Data API terms and YouTube's API Services terms; don't redistribute raw data.

## Results

See `docs/results.md` (numbers come from `data/processed/metrics.json` after a run).

## Next steps

Fine-tune the misinformation model on in-domain labelled posts; add sentiment evaluation per language; stream
collection instead of batch; build the Kibana alert rule into Watcher/Slack notifications.
