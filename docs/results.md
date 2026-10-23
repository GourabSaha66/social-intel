# Results

> Fill this in from `data/processed/metrics.json` and the CSVs after running the pipeline on real data.
> Do not quote numbers that the pipeline did not produce.

## Data collected
- YouTube videos / comments / replies: _
- Public benchmarks used: TweetEval, Amazon multilingual, LIAR, SNAP
- After cleaning: _ rows, language mix: _ (`metrics.json` -> `clean`)

## Sentiment
- Label distribution on our data: _
- TweetEval (English test sample) accuracy / macro-F1: _ / _  (`bench_tweeteval`)
- Multilingual Amazon Reviews, accuracy / macro-F1 per language: _ (`bench_amazon_by_language`)
- Throughput (docs/sec on CPU or GPU used): _
- Observations: sentiment over time chart, any events visible, languages where it looks weak.

## Topics
- Number of topics, outlier share: _ (`topics`)
- Topic coherence (optional, gensim `CoherenceModel`, c_v): _
- Topics that rose/fell: _ (`topics_over_time.csv`)

## Graph
- YouTube reply graph: nodes / edges / density / largest component / modularity: _ (`graph`)
- SNAP subreddit graph at scale: _ (`bench_snap_graph`), top subreddits by PageRank: _ (`snap_top_subreddits.csv`)
- Top 10 influencers by PageRank and what they posted: _ (`user_ranks.csv`)

## Misinformation (LIAR, binary)
| model | accuracy | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| TF-IDF + LogReg | _ | _ | _ | _ | _ |
| DistilBERT (optional) | _ | _ | _ | _ | _ |

- Error analysis: look at `misinfo_errors.csv`, describe 3 to 5 real failures and why.
- Domain-shift limitation: see README.

## Elasticsearch
- Docs indexed, errors, docs/sec, sample query latency: _ (`elasticsearch`)

## Alerts
- Negative-sentiment spikes found: _; flagged high-reach posts: _ (`alerts`)

## Challenges
_Rate limits, sparse graph, noisy short comments, language ID on short text, etc. Write what actually happened._
