# Results

All numbers come from `data/processed/metrics.json` or the CSVs next to it, from one run on 25 Oct 2026.
Items marked **TODO** have not been measured yet and are left open on purpose. Numbers labelled "my labels" come from my own hand check (`scripts/make_review_sheets.py`, `scripts/score_review.py`).

## Data collected
- YouTube: 299 videos from 12 search queries (English, Hindi, Spanish and German queries), top-level comments plus
  up to 5 inline replies per thread. Reddit was not collected (needs Reddit approval, see README).
- After cleaning and de-duplication: **11,233 rows**.
- Language mix (detected): en 66.1%, und 13.5% (too short or ambiguous), pt 4.7%, de 3.8%, mr 2.6%, hi 2.4%,
  id 2.1%, es 1.2%, ko 0.7%, ru 0.6%.
- Time spread: comments run from 2011 to 24 Oct 2026, but 16.9% are from before 2026 and **4,332 (38.6%) are from the
  last week of collection** (week of 19 Oct). YouTube search returns recent videos, and fresh videos collect most of
  their comments early. Weekly counts before that week are 241 to 572.
- Public benchmarks used: TweetEval, Multilingual Amazon Reviews (SetFit port), LIAR, SNAP Reddit hyperlinks.

## Sentiment
Model: `cardiffnlp/twitter-xlm-roberta-base-sentiment`, run on CPU (Intel i3) at 18.2 docs/s.

- Label share on our comments: negative 48.5%, neutral 29.1%, positive 22.4%.
- TweetEval (English tweets, 2,000 test samples): accuracy 0.661, macro-F1 0.660.
- Multilingual Amazon Reviews, 1,000 test reviews per language, 1-2 stars = negative, 3 = neutral, 4-5 = positive:

| language | accuracy | macro-F1 |
|---|---|---|
| en | 0.688 | 0.584 |
| de | 0.653 | 0.528 |
| es | 0.669 | 0.576 |
| fr | 0.686 | 0.553 |
| ja | 0.665 | 0.562 |
| zh | 0.629 | 0.525 |

- Reading these: the model was trained on tweets, not product reviews, and "3 stars = neutral" is a rough label, so
  macro-F1 sits well below accuracy. English is best, Chinese worst, the rest are within a few points of each other.
- **Check on our own data:** I labelled 100 random comments from the collected set by hand (one annotator, labels
  recorded before looking at the model's). Model vs my labels: **accuracy 0.73, macro-F1 0.695** (n = 100, so
  roughly +/- 9 points at 95% confidence). Per class recall: negative 43/53, neutral 14/23, positive 16/24.
  Confusion matrix (rows = my label, columns = model, order negative / neutral / positive):
  [[43, 8, 2], [5, 14, 4], [4, 4, 16]]. The most common error is calling a neutral or mildly negative comment
  negative. Label shares were close: negative 53% (mine) vs 52% (model), neutral 23% vs 26%, positive 24% vs 22%,
  so on this sample the 48.5% negative share over all comments does not look inflated. It is a small sample from one
  annotator, including languages the annotator reads only partly.
- Over time (Kibana, weekly share over the last year): negative stays around 40-60% every week with no sharp trend.

## Topics
- BERTopic with `paraphrase-multilingual-MiniLM-L12-v2` embeddings: **95 topics**, **37.5%** of comments are outliers
  (topic -1, mostly very short comments).
- Largest topics: Ukraine/Russia (644 comments), vaccines (530), climate (519), a Hinglish chat topic (361),
  AI and jobs (330), YouTube/video chatter (233), bitcoin (160), inflation (127), BJP/Congress (105),
  religion (104), fact-check videos (95), conspiracy theories (81). Topic names come from c-TF-IDF words.
- Some topics are not real subjects: topic 8 (`spcx9k_spx50c`) looks like spam or bot comments, and a few topics
  group by language rather than meaning. **TODO:** read 10 comments from topic 8 and say what they are.
- Topic coherence was not computed. Rising and falling topics: `topics_over_time.csv` exists, **TODO** review.

## Graph
Edge A -> B means A replied to B. Weighted PageRank and Louvain communities (NetworkX).

| graph | nodes | edges | components | largest component | modularity |
|---|---|---|---|---|---|
| YouTube, everyone incl. commenter -> video owner | 9,739 | 10,050 | 121 | 1,696 nodes (17.4%) | 0.98 |
| YouTube, comment-to-comment replies only | 1,492 | 1,085 | 434 | 52 nodes (3.5%) | 0.99 |
| SNAP subreddit hyperlinks | 54,075 | 234,792 | 758 | 52,468 nodes (97.0%) | 0.47 |

- The full YouTube graph is dominated by video owners by construction (every commenter points at the owner), so
  influencers are ranked on the replies-only graph.
- The YouTube graphs are very fragmented: most reply threads are a single exchange. The high modularity (0.98)
  is a side effect of many tiny disconnected clusters, not a sign of strong community structure.
  PageRank is flat: the top author has 0.004, so there are no dominant accounts in this sample.
- SNAP at scale: top subreddits by PageRank are iama, askreddit, pics, funny, videos, the_donald, gaming,
  todayilearned, music, worldnews (`snap_top_subreddits.csv`). Modularity 0.47 with real community structure.
- Neo4j: the full graph is loaded by the graph stage (`docker compose --profile graph up -d neo4j`) and was checked in
  Neo4j Browser; the top-PageRank node is a video owner with 150+ users replying to it (star-shaped cluster).

## Misinformation (LIAR, binary: pants-fire/false/barely-true = 1)
| model | split | accuracy | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| TF-IDF + LogReg | validation | 0.622 | 0.607 | 0.606 | 0.606 | 0.668 |
| TF-IDF + LogReg | test | 0.632 | 0.579 | 0.579 | 0.579 | 0.671 |
| DistilBERT (optional) | | not run | | | | |

- Strongest words for "misleading": wisconsin, obama, rep, illegals, obamacare, scott walker, muslim.
  For "truthful": than, countries, since, percent, million, debt, average. The model has partly learned US political
  topics and speakers and numeric phrasing, not truthfulness.
- Domain shift: LIAR is fact-checked political statements; our data is YouTube comments. Applied to our comments the
  score means "sounds like a statement fact-checkers rated misleading".
- Error analysis on our data: I read all 21 flagged posts (misinfo score >= 0.7, top-50 PageRank authors) and
  judged each myself: **5 misleading, 7 opinion, 9 other** (24% of the flags look like checkable false or misleading
  claims; judged by reading, not by fact-checking against sources). Examples of the false positives: "Bow down to
  the aggressor now...", "Stop saying that. Russia is the bad guy...", "shes not germanys president" are opinions or
  short reactions. The score picks up how political claims sound, not whether they are true, so it is a ranking
  aid for a human reviewer, not a detector.
  `misinfo_errors.csv` has the LIAR test errors to describe as well.

## Elasticsearch
- 11,233 documents indexed, 0 errors, 2,326 docs/s, sample text query ("vaccine", 261 hits) in 72 ms.
- Two Kibana dashboards (exported in `kibana/dashboards.ndjson`): Trends (sentiment share per week, topic volume,
  language mix) and Influencers and Alerts (PageRank table, flagged content).

## Alerts
- Negative-sentiment spikes: **9** topic-weeks out of 76 that had at least 15 posts. Rule: a topic-week whose
  negative share is at least 2 standard errors above that topic's share in its other weeks (binomial z-test).
  Six of the top eight are the week of 19 Oct, the same week that holds 39% of the data, so they most likely reflect
  fresh videos and the model's negative lean, not news events.
- Flagged high-reach content: **21** posts with misinfo score >= 0.7 by authors in the top 50 by PageRank
  (ties at the cut-off kept). These are posts to review, not judgements about accounts.

## Challenges
- Reddit API access needs approval and its policy limits ML use, so YouTube is the live source.
- YouTube quota: `search.list` costs 100 units; the config estimates about 1,700 of the 10,000 daily units for a full run. The collector had to
  skip results with no video id, skip videos with comments disabled, and resume.
- Language ID is poor on short and romanized text (romanized Telugu was labelled sn/la/jv before constraining the
  label set); about 13.5% of rows stay `und`.
- Topic names filled with Hinglish and other-language filler words until I added custom stop words.
- A standard-deviation z-score cannot fire when a topic has only 2-4 judgeable weeks (maximum is (k-1)/sqrt(k)); the
  first alert rule found zero spikes for that reason, so I switched to a binomial test.
- Collected comments are skewed to the most recent week, which limits what the time series can show.
- 8 GB laptop: Docker, Elasticsearch, Kibana and Neo4j do not fit together, so Neo4j is behind a Compose profile.