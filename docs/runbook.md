# Runbook: from clone to final submission

Run everything from the repo root, venv active, with `export PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH="src"`).

1. **Tests:** `python -m pytest -q` -> 35 passed.
2. **Keys:** `cp .env.example .env`. Fill `YOUTUBE_API_KEY` and put any random string in `AUTHOR_SALT`. Leave the Reddit lines empty.
3. **Services:** `docker compose up -d`, wait for Elasticsearch to be healthy (`docker compose ps`).
4. **Datasets:** put LIAR `train.tsv`, `valid.tsv`, `test.tsv` in `data/liar/`; put SNAP `soc-redditHyperlinks-title.tsv` in `data/snap/`.
5. **Benchmarks (no live data needed):** `python -m socintel.run_pipeline benchmarks`
   Gives TweetEval and per-language Amazon accuracy, plus the SNAP graph PageRank/communities.
6. **Small YouTube test:** set `videos_per_query: 3` in `config.yaml`, run `python -m socintel.collect_youtube`, check `data/raw/`.
7. **Smoke-run:** `python -m socintel.run_pipeline clean sentiment` on that small data.
8. **Real collection:** restore `videos_per_query: 20`, collect (quota is 10,000 units/day; each search is 100). Spread collection over 2 to 3 days so the time series has some length, and aim for a few thousand comments.
9. **Full pipeline:** `python -m socintel.run_pipeline all` (use Colab GPU for sentiment/topics if your laptop is slow).
10. **Dashboards:** `python scripts/setup_kibana.py`, then build per `docs/kibana.md`. Export to `kibana/dashboards.ndjson`.
11. **Results:** copy numbers from `data/processed/metrics.json` into `docs/results.md`; pick 3 to 5 real errors from `misinfo_errors.csv`.
12. **Deliverables:** screen-record the demo, make the 10-minute deck from `docs/presentation_outline.md`, write your own reflection, push to GitHub, and put everything in one Drive folder (do not put raw collected data in it).
