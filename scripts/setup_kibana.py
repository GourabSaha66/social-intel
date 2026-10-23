"""Creates the Kibana data views we need (idempotent-ish: 409 = already exists, fine).
Dashboards themselves are built in the UI, see docs/kibana.md, then exported to
kibana/dashboards.ndjson so they can be re-imported on a fresh stack.

    python scripts/setup_kibana.py
    python scripts/setup_kibana.py --import kibana/dashboards.ndjson
"""

import argparse
import os
import sys

import requests

KIBANA = os.environ.get("KIBANA_URL", "http://localhost:5601")
HEADERS = {"kbn-xsrf": "true"}

VIEWS = [
    {"id": "socintel-posts", "title": "socintel-posts", "name": "Posts", "timeFieldName": "created"},
]


def create_views():
    for v in VIEWS:
        r = requests.post(f"{KIBANA}/api/data_views/data_view", headers=HEADERS,
                          json={"data_view": v, "override": True}, timeout=30)
        print(v["title"], r.status_code)
        if r.status_code not in (200, 409):
            print(r.text)
            sys.exit(1)


def import_objects(path):
    with open(path, "rb") as f:
        r = requests.post(f"{KIBANA}/api/saved_objects/_import?overwrite=true", headers=HEADERS,
                          files={"file": f}, timeout=60)
    print(r.status_code, r.text[:300])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--import", dest="imp")
    a = ap.parse_args()
    create_views()
    if a.imp:
        import_objects(a.imp)
