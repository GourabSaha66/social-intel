import json
import os
import time
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def save_raw(obj, folder, name):
    """dump one raw API response as json, named with a timestamp so reruns don't overwrite."""
    Path(folder).mkdir(parents=True, exist_ok=True)
    fname = f"{name}_{int(time.time() * 1000)}.json"
    with open(os.path.join(folder, fname), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    return fname


def iter_raw(folder, prefix=""):
    for p in sorted(Path(folder).glob(f"{prefix}*.json")):
        with open(p, encoding="utf-8") as f:
            yield p.name, json.load(f)


def read_jsonl(path):
    return pd.read_json(path, lines=True)


def write_jsonl(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)
