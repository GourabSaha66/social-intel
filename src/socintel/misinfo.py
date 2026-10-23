"""Fake-news / misinformation classifier trained on LIAR.

LIAR has 6 truthfulness labels. For a first pass I collapse them into a binary target:
  false-ish: pants-fire, false, barely-true     -> 1 (misleading)
  true-ish : half-true, mostly-true, true       -> 0
That choice is debatable (half-true is arguably misleading too), so it's a config knob
and the README discusses it.

IMPORTANT limitation: LIAR is short political fact-check statements, not Reddit/YouTube
comments. The classifier output on our data is a "looks like known misleading claims"
signal, not ground truth. We say this in the results.
"""

import numpy as np
import pandas as pd

LIAR_LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
MISLEADING = {"pants-fire", "false", "barely-true"}


LIAR_TSV_COLS = ["id", "label_name", "text", "subject", "speaker", "job", "state", "party",
                 "c_barely", "c_false", "c_half", "c_mostly", "c_pants", "context"]


def _finish(d, binary):
    d = d.copy()
    d["y"] = d["label_name"].isin(MISLEADING).astype(int) if binary else d["label_name"].map(
        {n: i for i, n in enumerate(LIAR_LABELS)})
    return d[["text", "y", "label_name"]]


def load_liar(binary=True, local_dir="data/liar"):
    """returns {'train','val','test'} frames with columns text, y, label_name.

    Preferred: the original TSVs (train.tsv, valid.tsv, test.tsv) from the LIAR release
    dropped into data/liar/. Fallback: the Hugging Face copy, if the installed `datasets`
    version can still load it.
    """
    import os
    files = {"train": "train.tsv", "val": "valid.tsv", "test": "test.tsv"}
    if all(os.path.exists(os.path.join(local_dir, f)) for f in files.values()):
        return {k: _finish(pd.read_csv(os.path.join(local_dir, f), sep="\t", header=None,
                                       names=LIAR_TSV_COLS), binary)
                for k, f in files.items()}
    from datasets import load_dataset
    ds = load_dataset("ucsbnlp/liar")
    frames = {}
    for split_hf, name in [("train", "train"), ("validation", "val"), ("test", "test")]:
        d = ds[split_hf].to_pandas()
        d["label_name"] = d["label"].map(dict(enumerate(LIAR_LABELS)))
        frames[name] = _finish(d.rename(columns={"statement": "text"}), binary)
    return frames


def baseline_model(max_features=50000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features,
                                  sublinear_tf=True, lowercase=True)),
        ("clf", LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced")),
    ])


def evaluate(model, X, y):
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 confusion_matrix, roc_auc_score)
    pred = model.predict(X)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    res = {"accuracy": accuracy_score(y, pred), "precision": p, "recall": r, "f1": f,
           "confusion_matrix": confusion_matrix(y, pred).tolist()}
    if hasattr(model, "predict_proba"):
        res["roc_auc"] = roc_auc_score(y, model.predict_proba(X)[:, 1])
    return res


def top_features(model, k=15):
    """words the baseline associates most with each class - useful for the error analysis."""
    vec = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    return {"truthful": names[order[:k]].tolist(), "misleading": names[order[-k:]][::-1].tolist()}


def error_analysis(model, df, n=10, seed=42):
    """sample of wrong predictions, with the model's probability, for the write-up."""
    d = df.copy()
    d["prob"] = model.predict_proba(d["text"])[:, 1]
    d["pred"] = (d["prob"] >= 0.5).astype(int)
    wrong = d[d["pred"] != d["y"]]
    return wrong.sample(min(n, len(wrong)), random_state=seed)[["text", "y", "pred", "prob"]]


def fit_and_report(max_features=50000):
    data = load_liar()
    m = baseline_model(max_features)
    m.fit(data["train"]["text"], data["train"]["y"])
    return m, {"val": evaluate(m, data["val"]["text"], data["val"]["y"]),
               "test": evaluate(m, data["test"]["text"], data["test"]["y"])}, data


def score_posts(df, model):
    df = df.copy()
    df["misinfo_prob"] = model.predict_proba(df["clean_text"])[:, 1]
    return df


def finetune_transformer(train_df, val_df, out_dir="models/misinfo-distilbert", epochs=2):
    """optional stronger model. Needs a GPU to be pleasant (~10 min on a free Colab T4)."""
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                              TrainingArguments)
    name = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)

    def enc(batch):
        return tok(batch["text"], truncation=True, max_length=96, padding="max_length")

    tr = Dataset.from_pandas(train_df[["text", "y"]].rename(columns={"y": "label"})).map(enc, batched=True)
    va = Dataset.from_pandas(val_df[["text", "y"]].rename(columns={"y": "label"})).map(enc, batched=True)
    args = TrainingArguments(output_dir=out_dir, num_train_epochs=epochs,
                             per_device_train_batch_size=32, learning_rate=3e-5,
                             eval_strategy="epoch", save_strategy="no", report_to=[],
                             fp16=torch.cuda.is_available())
    Trainer(model=model, args=args, train_dataset=tr, eval_dataset=va).train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir
