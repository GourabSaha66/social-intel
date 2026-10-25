"""Topic discovery with BERTopic + topics-over-time.

We use a multilingual sentence-transformer so topics aren't split by language.
Docs that BERTopic can't place get topic -1 (outliers); we keep them labelled as such
instead of forcing them into a topic.
"""

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Filler words of the other languages in our data. They only affect how topics are *named*;
# without them a whole topic gets called "hai_ka_bhi_ko".
EXTRA_STOP = [
    # Hinglish
    "hai", "hain", "ka", "ke", "ki", "ko", "se", "mein", "bhi", "aur", "nahi", "nhi", "tha", "thi",
    "kya", "ye", "yeh", "ho", "hoga", "par", "ne", "wo", "woh", "bahut", "kar", "hi", "toh",
    # romanized Telugu
    "lo", "ni", "kuda", "ra", "ayithe", "vundhi", "chesindhi", "ante", "ela",
    # German
    "der", "das", "ist", "nicht", "ein", "und", "ich", "sie", "zu", "den", "mit", "auf", "nur", "auch", "wir",
    # Spanish / Portuguese
    "que", "de", "la", "el", "en", "los", "las", "para", "por", "con", "una", "un", "del", "mas",
    "não", "nao", "uma", "um", "os", "da", "em", "com", "mais",
]
STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) | set(EXTRA_STOP))


def fit_topics(docs, embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
               min_topic_size=15, seed=42):
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    # fixed seed so reruns give the same topics (UMAP is stochastic)
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                      random_state=seed)
    emb = SentenceTransformer(embedding_model)
    # only affects how topics are *named* (c-TF-IDF); stop-words out so names aren't "the_to_and"
    vec = CountVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), min_df=2)
    model = BERTopic(embedding_model=emb, umap_model=umap_model, vectorizer_model=vec,
                     min_topic_size=min_topic_size, calculate_probabilities=False, verbose=True)
    topics, _ = model.fit_transform(docs)
    return model, topics


def topic_labels(model):
    """topic id -> short readable label built from the top words."""
    info = model.get_topic_info()
    return {int(r.Topic): ("outlier" if r.Topic == -1 else "_".join(r.Name.split("_")[1:4]))
            for r in info.itertuples()}


def add_topics(df, model, topics):
    df = df.copy()
    df["topic_id"] = topics
    labels = topic_labels(model)
    df["topic_label"] = df["topic_id"].map(labels)
    return df


def topics_over_time(model, docs, timestamps, nr_bins=20):
    """thin wrapper so the runner doesn't need to know BERTopic internals."""
    ts = pd.to_datetime(timestamps, unit="s").tolist()
    return model.topics_over_time(docs, ts, nr_bins=nr_bins)


# ----- pure-pandas version: works from the labelled table, used for dashboards/alerts -----

def topic_volume(df, freq="D"):
    d = df[df["topic_id"] != -1].copy()
    d["bucket"] = pd.to_datetime(d["created_utc"], unit="s").dt.to_period(freq).dt.start_time
    return d.groupby(["bucket", "topic_id", "topic_label"]).size().rename("n").reset_index()


def rising_topics(vol, recent_buckets=3, top=10):
    """compare the last N buckets to the N before them; biggest relative growth first."""
    pivot = vol.pivot_table(index="topic_id", columns="bucket", values="n", fill_value=0)
    if pivot.shape[1] < 2 * recent_buckets:
        return pd.DataFrame(columns=["topic_id", "recent", "previous", "growth"])
    recent = pivot.iloc[:, -recent_buckets:].sum(axis=1)
    prev = pivot.iloc[:, -2 * recent_buckets:-recent_buckets].sum(axis=1)
    out = pd.DataFrame({"recent": recent, "previous": prev})
    out["growth"] = (out["recent"] - out["previous"]) / (out["previous"] + 1)   # +1 smoothing
    return out.sort_values("growth", ascending=False).head(top).reset_index()