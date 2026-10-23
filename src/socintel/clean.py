"""Text cleaning + language detection.

Emojis: we do NOT just delete them, they carry sentiment. They get turned into text
(":crying_face:" -> "crying face") so the transformer can still use them.
"""

import re

import emoji
import pandas as pd
import py3langid as langid

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"(?<!\w)(u/|r/|@)\w+")
HTML_RE = re.compile(r"&(amp|lt|gt|quot|#x200B);")
MD_RE = re.compile(r"[*~`>#]+")
EMOJI_NAME_RE = re.compile(r"§([^§\s]+)§")
SPACE_RE = re.compile(r"\s+")
REMOVED = {"[deleted]", "[removed]"}


def clean_text(text):
    if not isinstance(text, str):
        return ""
    t = HTML_RE.sub(" ", text)
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" @user ", t)
    t = emoji.demojize(t, delimiters=("§", "§"))
    t = EMOJI_NAME_RE.sub(lambda m: " " + m.group(1).replace("_", " ") + " ", t)
    t = MD_RE.sub(" ", t)
    t = SPACE_RE.sub(" ", t).strip()
    return t


def detect_language(text, min_chars=15):
    """py3langid is fast but unreliable on very short strings, so call those 'und'."""
    if not text or len(text) < min_chars:
        return "und"
    lang, _ = langid.classify(text)
    return lang


def clean_table(df):
    df = df.copy()
    df = df[~df["text"].isin(REMOVED)]
    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.len() >= 3]
    df["language"] = df["clean_text"].map(detect_language)
    # exact duplicate comments (bots / spam) add noise to topics, keep the first
    df = df.drop_duplicates(subset=["clean_text", "author"]).reset_index(drop=True)
    return df


def language_report(df):
    return (df["language"].value_counts(normalize=True).mul(100).round(1)
            .rename("percent").to_frame())
