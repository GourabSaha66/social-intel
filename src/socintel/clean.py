"""Text cleaning + language detection.

Emojis: we do NOT just delete them, they carry sentiment. They get turned into text
(":crying_face:" -> "crying face") so the transformer can still use them.
"""

import re

from pathlib import Path

import emoji
import pandas as pd
import py3langid as langid
from py3langid.langid import MODEL_DIR, MODEL_FILE, LanguageIdentifier

# Open-ended language ID on short, noisy comments picks absurd labels (Shona, Latin, Frisian...)
# for romanized Telugu/Hindi, so we restrict it to languages we actually expect, and anything
# the model isn't at least MIN_CONF sure about becomes "und".
ALLOWED_LANGS = ["en", "hi", "te", "ta", "bn", "mr", "gu", "kn", "ml", "pa", "ur", "es", "pt", "fr",
                 "de", "it", "ar", "ru", "tr", "id", "ja", "ko", "zh", "nl", "pl", "vi", "th", "fa"]
MIN_CONF = 0.5
_IDENT = None


def _identifier():
    global _IDENT
    if _IDENT is None:
        path = Path(MODEL_FILE)
        path = path if path.is_absolute() else Path(MODEL_DIR) / path
        _IDENT = LanguageIdentifier.from_modelpath(path, norm_probs=True)
        _IDENT.set_languages(ALLOWED_LANGS)
    return _IDENT

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


def topic_text(text):
    """text used for topic modelling: same cleaning but emojis are dropped, not spelled out.
    (spelled-out emojis help sentiment, but they flood topics with 'face', 'tears', 'heart'.)"""
    if not isinstance(text, str):
        return ""
    t = clean_text(emoji.replace_emoji(text, replace=" "))
    return SPACE_RE.sub(" ", t.replace("@user", " ")).strip()      # drop the mention placeholder too


def detect_language(text, min_chars=15):
    """py3langid is fast but unreliable on very short strings, so call those 'und'.
    Known weak spot: romanized Telugu/Hindi ("nenu vastanu") is often labelled en/id or und."""
    if not text or len(text) < min_chars:
        return "und"
    lang, conf = _identifier().classify(text)
    return lang if conf >= MIN_CONF else "und"


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