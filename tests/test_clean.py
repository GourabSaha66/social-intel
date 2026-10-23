import pandas as pd

from socintel.clean import clean_table, clean_text, detect_language


def test_urls_and_mentions_removed():
    out = clean_text("look at this https://example.com/x?a=1 cc u/someone and @other")
    assert "http" not in out
    assert "someone" not in out and "other" not in out
    assert "@user" in out


def test_emoji_kept_as_words_not_dropped():
    out = clean_text("so sad 😭")
    assert "crying" in out
    assert "😭" not in out


def test_underscores_in_normal_words_survive():
    assert clean_text("use snake_case_names") == "use snake_case_names"


def test_html_entities_and_markdown():
    assert clean_text("**bold** &amp; &gt; quote") == "bold quote"


def test_non_string_input():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_language_detection_basic():
    assert detect_language("This is clearly an english sentence about the weather") == "en"
    assert detect_language("Esto es una frase en español sobre el tiempo de hoy") == "es"


def test_short_text_is_undetermined():
    assert detect_language("hola") == "und"


def test_clean_table_drops_removed_and_dupes():
    df = pd.DataFrame({
        "text": ["[deleted]", "[removed]", "Buy cheap watches now!!!", "Buy cheap watches now!!!",
                 "A real comment with enough characters to detect language"],
        "author": ["a", "b", "spam", "spam", "c"],
    })
    out = clean_table(df)
    assert len(out) == 2
    assert set(out["author"]) == {"spam", "c"}
    assert {"clean_text", "language"} <= set(out.columns)
