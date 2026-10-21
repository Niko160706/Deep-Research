"""Unit tests for the text utilities (keywords + extractive ranking)."""
from __future__ import annotations

from drx.models import Document
from drx.textutil import (
    has_all_tokens,
    keyword_query,
    primary_entity,
    query_aliases,
    rank_sentences,
    sanitize_query,
    split_aliases,
    split_sentences,
    strip_boolean,
    tokenize,
    top_keywords,
)


def test_tokenize_filters_stopwords_and_short_words() -> None:
    tokens = tokenize("The Roman Empire fell because of many reasons")
    assert "the" not in tokens
    assert "of" not in tokens
    assert "roman" in tokens
    assert "empire" in tokens


def test_split_sentences_filters_fragments() -> None:
    long = "A sentence that is long enough to be kept and meaningful for testing."
    text = f"Short. {long} {long}"
    sentences = split_sentences(text)
    assert all(40 <= len(s) <= 400 for s in sentences)
    assert all(s == long for s in sentences)


def test_top_keywords_counts_by_frequency() -> None:
    docs = [Document(url="http://a", text="roman empire roman empire roman empire fall")]
    keywords = top_keywords(docs, n=3)
    assert keywords[0] == ("roman", 3)
    assert ("fall", 1) in keywords


def test_rank_sentences_prefers_query_relevant() -> None:
    docs = [
        Document(url="http://a", text="The roman empire was huge and lasted for centuries."),
        Document(url="http://b", text="Cooking pasta requires boiling water and salt."),
    ]
    ranked = rank_sentences(docs, query="roman empire", n=5)
    assert ranked
    assert "roman" in ranked[0]["text"].lower()


def test_keyword_query_strips_stopwords() -> None:
    assert keyword_query("why did the Roman Empire fall") == "Roman Empire fall"


def test_strip_boolean_reduces_anchored_syntax() -> None:
    anchored = '"Nur Royul Amin" AND ("Surabaya" OR "Informatika")'
    assert strip_boolean(anchored) == "Nur Royul Amin Surabaya Informatika"


def test_strip_boolean_keeps_plain_quoted_queries() -> None:
    assert strip_boolean('"Nur Royul Amin"') == '"Nur Royul Amin"'
    assert strip_boolean("roman empire fall") == "roman empire fall"


def test_has_all_tokens_folds_spelling_variants() -> None:
    tokens = ["nur", "royul", "amin"]
    assert has_all_tokens("Nur Ro'yul Amin menulis tentang Xinu.", tokens)
    assert has_all_tokens("Nur-Royul Amin, peneliti.", tokens)
    assert has_all_tokens("NUR ROYUL AMIN", tokens)
    assert not has_all_tokens("Nurul Amin adalah aktivis.", tokens)
    assert not has_all_tokens("Raka Amin Nur Soleh.", tokens)


# ---- Query sanitization: "Bayu Skak (Bayu Eko Moektito)" ----

def test_sanitize_query_turns_brackets_into_or_aliases() -> None:
    assert (
        sanitize_query("Bayu Skak (Bayu Eko Moektito)")
        == '"Bayu Skak" OR "Bayu Eko Moektito"'
    )


def test_sanitize_query_leaves_plain_and_boolean_queries_alone() -> None:
    assert sanitize_query("Ainun Najib") == "Ainun Najib"
    anchored = '"Ainun Najib" AND ("KawalCOVID19" OR "Data Scientist")'
    assert sanitize_query(anchored) == anchored


def test_sanitize_query_uses_brackets_only_when_present() -> None:
    assert sanitize_query("Siti (Rahayu) Aminah") == '"Siti Aminah" OR "Rahayu"'


def test_query_aliases_and_primary_entity() -> None:
    assert query_aliases("Bayu Skak (Bayu Eko Moektito)") == [
        "Bayu Skak",
        "Bayu Eko Moektito",
    ]
    assert primary_entity("Bayu Skak (Bayu Eko Moektito)") == "Bayu Skak"
    assert split_aliases('"Bayu Skak" OR "Bayu Eko Moektito"') == [
        "Bayu Skak",
        "Bayu Eko Moektito",
    ]
    assert split_aliases("Ainun Najib") == ["Ainun Najib"]


def test_split_aliases_keeps_boolean_subqueries_intact() -> None:
    anchored = '"Bayu Skak" AND ("Sleman" OR "Kesehatan")'
    assert split_aliases(anchored) == [anchored]


def test_css_layout_noise_is_stopworded() -> None:
    """CSS/HTML layout words leaked by naive extractors are never keywords."""
    noise = "normal type style width height align non display border padding"
    text = f"{noise} penelitian kesehatan masyarakat yogyakarta"
    assert tokenize(text) == ["penelitian", "kesehatan", "masyarakat", "yogyakarta"]

    doc = Document(url="http://x", text=(text + " ") * 5)
    keywords = dict(top_keywords([doc], n=10))
    for word in noise.split():
        assert word not in keywords
    assert "penelitian" in keywords
