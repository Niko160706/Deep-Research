"""Text utilities for keyword extraction and extractive summarization.

Pure Python, 0 tokens. Used by the entity extractor (keywords) and the context
packer (sentence ranking) to build a narrative report from raw parquet text —
without ever sending long text through the LLM.
"""
from __future__ import annotations

import re
from collections import Counter

from drx.models import Document

# A pragmatic stopword set (English + Indonesian) for keyword scoring.
STOPWORDS: frozenset[str] = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "he", "her", "his", "i", "in", "is", "it", "its", "of", "on", "or", "she", "that", "the", "they", "this", "to", "was", "were", "with", "you", "your", "we", "our", "their", "why", "what", "when", "where", "which", "who", "how", "did", "does", "do", "was", "were", "will", "would", "can", "could", "should", "may", "might", "must", "not", "no", "nor", "but", "than", "then", "so", "if", "because", "the", "this", "that", "these", "those", "there", "here", "dan", "yang", "di", "ke", "dari", "pada", "untuk", "dengan", "adalah", "ini", "itu", "atau", "juga", "akan", "telah", "tidak", "bisa", "bisa", "menjadi", "dalam", "oleh", "karena", "sebagai", "lebih", "paling", "sangat", "sebuah", "suatu", "para", "atas", "bawah", "antara", "serta", "namun", "tetapi", "ketika", "saat", "hingga", "ada", "tidak", "namun", "hanya", "siapa", "apa", "mengapa", "bagaimana", "kapan", "mana", "carikan", "profil", "lengkap", "nama", "lengkapnya", "isbn", "issn", "doi", "http", "https", "www", "retrieved", "cite", "references", "external", "links", "archive", "archived", "pdf", "page", "pages", "vol", "editor", "editors", "press", "university",
    # CSS/HTML layout noise: naive extractors leak these into key phrases
    # ("normal", "type", "style", "width", ...) — never report keywords.
    "normal", "type", "style", "width", "height", "align", "non", "display", "border", "padding",
]
)

_WORD_RE = re.compile(r"[a-z0-9]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_APOSTROPHE_RE = re.compile(r"['’`´]")
_WORD_SEP_RE = re.compile(r"[-_/]+")

#: Canvas cards use a deterministic compact label: 1-3 words, max 20 chars.
SHORT_LABEL_MAX_CHARS = 20
SHORT_LABEL_MAX_WORDS = 3


def compact_short_label(
    value: str,
    *,
    fallback: str = "",
    max_words: int = SHORT_LABEL_MAX_WORDS,
    max_chars: int = SHORT_LABEL_MAX_CHARS,
) -> str:
    """Return a display-safe 1-3 word label capped at 20 characters.

    The LLM is asked for ``short_label``, but this helper is also the
    deterministic fallback for imported/legacy nodes and ``--no-llm`` runs.
    Long candidates are cut at ``max_chars - 3`` and suffixed with ``...`` so
    the fixed-width Vis.js card can never overflow.
    """
    text = " ".join(str(value or "").split()).strip(
        " \t\r\n\"'“”‘’.,;:!?()[]{}"
    )
    if not text:
        text = " ".join(str(fallback or "").split()).strip(
            " \t\r\n\"'“”‘’.,;:!?()[]{}"
        )
    if not text:
        return ""
    words = text.split()[: max(1, int(max_words))]
    candidate = " ".join(words)
    limit = max(4, int(max_chars))
    if len(candidate) <= limit:
        return candidate
    # Prefer a clean word boundary ("Dinas Kesehatan..." rather than
    # "Dinas Kesehatan S...") while still keeping the 20-char hard cap.
    clipped = candidate[:limit]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    if len(clipped) + 3 > limit:
        clipped = candidate[: limit - 3]
    return clipped.rstrip() + "..."


def _fold_variants(text: str) -> str:
    """Lowercase + fold apostrophes and hyphen/underscore separators.

    ``Ro'yul`` -> ``royul``; ``Nur-Royul`` -> ``nur royul``.
    """
    folded = _APOSTROPHE_RE.sub("", text.lower())
    return _WORD_SEP_RE.sub(" ", folded)


def tokenize(text: str) -> list[str]:
    """Lowercase content words (len > 2, non-stopword, non-numeric).

    Apostrophes are folded first, so ``Ro'yul`` yields ``royul`` — keeping the
    real spelling variant as one keyword instead of fragmenting it into ``yul``.
    """
    words = _WORD_RE.findall(_APOSTROPHE_RE.sub("", text.lower()))
    return [w for w in words if len(w) > 2 and w not in STOPWORDS and not w.isdigit()]


def split_sentences(text: str) -> list[str]:
    """Split into substantial sentences (40–400 chars) for extractive use."""
    flat = re.sub(r"\s+", " ", text)
    parts = _SENT_SPLIT_RE.split(flat)
    return [p.strip() for p in parts if 40 <= len(p.strip()) <= 400]


def looks_like_name(query: str) -> bool:
    """Heuristic: does the query look like a person name (2–5 title-cased words)?

    Used to force exact-match search and strict name filtering so that a query
    like ``Nur Royul Amin`` does not drift onto ``Nurul Amin``.
    """
    words = [w for w in query.split() if w]
    if not 2 <= len(words) <= 5:
        return False
    for w in words:
        stripped = w.strip(".,!?;:")
        if not stripped:
            return False
        if stripped.lower() in STOPWORDS:
            return False
        if not stripped[0].isupper():
            return False
    return True


def name_tokens(query: str) -> list[str]:
    """Return the significant tokens of a name query (lowercased)."""
    return [
        w.lower().strip(".,!?;:")
        for w in query.split()
        if len(w) > 2 and w.lower() not in STOPWORDS
    ]


_PAREN_RE = re.compile(r"[（(]\s*([^()（）]*?)\s*[)）]")
_FRAGMENT_TRIM_RE = re.compile(r"^[\s\"'“”‘’,;:.!?]+|[\s\"'“”‘’,;:.!?]+$")


def _clean_fragment(text: str) -> str:
    """Trim quotes/punctuation/whitespace from a candidate alias."""
    return re.sub(r"\s+", " ", _FRAGMENT_TRIM_RE.sub("", text or "")).strip()


def query_aliases(query: str) -> list[str]:
    """Split a query into clean name aliases.

    ``"Bayu Skak (Bayu Eko Moektito)"`` -> ``["Bayu Skak", "Bayu Eko Moektito"]``.
    Boolean queries (anchored 70/30 sub-queries) are never touched. The first
    alias is the primary entity used for validation; the rest are alternates.
    """
    raw = (query or "").strip()
    if not raw or _BOOLEAN_RE.search(raw):
        return [raw] if raw else []
    aliases: list[str] = []
    outside = _clean_fragment(_PAREN_RE.sub(" ", raw))
    if outside:
        aliases.append(outside)
    for match in _PAREN_RE.finditer(raw):
        inner = _clean_fragment(match.group(1))
        if inner and inner.lower() not in {a.lower() for a in aliases}:
            aliases.append(inner)
    return aliases or [_clean_fragment(raw)]


def sanitize_query(query: str) -> str:
    """Build a clean search query from a parenthesised input.

    ``"Bayu Skak (Bayu Eko Moektito)"`` becomes
    ``'"Bayu Skak" OR "Bayu Eko Moektito"'`` so every alias is searched instead
    of the literal brackets (which backends treat as noise). Queries without
    brackets — including anchored boolean sub-queries — are returned unchanged.
    """
    raw = (query or "").strip()
    if not raw or _BOOLEAN_RE.search(raw) or "(" not in raw and "（" not in raw:
        return raw
    aliases = query_aliases(raw)
    if len(aliases) < 2:
        return aliases[0] if aliases else raw
    return " OR ".join(f'"{a}"' for a in aliases)


def primary_entity(query: str) -> str:
    """The entity name used for validation (first alias of the query)."""
    aliases = query_aliases(query)
    return aliases[0] if aliases else (query or "").strip()


def split_aliases(text: str) -> list[str]:
    """Split any alias form into clean names.

    Handles the sanitized search query (``"Bayu Skak" OR "Bayu Eko Moektito"``),
    the raw parenthesised input (``Bayu Skak (Bayu Eko Moektito)``) and plain
    names (``Ainun Najib`` -> ``["Ainun Najib"]``). Anchored boolean queries
    (``"A" AND ("x" OR "y")``) are returned untouched — their operators are not
    alias separators.
    """
    raw = (text or "").strip()
    if not raw:
        return []
    clause = _BOOLEAN_OR_RE.search(raw)
    if clause and not _BOOLEAN_HARD_RE.search(raw) and "(" not in raw:
        parts = [_clean_fragment(p) for p in _BOOLEAN_OR_RE.split(raw)]
        return [p for p in parts if p] or [raw]
    return query_aliases(raw)


def keyword_query(query: str) -> str:
    """Strip stopwords to produce a denser, more searchable query variant."""
    kept = [w for w in query.split() if w.lower().strip(".,!?;:") not in STOPWORDS]
    return " ".join(kept).strip()


_BOOLEAN_RE = re.compile(r"\b(?:AND|OR|NOT)\b")
_BOOLEAN_OR_RE = re.compile(r"\s+OR\s+")
_BOOLEAN_HARD_RE = re.compile(r"\b(?:AND|NOT)\b")


def strip_boolean(query: str) -> str:
    """Reduce ``"Q" AND ("a" OR "b")`` syntax to plain keywords.

    Semantic backends (Tavily/Exa) would otherwise treat the operators as
    literal noise words. Queries without boolean operators are returned
    untouched so exact-phrase searches keep their quotes.
    """
    if not _BOOLEAN_RE.search(query):
        return query
    reduced = _BOOLEAN_RE.sub(" ", query)
    reduced = reduced.replace("(", " ").replace(")", " ").replace('"', " ")
    return " ".join(reduced.split())


def has_all_tokens(text: str, tokens: list[str]) -> bool:
    """True if every token appears as a whole word in ``text``.

    Apostrophes are folded and hyphens/underscores act as separators *before*
    matching, so ``Ro'yul`` still matches ``royul`` and ``Nur-Royul`` matches
    both ``nur`` and ``royul`` — spelling variants of a real name are kept,
    while lookalikes (``Nurul Amin``) remain rejected.
    """
    folded = _fold_variants(text)
    for token in tokens:
        tok = _fold_variants(token).strip()
        if not tok:
            continue
        if re.search(rf"\b{re.escape(tok)}\b", folded) is None:
            return False
    return True


def top_keywords(documents: list[Document], n: int = 15) -> list[tuple[str, int]]:
    """Return the most frequent content words across documents."""
    counter: Counter[str] = Counter()
    for doc in documents:
        counter.update(tokenize(doc.text))
    return counter.most_common(n)


def rank_sentences(
    documents: list[Document],
    query: str,
    n: int = 12,
) -> list[dict]:
    """Score sentences by overlap with query terms; return top-N (deduped)."""
    qterms = set(tokenize(query))
    scored: list[tuple[float, str, Document]] = []
    for doc in documents:
        for sent in split_sentences(doc.text):
            terms = tokenize(sent)
            overlap = len(qterms & set(terms))
            # Overlap with the query dominates; length rewards substance.
            score = overlap * 2.0 + min(len(terms), 40) / 40.0
            if overlap == 0:
                continue  # skip sentences with no topical link to the query
            scored.append((score, sent, doc))

    scored.sort(key=lambda item: item[0], reverse=True)

    seen: set[str] = set()
    out: list[dict] = []
    for score, sent, doc in scored:
        key = sent[:80].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "text": sent,
                "source": doc.url,
                "title": doc.title or "",
                "score": round(score, 2),
            }
        )
        if len(out) >= n:
            break
    return out
