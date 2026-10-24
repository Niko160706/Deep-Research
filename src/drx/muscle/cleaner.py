"""HTML cleaner: strip ads/nav/boilerplate and trim via BM25 relevance.

100% Python, 0 tokens. Primary extractor is ``trafilatura``; falls back to
BeautifulSoup (drop nav/footer/script), then to a regex strip. Paragraph
selection uses ``rank_bm25`` with a simple term-frequency fallback.
"""
from __future__ import annotations

import itertools
import logging
import re
from urllib.parse import urlparse

from drx.gate.decomposer import split_context
from drx.models import Document
from drx.textutil import (
    STOPWORDS,
    has_all_tokens,
    looks_like_name,
    name_tokens,
    split_aliases,
    tokenize,
)

log = logging.getLogger("drx.muscle.cleaner")

# Minimum Fellegi-Sunter score for a person-name document. Lowered from 0.50
# to 0.35 to retain relevant news/forum coverage (ArsTechnica, Reddit, NYTimes)
# that often misses one context/name signal. Strong lookalikes still score <=0.1.
ENTITY_LIKELIHOOD_MIN = 0.35

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")

_BOILERPLATE_TAGS = {
    "script", "style", "nav", "header", "footer", "aside", "form",
    "iframe", "noscript", "button", "input", "select", "svg",
}

# Profile platforms where the URL username itself is evidence of identity.
_PROFILE_DOMAINS = ("github.com", "linkedin.com", "twitter.com", "x.com")
_PROFILE_PATH_PREFIXES = {"in", "company", "pub", "profile", "user", "u", "pages", "org"}

# Social-media platforms: a profile URL there is NOT identity evidence. Market
# names (facebook.com/sri-lestari) belong to thousands of people at once and the
# page body is mostly friend lists / auto-generated widgets, so in strict mode a
# social profile whose slug is the name must still carry a context keyword.
_SOCIAL_DOMAINS = ("facebook.com", "instagram.com", "tiktok.com")


def is_social_media_url(url: str) -> bool:
    """True for facebook/instagram/tiktok URLs (mobile subdomains included)."""
    host = (urlparse(url).netloc or "").lower().removeprefix("www.")
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in _SOCIAL_DOMAINS)

# Capitalized words that commonly follow a name in prose/headlines (EN + ID) and
# must NOT trigger the homonym penalty ("Ainun Najib Raih People Of The Year",
# "Ainun Najib Bersama dua rekannya...").
_NAME_CONNECTORS = frozenset(
    {
        "bersama", "kata", "ujar", "menurut", "sebagai", "dalam", "pada", "adalah",
        "dari", "yang", "ini", "itu", "untuk", "dengan", "saat", "ketika", "dan",
        "seorang", "merupakan", "kembali", "juga", "masih", "sang", "bapak", "ibu",
        "the", "a", "an", "and", "who", "said", "says", "from", "with", "was",
        "is", "are", "his", "her", "their", "this", "that",
        # Indonesian headline verbs/nouns that follow the subject name
        "raih", "meraih", "pamer", "disebut", "diminta", "berikut", "kenal",
        "mengenal", "sosok", "fakta", "karya", "cerita", "prestasi", "inisiasi",
        "jejak", "kiprah", "sepakterjang", "mantan", "kini", "hingga", "sampai",
        "setelah", "sebelum", "usai", "selepas", "jelang", "demi", "bagi", "oleh",
        "tanpa", "soal", "terkait", "lalu", "kemudian", "bahkan", "malah", "tak",
        "tidak", "bukan", "jadi", "punya", "milik", "bikin", "buat", "hadir",
        "pernah", "sedang", "tengah", "bakal", "akan", "siap", "mampu",
        # common roles/titles that legitimately follow a real name
        "head", "chief", "founder", "co-founder", "ceo", "cto", "coo", "cfo",
        "director", "manager", "analyst", "engineer", "scientist", "researcher",
        "professor", "lecturer", "student", "author", "writer", "reporter",
        "journalist", "activist", "president", "minister", "doctor", "dr", "ir",
        "mr", "mrs", "ms", "inisiator", "pendiri", "penggagas", "praktisi",
        "pemuda", "santri", "alumni", "mahasiswa", "dosen", "guru", "ketua",
        "wakil", "menteri", "presiden", "gubernur", "bupati", "walikota",
        "rektor", "dekan", "putra", "putri", "bin", "binti", "ahli", "pria",
        "wanita", "anak", "pengusaha", "karyawan", "pegawai", "tokoh",
    }
)


def profile_name_match(url: str, entity: str) -> str:
    """Classify a profile URL against the entity name.

    Returns:
        ``"strong"`` — username carries the full name (``github.com/ainunnajib``).
        ``"weak"``   — username carries one long name token (``github.com/ec2ainun``).
        ``""``       — not a profile platform, or no match.
    """
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if not any(host == d or host.endswith("." + d) for d in _PROFILE_DOMAINS):
        return ""
    segments = [s for s in parsed.path.split("/") if s]
    if not segments:
        return ""
    index = (
        1
        if segments[0].lower() in _PROFILE_PATH_PREFIXES and len(segments) > 1
        else 0
    )
    username = re.sub(r"[^a-z0-9]", "", segments[index].lower())
    if not username:
        return ""
    tokens = name_tokens(entity)
    if not tokens:
        return ""
    concat = "".join(tokens)
    if concat and concat in username:
        return "strong"
    if any(len(t) >= 4 and t in username for t in tokens):
        return "weak"
    return ""


_IDENTITY_EXT_RE = re.compile(r"\.(?:html?|php|aspx?|jsp)$", re.IGNORECASE)


# Table/administrative labels that legitimately follow a name inside ASN /
# employee dumps ("Ainun Najib Golongan III/b", "Nama Umum 12 Golongan III/a").
# These are never a third name word, so the homonym guard must not fire on them.
_TABLE_WORDS = frozenset(
    {
        "nip", "pns", "pppk", "cpns", "golongan", "pangkat", "jabatan", "pelaksana",
        "nama", "nomor", "no", "nik", "npwp", "nuptk", "sk", "tmt", "tugas",
        "instansi", "unit", "kerja", "status", "keterangan", "jumlah", "total",
        "alamat", "tanggal", "lahir", "pendidikan", "jurusan", "tahun", "tabel",
        "daftar", "kabupaten", "kecamatan", "kelurahan", "desa", "provinsi",
        "kota", "laki", "perempuan", "pria", "wanita", "pegawai",
    }
)


# Login/consent wall shells (LinkedIn "join now", cookie banners, newsletter
# interstitials): the SEO title carries the name, the body is pure boilerplate.
# Real articles can mention one or two of these in a footer, so a page is only
# treated as a wall when it is BOTH phrase-dense and short. Measured on real
# runs: walls = 16 signs / 388 chars, articles = <=5 signs / >=3.3k chars.
_WALL_SIGNS = (
    "cookie", "privacy policy", "user agreement", "terms of service",
    "sign in", "log in", "join now", "sign up", "newsletter", "subscribe",
    "by clicking", "continue",
)
_WALL_MIN_SIGNS = 6
_WALL_MAX_CHARS = 2000


def _is_login_wall(text: str) -> bool:
    """True for login/consent shells that carry no content of their own."""
    if not text or len(text) > _WALL_MAX_CHARS:
        return False
    lowered = text.lower()
    return sum(lowered.count(sign) for sign in _WALL_SIGNS) >= _WALL_MIN_SIGNS


def identity_slug_match(url: str, entity: str) -> bool:
    """True when a URL path segment is exactly the person's name slug.

    A verified identity page (``cphof.org/profile/ainun-najib``,
    ``medium.com/@ainunnajib``, ``generalassemb.ly/ainunnajib``) carries the
    full name as one path segment. A news article URL
    (``.../institusi-kawal-covid-19-ainun-najib-raih-people-of-the-year-2021``)
    does not, because that segment contains extra words — so this rule cannot
    be used to smuggle name-dropping articles past the context gate.
    """
    tokens = name_tokens(entity)
    if len(tokens) < 2:
        return False
    concat = "".join(tokens)
    joined = "-".join(tokens)
    parsed = urlparse(url)

    def _is_name_slug(candidate: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", "-", candidate.lower()).strip("-")
        if not normalized:
            return False
        return normalized.replace("-", "") == concat or normalized == joined

    # Personal domain (ainunnajib.com, ainun-najib.id) — the host IS the name.
    host = (parsed.netloc or "").lower().removeprefix("www.")
    if host and _is_name_slug(host.split(".")[0]):
        return True

    for seg in (parsed.path or "").split("/"):
        if seg and _is_name_slug(_IDENTITY_EXT_RE.sub("", seg)):
            return True
    return False


def _name_from_boolean(text: str) -> str:
    """First quoted phrase of a boolean query that looks like a person name."""
    for match in re.findall(r'"([^"]{3,60})"', text or ""):
        candidate = match.strip()
        if looks_like_name(candidate):
            return candidate
    return ""


def third_name_word_penalty(text: str, query: str, context: str = "") -> float:
    """Return ``-0.4`` when a 2-word name is glued to a *different* person's name.

    Semantic backends (Exa) surface homonyms such as ``Ainun Najib Febrya
    Rahman`` for the query ``Ainun Najib``. The document is treated as that
    *other* person only when **both** hold:

    1. the name never appears "free" in the body — every occurrence is followed
       on the same line by another word (``Ainun Najib Febrya``), never by
       punctuation/newline/end of text, and
    2. that word is capitalized, is not a role/connector/context term, is not an
       acronym, and never occurs lowercase in the document (``Febrya`` vs
       ``Pria``/``Ahli``/``Raih``).

    News articles about the target always repeat the name in free positions
    ("Ainun Najib lahir pada...", "Ainun Najib, Ahli Sains...") and are therefore
    never penalized. Evaluate this on the page *body*: SERP titles are unreliable
    ("Ainun Najib | General Assembly" is a real profile, not a homonym).
    """
    tokens = name_tokens(query)
    if len(tokens) != 2:
        return 0.0
    allowed = set(tokens)
    for term in split_context(context):
        allowed.update(tokenize(term))

    # Ordinary words the document also spells in lowercase are prose, not names.
    lowercase_words = set(re.findall(r"\b[a-z][a-z'’.-]*", text))
    name_re = re.compile(
        rf"\b{re.escape(tokens[0])}\s+{re.escape(tokens[1])}\b", re.IGNORECASE
    )
    run_hit = False
    trigger = ""
    for match in name_re.finditer(text):
        rest = text[match.end() :]
        # Skip intra-line spaces only: punctuation/newline/end of text means the
        # name stands alone -> this is the target person.
        stripped = rest.lstrip(" \t")
        if not stripped or not stripped[0].isalpha():
            return 0.0
        first_word = re.match(r"[^\W\d_][\w'’.-]*", stripped)
        if first_word is None or not first_word.group(0)[0].isupper():
            return 0.0  # lowercase follow-up ("Ainun Najib lahir di...") -> prose
        # Walk the following words: a homonym page glues the name to another
        # name ("Ainun Najib Febrya [Rahman]").
        cursor = stripped
        while True:
            word_match = re.match(r"[^\W\d_][\w'’.-]*", cursor)
            if word_match is None:
                break
            raw_word = word_match.group(0)
            word = raw_word.strip("'’.-").lower()
            if not raw_word[0].isupper() or (len(raw_word) >= 2 and raw_word.isupper()):
                break  # lowercase word or acronym ("NU", "WNI") -> prose
            if (
                word in allowed
                or word in _NAME_CONNECTORS
                or word in _TABLE_WORDS
                or word in STOPWORDS
                or word in lowercase_words
            ):
                break
            run_hit = True
            if not trigger:
                trigger = raw_word
            cursor = cursor[word_match.end() :].lstrip(" \t")
    if run_hit:
        log.debug(
            "homonym guard: %r never appears free in the body (glued to %r)",
            f"{tokens[0]} {tokens[1]}",
            trigger,
        )
        return -0.4
    return 0.0


def validate_entity_likelihood(
    text: str,
    query: str,
    context: str = "",
    exploratory_terms: list[str] | None = None,
    url: str = "",
    body: str = "",
) -> float:
    """Fellegi-Sunter style probabilistic entity match score.

    ``text`` is what the name is matched against (title + body); ``body`` is the
    page body used by the homonym guard (defaults to ``text``).

    Weights:
        * Full/unique name match             -> +0.5
        * Anchored context keyword overlap   -> +0.3
        * Exploratory context entity overlap -> +0.2
        * Official profile URL (github/linkedin/x) with the name in the
          username                            -> +0.3
        * Homonym guard: 2-word name glued to a non-context name word
          (``Ainun Najib Febrya``)             -> -0.4
    """
    score = 0.0
    q_tokens = name_tokens(query)
    text_matched = bool(q_tokens) and has_all_tokens(text, q_tokens)
    profile = profile_name_match(url, query) if url else ""

    if text_matched:
        score += 0.5
        # Homonym guard: "Ainun Najib Febrya Rahman" is a different person.
        score += third_name_word_penalty(body or text, query, context)
    elif profile == "strong":
        # github.com/ainunnajib: the URL username *is* the identity evidence, so
        # a short or JS-thin profile body must not disqualify the page.
        log.debug("profile username carries the full name: %s", url)
        score += 0.5

    for term in split_context(context):
        term_tokens = tokenize(term)
        if term_tokens and has_all_tokens(text, term_tokens):
            score += 0.3
            break

    if exploratory_terms:
        for term in exploratory_terms:
            term_tokens = tokenize(term)
            if term_tokens and has_all_tokens(text, term_tokens):
                score += 0.2
                break

    if profile:
        # Genuine GitHub/LinkedIn/X profiles must not be dropped for being short
        # or for lacking the 70/30 context keywords.
        score += 0.3

    return round(max(score, 0.0), 2)


class Cleaner:
    """Turns raw HTML into concise, relevant plain text."""

    def __init__(
        self,
        min_text_length: int = 400,
        bm25_keep_top_k: int = 12,
        target_lang: str = "",
        require_context: bool = True,
        asn_noise_threshold: int = 10,
        asn_window_lines: int = 2,
    ) -> None:
        self.min_text_length = min_text_length
        self.bm25_keep_top_k = bm25_keep_top_k
        self.target_lang = target_lang.strip().lower() or None
        # Strict disambiguation: with --context, a doc must contain at least one
        # context keyword (kills common-name contamination).
        self.require_context = require_context
        self.asn_noise_threshold = asn_noise_threshold
        self.asn_window_lines = asn_window_lines

    def clean(
        self,
        documents: list[Document],
        query: str = "",
        context: str = "",
        entity_query: str = "",
    ) -> list[Document]:
        """Return documents that survive de-boilerplate + length + lang + entity.

        For person-name queries, a document is kept when its Fellegi-Sunter
        score reaches ``ENTITY_LIKELIHOOD_MIN`` (0.35; loosened from 0.50 to
        retain relevant news/forum coverage). The anchored context (+0.3) and
        exploratory entities (+0.2) boost the score. When ``context`` is given
        and ``require_context`` is on, a context keyword is **mandatory** for
        ordinary pages — common names ("Sri Lestari", "Ahmad Zaki") otherwise
        drag in unrelated people with the same name. This gate is *global*: it
        applies to every sub-query path (anchored 70% **and** exploratory 30%),
        so a high ``min_docs`` cannot smuggle homonyms in through the broad
        probe. Two kinds of page are exempt because they are about the person by
        construction:

        * official profiles (GitHub/LinkedIn/X) whose username carries the name;
        * identity pages whose URL slug *is* the name
          (``cphof.org/profile/ainun-najib``, ``medium.com/@ainunnajib``).

        Social media (facebook/instagram/tiktok) is **not** exempt: a
        ``facebook.com/sri-lestari`` URL belongs to some other "Sri Lestari" just
        as well and the body is friend-list noise, so the context keyword stays
        mandatory there (with no ``context`` given at all, such profiles are
        dropped in strict mode because they can never satisfy it).

        Login/consent walls (the name in the SEO title, the body made of "join
        now"/"cookie policy" boilerplate) are dropped in **every** mode: they are
        empty shells, and no context keyword can turn them into evidence.

        Context terms are matched whitespace-insensitively (``KawalCOVID19``
        still matches ``Kawal Covid-19``).

        The gates always judge the **full page** (title + complete extracted
        text), while the text stored for synthesis is the ASN-windowed excerpt:
        a "DINAS KESEHATAN" header outside the target's table row still counts
        as context evidence for that page.

        ``entity_query`` is the entity to validate against and defaults to
        ``query``. Anchored 70/30 queries pass the original entity here so that
        boolean/context syntax never disables the name filter, while ``query``
        still drives the BM25 relevance trim. ``entity_query`` may hold several
        aliases separated by ``" OR "`` (``"Bayu Skak" OR "Bayu Eko Moektito"``):
        a document is kept when **any** alias validates.
        """
        entities = split_aliases(entity_query or query)
        entity = entities[0] if entities else ""
        if not looks_like_name(entity):
            # A boolean anchored query (``"A" AND ("x" OR "y")``) must never
            # disable the name filter: fall back to its quoted name phrase.
            fallback = _name_from_boolean(entity_query) or _name_from_boolean(query)
            if fallback:
                entities = [fallback, *[e for e in entities if e != fallback]]
                entity = fallback
        is_name = looks_like_name(entity)
        context_terms = split_context(context)
        cleaned: list[Document] = []
        dropped_context = 0
        dropped_wall = 0
        for doc in documents:
            text = self._extract_main_content(doc)
            text = self._normalize(text)
            # trafilatura occasionally latches onto a sidebar/widget and returns a
            # block that never names the target, while the raw page does. For a
            # person query, prefer the full de-boilerplated text in that case.
            if is_name and not self._has_any_name(text, entities):
                fallback = self._fallback_text(doc)
                if fallback and self._has_any_name(fallback, entities):
                    log.debug(
                        "extractor missed the name -> full-text fallback: %s", doc.url
                    )
                    text = self._normalize(fallback)
            # Login/consent walls: the name sits in the SEO title while the body
            # is "join now / cookie policy" boilerplate (LinkedIn shells). They
            # are dropped in every mode — no context keyword could make such a
            # page evidence.
            if _is_login_wall(text):
                dropped_wall += 1
                log.debug("skip %s: login/consent wall (no real content)", doc.url)
                continue
            # PDF/tabular ASN dumps: keep only the window around the target name
            # so "NIP"/"Golongan"/"Pelaksana" cannot dominate the report.
            full_text = text
            text = self._window_asn_table(full_text, entities)
            profile = profile_name_match(doc.url, entity) if is_name else ""
            # Identity page = platform profile OR URL slug that *is* the name
            # (cphof.org/profile/ainun-najib, medium.com/@ainunnajib): the page
            # is about this person by construction, so it needs no context word.
            identity = bool(profile) or (
                identity_slug_match(doc.url, entity) if is_name else False
            )
            # Social media is the exception: facebook.com/sri-lestari belongs to
            # some other "Sri Lestari" just as well, and the body is friend-list
            # noise — so identity-by-slug earns no context exemption there.
            social_identity = identity and is_social_media_url(doc.url)
            if len(text) < self.min_text_length and not identity:
                log.debug("skip %s: too short (%d chars)", doc.url, len(text))
                continue
            # Probe = title + FULL page text: the gates judge the page as a whole
            # (an ASN dump's "DINAS KESEHATAN" header lives outside the target's
            # table row, i.e. outside the window that gets stored). The stored
            # payload stays the windowed excerpt.
            probe = f"{doc.title}\n{full_text}" if doc.title else full_text
            if is_name:
                score, matched = self._best_entity_score(
                    probe, entities, context, doc.url, text
                )
                if score < ENTITY_LIKELIHOOD_MIN:
                    log.debug(
                        "skip %s: entity likelihood %.2f < %.2f (entity=%r)",
                        doc.url,
                        score,
                        ENTITY_LIKELIHOOD_MIN,
                        entity,
                    )
                    continue
                # The lower 0.35 floor must not let a known homonym through:
                # negative evidence (2-word name glued to another name) still
                # vetoes the document even when a profile bonus lifts it to 0.4.
                if matched and third_name_word_penalty(text, matched, context) < 0:
                    log.debug(
                        "skip %s: homonym guard (name glued to another name)",
                        doc.url,
                    )
                    continue
                if identity and not profile and not social_identity:
                    log.debug("identity page (URL slug == name) -> kept: %s", doc.url)

            # ---- GLOBAL strict --context gate ------------------------------
            # Identical rule on every sub-query path: anchored 70% *and*
            # exploratory 30%. A high min_docs must not let homonyms in through
            # the broad probe. Only the two identity exemptions above survive it
            # (profile username / slug == name), and social-media profiles do
            # not: with no context given at all they can never satisfy the rule.
            if self.require_context:
                if social_identity:
                    if not context_terms or not self._context_hit(probe, context_terms):
                        dropped_context += 1
                        log.debug(
                            "skip %s: social-media identity page without a "
                            "--context keyword (strict disambiguation)",
                            doc.url,
                        )
                        continue
                elif (
                    is_name
                    and context_terms
                    and not identity
                    and not self._context_hit(probe, context_terms)
                ):
                    dropped_context += 1
                    log.debug(
                        "skip %s: no --context keyword present (strict "
                        "disambiguation, entity=%r)",
                        doc.url,
                        matched,
                    )
                    continue
            if self.target_lang and not self._lang_matches(text):
                log.debug("skip %s: language != %s", doc.url, self.target_lang)
                continue
            if query:
                text = self._trim_by_relevance(text, query)
            if not text.strip():
                continue
            cleaned.append(doc.model_copy(update={"text": text}))
        if dropped_context:
            log.info(
                "strict --context gate: %d/%d docs dropped (no context keyword; "
                "applies to every sub-query path)",
                dropped_context,
                len(documents),
            )
        if dropped_wall:
            log.info(
                "login/consent walls dropped: %d/%d docs (no real content)",
                dropped_wall,
                len(documents),
            )
        return cleaned

    @staticmethod
    def _has_any_name(text: str, entities: list[str]) -> bool:
        """True when the full name of any alias appears in ``text``."""
        for alias in entities:
            tokens = name_tokens(alias)
            if tokens and has_all_tokens(text, tokens):
                return True
        return False

    @staticmethod
    def _fallback_text(doc: Document) -> str:
        """Full de-boilerplated text (BeautifulSoup), trafilatura bypassed."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return ""
        try:
            soup = BeautifulSoup(doc.text or "", "lxml")
            for tag in soup(_BOILERPLATE_TAGS):
                tag.decompose()
            body = soup.body or soup
            return body.get_text("\n", strip=True)
        except Exception:  # pragma: no cover - parser edge cases
            return ""

    @staticmethod
    def _best_entity_score(
        probe: str,
        entities: list[str],
        context: str,
        url: str,
        body: str,
    ) -> tuple[float, str]:
        """Score the probe against every alias; return the best (score, alias)."""
        best = (0.0, entities[0] if entities else "")
        for alias in entities:
            if not alias:
                continue
            score = validate_entity_likelihood(
                probe, alias, context, url=url, body=body
            )
            if score > best[0]:
                best = (score, alias)
        return best

    @staticmethod
    def _context_hit(probe: str, context_terms: list[str]) -> bool:
        """True when at least one --context term appears in the document.

        Whole-token matching is tried first; single-token terms are additionally
        matched against the whitespace-stripped text so a compact spelling
        (``KawalCOVID19``, ``DataScientist``) still hits the spaced form used in
        prose (``Kawal Covid-19``).
        """
        compact = re.sub(r"[^a-z0-9]+", "", probe.lower())
        for term in context_terms:
            tokens = tokenize(term)
            if tokens and has_all_tokens(probe, tokens):
                return True
            slug = re.sub(r"[^a-z0-9]+", "", term.lower())
            # >= 5 chars avoids accidental substring hits on short words.
            if len(slug) >= 5 and slug in compact:
                return True
        return False

    def _window_asn_table(self, text: str, entities: list[str]) -> str:
        """Collapse massive ASN/employee tables to the target's ±N row window.

        Dumps like "Daftar Nama Pegawai" repeat ``NIP``/``PNS``/``Golongan``
        hundreds of times; only the rows naming the target are relevant. HTML
        tables and most PDF extractions keep their line structure and are
        windowed line-by-line. A PDF text layer can also arrive as ONE gigantic
        line (no ``\\n`` at all): ±N lines would then keep the entire dump, so
        that case is windowed by characters instead (≈N table rows).
        """
        if not text or not entities:
            return text
        noise = sum(
            len(re.findall(rf"\b{kw}\b", text, re.IGNORECASE))
            for kw in ("NIP", "PNS", "Golongan")
        )
        if noise < self.asn_noise_threshold:
            return text
        lines = text.split("\n")
        if len(lines) <= 2 * self.asn_window_lines + 1:
            return self._window_asn_blob(text, entities, noise)
        keep: set[int] = set()
        for alias in entities:
            tokens = name_tokens(alias)
            if not tokens:
                continue
            for index, line in enumerate(lines):
                if has_all_tokens(line, tokens):
                    keep.update(
                        range(
                            max(0, index - self.asn_window_lines),
                            min(len(lines), index + self.asn_window_lines + 1),
                        )
                    )
        if not keep:
            log.debug(
                "ASN table noise (%d hits) but target name not found in any row "
                "-> drop table body", noise,
            )
            return ""
        windowed = "\n".join(lines[i] for i in sorted(keep))
        log.info(
            "ASN/PDF table noise (%d hits) -> windowed to %d/%d lines",
            noise,
            len(keep),
            len(lines),
        )
        return windowed

    _ASN_ROW_CHARS = 160  # fallback width of one "… NIP: … Golongan …" row

    @classmethod
    def _estimate_row_chars(cls, text: str) -> int:
        """Median gap between consecutive ``NIP`` hits ≈ one table row's width."""
        hits = [m.start() for m in re.finditer(r"\bNIP\b", text, re.IGNORECASE)]
        gaps = sorted(b - a for a, b in itertools.pairwise(hits) if b > a)
        if not gaps:
            return cls._ASN_ROW_CHARS
        return max(80, min(400, gaps[len(gaps) // 2]))

    def _window_asn_blob(self, text: str, entities: list[str], noise: int) -> str:
        """Character window (±N rows) for single-line PDF text layers.

        ``pad`` is :attr:`asn_window_lines` table rows' worth of characters on
        each side of every name occurrence (row width estimated from the dump
        itself); overlapping windows are merged so a name repeated in the same
        block does not duplicate text.
        """
        pad = max(1, self.asn_window_lines) * self._estimate_row_chars(text)
        spans: list[list[int]] = []
        for alias in entities:
            tokens = name_tokens(alias)
            if not tokens:
                continue
            pattern = re.compile(
                r"\s+".join(re.escape(t) for t in tokens), re.IGNORECASE
            )
            for match in pattern.finditer(text):
                spans.append(
                    [max(0, match.start() - pad), min(len(text), match.end() + pad)]
                )
        if not spans:
            log.debug(
                "ASN blob noise (%d hits) but target name not found -> drop "
                "table body",
                noise,
            )
            return ""
        spans.sort()
        merged = [spans[0]]
        for start, end in spans[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        windowed = "\n".join(text[start:end].strip() for start, end in merged)
        log.info(
            "ASN/PDF blob noise (%d hits) -> windowed to %d/%d chars "
            "(%d row windows)",
            noise,
            len(windowed),
            len(text),
            len(merged),
        )
        return windowed

    # ---- main-content extraction ----
    def _extract_main_content(self, doc: Document) -> str:
        html = doc.text or ""
        if not html.strip():
            return ""
        # 1) trafilatura (best signal for article/main content)
        try:
            import trafilatura

            extracted = trafilatura.extract(
                html, include_comments=False, include_tables=False
            )
            if extracted:
                return extracted
        except ImportError:
            pass
        # 2) BeautifulSoup de-boilerplate
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "lxml")
            for tag in soup(_BOILERPLATE_TAGS):
                tag.decompose()
            body = soup.body or soup
            return body.get_text("\n", strip=True)
        except ImportError:
            pass
        # 3) regex strip (last resort)
        return _TAG_RE.sub(" ", html)

    @staticmethod
    def _normalize(text: str) -> str:
        text = _WS_RE.sub(" ", text)
        text = _NL_RE.sub("\n\n", text)
        return text.strip()

    # ---- language filter ----
    def _lang_matches(self, text: str) -> bool:
        try:
            from langdetect import detect

            return detect(text[:2000]) == self.target_lang
        except Exception:
            return True  # on detection failure, keep the document

    # ---- relevance trim ----
    def _trim_by_relevance(self, text: str, query: str) -> str:
        paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
        if not paragraphs:
            return text
        try:
            from rank_bm25 import BM25Okapi

            corpus = [self._tokenize(p) for p in paragraphs]
            bm25 = BM25Okapi(corpus)
            scores = bm25.get_scores(self._tokenize(query))
            idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            idx = sorted(idx[: self.bm25_keep_top_k])
            return "\n".join(paragraphs[i] for i in idx)
        except ImportError:
            qtokens = set(self._tokenize(query))
            scored = [
                (sum(1 for t in qtokens if t in self._tokenize(p)), p)
                for p in paragraphs
            ]
            scored.sort(key=lambda pair: pair[0], reverse=True)
            return "\n".join(p for _, p in scored[: self.bm25_keep_top_k])

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1]
