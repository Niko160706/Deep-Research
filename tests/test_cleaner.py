"""Unit tests for the HTML cleaner."""
from __future__ import annotations

from drx.models import Document
from drx.muscle.cleaner import (
    ENTITY_LIKELIHOOD_MIN,
    Cleaner,
    _is_login_wall,
    identity_slug_match,
    is_social_media_url,
    profile_name_match,
    third_name_word_penalty,
    validate_entity_likelihood,
)


def _doc(html: str, url: str = "http://example.com/a") -> Document:
    return Document(url=url, text=html)


def test_strips_boilerplate_and_scripts() -> None:
    cleaner = Cleaner(min_text_length=10, bm25_keep_top_k=5)
    html = (
        "<html><body>"
        "<nav>menu</nav><script>var x = 1;</script>"
        "<p>" + "hello world " * 50 + "</p>"
        "</body></html>"
    )
    docs = cleaner.clean([_doc(html)])
    assert len(docs) == 1
    assert "var x" not in docs[0].text
    assert "menu" not in docs[0].text


def test_too_short_document_rejected() -> None:
    cleaner = Cleaner(min_text_length=1000, bm25_keep_top_k=5)
    assert cleaner.clean([_doc("<p>tiny</p>")]) == []


def test_query_trim_reduces_text() -> None:
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=1)
    paragraphs = ["apple banana cherry", "dog cat mouse", "apple pie recipe"]
    text = "\n\n".join(paragraphs)
    docs = cleaner.clean([_doc(f"<p>{text}</p>")], query="apple pie")
    assert len(docs) == 1
    assert "apple pie recipe" in docs[0].text


# ---- Fellegi-Sunter entity likelihood ----

def test_entity_likelihood_full_name_plus_context() -> None:
    text = "Nur Royul Amin dari Surabaya bekerja di bidang informatika."
    assert validate_entity_likelihood(text, "Nur Royul Amin", "Surabaya, Informatika") == 0.8


def test_entity_likelihood_full_name_only_survives_exploratory() -> None:
    text = "Nur Royul Amin adalah seorang peneliti muda."
    assert validate_entity_likelihood(text, "Nur Royul Amin", "Surabaya") == 0.5


def test_entity_likelihood_lookalike_scores_zero() -> None:
    text = "Nurul Amin dari Pakistan adalah aktivis politik."
    assert validate_entity_likelihood(text, "Nur Royul Amin", "Surabaya") == 0.0


def test_entity_likelihood_exploratory_bonus() -> None:
    text = "Nur Royul Amin aktif di komunitas Jaringan Riset."
    score = validate_entity_likelihood(
        text, "Nur Royul Amin", "", exploratory_terms=["Jaringan Riset"]
    )
    assert score == 0.7  # 0.5 name + 0.2 exploratory entity


def test_cleaner_uses_loosened_entity_threshold(monkeypatch) -> None:
    """A 0.40 score now survives; 0.34 still fails the 0.35 gate."""
    assert ENTITY_LIKELIHOOD_MIN == 0.35
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    doc_keep = Document(url="http://keep", text="Nur Royul Amin menulis riset.")
    doc_drop = Document(url="http://drop", text="Nur Royul Amin menulis riset.")

    def fake_score(probe, entities, context, url, body):
        return (0.40 if url.endswith("/keep") else 0.34), entities[0]

    monkeypatch.setattr(
        Cleaner, "_best_entity_score", staticmethod(fake_score)
    )
    docs = cleaner.clean([doc_keep, doc_drop], query="Nur Royul Amin")
    assert [d.url for d in docs] == ["http://keep"]


def test_cleaner_keeps_exploratory_doc_with_strong_name_match() -> None:
    """Docs on the 30% exploratory path are kept when the full name matches.

    Loose mode (``require_context=False``): context words are a bonus, not a
    requirement — the exploratory sweep keeps a strong name match.
    """
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5, require_context=False)
    doc = Document(url="http://broad", text="Nur Royul Amin menulis tentang riset terbuka.")
    docs = cleaner.clean([doc], query="Nur Royul Amin", context="Surabaya, Informatika")
    assert len(docs) == 1
    assert docs[0].url == "http://broad"


def test_strict_context_gate_drops_doc_without_context_keyword() -> None:
    """Common-name disambiguation: with --context, a context word is mandatory."""
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    ctx_doc = Document(url="http://ctx", text="Sri Lestari bekerja sebagai perawat di Sleman.")
    other = Document(url="http://other", text="Sri Lestari berdagang di pasar Beringharjo.")
    docs = cleaner.clean(
        [ctx_doc, other], query="Sri Lestari", context="Sleman, Kesehatan"
    )
    assert [d.url for d in docs] == ["http://ctx"]


def test_strict_context_matches_compact_spelling() -> None:
    """``KawalCOVID19`` (query) must match ``Kawal Covid-19`` (prose)."""
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    doc = Document(
        url="http://nu.or.id/x",
        title="Inisiasi Kawal Covid-19, Ainun Najib Raih Penghargaan",
        text="Ainun Najib menginisiasi gerakan relawan itu dari Singapura.",
    )
    docs = cleaner.clean(
        [doc], query="Ainun Najib", context="KawalCOVID19, Data Scientist"
    )
    assert [d.url for d in docs] == ["http://nu.or.id/x"]


def test_identity_slug_pages_exempt_from_context_gate() -> None:
    """URL slug == name -> identity page (kept); a news slug is not (dropped)."""
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    identity = Document(
        url="https://cphof.org/profile/ainun-najib",
        text="Ainun Najib is listed as a competitor in the 2015 contest.",
    )
    news = Document(
        url="https://example.com/ainun-najib-raih-penghargaan-bergengsi-2021",
        text="Ainun Najib meraih penghargaan bergengsi tahun ini.",
    )
    docs = cleaner.clean(
        [identity, news], query="Ainun Najib", context="KawalCOVID19, Data Scientist"
    )
    assert [d.url for d in docs] == [identity.url]


def test_cleaner_entity_query_filters_anchored_boolean_queries() -> None:
    """Boolean anchored syntax must not disable the entity (name) filter."""
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    good = Document(
        url="http://good",
        text="Nur Royul Amin menulis tentang sistem operasi Xinu di Surabaya.",
    )
    bad = Document(
        url="http://bad",
        text="Raka Amin Nur Soleh menulis tentang jaringan komputer di Surabaya.",
    )
    anchored = '"Nur Royul Amin" AND ("Surabaya" OR "Informatika")'
    docs = cleaner.clean(
        [good, bad],
        query=anchored,
        context="Surabaya, Informatika",
        entity_query="Nur Royul Amin",
    )
    assert [d.url for d in docs] == ["http://good"]


def test_cleaner_keeps_apostrophe_name_variant() -> None:
    """A real spelling variant (Ro'yul) must survive the entity filter."""
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    doc = Document(
        url="http://variant",
        text="Nur Ro'yul Amin menulis tentang manajemen memori Xinu.",
    )
    docs = cleaner.clean([doc], query="Nur Royul Amin")
    assert [d.url for d in docs] == ["http://variant"]


def test_cleaner_keeps_doc_when_name_only_in_title() -> None:
    """A page named after the person is kept even if the body omits the name."""
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    doc = Document(
        url="http://title-only",
        title="Nur Ro'yul Amin – Medium",
        text="Instalasi Ubuntu di Oracle VirtualBox dan setup guest additions.",
    )
    docs = cleaner.clean([doc], query="Nur Royul Amin")
    assert [d.url for d in docs] == ["http://title-only"]


def test_cleaner_drops_lookalike_title() -> None:
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    doc = Document(
        url="http://lookalike",
        title="Md Nurul Amin - For Your Health",
        text="Kesehatan dan olahraga untuk semua orang.",
    )
    assert cleaner.clean([doc], query="Nur Royul Amin") == []


# ---- Homonym guard: 2-word name glued to a third (non-context) word ----

def test_third_word_penalty_eliminates_exa_homonym() -> None:
    text = (
        "Person Ainun Najib Febrya Rahman Career map 2024 Muhammadiyah "
        "University of Surabaya student profile."
    )
    assert third_name_word_penalty(text, "Ainun Najib") == -0.4
    assert validate_entity_likelihood(text, "Ainun Najib") == 0.1


def test_cleaner_drops_exa_homonym_document() -> None:
    """Real noise from the 'Ainun Najib' benchmark (exa.ai person library)."""
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    homonym = Document(
        url="https://exa.ai/library/person/gfqm9z9q7b0",
        title="ainun najib febrya rahman",
        text=(
            "Person Ainun Najib Febrya Rahman Career map 20242026 Muhammadiyah "
            "University Purwokerto Work Education 1Work Experience - "
            "MUMuhammadiyah University Purwokerto2024 2Publications Total of 1 "
            "work and 1 citation, including 1 work in 2024, with an h-index of 1."
        ),
    )
    assert cleaner.clean([homonym], query="Ainun Najib") == []


def test_bare_extended_name_title_penalty() -> None:
    """A single extra name word after the query name is already a homonym."""
    assert third_name_word_penalty("Person Ainun Najib Febrya.", "Ainun Najib") == -0.4


def test_role_after_name_not_penalized() -> None:
    """"Ainun Najib Head of Analytics" is a role, not a longer name."""
    assert third_name_word_penalty("Ainun Najib Head of Analytics at Grab.", "Ainun Najib") == 0.0


def test_site_title_not_penalized() -> None:
    """"Ainun Najib | General Assembly" is a real profile, not a homonym."""
    body = "Instructor profile. Ainun Najib teaches product management."
    assert third_name_word_penalty(body, "Ainun Najib") == 0.0


def test_profile_card_body_not_penalized() -> None:
    """cphof.org-style profile card: the name stands alone on its own line."""
    text = "Ainun Najib\nCountry:\nSingapore\nPrize Money:\n$100\nRank: 12"
    assert third_name_word_penalty(text, "Ainun Najib") == 0.0


def test_prefix_full_name_not_penalized() -> None:
    """"Moch. Ainun Najib" is still the target (prefix, not suffix)."""
    assert third_name_word_penalty(
        "Moch. Ainun Najib\nJakarta\nSoftware engineer.", "Ainun Najib"
    ) == 0.0


def test_no_penalty_when_name_also_appears_free() -> None:
    """News articles repeat the name in free positions -> never penalized."""
    text = (
        "Inisiasi Kawal Covid-19, Ainun Najib Raih People Of The Year 2021. "
        "Ainun Najib seorang praktisi teknologi informasi asal Gresik."
    )
    assert third_name_word_penalty(text, "Ainun Najib") == 0.0
    assert validate_entity_likelihood(text, "Ainun Najib") == 0.5


def test_no_penalty_for_headline_case_article() -> None:
    """Indonesian headlines are Title Case: 'Prestasi Ainun Najib Pria NU'."""
    text = (
        "Diminta Jokowi Pulang, Berikut Prestasi Ainun Najib Pria NU Bidang IT. "
        "Ainun Najib kini bekerja di Singapura."
    )
    assert third_name_word_penalty(text, "Ainun Najib") == 0.0


def test_no_penalty_when_name_stands_alone_on_a_line() -> None:
    """Profile cards ('Ainun Najib\\nCountry:') must not be penalized."""
    text = "Ainun Najib\nCountry:\nSingapore\nPrize Money:\n$100"
    assert third_name_word_penalty(text, "Ainun Najib") == 0.0
    assert validate_entity_likelihood(text, "Ainun Najib") == 0.5


def test_no_penalty_when_following_word_is_lowercase_elsewhere() -> None:
    """'Ainun Najib Pamer Gaji' -> 'pamer' is prose, not a name part."""
    text = "Ainun Najib Pamer Gaji. Ia pamer gaji Rp53 miliar di Singapura."
    assert third_name_word_penalty(text, "Ainun Najib") == 0.0


def test_third_word_penalty_skips_context_word() -> None:
    text = "Ainun Najib Data Scientist di Grab memimpin tim analitik."
    assert third_name_word_penalty(text, "Ainun Najib", "KawalCOVID19, Data Scientist") == 0.0
    score = validate_entity_likelihood(
        text, "Ainun Najib", "KawalCOVID19, Data Scientist"
    )
    assert score == 0.8  # 0.5 name + 0.3 context, no homonym penalty


def test_third_word_penalty_skips_sentence_connector() -> None:
    text = "Ainun Najib Bersama dua rekannya membangun layanan warga."
    assert third_name_word_penalty(text, "Ainun Najib") == 0.0
    assert validate_entity_likelihood(text, "Ainun Najib") == 0.5


def test_third_word_penalty_only_applies_to_two_word_names() -> None:
    text = "Nur Royul Amin Pratama sedang menulis tesis."
    assert third_name_word_penalty(text, "Nur Royul Amin") == 0.0


def test_third_word_penalty_ignores_comma_separated_roles() -> None:
    text = "Ainun Najib, head of Analytics at Grab, berbicara di panel."
    assert third_name_word_penalty(text, "Ainun Najib") == 0.0


def test_cleaner_keeps_news_article_about_the_target() -> None:
    """Regression: the homonym guard must not eat real coverage."""
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    doc = Document(
        url="https://www.nu.or.id/nasional/inisiasi-kawal-covid-19-ainun-najib-raih-people-of-the-year-2021",
        title="Inisiasi Kawal Covid-19, Ainun Najib Raih People Of The Year 2021",
        text=(
            "Jakarta, NU Online. Ainun Najib seorang praktisi teknologi informasi "
            "yang kini berdomisili di Singapura."
        ),
    )
    docs = cleaner.clean([doc], query="Ainun Najib", context="KawalCOVID19, Data Scientist")
    assert [d.url for d in docs] == [doc.url]


# ---- Profile URLs (github / linkedin / twitter / x) ----

def test_profile_name_match_strong_weak_none() -> None:
    assert profile_name_match("https://github.com/ainunnajib", "Ainun Najib") == "strong"
    assert profile_name_match("https://github.com/ec2ainun", "Ainun Najib") == "weak"
    assert profile_name_match("https://github.com/torvalds", "Ainun Najib") == ""
    assert profile_name_match("https://example.com/ainunnajib", "Ainun Najib") == ""


def test_profile_name_match_linkedin_prefix_and_x_domain() -> None:
    assert profile_name_match("https://www.linkedin.com/in/ainunnajib", "Ainun Najib") == "strong"
    assert profile_name_match("https://x.com/ec2ainun", "Ainun Najib") == "weak"


def test_profile_bonus_keeps_short_github_profile() -> None:
    """A genuine profile is not dropped for being short (min_text_length)."""
    cleaner = Cleaner(min_text_length=500, bm25_keep_top_k=5)
    doc = Document(
        url="https://github.com/ainunnajib",
        title="Ainun Najib (ainunnajib) · GitHub",
        text="Ainun Najib ainunnajib butterfly's effect learner and bug tamer.",
    )
    docs = cleaner.clean([doc], query="Ainun Najib", context="KawalCOVID19")
    assert [d.url for d in docs] == ["https://github.com/ainunnajib"]


def test_profile_url_username_is_identity_evidence_for_thin_body() -> None:
    """JS-thin profile pages still count: the username carries the full name."""
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    doc = Document(url="https://github.com/ainunnajib", text="Sign in to GitHub.")
    docs = cleaner.clean([doc], query="Ainun Najib")
    assert [d.url for d in docs] == ["https://github.com/ainunnajib"]


def test_weak_profile_username_alone_is_not_enough() -> None:
    """A single-token username match needs the name in the page text too."""
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    doc = Document(url="https://github.com/ec2ainun", text="Sign in to GitHub.")
    assert cleaner.clean([doc], query="Ainun Najib") == []


def test_weak_profile_username_kept_with_name_in_text() -> None:
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    doc = Document(
        url="https://github.com/ec2ainun",
        title="Ainun Najib · GitHub",
        text="Ainun Najib builds open data tools.",
    )
    docs = cleaner.clean([doc], query="Ainun Najib")
    assert [d.url for d in docs] == ["https://github.com/ec2ainun"]


def test_linkedin_lookalike_username_not_rescued() -> None:
    """A lookalike profile must not pass on the +0.3 username bonus alone."""
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    doc = Document(
        url="https://www.linkedin.com/in/nurul-amin-b5176a22",
        title="Nurul Amin - For Your Health | LinkedIn",
        text="Kesehatan dan olahraga untuk semua orang di Pakistan.",
    )
    assert cleaner.clean([doc], query="Nur Royul Amin") == []


def test_homonym_profile_bonus_cannot_outweigh_penalty() -> None:
    """0.5 name - 0.4 homonym + 0.3 profile bonus = 0.4."""
    text = "Ainun Najib Febrya Rahman profil mahasiswa Universitas Muhammadiyah."
    assert validate_entity_likelihood(
        text, "Ainun Najib", url="https://github.com/ainun-najib-febrya"
    ) == 0.4
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    doc = Document(url="https://github.com/ainun-najib-febrya", text=text)
    # Score 0.4 passes the loosened 0.35 floor, but the homonym veto still wins.
    assert cleaner.clean([doc], query="Ainun Najib") == []


# ---- Identity pages (URL slug == name) ----

def test_identity_slug_match_accepts_name_slugs() -> None:
    assert identity_slug_match("https://cphof.org/profile/ainun-najib", "Ainun Najib")
    assert identity_slug_match("https://medium.com/@ainunnajib", "Ainun Najib")
    assert identity_slug_match("https://generalassemb.ly/ainunnajib", "Ainun Najib")
    assert identity_slug_match("https://x.id/author/Ainun-Najib.html", "Ainun Najib")


def test_identity_slug_match_accepts_personal_domain() -> None:
    """A domain that IS the name belongs to the person (ainunnajib.com)."""
    assert identity_slug_match("https://ainunnajib.com/", "Ainun Najib")
    assert identity_slug_match("https://www.ainun-najib.id/tentang", "Ainun Najib")
    assert not identity_slug_match("https://beritaainun.com/x", "Ainun Najib")


def test_identity_slug_match_rejects_article_slugs() -> None:
    """A news slug with extra words is NOT an identity page."""
    assert not identity_slug_match(
        "https://nu.or.id/nasional/inisiasi-kawal-covid-19-ainun-najib-raih-people-of-the-year-2021",
        "Ainun Najib",
    )
    assert not identity_slug_match("https://exa.ai/library/person/gfqm9z9q7b0", "Ainun Najib")
    assert not identity_slug_match("https://situs.id/berita/1001", "Ainun Najib")


def test_identity_slug_single_token_name_not_exempt() -> None:
    """Single-token entities are too weak to be exempted by a slug."""
    assert not identity_slug_match("https://situs.id/zaki", "Zaki")


# ---- ASN / employee-table windowing (PDF noise) ----

def _asn_dump(rows: int = 40, target_row: int = 41) -> str:
    head = ["DAFTAR NAMA PEGAWAI", "No NIP Nama Golongan Jabatan"]
    body = [
        f"{i:03d} NIP:19850101{i:04d} Nama Umum {i} Golongan III/a Pelaksana Kesehatan"
        for i in range(1, rows + 1)
    ]
    target = f"{target_row:03d} NIP:198501010999 Ainun Najib Golongan III/b Analis Sistem"
    return "\n".join([*head, *body, target, *body])


def test_asn_table_windowed_to_target_rows() -> None:
    """NIP/Golongan dumps are cut to the ±2 line window around the target."""
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=50)
    text = _asn_dump()
    assert text.count("Golongan") > 10  # noisy dump by construction
    docs = cleaner.clean(
        [Document(url="https://sleman.go.id/dokumen/daftar-pegawai.pdf", text=text)],
        query="Ainun Najib",
    )
    assert len(docs) == 1
    out = docs[0].text
    assert "Ainun Najib" in out
    # window +/-2 lines -> 5 lines kept (plus the header lines if adjacent)
    assert out.count("Nama Umum") <= 4
    assert "Nama Umum 5 " not in out  # far-away rows are gone
    assert out.count("NIP:") <= 5


def test_asn_table_without_target_name_is_dropped() -> None:
    """A bulk ASN dump that never names the target contributes nothing."""
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=50)
    docs = cleaner.clean(
        [Document(url="https://sleman.go.id/dokumen/pegawai.pdf", text=_asn_dump())],
        query="Sri Lestari",
    )
    assert docs == []


def test_name_filter_survives_boolean_entity_query() -> None:
    """An anchored boolean entity_query still enforces the name filter."""
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    good = Document(url="http://good", text="Ainun Najib bekerja di Surabaya.")
    bad = Document(url="http://bad", text="Ainun Rahman bekerja di Surabaya.")
    anchored = '"Ainun Najib" AND ("Surabaya" OR "Informatika")'
    docs = cleaner.clean(
        [good, bad],
        query=anchored,
        context="Surabaya, Informatika",
        entity_query=anchored,
    )
    assert [d.url for d in docs] == ["http://good"]


def test_extractor_miss_falls_back_to_full_text(monkeypatch) -> None:
    """When trafilatura returns a sidebar that omits the name, use full text."""
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    html = (
        "<html><body>"
        "<div class='sidebar'>Berita Lain Opini Kalam Sejawat</div>"
        "<article><h1>Inisiasi Kawal Covid-19, Ainun Najib Raih Penghargaan</h1>"
        "<p>Ainun Najib adalah praktisi teknologi informasi di Singapura.</p>"
        "</article></body></html>"
    )
    doc = Document(url="http://nu.or.id/x", text=html)
    # force the bad extraction (trafilatura latching onto the sidebar)
    monkeypatch.setattr(
        cleaner, "_extract_main_content", lambda _doc: "Berita Lain Opini Kalam"
    )
    docs = cleaner.clean([doc], query="Ainun Najib")
    assert [d.url for d in docs] == ["http://nu.or.id/x"]
    assert "Ainun Najib" in docs[0].text


def test_extractor_miss_without_name_anywhere_still_drops(monkeypatch) -> None:
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    doc = Document(url="http://x", text="<html><body><p>Halaman lain.</p></body></html>")
    monkeypatch.setattr(cleaner, "_extract_main_content", lambda _doc: "Halaman lain.")
    assert cleaner.clean([doc], query="Ainun Najib") == []


# ---- social-media identity pages: no context exemption (strict mode) --------

_BODY_NO_CTX = (
    "Sri Lestari tinggal di kota lain dan berdagang di pasar tradisional. "
    "Sri Lestari memiliki banyak teman di komunitasnya. "
) * 6
_BODY_WITH_CTX = (
    "Sri Lestari bekerja sebagai perawat di Sleman sejak tahun lalu. "
    "Sri Lestari aktif di klinik kesehatan desa. "
) * 6


def test_social_identity_pages_need_context_keyword_in_strict_mode() -> None:
    """facebook/instagram/tiktok slug==name is NOT identity evidence."""
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    social = [
        Document(url="https://facebook.com/sri-lestari", text=_BODY_NO_CTX),
        Document(url="https://m.facebook.com/srilestari", text=_BODY_NO_CTX),
        Document(url="https://instagram.com/sri.lestari", text=_BODY_NO_CTX),
        Document(url="https://tiktok.com/@srilestari", text=_BODY_NO_CTX),
    ]
    with_ctx = Document(
        url="https://facebook.com/pages/sri-lestari-sleman", text=_BODY_WITH_CTX
    )
    docs = cleaner.clean(
        [*social, with_ctx], query="Sri Lestari", context="Sleman, Kesehatan"
    )
    assert [d.url for d in docs] == [with_ctx.url]


def test_social_identity_pages_dropped_without_any_context() -> None:
    """No --context at all: a social slug==name profile can never satisfy it."""
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    social = Document(url="https://facebook.com/sri-lestari", text=_BODY_NO_CTX)
    identity = Document(url="https://medium.com/@srilestari", text=_BODY_NO_CTX)
    docs = cleaner.clean([social, identity], query="Sri Lestari", context="")
    # Non-social identity pages stay exempt; the social profile is dropped.
    assert [d.url for d in docs] == [identity.url]


def test_social_identity_pages_kept_in_loose_mode() -> None:
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5, require_context=False)
    doc = Document(url="https://facebook.com/sri-lestari", text=_BODY_NO_CTX)
    docs = cleaner.clean([doc], query="Sri Lestari", context="Sleman, Kesehatan")
    assert [d.url for d in docs] == [doc.url]


def test_social_profile_with_context_keyword_is_kept_on_any_path() -> None:
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    doc = Document(url="https://instagram.com/srilestari", text=_BODY_WITH_CTX)
    docs = cleaner.clean([doc], query="Sri Lestari", context="Sleman, Kesehatan")
    assert [d.url for d in docs] == [doc.url]


def test_is_social_media_url_matches_subdomains_not_lookalikes() -> None:
    assert is_social_media_url("https://facebook.com/sri-lestari")
    assert is_social_media_url("https://m.facebook.com/srilestari")
    assert is_social_media_url("https://www.instagram.com/sri.lestari")
    assert is_social_media_url("https://tiktok.com/@srilestari")
    assert not is_social_media_url("https://notfacebook.com/x")
    assert not is_social_media_url("https://facebook.com.evil.io/x")
    assert not is_social_media_url("https://github.com/srilestari")
    assert not is_social_media_url("")


def test_strict_gate_logs_global_drop_count(caplog) -> None:
    """The gate reports how many docs it killed, for every path at once."""
    import logging

    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    docs_in = [
        Document(url="http://ctx", text=_BODY_WITH_CTX),
        Document(url="http://other", text=_BODY_NO_CTX),
        Document(url="https://facebook.com/sri-lestari", text=_BODY_NO_CTX),
    ]
    with caplog.at_level(logging.INFO, logger="drx.muscle.cleaner"):
        docs = cleaner.clean(docs_in, query="Sri Lestari", context="Sleman, Kesehatan")
    assert [d.url for d in docs] == ["http://ctx"]
    assert "strict --context gate: 2/3 docs dropped" in caplog.text


# ---- ASN blob windowing: PDF text layer without newlines --------------------

def _asn_blob(rows: int = 40, target_row: int = 41) -> str:
    """One gigantic line (no ``\\n``), as produced by many PDF text layers."""
    cells = ["DAFTAR NAMA PEGAWAI No Nama Lengkap Jabatan Golongan Jenis Kelamin"]
    cells += [
        f"{i} NIP:19850101{i:04d} Nama Umum {i} Golongan III/a Pelaksana Laki-laki"
        for i in range(1, rows + 1)
    ]
    cells.append(
        f"{target_row} NIP:198501010999 Ainun Najib Golongan III/b Analis Perempuan"
    )
    cells += [
        f"{i} NIP:19850101{i:04d} Nama Umum {i} Golongan III/a Pelaksana Laki-laki"
        for i in range(1, rows + 1)
    ]
    return " ".join(cells)


def test_asn_pdf_blob_is_windowed_by_character() -> None:
    """A one-line PDF dump must not survive whole (106k-char Temanggung case)."""
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=50)
    blob = _asn_blob()
    assert "\n" not in blob
    assert blob.count("Golongan") > 10  # noisy dump by construction
    docs = cleaner.clean(
        [Document(url="https://dinkes.example.go.id/pegawai.pdf", text=blob)],
        query="Ainun Najib",
    )
    assert len(docs) == 1
    out = docs[0].text
    assert "Ainun Najib" in out
    assert len(out) < len(blob) // 4  # hanya sekitar baris target yang tersisa
    assert out.count("NIP:") <= 6
    assert "Nama Umum 20 Golongan" not in out  # baris jauh hilang


def test_asn_pdf_blob_without_target_name_is_dropped() -> None:
    """No name anywhere in the blob -> nothing left to contribute."""
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=50)
    docs = cleaner.clean(
        [Document(url="https://dinkes.example.go.id/pegawai.pdf", text=_asn_blob())],
        query="Sri Lestari",
    )
    assert docs == []


def test_asn_blob_windows_merge_for_repeated_name() -> None:
    """The same name twice in adjacent rows yields merged, non-duplicated text."""
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=50)
    row = "NIP:198501010999 Ainun Najib Golongan III/b Analis"
    blob = " ".join([row, row, *[f"NIP:1985010{i:05d} Nama Umum {i} Golongan III/a"
                                for i in range(20)]])
    out = cleaner._window_asn_table(blob, ["Ainun Najib"])
    assert out.count("Ainun Najib") == 2  # both mentions kept, no duplication
    assert out.count("NIP:") <= 6


def test_asn_blob_gate_uses_full_page_but_stores_window() -> None:
    """Context word in the dump header (outside the window) still counts."""
    cleaner = Cleaner(min_text_length=10, bm25_keep_top_k=50)
    rows = " ".join(
        f"{i} NIP:19850101{i:04d} Nama Umum {i} Golongan III/a Pelaksana Laki-laki"
        for i in range(1, 40)
    )
    blob = (
        "DINAS KESEHATAN KABUPATEN SLEMAN Daftar Nama Pegawai "
        + rows
        + " 41 NIP:198501010999 Ainun Najib Golongan III/b Analis Perempuan "
        + rows
    )
    docs = cleaner.clean(
        [Document(url="https://dinkes.example.go.id/pegawai.pdf", text=blob)],
        query="Ainun Najib",
        context="Dinas Kesehatan, Sleman",
    )
    assert len(docs) == 1
    out = docs[0].text
    assert "Ainun Najib" in out  # the target's row survived the windowing
    assert len(out) < len(blob) // 4  # ...without the bulk rows
    assert out.count("NIP:") <= 6
    assert "DINAS KESEHATAN" not in out  # the header itself is not stored


# ---- login/consent wall shells (LinkedIn "join now", cookie banners) --------

_WALL_TEXT = (
    "Sri Lestari - LinkedIn. Join now Sign in. Cookie policy. By clicking "
    "Continue to join LinkedIn, you agree to the User Agreement, Privacy "
    "Policy and Cookie Policy. LinkedIn Corporation. Join now to see who you "
    "already know at Sri Lestari. Skip to main content. Sign in to see who you "
    "know. New to LinkedIn? Sign up. Email or phone Password Forgot password. "
    "By clicking Continue, you agree to LinkedIn's User Agreement, Privacy "
    "Policy and Cookie Policy."
)


def test_login_wall_shell_is_dropped_in_every_mode() -> None:
    """A wall carries the name but no content: not even profile URLs survive."""
    assert _is_login_wall(_WALL_TEXT)
    doc = Document(
        url="https://id.linkedin.com/in/sri-lestari",
        title="Sri Lestari - LinkedIn",
        text=_WALL_TEXT,
    )
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    assert cleaner.clean([doc], query="Sri Lestari", context="Sleman, Kesehatan") == []
    loose = Cleaner(min_text_length=1, bm25_keep_top_k=5, require_context=False)
    assert loose.clean([doc], query="Sri Lestari") == []


def test_real_profile_with_footer_consent_mentions_is_kept() -> None:
    """A long page is never a wall, even with a cookie footer."""
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    body = (
        "Sri Lestari is a public health nurse in Sleman. She leads the maternal "
        "care programme and has published three papers. "
    ) * 20
    footer = " Cookie policy. Sign in. Join now. Privacy policy."
    doc = Document(
        url="https://example.org/sri-lestari", title="Sri Lestari", text=body + footer
    )
    docs = cleaner.clean([doc], query="Sri Lestari", context="Sleman, Kesehatan")
    assert [d.url for d in docs] == [doc.url]


def test_short_page_with_a_few_consent_mentions_is_kept() -> None:
    """Below the sign threshold a short page is still real content."""
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    doc = Document(
        url="https://example.org/sri-lestari-sleman",
        title="Sri Lestari",
        text=(
            "Sri Lestari di Sleman bekerja sebagai bidan di puskesmas. " * 6
            + "Subscribe to our newsletter; cookie policy applies."
        ),
    )
    docs = cleaner.clean([doc], query="Sri Lestari", context="Sleman, Kesehatan")
    assert [d.url for d in docs] == [doc.url]


def test_login_wall_drop_is_reported(caplog) -> None:
    import logging

    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    doc = Document(
        url="https://id.linkedin.com/in/sri-lestari", title="Sri Lestari", text=_WALL_TEXT
    )
    with caplog.at_level(logging.INFO, logger="drx.muscle.cleaner"):
        assert cleaner.clean([doc], query="Sri Lestari") == []
    assert "login/consent walls dropped: 1/1 docs" in caplog.text
