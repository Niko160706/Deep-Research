-- External-content FTS is maintained explicitly in the same authority transaction.
-- No FTS trigger requires trusted_schema=ON.
CREATE TABLE source_index (
    rowid INTEGER PRIMARY KEY, source_id TEXT NOT NULL REFERENCES source_candidates(source_id),
    chunk_id TEXT REFERENCES chunks(chunk_id), title TEXT NOT NULL, snippet TEXT NOT NULL,
    extracted_text TEXT NOT NULL CHECK(length(extracted_text)<=262144),
    chunk_key TEXT GENERATED ALWAYS AS (coalesce(chunk_id,'')) STORED,
    UNIQUE(source_id,chunk_key)
) STRICT;
CREATE VIRTUAL TABLE source_fts USING fts5(
    title, snippet, extracted_text, content='source_index', content_rowid='rowid', tokenize='unicode61'
);
CREATE TABLE projection_cursors (
    projection_id TEXT PRIMARY KEY, last_seq INTEGER NOT NULL CHECK(last_seq>=0),
    generation TEXT NOT NULL, manifest_sha256 TEXT NOT NULL CHECK(length(manifest_sha256)=64), updated_at_utc TEXT NOT NULL
) STRICT;
