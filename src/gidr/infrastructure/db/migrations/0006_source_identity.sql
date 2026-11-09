CREATE TABLE source_identities (
    identity_type TEXT NOT NULL CHECK(identity_type IN ('DOI','URL','WAYBACK_SNAPSHOT')),
    identity_value TEXT NOT NULL CHECK(length(identity_value) BETWEEN 1 AND 4096),
    source_id TEXT NOT NULL REFERENCES source_candidates(source_id),
    created_at_utc TEXT NOT NULL,
    PRIMARY KEY(identity_type,identity_value,source_id)
) STRICT;
CREATE INDEX idx_source_identity_lookup
    ON source_identities(identity_type,identity_value);
