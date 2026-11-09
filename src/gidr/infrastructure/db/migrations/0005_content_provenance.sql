ALTER TABLE documents ADD COLUMN language TEXT NOT NULL DEFAULT 'und'
    CHECK(length(language)<=32 AND length(language)>0);
ALTER TABLE chunks ADD COLUMN quality_flags_json TEXT NOT NULL DEFAULT '[]'
    CHECK(json_valid(quality_flags_json) AND length(quality_flags_json)<=65536);
