ALTER TABLE model_executions ADD COLUMN parent_attempt_id TEXT REFERENCES task_attempts(attempt_id);
ALTER TABLE model_executions ADD COLUMN result_json TEXT CHECK(result_json IS NULL OR json_valid(result_json));
CREATE UNIQUE INDEX one_model_execution_per_parent_attempt
    ON model_executions(parent_attempt_id) WHERE parent_attempt_id IS NOT NULL;
CREATE TABLE model_attempt_controls (
    model_attempt_id TEXT PRIMARY KEY REFERENCES model_attempts(model_attempt_id),
    parent_attempt_id TEXT NOT NULL REFERENCES task_attempts(attempt_id),
    provider_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('RESERVED','DISPATCHED','SUCCEEDED','FAILED')),
    permit_sha256 TEXT NOT NULL CHECK(length(permit_sha256)=64 AND permit_sha256 NOT GLOB '*[^0-9a-f]*'),
    input_sha256 TEXT NOT NULL CHECK(length(input_sha256)=64 AND input_sha256 NOT GLOB '*[^0-9a-f]*'),
    reserved_cost_usd TEXT NOT NULL,
    reserved_tokens INTEGER NOT NULL CHECK(reserved_tokens>=0),
    reserved_requests INTEGER NOT NULL CHECK(reserved_requests>=0),
    input_tokens INTEGER CHECK(input_tokens>=0),
    output_tokens INTEGER CHECK(output_tokens>=0),
    actual_cost_usd TEXT,
    usage_state TEXT NOT NULL CHECK(usage_state IN ('KNOWN','ESTIMATED','UNKNOWN_USAGE')),
    error_code TEXT,
    response_status INTEGER CHECK(response_status>=0),
    response_sha256 TEXT CHECK(response_sha256 IS NULL OR (length(response_sha256)=64 AND response_sha256 NOT GLOB '*[^0-9a-f]*')),
    dispatch_url_sha256 TEXT CHECK(dispatch_url_sha256 IS NULL OR (length(dispatch_url_sha256)=64 AND dispatch_url_sha256 NOT GLOB '*[^0-9a-f]*')),
    dispatch_body_sha256 TEXT CHECK(dispatch_body_sha256 IS NULL OR (length(dispatch_body_sha256)=64 AND dispatch_body_sha256 NOT GLOB '*[^0-9a-f]*')),
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    UNIQUE(parent_attempt_id,model_attempt_id)
) STRICT;
CREATE INDEX idx_model_attempt_controls_parent
    ON model_attempt_controls(parent_attempt_id,state);
