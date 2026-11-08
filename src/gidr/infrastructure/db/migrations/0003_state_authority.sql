ALTER TABLE grants ADD COLUMN revision INTEGER NOT NULL DEFAULT 0 CHECK(revision>=0);
ALTER TABLE grants ADD COLUMN revoked INTEGER NOT NULL DEFAULT 0 CHECK(revoked IN (0,1));
ALTER TABLE tasks ADD COLUMN policy_json TEXT CHECK(policy_json IS NULL OR json_valid(policy_json));
ALTER TABLE task_attempts ADD COLUMN gate_generation INTEGER NOT NULL DEFAULT 0 CHECK(gate_generation>=0);
ALTER TABLE task_attempts ADD COLUMN plan_version_id TEXT REFERENCES plan_versions(plan_version_id);
ALTER TABLE task_attempts ADD COLUMN grant_id TEXT REFERENCES grants(grant_id);
ALTER TABLE task_attempts ADD COLUMN revision INTEGER NOT NULL DEFAULT 0 CHECK(revision>=0);
CREATE UNIQUE INDEX one_unfinished_attempt ON task_attempts(task_id) WHERE state IN ('RESERVED','RUNNING');
CREATE TABLE attempt_accounts (
    attempt_id TEXT PRIMARY KEY REFERENCES task_attempts(attempt_id),
    status TEXT NOT NULL CHECK(status IN ('HELD','KNOWN','UNCERTAIN','RELEASED')),
    max_cost_usd TEXT NOT NULL, max_tokens INTEGER NOT NULL CHECK(max_tokens>=0),
    max_requests INTEGER NOT NULL CHECK(max_requests>=0),
    cost_usd TEXT, tokens INTEGER CHECK(tokens>=0), requests INTEGER CHECK(requests>=0),
    CHECK(status<>'KNOWN' OR (cost_usd IS NOT NULL AND tokens IS NOT NULL AND requests IS NOT NULL))
) STRICT;
CREATE TABLE run_checkpoints (
    checkpoint_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
    revision INTEGER NOT NULL, artifact_id TEXT NOT NULL REFERENCES artifact_catalog(artifact_id),
    created_at_utc TEXT NOT NULL
) STRICT;
CREATE TABLE plan_control (
    plan_version_id TEXT PRIMARY KEY REFERENCES plan_versions(plan_version_id),
    categories_json TEXT NOT NULL CHECK(json_valid(categories_json))
) STRICT;
CREATE TABLE state_command_keys (
    run_id TEXT NOT NULL REFERENCES runs(run_id), idempotency_key TEXT NOT NULL,
    request_sha256 TEXT NOT NULL, result_json TEXT NOT NULL CHECK(json_valid(result_json)),
    command_id TEXT NOT NULL REFERENCES processed_commands(command_id) DEFERRABLE INITIALLY DEFERRED,
    PRIMARY KEY(run_id,idempotency_key)
) STRICT;
CREATE TABLE navigation_state (
    singleton INTEGER PRIMARY KEY CHECK(singleton=1), session_id TEXT REFERENCES sessions(session_id),
    pending_session_id TEXT REFERENCES sessions(session_id)
) STRICT;
INSERT INTO navigation_state(singleton) VALUES(1);
ALTER TABLE leads ADD COLUMN plan_version_id TEXT REFERENCES plan_versions(plan_version_id);
ALTER TABLE critic_proposals ADD COLUMN revision INTEGER NOT NULL DEFAULT 0 CHECK(revision>=0);
ALTER TABLE critic_proposals ADD COLUMN plan_version_id TEXT REFERENCES plan_versions(plan_version_id);
ALTER TABLE plan_change_proposals ADD COLUMN revision INTEGER NOT NULL DEFAULT 0 CHECK(revision>=0);
ALTER TABLE assessments ADD COLUMN revision INTEGER NOT NULL DEFAULT 0 CHECK(revision>=0);
ALTER TABLE assessments ADD COLUMN validation_attempts INTEGER NOT NULL DEFAULT 0 CHECK(validation_attempts>=0);
CREATE INDEX idx_grants_run_validity ON grants(run_id,plan_version_id,revoked,expires_at_utc);
CREATE INDEX idx_gate_attempts ON task_attempts(gate_generation,state);
CREATE UNIQUE INDEX one_global_active_run ON runs((1)) WHERE state IN ('RUNNING','PAUSING','RECOVERING','COMPLETING');
CREATE TRIGGER one_active_run_insert BEFORE INSERT ON runs WHEN NEW.state IN ('RUNNING','PAUSING','RECOVERING','COMPLETING') AND EXISTS(SELECT 1 FROM runs WHERE state IN ('RUNNING','PAUSING','RECOVERING','COMPLETING')) BEGIN
    SELECT RAISE(ABORT,'single_active_run');
END;
CREATE TRIGGER one_active_run_update BEFORE UPDATE OF state ON runs WHEN NEW.state IN ('RUNNING','PAUSING','RECOVERING','COMPLETING') AND EXISTS(SELECT 1 FROM runs WHERE run_id<>NEW.run_id AND state IN ('RUNNING','PAUSING','RECOVERING','COMPLETING')) BEGIN
    SELECT RAISE(ABORT,'single_active_run');
END;
