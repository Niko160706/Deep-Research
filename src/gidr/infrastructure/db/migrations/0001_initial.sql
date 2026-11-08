CREATE TABLE artifact_catalog (
    artifact_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL UNIQUE CHECK(length(sha256)=64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
    size_bytes INTEGER NOT NULL CHECK(size_bytes>=0), category TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('AVAILABLE','QUARANTINED','DELETING','DELETED')),
    created_at_utc TEXT NOT NULL, deletion_epoch INTEGER NOT NULL DEFAULT 0 CHECK(deletion_epoch>=0),
    UNIQUE(artifact_id,sha256)
) STRICT;
CREATE TABLE projects (
    project_id TEXT PRIMARY KEY, title TEXT NOT NULL, owner TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0 CHECK(revision>=0), created_at_utc TEXT NOT NULL
) STRICT;
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(project_id),
    revision INTEGER NOT NULL DEFAULT 0 CHECK(revision>=0), created_at_utc TEXT NOT NULL,
    UNIQUE(session_id,project_id)
) STRICT;
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(project_id),
    session_id TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN
      ('CREATED','PLANNING','PLAN_READY','RUNNING','PAUSING','PAUSED','INTERRUPTED',
       'RECOVERING','COMPLETING','COMPLETED','FAILED','CANCELLED')),
    revision INTEGER NOT NULL DEFAULT 0 CHECK(revision>=0),
    active_plan_version_id TEXT REFERENCES plan_versions(plan_version_id) DEFERRABLE INITIALLY DEFERRED,
    created_at_utc TEXT NOT NULL, updated_at_utc TEXT NOT NULL, stop_reason TEXT,
    FOREIGN KEY(session_id,project_id) REFERENCES sessions(session_id,project_id),
    FOREIGN KEY(active_plan_version_id,run_id) REFERENCES plan_versions(plan_version_id,run_id) DEFERRABLE INITIALLY DEFERRED
) STRICT;
CREATE INDEX idx_runs_session_time ON runs(session_id,created_at_utc);
CREATE TABLE active_run_gate (
    singleton INTEGER PRIMARY KEY CHECK(singleton=1), run_id TEXT REFERENCES runs(run_id),
    generation INTEGER NOT NULL DEFAULT 0 CHECK(generation>=0), holder TEXT,
    expires_at_utc TEXT, updated_at_utc TEXT NOT NULL
) STRICT;
INSERT INTO active_run_gate(singleton,generation,updated_at_utc) VALUES(1,0,strftime('%Y-%m-%dT%H:%M:%fZ','now'));
CREATE TABLE plans (
    plan_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), created_at_utc TEXT NOT NULL,
    UNIQUE(plan_id,run_id)
) STRICT;
CREATE TABLE plan_versions (
    plan_version_id TEXT PRIMARY KEY, plan_id TEXT NOT NULL REFERENCES plans(plan_id),
    run_id TEXT NOT NULL REFERENCES runs(run_id), version INTEGER NOT NULL CHECK(version>0),
    parent_version_id TEXT REFERENCES plan_versions(plan_version_id),
    snapshot_artifact_id TEXT NOT NULL REFERENCES artifact_catalog(artifact_id),
    created_at_utc TEXT NOT NULL, change_reason TEXT NOT NULL,
    UNIQUE(run_id,version), UNIQUE(plan_version_id,run_id),
    FOREIGN KEY(plan_id,run_id) REFERENCES plans(plan_id,run_id)
) STRICT;
CREATE TRIGGER immutable_plan_version BEFORE UPDATE ON plan_versions BEGIN
    SELECT RAISE(ABORT,'immutable_plan_version');
END;
CREATE TABLE questions (
    question_id TEXT NOT NULL, plan_version_id TEXT NOT NULL REFERENCES plan_versions(plan_version_id),
    parent_question_id TEXT, methodology TEXT NOT NULL,
    coverage TEXT NOT NULL CHECK(coverage IN ('UNEXPLORED','PARTIAL','SUFFICIENT_FOR_SCOPE','UNRESOLVED')),
    payload_json TEXT NOT NULL CHECK(json_valid(payload_json) AND length(payload_json)<=65536),
    PRIMARY KEY(question_id,plan_version_id)
) STRICT;
CREATE TABLE grants (
    grant_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), plan_version_id TEXT NOT NULL,
    scope_json TEXT NOT NULL CHECK(json_valid(scope_json) AND length(scope_json)<=65536),
    budget_json TEXT NOT NULL CHECK(json_valid(budget_json) AND length(budget_json)<=65536),
    approval_source TEXT NOT NULL CHECK(approval_source IN ('USER_APPROVED','PRIOR_GRANT')),
    expires_at_utc TEXT, created_at_utc TEXT NOT NULL,
    FOREIGN KEY(plan_version_id,run_id) REFERENCES plan_versions(plan_version_id,run_id)
) STRICT;
CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), plan_version_id TEXT NOT NULL,
    type TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN
      ('PENDING','WAITING_DEPENDENCY','READY','RESERVED','RUNNING','RETRY_WAIT','DEFERRED',
       'AWAITING_APPROVAL','BLOCKED','SUCCEEDED','FAILED','CANCELLED')),
    priority REAL NOT NULL DEFAULT 0, revision INTEGER NOT NULL DEFAULT 0 CHECK(revision>=0),
    fingerprint TEXT NOT NULL, not_before_utc TEXT, created_at_utc TEXT NOT NULL, updated_at_utc TEXT NOT NULL,
    FOREIGN KEY(plan_version_id,run_id) REFERENCES plan_versions(plan_version_id,run_id),
    UNIQUE(run_id,plan_version_id,fingerprint)
) STRICT;
CREATE INDEX idx_tasks_dispatch ON tasks(run_id,state,not_before_utc,priority DESC);
CREATE TABLE task_dependencies (
    task_id TEXT NOT NULL REFERENCES tasks(task_id), dependency_task_id TEXT NOT NULL REFERENCES tasks(task_id),
    CHECK(task_id<>dependency_task_id), PRIMARY KEY(task_id,dependency_task_id)
) STRICT;
CREATE TABLE task_attempts (
    attempt_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(task_id),
    attempt_no INTEGER NOT NULL CHECK(attempt_no>0), lease_generation INTEGER NOT NULL CHECK(lease_generation>=0),
    holder TEXT NOT NULL, expires_at_utc TEXT NOT NULL, state TEXT NOT NULL,
    usage_state TEXT NOT NULL DEFAULT 'UNKNOWN_USAGE' CHECK(usage_state IN ('KNOWN','ESTIMATED','UNKNOWN_USAGE')),
    output_artifact_id TEXT REFERENCES artifact_catalog(artifact_id), UNIQUE(task_id,attempt_no)
) STRICT;
CREATE INDEX idx_attempt_expiry ON task_attempts(state,expires_at_utc);
CREATE TABLE budget_ledger (
    entry_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
    task_id TEXT REFERENCES tasks(task_id), attempt_id TEXT REFERENCES task_attempts(attempt_id),
    kind TEXT NOT NULL CHECK(kind IN ('RESERVE','SETTLE','RELEASE','USAGE_UNCERTAIN')),
    amount_usd TEXT, usage_state TEXT NOT NULL CHECK(usage_state IN ('KNOWN','ESTIMATED','UNKNOWN_USAGE')),
    idempotency_key TEXT NOT NULL UNIQUE, created_at_utc TEXT NOT NULL,
    CHECK(amount_usd IS NULL OR (length(amount_usd)>0 AND amount_usd NOT GLOB '*[^0-9.]*'
      AND length(amount_usd)-length(replace(amount_usd,'.',''))<=1 AND amount_usd<>'.'))
) STRICT;
CREATE INDEX idx_budget_run ON budget_ledger(run_id,kind);
CREATE TABLE source_candidates (
    source_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), provider_id TEXT NOT NULL,
    external_id TEXT, canonical_url TEXT NOT NULL, title TEXT, snippet TEXT,
    metadata_artifact_id TEXT REFERENCES artifact_catalog(artifact_id), discovered_at_utc TEXT NOT NULL,
    UNIQUE(provider_id,external_id)
) STRICT;
CREATE INDEX idx_sources_run_provider ON source_candidates(run_id,provider_id);
CREATE TABLE source_observations (
    observation_id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES source_candidates(source_id),
    run_id TEXT NOT NULL REFERENCES runs(run_id), artifact_id TEXT REFERENCES artifact_catalog(artifact_id),
    observed_at_utc TEXT NOT NULL, access_status TEXT NOT NULL, UNIQUE(observation_id,source_id)
) STRICT;
CREATE TABLE source_aliases (
    source_id TEXT NOT NULL REFERENCES source_candidates(source_id), alias_url TEXT NOT NULL,
    PRIMARY KEY(source_id,alias_url)
) STRICT;
CREATE TABLE source_dependencies (
    from_source_id TEXT NOT NULL REFERENCES source_candidates(source_id),
    to_source_id TEXT NOT NULL REFERENCES source_candidates(source_id), kind TEXT NOT NULL,
    checking_method TEXT NOT NULL, certainty TEXT NOT NULL CHECK(certainty IN ('CONFIRMED','LIKELY','UNKNOWN')),
    CHECK(from_source_id<>to_source_id), PRIMARY KEY(from_source_id,to_source_id,kind)
) STRICT;
CREATE TABLE leads (
    lead_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), question_id TEXT,
    source_id TEXT REFERENCES source_candidates(source_id),
    state TEXT NOT NULL CHECK(state IN ('DISCOVERED','TRIAGED','QUEUED','DEFERRED','AWAITING_APPROVAL',
      'REJECTED_WITH_REASON','INVESTIGATING','RESOLVED','BLOCKED')),
    validation_label TEXT NOT NULL DEFAULT 'UNVERIFIED_LEAD',
    revision INTEGER NOT NULL DEFAULT 0 CHECK(revision>=0), reason_code TEXT NOT NULL
) STRICT;
CREATE TABLE lead_edges (
    parent_lead_id TEXT NOT NULL REFERENCES leads(lead_id), child_lead_id TEXT NOT NULL REFERENCES leads(lead_id),
    reason_code TEXT NOT NULL, PRIMARY KEY(parent_lead_id,child_lead_id)
) STRICT;
CREATE TABLE frontier_scores (
    lead_id TEXT NOT NULL REFERENCES leads(lead_id), rank_version INTEGER NOT NULL CHECK(rank_version>0),
    score REAL, reason_code TEXT NOT NULL, created_at_utc TEXT NOT NULL, PRIMARY KEY(lead_id,rank_version)
) STRICT;
CREATE INDEX idx_leads_run_state ON leads(run_id,state);
CREATE TABLE documents (
    document_id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES source_candidates(source_id),
    observation_id TEXT NOT NULL REFERENCES source_observations(observation_id),
    original_artifact_id TEXT REFERENCES artifact_catalog(artifact_id), mime_type TEXT, sha256 TEXT,
    retrieval_status TEXT NOT NULL CHECK(retrieval_status IN ('RETRIEVED','FAILED','BLOCKED')),
    availability TEXT NOT NULL CHECK(availability IN ('AVAILABLE','MISSING','QUARANTINED','DELETED')),
    fetched_at_utc TEXT NOT NULL,
    CHECK(retrieval_status<>'RETRIEVED' OR (original_artifact_id IS NOT NULL AND sha256 IS NOT NULL AND mime_type IS NOT NULL)),
    CHECK(retrieval_status='RETRIEVED' OR (original_artifact_id IS NULL AND sha256 IS NULL)),
    FOREIGN KEY(original_artifact_id,sha256) REFERENCES artifact_catalog(artifact_id,sha256),
    FOREIGN KEY(observation_id,source_id) REFERENCES source_observations(observation_id,source_id)
) STRICT;
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(document_id),
    text_artifact_id TEXT NOT NULL REFERENCES artifact_catalog(artifact_id),
    locator_json TEXT NOT NULL CHECK(json_valid(locator_json) AND length(locator_json)<=65536),
    language TEXT NOT NULL, parser_id TEXT NOT NULL, parser_version TEXT NOT NULL, quality REAL
) STRICT;
CREATE INDEX idx_chunks_document ON chunks(document_id,chunk_id);
CREATE TABLE translations (
    translation_id TEXT PRIMARY KEY, chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id),
    translated_artifact_id TEXT NOT NULL REFERENCES artifact_catalog(artifact_id),
    source_language TEXT NOT NULL, target_language TEXT NOT NULL, transformer_version TEXT NOT NULL,
    risks_json TEXT NOT NULL CHECK(json_valid(risks_json) AND length(risks_json)<=65536)
) STRICT;
CREATE TABLE claims (
    claim_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
    proposition_artifact_id TEXT NOT NULL REFERENCES artifact_catalog(artifact_id), claim_type TEXT NOT NULL,
    scope_json TEXT NOT NULL CHECK(json_valid(scope_json) AND length(scope_json)<=65536)
) STRICT;
CREATE TABLE claim_relations (
    from_claim_id TEXT NOT NULL REFERENCES claims(claim_id), to_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    relation TEXT NOT NULL, PRIMARY KEY(from_claim_id,to_claim_id,relation)
) STRICT;
CREATE TABLE evidence (
    evidence_id TEXT PRIMARY KEY, claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id), excerpt_artifact_id TEXT NOT NULL REFERENCES artifact_catalog(artifact_id),
    locator_json TEXT NOT NULL CHECK(json_valid(locator_json) AND length(locator_json)<=65536),
    relation TEXT NOT NULL CHECK(relation IN ('SUPPORTS','REFUTES','INSUFFICIENT','CONTEXTUALIZES')),
    validation_status TEXT NOT NULL CHECK(validation_status IN ('UNCHECKED','MATCH','MISMATCH','UNAVAILABLE'))
) STRICT;
CREATE INDEX idx_evidence_claim ON evidence(claim_id);
CREATE TABLE assessments (
    assessment_id TEXT PRIMARY KEY, claim_id TEXT NOT NULL REFERENCES claims(claim_id), version INTEGER NOT NULL CHECK(version>0),
    contract_id TEXT NOT NULL, contract_version TEXT NOT NULL,
    lifecycle TEXT NOT NULL CHECK(lifecycle IN ('DRAFT','VALIDATING','COMMITTED','REJECTED_NEEDS_EVIDENCE','CONFLICT_REVIEW')),
    verdict TEXT NOT NULL CHECK(verdict IN ('UNASSESSED','INSUFFICIENT','SUPPORTED','CONTESTED','REFUTED','CONTEXT_DEPENDENT')),
    snapshot_artifact_id TEXT NOT NULL REFERENCES artifact_catalog(artifact_id),
    predecessor_id TEXT REFERENCES assessments(assessment_id), created_at_utc TEXT NOT NULL, UNIQUE(claim_id,version)
) STRICT;
CREATE TRIGGER immutable_committed_assessment BEFORE UPDATE ON assessments WHEN OLD.lifecycle='COMMITTED' BEGIN
    SELECT RAISE(ABORT,'immutable_committed_assessment');
END;
CREATE TABLE assessment_evidence (
    assessment_id TEXT NOT NULL REFERENCES assessments(assessment_id), evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id),
    PRIMARY KEY(assessment_id,evidence_id)
) STRICT;
CREATE TABLE critic_proposals (
    proposal_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), fingerprint TEXT NOT NULL,
    material_version TEXT NOT NULL, snapshot_artifact_id TEXT NOT NULL REFERENCES artifact_catalog(artifact_id),
    status TEXT NOT NULL CHECK(status IN ('PROPOSED','VALIDATED','AUTO_AUTHORIZED','AWAITING_APPROVAL',
      'REJECTED','DUPLICATE_SUPPRESSED','SCHEDULED','APPLIED','FAILED_APPLICATION','USER_APPROVED','USER_REJECTED','EXPIRED')),
    approval_grant_id TEXT REFERENCES grants(grant_id), UNIQUE(run_id,fingerprint,material_version)
) STRICT;
CREATE TABLE plan_change_proposals (
    proposal_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), expected_version INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('PROPOSED','REVIEWED','APPROVED','REJECTED')),
    snapshot_artifact_id TEXT NOT NULL REFERENCES artifact_catalog(artifact_id)
) STRICT;
CREATE TABLE model_executions (
    model_execution_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(task_id),
    provider_id TEXT NOT NULL, model_id TEXT, model_version TEXT, prompt_version TEXT NOT NULL,
    input_artifact_id TEXT REFERENCES artifact_catalog(artifact_id), output_artifact_id TEXT REFERENCES artifact_catalog(artifact_id),
    usage_state TEXT NOT NULL CHECK(usage_state IN ('KNOWN','ESTIMATED','UNKNOWN_USAGE')),
    cost_state TEXT NOT NULL CHECK(cost_state IN ('KNOWN','ESTIMATED','UNKNOWN')), cost_usd TEXT,
    error_code TEXT, CHECK(cost_state<>'UNKNOWN' OR cost_usd IS NULL)
) STRICT;
CREATE INDEX idx_models_provider ON model_executions(provider_id,task_id);
CREATE TABLE model_attempts (
    model_attempt_id TEXT PRIMARY KEY, model_execution_id TEXT NOT NULL REFERENCES model_executions(model_execution_id),
    attempt_no INTEGER NOT NULL CHECK(attempt_no>0), idempotency_key TEXT NOT NULL UNIQUE,
    output_artifact_id TEXT REFERENCES artifact_catalog(artifact_id), error_code TEXT,
    UNIQUE(model_execution_id,attempt_no)
) STRICT;
CREATE TABLE provider_health (
    adapter_id TEXT PRIMARY KEY, adapter_version TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN
      ('NOT_IMPLEMENTED','MOCK_TESTED','LIVE_VERIFIED','DEGRADED','BLOCKED','DISABLED')),
    checked_at_utc TEXT, verification_artifact_id TEXT REFERENCES artifact_catalog(artifact_id), reason_code TEXT,
    CHECK(status<>'LIVE_VERIFIED' OR (checked_at_utc IS NOT NULL AND verification_artifact_id IS NOT NULL))
) STRICT;
CREATE TABLE provider_cooldowns (
    adapter_id TEXT PRIMARY KEY REFERENCES provider_health(adapter_id), until_utc TEXT NOT NULL, reason_code TEXT NOT NULL
) STRICT;
CREATE TABLE artifact_refs (
    artifact_id TEXT NOT NULL REFERENCES artifact_catalog(artifact_id), owner_type TEXT NOT NULL, owner_id TEXT NOT NULL,
    role TEXT NOT NULL, created_at_utc TEXT NOT NULL, PRIMARY KEY(artifact_id,owner_type,owner_id,role)
) STRICT;
CREATE INDEX idx_artifact_owner ON artifact_refs(owner_type,owner_id);
CREATE TABLE artifact_tombstones (
    artifact_id TEXT NOT NULL REFERENCES artifact_catalog(artifact_id), deletion_epoch INTEGER NOT NULL,
    reason_code TEXT NOT NULL, created_at_utc TEXT NOT NULL, PRIMARY KEY(artifact_id,deletion_epoch)
) STRICT;
CREATE TABLE processed_commands (
    command_id TEXT PRIMARY KEY, request_sha256 TEXT NOT NULL CHECK(length(request_sha256)=64),
    result_type TEXT NOT NULL, result_json TEXT NOT NULL CHECK(json_valid(result_json) AND length(result_json)<=131072),
    committed_at_utc TEXT NOT NULL
) STRICT;
CREATE TRIGGER immutable_processed_command BEFORE UPDATE ON processed_commands BEGIN
    SELECT RAISE(ABORT,'immutable_processed_command');
END;
CREATE TRIGGER retained_processed_command BEFORE DELETE ON processed_commands BEGIN
    SELECT RAISE(ABORT,'retained_processed_command');
END;
CREATE TABLE audit_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE,
    command_id TEXT NOT NULL REFERENCES processed_commands(command_id) DEFERRABLE INITIALLY DEFERRED,
    run_id TEXT REFERENCES runs(run_id), trace_id TEXT NOT NULL, aggregate_type TEXT NOT NULL,
    aggregate_id TEXT NOT NULL, revision_before INTEGER, revision_after INTEGER, component TEXT NOT NULL,
    event_type TEXT NOT NULL, event_payload_json TEXT NOT NULL CHECK(json_valid(event_payload_json) AND length(event_payload_json)<=65536),
    committed_at_utc TEXT NOT NULL
) STRICT;
CREATE INDEX idx_events_run_seq ON audit_events(run_id,seq);
CREATE INDEX idx_events_trace_seq ON audit_events(trace_id,seq);
CREATE INDEX idx_events_command ON audit_events(command_id);
CREATE TRIGGER immutable_audit_event BEFORE UPDATE ON audit_events BEGIN
    SELECT RAISE(ABORT,'immutable_audit_event');
END;
CREATE TRIGGER retained_audit_event BEFORE DELETE ON audit_events BEGIN
    SELECT RAISE(ABORT,'retained_audit_event');
END;
CREATE TABLE coverage_observations (
    coverage_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), question_id TEXT,
    provider_id TEXT NOT NULL, status TEXT NOT NULL, counts_json TEXT NOT NULL CHECK(json_valid(counts_json) AND length(counts_json)<=65536),
    created_at_utc TEXT NOT NULL
) STRICT;
CREATE TABLE report_snapshots (
    report_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), version INTEGER NOT NULL CHECK(version>0),
    snapshot_artifact_id TEXT NOT NULL REFERENCES artifact_catalog(artifact_id), created_at_utc TEXT NOT NULL, UNIQUE(run_id,version)
) STRICT;
CREATE TABLE runtime_lock_observations (
    observation_id TEXT PRIMARY KEY, process_id INTEGER NOT NULL, startup_nonce TEXT NOT NULL,
    heartbeat_at_utc TEXT NOT NULL, prior_state_json TEXT NOT NULL CHECK(json_valid(prior_state_json) AND length(prior_state_json)<=65536)
) STRICT;
CREATE TABLE query_cache (
    query_hash TEXT NOT NULL, provider_id TEXT NOT NULL, result_artifact_id TEXT NOT NULL REFERENCES artifact_catalog(artifact_id),
    expires_at_utc TEXT NOT NULL, rights_policy TEXT NOT NULL, PRIMARY KEY(query_hash,provider_id)
) STRICT;
CREATE TABLE dedup_fingerprints (
    fingerprint TEXT NOT NULL, scope TEXT NOT NULL, source_id TEXT NOT NULL REFERENCES source_candidates(source_id),
    PRIMARY KEY(fingerprint,scope,source_id)
) STRICT;
