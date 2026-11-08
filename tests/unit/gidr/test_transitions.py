"""Normative SRS edge cases, terminal protection and distinct vocabularies."""

import pytest

from gidr.domain.states import (
    AssessmentState,
    LeadState,
    PlanChangeState,
    ProposalState,
    RunState,
    TaskState,
    Verdict,
)
from gidr.domain.transitions import GRAPHS, allowed


@pytest.mark.parametrize(
    ("kind", "before", "after"),
    [
        ("RUN", "PAUSED", "RECOVERING"),
        ("RUN", "RECOVERING", "PAUSED"),
        ("RUN", "COMPLETING", "PAUSED"),
        ("TASK", "RESERVED", "READY"),
        ("TASK", "RUNNING", "RETRY_WAIT"),
        ("TASK", "BLOCKED", "READY"),
        ("LEAD", "DEFERRED", "QUEUED"),
        ("PROPOSAL", "AWAITING_APPROVAL", "USER_APPROVED"),
        ("PROPOSAL", "USER_APPROVED", "SCHEDULED"),
        ("ASSESSMENT", "CONFLICT_REVIEW", "VALIDATING"),
        ("PLAN_CHANGE", "REVIEWED", "APPROVED"),
    ],
)
def test_srs_recovery_and_approval_edges(kind, before, after):
    assert allowed(kind, before, after)


@pytest.mark.parametrize(
    ("kind", "before", "after"),
    [
        ("RUN", "CREATED", "RUNNING"),
        ("RUN", "PAUSING", "RUNNING"),
        ("RUN", "PAUSED", "RUNNING"),
        ("RUN", "FAILED", "RUNNING"),
        ("RUN", "COMPLETING", "INTERRUPTED"),
        ("RUN", "COMPLETED", "PLANNING"),
        ("TASK", "READY", "SUCCEEDED"),
        ("TASK", "RUNNING", "READY"),
        ("TASK", "SUCCEEDED", "READY"),
        ("LEAD", "DISCOVERED", "RESOLVED"),
        ("PROPOSAL", "PROPOSED", "APPLIED"),
        ("ASSESSMENT", "DRAFT", "COMMITTED"),
        ("ASSESSMENT", "COMMITTED", "VALIDATING"),
        ("PLAN_CHANGE", "PROPOSED", "APPROVED"),
    ],
)
def test_srs_prohibits_skips_and_terminal_resurrection(kind, before, after):
    assert not allowed(kind, before, after)


def test_graphs_cover_exact_srs_enums_and_never_confuse_verdicts():
    for name, enum in [
        ("RUN", RunState),
        ("TASK", TaskState),
        ("LEAD", LeadState),
        ("PROPOSAL", ProposalState),
        ("ASSESSMENT", AssessmentState),
        ("PLAN_CHANGE", PlanChangeState),
    ]:
        assert set(GRAPHS[name]) == {e.value for e in enum}
        assert all(set(v) <= set(GRAPHS[name]) for v in GRAPHS[name].values())
        assert "SUPPORTED" not in GRAPHS[name]
    assert Verdict.SUPPORTED.value == "SUPPORTED"
