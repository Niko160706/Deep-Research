"""Strict control boundary tests, without persistence or live providers."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from gidr.contracts import commands as dto


@pytest.mark.parametrize(
    "values",
    [
        {"state": "KNOWN"},
        {"state": "UNKNOWN_USAGE", "cost_usd": Decimal(0)},
        {"state": "ESTIMATED"},
        {"state": "KNOWN", "cost_usd": Decimal(-1), "tokens": 0, "requests": 0},
        {"state": "KNOWN", "cost_usd": Decimal("NaN"), "tokens": 0, "requests": 0},
        {"state": "KNOWN", "cost_usd": Decimal("1e50"), "tokens": 0, "requests": 0},
        {"state": "KNOWN", "cost_usd": Decimal(0), "tokens": -1, "requests": 0},
    ],
)
def test_usage_never_invents_unknown_zero_or_accepts_unbounded_values(values):
    with pytest.raises(ValidationError):
        dto.Usage(**values)


@pytest.mark.parametrize("ttl", [0, -1, 3601, True])
def test_attempt_ttl_is_bounded_and_strict(ttl):
    with pytest.raises(ValidationError):
        dto.ReserveTask(
            task_id="TASK-fixture",
            attempt_id="ATT-fixture",
            worker_id="worker",
            lease_ttl_seconds=ttl,
        )


def test_success_requires_result_and_fencing_has_valid_identity():
    token = dto.AttemptToken(
        task_id="TASK-fixture",
        attempt_id="ATT-fixture",
        holder="worker",
        generation=1,
        gate_generation=1,
    )
    with pytest.raises(ValidationError):
        dto.FinishAttempt(
            token=token, target="SUCCEEDED", usage=dto.Usage(state="UNKNOWN_USAGE"), reason="TEST"
        )
    with pytest.raises(ValidationError):
        dto.AttemptToken(
            task_id="TASK-fixture",
            attempt_id="ATT-fixture",
            holder="worker",
            generation=0,
            gate_generation=1,
        )
