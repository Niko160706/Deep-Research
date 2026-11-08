"""Exact Decimal accounting; uncertainty retains the upper reservation."""

from decimal import Decimal

from gidr.contracts.commands import GrantPolicy, ScheduledTask, Usage
from gidr.infrastructure.db.authority import WriteContext


def total(context: WriteContext, run_id: str) -> tuple[Decimal, int, int]:
    cost = Decimal(0)
    tokens = requests = 0
    rows = context.execute(
        "SELECT a.* FROM attempt_accounts a JOIN task_attempts t USING(attempt_id) JOIN tasks k USING(task_id) WHERE k.run_id=? AND a.status<>'RELEASED'",
        (run_id,),
    )
    for row in rows:
        known = row["status"] == "KNOWN"
        cost += (
            Decimal(row["cost_usd"])
            if known
            else max(Decimal(row["max_cost_usd"]), Decimal(row["cost_usd"] or "0"))
        )
        tokens += (
            int(row["tokens"]) if known else max(int(row["max_tokens"]), int(row["tokens"] or 0))
        )
        requests += (
            int(row["requests"])
            if known
            else max(int(row["max_requests"]), int(row["requests"] or 0))
        )
    return cost, tokens, requests


def fits(
    context: WriteContext, run_id: str, grant: GrantPolicy, task: ScheduledTask | None = None
) -> bool:
    cost, tokens, requests = total(context, run_id)
    if task is not None:
        cost += task.max_cost_usd
        tokens += task.max_tokens
        requests += task.max_requests
    return (
        cost <= grant.max_cost_usd and tokens <= grant.max_tokens and requests <= grant.max_requests
    )


def ledger(
    context: WriteContext,
    run_id: str,
    task_id: str,
    attempt_id: str,
    kind: str,
    amount: Decimal | None,
    usage: str,
    stamp: str,
) -> None:
    from gidr.contracts.common import new_id

    key = context.command_id + ":" + attempt_id + ":" + kind
    context.execute(
        "INSERT INTO budget_ledger VALUES(?,?,?,?,?,?,?,?,?)",
        (
            new_id("EVT"),
            run_id,
            task_id,
            attempt_id,
            kind,
            None if amount is None else str(amount),
            usage,
            key,
            stamp,
        ),
    )


def settle(
    context: WriteContext, run_id: str, task_id: str, attempt_id: str, usage: Usage, stamp: str
) -> None:
    known = usage.state == "KNOWN"
    context.execute(
        "UPDATE attempt_accounts SET status=?,cost_usd=?,tokens=?,requests=? WHERE attempt_id=?",
        (
            "KNOWN" if known else "UNCERTAIN",
            None if usage.cost_usd is None else str(usage.cost_usd),
            usage.tokens,
            usage.requests,
            attempt_id,
        ),
    )
    context.execute(
        "UPDATE task_attempts SET usage_state=? WHERE attempt_id=?", (usage.state, attempt_id)
    )
    ledger(
        context,
        run_id,
        task_id,
        attempt_id,
        "SETTLE" if known else "USAGE_UNCERTAIN",
        usage.cost_usd if known else None,
        usage.state,
        stamp,
    )
