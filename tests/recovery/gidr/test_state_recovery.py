"""SIGKILL recovery cannot auto-resume research or refund uncertain usage."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gidr.application.commands import StateAuthority
from gidr.contracts import commands as dto

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX SIGKILL only; Windows unverified"
)
WORKER = Path(__file__).with_name("state_crash_worker.py")


def support():
    path = Path(__file__).resolve().parents[2] / "integration/gidr/state_support.py"
    spec = importlib.util.spec_from_file_location("state_recovery_parent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launch(root, stage):
    result = subprocess.run(
        [sys.executable, "-I", str(WORKER), str(root), stage],
        cwd=root,
        env={"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == -9, (result.returncode, result.stdout, result.stderr)


@pytest.mark.parametrize(
    "stage",
    [
        "reserved",
        "issued",
        "result_before_commit",
        "result_after_commit",
        "pause_after_stop",
        "checkpoint_published",
    ],
)
def test_exclusive_restart_reconciles_to_paused_with_no_duplicate_success(tmp_path, stage):
    launch(tmp_path, stage)
    h = support().Harness(tmp_path)
    token = dto.AttemptToken.model_validate_json((tmp_path / "token.json").read_text())
    with h.authority:
        assert h.row("runs", "run_id", "RUN-one")["state"] == "PAUSED"
        with h.db.reader() as conn:
            assert conn.execute("SELECT run_id FROM active_run_gate").fetchone()[0] is None
            assert conn.execute("SELECT count(*) FROM run_checkpoints").fetchone()[0] >= 1
            h.db.check(conn)
        task = h.row("tasks", "task_id", "TASK-one")
        account = h.row("attempt_accounts", "attempt_id", "ATT-one")
        if stage == "reserved":
            assert task["state"] == "READY" and account["status"] == "RELEASED"
        elif stage in {"issued", "result_before_commit"}:
            assert (
                task["state"] == "RETRY_WAIT"
                and account["status"] == "UNCERTAIN"
                and account["cost_usd"] is None
            )
        else:
            assert task["state"] == "SUCCEEDED"
        if stage == "result_after_commit":
            command = dto.StateCommand.model_validate_json(
                (tmp_path / "result-command.json").read_text()
            )
            assert h.authority.port("WORKER", "worker").handle(command).accepted
        if stage == "result_before_commit":
            assert h.repo.inventory()["orphans"]  # Published result has no fabricated success.
        if stage in {"reserved", "issued", "result_before_commit"}:
            assert not h.send(dto.StartAttempt(token=token), actor="WORKER").accepted
        with h.db.reader() as conn:
            assert (
                conn.execute(
                    "SELECT count(*) FROM audit_events WHERE event_type='finish_attempt.committed'"
                ).fetchone()[0]
                <= 1
            )
    # Reopening a paused session does not create another recovery or resume.
    with StateAuthority(h.storage, h.repo, clock=h.clock):
        assert h.row("runs", "run_id", "RUN-one")["state"] == "PAUSED"


def test_startup_checkpoint_failure_retains_gate_until_recovery_retries(tmp_path, monkeypatch):
    launch(tmp_path, "issued")
    h = support().Harness(tmp_path)

    def fail(_):
        raise OSError("Synthetic disk failure")

    with monkeypatch.context() as patch:
        patch.setattr(h.store, "publish", fail)
        with pytest.raises(OSError):
            with h.authority:
                pass
    assert h.row("runs", "run_id", "RUN-one")["state"] == "RECOVERING"
    with h.db.reader() as conn:
        assert conn.execute("SELECT run_id FROM active_run_gate").fetchone()[0] == "RUN-one"
    with StateAuthority(h.storage, h.repo, clock=h.clock):
        assert h.row("runs", "run_id", "RUN-one")["state"] == "PAUSED"
        assert h.row("attempt_accounts", "attempt_id", "ATT-one")["status"] == "UNCERTAIN"
