"""Real process crash against synthetic local B03 state, never an API provider."""

import importlib.util
import os
import signal
import sys
from pathlib import Path

from gidr.contracts import commands as dto


def support():
    path = Path(__file__).resolve().parents[2] / "integration/gidr/state_support.py"
    spec = importlib.util.spec_from_file_location("state_recovery_support", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kill():
    os.kill(os.getpid(), signal.SIGKILL)


def main():
    root, stage = Path(sys.argv[1]), sys.argv[2]
    h = support().Harness(root)
    with h.authority:
        h.setup()
        h.add(h.task())
        assert h.ready().accepted
        if stage == "reserved":
            result = h.reserve()
            assert result.accepted
            (root / "token.json").write_text(result.token.model_dump_json())
            kill()
        token = h.running()
        (root / "token.json").write_text(token.model_dump_json())
        if stage == "issued":
            kill()
        if stage in {"result_before_commit", "result_after_commit"}:
            ref = h.store.publish(b"Synthetic result at crash boundary")
            payload = dto.FinishAttempt(
                token=token,
                target="SUCCEEDED",
                usage=dto.Usage(state="UNKNOWN_USAGE"),
                reason="SYNTHETIC_VALIDATED",
                result_ref=ref,
            )
            command = h.command(payload, actor="WORKER", command_id="CMD-crash-result")
            (root / "result-command.json").write_text(command.model_dump_json())
            boundary = stage.removeprefix("result_")
            h.authority.handle(
                command,
                actor="WORKER",
                worker_id="worker",
                fault=lambda at: kill() if at == boundary else None,
            )
        assert h.finish(token).accepted
        assert h.send(dto.ChangeRun(target="PAUSING", reason="SYNTHETIC_PAUSE")).accepted
        if stage == "pause_after_stop":
            kill()
        h.authority.checkpoint("RUN-one")
        if stage == "checkpoint_published":
            kill()
    raise RuntimeError("Crash boundary not reached")


if __name__ == "__main__":
    main()
