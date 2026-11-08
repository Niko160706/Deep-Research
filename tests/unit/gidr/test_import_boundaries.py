"""Guard dependency direction and ensure no SDK/network enters pure modules."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "src/gidr"


def test_pure_layers_do_not_import_infrastructure_or_cli():
    forbidden = (
        "gidr.infrastructure",
        "gidr.interfaces",
        "gidr.bootstrap",
        "httpx",
        "aiohttp",
        "typesafe",
        "litellm",
        "keyring",
        "sqlite3",
        "subprocess",
        "socket",
        "drx",
    )
    for layer in ("contracts", "domain"):
        for path in (ROOT / layer).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom):
                    assert node.level == 0, f"Relative import hides dependency: {path}"
                    names = [node.module or ""]
                for name in names:
                    assert not any(name == p or name.startswith(p + ".") for p in forbidden), (
                        path,
                        name,
                    )


def test_new_runtime_has_no_provider_credentials_or_secret_loading():
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(n.name != "dotenv" for n in node.names)
            if isinstance(node, ast.ImportFrom):
                assert node.module not in {"dotenv", "drx.config"}


def test_state_vocabulary_is_separate_from_epistemic_verdicts():
    from gidr.domain.states import RunState, TaskState, Verdict

    assert set(RunState.__members__) == {
        "CREATED",
        "PLANNING",
        "PLAN_READY",
        "RUNNING",
        "PAUSING",
        "PAUSED",
        "INTERRUPTED",
        "RECOVERING",
        "COMPLETING",
        "COMPLETED",
        "FAILED",
        "CANCELLED",
    }
    assert set(TaskState.__members__) == {
        "PENDING",
        "WAITING_DEPENDENCY",
        "READY",
        "RESERVED",
        "RUNNING",
        "RETRY_WAIT",
        "DEFERRED",
        "AWAITING_APPROVAL",
        "BLOCKED",
        "SUCCEEDED",
        "FAILED",
        "CANCELLED",
    }
    assert set(Verdict.__members__) == {
        "UNASSESSED",
        "INSUFFICIENT",
        "SUPPORTED",
        "CONTESTED",
        "REFUTED",
        "CONTEXT_DEPENDENT",
    }
