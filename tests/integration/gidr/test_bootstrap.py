"""Installed-package bootstrap check; no live API or research integration."""

import os
import subprocess
import sys


def test_installed_package_imports_without_network_dotenv_or_legacy(tmp_path):
    (tmp_path / ".env").write_text("TYPESAFE_API_KEY=SYNTHETIC-CREDENTIAL-CANARY\n")
    code = r"""
import sys, pathlib
def guard(event, args):
    if event.startswith("socket.") or event == "subprocess.Popen":
        raise RuntimeError("Unexpected external side effect")
    if event == "open" and args and str(args[0]).endswith(".env"):
        raise RuntimeError("Unexpected dotenv access")
sys.addaudithook(guard)
import gidr
import gidr.contracts.common, gidr.contracts.sources, gidr.contracts.models
import gidr.contracts.control, gidr.contracts.evidence, gidr.domain.links
import gidr.interfaces.provider_ports, gidr.interfaces.model_ports
import gidr.interfaces.storage_ports, gidr.interfaces.event_ports
from gidr.bootstrap import main
assert not any(n == "drx" or n.startswith("drx.") for n in sys.modules)
assert main() == 0
assert "SYNTHETIC-CREDENTIAL-CANARY" not in repr(gidr)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        env={
            "PATH": os.defpath,
            "LANG": "C.UTF-8",
            "TYPESAFE_API_KEY": "SYNTHETIC-CREDENTIAL-CANARY",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "belum diaktifkan" in result.stdout
    assert "SYNTHETIC-CREDENTIAL-CANARY" not in result.stdout + result.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == [".env"]
