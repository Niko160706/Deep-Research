"""Secret-free repository audit. Findings expose paths and codes, never matched values."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_SECRET_PATH = re.compile(r"(^|/)\.env(?:\..+)?$|\.(?:pem|key|p12|pfx)$", re.IGNORECASE)
_TOKEN_PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{30,}\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{35,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(
        r"\b(?:TYPESAFE|TAVILY|EXA|GEMINI|GROQ|SILICONFLOW)_API_KEY\s*[:=]\s*['\"]?(?!synthetic[-_]|fake[-_]|placeholder|example)[A-Za-z0-9_./+=-]{16,}",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True, slots=True)
class SecretFinding:
    path: str
    reason_code: str


def audit_repository(root: Path) -> tuple[SecretFinding, ...]:
    """Check Git path names in current and historical trees; inspect current text only."""
    root = root.resolve(strict=True)
    current_names = _git_names(root, ["ls-files", "-co", "--exclude-standard", "-z"])
    staged_names = _git_names(
        root, ["diff", "--cached", "--name-only", "--diff-filter=ACMRTUXB", "-z"]
    )
    historical_names = _git_names(root, ["log", "--all", "--name-only", "--format=", "-z"])
    findings: set[SecretFinding] = set()
    for name in current_names:
        if _SECRET_PATH.search(name) and not name.endswith(".env.example"):
            findings.add(SecretFinding(name, "SECRET_FILE_IN_GIT_TREE"))
    for name in historical_names:
        if name and _SECRET_PATH.search(name) and not name.endswith(".env.example"):
            findings.add(SecretFinding(name, "SECRET_FILE_IN_GIT_HISTORY"))
    for name in current_names:
        path = root / name
        if (
            (_SECRET_PATH.search(name) and not name.endswith(".env.example"))
            or not path.is_file()
            or path.is_symlink()
        ):
            continue
        try:
            if path.stat().st_size > 10_000_000:
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        for pattern in _TOKEN_PATTERNS:
            if pattern.search(text):
                findings.add(SecretFinding(name, "SECRET_LIKE_CONTENT"))
                break
    for name in staged_names:
        if _SECRET_PATH.search(name) and not name.endswith(".env.example"):
            continue
        staged = _git_blob(root, name)
        if staged is None:
            continue
        try:
            text = staged.decode("utf-8")
        except UnicodeError:
            continue
        for pattern in _TOKEN_PATTERNS:
            if pattern.search(text):
                findings.add(SecretFinding(name, "SECRET_LIKE_STAGED_CONTENT"))
                break
    return tuple(sorted(findings, key=lambda finding: (finding.path, finding.reason_code)))


def _git_names(root: Path, arguments: list[str]) -> tuple[str, ...]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    if result.returncode != 0:
        return ()
    return tuple(
        item.decode("utf-8", errors="replace") for item in result.stdout.split(b"\0") if item
    )


def _git_blob(root: Path, name: str) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "show", f":{name}"],
            check=False,
            capture_output=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or len(result.stdout) > 10_000_000:
        return None
    return result.stdout


def main() -> int:
    findings = audit_repository(Path.cwd())
    if findings:
        for finding in findings:
            print(f"{finding.reason_code}: {finding.path}")
        return 1
    print("Secret audit passed; matched values were not emitted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
