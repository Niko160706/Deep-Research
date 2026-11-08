"""Linux bubblewrap launcher. No usable sandbox means parser execution is denied."""

from __future__ import annotations

import hashlib
import os
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from gidr.infrastructure.security.events import SecurityEventSink, SecurityObservation


class SandboxError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class SandboxResult:
    returncode: int
    stdout: bytes
    stderr: str
    elapsed_seconds: float


class ParserSandbox:
    def __init__(
        self,
        *,
        approved_executables: frozenset[Path],
        quarantine_root: Path,
        bwrap: Path = Path("/usr/bin/bwrap"),
        prlimit: Path | None = None,
        max_output_bytes: int = 2_000_000,
        max_memory_bytes: int = 1_000_000_000,
        event_sink: SecurityEventSink | None = None,
    ) -> None:
        self._approved = frozenset(p.resolve(strict=True) for p in approved_executables)
        if quarantine_root.is_symlink() or not quarantine_root.is_dir():
            raise ValueError("PARSER_QUARANTINE_ROOT_INVALID")
        self._quarantine_root = quarantine_root.resolve(strict=True)
        self._bwrap = bwrap.resolve(strict=True) if bwrap.exists() else None
        found_prlimit = shutil.which("prlimit")
        prlimit_path = prlimit or (Path(found_prlimit) if found_prlimit is not None else None)
        self._prlimit = (
            prlimit_path.resolve(strict=True) if prlimit_path and prlimit_path.exists() else None
        )
        self._max_output = max_output_bytes
        self._max_memory = max_memory_bytes
        self._event_sink = event_sink

    def available(self) -> bool:
        return (
            os.name == "posix"
            and self._bwrap is not None
            and os.access(self._bwrap, os.X_OK)
            and self._prlimit is not None
            and os.access(self._prlimit, os.X_OK)
            and self._max_output > 0
            and self._max_memory >= 128_000_000
        )

    def run(
        self, argv: Sequence[str], *, input_path: Path, timeout_seconds: float = 10.0
    ) -> SandboxResult:
        if not self.available():
            raise SandboxError("PARSER_SANDBOX_UNAVAILABLE")
        if self._event_sink is None:
            raise SandboxError("PARSER_AUDIT_SINK_REQUIRED")
        if not argv or not 0 < timeout_seconds <= 120:
            raise SandboxError("PARSER_REQUEST_INVALID")
        requested_executable = Path(argv[0])
        if requested_executable.is_symlink():
            raise SandboxError("PARSER_EXECUTABLE_NOT_APPROVED")
        executable = requested_executable.resolve(strict=True)
        forbidden_interpreters = {
            "sh",
            "bash",
            "dash",
            "zsh",
            "fish",
            "cmd",
            "powershell",
            "pwsh",
            "python",
            "python3",
            "python3.12",
            "node",
            "ruby",
            "perl",
        }
        if (
            executable not in self._approved
            or executable.name.casefold() in forbidden_interpreters
            or not executable.is_file()
            or not os.access(executable, os.X_OK)
            or any("\x00" in value or len(value) > 4096 for value in argv[1:])
        ):
            raise SandboxError("PARSER_EXECUTABLE_NOT_APPROVED")
        if input_path.is_symlink():
            raise SandboxError("PARSER_INPUT_INVALID")
        source = input_path.resolve(strict=True)
        if not source.is_file() or not source.is_relative_to(self._quarantine_root):
            raise SandboxError("PARSER_INPUT_INVALID")
        if source.stat().st_size > self._max_output * 5:
            raise SandboxError("PARSER_INPUT_LIMIT")
        input_hash = hashlib.sha256()
        with source.open("rb") as input_stream:
            for chunk in iter(lambda: input_stream.read(65536), b""):
                input_hash.update(chunk)
        observation = SecurityObservation(
            category="PARSER_DISPATCH",
            reason_code="SANDBOXED_PARSER_DISPATCH",
            request_id="PARSE-" + input_hash.hexdigest()[:24],
            content_sha256=input_hash.hexdigest(),
            byte_count=source.stat().st_size,
        )
        self._record(observation)
        started = __import__("time").monotonic()
        with tempfile.TemporaryDirectory(prefix="gidr-parser-") as temporary:
            del temporary
            sandbox_args = [
                str(self._bwrap),
                "--die-with-parent",
                "--new-session",
                "--unshare-net",
                "--unshare-pid",
                "--unshare-ipc",
                "--unshare-uts",
                "--clearenv",
            ]
            for system_path in (
                Path("/usr"),
                Path("/bin"),
                Path("/lib"),
                Path("/lib64"),
                Path("/etc/ld.so.cache"),
            ):
                if system_path.exists():
                    sandbox_args.extend(["--ro-bind", str(system_path), str(system_path)])
            sandbox_args.extend(
                [
                    "--proc",
                    "/proc",
                    "--dev",
                    "/dev",
                    "--size",
                    str(min(self._max_memory // 8, 128_000_000)),
                    "--tmpfs",
                    "/tmp",
                    "--dir",
                    "/input",
                    "--size",
                    str(min(self._max_output * 4, 64_000_000)),
                    "--tmpfs",
                    "/output",
                ]
            )
            sandbox_args.extend(["--ro-bind", str(source), "/input/document", "--chdir", "/output"])
            sandbox_args.extend(
                [
                    "--setenv",
                    "PATH",
                    "/usr/bin:/bin",
                    "--setenv",
                    "HOME",
                    "/nonexistent",
                    "--setenv",
                    "LANG",
                    "C.UTF-8",
                    "--setenv",
                    "PYTHONNOUSERSITE",
                    "1",
                ]
            )
            sandbox_args.extend(["--", str(executable), *argv[1:]])
            command = [
                str(self._prlimit),
                f"--cpu={max(1, int(timeout_seconds) + 1)}",
                f"--as={self._max_memory}",
                f"--fsize={self._max_output}",
                "--nofile=64",
                "--",
                *sandbox_args,
            ]
            try:
                with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                    result = subprocess.run(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                        timeout=timeout_seconds,
                        check=False,
                        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
                        close_fds=True,
                        start_new_session=True,
                    )
                    stdout.seek(0)
                    stderr.seek(0)
                    stdout_bytes = stdout.read(self._max_output + 1)
                    stderr_bytes = stderr.read(65537)
            except subprocess.TimeoutExpired as exc:
                self._record_failure(input_hash.hexdigest(), "PARSER_TIMEOUT")
                raise SandboxError("PARSER_TIMEOUT") from exc
            except OSError as exc:
                self._record_failure(input_hash.hexdigest(), "PARSER_SANDBOX_FAILED")
                raise SandboxError("PARSER_SANDBOX_FAILED") from exc
            elapsed = __import__("time").monotonic() - started
            if stderr_bytes.startswith(b"bwrap:"):
                self._record_failure(input_hash.hexdigest(), "PARSER_SANDBOX_FAILED")
                raise SandboxError("PARSER_SANDBOX_FAILED")
            if (
                result.returncode in {-signal.SIGXFSZ, 128 + signal.SIGXFSZ}
                or len(stdout_bytes) > self._max_output
                or len(stderr_bytes) > 65536
            ):
                self._record_failure(input_hash.hexdigest(), "PARSER_OUTPUT_LIMIT")
                raise SandboxError("PARSER_OUTPUT_LIMIT")
            if result.returncode < 0:
                self._record_failure(input_hash.hexdigest(), "PARSER_SIGNAL_EXIT")
            else:
                self._record(
                    SecurityObservation(
                        category="PARSER_COMPLETED",
                        reason_code="PARSER_EXIT",
                        request_id=observation.request_id,
                        content_sha256=input_hash.hexdigest(),
                        byte_count=source.stat().st_size,
                        status_code=result.returncode,
                    )
                )
            # Do not return untrusted stderr as an exception or audit message.
            return SandboxResult(
                result.returncode,
                stdout_bytes,
                f"stderr_sha256:{hashlib.sha256(stderr_bytes).hexdigest()}",
                elapsed,
            )

    def _record_failure(self, digest: str, reason: str) -> None:
        assert self._event_sink is not None
        self._record(
            SecurityObservation(
                category="PARSER_SANDBOX_FAILURE",
                reason_code=reason,
                request_id="PARSE-" + digest[:24],
                content_sha256=digest,
            )
        )

    def _record(self, observation: SecurityObservation) -> None:
        assert self._event_sink is not None
        try:
            self._event_sink.record_security(observation)
        except Exception:
            raise SandboxError("PARSER_AUDIT_FAILURE") from None
