"""Owner-only local paths, durable publication and cross-process locks."""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def reject_symlinks(path: Path) -> None:
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("SYMLINK_PATH_BLOCKED")


def private_directory(path: Path) -> Path:
    path = path.absolute()
    reject_symlinks(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError("DIRECTORY_REQUIRED")
    if os.name == "posix":
        if path.stat().st_uid != os.getuid():
            raise PermissionError("DIRECTORY_OWNER_MISMATCH")
        path.chmod(0o700)
    return path


def regular_fd(path: Path, *, create: bool = False) -> int:
    reject_symlinks(path)
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    if create:
        flags |= os.O_CREAT
    fd = os.open(path, flags, 0o600)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("REGULAR_FILE_REQUIRED")
    if os.name == "posix":
        os.fchmod(fd, 0o600)
    return fd


def sync_directory(path: Path) -> None:
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_write(path: Path, data: bytes) -> None:
    private_directory(path.parent)
    reject_symlinks(path)
    fd, name = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        sync_directory(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


class LocalLock:
    """Reentrant per instance; an OS lock fences other processes, not a PID file."""

    def __init__(self, path: Path, timeout: float = 5.0):
        self.path = private_directory(path.parent) / path.name
        self.timeout = timeout
        self._thread = threading.RLock()
        self._depth = 0

    @contextmanager
    def hold(self) -> Iterator[None]:
        with self._thread:
            if self._depth:
                self._depth += 1
                try:
                    yield
                finally:
                    self._depth -= 1
                return
            fd = regular_fd(self.path, create=True)
            deadline = time.monotonic() + self.timeout
            try:
                while True:
                    try:
                        if sys.platform != "win32":
                            import fcntl

                            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        else:
                            import msvcrt

                            if os.fstat(fd).st_size == 0:
                                os.write(fd, b"\0")
                            os.lseek(fd, 0, os.SEEK_SET)
                            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                        break
                    except (BlockingIOError, PermissionError, OSError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("LOCAL_LOCK_TIMEOUT") from None
                        time.sleep(0.02)
                self._depth = 1
                try:
                    yield
                finally:
                    self._depth = 0
                    if sys.platform != "win32":
                        import fcntl

                        fcntl.flock(fd, fcntl.LOCK_UN)
                    else:
                        import msvcrt

                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            finally:
                os.close(fd)
