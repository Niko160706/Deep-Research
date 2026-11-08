"""Private, bounded quarantine and archive-manifest inspection; never extract."""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from gidr.infrastructure.audit.policy import SecretPolicy


class QuarantineError(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class QuarantineRecord:
    path: Path
    sha256: str
    size_bytes: int
    declared_mime: str | None
    detected_mime: str
    status: str
    reason_code: str | None = None


_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "application/pdf"),
    (b"PK\x03\x04", "application/zip"),
    (b"\x7fELF", "application/x-executable"),
    (b"MZ", "application/x-dosexec"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
)
_ALLOWED = {
    "application/pdf",
    "application/zip",
    "image/png",
    "image/jpeg",
    "text/plain",
    "text/html",
    "application/json",
    "text/csv",
}


def detect_mime(data: bytes) -> str:
    for signature, mime in _MAGIC:
        if data.startswith(signature):
            return mime
    sample = data[:8192]
    if b"\x00" not in sample:
        try:
            sample.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            stripped = sample.lstrip().lower()
            if stripped.startswith((b"<!doctype html", b"<html")):
                return "text/html"
            if stripped.startswith((b"{", b"[")):
                return "application/json"
            return "text/plain"
    return "application/octet-stream"


def inspect_zip_manifest(
    data: bytes,
    *,
    max_entries: int = 1000,
    max_uncompressed: int = 50_000_000,
    max_ratio: int = 100,
) -> tuple[str, ...]:
    if not data.startswith(b"PK\x03\x04"):
        raise QuarantineError("ARCHIVE_SIGNATURE_INVALID")
    try:
        with zipfile.ZipFile(__import__("io").BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > max_entries:
                raise QuarantineError("ARCHIVE_ENTRY_LIMIT")
            total = 0
            names: list[str] = []
            for entry in entries:
                path = PurePosixPath(entry.filename.replace("\\", "/"))
                mode = entry.external_attr >> 16
                if path.is_absolute() or ".." in path.parts or (mode and stat.S_ISLNK(mode)):
                    raise QuarantineError("ARCHIVE_PATH_OR_LINK_BLOCKED")
                if entry.flag_bits & 1:
                    raise QuarantineError("ARCHIVE_ENCRYPTED_BLOCKED")
                total += entry.file_size
                if total > max_uncompressed:
                    raise QuarantineError("ARCHIVE_EXPANSION_LIMIT")
                if entry.file_size > max_ratio * max(entry.compress_size, 1):
                    raise QuarantineError("ARCHIVE_RATIO_LIMIT")
                names.append(entry.filename[:1024])
            return tuple(names)
    except zipfile.BadZipFile as exc:
        raise QuarantineError("ARCHIVE_INVALID") from exc


class QuarantineStore:
    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int = 10_000_000,
        secret_policy: SecretPolicy | None = None,
    ):
        self.root = root
        self.max_bytes = max_bytes
        self.secret_policy = secret_policy
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if root.is_symlink() or not root.is_dir():
            raise QuarantineError("QUARANTINE_ROOT_INVALID")
        os.chmod(root, 0o700)

    def store(
        self, data: bytes, *, declared_mime: str | None, allow_archive: bool = False
    ) -> QuarantineRecord:
        if len(data) > self.max_bytes:
            raise QuarantineError("QUARANTINE_SIZE_LIMIT")
        if self.secret_policy is not None:
            try:
                self.secret_policy.validate_text(data.decode("utf-8", errors="replace"))
            except ValueError as exc:
                raise QuarantineError("SECRET_CONTENT_BLOCKED") from exc
        detected = detect_mime(data)
        mismatch = (
            declared_mime is not None and declared_mime.split(";", 1)[0].strip().lower() != detected
        )
        risky = (
            detected not in _ALLOWED
            or mismatch
            or (detected == "application/zip" and not allow_archive)
        )
        reason = (
            "MIME_MISMATCH"
            if mismatch
            else "MIME_DISALLOWED"
            if detected not in _ALLOWED
            else "ARCHIVE_REVIEW_REQUIRED"
            if detected == "application/zip" and not allow_archive
            else None
        )
        if detected == "application/zip" and allow_archive:
            try:
                inspect_zip_manifest(data)
            except QuarantineError as exc:
                risky = True
                reason = exc.reason
        name = secrets.token_hex(24) + ".blob"
        path = self.root / name
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        try:
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(fd)
        return QuarantineRecord(
            path,
            hashlib.sha256(data).hexdigest(),
            len(data),
            declared_mime,
            detected,
            "QUARANTINED" if risky else "STAGED",
            reason,
        )
