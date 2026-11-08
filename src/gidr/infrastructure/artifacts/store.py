"""Publish-before-reference SHA-256 blobs; database and filesystem stay distinct."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from uuid import uuid4

from pydantic import TypeAdapter

from gidr.contracts.common import ArtifactRef, Digest
from gidr.infrastructure.audit.policy import SecretPolicy
from gidr.infrastructure.localfs import (
    LocalLock,
    private_directory,
    reject_symlinks,
    sync_directory,
)


class ArtifactError(RuntimeError):
    pass


class ArtifactStore:
    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int = 10 * 1024 * 1024,
        forbidden_values: tuple[bytes, ...] = (),
        secret_policy: SecretPolicy | None = None,
    ):
        if max_bytes < 1:
            raise ValueError("INVALID_ARTIFACT_LIMIT")
        self.root = private_directory(root)
        self.blobs = private_directory(self.root / "blobs/sha256")
        self.staging = private_directory(self.root / "staging")
        self.quarantine = private_directory(self.root / "quarantine")
        if len({p.stat().st_dev for p in (self.blobs, self.staging, self.quarantine)}) != 1:
            raise ArtifactError("ARTIFACT_PATHS_REQUIRE_ONE_FILESYSTEM")
        self.lock = LocalLock(self.root / ".artifacts.lock")
        self.max_bytes = max_bytes
        self.forbidden_values = tuple(value for value in forbidden_values if value)
        self.secret_policy = secret_policy

    def path(self, digest: str) -> Path:
        TypeAdapter(Digest).validate_python(digest, strict=True)
        target = self.blobs / digest[:2] / digest[2:4] / digest
        reject_symlinks(target)
        return target

    def publish(self, content: bytes) -> ArtifactRef:
        return self.publish_chunks((content,))

    def publish_chunks(
        self, chunks: Iterable[bytes], *, fault: Callable[[str], None] = lambda _: None
    ) -> ArtifactRef:
        with self.lock.hold():
            fd, name = tempfile.mkstemp(prefix="stage-", dir=self.staging)
            temporary = Path(name)
            digest = hashlib.sha256()
            size = 0
            tail = b""
            carry = max((len(value) for value in self.forbidden_values), default=1) - 1
            policy_tail = b""
            try:
                with os.fdopen(fd, "wb") as handle:
                    for chunk in chunks:
                        if not isinstance(chunk, bytes):
                            raise ArtifactError("ARTIFACT_CHUNKS_MUST_BE_BYTES")
                        size += len(chunk)
                        if size > self.max_bytes:
                            raise ArtifactError("ARTIFACT_BYTE_LIMIT")
                        combined = tail + chunk
                        if any(value in combined for value in self.forbidden_values):
                            raise ArtifactError("SECRET_ARTIFACT_BLOCKED")
                        tail = combined[-carry:] if carry else b""
                        if self.secret_policy is not None:
                            policy_bytes = policy_tail + chunk
                            try:
                                self.secret_policy.validate_text(
                                    policy_bytes.decode("utf-8", errors="replace")
                                )
                            except ValueError as exc:
                                raise ArtifactError("SECRET_ARTIFACT_BLOCKED") from exc
                            policy_tail = policy_bytes[-8192:]
                        handle.write(chunk)
                        digest.update(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                fault("staged")
                sha = digest.hexdigest()
                reference = ArtifactRef(artifact_id="ART-" + sha, sha256=sha, size_bytes=size)
                destination = self.path(sha)
                private_directory(destination.parent)
                try:
                    os.link(temporary, destination)
                    sync_directory(destination.parent)
                except FileExistsError:
                    self.read(reference)
                fault("published")
                return reference
            finally:
                temporary.unlink(missing_ok=True)
                sync_directory(self.staging)

    def read(self, reference: ArtifactRef) -> bytes:
        reference = ArtifactRef.model_validate(reference)
        if (
            reference.artifact_id != "ART-" + reference.sha256
            or reference.size_bytes > self.max_bytes
        ):
            raise ArtifactError("ARTIFACT_ID_OR_SIZE_INVALID")
        with self.lock.hold():
            path = self.path(reference.sha256)
            try:
                fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            except FileNotFoundError:
                raise ArtifactError("ARTIFACT_UNAVAILABLE") from None
            with os.fdopen(fd, "rb") as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > self.max_bytes:
                    raise ArtifactError("ARTIFACT_FILE_INVALID")
                data = handle.read(self.max_bytes + 1)
            if hashlib.sha256(data).hexdigest() != reference.sha256:
                quarantine = self.quarantine / (reference.sha256 + "-" + uuid4().hex)
                os.replace(path, quarantine)
                sync_directory(path.parent)
                sync_directory(self.quarantine)
                raise ArtifactError("ARTIFACT_INTEGRITY_FAILURE")
            if len(data) != reference.size_bytes:
                raise ArtifactError("ARTIFACT_REFERENCE_SIZE_MISMATCH")
            return data

    def published(self) -> tuple[ArtifactRef, ...]:
        result = []
        for path in self.blobs.glob("*/*/*"):
            if path.is_symlink():
                raise ArtifactError("SYMLINK_ARTIFACT_BLOCKED")
            if path.is_file():
                TypeAdapter(Digest).validate_python(path.name, strict=True)
                if path != self.path(path.name):
                    raise ArtifactError("INVALID_ARTIFACT_LOCATION")
                result.append(
                    ArtifactRef(
                        artifact_id="ART-" + path.name,
                        sha256=path.name,
                        size_bytes=path.stat().st_size,
                    )
                )
        return tuple(result)
