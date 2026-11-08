"""Stream a consistent committed prefix into an atomically published generation."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from pydantic import StringConstraints

from gidr.contracts.common import Contract, Digest, Name, Nonnegative, RunId, parse_contract
from gidr.contracts.control import AuditEvent
from gidr.infrastructure.db.authority import PersistenceAuthority
from gidr.infrastructure.db.database import Database
from gidr.infrastructure.localfs import (
    LocalLock,
    atomic_write,
    private_directory,
    reject_symlinks,
    sync_directory,
)

CHANNELS = {
    "fast_decision": "jev_decisions.jsonl",
    "llm": "llm_calls.jsonl",
    "retrieval": "retrieval.jsonl",
    "parsing": "parsing.jsonl",
    "evidence": "evidence.jsonl",
    "scheduler": "scheduler.jsonl",
    "security": "security.jsonl",
    "errors": "errors.jsonl",
}


class ProjectionManifest(Contract):
    projection_id: Name
    run_id: RunId | None
    last_seq: Nonnegative
    generation: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
    files: dict[str, Digest]


def events(
    database: Database,
    *,
    run_id: str | None = None,
    after: int = 0,
    through: int | None = None,
    batch_size: int = 200,
) -> Iterator[AuditEvent]:
    if after < 0 or not 1 <= batch_size <= 10000:
        raise ValueError("INVALID_EVENT_CURSOR")
    with database.reader() as conn:
        end = (
            conn.execute("SELECT coalesce(max(seq),0) FROM audit_events").fetchone()[0]
            if through is None
            else through
        )
        cursor = conn.execute(
            "SELECT seq,committed_at_utc,event_payload_json FROM audit_events WHERE seq>? AND seq<=? AND (? IS NULL OR run_id=?) ORDER BY seq",
            (after, end, run_id, run_id),
        )
        while rows := cursor.fetchmany(batch_size):
            for row in rows:
                body = json.loads(row["event_payload_json"])
                body.update(seq=row["seq"], committed_at_utc=row["committed_at_utc"])
                yield parse_contract(AuditEvent, json.dumps(body))


class AuditExporter:
    def __init__(self, authority: PersistenceAuthority):
        self.authority = authority

    def export(
        self,
        root: Path,
        *,
        projection_id: str = "audit",
        run_id: str | None = None,
        fault: Callable[[str], None] = lambda _: None,
    ) -> Path:
        root = private_directory(root)
        with LocalLock(root / ".projection.lock").hold():
            generation = uuid4().hex
            destination = private_directory(root / "generations" / generation)
            with self.authority.database.reader() as conn:
                end = conn.execute("SELECT coalesce(max(seq),0) FROM audit_events").fetchone()[0]
            names = ("events.jsonl", *CHANNELS.values())
            handles = {
                name: os.fdopen(
                    os.open(destination / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
                )
                for name in names
            }
            hashes = {name: hashlib.sha256() for name in names}
            try:
                for event in events(self.authority.database, run_id=run_id, through=end):
                    line = event.model_dump_json() + "\n"
                    self.authority.policy.validate_json(line)
                    encoded = line.encode()
                    outputs = ["events.jsonl"]
                    if event.component in CHANNELS:
                        outputs.append(CHANNELS[event.component])
                    if event.component != "errors" and event.event_type.endswith(
                        (".failed", "_failed")
                    ):
                        outputs.append("errors.jsonl")
                    for name in outputs:
                        handles[name].write(encoded)
                        hashes[name].update(encoded)
                for handle in handles.values():
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                for handle in handles.values():
                    handle.close()
            sync_directory(destination)
            snapshot = ProjectionManifest(
                projection_id=projection_id,
                run_id=run_id,
                last_seq=end,
                generation=generation,
                files={name: digest.hexdigest() for name, digest in hashes.items()},
            )
            manifest = json.dumps(snapshot.model_dump(mode="json"), sort_keys=True).encode()
            fault("before_projection_publish")
            atomic_write(root / "CURRENT.json", manifest)
            fault("after_projection_publish")
            self.authority.checkpoint_projection(
                projection_id, end, generation, hashlib.sha256(manifest).hexdigest()
            )
            return destination

    @staticmethod
    def current(root: Path) -> Path:
        reject_symlinks(root / "CURRENT.json")
        manifest = parse_contract(ProjectionManifest, (root / "CURRENT.json").read_bytes())
        generation = manifest.generation
        target = root / "generations" / generation
        reject_symlinks(target)
        if set(manifest.files) != {"events.jsonl", *CHANNELS.values()}:
            raise ValueError("INVALID_PROJECTION_FILES")
        for name, digest in manifest.files.items():
            path = target / name
            reject_symlinks(path)
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(65536):
                    hasher.update(chunk)
            if hasher.hexdigest() != digest:
                raise ValueError("PROJECTION_HASH_MISMATCH_REBUILD_REQUIRED")
        return target
