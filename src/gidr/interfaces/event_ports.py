"""Audit projections read committed events; adapters have no audit write port."""

from collections.abc import Iterable
from typing import Protocol

from gidr.contracts.common import RunId
from gidr.contracts.control import AuditEvent


class AuditReader(Protocol):
    def events(self, run_id: RunId, after_sequence: int = 0) -> Iterable[AuditEvent]: ...
