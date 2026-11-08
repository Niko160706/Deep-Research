"""Ports express ownership without creating a second state authority."""

from typing import Protocol

from gidr.contracts.common import ArtifactRef, CommandId, Contract, Record
from gidr.contracts.control import TaskCommand


class CommandReceipt(Record):
    command_id: CommandId
    outcome: str


class StateAuthority(Protocol):
    async def handle(self, command: TaskCommand[Contract]) -> CommandReceipt: ...


class ArtifactStore(Protocol):
    def publish(self, content: bytes) -> ArtifactRef: ...

    def read(self, reference: ArtifactRef) -> bytes: ...
