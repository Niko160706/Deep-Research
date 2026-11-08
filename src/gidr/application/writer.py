"""One bounded async command consumer, with no awaits inside DB transactions."""

import asyncio
from typing import Self

from gidr.application.commands import CommandPort
from gidr.contracts.commands import CommandResult, StateCommand


class AsyncStateWriter:
    def __init__(self, port: CommandPort, capacity: int = 64):
        if not 1 <= capacity <= 1024:
            raise ValueError("INVALID_WRITER_CAPACITY")
        self._port = port
        self._queue: asyncio.Queue[tuple[StateCommand, asyncio.Future[CommandResult]] | None] = (
            asyncio.Queue(capacity)
        )
        self._consumer: asyncio.Task[None] | None = None
        self._closing = True
        self._admission = asyncio.Lock()

    async def __aenter__(self) -> Self:
        if self._consumer is not None:
            raise RuntimeError("WRITER_ALREADY_STARTED")
        self._closing = False
        self._consumer = asyncio.create_task(self._consume())
        return self

    async def __aexit__(self, *args: object) -> None:
        async with self._admission:
            self._closing = True
            await self._queue.put(None)
        if self._consumer:
            await self._consumer
        self._consumer = None

    async def submit(self, command: StateCommand) -> CommandResult:
        future: asyncio.Future[CommandResult] = asyncio.get_running_loop().create_future()
        async with self._admission:
            if self._closing:
                raise RuntimeError("WRITER_CLOSED")
            await self._queue.put((command, future))
        # Caller cancellation cannot roll back an already committed command.
        return await asyncio.shield(future)

    async def _consume(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    return
                command, future = item
                try:
                    result = self._port.handle(command)
                except Exception as error:
                    future.set_exception(error)
                else:
                    future.set_result(result)
            finally:
                self._queue.task_done()
