"""Only the application gateway owns routing, retry and spending policy."""

from typing import Protocol

from gidr.contracts.models import ModelExecutionResult, ModelTask


class ModelGateway(Protocol):
    async def execute(self, task: ModelTask) -> ModelExecutionResult: ...
