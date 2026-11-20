"""Keep exhausted model-correctable tool calls from ending the run."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, override

from pydantic import ValidationError
from pydantic_ai import CallToolsNode, ModelRequestNode
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior
from pydantic_ai.messages import ModelRequest, RetryPromptPart, ToolCallPart, ToolReturnPart

if TYPE_CHECKING:
    from pydantic_ai.capabilities import AgentNode, NodeResult
    from pydantic_ai.tools import RunContext, ToolDefinition


def _retry_message(error: Exception) -> str | None:
    if not isinstance(error, UnexpectedModelBehavior) or error.message.startswith("Exceeded maximum output retries ("):
        return None
    cause = error.__cause__
    if isinstance(cause, ModelRetry | ValidationError):
        return str(cause)
    return None


class ToolErrorRecovery(AbstractCapability[object]):
    """Return an ordinary tool error when a bad call exhausts its retries."""

    @override
    async def on_tool_execute_error(
        self,
        ctx: RunContext[object],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, object],
        error: Exception,
    ) -> object:
        message = _retry_message(error)
        if ctx.last_attempt and message is not None:
            return f"Tool error: {message}"
        raise error

    @override
    async def on_node_run_error(
        self,
        ctx: RunContext[object],
        *,
        node: AgentNode[object],
        error: Exception,
    ) -> NodeResult[object]:
        message = _retry_message(error)
        if message is None or not isinstance(node, CallToolsNode) or not ctx.messages:
            raise error
        interrupted = ctx.messages[-1]
        if not isinstance(interrupted, ModelRequest) or interrupted.state != "interrupted":
            raise error

        answered = {
            part.tool_call_id for part in interrupted.parts if isinstance(part, ToolReturnPart | RetryPromptPart)
        }
        unanswered = [
            part
            for part in node.model_response.parts
            if isinstance(part, ToolCallPart) and part.tool_call_id not in answered
        ]
        if len(unanswered) != 1:
            raise error

        call = unanswered[0]
        result = ToolReturnPart(
            tool_name=call.tool_name,
            content=f"Tool error: {message}",
            tool_call_id=call.tool_call_id,
            tool_kind=call.tool_kind,
        )
        request = replace(interrupted, parts=[*interrupted.parts, result], state="complete")
        ctx.messages.pop()
        return ModelRequestNode(request)
