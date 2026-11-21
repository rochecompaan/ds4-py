"""Exact, observation-only request measurement for RollingWindow selection."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models import ModelRequestParameters

    from ds4py.binding import MessageBinder
    from ds4py.binding_projection import RequestProjection


@dataclass(frozen=True, slots=True)
class RequestMeter:
    """Measure each candidate afresh through the binder's scratch planner."""

    binder: MessageBinder
    params: ModelRequestParameters

    def measure(
        self,
        *,
        canonical_messages: Sequence[ModelMessage],
        candidate_messages: Sequence[ModelMessage],
    ) -> RequestProjection:
        return self.binder.project_request_view(
            canonical_messages=canonical_messages,
            candidate_messages=candidate_messages,
            params=self.params,
        )
