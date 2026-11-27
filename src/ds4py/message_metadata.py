"""Canonical metadata keys shared by rendering, binding, and persistence."""

from typing import Literal

DS4_KEY = "ds4.key"
DS4_SOURCE = "ds4.source"
DS4_SAMPLED = "ds4.sampled"
DS4_CONTEXT_PAGING = "ds4.context_paging"
DS4_VIEW_OF = "ds4.view_of"
DS4_PAGING_INPUT_TOKENS = "ds4.paging_input_tokens"
DS4_REQUEST_SYNC_COUNT = "ds4.request_sync_count"
DS4_TIMESTAMP = "ds4.timestamp"

type PagingMarker = Literal["notice", "recovery"]
