"""ctypes mirrors of the pinned ds4 public structures."""

import ctypes
from ctypes import (
    CFUNCTYPE, POINTER, c_bool, c_char_p, c_float, c_int,
    c_uint8, c_uint32, c_uint64, c_void_p,
)
from dataclasses import dataclass
from typing import Any


PROGRESS_FN = CFUNCTYPE(None, c_void_p, c_char_p, c_int, c_int)
CANCEL_FN = CFUNCTYPE(c_bool, c_void_p)


class Tokens(ctypes.Structure):
    _c_name_ = "ds4_tokens"
    _fields_ = [
        ("v", POINTER(c_int)),
        ("len", c_int),
        ("cap", c_int),
    ]


class TokenScore(ctypes.Structure):
    _c_name_ = "ds4_token_score"
    _fields_ = [
        ("id", c_int),
        ("logit", c_float),
        ("logprob", c_float),
    ]


class SessionSnapshot(ctypes.Structure):
    _c_name_ = "ds4_session_snapshot"
    _fields_ = [
        ("ptr", POINTER(c_uint8)),
        ("len", c_uint64),
        ("cap", c_uint64),
    ]


class SessionPayloadFile(ctypes.Structure):
    _c_name_ = "ds4_session_payload_file"
    _fields_ = [
        ("path", c_char_p),
        ("bytes", c_uint64),
    ]


class DistributedLayers(ctypes.Structure):
    _c_name_ = "ds4_distributed_layers"
    _fields_ = [
        ("start", c_uint32),
        ("end", c_uint32),
        ("has_output", c_bool),
        ("set", c_bool),
    ]


class DistributedOptions(ctypes.Structure):
    _c_name_ = "ds4_distributed_options"
    _fields_ = [
        ("role", c_int),
        ("layers", DistributedLayers),
        ("listen_host", c_char_p),
        ("listen_port", c_int),
        ("coordinator_host", c_char_p),
        ("coordinator_port", c_int),
        ("prefill_chunk", c_uint32),
        ("prefill_window", c_uint32),
        ("activation_bits", c_uint32),
        ("replay_check", c_bool),
        ("debug", c_bool),
    ]


class TpOptions(ctypes.Structure):
    _c_name_ = "ds4_tp_options"
    _fields_ = [
        ("role", c_int),
        ("requested", c_bool),
        ("listen_host", c_char_p),
        ("listen_port", c_int),
        ("leader_host", c_char_p),
        ("leader_port", c_int),
        ("transport", c_int),
        ("rdma_device", c_char_p),
        ("rdma_gid_index", c_int),
        ("rdma_gid_index_set", c_bool),
        ("glm_token_prefill", c_bool),
        ("debug_hash", c_int),
    ]


class EngineOptions(ctypes.Structure):
    _c_name_ = "ds4_engine_options"
    _fields_ = [
        ("model_path", c_char_p),
        ("mtp_path", c_char_p),
        ("vision_path", c_char_p),
        ("backend", c_int),
        ("n_threads", c_int),
        ("context_size", c_int),
        ("prefill_chunk", c_uint32),
        ("mtp_draft_tokens", c_int),
        ("mtp_margin", c_float),
        ("dspark_confidence_threshold", c_float),
        ("directional_steering_file", c_char_p),
        ("expert_profile_path", c_char_p),
        ("directional_steering_attn", c_float),
        ("directional_steering_ffn", c_float),
        ("power_percent", c_int),
        ("ssd_streaming_cache_experts", c_uint32),
        ("ssd_streaming_cache_bytes", c_uint64),
        ("ssd_streaming_full_layers", c_uint32),
        ("ssd_streaming_preload_experts", c_uint32),
        ("simulate_used_memory_bytes", c_uint64),
        ("warm_weights", c_bool),
        ("quality", c_bool),
        ("glm_mtp", c_bool),
        ("glm_mtp_timing", c_bool),
        ("dspark", c_bool),
        ("dspark_strict", c_bool),
        ("dspark_exact_sampling", c_bool),
        ("dspark_confidence_threshold_set", c_bool),
        ("cuda_tensor_parallel", c_bool),
        ("ssd_streaming", c_bool),
        ("ssd_streaming_cold", c_bool),
        ("ssd_streaming_full_layers_set", c_bool),
        ("inspect_only", c_bool),
        ("placement_ctx_hint", c_int),
        ("placement_session_count_hint", c_int),
        ("share_session_prefill_workspace", c_bool),
        ("first_token_test", c_bool),
        ("metal_graph_test", c_bool),
        ("load_slice", c_bool),
        ("load_layer_start", c_uint32),
        ("load_layer_end", c_uint32),
        ("load_output", c_bool),
        ("distributed", DistributedOptions),
        ("tp", TpOptions),
    ]


STRUCTS: dict[str, type[ctypes.Structure]] = {
    "ds4_tokens": Tokens,
    "ds4_token_score": TokenScore,
    "ds4_session_snapshot": SessionSnapshot,
    "ds4_session_payload_file": SessionPayloadFile,
    "ds4_distributed_layers": DistributedLayers,
    "ds4_distributed_options": DistributedOptions,
    "ds4_tp_options": TpOptions,
    "ds4_engine_options": EngineOptions,
}

@dataclass(frozen=True)
class StructLayout:
    size: int
    fields: dict[str, int]


def struct_layout(cls: type[ctypes.Structure]) -> StructLayout:
    """Record offsets, including nested fields relative to the outer struct."""
    fields = {}

    def walk(current: Any, prefix: str, base: int) -> None:
        for name, field_type in current._fields_:
            offset = base + getattr(current, name).offset
            path = prefix + name
            fields[path] = offset
            if issubclass(field_type, ctypes.Structure):
                walk(field_type, path + ".", offset)

    walk(cls, "", 0)
    return StructLayout(ctypes.sizeof(cls), fields)
