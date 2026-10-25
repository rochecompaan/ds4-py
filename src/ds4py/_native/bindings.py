"""The M1 native declarations and installed extension declaration discovery."""

from dataclasses import dataclass, replace
from importlib import metadata

from .typemap import ctypes_signature, parse_prototype

ENTRY_POINT_GROUP = "ds4py.bindings"


@dataclass(frozen=True)
class Binding:
    name: str
    prototype: str
    required: bool = True
    source: str = "core"

    def __post_init__(self):
        parsed = parse_prototype(self.prototype)
        if parsed.name != self.name:
            raise ValueError(f"binding {self.name} has prototype for {parsed.name}")
        ctypes_signature(parsed)

    @property
    def normalized(self) -> str:
        return str(parse_prototype(self.prototype))


# Prototypes retain the pinned upstream header's parameter names.
CORE_BINDINGS: tuple[Binding, ...] = (
    Binding("ds4_engine_open", "int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt)"),
    Binding("ds4_engine_close", "void ds4_engine_close(ds4_engine *e)"),
    Binding("ds4_engine_model_name", "const char *ds4_engine_model_name(ds4_engine *e)"),
    Binding("ds4_engine_model_id", "int ds4_engine_model_id(ds4_engine *e)"),
    Binding("ds4_engine_is_deepseek41", "bool ds4_engine_is_deepseek41(ds4_engine *e)"),
    Binding("ds4_engine_is_glm_dsa", "bool ds4_engine_is_glm_dsa(ds4_engine *e)"),
    Binding("ds4_engine_is_qwen4", "bool ds4_engine_is_qwen4(ds4_engine *e)"),
    Binding("ds4_engine_has_vision", "bool ds4_engine_has_vision(ds4_engine *e)"),
    Binding("ds4_engine_routed_quant_bits", "int ds4_engine_routed_quant_bits(ds4_engine *e)"),
    Binding("ds4_engine_mtp_draft_tokens", "int ds4_engine_mtp_draft_tokens(ds4_engine *e)"),
    Binding("ds4_engine_vocab_size", "int ds4_engine_vocab_size(ds4_engine *e)"),
    Binding("ds4_tokenize_text", "void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out)"),
    Binding("ds4_tokenize_rendered_chat", "void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out)"),
    Binding("ds4_chat_begin", "void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens)"),
    Binding("ds4_chat_append_think_prefix", "void ds4_chat_append_think_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode mode)"),
    Binding("ds4_chat_append_message", "void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content)"),
    Binding("ds4_chat_append_assistant_prefix", "void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode)"),
    Binding("ds4_qwen4_reasoning_effort_text", "const char *ds4_qwen4_reasoning_effort_text(ds4_think_mode mode)"),
    Binding("ds4_token_text", "char *ds4_token_text(ds4_engine *e, int token, size_t *len)"),
    Binding("ds4_token_eos", "int ds4_token_eos(ds4_engine *e)"),
    Binding("ds4_token_is_stop", "bool ds4_token_is_stop(ds4_engine *e, int token)"),
    Binding("ds4_token_is_stop_for_think_mode", "bool ds4_token_is_stop_for_think_mode(ds4_engine *e, int token, ds4_think_mode mode)"),
    Binding("ds4_tokens_free", "void ds4_tokens_free(ds4_tokens *tv)"),
    Binding("ds4_tokens_push", "void ds4_tokens_push(ds4_tokens *tv, int token)"),
    Binding("ds4_session_create", "int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size)"),
    Binding("ds4_session_free", "void ds4_session_free(ds4_session *s)"),
    Binding("ds4_session_sync", "int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen)"),
    Binding("ds4_session_set_progress", "void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud)"),
    Binding("ds4_session_set_display_progress", "void ds4_session_set_display_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud)"),
    Binding("ds4_session_set_cancel", "void ds4_session_set_cancel(ds4_session *s, ds4_session_cancel_fn fn, void *ud)"),
    Binding("ds4_session_sample", "int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng)"),
    Binding("ds4_session_argmax", "int ds4_session_argmax(ds4_session *s)"),
    Binding("ds4_session_eval", "int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen)"),
    Binding("ds4_session_rewind", "void ds4_session_rewind(ds4_session *s, int pos)"),
    Binding("ds4_session_pos", "int ds4_session_pos(ds4_session *s)"),
    Binding("ds4_session_ctx", "int ds4_session_ctx(ds4_session *s)"),
    Binding("ds4_session_tokens", "const ds4_tokens *ds4_session_tokens(ds4_session *s)"),
    Binding("ds4_session_common_prefix", "int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt)"),
    Binding("ds4_session_invalidate", "void ds4_session_invalidate(ds4_session *s)"),
    Binding("ds4_session_checkpoint_valid", "bool ds4_session_checkpoint_valid(const ds4_session *s)"),
    Binding("ds4_session_save_snapshot", "int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t errlen)"),
    Binding("ds4_session_load_snapshot", "int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen)"),
    Binding("ds4_session_snapshot_free", "void ds4_session_snapshot_free(ds4_session_snapshot *snap)"),
    Binding("ds4_session_payload_bytes", "uint64_t ds4_session_payload_bytes(ds4_session *s)"),
    Binding("ds4_session_copy_logits", "int ds4_session_copy_logits(ds4_session *s, float *out, int cap)"),
    Binding("ds4_session_set_logits", "int ds4_session_set_logits(ds4_session *s, const float *logits, int n)"),
    Binding("ds4_think_mode_for_context", "ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size)"),
    Binding("ds4_think_mode_parse_level", "bool ds4_think_mode_parse_level(const char *text, ds4_think_mode *out)"),
)


def discover_extension_bindings() -> tuple[Binding, ...]:
    """Load installed declarations, rejecting core names and incompatible ABIs.

    Compatible duplicates retain the first declaration's position and source,
    and are required if any provider requires them.
    """
    core_names = {binding.name for binding in CORE_BINDINGS}
    found: dict[str, Binding] = {}
    for entry_point in metadata.entry_points(group=ENTRY_POINT_GROUP):
        for binding in entry_point.load():
            if not isinstance(binding, Binding):
                raise TypeError(f"{entry_point.name} must provide Binding records")
            if binding.name in core_names:
                raise ValueError(f"extension {entry_point.name} redeclares core binding {binding.name}")
            previous = found.get(binding.name)
            if previous is None:
                found[binding.name] = replace(binding, source=entry_point.name)
            elif previous.normalized != binding.normalized:
                raise ValueError(f"conflicting extension prototypes for {binding.name}")
            elif binding.required and not previous.required:
                found[binding.name] = replace(previous, required=True)
    return tuple(found.values())


def all_bindings() -> tuple[Binding, ...]:
    """Return core declarations first, followed by installed extensions."""
    return CORE_BINDINGS + discover_extension_bindings()
