from typing import Any
import ctypes
import pytest
from ds4py._native import typemap, structs
from ds4py.errors import TypeMapError

@pytest.mark.parametrize("text, normal", [
    ("int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen)",
     "int ds4_session_eval(ds4_session *, int, char *, size_t)"),
    ("char *ds4_token_text(ds4_engine *e, int token, size_t *len)",
     "char *ds4_token_text(ds4_engine *, int, size_t *)"),
    ("const ds4_tokens *ds4_session_tokens(ds4_session *s)",
     "const ds4_tokens *ds4_session_tokens(ds4_session *)"),
    ("int  ds4_engine_open( ds4_engine ** out , const ds4_engine_options*opt )",
     "int ds4_engine_open(ds4_engine **, const ds4_engine_options *)"),
    ("unsigned long long f(unsigned int, void *p)", "unsigned long long f(unsigned int, void *)"),
    ("void f(void)", "void f(void)"),
])
def test_normal_form(text: Any, normal: Any) -> None:
    assert typemap.normalize_prototype(text) == normal

def test_ctypes_signature_maps_pointers_structs_and_callbacks() -> None:
    p = typemap.parse_prototype(
        "void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud)")
    assert typemap.ctypes_signature(p) == (None, [ctypes.c_void_p, structs.PROGRESS_FN, ctypes.c_void_p])
    p = typemap.parse_prototype("char *ds4_token_text(ds4_engine *e, int token, size_t *len)")
    assert typemap.ctypes_signature(p) == (
        ctypes.POINTER(ctypes.c_char), [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)])
    p = typemap.parse_prototype("int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen)")
    assert typemap.ctypes_signature(p)[1][1] is ctypes.POINTER(structs.Tokens)

def test_unknown_type_is_an_error_naming_type_and_function() -> None:
    with pytest.raises(TypeMapError, match=r"ds4_widget.*ds4_frob"):
        typemap.ctypes_signature(typemap.parse_prototype("int ds4_frob(ds4_widget *w)"))

def test_probe_declaration() -> None:
    p = typemap.parse_prototype("char *ds4_token_text(ds4_engine *e, int token, size_t *len)")
    assert typemap.probe_declaration(p) == "char *(*ds4py_probe)(ds4_engine *, int, size_t *)"

def test_struct_layout_walks_nested_fields() -> None:
    layout = structs.struct_layout(structs.EngineOptions)
    assert layout.size == ctypes.sizeof(structs.EngineOptions)
    assert layout.fields["model_path"] == 0
    assert "distributed.role" in layout.fields and "tp.debug_hash" in layout.fields


@pytest.mark.parametrize("c_type, expected", [
    ("int", ctypes.c_int), ("unsigned int", ctypes.c_uint),
    ("short", ctypes.c_short), ("unsigned short", ctypes.c_ushort),
    ("long", ctypes.c_long), ("unsigned long", ctypes.c_ulong),
    ("long long", ctypes.c_longlong), ("unsigned long long", ctypes.c_ulonglong),
    ("bool", ctypes.c_bool), ("float", ctypes.c_float),
    ("double", ctypes.c_double), ("size_t", ctypes.c_size_t), ("char", ctypes.c_char),
    ("int8_t", ctypes.c_int8), ("uint8_t", ctypes.c_uint8),
    ("int16_t", ctypes.c_int16), ("uint16_t", ctypes.c_uint16),
    ("int32_t", ctypes.c_int32), ("uint32_t", ctypes.c_uint32),
    ("int64_t", ctypes.c_int64), ("uint64_t", ctypes.c_uint64),
    ("ds4_backend", ctypes.c_int), ("ds4_think_mode", ctypes.c_int),
    ("ds4_log_type", ctypes.c_int),
])
def test_scalar_and_scalar_pointer_mapping(c_type: Any, expected: Any) -> None:
    proto = typemap.parse_prototype(f"{c_type} f({c_type} *value)")
    assert typemap.ctypes_signature(proto) == (expected, [ctypes.POINTER(expected)])


@pytest.mark.parametrize("c_type, expected", [
    ("const char *", ctypes.c_char_p), ("char *", ctypes.POINTER(ctypes.c_char)),
    ("void *", ctypes.c_void_p), ("const void *", ctypes.c_void_p),
    ("FILE *", ctypes.c_void_p), ("const ds4_engine *", ctypes.c_void_p),
    ("ds4_session *", ctypes.c_void_p),
    ("ds4_engine **", ctypes.POINTER(ctypes.c_void_p)),
    ("ds4_session **", ctypes.POINTER(ctypes.c_void_p)),
])
def test_special_pointer_mapping(c_type: Any, expected: Any) -> None:
    assert typemap.ctypes_signature(typemap.parse_prototype(f"{c_type} f(void)")) == (expected, [])


def test_cancel_callback_mapping() -> None:
    proto = typemap.parse_prototype("void f(ds4_session_cancel_fn cancel)")
    assert typemap.ctypes_signature(proto) == (None, [structs.CANCEL_FN])


@pytest.mark.parametrize("c_type", ["void", "FILE", "ds4_engine", "ds4_tokens", "int **", "ds4_widget *"])
def test_unlisted_parameter_types_are_rejected(c_type: Any) -> None:
    proto = typemap.parse_prototype(f"int f({c_type} value)")
    with pytest.raises(TypeMapError, match="in f; add it to"):
        typemap.ctypes_signature(proto)


def test_normal_form_relocates_const_and_accepts_unnamed_parameters() -> None:
    assert typemap.normalize_prototype("char const* f(ds4_tokens const*, uint64_t)") == (
        "const char *f(const ds4_tokens *, uint64_t)")


@pytest.mark.parametrize("text", ["", "int f", "int f(int x", "int f(int x,)", "int f(int x) trailing"])
def test_malformed_prototypes_raise_type_map_error(text: Any) -> None:
    with pytest.raises(TypeMapError):
        typemap.parse_prototype(text)


def test_nested_layout_offsets_include_parent_offset() -> None:
    layout = structs.struct_layout(structs.EngineOptions)
    assert layout.fields["distributed.layers.start"] == (
        structs.EngineOptions.distributed.offset + structs.DistributedOptions.layers.offset)
    assert layout.fields["tp.debug_hash"] == (
        structs.EngineOptions.tp.offset + structs.TpOptions.debug_hash.offset)
