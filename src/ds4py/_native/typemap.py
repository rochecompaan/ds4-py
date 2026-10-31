"""Parse and map the deliberately limited ds4 C declaration vocabulary."""

import ctypes
import re
from dataclasses import dataclass

from ds4py.errors import TypeMapError

from .structs import CANCEL_FN, PROGRESS_FN, STRUCTS

TYPE_MAP_VERSION = 1
_ALL_STRUCTS = STRUCTS

_SCALARS = {
    "int": ctypes.c_int,
    "unsigned int": ctypes.c_uint,
    "short": ctypes.c_short,
    "unsigned short": ctypes.c_ushort,
    "long": ctypes.c_long,
    "unsigned long": ctypes.c_ulong,
    "long long": ctypes.c_longlong,
    "unsigned long long": ctypes.c_ulonglong,
    "bool": ctypes.c_bool,
    "float": ctypes.c_float,
    "double": ctypes.c_double,
    "size_t": ctypes.c_size_t,
    "char": ctypes.c_char,
    "int8_t": ctypes.c_int8,
    "uint8_t": ctypes.c_uint8,
    "int16_t": ctypes.c_int16,
    "uint16_t": ctypes.c_uint16,
    "int32_t": ctypes.c_int32,
    "uint32_t": ctypes.c_uint32,
    "int64_t": ctypes.c_int64,
    "uint64_t": ctypes.c_uint64,
    "ds4_backend": ctypes.c_int,
    "ds4_think_mode": ctypes.c_int,
    "ds4_log_type": ctypes.c_int,
}
_CALLBACKS = {
    "ds4_session_progress_fn": PROGRESS_FN,
    "ds4_session_cancel_fn": CANCEL_FN,
}
_TYPE_WORDS: set[str] = set(["void", "int", "unsigned", "short", "long", "bool", "float", "double", "char", "const"])
_TYPE_WORDS.update(_SCALARS)
_TYPE_WORDS.update(_CALLBACKS)
_TYPE_WORDS.update(_ALL_STRUCTS)
_TYPE_WORDS.update(("FILE", "ds4_engine", "ds4_session"))


@dataclass(frozen=True)
class CType:
    base: str
    pointers: int
    const: bool

    def __str__(self) -> str:
        return ("const " if self.const else "") + self.base + (" " + "*" * self.pointers if self.pointers else "")


@dataclass(frozen=True)
class CPrototype:
    name: str
    restype: CType
    params: tuple[CType, ...]

    def __str__(self) -> str:
        return _declaration(self, self.name)


def _declaration(proto: CPrototype, name: str) -> str:
    params = ", ".join(map(str, proto.params)) or "void"
    separator = "" if proto.restype.pointers else " "
    return f"{proto.restype}{separator}{name}({params})"


def _parse_type(text: str, *, parameter: bool) -> CType:
    tokens = re.findall(r"[A-Za-z_]\w*|\*", text)
    if re.sub(r"\s+", "", text) != "".join(tokens) or not tokens:
        raise TypeMapError(f"invalid C type {text!r}")
    if parameter and len(tokens) > 1 and tokens[-1] != "*" and tokens[-1] not in _TYPE_WORDS:
        tokens.pop()
    pointers = tokens.count("*")
    if pointers:
        first = tokens.index("*")
        if tokens[first:] != ["*"] * pointers:
            raise TypeMapError(f"invalid C type {text!r}")
        tokens = tokens[:first]
    const = "const" in tokens
    tokens = [token for token in tokens if token != "const"]
    if not tokens:
        raise TypeMapError(f"invalid C type {text!r}")
    return CType(" ".join(tokens), pointers, const)


def parse_prototype(text: str) -> CPrototype:
    """Parse a simple function prototype, discarding parameter names."""
    match = re.fullmatch(r"\s*(.+?)\b([A-Za-z_]\w*)\s*\(([^()]*)\)\s*;?\s*", text)
    if not match:
        raise TypeMapError(f"invalid C prototype {text!r}")
    result, name, params = match.groups()
    restype = _parse_type(result, parameter=False)
    if params.strip() in ("", "void"):
        parsed = ()
    else:
        parsed = tuple(_parse_type(part.strip(), parameter=True) for part in params.split(","))
    return CPrototype(name, restype, parsed)


def normalize_prototype(text: str) -> str:
    return str(parse_prototype(text))


def _ctypes_type(ctype: CType, function: str, *, returning: bool = False) -> object | None:
    base, pointers = ctype.base, ctype.pointers
    if base == "void" and pointers == 0 and returning:
        return None
    if pointers == 0:
        if base in _SCALARS:
            return _SCALARS[base]
        if base in _CALLBACKS:
            return _CALLBACKS[base]
        if base == "ds4_eff_logits":
            return _ALL_STRUCTS[base]
    if pointers == 1:
        if base == "char" and ctype.const:
            return ctypes.c_char_p
        if base in ("void", "FILE", "ds4_engine", "ds4_session"):
            return ctypes.c_void_p
        if base in _ALL_STRUCTS:
            return ctypes.POINTER(_ALL_STRUCTS[base])
        if base in _SCALARS:
            return ctypes.POINTER(_SCALARS[base])
    if pointers == 2 and base in ("ds4_engine", "ds4_session") and not ctype.const:
        return ctypes.POINTER(ctypes.c_void_p)
    raise TypeMapError(f"unknown C type '{ctype}' in {function}; add it to ds4py/_native/typemap.py")


def ctypes_signature(proto: CPrototype) -> tuple[object | None, list[object]]:
    return (
        _ctypes_type(proto.restype, proto.name, returning=True),
        [_ctypes_type(param, proto.name) for param in proto.params],
    )


def probe_declaration(proto: CPrototype) -> str:
    return _declaration(proto, "(*ds4py_probe)")
