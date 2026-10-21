# M1: Native Layer and Raw Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `libds4` from a ds4 tree, load it through a checked ctypes layer, and run the engine from Python: sync with progress and cancel, checkpoints and sync plans, a plain decode loop, and `ds4py raw` with the same tokens as `ds4-agent --raw`.

**Architecture:** `ds4py build` links the tree's own objects into a shared library and writes `libds4.abi.json`, which records how each binding's prototype and each struct layout compiled against that tree. `ds4py._native` refuses to load a library whose manifest disagrees with the Python bindings, then wraps the C calls in `NativeEngine` and `NativeSession`. `FakeNative` implements the same interface in pure Python. `ds4py.engine` runs every engine call on one worker thread and exposes jobs that yield events.

**Tech Stack:** Python 3.11+, standard library only (`ctypes`, `threading`, `argparse`, `importlib.metadata`), `uv`, `pytest`, GNU make and a C compiler for `ds4py build`, GitHub Actions.

**Spec:** `docs/specs/2026-10-21-ds4-py-design.md`. M1 scope and done-when are in section 23. The native layer is section 9, the facade section 10, build section 19 and tests section 20.

## Global Constraints

- Python 3.11 or later. `uv` project, src layout. The wheel has no compiled extension.
- The core package `ds4py` imports only the standard library. `pytest` is a dev dependency only.
- Nothing in the ds4 engine changes. `Makefile.lib` lives in this repo and includes the tree's own `Makefile`.
- The engine worker thread is the only thread that calls `libds4`. `SessionHandle` methods run on one thread.
- Pins live in one module, `src/ds4py/abi/pins.py`:
  - upstream `https://github.com/antirez/ds4.git` at `5379c1c9ac315f1540371b06a3db88491d4059ff`
  - downstream fork `https://github.com/pauleveritt/ds4.git` at `cfb2faf5c03199790db7c22166c4c71b549f1e66`
  - known headers: `ds4.h` `fd34c9485866cbf9b55e6e9cefeb0c19db4023de567d9c2196ddc3b91d6bc0d4`, `ds4_kvstore.h` `e6d0e2836385285718436d89292d1052099b7cc9a2c8a1c79d3d59e214d81c3b`. The fork's `ds4.h` (`b1e0ad19…`) is deliberately not known.
- Probe compile flags: `-Werror=incompatible-pointer-types -Werror=implicit-function-declaration`.
- Library resolution order: `ds4py.open(lib=PATH)`, then `DS4_LIB`, then the default directory `~/.ds4py/lib`. Load once per process with `ctypes.CDLL(path, mode=RTLD_LOCAL)`; a second load of a different path is refused.
- Manifest name: `libds4.abi.json`, next to the library. `TYPE_MAP_VERSION = 1`.
- Error buffers passed to C are 512 bytes.
- `Prefill` events arrive at most every 100 ms.
- Checkpoints: `system` plus at most `max_checkpoints = 4` others. A request beyond the limit is refused; nothing is evicted automatically.
- `ds4py raw` defaults match `ds4-agent`: model `ds4flash.gguf`, `ctx` 100000, `n_predict` 50000, temperature 1.0, `top_p` 1.0, `min_p` 0.05, `top_k` 0, think `HIGH`. On GLM (`is_glm_dsa`): `top_p` 0.95 and `min_p` 0.0 unless set.
- Live tests are marked `live` and need `DS4_TEST_MODEL`. Library build tests are marked `build` and need `DS4_SRC_UPSTREAM` or `DS4_SRC_FORK`. Both are skipped when the variable is unset.

## Review Focus

- A wrong model path (missing, a directory, or not a GGUF file): `ds4_engine_open` calls `exit()`, so Python must raise `ModelFileError` before calling it (Task 7).
- Another ds4 process holds the instance lock (`DS4_LOCK_FILE`, default `/tmp/ds4.lock`): the engine would `exit(2)`, so Python must raise `EngineLockedError` naming the lock path (Task 7).
- A raw prompt that does not fit the context: `ds4py raw` prints one line naming the prompt length and context size and exits 2, with no traceback (Task 13).
- Ctrl+C during `ds4py raw`: generation stops, the text so far is flushed, and the process exits 130 (Task 13).
- `ds4py build --src` pointing at a directory that is not a ds4 tree, or with no C compiler or `make`: one error line naming what is missing, before `make` runs (Task 5).

## File Structure

| Path | Responsibility |
| --- | --- |
| `pyproject.toml` | uv project, `ds4py` console script, pytest markers, package data |
| `src/ds4py/__init__.py` | public names: `open`, `ThinkMode`, the errors |
| `src/ds4py/errors.py` | exception hierarchy |
| `src/ds4py/_native/typemap.py` | prototype parsing, normal form, C type → ctypes map, probe declarations |
| `src/ds4py/_native/structs.py` | ctypes mirrors of the ds4 structs, callback types, `struct_layout()` |
| `src/ds4py/_native/bindings.py` | `Binding`, `CORE_BINDINGS`, extension discovery |
| `src/ds4py/_native/loader.py` | find, validate, load once and bind the library |
| `src/ds4py/_native/protocol.py` | `ThinkMode`, `EngineInfo`, `SyncStatus`, the `Native*` protocols |
| `src/ds4py/_native/api.py` | `NativeEngine`, `NativeSession`, `Snapshot` over the real library |
| `src/ds4py/_native/fake.py` | `FakeNative`: the same protocols in pure Python |
| `src/ds4py/abi/pins.py` | pins and known header hashes |
| `src/ds4py/abi/probe.py` | signature probes, layout program, header hashes |
| `src/ds4py/abi/manifest.py` | `Manifest` read/write and `validate()` |
| `src/ds4py/abi/build.py` + `Makefile.lib` | `ds4py build` |
| `src/ds4py/abi/check.py` | `ds4py check`, including `--live` |
| `src/ds4py/engine/transcript.py` | `Transcript`, `Segment` |
| `src/ds4py/engine/plan.py` | `Plan`, `plan_sync()` |
| `src/ds4py/engine/sampling.py` | `SamplingPolicy`, `Rng` |
| `src/ds4py/engine/events.py` | event dataclasses |
| `src/ds4py/engine/worker.py` | `EngineWorker`, jobs, sync, checkpoints, worker calls |
| `src/ds4py/engine/decode.py` | decode loop, `DecodeContext`, `PlainStep` |
| `src/ds4py/engine/session.py` | `Engine`, `Session`, `open()` |
| `src/ds4py/cli.py` | `ds4py build | check | raw` argument parsing |
| `src/ds4py/raw.py` | the raw command and its trace writer |
| `tests/fixtures/ds4-headers/{upstream,fork}/` | `ds4.h`, `ds4_kvstore.h`, `ds4_ssd.h` at the two pins, plus ds4's MIT `LICENSE` |
| `tests/fixtures/ext_site/` | an installed test extension: `ds4py_test_ext` and its `.dist-info` |
| `.github/workflows/ci.yml` | unit tests, CPU builds of both pins, `ds4py check` |

---

### Task 1: Project scaffold, errors, structs and the type map

**Files:**
- Create: `pyproject.toml`, `src/ds4py/__init__.py`, `src/ds4py/errors.py`, `src/ds4py/_native/__init__.py`, `src/ds4py/_native/typemap.py`, `src/ds4py/_native/structs.py`, `tests/conftest.py`
- Test: `tests/native/test_typemap.py`

**Interfaces:**
- Produces:
  - `errors.py`: `Ds4pyError`; `AbiError(Ds4pyError)` with subclasses `TypeMapError`, `AbiMismatch`, `LibraryNotFound`, `LibraryAlreadyLoaded`, `MissingBinding`; `BuildError(Ds4pyError)`; `EngineError(Ds4pyError)` with subclasses `ModelFileError`, `EngineLockedError`; `CheckpointError(Ds4pyError)`.
  - `typemap.py`: `TYPE_MAP_VERSION = 1`; `CType(base: str, pointers: int, const: bool)` with `__str__` giving `"const char *"`; `CPrototype(name: str, restype: CType, params: tuple[CType, ...])` with `__str__` giving the normal form; `parse_prototype(text: str) -> CPrototype`; `normalize_prototype(text: str) -> str`; `ctypes_signature(proto: CPrototype) -> tuple[object, list[object]]` (restype, argtypes); `probe_declaration(proto: CPrototype) -> str`.
  - `structs.py`: `Tokens`, `TokenScore`, `SessionSnapshot`, `DistributedLayers`, `DistributedOptions`, `TpOptions`, `EngineOptions` (each `ctypes.Structure` with `_c_name_`); `STRUCTS: dict[str, type[ctypes.Structure]]` keyed by C name; `PROGRESS_FN = CFUNCTYPE(None, c_void_p, c_char_p, c_int, c_int)`; `CANCEL_FN = CFUNCTYPE(c_bool, c_void_p)`; `StructLayout(size: int, fields: dict[str, int])`; `struct_layout(cls) -> StructLayout`.

- [ ] **Step 1: Scaffold the project**

Run `uv init --package --name ds4py --python ">=3.11"` in the repo root, keep the src layout, add `pytest` with `uv add --dev pytest`, and add to `pyproject.toml`:

```toml
[project.scripts]
ds4py = "ds4py.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "build: builds libds4 from DS4_SRC_UPSTREAM / DS4_SRC_FORK",
  "live: needs a model in DS4_TEST_MODEL",
]
```

`tests/conftest.py` provides a `cc` fixture returning `os.environ.get("CC") or shutil.which("cc")` and skipping when neither exists.

- [ ] **Step 2: Write the failing type-map tests**

```python
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
def test_normal_form(text, normal):
    assert typemap.normalize_prototype(text) == normal

def test_ctypes_signature_maps_pointers_structs_and_callbacks():
    p = typemap.parse_prototype(
        "void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud)")
    assert typemap.ctypes_signature(p) == (None, [ctypes.c_void_p, structs.PROGRESS_FN, ctypes.c_void_p])
    p = typemap.parse_prototype("char *ds4_token_text(ds4_engine *e, int token, size_t *len)")
    assert typemap.ctypes_signature(p) == (
        ctypes.POINTER(ctypes.c_char), [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)])
    p = typemap.parse_prototype("int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen)")
    assert typemap.ctypes_signature(p)[1][1] is ctypes.POINTER(structs.Tokens)

def test_unknown_type_is_an_error_naming_type_and_function():
    with pytest.raises(TypeMapError, match=r"ds4_widget.*ds4_frob"):
        typemap.ctypes_signature(typemap.parse_prototype("int ds4_frob(ds4_widget *w)"))

def test_probe_declaration():
    p = typemap.parse_prototype("char *ds4_token_text(ds4_engine *e, int token, size_t *len)")
    assert typemap.probe_declaration(p) == "char *(*ds4py_probe)(ds4_engine *, int, size_t *)"

def test_struct_layout_walks_nested_fields():
    layout = structs.struct_layout(structs.EngineOptions)
    assert layout.size == ctypes.sizeof(structs.EngineOptions)
    assert layout.fields["model_path"] == 0
    assert "distributed.role" in layout.fields and "tp.debug_hash" in layout.fields
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/native/test_typemap.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ds4py._native'`.

- [ ] **Step 4: Implement `errors.py`, `structs.py` and `typemap.py`**

`structs.py` mirrors `ds4.h` field for field and in order: `ds4_tokens`, `ds4_token_score`, `ds4_session_snapshot`, and `ds4_engine_options` (`ds4.h:131-179`) with its nested `ds4_distributed_layers`, `ds4_distributed_options` and `ds4_tp_options`. Enums are `c_int`; `const char *` fields are `c_char_p`. Task 3's layout test is what proves the mirror. `struct_layout` records every field offset with dotted names for nested structs (`"distributed.role"`, `"distributed.layers.start"`).

Type map, version 1. Anything not listed raises `TypeMapError("unknown C type '<type>' in <function>; add it to ds4py/_native/typemap.py")`:

| C type | ctypes |
| --- | --- |
| `void` (return only) | `None` |
| `int`, `unsigned int`, `short`, `unsigned short`, `long`, `unsigned long`, `long long`, `unsigned long long` | the matching `c_*` type |
| `bool`, `float`, `double`, `size_t`, `char` | `c_bool`, `c_float`, `c_double`, `c_size_t`, `c_char` |
| `int8_t` … `uint64_t` | `c_int8` … `c_uint64` |
| `const char *` | `c_char_p` |
| `char *` | `POINTER(c_char)`, so a returned buffer can be freed and an error buffer can be passed |
| `void *`, `const void *`, `FILE *` | `c_void_p` |
| `ds4_engine *`, `ds4_session *`, with or without `const` | `c_void_p` |
| `ds4_engine **`, `ds4_session **` | `POINTER(c_void_p)` |
| `ds4_backend`, `ds4_think_mode`, `ds4_log_type` | `c_int` |
| pointer to a `STRUCTS` name, with or without `const` | `POINTER(struct)` |
| pointer to a scalar or enum above | `POINTER(scalar)` |
| `ds4_session_progress_fn`, `ds4_session_cancel_fn` | `PROGRESS_FN`, `CANCEL_FN` |

Normal form: single spaces, parameter names dropped, `const` only as a leading qualifier, and a space before the first `*` but none after the last (`char *ds4_token_text(...)`, `ds4_engine **`). A parameter's last word is its name unless it is a C type keyword or a type-map name.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/native/test_typemap.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .python-version src tests
git commit -m "feat(native): add project scaffold, ds4 structs and the C type map"
```

### Task 2: Binding list and extension discovery

**Files:**
- Create: `src/ds4py/_native/bindings.py`, `tests/fixtures/ext_site/ds4py_test_ext/__init__.py`, `tests/fixtures/ext_site/ds4py_test_ext-0.1.dist-info/METADATA`, `tests/fixtures/ext_site/ds4py_test_ext-0.1.dist-info/entry_points.txt`
- Test: `tests/native/test_bindings.py`

**Interfaces:**
- Consumes: `typemap.parse_prototype`, `typemap.ctypes_signature`.
- Produces: `Binding(name: str, prototype: str, required: bool = True, source: str = "core")` (frozen; `.normalized -> str`; construction raises `ValueError` when the prototype's function name differs from `name`); `CORE_BINDINGS: tuple[Binding, ...]`; `ENTRY_POINT_GROUP = "ds4py.bindings"`; `discover_extension_bindings() -> tuple[Binding, ...]`; `all_bindings() -> tuple[Binding, ...]` (core first, then extensions).

- [ ] **Step 1: Write the test extension fixture**

`ds4py_test_ext/__init__.py` defines `BINDINGS` as two `Binding`s: `int ds4_engine_layer_count(ds4_engine *e)` (required; in both pins) and `void ds4_engine_sampling_defaults(ds4_engine *e, float *temperature, int *top_k, float *top_p, float *min_p)` with `required=False` (fork only). `entry_points.txt`:

```ini
[ds4py.bindings]
test_ext = ds4py_test_ext:BINDINGS
```

`METADATA` holds `Metadata-Version: 2.1`, `Name: ds4py-test-ext`, `Version: 0.1`.

- [ ] **Step 2: Write the failing tests**

```python
from ds4py._native import bindings, typemap

def test_core_bindings_are_the_m1_list():
    names = [b.name for b in bindings.CORE_BINDINGS]
    assert len(names) == len(set(names)) == 43
    assert all(b.required for b in bindings.CORE_BINDINGS)
    assert {"ds4_session_sync", "ds4_session_save_snapshot", "ds4_session_checkpoint_valid",
            "ds4_think_mode_parse_level"} <= set(names)

def test_every_core_binding_maps_to_ctypes():
    for b in bindings.CORE_BINDINGS:
        typemap.ctypes_signature(typemap.parse_prototype(b.prototype))

def test_binding_name_must_match_prototype():
    with pytest.raises(ValueError):
        bindings.Binding("ds4_a", "int ds4_b(void)")

def test_no_extensions_installed_by_default():
    assert bindings.discover_extension_bindings() == ()

def test_extension_discovered_through_entry_point(ext_site):
    found = {b.name: b for b in bindings.discover_extension_bindings()}
    assert found["ds4_engine_layer_count"].required
    assert not found["ds4_engine_sampling_defaults"].required
    assert found["ds4_engine_layer_count"].source == "test_ext"
    assert [b.name for b in bindings.all_bindings()][-2:] == list(found)

def test_extension_may_not_redeclare_a_core_binding(ext_site, monkeypatch):
    import ds4py_test_ext
    monkeypatch.setattr(ds4py_test_ext, "BINDINGS",
                        (bindings.Binding("ds4_session_pos", "long ds4_session_pos(ds4_session *s)"),))
    with pytest.raises(ValueError, match="ds4_session_pos"):
        bindings.discover_extension_bindings()
```

Add an `ext_site` fixture to `tests/conftest.py` that runs `monkeypatch.syspath_prepend("tests/fixtures/ext_site")` (absolute path) and clears any cached `importlib.metadata` state.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/native/test_bindings.py -v`
Expected: FAIL with `ImportError: cannot import name 'bindings'`.

- [ ] **Step 4: Implement `bindings.py`**

`CORE_BINDINGS` holds these 43 functions, with prototypes copied exactly from upstream `ds4.h`. All 43 are identical in the fork. Each file keeps the header's parameter names.

- Engine: `ds4_engine_open`, `ds4_engine_close`, `ds4_engine_model_name`, `ds4_engine_model_id`, `ds4_engine_is_deepseek41`, `ds4_engine_is_glm_dsa`, `ds4_engine_is_qwen4`, `ds4_engine_has_vision`, `ds4_engine_routed_quant_bits`, `ds4_engine_mtp_draft_tokens`, `ds4_engine_vocab_size`
- Tokens: `ds4_tokenize_text`, `ds4_tokenize_rendered_chat`, `ds4_token_text`, `ds4_token_eos`, `ds4_token_is_stop`, `ds4_token_is_stop_for_think_mode`, `ds4_tokens_free`, `ds4_tokens_push`
- Session: `ds4_session_create`, `ds4_session_free`, `ds4_session_sync`, `ds4_session_set_progress`, `ds4_session_set_display_progress`, `ds4_session_set_cancel`, `ds4_session_sample`, `ds4_session_argmax`, `ds4_session_eval`, `ds4_session_rewind`, `ds4_session_pos`, `ds4_session_ctx`, `ds4_session_tokens`, `ds4_session_common_prefix`, `ds4_session_invalidate`, `ds4_session_checkpoint_valid`, `ds4_session_save_snapshot`, `ds4_session_load_snapshot`, `ds4_session_snapshot_free`, `ds4_session_payload_bytes`, `ds4_session_copy_logits`, `ds4_session_set_logits`
- Think: `ds4_think_mode_for_context`, `ds4_think_mode_parse_level`

Discovery loads each entry point in group `ds4py.bindings`, expects an iterable of `Binding`, and sets `source` to the entry point's name. It raises `ValueError` naming the function when an extension redeclares a core name, or when two extensions declare one name with different normal forms.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/native -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ds4py/_native/bindings.py tests
git commit -m "feat(native): declare the core binding list and discover extension bindings"
```

### Task 3: Pins, signature probes and the layout program

**Files:**
- Create: `src/ds4py/abi/__init__.py`, `src/ds4py/abi/pins.py`, `src/ds4py/abi/probe.py`, `tests/fixtures/ds4-headers/upstream/{ds4.h,ds4_kvstore.h,ds4_ssd.h,LICENSE}`, `tests/fixtures/ds4-headers/fork/{ds4.h,ds4_kvstore.h,ds4_ssd.h,LICENSE}`
- Test: `tests/abi/test_probe.py`

**Interfaces:**
- Consumes: `Binding`, `CORE_BINDINGS`, `typemap.probe_declaration`, `structs.STRUCTS`, `structs.struct_layout`, `StructLayout`.
- Produces:
  - `pins.py`: `Pin(repo: str, commit: str)`; `UPSTREAM`, `FORK`; `HEADER_FILES = ("ds4.h", "ds4_kvstore.h")`; `KNOWN_HEADERS: tuple[dict[str, str], ...]` (one entry: the upstream pair from Global Constraints).
  - `probe.py`: `ProbeResult(ok: bool, message: str)`; `header_hashes(include_dir: Path) -> dict[str, str]`; `headers_known(hashes: dict[str, str]) -> bool`; `probe_signatures(include_dir: Path, bindings: Sequence[Binding], cc: str, workdir: Path) -> dict[str, ProbeResult]`; `probe_layouts(include_dir: Path, cc: str, workdir: Path) -> dict[str, StructLayout]`.

- [ ] **Step 1: Copy the pinned headers into the fixtures**

For each of `ds4.h`, `ds4_kvstore.h`, `ds4_ssd.h` and `LICENSE`:

```bash
curl -fsSL -o tests/fixtures/ds4-headers/upstream/ds4.h \
  https://raw.githubusercontent.com/antirez/ds4/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4.h
curl -fsSL -o tests/fixtures/ds4-headers/fork/ds4.h \
  https://raw.githubusercontent.com/pauleveritt/ds4/cfb2faf5c03199790db7c22166c4c71b549f1e66/ds4.h
```

Check: `sha256sum tests/fixtures/ds4-headers/*/ds4.h` prints `fd34c948…` for upstream and `b1e0ad19…` for the fork.

- [ ] **Step 2: Write the failing tests**

Fixtures `upstream_headers` and `fork_headers` return the two fixture directories. `doctored(tmp_path, old, new)` copies the upstream headers to `tmp_path` and replaces `old` with `new` exactly once in `ds4.h`, asserting the replacement happened.

```python
def test_core_bindings_compile_against_both_pins(cc, tmp_path, upstream_headers, fork_headers):
    for inc in (upstream_headers, fork_headers):
        results = probe.probe_signatures(inc, CORE_BINDINGS, cc, tmp_path / inc.name)
        assert {n for n, r in results.items() if not r.ok} == set()

def test_fork_only_binding_fails_upstream(cc, tmp_path, upstream_headers, fork_headers, ext_site):
    ext = discover_extension_bindings()
    up = probe.probe_signatures(upstream_headers, ext, cc, tmp_path / "u")
    fk = probe.probe_signatures(fork_headers, ext, cc, tmp_path / "f")
    assert up["ds4_engine_layer_count"].ok and fk["ds4_engine_sampling_defaults"].ok
    assert not up["ds4_engine_sampling_defaults"].ok
    assert "ds4_engine_sampling_defaults" in up["ds4_engine_sampling_defaults"].message

def test_changed_argument_type_fails_its_probe(cc, tmp_path):
    inc = doctored(tmp_path, "int ds4_session_eval(ds4_session *s, int token,",
                             "int ds4_session_eval(ds4_session *s, long token,")
    results = probe.probe_signatures(inc, CORE_BINDINGS, cc, tmp_path / "w")
    assert not results["ds4_session_eval"].ok
    assert "incompatible" in results["ds4_session_eval"].message
    assert all(r.ok for n, r in results.items() if n != "ds4_session_eval")

def test_ctypes_structs_match_the_compiled_layout(cc, tmp_path, upstream_headers, fork_headers):
    for inc in (upstream_headers, fork_headers):
        compiled = probe.probe_layouts(inc, cc, tmp_path / inc.name)
        assert compiled == {name: struct_layout(cls) for name, cls in STRUCTS.items()}

def test_moved_field_changes_the_compiled_layout(cc, tmp_path):
    inc = doctored(tmp_path, "    int n_threads;\n    int context_size;",
                             "    int context_size;\n    int n_threads;")
    compiled = probe.probe_layouts(inc, cc, tmp_path / "w")
    assert compiled["ds4_engine_options"] != struct_layout(EngineOptions)

def test_header_hashes(upstream_headers, fork_headers):
    assert probe.headers_known(probe.header_hashes(upstream_headers))
    assert not probe.headers_known(probe.header_hashes(fork_headers))
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/abi/test_probe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ds4py.abi'`.

- [ ] **Step 4: Implement `pins.py` and `probe.py`**

Each signature probe is one file, `<workdir>/<name>.c`:

```c
#include "ds4.h"
#include "ds4_kvstore.h"
<probe_declaration(proto)> = <name>;
```

It is compiled with `<cc> -std=c11 -c -o /dev/null -Werror=incompatible-pointer-types -Werror=implicit-function-declaration -I<include_dir> <file>`. `ok` is a zero exit code; `message` is the compiler's stderr. Run the compiles in a `ThreadPoolExecutor`.

The layout program includes the same two headers and `<stddef.h>`/`<stdio.h>`. For each `STRUCTS` entry it prints `<struct> size <sizeof>` and one `<struct> field <dotted.path> <offsetof>` line per entry in `struct_layout(cls).fields`, using the dotted path directly as the `offsetof` member designator. Compile it with `<cc> -std=c11 -I<include_dir>`, run it, and parse the output into `StructLayout`s.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/abi/test_probe.py -v`
Expected: PASS. If the layout test fails, fix `structs.py`, not the test.

- [ ] **Step 6: Commit**

```bash
git add src/ds4py/abi tests
git commit -m "feat(abi): probe binding signatures and struct layouts against ds4 headers"
```

### Task 4: Manifest validation and the loader

**Files:**
- Create: `src/ds4py/abi/manifest.py`, `src/ds4py/_native/loader.py`
- Test: `tests/abi/test_manifest.py`, `tests/native/test_loader.py`

**Interfaces:**
- Consumes: `Binding`, `all_bindings()`, `typemap.ctypes_signature`, `StructLayout`, `struct_layout`, `STRUCTS`, `probe.*`.
- Produces:
  - `manifest.py`: `MANIFEST_NAME = "libds4.abi.json"`; `manifest_path(lib: Path) -> Path`; `BindingRecord(prototype: str, required: bool, source: str, ok: bool, message: str = "")`; `Manifest(type_map_version: int, library_file: str, library_sha256: str, source_path: str, source_commit: str | None, backend: str, headers: dict[str, str], headers_known: bool, bindings: dict[str, BindingRecord], layouts: dict[str, StructLayout], live_checks: dict[str, dict] = {})` with `Manifest.read(path) -> Manifest` and `.write(path)`; `Validation(available: frozenset[str], absent: frozenset[str], warnings: tuple[str, ...])`; `validate(m: Manifest, *, lib_sha256: str, bindings: Sequence[Binding], layouts: dict[str, StructLayout]) -> Validation`; `file_sha256(path: Path) -> str`.
  - `loader.py`: `LIB_NAME` (`libds4.dylib` on macOS, else `libds4.so`); `default_lib_dir() -> Path` (`~/.ds4py/lib`); `resolve_lib_path(lib: str | Path | None) -> Path`; `Library` with `.path`, `.manifest`, `.validation`, `.fn(name) -> ctypes function`, `.has(name) -> bool`, `.free(ptr) -> None` (libc `free` found through the library handle); `load_library(lib: str | Path | None = None) -> Library`; module seam `_dlopen(path: Path) -> ctypes.CDLL`.

- [ ] **Step 1: Write the failing manifest tests**

A session fixture `probed(cc, headers_dir, bindings)` runs `probe_signatures` and `probe_layouts` once per header set and caches them. `make_manifest(tmp_path, headers_dir, bindings=CORE_BINDINGS)` writes a dummy `libds4.so` (any bytes) and returns `(lib_path, Manifest)` built from the cached probe results.

```python
def test_matching_manifest_is_accepted(cc, tmp_path, upstream_headers):
    lib, m = make_manifest(tmp_path, upstream_headers)
    v = manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=CORE_BINDINGS, layouts=LAYOUTS)
    assert v.available == {b.name for b in CORE_BINDINGS} and v.warnings == ()

def test_unknown_headers_only_warn(cc, tmp_path, fork_headers):
    lib, m = make_manifest(tmp_path, fork_headers)
    v = manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=CORE_BINDINGS, layouts=LAYOUTS)
    assert any("b1e0ad190897" in w for w in v.warnings)

def test_changed_prototype_is_refused(cc, tmp_path, upstream_headers):
    lib, m = make_manifest(tmp_path, upstream_headers)
    m.bindings["ds4_session_eval"].prototype = "int ds4_session_eval(ds4_session *, long, char *, size_t)"
    with pytest.raises(AbiMismatch) as e:
        manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=CORE_BINDINGS, layouts=LAYOUTS)
    msg = str(e.value)
    assert "ds4_session_eval(ds4_session *, long, char *, size_t)" in msg
    assert "ds4_session_eval(ds4_session *, int, char *, size_t)" in msg
    assert "ds4py build" in msg

def test_moved_field_is_refused(cc, tmp_path):
    inc = doctored(tmp_path, "    int n_threads;\n    int context_size;", "    int context_size;\n    int n_threads;")
    lib, m = make_manifest(tmp_path, inc)
    with pytest.raises(AbiMismatch, match=r"ds4_engine_options\.(n_threads|context_size)"):
        manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=CORE_BINDINGS, layouts=LAYOUTS)

def test_failed_required_probe_is_refused(cc, tmp_path): ...   # doctored `long token` header; message names ds4_session_eval
def test_library_hash_mismatch_is_refused(cc, tmp_path, upstream_headers): ...  # match "ds4py build"
def test_type_map_version_mismatch_is_refused(cc, tmp_path, upstream_headers): ...
def test_unprobed_extension_binding_is_refused(cc, tmp_path, upstream_headers, ext_site): ...  # match "ds4_engine_layer_count"
def test_absent_optional_binding_is_listed(cc, tmp_path, upstream_headers, ext_site):
    lib, m = make_manifest(tmp_path, upstream_headers, bindings=all_bindings())
    v = manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=all_bindings(), layouts=LAYOUTS)
    assert v.absent == {"ds4_engine_sampling_defaults"} and "ds4_engine_layer_count" in v.available

def test_manifest_round_trips(cc, tmp_path, upstream_headers):
    lib, m = make_manifest(tmp_path, upstream_headers)
    m.write(tmp_path / "m.json")
    assert manifest.Manifest.read(tmp_path / "m.json") == m
```

`LAYOUTS = {name: struct_layout(cls) for name, cls in STRUCTS.items()}`.

- [ ] **Step 2: Write the failing loader tests**

```python
def test_resolution_order(tmp_path, monkeypatch):
    monkeypatch.setenv("DS4_LIB", str(tmp_path / "env" / LIB_NAME))
    assert resolve_lib_path(tmp_path / "x.so") == tmp_path / "x.so"
    assert resolve_lib_path(None) == tmp_path / "env" / LIB_NAME
    monkeypatch.delenv("DS4_LIB")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert resolve_lib_path(None) == tmp_path / ".ds4py" / "lib" / LIB_NAME

def test_missing_library_says_how_to_build(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path)); monkeypatch.delenv("DS4_LIB", raising=False)
    with pytest.raises(LibraryNotFound, match="ds4py build --src"):
        load_library()

def test_missing_manifest_is_refused_before_dlopen(tmp_path, monkeypatch):
    (tmp_path / LIB_NAME).write_bytes(b"x")
    monkeypatch.setattr(loader, "_dlopen", lambda p: pytest.fail("dlopen called"))
    with pytest.raises(AbiMismatch, match="libds4.abi.json"):
        load_library(tmp_path / LIB_NAME)

def test_one_library_per_process(cc, tmp_path, upstream_headers, monkeypatch):
    # two valid manifests at two paths; _dlopen patched to a stub whose attributes accept argtypes/restype
    assert load_library(lib_a) is load_library(lib_a)
    with pytest.raises(LibraryAlreadyLoaded, match=str(lib_a)):
        load_library(lib_b)

def test_absent_optional_binding_raises_missing_binding(...):  # ext_site; lib.fn("ds4_engine_sampling_defaults") -> MissingBinding
def test_undeclared_name_raises_missing_binding(...):         # lib.fn("ds4_session_tp_spec_cycle") -> MissingBinding
```

An autouse fixture in `tests/native/test_loader.py` resets `loader._loaded` to `None`.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/abi/test_manifest.py tests/native/test_loader.py -v`
Expected: FAIL with `ImportError` for `ds4py.abi.manifest`.

- [ ] **Step 4: Implement `manifest.py` and `loader.py`**

`validate` checks in this order and raises `AbiMismatch` on the first failure: type-map version, library hash, then for each installed binding: a record exists, the recorded prototype equals `binding.normalized`, and a required binding's probe passed. Then layouts: struct sizes, then each field offset. An optional binding whose probe failed goes in `absent`. Unknown headers add a warning naming the file and the first 12 hex digits of its hash. Every refusal message ends with `` Run `ds4py build` to rebuild the library and its manifest. `` The prototype refusal has this shape:

```text
ds4_session_eval: the library was probed with
  int ds4_session_eval(ds4_session *, long, char *, size_t)
but ds4py declares
  int ds4_session_eval(ds4_session *, int, char *, size_t)
```

`load_library` resolves the path, reads and validates the manifest with `all_bindings()` and the `STRUCTS` layouts, and only then calls `_dlopen(path)`, which is `ctypes.CDLL(str(path), mode=os.RTLD_LOCAL)`. It sets `argtypes` and `restype` from `ctypes_signature` on every available binding. A symbol missing from the library raises `AbiMismatch`. `_loaded` holds the one `Library` for the process.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/abi tests/native -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ds4py/abi/manifest.py src/ds4py/_native/loader.py tests
git commit -m "feat(abi): validate libds4.abi.json and load the library once per process"
```

### Task 5: `ds4py build`

**Files:**
- Create: `src/ds4py/abi/Makefile.lib`, `src/ds4py/abi/build.py`, `src/ds4py/cli.py`, `src/ds4py/__main__.py`
- Test: `tests/abi/test_build.py`

**Interfaces:**
- Consumes: `probe.*`, `Manifest`, `BindingRecord`, `file_sha256`, `all_bindings()`, `LIB_NAME`, `default_lib_dir()`, `pins.UPSTREAM`.
- Produces: `default_backend() -> str` (`"metal"` on macOS, else `"cpu"`); `build_library(src: Path, *, backend: str, out_dir: Path, jobs: int | None = None, cc: str | None = None, log: TextIO = sys.stderr) -> Path` (the installed library); `cli.main(argv: list[str] | None = None) -> int` with subcommand `build --src PATH [--backend cpu|metal|cuda|rocm] [--out DIR] [-j N]`. `--out` defaults to `default_lib_dir()`.

- [ ] **Step 1: Write the failing tests**

```python
def test_not_a_ds4_tree(tmp_path):
    with pytest.raises(BuildError, match="not a ds4 source tree: missing ds4.h"):
        build_library(tmp_path, backend="cpu", out_dir=tmp_path / "out")

def test_no_compiler(tmp_path, fake_tree, monkeypatch):
    monkeypatch.setenv("CC", str(tmp_path / "no-such-cc"))
    with pytest.raises(BuildError, match="no C compiler"):
        build_library(fake_tree, backend="cpu", out_dir=tmp_path / "out")

def test_no_make(tmp_path, fake_tree, monkeypatch): ...      # shutil.which("make") -> None; match "make"
def test_rocm_is_refused(tmp_path, fake_tree): ...           # match "rocm"
def test_metal_needs_macos(tmp_path, fake_tree): ...         # skip on darwin; match "macOS"

def test_changed_argument_type_fails_before_make(cc, tmp_path, monkeypatch):
    tree = doctored(tmp_path, "int ds4_session_eval(ds4_session *s, int token,",
                              "int ds4_session_eval(ds4_session *s, long token,")
    (tree / "Makefile").write_text(""); (tree / "ds4.c").write_text("")
    monkeypatch.setattr(build, "_run_make", lambda *a, **k: pytest.fail("make ran"))
    with pytest.raises(BuildError) as e:
        build_library(tree, backend="cpu", out_dir=tmp_path / "out", cc=cc)
    assert "ds4_session_eval" in str(e.value) and "incompatible" in str(e.value)
    assert not (tmp_path / "out" / LIB_NAME).exists()

def test_mirror_skips_git_and_objects(tmp_path): ...  # build._mirror(src, dst): .git/ and *.o, *.a, *.so, *.dylib not linked; others are symlinks

@pytest.mark.build
def test_cpu_build_of_upstream(built_upstream):
    lib, m = built_upstream
    assert lib.name == LIB_NAME and m.library_sha256 == file_sha256(lib)
    assert all(r.ok for r in m.bindings.values()) and m.headers_known and m.backend == "cpu"
    if (Path(os.environ["DS4_SRC_UPSTREAM"]) / ".git").exists():
        assert m.source_commit == pins.UPSTREAM.commit

@pytest.mark.build
def test_cpu_build_of_fork_with_extension(built_fork):   # built with ext_site on sys.path
    lib, m = built_fork
    assert not m.headers_known and m.bindings["ds4_engine_sampling_defaults"].ok
```

`fake_tree` copies the upstream fixture headers and writes empty `Makefile` and `ds4.c`. `built_upstream` and `built_fork` are session fixtures in `tests/conftest.py` that skip unless `DS4_SRC_UPSTREAM` or `DS4_SRC_FORK` is set, then call `build_library(..., backend="cpu", out_dir=tmp_path_factory.mktemp(...))` and return `(lib_path, Manifest)`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/abi/test_build.py -v`
Expected: FAIL with `ImportError` for `ds4py.abi.build`.

- [ ] **Step 3: Write `Makefile.lib`**

```make
# Links a ds4 tree's own objects into libds4. ds4py build runs it in a
# mirror of the tree: make -f Makefile.lib libds4 LIBDS4_BACKEND=cpu|metal|cuda
include Makefile

LIBDS4_BACKEND ?= cpu
CFLAGS += -fPIC
OBJCFLAGS += -fPIC
NVCCFLAGS += -Xcompiler -fPIC

ifeq ($(UNAME_S),Darwin)
LIBDS4 := libds4.dylib
LIBDS4_SHARED := -dynamiclib
else
LIBDS4 := libds4.so
LIBDS4_SHARED := -shared
endif

ifeq ($(LIBDS4_BACKEND),cpu)
$(LIBDS4): $(CPU_CORE_OBJS) ds4_kvstore.o
	$(CC) $(CFLAGS) $(LIBDS4_SHARED) -o $@ $^ $(LDLIBS)
else ifeq ($(LIBDS4_BACKEND),metal)
$(LIBDS4): $(CORE_OBJS) ds4_kvstore.o
	$(CC) $(CFLAGS) $(LIBDS4_SHARED) -o $@ $^ $(METAL_LDLIBS)
else ifeq ($(LIBDS4_BACKEND),cuda)
$(LIBDS4): $(CORE_OBJS) ds4_kvstore.o
	$(DS4_LINK) $(LIBDS4_SHARED) -o $@ $^ $(DS4_LINK_LIBS)
else
$(error LIBDS4_BACKEND must be cpu, metal or cuda)
endif

.PHONY: libds4
libds4: $(LIBDS4)
```

Make sure `uv build` puts `Makefile.lib` in the wheel (`unzip -l dist/*.whl | grep Makefile.lib`).

- [ ] **Step 4: Implement `build.py`, `cli.py` and `__main__.py`**

`build_library` runs these steps in order and stops at the first error:

1. Preflight. Each failure raises `BuildError` with one line: `<src> is not a ds4 source tree: missing <file>` (checks `Makefile`, `ds4.h`, `ds4_kvstore.h`, `ds4.c`); `no C compiler: set CC or put cc on PATH`; `make not found on PATH`; `backend rocm is not supported by ds4py build yet`; `backend metal needs macOS`.
2. Probe signatures for `all_bindings()` against `src`. If any required binding fails, raise `BuildError` listing each failed name with its compiler output.
3. Mirror the tree with symlinks into `<out_dir>/build/<backend>-<first 12 hex of sha256(resolved src path)>`, skipping `.git` and `*.o`, `*.a`, `*.so`, `*.dylib`. Objects stay between builds, so rebuilds are incremental. Copy `Makefile.lib` in from package data.
4. `_run_make(farm, backend, cc, jobs, log)` runs `make -f Makefile.lib libds4 LIBDS4_BACKEND=<backend> CC=<cc> -j<jobs or os.cpu_count()>`, streaming output to `log`. On failure, raise `BuildError` with the last 40 lines of output.
5. Probe layouts. Copy the library to `<out_dir>/<LIB_NAME>` through a temporary file and `os.replace`, then write the manifest. `source_commit` is `git -C <src> rev-parse HEAD`, or `None` when that fails. Log a warning when the headers are unknown.

`cli.main` uses `argparse` with subparsers. For `build`, it prints the installed path. A `Ds4pyError` prints `ds4py: <message>` to stderr and returns 1.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/abi/test_build.py -v`
Expected: PASS, with the `build` tests skipped.

Then run the real builds against local checkouts of both pins:

Run: `DS4_SRC_UPSTREAM=<upstream tree> DS4_SRC_FORK=<fork tree> uv run pytest tests/abi/test_build.py -v -m build`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ds4py/abi src/ds4py/cli.py src/ds4py/__main__.py tests
git commit -m "feat(abi): add ds4py build with Makefile.lib, probes and manifest"
```

### Task 6: `ds4py check`

**Files:**
- Create: `src/ds4py/abi/check.py`
- Modify: `src/ds4py/cli.py`
- Test: `tests/abi/test_check.py`

**Interfaces:**
- Consumes: `load_library`, `Library`, `Validation`, `Manifest`, `AbiError`.
- Produces: `CheckReport(lib_path: Path, manifest: Manifest | None, validation: Validation | None, extensions: dict[str, str], error: str | None)`; `check_library(lib: str | Path | None) -> CheckReport`; `format_report(r: CheckReport) -> str`; CLI `ds4py check [--lib PATH]`, which prints the report and exits 0, or 1 when `error` is set.

- [ ] **Step 1: Write the failing tests**

A process can load only one library, so the tests that load a real library run the CLI in a subprocess: `run_check(lib, extra_env={}) -> CompletedProcess` calls `[sys.executable, "-m", "ds4py", "check", "--lib", str(lib)]`.

```python
def test_report_lines_for_unknown_headers_and_optional_bindings():
    text = format_report(report_with(headers_known=False, absent={"ds4_engine_sampling_defaults"},
                                     extensions={"ds4_engine_layer_count": "test_ext",
                                                 "ds4_engine_sampling_defaults": "test_ext"}))
    assert "warning: unknown header ds4.h b1e0ad190897" in text
    assert "ds4_engine_sampling_defaults  absent  (test_ext)" in text
    assert "ds4_engine_layer_count  present  (test_ext)" in text
    assert "restore:  not verified" in text

def test_refused_library_exits_1(tmp_path): ...  # lib file without manifest; stdout or stderr contains "libds4.abi.json"; returncode 1

@pytest.mark.build
def test_check_upstream_build(built_upstream):
    r = run_check(built_upstream[0])
    assert r.returncode == 0 and "headers:  known" in r.stdout and "layouts:  ok" in r.stdout

@pytest.mark.build
def test_check_fork_build_lists_extension(built_fork):
    r = run_check(built_fork[0], extra_env={"PYTHONPATH": EXT_SITE})
    assert r.returncode == 0
    assert "warning: unknown header ds4.h b1e0ad190897" in r.stdout
    assert "ds4_engine_sampling_defaults  present  (test_ext)" in r.stdout
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/abi/test_check.py -v`
Expected: FAIL with `ImportError` for `ds4py.abi.check`.

- [ ] **Step 3: Implement `check.py` and the `check` subcommand**

The report has these lines, in this order. Fixed text is exact; `<…>` is filled in:

```text
library:  <path> (<backend>)
ds4 tree: <source path> at <commit[:12]>        # "<source path> (not a git checkout)" when commit is None
headers:  known                                   # or: unknown, followed by one line per warning
          warning: unknown header ds4.h b1e0ad190897; bindings and layouts were checked, but this tree is not a tested pin
layouts:  ok
bindings: <n> required ok
optional: <name>  present|absent  (<source>)      # one line per optional or extension binding; "optional: none" when there are none
restore:  not verified (run `ds4py check --live --model PATH`)
```

Task 14 fills in the `restore:` line from `manifest.live_checks`. When loading fails, the report is `library:  <path>` followed by `error:    <AbiError message>`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/abi/test_check.py -v`, then again with `DS4_SRC_UPSTREAM` and `DS4_SRC_FORK` set and `-m build`.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/abi/check.py src/ds4py/cli.py tests
git commit -m "feat(abi): add ds4py check"
```
