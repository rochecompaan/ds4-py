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

- A bad model file (missing, too small, or a GGUF that ds4 cannot parse): ds4 calls `exit()` while parsing, so a child process opens the model with `inspect_only` first, and Python raises `ModelFileError` with ds4's own message (Task 7).
- Another ds4 process holds the instance lock (`DS4_LOCK_FILE`, default `/tmp/ds4.lock`): the engine would `exit(2)`. Python takes that lock itself and holds it while the engine is open, so there is no gap between the check and the native open. A held lock raises `EngineLockedError` naming the lock path (Task 7).
- A relative model path, such as the default `ds4flash.gguf`: `EngineConfig` makes it absolute on the caller's thread. ds4py never changes the working directory (Task 7).
- A raw prompt that does not fit the context: `ds4py raw` prints one line naming the prompt length and context size and exits 2, with no traceback (Task 13).
- Ctrl+C during `ds4py raw`: generation stops, the text so far is flushed, and the process exits 130 (Task 13).
- An eval that fails during generation: the worker commits a token only after its eval succeeds, so the token reaches neither the transcript nor stdout (Task 11).
- Replacing a checkpoint when the new save fails (for example, out of memory): the worker saves the new snapshot before it closes the old one, so the old checkpoint still restores (Task 10).
- `ds4py build --src` pointing at a directory that is not a ds4 tree, or with no C compiler or `make`: one error line naming what is missing, before `make` runs (Task 5).
- A build that fails or is stopped while it installs: one symlink switch publishes the library and its manifest together, so the previous pair stays installed and loadable (Task 5).

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
| `src/ds4py/_native/api.py` | `NativeEngine`, `NativeSession`, `Snapshot` over the real library, the instance lock |
| `src/ds4py/_native/modelcheck.py` | child process that opens a model with `inspect_only` |
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
- Modify: `tests/conftest.py`
- Test: `tests/abi/test_manifest.py`, `tests/native/test_loader.py`

**Interfaces:**
- Consumes: `Binding`, `all_bindings()`, `typemap.ctypes_signature`, `StructLayout`, `struct_layout`, `STRUCTS`, `probe.*`.
- Produces:
  - `manifest.py`: `MANIFEST_NAME = "libds4.abi.json"`; `manifest_path(lib: Path) -> Path` (`MANIFEST_NAME` beside the file `lib` resolves to, following symlinks); `BindingRecord(prototype: str, required: bool, source: str, ok: bool, message: str = "")`; `Manifest(type_map_version: int, library_file: str, library_sha256: str, source_path: str, source_commit: str | None, backend: str, headers: dict[str, str], headers_known: bool, bindings: dict[str, BindingRecord], layouts: dict[str, StructLayout], live_checks: dict[str, dict] = {})` with `Manifest.read(path) -> Manifest` and `.write(path)` (atomic: a temporary file in the same directory, then `os.replace`); `Validation(available: frozenset[str], absent: frozenset[str], warnings: tuple[str, ...])`; `validate(m: Manifest, *, lib_sha256: str, bindings: Sequence[Binding], layouts: dict[str, StructLayout]) -> Validation`; `file_sha256(path: Path) -> str`.
  - `loader.py`: `LIB_NAME` (`libds4.dylib` on macOS, else `libds4.so`); `default_lib_dir() -> Path` (`~/.ds4py/lib`); `resolve_lib_path(lib: str | Path | None) -> Path`; `Library` with `.path` (the real file, with symlinks resolved), `.manifest`, `.validation`, `.fn(name) -> ctypes function`, `.has(name) -> bool`, `.free(ptr) -> None` (libc `free` found through the library handle); `load_library(lib: str | Path | None = None) -> Library`; module seam `_dlopen(path: Path) -> ctypes.CDLL`.

- [ ] **Step 1: Write the failing manifest tests**

Two fixtures in `tests/conftest.py` build test manifests, because the manifest, loader and install tests (Task 5) all need them. `probed` is a session-scoped dict that caches probe results by header directory and binding names. `make_manifest` depends on `cc` and `probed` and returns a function `(where, headers_dir, bindings=CORE_BINDINGS) -> (lib_path, Manifest)`. That function writes a dummy `LIB_NAME` file (any bytes) in `where`, runs `probe_signatures` and `probe_layouts` only when `probed` has no entry yet, and builds the manifest from the results. Tests that call it take `make_manifest` as a parameter instead of `cc`, so they still skip when there is no compiler.

```python
def test_matching_manifest_is_accepted(make_manifest, tmp_path, upstream_headers):
    lib, m = make_manifest(tmp_path, upstream_headers)
    v = manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=CORE_BINDINGS, layouts=LAYOUTS)
    assert v.available == {b.name for b in CORE_BINDINGS} and v.warnings == ()

def test_unknown_headers_only_warn(make_manifest, tmp_path, fork_headers):
    lib, m = make_manifest(tmp_path, fork_headers)
    v = manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=CORE_BINDINGS, layouts=LAYOUTS)
    assert any("b1e0ad190897" in w for w in v.warnings)

def test_changed_prototype_is_refused(make_manifest, tmp_path, upstream_headers):
    lib, m = make_manifest(tmp_path, upstream_headers)
    m.bindings["ds4_session_eval"].prototype = "int ds4_session_eval(ds4_session *, long, char *, size_t)"
    with pytest.raises(AbiMismatch) as e:
        manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=CORE_BINDINGS, layouts=LAYOUTS)
    msg = str(e.value)
    assert "ds4_session_eval(ds4_session *, long, char *, size_t)" in msg
    assert "ds4_session_eval(ds4_session *, int, char *, size_t)" in msg
    assert "ds4py build" in msg

def test_moved_field_is_refused(make_manifest, tmp_path):
    inc = doctored(tmp_path, "    int n_threads;\n    int context_size;", "    int context_size;\n    int n_threads;")
    lib, m = make_manifest(tmp_path, inc)
    with pytest.raises(AbiMismatch, match=r"ds4_engine_options\.(n_threads|context_size)"):
        manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=CORE_BINDINGS, layouts=LAYOUTS)

def test_failed_required_probe_is_refused(make_manifest, tmp_path): ...   # doctored `long token` header; message names ds4_session_eval
def test_library_hash_mismatch_is_refused(make_manifest, tmp_path, upstream_headers): ...  # match "ds4py build"
def test_type_map_version_mismatch_is_refused(make_manifest, tmp_path, upstream_headers): ...
def test_unprobed_extension_binding_is_refused(make_manifest, tmp_path, upstream_headers, ext_site): ...  # match "ds4_engine_layer_count"
def test_absent_optional_binding_is_listed(make_manifest, tmp_path, upstream_headers, ext_site):
    lib, m = make_manifest(tmp_path, upstream_headers, bindings=all_bindings())
    v = manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=all_bindings(), layouts=LAYOUTS)
    assert v.absent == {"ds4_engine_sampling_defaults"} and "ds4_engine_layer_count" in v.available

def test_manifest_round_trips(make_manifest, tmp_path, upstream_headers):
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

def test_one_library_per_process(make_manifest, tmp_path, upstream_headers, monkeypatch):
    # two valid manifests at two paths; _dlopen patched to a stub whose attributes accept argtypes/restype
    assert load_library(lib_a) is load_library(lib_a)
    with pytest.raises(LibraryAlreadyLoaded, match=re.escape(str(lib_a.resolve()))):  # the loader reports real paths
        load_library(lib_b)

def test_symlink_pairs_with_the_manifest_beside_the_real_file(make_manifest, tmp_path, upstream_headers, monkeypatch):
    (tmp_path / "installs").mkdir()
    real, m = make_manifest(tmp_path / "installs", upstream_headers)
    m.write(manifest_path(real))                              # no manifest beside the symlink
    link = tmp_path / LIB_NAME
    link.symlink_to(real)
    opened = []
    monkeypatch.setattr(loader, "_dlopen", lambda p: opened.append(p) or mock.MagicMock())
    real = real.resolve()                                     # tmp_path may itself sit under a symlink
    assert load_library(link).path == real and opened == [real]

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

`load_library` resolves the path, then follows its symlinks once with `Path.resolve(strict=True)`; a missing file raises `LibraryNotFound`. Every later step uses that real path: it reads the manifest beside it, validates it with `all_bindings()` and the `STRUCTS` layouts, and only then calls `_dlopen(real)`, which is `ctypes.CDLL(str(real), mode=os.RTLD_LOCAL)`. `ds4py build` publishes a library by switching a symlink (Task 5), so following it once means the loader never pairs one build's library with another build's manifest. `Library.path` is the real path, and the one-library rule compares real paths: after a rebuild, a process that has already loaded a library refuses the new one until it restarts. It sets `argtypes` and `restype` from `ctypes_signature` on every available binding. A symbol missing from the library raises `AbiMismatch`. `_loaded` holds the one `Library` for the process.

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
- Consumes: `probe.*`, `Manifest`, `BindingRecord`, `validate`, `manifest_path`, `file_sha256`, `all_bindings()`, `STRUCTS`, `struct_layout`, `AbiMismatch`, `LIB_NAME`, `default_lib_dir()`, `pins.UPSTREAM`.
- Produces: `default_backend() -> str` (`"metal"` on macOS, else `"cpu"`); `build_library(src: Path, *, backend: str, out_dir: Path, jobs: int | None = None, cc: str | None = None, log: TextIO = sys.stderr) -> Path` (the installed library); `install(lib: Path, manifest: Manifest, out_dir: Path) -> Path`, which publishes a library and its manifest as one pair and returns `out_dir / LIB_NAME`; `cli.main(argv: list[str] | None = None) -> int` with subcommand `build --src PATH [--backend cpu|metal|cuda|rocm] [--out DIR] [-j N]`. `--out` defaults to `default_lib_dir()`.

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

def test_one_build_per_out_dir(cc, tmp_path, fake_tree, monkeypatch):
    out = tmp_path / "out"; out.mkdir()
    held = os.open(out / ".build.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(held, fcntl.LOCK_EX)                        # another build is running
    monkeypatch.setattr(build, "_run_make", lambda *a, **k: pytest.fail("make ran"))
    with pytest.raises(BuildError, match="another ds4py build is using"):
        build_library(fake_tree, backend="cpu", out_dir=out, cc=cc)
    os.close(held)

def installed(out):
    lib = out / LIB_NAME
    return lib.read_bytes(), Manifest.read(manifest_path(lib)).library_sha256 == file_sha256(lib)

def test_install_switches_the_library_and_manifest_together(make_manifest, tmp_path, upstream_headers, monkeypatch):
    def pair(name, **changes):                                # a library whose bytes are its name, and its manifest
        (tmp_path / name).mkdir()
        lib, m = make_manifest(tmp_path / name, upstream_headers)
        lib.write_bytes(name.encode())
        return lib, replace(m, library_sha256=file_sha256(lib), **changes)

    out = tmp_path / "out"
    one, two, four = pair("one"), pair("two"), pair("four")
    three = pair("three", type_map_version=0)
    assert build.install(*one, out) == out / LIB_NAME and installed(out) == (b"one", True)
    with monkeypatch.context() as mp:
        mp.setattr(Manifest, "write", lambda self, path: (_ for _ in ()).throw(OSError("disk full")))
        with pytest.raises(OSError, match="disk full"): build.install(*two, out)
    with pytest.raises(BuildError): build.install(*three, out)      # the loader would refuse this pair
    assert installed(out) == (b"one", True)                         # failed installs leave the old pair
    build.install(*four, out)
    assert installed(out) == (b"four", True)
    assert len(list((out / "installs").iterdir())) == 2             # "four" and the "one" it replaced

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
3. Create `out_dir` with its parents, then take an exclusive `flock` on `<out_dir>/.build.lock` without waiting, and hold it until `build_library` returns. If another build holds it, raise `BuildError(f"another ds4py build is using {out_dir}")`: every build into `out_dir` shares the mirror and the installs. Then mirror the tree with symlinks into `<out_dir>/build/<backend>-<first 12 hex of sha256(resolved src path)>`, skipping `.git` and `*.o`, `*.a`, `*.so`, `*.dylib`. Objects stay between builds, so rebuilds are incremental. Copy `Makefile.lib` in from package data.
4. `_run_make(farm, backend, cc, jobs, log)` runs `make -f Makefile.lib libds4 LIBDS4_BACKEND=<backend> CC=<cc> -j<jobs or os.cpu_count()>`, streaming output to `log`. On failure, raise `BuildError` with the last 40 lines of output.
5. Probe layouts and build the `Manifest` for the library that `make` produced. `source_commit` is `git -C <src> rev-parse HEAD`, or `None` when that fails. Log a warning when the headers are unknown.
6. Return `install(<mirror>/<LIB_NAME>, manifest, out_dir)`.

`install` publishes the library and its manifest as one pair. It takes no lock: its caller holds `<out_dir>/.build.lock`, as `build_library` does, so no other build can remove the new directory before the switch.

1. Create `<out_dir>/installs` and its parents if needed, then make a new directory in it with `tempfile.mkdtemp`. Copy the library into it as `LIB_NAME` and write the manifest beside it.
2. Check the staged pair as the loader would: `validate(Manifest.read(<staged manifest>), lib_sha256=file_sha256(<staged library>), bindings=all_bindings(), layouts=<STRUCTS layouts>)`. An `AbiMismatch` becomes a `BuildError` with the same message, so a build never publishes a library that ds4py would refuse to load.
3. Remove any stale `<out_dir>/.<LIB_NAME>.new` left by a killed build, create a symlink there with the relative target `installs/<dir>/<LIB_NAME>`, and `os.replace` it onto `<out_dir>/<LIB_NAME>`. The build lock makes the fixed temporary name safe. This rename is the only switch: before it, readers see the old pair, and after it, the new one. The loader follows the symlink once and reads the manifest beside the real file (Task 4). So where the spec puts `libds4.abi.json` "next to the library", this plan means next to the real file in `installs/<dir>`, not next to the symlink.
4. Remove every other directory in `installs`, except the one the replaced symlink pointed to. Step 3 reads that target with `os.readlink` just before the switch; there is none when `<out_dir>/<LIB_NAME>` did not exist. A process that followed the old symlink just before the switch can still load that library, and a library that a process has already loaded stays mapped after its file is removed. Removal errors are ignored, because the switch has already happened.

If a step before the switch fails, `install` removes the new directory and the temporary symlink, then re-raises, so the previous pair stays installed and loadable. After the switch, nothing changes an installed pair except `live_checks` in its manifest (Task 14), and `Manifest.write` replaces that file atomically.

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

The report tests need no library. `report_with(*, headers_known=True, absent=frozenset(), extensions={}) -> CheckReport` builds one by hand: a `Manifest` with backend `cpu`, source path `/src/ds4`, commit `UPSTREAM.commit`, the known header hashes (with `ds4.h` replaced by the fork's `b1e0ad190897de401780d02d246e610882bca676a531bd22d2e06b3769c52d9b` when `headers_known` is false), a passing `BindingRecord` for every name in `CORE_BINDINGS` and in `absent`, `headers_known` to match, and no layouts; a `Validation` with the core bindings available, `absent`, and, when `headers_known` is false, the warning `validate` writes for the fork's `ds4.h`; and the given `extensions`.

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

### Task 7: Native protocols and the ctypes API

**Files:**
- Create: `src/ds4py/_native/protocol.py`, `src/ds4py/_native/api.py`, `src/ds4py/_native/modelcheck.py`, `tests/native/contract.py`
- Modify: `tests/conftest.py`
- Test: `tests/native/test_api.py`

**Interfaces:**
- Consumes: `Library`, `load_library`, `structs.*`, `EngineError`, `ModelFileError`, `EngineLockedError`, `MissingBinding`.
- Produces:
  - `protocol.py`: `ThinkMode(IntEnum)` with `NONE=0, HIGH=1, MAX=2, LOW=3, MEDIUM=4`; `THINK_LEVEL_BASE = 1000`; `BACKENDS = {"metal": 0, "cuda": 1, "rocm": 1, "cpu": 2}`; `EngineConfig(model: Path, ctx: int = 100_000, backend: str = "cpu", threads: int = 0)` (frozen; `__post_init__` makes `model` absolute with `Path.absolute()`, so a relative path keeps the meaning it had on the caller's thread); `EngineInfo(name: str, model_id: int, family: str, is_deepseek41: bool, is_glm_dsa: bool, is_qwen4: bool, has_vision: bool, routed_quant_bits: int, mtp_draft_tokens: int, vocab_size: int)` (frozen); `family_of(*, deepseek41: bool, glm: bool, qwen: bool) -> str` (`"deepseek41"`, `"glm"`, `"qwen"`, else `"deepseek4"`); `SyncStatus(Enum)` with `DONE`, `INTERRUPTED`; `ProgressFn = Callable[[str, int, int], None]`; `CancelFn = Callable[[], bool]`; protocols `Snapshot`, `SessionHandle`, `EngineHandle`, `Native` with the methods below.
  - `api.py`: `lock_path() -> Path`; `engine_lock_path() -> Path`; `InstanceLock` with `InstanceLock.acquire(path: Path) -> InstanceLock`, `.close()` and context-manager use; `metal_source_env(src: Path) -> dict[str, str]`; `engine_options(config: EngineConfig, *, inspect_only: bool = False) -> structs.EngineOptions`; `LibNative(lib: Library)`, `NativeEngine`, `NativeSession`, `NativeSnapshot` implementing the protocols.
  - `modelcheck.py`: `NATIVE_REFUSED = 3`; `PYTHON_ERROR = 4`; `exit_error(status: int, stderr: str, model: Path, lock_file: Path) -> EngineError | None`; `inspect_model(lib: Path, config: EngineConfig, lock_file: Path) -> None`; `main(argv: list[str]) -> int`, run as `python -m ds4py._native.modelcheck`.

The protocols, which `FakeNative` also implements in Task 8:

```python
class Snapshot(Protocol):
    nbytes: int
    def close(self) -> None: ...

class SessionHandle(Protocol):
    raw: int                                  # session pointer, for extensions
    def sync(self, tokens: Sequence[int], *, progress: ProgressFn | None = None,
             cancel: CancelFn | None = None) -> SyncStatus: ...
    def sample(self, temperature: float, top_k: int, top_p: float, min_p: float,
               rng: ctypes.c_uint64) -> int: ...
    def argmax(self) -> int: ...
    def eval(self, token: int) -> None: ...
    def rewind(self, pos: int) -> None: ...
    def pos(self) -> int: ...
    def ctx(self) -> int: ...
    def tokens(self) -> list[int]: ...
    def common_prefix(self, tokens: Sequence[int]) -> int: ...
    def checkpoint_valid(self) -> bool: ...
    def invalidate(self) -> None: ...
    def payload_bytes(self) -> int: ...
    def save_snapshot(self) -> Snapshot: ...
    def load_snapshot(self, snap: Snapshot) -> None: ...
    def close(self) -> None: ...

class EngineHandle(Protocol):
    raw: int                                  # engine pointer, for extensions
    def info(self) -> EngineInfo: ...
    def tokenize(self, text: str, *, rendered: bool = False) -> list[int]: ...
    def token_text(self, token: int) -> bytes: ...
    def eos(self) -> int: ...
    def is_stop(self, token: int) -> bool: ...
    def create_session(self, ctx: int) -> SessionHandle: ...
    def close(self) -> None: ...

class Native(Protocol):
    backend: str                              # the backend the library was built for
    restore_enabled: bool
    def open_engine(self, config: EngineConfig) -> EngineHandle: ...
    def bind(self, name: str) -> Callable[..., Any]: ...
    def parse_think_level(self, text: str) -> int | None: ...
    def think_mode_for_context(self, mode: int, ctx: int) -> int: ...
```

`copy_logits`, `set_logits` and the payload functions are bound by the manifest but get Python wrappers in later milestones.

- [ ] **Step 1: Write the shared contract**

`tests/native/contract.py` holds behaviour that both the real library and `FakeNative` must show. Each function takes an `EngineHandle` and a list of prompt tokens with at least 8 entries, whose last token differs from its first. ds4's progress events differ by backend: the CPU path reports one event per extended token and a single `(len, len)` event for a rebuild, while Metal reports the start position first. So the contract checks reuse by its result: after an extension or a restore, the next greedy token must match a full rebuild.

```python
def rebuilt_argmax(s, tokens):
    s.invalidate(); s.sync(tokens)
    return s.argmax()

def check_extend_matches_rebuild(engine, prompt):
    s = engine.create_session(len(prompt) + 64)
    assert s.sync(prompt[:-2]) is SyncStatus.DONE and s.pos() == len(prompt) - 2
    events = []
    assert s.sync(prompt, progress=lambda *e: events.append(e)) is SyncStatus.DONE
    assert s.tokens() == list(prompt) and s.checkpoint_valid()
    chunks = [e for e in events if e[0] == "prefill_chunk"]    # Metal interleaves prefill_display
    assert chunks[-1] == ("prefill_chunk", len(prompt), len(prompt))
    assert s.argmax() == rebuilt_argmax(s, prompt)

def check_common_prefix(engine, prompt):
    assert prompt[-1] != prompt[0]
    s = engine.create_session(len(prompt) + 64)
    s.sync(prompt)
    diverged = prompt[:-1] + [prompt[0]]
    assert s.common_prefix(diverged) == len(prompt) - 1
    s.sync(diverged); assert s.tokens() == diverged
    s.invalidate(); assert s.common_prefix(diverged) == 0

def check_eval_rewind_and_greedy(engine, prompt):
    s = engine.create_session(len(prompt) + 64)
    s.sync(prompt)
    tok = s.argmax()
    assert s.sample(0.0, 0, 1.0, 0.0, ctypes.c_uint64(1)) == tok      # temperature 0 is greedy
    s.eval(tok); assert s.pos() == len(prompt) + 1
    s.rewind(len(prompt)); assert s.tokens() == list(prompt)
    s.sync(prompt); assert s.argmax() == tok      # rewind usually invalidates; sync rebuilds

def check_snapshot_restore_matches_rebuild(engine, prompt):
    s = engine.create_session(len(prompt) + 64)
    s.sync(prompt[:4]); snap = s.save_snapshot()
    assert snap.nbytes > 0 and s.payload_bytes() > 0
    s.sync(prompt[:2] + prompt[:2])
    s.load_snapshot(snap); snap.close()
    with pytest.raises(EngineError, match="closed"): s.load_snapshot(snap)   # and the session is unchanged
    assert s.tokens() == list(prompt[:4]) and s.checkpoint_valid()
    s.sync(prompt)
    assert s.argmax() == rebuilt_argmax(s, prompt)

def check_cancel_interrupts(engine, prompt):
    s = engine.create_session(len(prompt) + 64)
    assert s.sync(prompt, cancel=lambda: True) is SyncStatus.INTERRUPTED
    assert s.pos() < len(prompt)

def check_context_errors(engine, prompt):
    s = engine.create_session(len(prompt))
    with pytest.raises(EngineError): s.sync(prompt)           # needs one free slot
    with pytest.raises(EngineError): s.sync([])

CONTRACT = [check_extend_matches_rebuild, check_common_prefix, check_eval_rewind_and_greedy,
            check_snapshot_restore_matches_rebuild, check_cancel_interrupts, check_context_errors]
```

- [ ] **Step 2: Write the failing API tests**

Add a `fresh_loader` fixture to `tests/conftest.py` that sets `loader._loaded = None` before and after the test, so tests can load different libraries in one process.

```python
def test_instance_lock(tmp_path, monkeypatch):
    monkeypatch.delenv("DS4_LOCK_FILE", raising=False)
    assert lock_path() == Path("/tmp/ds4.lock")
    path = tmp_path / "ds4.lock"
    monkeypatch.setenv("DS4_LOCK_FILE", str(path))
    assert (lock_path(), engine_lock_path()) == (path, tmp_path / "ds4.lock.engine")
    with open(path, "w") as held:
        held.write("4242"); held.flush()
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(EngineLockedError, match=rf"{re.escape(str(path))}.*pid 4242"):
            InstanceLock.acquire(path)
    with InstanceLock.acquire(path):
        assert path.read_text() == f"{os.getpid()}\n"      # a C tool's refusal names this process
        with pytest.raises(EngineLockedError):              # a second descriptor conflicts, as ds4's would
            InstanceLock.acquire(path)
    InstanceLock.acquire(path).close()                      # closing released it

def test_exit_error():
    m, lk = Path("/m.gguf"), Path("/tmp/ds4.lock.engine")
    assert exit_error(0, "", m, lk) is None
    e = exit_error(1, "ds4: loading\nds4: only GGUF v3 is supported\n", m, lk)
    assert type(e) is ModelFileError and "/m.gguf" in str(e) and str(e).endswith("only GGUF v3 is supported")
    e = exit_error(2, "ds4: another ds4 process is already running (pid 7); refusing to start\n", m, lk)
    assert type(e) is EngineLockedError and str(lk) in str(e) and "pid 7" in str(e)
    assert type(exit_error(NATIVE_REFUSED, "ds4: V4.1 vision requires Metal\n", m, lk)) is EngineError
    tb = "Traceback (most recent call last):\n  ...\nImportError: no module named ds4py\n"
    for status in (1, 2, NATIVE_REFUSED, PYTHON_ERROR, -9):      # no ds4 line: not a ds4 refusal
        e = exit_error(status, tb, m, lk)
        assert type(e) is EngineError and "ImportError" in str(e)

def test_metal_source_env(tmp_path, monkeypatch):
    (tmp_path / "metal").mkdir()
    for stem in ("flash_attn", "moe"):
        (tmp_path / "metal" / f"{stem}.metal").write_text("")
    monkeypatch.chdir(tmp_path)
    here = Path.cwd()                  # tmp_path may be a symlink, as on macOS
    assert metal_source_env(Path(".")) == {
        "DS4_METAL_FLASH_ATTN_SOURCE": str(here / "metal" / "flash_attn.metal"),
        "DS4_METAL_MOE_SOURCE": str(here / "metal" / "moe.metal"),
    }
    with pytest.raises(EngineError, match="no Metal sources"):
        metal_source_env(tmp_path / "metal")

def test_engine_options(tmp_path, monkeypatch):
    o = engine_options(EngineConfig(Path("/m.gguf"), ctx=4096, backend="rocm", threads=3))
    assert (o.model_path, o.backend, o.context_size, o.placement_ctx_hint, o.n_threads) == (b"/m.gguf", 1, 4096, 4096, 3)
    assert (o.mtp_draft_tokens, o.mtp_margin, o.inspect_only) == (1, 3.0, False)
    assert engine_options(EngineConfig(Path("/m.gguf")), inspect_only=True).inspect_only
    monkeypatch.chdir(tmp_path)
    assert EngineConfig(Path("m.gguf")).model == Path.cwd() / "m.gguf"   # absolute on the caller's thread
    with pytest.raises(ValueError, match="vulkan"): engine_options(EngineConfig(Path("/m"), backend="vulkan"))

@pytest.mark.build
def test_open_failures_become_exceptions(built_upstream, fresh_loader, tmp_path, monkeypatch):
    lock = tmp_path / "ds4.lock"
    monkeypatch.setenv("DS4_LOCK_FILE", str(lock))
    native = LibNative(load_library(built_upstream[0]))
    assert native.parse_think_level("42") == 1042 and native.parse_think_level("high") is None
    assert native.think_mode_for_context(ThinkMode.MAX, 4096) == ThinkMode.HIGH
    (tmp_path / "v0.gguf").write_bytes(b"GGUF" + bytes(60))        # GGUF version 0
    for model, text in [("missing.gguf", "cannot open model"), ("v0.gguf", "only GGUF v3")]:
        with pytest.raises(ModelFileError, match=text):           # ds4 exit()s; only the child dies
            native.open_engine(EngineConfig(tmp_path / model))
    with InstanceLock.acquire(lock):                               # another holder of the real lock
        with pytest.raises(EngineLockedError):
            native.open_engine(EngineConfig(tmp_path / "v0.gguf"))
    with InstanceLock.acquire(engine_lock_path()):                 # ds4's own lock: the child exits 2
        with pytest.raises(EngineLockedError, match="already running"):
            native.open_engine(EngineConfig(tmp_path / "v0.gguf"))
    InstanceLock.acquire(lock).close()                             # the failed opens released it

@pytest.mark.live
@pytest.mark.parametrize("check", CONTRACT, ids=lambda f: f.__name__)
def test_contract_live(live_engine, check):
    check(live_engine, live_engine.tokenize("The quick brown fox jumps over the lazy dog. " * 2))
```

`live_engine` is a module fixture that skips unless `DS4_TEST_MODEL` is set, loads the library from `DS4_LIB` or the default directory, and opens an engine on the manifest's backend with `ctx=4096`. `test_contract_live` skips `check_snapshot_restore_matches_rebuild` when `LibNative.restore_enabled` is `False`. Add two more live tests: `engine.token_text` over `engine.tokenize("Hello world")` joins to `b"Hello world"`; and while `live_engine` is open, `InstanceLock.acquire(lock_path())` raises `EngineLockedError` and `os.environ.get("DS4_LOCK_FILE")` is what it was before the open.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/native/test_api.py -v`
Expected: FAIL with `ImportError` for `ds4py._native.api`.

- [ ] **Step 4: Implement `protocol.py`, `api.py` and `modelcheck.py`**

ds4 calls `exit()` when another process holds the instance lock (`ds4_acquire_instance_lock`, `ds4.c:61061`) and when it cannot parse a model (`model_open`, `ds4.c:2855`, through `ds4_die`). Python cannot catch either, and a Python copy of ds4's checks would always miss cases. `LibNative.open_engine` therefore runs these steps in order, under a module-level `threading.Lock` so that two opens in one process cannot overlap:

1. `lock = InstanceLock.acquire(lock_path())`. Python takes the lock that every other ds4 process checks and keeps it until the engine closes, so no process can take it between the check and the native open.
2. `inspect_model(library.path, config, engine_lock_path())`. A child process lets ds4 parse the model; if ds4 exits, only the child dies.
3. For the Metal backend, add `metal_source_env(Path(manifest.source_path))` to `os.environ` with `setdefault`, and leave it set.
4. Set `DS4_LOCK_FILE` to `engine_lock_path()`, call `ds4_engine_open`, and restore the previous value (or remove the variable) in `finally`. A non-zero return raises `EngineError`.
5. If any step fails, close `lock` before re-raising. On success, `NativeEngine` owns `lock`.

Details:

- `lock_path()` is `DS4_LOCK_FILE`, or `/tmp/ds4.lock` when that is unset or empty, as in ds4. `engine_lock_path()` is `lock_path()` with `.engine` appended to the name. ds4 opens its lock file again and calls `flock` on the new descriptor, which conflicts with the lock Python holds even inside one process, so ds4's own lock must use a different file. Only the holder of the real lock ever opens the engine file, so a fixed name is safe.
- `InstanceLock.acquire(path)` opens `path` with `O_RDWR | O_CREAT`, mode `0600` (Python descriptors are not inherited, like ds4's `FD_CLOEXEC`), and calls `fcntl.flock(fd, LOCK_EX | LOCK_NB)`. On `BlockingIOError` it reads a pid from the file, closes it and raises `EngineLockedError("another ds4 process holds <path> (pid <n>); stop it or set DS4_LOCK_FILE")`, leaving out `(pid <n>)` when there is none. On success it truncates the file and writes its own pid, as ds4 does, so a C tool's refusal names this process. `close()` closes the descriptor once, which releases the lock.
- `inspect_model` runs `[sys.executable, "-P", "-m", "ds4py._native.modelcheck", <lib>, <model>, <backend>]` with the current environment plus `DS4_LOCK_FILE=<lock_file>`, captures its output, and raises `exit_error(returncode, stderr, model, lock_file)` when that is not `None`. `modelcheck.main` loads the library, calls `ds4_engine_open` with `engine_options(config, inspect_only=True)` and closes the engine. With `inspect_only`, ds4 parses the whole model without prefetching weights and returns before it starts the GPU (`ds4.c:72293`). `main` returns 0 when the open succeeds and `NATIVE_REFUSED` when `ds4_engine_open` returns non-zero. It catches any Python exception, prints the traceback and returns `PYTHON_ERROR`, so only ds4's own `exit()` calls produce statuses 1 and 2. It reads its three arguments by position, with no argument parser that could exit with 2.
- `exit_error` treats a failure as a ds4 refusal only when stderr has a line starting with `ds4: `, and the exception carries the last such line. Status 2 comes only from ds4's lock code (`ds4.c:61068` to `61098`) and raises `EngineLockedError` naming `lock_file`. Status 1 comes from `ds4_die` and from ds4's metadata and tensor checks, and raises `ModelFileError` naming the model path. `NATIVE_REFUSED` raises `EngineError`, as a failed native open does in step 4. Any other status, or a failure with no `ds4: ` line, raises `EngineError` with the status and the last 20 lines of stderr. That covers a signal, and a failed import, which Python reports with status 1. ds4 also exits with 1 for `out of memory` in `xmalloc`; that case raises `ModelFileError` as well, and the message shows ds4's own words.
- `metal_source_env(src)` maps each `<src>/metal/<stem>.metal` to `DS4_METAL_<STEM in upper case>_SOURCE`, built on `src.absolute()`, so every value is an absolute path. This matches all 26 entries in the source table of `ds4_metal.m` at the upstream pin (`ds4_metal.m:4763`), and ds4 tries an override before its relative path. When `<src>/metal` holds no `.metal` files, it raises `EngineError("no Metal sources in <src>/metal")`. The variables stay set because ds4 may start the GPU later, for example for vision (`ds4.c:73608`). A value the user has already set wins.
- ds4py never changes the working directory, so relative paths on other threads keep their meaning. CI does not run the Metal path; Task 14's live check on the bake-off machine does.
- Step 4 changes `DS4_LOCK_FILE` for the whole process while `ds4_engine_open` runs, because ds4 reads the lock path only from the environment, and only there (`ds4.c:61062`). If another thread starts a ds4 process in that window, the child can take the engine file first, and then the native open exits. ds4py starts no processes while it opens an engine, and the `open_engine` docstring tells callers not to either.
- ds4 exits after the model has parsed, such as inside GPU setup, are not caught. They stop the process, as they do in the C tools.

`engine_options` zero-fills the struct and sets `model_path`, `backend` (from `BACKENDS`; any other name raises `ValueError`), `n_threads`, `context_size`, `placement_ctx_hint = ctx`, `mtp_draft_tokens = 1` and `mtp_margin = 3.0` as `ds4_agent.c` does, and `inspect_only` from its argument. `NativeEngine` keeps the options struct and its encoded path alive for its lifetime.

`NativeSession`:
- Creates one `PROGRESS_FN` and one `CANCEL_FN` object in `__init__` and keeps them on the instance. `sync` points them at the call's `progress` and `cancel`, installs them with `ds4_session_set_progress`, `ds4_session_set_display_progress` and `ds4_session_set_cancel`, and clears all three with `PROGRESS_FN()` / `CANCEL_FN()` in `finally`.
- The callbacks never raise into C. They store the first exception, make cancel return `True`, and `sync` re-raises it after the C call returns.
- `sync` returns `DONE` for 0 and `INTERRUPTED` for `DS4_SESSION_SYNC_INTERRUPTED` (2). Any other code raises `EngineError` with the text from the 512-byte error buffer. `eval`, `save_snapshot` and `load_snapshot` raise the same way.
- Token lists cross into C as a `Tokens` struct over a `(c_int * n)` array that lives for the call. `tokenize` copies `v[:len]` out and then calls `ds4_tokens_free`. `token_text` copies the returned bytes and frees them with `Library.free`. `tokens()` copies the borrowed `ds4_session_tokens` array.
- `NativeEngine.is_stop` wraps `ds4_token_is_stop`, which the `ds4-agent --raw` loop uses. The think-mode variant gets a wrapper with chat rendering in M2.
- Every `NativeEngine` and `NativeSession` method asserts it runs on the thread that created the object.
- `close()` frees the session once; `__del__` calls `close()`. `NativeEngine.close()` calls `ds4_engine_close` once and then closes its `InstanceLock`, so the real lock is released only after ds4 has released the engine file. `NativeSnapshot.close()` calls `ds4_session_snapshot_free` once and marks the snapshot closed; `load_snapshot` of a closed snapshot raises `EngineError("snapshot is closed")` instead of handing freed memory to ds4. `nbytes` is the struct's `len`.

`LibNative.backend` is `manifest.backend`. `LibNative.restore_enabled` is `False` only when `manifest.live_checks["restore"]["ok"]` is `False`. `LibNative.bind(name)` returns `Library.fn(name)`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/native/test_api.py -v`
Expected: PASS, with `build` and `live` tests skipped.

Run: `DS4_SRC_UPSTREAM=<upstream tree> uv run pytest tests/native/test_api.py -v -m build`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ds4py/_native tests
git commit -m "feat(native): wrap the ds4 engine and session API with ctypes"
```

### Task 8: `FakeNative`

**Files:**
- Create: `src/ds4py/_native/fake.py`
- Test: `tests/native/test_fake.py`

**Interfaces:**
- Consumes: the Task 7 protocols, `all_bindings()`, `MissingBinding`, `EngineError`.
- Produces: `FAKE_EOS = 0`; `Script = Callable[[Sequence[int], Sequence[int]], int]` (prompt tokens, generated tokens → next token); `reply(text: bytes) -> Script` (emits the bytes, then `FAKE_EOS`); `FakeClock` (`__call__() -> float`, `advance(seconds)`); `FakeNative(*, script: Script = reply(b"ok"), chunk: int = 4, clock: FakeClock | None = None, chunk_seconds: float = 0.0, bindings: Mapping[str, Callable] = {}, absent: Iterable[str] = (), restore_enabled: bool = True, info: EngineInfo | None = None)`; `FakeEngine`, `FakeSession` (with `calls: list[tuple]`), `FakeSnapshot` (with `closed: bool`).

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("check", CONTRACT, ids=lambda f: f.__name__)
def test_contract(check):
    engine = FakeNative().open_engine(EngineConfig(Path("fake.gguf")))
    check(engine, engine.tokenize("The quick brown fox"))

def test_tokenizer_round_trips_bytes():
    e = FakeNative().open_engine(EngineConfig(Path("f")))
    assert b"".join(e.token_text(t) for t in e.tokenize("héllo")) == "héllo".encode()
    assert e.eos() == FAKE_EOS and e.is_stop(FAKE_EOS) and not e.is_stop(e.tokenize("a")[0])

def test_progress_events_follow_ds4():
    clock = FakeClock()
    s = FakeNative(chunk=4, clock=clock, chunk_seconds=0.05).open_engine(EngineConfig(Path("f"))).create_session(64)
    s.sync(list(range(1, 7)))
    events = []
    s.sync(list(range(1, 17)), progress=lambda *e: events.append(e))
    assert events == [("prefill_chunk", 6, 16), ("prefill_chunk", 10, 16),
                      ("prefill_chunk", 14, 16), ("prefill_chunk", 16, 16)]
    assert clock() == pytest.approx(0.05 * 2 + 0.05 * 3)       # 2 chunks, then 3

def test_cancel_keeps_the_partial_prefix_valid():
    s = FakeNative(chunk=4).open_engine(EngineConfig(Path("f"))).create_session(64)
    seen = []
    status = s.sync(list(range(1, 17)), progress=lambda *e: seen.append(e), cancel=lambda: len(seen) >= 2)
    assert status is SyncStatus.INTERRUPTED and s.pos() == 4 and s.checkpoint_valid()

def test_script_drives_sampling_and_calls_are_logged():
    s = FakeNative(script=reply(b"hi")).open_engine(EngineConfig(Path("f"))).create_session(64)
    s.sync([5, 6])
    out = []
    while (t := s.sample(1.0, 0, 1.0, 0.05, ctypes.c_uint64(3))) != FAKE_EOS:
        s.eval(t); out.append(t)
    assert bytes(t - 1 for t in out) == b"hi"
    assert ("eval", FAKE_EOS) not in s.calls and s.calls[0] == ("sync", 2)

def test_bind(ext_site):
    native = FakeNative(bindings={"ds4_engine_layer_count": lambda e: 43},
                        absent={"ds4_engine_sampling_defaults"})
    assert native.bind("ds4_engine_layer_count")(None) == 43
    with pytest.raises(MissingBinding, match="absent"): native.bind("ds4_engine_sampling_defaults")
    with pytest.raises(MissingBinding, match="does not declare"): native.bind("ds4_session_tp_spec_cycle")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/native/test_fake.py -v`
Expected: FAIL with `ModuleNotFoundError` for `ds4py._native.fake`.

- [ ] **Step 3: Implement `fake.py`**

- Tokenizer: byte `b` is token `b + 1`; token 0 is `FAKE_EOS` with text `b"</s>"`; `vocab_size` 257. `rendered=True` tokenizes the same way. `is_stop` is true only for `FAKE_EOS`. The default `EngineInfo` has name `"fake"`, family `"deepseek4"` and `mtp_draft_tokens` 1.
- `sync` follows ds4's rules. Empty input, or input with `len >= ctx`, raises `EngineError`. State is kept only when the checkpoint is valid and the new tokens start with all of it; otherwise prefill starts at 0. Progress follows ds4's Metal path: `("prefill_chunk", start, total)` first, then one event after each `chunk` tokens, with absolute positions. The fake emits no `prefill_display` events. Before each chunk, the fake checks `cancel` and returns `INTERRUPTED`, keeping the prefix done so far as a valid checkpoint. Each chunk advances `clock` by `chunk_seconds`. The sync also records `("sync", n_prefilled)` in `calls`.
- `sample` calls `script(prompt, generated)`, where `prompt` is the token list at the end of the last sync and `generated` is the tokens evaluated since then, and steps `rng.value` with splitmix64. `argmax` calls the script without touching `rng`. `eval` appends, records `("eval", token)`, and raises `EngineError` when the context is full. `rewind` truncates, marks the checkpoint invalid and records `("rewind", pos)`; it is a no-op when `pos >= len`. This is ds4's rule for DeepSeek outside a speculative step (`ds4_session_rewind`): compressor state cannot be rolled back, so the next sync rebuilds. `sample` and `argmax` return -1 while the checkpoint is invalid, as ds4 does.
- `payload_bytes` is `1024 * pos`. `save_snapshot` returns a `FakeSnapshot` holding a copy of the tokens and the prompt length, with that `nbytes`. `load_snapshot` restores them and marks the checkpoint valid. `FakeSnapshot.close()` sets `closed`; loading a closed snapshot raises `EngineError("snapshot is closed")`, as the native class does.
- `raw` is `id(self)` on `FakeEngine` and `FakeSession`. `parse_think_level` and `think_mode_for_context` follow ds4 (`ds4.c:61007`, `ds4.c:61044`): the text `"0"` to `"100"` gives `1000 + n` and anything else `None`; `MAX` becomes `HIGH` when `ctx < 393216`.
- `FakeNative.backend` is `"cpu"`. `FakeEngine` and `FakeSession` methods assert they run on the thread that created the object, as the native classes do, so worker tests catch engine calls from the wrong thread.
- `bind` raises `MissingBinding` with `ds4py does not declare <name>` for names not in `all_bindings()`, and `<name> is absent from this library` for names in `absent`. Otherwise it returns `bindings[name]`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/native -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/_native/fake.py tests/native/test_fake.py
git commit -m "feat(native): add FakeNative with ds4's sync, progress and cancel rules"
```

### Task 9: Transcript, sync plans and sampling

**Files:**
- Create: `src/ds4py/engine/__init__.py`, `src/ds4py/engine/transcript.py`, `src/ds4py/engine/plan.py`, `src/ds4py/engine/sampling.py`
- Test: `tests/engine/test_transcript.py`, `tests/engine/test_plan.py`, `tests/engine/test_sampling.py`

**Interfaces:**
- Consumes: `EngineInfo`.
- Produces:
  - `transcript.py`: `KINDS = ("system", "user", "assistant_prefix", "assistant", "tool", "eos", "note", "reminder", "compaction")`; `Segment(start: int, end: int, kind: str, meta: Mapping[str, object])` (frozen); `Transcript` with `.tokens -> tuple[int, ...]`, `.segments -> tuple[Segment, ...]`, `__len__`, `append(tokens: Sequence[int], kind: str, meta: Mapping | None = None) -> Segment`, `extend(tokens: Sequence[int]) -> None` (grows the last segment), `truncate(n: int) -> None`.
  - `plan.py`: `Plan(kind: Literal["extend", "restore", "rebuild"], reused: int, to_prefill: int, fits: bool, checkpoint: str | None)` (frozen); `plan_sync(tokens: Sequence[int], *, live: Sequence[int], live_valid: bool, checkpoints: Mapping[str, Sequence[int]], ctx: int, reserve: int = 1, restore_enabled: bool = True) -> Plan`.
  - `sampling.py`: `SamplingPolicy(temperature: float = 1.0, top_p: float = 1.0, min_p: float = 0.05, top_k: int = 0, seed: int | None = None)` (frozen); `SamplingPolicy.for_engine(info: EngineInfo, *, temperature=None, top_p=None, min_p=None, top_k=None, seed=None) -> SamplingPolicy`; `Rng(seed: int | None)` with `.state: ctypes.c_uint64`.

- [ ] **Step 1: Write the failing tests**

```python
def test_transcript_segments():
    t = Transcript()
    t.append([1, 2, 3], "user"); t.append([4], "assistant"); t.extend([5, 6])
    assert t.tokens == (1, 2, 3, 4, 5, 6)
    assert [(s.start, s.end, s.kind) for s in t.segments] == [(0, 3, "user"), (3, 6, "assistant")]
    t.truncate(4)
    assert [(s.start, s.end) for s in t.segments] == [(0, 3), (3, 4)]
    t.truncate(3)
    assert len(t.segments) == 1
    with pytest.raises(ValueError, match="kind"): t.append([7], "chat")
    with pytest.raises(ValueError): t.append([], "user")

P = list(range(1, 21))
def plan(tokens, live=(), valid=True, cps=None, ctx=100, restore=True):
    return plan_sync(tokens, live=list(live), live_valid=valid, checkpoints=cps or {}, ctx=ctx, restore_enabled=restore)

def test_extend():
    assert plan(P, live=P[:5]) == Plan("extend", 5, 15, True, None)
    assert plan(P, live=P) == Plan("extend", 20, 0, True, None)

def test_restore_picks_the_longest_checkpoint():
    p = plan(P, live=[9, 9], cps={"system": P[:3], "turn": P[:8], "other": [7]})
    assert p == Plan("restore", 8, 12, True, "turn")

def test_rebuild():
    assert plan(P, live=P[:5], valid=False) == Plan("rebuild", 0, 20, True, None)
    assert plan(P).kind == "rebuild"                                       # empty live is not an extension
    assert plan(P, live=[9], cps={"system": P[:3]}, restore=False).kind == "rebuild"

def test_fits_leaves_the_reserve():
    assert plan(P, ctx=21).fits and not plan(P, ctx=20).fits

INFO = FakeNative().open_engine(EngineConfig(Path("f"))).info()

def test_glm_defaults_unless_set():
    glm = replace(INFO, family="glm", is_glm_dsa=True)
    assert SamplingPolicy.for_engine(glm) == SamplingPolicy(top_p=0.95, min_p=0.0)
    assert SamplingPolicy.for_engine(glm, min_p=0.1).min_p == 0.1
    assert SamplingPolicy.for_engine(INFO, temperature=0.0) == SamplingPolicy(temperature=0.0)

def test_rng_seed():
    assert Rng(7).state.value == 7
    assert Rng(None).state.value != 0 and Rng(0).state.value != 0      # ds4: seed 0 means "pick one"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/engine -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ds4py.engine'`.

- [ ] **Step 3: Implement the three modules**

`Transcript` keeps its token list and segment list behind one `threading.Lock`. Every method and property takes the lock, and the properties return copies, so a thread that reads the transcript while the worker extends it (Task 11) never sees a half-made change.

`plan_sync` follows spec section 10:
1. `extend` when `live_valid`, `live` is not empty, and `tokens` starts with all of `live`. `reused = len(live)`.
2. Otherwise `restore` when `restore_enabled` and some checkpoint's tokens are a non-empty prefix of `tokens`. The longest one wins; ties go to the first in mapping order. `reused` is its length.
3. Otherwise `rebuild` with `reused = 0`.

`to_prefill = len(tokens) - reused`. `fits = len(tokens) + reserve <= ctx`. ds4 refuses a prompt that leaves no free slot, so `reserve` is at least 1.

`Rng(seed)` uses `seed` when it is non-zero. Otherwise it uses `int(time.time()) ^ (os.getpid() << 32) ^ time.process_time_ns()`, masked to 64 bits and forced non-zero, matching `ds4_agent.c`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/engine -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/engine tests/engine
git commit -m "feat(engine): add transcript, sync plans and sampling policy"
```

### Task 10: Events and the engine worker

**Files:**
- Create: `src/ds4py/engine/events.py`, `src/ds4py/engine/worker.py`
- Test: `tests/engine/test_worker.py`

**Interfaces:**
- Consumes: `Native`, `EngineHandle`, `SessionHandle`, `EngineConfig`, `EngineInfo`, `SyncStatus`, `Plan`, `plan_sync`, `FakeNative`, `FakeClock`, `EngineError`, `CheckpointError`.
- Produces:
  - `events.py`, all frozen dataclasses: `Prefill(done: int, total: int, cached_prefix: int, tps: float)`; `Synced(pos: int, plan: Plan)`; `Interrupted(pos: int)`; `Checkpointed(key: str, tokens: int, bytes: int)`, where `tokens` is the prefix length; `Dropped(key: str)`; `Token(id: int, text_bytes: bytes)`; `Stopped(reason: str)`; `STOP_REASONS = ("stop_token", "tool_call_complete", "max_tokens", "context_reserve", "interrupted", "parser_error", "context_exhausted")`; `Event`, the union of these types.
  - `worker.py`: `SYSTEM = "system"`; `SyncJob(tokens: tuple[int, ...])`; `CheckpointJob(key: str, drop: bool = False)`; `Handles(engine: int, session: int)`; `EngineWorker(native: Native, config: EngineConfig, *, cancel: threading.Event | None = None, clock: Callable[[], float] = time.monotonic, max_checkpoints: int = 4, throttle: float = 0.1)` with `.info: EngineInfo`, `.cancel_event: threading.Event`, `submit(job) -> Iterator[Event]`, `call(fn: Callable[[EngineHandle, SessionHandle], T]) -> Future[T]`, `plan(tokens: Sequence[int]) -> Plan` and `close() -> None`.

`call` is the spec's `WorkerCallJob`: a `Future` takes the place of the `Done` and `Failed` events. `SyncJob` takes no images until M5.

- [ ] **Step 1: Write the failing tests**

```python
P = list(range(1, 41))

def worker(*, ctx=64, clock=None, cancel=None, max_checkpoints=4, **fake):
    clock = clock or FakeClock()
    return EngineWorker(FakeNative(clock=clock, **fake), EngineConfig(Path("f.gguf"), ctx=ctx),
                        clock=clock, cancel=cancel, max_checkpoints=max_checkpoints)

def run(w, job):
    return list(w.submit(job))

def sync_calls(w):
    return w.call(lambda e, s: [n for name, n in s.calls if name == "sync"]).result()

def test_each_sync_reports_the_plan_it_carried_out():
    w = worker()
    run(w, SyncJob(tuple(P[:8])))
    assert run(w, CheckpointJob("turn")) == [Checkpointed("turn", 8, 8 * 1024)]
    plans = []
    for tokens, kind in [(P[:20], "extend"), (P[:8] + [99, 98], "restore"), ([50, 51], "rebuild")]:
        predicted = w.plan(tokens)
        *_, synced = run(w, SyncJob(tuple(tokens)))
        assert predicted.kind == kind and synced == Synced(len(tokens), predicted)
        plans.append(predicted)
    assert sync_calls(w)[1:] == [p.to_prefill for p in plans] == [12, 2, 2]

def test_prefill_events_come_at_most_every_100ms():
    clock = FakeClock()
    w = worker(clock=clock, chunk=4, chunk_seconds=0.03)
    events = run(w, SyncJob(tuple(P)))
    prefills = [e for e in events if isinstance(e, Prefill)]
    assert [(e.done, e.total, e.cached_prefix) for e in prefills] == [(0, 40, 0), (16, 40, 0), (32, 40, 0)]
    assert prefills[1].tps == pytest.approx(16 / 0.12)
    assert events[-1] == Synced(40, Plan("rebuild", 0, 40, True, None))

def test_prefill_counts_only_the_new_tokens():
    w = worker(chunk=4)
    run(w, SyncJob(tuple(P[:6])))
    assert run(w, SyncJob(tuple(P)))[0] == Prefill(0, 34, 6, 0.0)

def test_cancel_interrupts_sync_and_clears_the_flag():
    cancel = threading.Event()
    w = worker(cancel=cancel)
    run(w, SyncJob(tuple(P[:6])))
    cancel.set()                                                    # Ctrl+C
    assert run(w, SyncJob(tuple(P)))[-1] == Interrupted(6)
    assert w.cancel_event is cancel and not cancel.is_set()
    assert w.plan(P) == Plan("extend", 6, 34, True, None)          # the prefix done so far is kept

def test_checkpoint_limits():
    w = worker(max_checkpoints=2)
    with pytest.raises(CheckpointError, match="no valid state"): run(w, CheckpointJob("a"))
    run(w, SyncJob(tuple(P[:4])))
    for key in (SYSTEM, "a", "b", "a"):                             # system is not counted; "a" is replaced
        run(w, CheckpointJob(key))
    with pytest.raises(CheckpointError, match="max_checkpoints"): run(w, CheckpointJob("c"))
    assert run(w, CheckpointJob("a", drop=True)) == [Dropped("a")]
    run(w, CheckpointJob("c"))
    with pytest.raises(CheckpointError, match="unknown checkpoint"): run(w, CheckpointJob("zz", drop=True))

def test_a_checkpoint_is_kept_until_its_replacement_is_saved():
    w = worker()
    saved = []
    def wrap_save(e, s):                                            # the third save runs out of memory
        real = s.save_snapshot
        def save():
            if len(saved) == 2: raise EngineError("out of memory")
            saved.append(real())
            return saved[-1]
        s.save_snapshot = save
    w.call(wrap_save).result()
    for n in (8, 12):
        run(w, SyncJob(tuple(P[:n]))); run(w, CheckpointJob("a"))
    assert [snap.closed for snap in saved] == [True, False]         # the replaced snapshot is closed
    run(w, SyncJob(tuple(P[:16])))
    with pytest.raises(EngineError, match="out of memory"): run(w, CheckpointJob("a"))
    *_, synced = run(w, SyncJob(tuple(P[:12] + [99])))
    assert synced.plan == Plan("restore", 12, 1, True, "a")         # closing first would fail this restore

def test_sync_refuses_a_prompt_that_does_not_fit():
    w = worker(ctx=8)
    with pytest.raises(EngineError, match="8 tokens does not fit a context of 8"):
        run(w, SyncJob(tuple(P[:8])))
    assert sync_calls(w) == []

def test_engine_calls_run_on_one_worker_thread():
    w = worker()
    ids = {w.call(lambda e, s: threading.get_ident()).result() for _ in range(3)}
    assert len(ids) == 1 and threading.get_ident() not in ids
    assert isinstance(w.call(lambda e, s: 1 / 0).exception(), ZeroDivisionError)
    assert w.call(lambda e, s: s.pos()).result() == 0               # the worker survives a failed call
    assert w.info.name == "fake"

def test_open_errors_reach_the_caller():
    class Broken(FakeNative):
        def open_engine(self, config): raise ModelFileError("no model")
    with pytest.raises(ModelFileError): EngineWorker(Broken(), EngineConfig(Path("f")))

def test_close():
    w = worker(); w.close(); w.close()
    with pytest.raises(EngineError, match="closed"): w.submit(SyncJob((1,)))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/engine/test_worker.py -v`
Expected: FAIL with `ModuleNotFoundError` for `ds4py.engine.events`.

- [ ] **Step 3: Implement `events.py` and `worker.py`**

The thread:
- It opens the engine with `native.open_engine(config)`, creates the session with `config.ctx` and reads `info()`. `__init__` waits for this and re-raises any error, such as `ModelFileError`. The thread is a daemon.
- It takes work items from one `queue.SimpleQueue`, in order. `submit(job)` enqueues the job at once with its own event queue and returns a generator over that queue. The worker puts each event, then an end marker, or the exception, which the generator re-raises in the caller's thread. `call(fn)` enqueues `fn(engine, session)` with a `Future`.
- `plan(tokens)` is a `call` of `plan_sync(tokens, live=s.tokens(), live_valid=s.checkpoint_valid(), checkpoints=<key → tokens of each held checkpoint>, ctx=s.ctx(), restore_enabled=native.restore_enabled)`. Like every `call`, it waits for the jobs queued before it.
- `close()` enqueues a stop item and joins. The thread closes the held snapshots, the session and the engine, in that order. A second `close()` does nothing. `submit` and `call` after `close` raise `EngineError("engine worker is closed")`.

`SyncJob`:
1. Plan again on the worker. When the plan does not fit, raise `EngineError(f"prompt of {n} tokens does not fit a context of {ctx}")` without syncing.
2. For `restore`, load the plan's checkpoint with `load_snapshot`.
3. Call `sync` with `cancel=cancel_event.is_set` and a progress function. The progress function counts done tokens as `agent_worker_sync_tokens` does (`ds4_agent.c`): `prefill_work` gives `current`; `prefill_chunk` and `prefill_display` give `current - plan.reused`; other events are ignored; the count is clamped to `0..plan.to_prefill`. It emits `Prefill(done, plan.to_prefill, plan.reused, tps)` for the first event, then only when `clock() - last_emit >= throttle`. `tps` is `done / (clock() - start)`, or `0.0` when no time has passed.
4. `DONE` gives `Synced(s.pos(), plan)`. `INTERRUPTED` clears `cancel_event` first and then gives `Interrupted(s.pos())`.

`CheckpointJob`:
- Saving needs a valid checkpoint with `pos() > 0`; otherwise raise `CheckpointError("no valid state to checkpoint")`. A new key other than `SYSTEM` is refused with `CheckpointError(f"max_checkpoints ({n}) reached; drop one first")` when `n` others are held. The worker keeps `(tuple(s.tokens()), snapshot)` per key and emits `Checkpointed(key, len(tokens), snapshot.nbytes)`. Saving an existing key keeps the old entry until the new snapshot exists: save, then swap the entry, then close the old snapshot. If the save raises, the old entry is untouched and still restores. Replacing a key therefore holds two snapshots for a moment. A caller short of memory drops the key first with `CheckpointJob(key, drop=True)`; replacement never drops a snapshot on its own.
- `drop=True` closes the snapshot and emits `Dropped(key)`. An unknown key raises `CheckpointError(f"unknown checkpoint {key!r}")`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/engine -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/engine tests/engine/test_worker.py
git commit -m "feat(engine): add the engine worker with sync, checkpoints and calls"
```

### Task 11: The decode loop and `PlainStep`

**Files:**
- Create: `src/ds4py/engine/decode.py`
- Modify: `src/ds4py/engine/worker.py`
- Test: `tests/engine/test_decode.py`

**Interfaces:**
- Consumes: `SessionHandle`, `EngineHandle`, `SamplingPolicy`, `Rng`, `Transcript`, `Token`, `Stopped`, `Event`, `EngineWorker`, `EngineError`.
- Produces: `Verdict(Enum)` with `CONTINUE`, `STOP_BEFORE`, `STOP_AFTER`; `GenerateJob(sampling: SamplingPolicy, rng: Rng, max_tokens: int, transcript: Transcript, reserve: int = 1)` (frozen; `worker.py` re-exports it); `DecodeContext` with `rng: ctypes.c_uint64`, `reason: str | None`, `before_sample() -> None`, `sampling() -> SamplingPolicy`, `cancelled() -> bool`, `accept(token: int) -> Verdict` and `commit(token: int) -> None`; the `DecodeStep` protocol (`hooks_every_sample: bool`, `run(session: SessionHandle, ctx: DecodeContext) -> None`); `PlainStep`; `generate(engine: EngineHandle, session: SessionHandle, job: GenerateJob, *, cancelled: Callable[[], bool], emit: Callable[[Event], None], step: DecodeStep | None = None) -> str`.

`GenerateJob` has no `stop_policy` yet: the tool-call stop arrives with the parsers in M2, and so do the logits hooks behind `before_sample` and the greedy switch behind `sampling()`.

- [ ] **Step 1: Write the failing tests**

```python
def worker(script, *, ctx=64, cancel=None):
    return EngineWorker(FakeNative(script=script), EngineConfig(Path("f"), ctx=ctx), cancel=cancel)

def gen(w, prompt, max_tokens=50, transcript=None):
    transcript = Transcript() if transcript is None else transcript
    list(w.submit(SyncJob(tuple(prompt))))
    return list(w.submit(GenerateJob(SamplingPolicy(), Rng(1), max_tokens, transcript)))

def tok(c):
    return Token(ord(c) + 1, c.encode())

def test_stop_token_is_neither_kept_nor_evaluated():
    w, t = worker(reply(b"hi")), Transcript()
    assert gen(w, [5, 6], transcript=t) == [tok("h"), tok("i"), Stopped("stop_token")]
    assert t.tokens == (tok("h").id, tok("i").id) and t.segments[0].kind == "assistant"
    calls, pos = w.call(lambda e, s: (s.calls, s.pos())).result()
    assert ("eval", FAKE_EOS) not in calls and pos == 4

def test_max_tokens_keeps_the_last_token():
    w = worker(reply(b"hello"))
    assert gen(w, [5, 6], max_tokens=2) == [tok("h"), tok("e"), Stopped("max_tokens")]
    assert w.call(lambda e, s: s.pos()).result() == 4
    assert gen(w, [5, 6], max_tokens=0) == [Stopped("max_tokens")]

def test_context_reserve_matches_the_ds4_agent_clamp():
    w = worker(reply(b"abcdef"), ctx=8)                    # room 3, less a reserve of 1
    assert gen(w, [1, 2, 3, 4, 5]) == [tok("a"), tok("b"), Stopped("context_reserve")]

def test_ctrl_c_keeps_the_partial_tokens():
    cancel = threading.Event()
    def script(prompt, generated):
        if len(generated) == 2: cancel.set()               # Ctrl+C while the third token is sampled
        return ord("a") + 1
    w, t = worker(script, cancel=cancel), Transcript()
    assert gen(w, [5, 6], transcript=t) == [tok("a")] * 3 + [Stopped("interrupted")]
    assert len(t) == 3 and not cancel.is_set() and w.call(lambda e, s: s.pos()).result() == 5

def test_sampling_needs_a_valid_state():
    w = worker(reply(b"x"))
    list(w.submit(SyncJob((5, 6, 7))))
    w.call(lambda e, s: s.rewind(1)).result()               # invalidates the checkpoint, as in ds4
    with pytest.raises(EngineError, match="sync first"):
        list(w.submit(GenerateJob(SamplingPolicy(), Rng(1), 5, Transcript())))

class StubEngine:
    def is_stop(self, token): return token == 0
    def token_text(self, token): return b"x"

class StubSession:                                          # just what PlainStep and generate use
    def __init__(self, tokens, fail_eval=None):
        self.tokens, self.fail_eval, self.seen = list(tokens), fail_eval, []
    def ctx(self): return 100
    def pos(self): return 10
    def sample(self, temperature, top_k, top_p, min_p, rng):
        self.seen.append(("sample", temperature, top_k, top_p, min_p, rng.value))
        return self.tokens.pop(0)
    def eval(self, token):
        if token == self.fail_eval: raise EngineError("eval failed")
        self.seen.append(("eval", token))

def test_plain_step_follows_the_contract():
    s, t, out = StubSession([7, 0]), Transcript(), []
    policy = SamplingPolicy(temperature=0.5, top_p=0.9, min_p=0.0, top_k=3)
    reason = generate(StubEngine(), s, GenerateJob(policy, Rng(9), 10, t), cancelled=lambda: False, emit=out.append)
    assert reason == "stop_token" and out == [Token(7, b"x")] and PlainStep.hooks_every_sample
    assert s.seen == [("sample", 0.5, 3, 0.9, 0.0, 9), ("eval", 7), ("sample", 0.5, 3, 0.9, 0.0, 9)]
    assert t.tokens == (7,)

def test_a_token_is_committed_after_its_eval_and_before_its_event():
    s, t, out = StubSession([7, 8], fail_eval=8), Transcript(), []
    t.append([1, 2], "user")
    job = GenerateJob(SamplingPolicy(), Rng(9), 10, t)
    with pytest.raises(EngineError, match="eval failed"):
        generate(StubEngine(), s, job, cancelled=lambda: False, emit=lambda ev: out.append((ev, t.tokens)))
    assert out == [(Token(7, b"x"), (1, 2, 7))]             # 8 failed its eval: not kept, not published
    assert [(seg.kind, seg.end) for seg in t.segments] == [("user", 2), ("assistant", 3)]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/engine/test_decode.py -v`
Expected: FAIL with `ModuleNotFoundError` for `ds4py.engine.decode`.

- [ ] **Step 3: Implement `decode.py` and the `GenerateJob` branch of the worker**

`generate`:
- `room = session.ctx() - session.pos()` and `limit = min(job.max_tokens, max(room - job.reserve, 0))`. The reason at the limit is `max_tokens` when `job.max_tokens <= room - job.reserve`, else `context_reserve`. With `reserve=1` this is the `ds4-agent --raw` clamp, `min(n_predict, room - 1)`. When `limit` is 0, return that reason without sampling.
- `DecodeContext.accept(token)` judges a token and changes nothing outside the context. A token with `engine.is_stop(token)` sets `reason = "stop_token"` and returns `STOP_BEFORE`. Any other token is counted; at the limit it sets the limit reason and returns `STOP_AFTER`, and otherwise it returns `CONTINUE`.
- `DecodeContext.commit(token)` runs once the KV holds an accepted token. It reads `engine.token_text(token)`, adds the token to `job.transcript`, and then emits `Token(token, text)`. The job's first commit appends an `assistant` segment and later ones extend it. This is the order of the `ds4-agent` loop: eval, push onto the transcript, then stream the text (spec section 6, item 2). A token whose eval raises is never committed, so it reaches neither the transcript nor the caller.
- `before_sample()` does nothing in M1. `sampling()` returns `job.sampling`. `cancelled()` calls `cancelled`.
- `step` defaults to `PlainStep()`. After it returns, `generate` returns `ctx.reason`, or `"interrupted"` when the step stopped because of Ctrl+C.

`PlainStep.run` is the raw loop of `ds4_agent.c`:

```python
while not ctx.cancelled():
    ctx.before_sample()
    p = ctx.sampling()
    token = session.sample(p.temperature, p.top_k, p.top_p, p.min_p, ctx.rng)
    # token < 0: raise EngineError("cannot sample: the session state is not valid; sync first")
    verdict = ctx.accept(token)
    if verdict is Verdict.STOP_BEFORE: return
    session.eval(token)
    ctx.commit(token)
    if verdict is Verdict.STOP_AFTER: return
```

The spec's decode-step contract (section 10) has the step call only `accept`. This plan adds one rule: once the KV holds an accepted token, the step calls `ctx.commit(token)`. A multi-token step commits its kept tokens in order, after it has rewound the rest. The loop still owns pushing tokens onto the transcript, as the spec says, but it pushes only tokens that the engine has evaluated.

The worker runs `GenerateJob` with `generate(engine, session, job, cancelled=cancel_event.is_set, emit=...)`. While the job runs, the worker thread is the only writer of `job.transcript`. For `"interrupted"` the worker clears `cancel_event` first. Then it emits `Stopped(reason)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/engine -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/engine tests/engine/test_decode.py
git commit -m "feat(engine): add the decode loop and PlainStep"
```

### Task 12: `Engine`, `Session` and `ds4py.open`

**Files:**
- Create: `src/ds4py/engine/session.py`
- Modify: `src/ds4py/__init__.py`
- Test: `tests/engine/test_session.py`

**Interfaces:**
- Consumes: Tasks 7–11, `LibNative`, `load_library`.
- Produces:
  - `open(model: str | Path = "ds4flash.gguf", *, ctx: int = 100_000, backend: str | None = None, think: int = ThinkMode.HIGH, seed: int | None = None, threads: int = 0, lib: str | Path | None = None, native: Native | None = None, cancel: threading.Event | None = None) -> Engine`.
  - `Engine` with `.native`, `.info`, `.config`, `.think`, `.sampling_defaults` (settable), `tokenize(text, *, rendered=False) -> list[int]`, `token_text(token) -> bytes`, `session() -> Session`, `close()`, and context-manager support.
  - `Session` with `.transcript`, `.ctx_used`, `append_tokens(tokens, kind, meta=None) -> Segment`, `run_round(*, max_tokens: int = 50_000, sampling: SamplingPolicy | None = None) -> Iterator[Event]`, `plan(tokens=None) -> Plan`, `checkpoint(key) -> Checkpointed`, `drop_checkpoint(key) -> None`, `on_worker(fn: Callable[[Handles], T]) -> Future[T]` and `cancel() -> None`.
  - `ds4py` exports `open`, `ThinkMode`, `Engine`, `Session` and the errors.

M1 leaves out the spec's `append_user`, `append_tool_results` and `append_system_note`; they need `PromptBuilder` (M2). `ReserveReached` with `resume()` and `fail()` is also M2. Until then `run_round` passes `Stopped("context_reserve")` through, which is what `ds4-agent --raw` does when the context fills.

- [ ] **Step 1: Write the failing tests**

```python
def fake_engine(*, ctx=64, cancel=None, **fake):
    return ds4py.open("f.gguf", ctx=ctx, cancel=cancel, native=FakeNative(**fake))

def test_open_checks_backend_and_think_mode():
    with fake_engine() as e:
        assert e.info.name == "fake" and e.config.backend == "cpu" and e.think == ThinkMode.HIGH
    with pytest.raises(EngineError, match="built for cpu"):
        ds4py.open("f.gguf", backend="metal", native=FakeNative())
    with ds4py.open("f.gguf", ctx=64, think=ThinkMode.MAX, native=FakeNative()) as e:
        assert e.think == ThinkMode.HIGH                    # MAX needs 393216 tokens of context

def test_sampling_defaults_follow_the_engine():
    glm = replace(FakeNative().open_engine(EngineConfig(Path("f"))).info(), family="glm", is_glm_dsa=True)
    with ds4py.open("f.gguf", seed=5, native=FakeNative(info=glm)) as e:
        assert e.sampling_defaults == SamplingPolicy(top_p=0.95, min_p=0.0, seed=5)

def test_a_round_syncs_generates_and_extends_the_transcript():
    with fake_engine(script=reply(b"ok")) as e:
        s = e.session()
        s.append_tokens(e.tokenize("hi"), "user")
        events = list(s.run_round(max_tokens=10))
        assert [type(ev).__name__ for ev in events if not isinstance(ev, Prefill)] == ["Synced", "Token", "Token", "Stopped"]
        assert s.transcript.tokens == tuple(e.tokenize("hiok"))
        assert [seg.kind for seg in s.transcript.segments] == ["user", "assistant"]
        assert s.ctx_used == 4 and s.plan() == Plan("extend", 4, 0, True, None)
        with pytest.raises(EngineError, match="one session"): e.session()

def test_ctrl_c_during_sync_keeps_the_transcript():
    with fake_engine() as e:
        s = e.session()
        s.append_tokens(e.tokenize("hello"), "user")
        s.cancel()
        assert list(s.run_round())[-1] == Interrupted(0)
        assert s.transcript.tokens == tuple(e.tokenize("hello"))
        assert isinstance(list(s.run_round())[-1], Stopped)          # the flag was cleared

def test_closing_a_round_early_keeps_what_was_evaluated():
    cancel = threading.Event()
    with fake_engine(script=reply(b"abcdefgh"), cancel=cancel) as e:
        s = e.session()
        s.append_tokens([5, 6], "user")
        rnd = s.run_round()
        first = next(ev for ev in rnd if isinstance(ev, Token))
        assert s.transcript.tokens[2] == first.id           # committed before it was published
        with pytest.raises(EngineError, match="a round is open"): s.append_tokens([7], "user")
        rnd.close()
        assert s.ctx_used == len(s.transcript)              # every evaluated token is in the transcript
        assert s.plan().kind == "extend" and not cancel.is_set()
        s.append_tokens([7], "user")                        # the round is over

def test_checkpoints_through_the_session():
    with fake_engine() as e:
        s = e.session()
        s.append_tokens(list(range(1, 9)), "system")
        list(s.run_round(max_tokens=0))
        assert s.checkpoint(SYSTEM) == Checkpointed(SYSTEM, 8, 8 * 1024)
        s.drop_checkpoint(SYSTEM)
        with pytest.raises(CheckpointError): s.drop_checkpoint(SYSTEM)

def test_extension_binding_runs_on_the_worker(ext_site):
    native = FakeNative(bindings={"ds4_engine_layer_count": lambda engine_ptr: 43})
    with ds4py.open("f.gguf", ctx=64, native=native) as e:
        layer_count = e.native.bind("ds4_engine_layer_count")
        count, session_ptr, thread = e.session().on_worker(
            lambda h: (layer_count(h.engine), h.session, threading.get_ident())).result()
        assert count == 43 and isinstance(session_ptr, int) and thread != threading.get_ident()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/engine/test_session.py -v`
Expected: FAIL with `ModuleNotFoundError` for `ds4py.engine.session`.

- [ ] **Step 3: Implement `session.py` and the exports**

`open`:
1. `native = native or LibNative(load_library(lib))`.
2. A `backend` other than `native.backend` raises `EngineError` with the message `this libds4 was built for <built>, not <asked>; run ds4py build --backend <asked>`.
3. `think = native.think_mode_for_context(think, ctx)`.
4. Start `EngineWorker(native, EngineConfig(Path(model), ctx, native.backend, threads), cancel=cancel)`.
5. `sampling_defaults = SamplingPolicy.for_engine(worker.info, seed=seed)`.

`session.py` defines `open`, so it must not use the builtin `open`.

`Engine.tokenize` and `Engine.token_text` go through `worker.call(...).result()`, because only the worker thread calls the engine. A second `session()` raises `EngineError("one session per engine")`. `close()` closes the worker.

`Session` makes one `Rng(engine.sampling_defaults.seed)` when it is created and uses it for every round. `run_round` is a generator:
1. Submit `SyncJob(transcript.tokens)` and yield its events. After `Interrupted`, return; the transcript is unchanged.
2. Submit `GenerateJob(sampling or engine.sampling_defaults, rng, max_tokens, transcript)` and yield its events. The worker adds each token to the transcript before it publishes the token's event (Task 11), so `run_round` never writes the transcript.
3. If the caller closes the generator while a job is running, set the cancel flag, then read and drop the job's remaining events until the job ends. Nothing is rebuilt, because every evaluated token is already in the transcript. Clear the flag if the job ended without reporting an interrupt.

A round is open from its first step until it returns or is closed. While a round is open, `append_tokens` raises `EngineError("a round is open")`. So the transcript has one writer at a time: the caller between rounds, and the worker during a round.

`ctx_used` is `worker.call(lambda e, s: s.pos()).result()`. `plan(tokens=None)` plans the transcript's tokens by default. `checkpoint` and `drop_checkpoint` run a `CheckpointJob` and return its event. `on_worker(fn)` is `worker.call(lambda e, s: fn(Handles(e.raw, s.raw)))`. `cancel()` sets `worker.cancel_event` and is safe from any thread. `transcript` returns the session's `Transcript`. Callers change it only through `append_tokens`; the worker extends it during a round.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -v`
Expected: PASS, with `build` and `live` tests skipped.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py tests/engine/test_session.py
git commit -m "feat(engine): add ds4py.open, Engine and Session"
```

### Task 13: `ds4py raw`

**Files:**
- Create: `src/ds4py/raw.py`
- Modify: `src/ds4py/cli.py`
- Test: `tests/test_raw.py`

**Interfaces:**
- Consumes: `ds4py.open`, `Engine`, `Session`, `SamplingPolicy`, `Token`, `Stopped`, `Interrupted`, `EngineError`, `FakeNative`, `reply`.
- Produces: `Trace(path: Path, *, now: Callable[[], datetime] = datetime.now)`, a context manager with `line(message: str) -> None`, `text(label: str, data: bytes) -> None` and `token(index: int, token: int, text: bytes) -> None`; `read_trace_tokens(path: Path) -> tuple[list[int], list[int]]`, the prompt and generated token ids of the last raw run in a trace written by either `ds4py raw` or `ds4-agent --raw`; `run_raw(engine: Engine, prompt: str, *, max_tokens: int, sampling: SamplingPolicy, out: BinaryIO, err: TextIO, trace: Trace | None = None) -> int`; `command(args: argparse.Namespace) -> int`; module seam `open_engine` (`ds4py.open`).
- CLI: `ds4py raw (PROMPT | -p TEXT | --prompt-file PATH) [-m PATH] [-c N] [-n N] [--temp X] [--top-p X] [--min-p X] [--seed N] [--backend NAME | --cpu | --metal | --cuda] [-t N] [--trace FILE] [--lib PATH]`. Defaults: model `ds4flash.gguf`, `-c 100000`, `-n 50000`, backend from the manifest, threads 0.

Exit codes: 0 when generation stops at a stop token, `-n` or the context reserve; 1 on an error (`cli.main` prints `ds4py: <message>`); 2 when the prompt does not fit; 130 after Ctrl+C. `ds4-agent --raw` writes the same stdout: token bytes only, with no trailing newline (`worker_run_raw_prompt`, `ds4_agent.c:10830`).

- [ ] **Step 1: Write the failing tests**

```python
import ds4py.raw as raw_mod

def raw(prompt="hi", *, ctx=64, max_tokens=50, script=reply(b"ok"), trace=None, cancel=None):
    out, err = io.BytesIO(), io.StringIO()
    with ds4py.open("f.gguf", ctx=ctx, cancel=cancel, native=FakeNative(script=script)) as e:
        code = run_raw(e, prompt, max_tokens=max_tokens, sampling=e.sampling_defaults,
                       out=out, err=err, trace=trace)
    return code, out.getvalue(), err.getvalue()

def test_writes_the_token_bytes_and_nothing_else():
    assert raw(script=reply(b"hello")) == (0, b"hello", "")
    assert raw(script=reply(b"hello"), max_tokens=3) == (0, b"hel", "")

def test_a_prompt_that_does_not_fit_exits_2():
    assert raw("x" * 64, ctx=64) == (2, b"", "ds4py: prompt of 64 tokens does not fit a context of 64\n")

def test_an_empty_prompt_is_an_error():
    with pytest.raises(EngineError, match="raw prompt tokenized to no tokens"):
        raw("")

def test_ctrl_c_flushes_the_text_and_exits_130():
    main, cancel = threading.get_ident(), threading.Event()
    def script(prompt, generated):
        if len(generated) == 2:                              # Ctrl+C while the third token is sampled
            signal.pthread_kill(main, signal.SIGINT)
            assert cancel.wait(5)
        return ord("abc"[len(generated)]) + 1
    before = signal.getsignal(signal.SIGINT)
    assert raw(script=script, cancel=cancel) == (130, b"abc", "")
    assert signal.getsignal(signal.SIGINT) is before

def test_trace_uses_the_ds4_agent_format(tmp_path):
    path = tmp_path / "trace.log"
    for _ in range(2):                                       # the file is appended to; the reader takes the last run
        with Trace(path, now=lambda: datetime(2026, 10, 21, 9, 5, 7, 42_000)) as t:
            raw("a\n", script=reply(b'"\x7f\xc3'), trace=t)
    assert path.read_bytes().splitlines()[:7] == [
        b'2026-10-21 09:05:07.042 raw-prompt="a\\n"',
        b"2026-10-21 09:05:07.042 tokens label=raw_prompt_tokens start=0 len=2",
        b'2026-10-21 09:05:07.042 token index=0 id=98 bytes=1 text="a" hex=61',
        b'2026-10-21 09:05:07.042 token index=1 id=11 bytes=1 text="\\n" hex=0a',
        b'2026-10-21 09:05:07.042 token index=1 id=35 bytes=1 text="\\"" hex=22',
        b'2026-10-21 09:05:07.042 token index=2 id=128 bytes=1 text="\\x7f" hex=7f',
        b'2026-10-21 09:05:07.042 token index=3 id=196 bytes=1 text="\xc3" hex=c3',
    ]
    assert read_trace_tokens(path) == ([98, 11], [35, 128, 196])

@pytest.fixture
def opened(monkeypatch):
    seen = {}
    def open_engine(model, **kw):
        seen.update(model=model, **kw)
        return ds4py.open(model, ctx=kw["ctx"], native=FakeNative(script=reply(b"ok")))
    monkeypatch.setattr(raw_mod, "open_engine", open_engine)
    return seen

def test_cli_passes_the_flags_through(opened, monkeypatch):
    ran = []
    monkeypatch.setattr(raw_mod, "run_raw", lambda engine, prompt, **kw: ran.append((prompt, kw)) or 0)
    assert cli.main(["raw", "hi", "-m", "m.gguf", "-c", "64", "-n", "5", "--temp", "0",
                     "--top-p", "0.9", "--seed", "1", "--cpu", "-t", "2"]) == 0
    assert opened == {"model": "m.gguf", "ctx": 64, "backend": "cpu", "seed": 1, "threads": 2, "lib": None}
    (prompt, kw), = ran
    assert (prompt, kw["max_tokens"], kw["trace"]) == ("hi", 5, None)
    assert kw["sampling"] == SamplingPolicy(temperature=0.0, top_p=0.9, seed=1)

def test_cli_defaults_and_prompt_sources(opened, capsysbinary, tmp_path):
    (tmp_path / "p.txt").write_text("hi")
    assert cli.main(["raw", "--prompt-file", str(tmp_path / "p.txt")]) == 0
    assert capsysbinary.readouterr().out == b"ok"
    assert opened == {"model": "ds4flash.gguf", "ctx": 100_000, "backend": None, "seed": None, "threads": 0, "lib": None}
    with pytest.raises(SystemExit): cli.main(["raw", "hi", "-p", "hi"])     # exactly one prompt source
    with pytest.raises(SystemExit): cli.main(["raw"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_raw.py -v`
Expected: FAIL with `ModuleNotFoundError` for `ds4py.raw`.

- [ ] **Step 3: Implement `raw.py` and the `raw` subcommand**

`run_raw` follows `worker_run_raw_prompt`:
1. `tokens = engine.tokenize(prompt)`, as `ds4_tokenize_text` with no chat template. No tokens raises `EngineError("raw prompt tokenized to no tokens")`.
2. With a trace, write the text line `raw-prompt="<escaped prompt>"`, the line `tokens label=raw_prompt_tokens start=0 len=<n>`, then one token line per prompt token with indexes from 0. `ds4-agent` writes the same three parts (`ds4_agent.c:10847`).
3. Create the session and `append_tokens(tokens, "user")`. When `session.plan().fits` is false, write `ds4py: prompt of <n> tokens does not fit a context of <ctx>` and a newline to `err`, and return 2 without syncing.
4. On the main thread only, install a SIGINT handler that calls `session.cancel()` and sets SIGINT back to `SIG_DFL`, so a second Ctrl+C ends the process at once. Restore the previous handler in `finally`.
5. Run `session.run_round(max_tokens=max_tokens, sampling=sampling)`. For each `Token`, write its bytes to `out`, flush, and trace it with indexes from 1. `Interrupted` and `Stopped("interrupted")` return 130; any other `Stopped` returns 0.

`Trace` opens the file in append-binary mode, as `ds4-agent` does, and flushes each line. A line is `now()` as `%Y-%m-%d %H:%M:%S.mmm`, a space, and the message. A text line's message is `<label>="<escaped>"` (`agent_trace_text`). A token line's message is `token index=<i> id=<id> bytes=<n> text="<escaped>" hex=<lowercase hex>`. Both are escaped as in `agent_trace_escaped` (`ds4_agent.c:1733`): `\\`, `\n`, `\r`, `\t` and `\"`; other bytes below 32 and byte 127 as `\xNN`; every other byte as it is.

`read_trace_tokens` finds the last `tokens label=raw_prompt_tokens start=0 len=<n>` line. The next `n` token lines are the prompt and every later token line is generated output. It reads `id=` from each with `rb" token index=\d+ id=(-?\d+) "` and raises `ValueError` when the file holds no raw run.

`command(args)`:
1. Read the prompt from `PROMPT`, `-p` or `--prompt-file` (UTF-8). The three form a required mutually exclusive group.
2. `engine = open_engine(args.model, ctx=args.ctx, backend=args.backend, seed=args.seed, threads=args.threads, lib=args.lib)`. `--cpu`, `--metal` and `--cuda` set `backend`.
3. `sampling = SamplingPolicy.for_engine(engine.info, temperature=args.temp, top_p=args.top_p, min_p=args.min_p, seed=args.seed)`.
4. Inside `with engine` and the trace, return `run_raw(engine, prompt, max_tokens=args.tokens, sampling=sampling, out=sys.stdout.buffer, err=sys.stderr, trace=trace)`.

`cli.py` only adds the parser and calls `raw.command`. Look up `open_engine` and `run_raw` as module globals at call time, so the tests can replace them.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_raw.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/raw.py src/ds4py/cli.py tests/test_raw.py
git commit -m "feat(cli): add ds4py raw with ds4-agent's trace format"
```

### Task 14: `ds4py check --live` and the live parity tests

**Files:**
- Modify: `src/ds4py/abi/check.py`, `src/ds4py/cli.py`, `tests/conftest.py`, `tests/abi/test_check.py`
- Create: `tests/live/test_live_check_cli.py`, `tests/live/test_raw_parity.py`
- Test: `tests/abi/test_live_checks.py`

**Interfaces:**
- Consumes: `EngineWorker`, `SyncJob`, `CheckpointJob`, `GenerateJob`, `SamplingPolicy`, `Rng`, `Transcript`, `Synced`, `Token`, `Stopped`, `EngineConfig`, `LibNative`, `load_library`, `manifest_path`, `Manifest`, `check_library`, `format_report`, `read_trace_tokens`, `FakeNative`, `FakeClock`, `FakeSession`.
- Produces: `LIVE_TEXT: str`; `LIVE_CHECKPOINT = "live-check"`; `run_live_checks(worker: EngineWorker, *, clock: Callable[[], float] = time.monotonic, prefix: int = 2048, tail: int = 64, generate: int = 256) -> dict[str, dict]`; `check_live(lib: str | Path | None, model: Path, *, ctx: int = 4096) -> CheckReport`; a `live_lib` fixture in `tests/conftest.py`.
- CLI: `ds4py check [--lib PATH] [--live --model PATH [--ctx N]]`. `--live` without `--model` is a usage error (exit 2).

The spec's sync-plan check asks that "the prefill count in `Synced` equals `plan().to_prefill`". ds4 cannot report that count the same way on every backend: the CPU path reports one progress event per extended token and a single `(len, len)` event for a rebuild, and Metal reports the start position first (Task 7). So `Synced` carries the plan, and the live check measures reuse by time instead: an extend or a restore of `tail` tokens must take at most half as long as a rebuild of all `prefix` tokens. With the defaults that is 64 tokens against 2048, so a backend that reuses its KV passes by a wide margin, and one that silently rebuilds fails.

The result is recorded in the manifest as `live_checks`:

```python
{"sync_plans": {"ok": bool, "message": str,
                "tokens": {"extend": int, "restore": int, "rebuild": int},
                "seconds": {"extend": float, "restore": float, "rebuild": float}},
 "restore": {"ok": bool, "message": str, "tokens": int}}
```

- [ ] **Step 1: Write the failing tests**

`tests/abi/test_live_checks.py` runs the checks on `FakeNative`:

```python
def live(script=lambda prompt, generated: 98, **fake):    # 98 is "a"; the run never stops early
    clock = FakeClock()
    w = EngineWorker(FakeNative(script=script, clock=clock, chunk=4, chunk_seconds=0.01, **fake),
                     EngineConfig(Path("f.gguf"), ctx=512), clock=clock)
    try:
        return run_live_checks(w, clock=clock, prefix=128, tail=16, generate=32)
    finally:
        w.close()

def test_the_checks_pass_on_an_engine_that_reuses_its_kv():
    r = live()
    assert r["sync_plans"]["ok"] and r["restore"] == {
        "ok": True, "message": "32 greedy tokens after a restore match a rebuild", "tokens": 32}
    assert r["sync_plans"]["tokens"] == {"extend": 16, "restore": 16, "rebuild": 128}
    assert r["sync_plans"]["seconds"] == pytest.approx({"extend": 0.04, "restore": 0.04, "rebuild": 0.32})
    assert r["sync_plans"]["message"] == ("extend of 16 tokens took 0.040 s, restore of 16 took 0.040 s, "
                                          "rebuild of 128 took 0.320 s")

def test_a_restore_that_changes_the_output_fails_the_restore_check():
    starts = []
    def drifting(prompt, generated):                    # each run answers with a different token
        if not generated:
            starts.append(None)
        return 2 + len(starts)
    r = live(script=drifting)
    assert r["sync_plans"]["ok"] and not r["restore"]["ok"]
    assert r["restore"]["message"] == "the runs differ at token 0: 3 after a restore, 4 after a rebuild"

def test_an_engine_that_always_rebuilds_fails_the_sync_check(monkeypatch):
    sync = FakeSession.sync
    def rebuild_always(self, tokens, **kw):
        self.invalidate()
        return sync(self, tokens, **kw)
    monkeypatch.setattr(FakeSession, "sync", rebuild_always)
    r = live()
    assert not r["sync_plans"]["ok"]
    assert r["sync_plans"]["message"] == ("extend of 16 tokens took 0.320 s, more than half of the "
                                          "0.320 s rebuild; the engine did not reuse its KV")

def test_the_context_must_hold_the_check():
    clock = FakeClock()
    w = EngineWorker(FakeNative(clock=clock), EngineConfig(Path("f.gguf"), ctx=64), clock=clock)
    try:
        with pytest.raises(EngineError, match="needs a context of at least 161"):
            run_live_checks(w, clock=clock, prefix=128, tail=16, generate=32)
    finally:
        w.close()
```

Extend `report_with` in `tests/abi/test_check.py` with a `live_checks` argument and add:

```python
def test_report_shows_the_live_checks():
    text = format_report(report_with(live_checks={
        "sync_plans": {"ok": True, "message": "extend of 64 tokens took 0.120 s"},
        "restore": {"ok": False, "message": "the runs differ at token 3: 7 after a restore, 9 after a rebuild"}}))
    assert "sync:     ok (extend of 64 tokens took 0.120 s)" in text
    assert "restore:  disabled (the runs differ at token 3: 7 after a restore, 9 after a rebuild)" in text
    assert "sync:" not in format_report(report_with())

def test_live_needs_a_model(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["check", "--live"])
    assert e.value.code == 2 and "--live needs --model" in capsys.readouterr().err
```

`live_lib` in `tests/conftest.py` is a session fixture that skips unless `DS4_TEST_MODEL` is set and returns `resolve_lib_path(None)`, skipping when that file does not exist.

`tests/live/test_live_check_cli.py` runs the check on a copy of the library, so the user's manifest is not changed:

```python
@pytest.mark.live
def test_check_live_records_its_results(live_lib, tmp_path):
    lib = tmp_path / live_lib.name
    shutil.copy(live_lib, lib)
    shutil.copy(manifest_path(live_lib), manifest_path(lib))
    r = subprocess.run([sys.executable, "-m", "ds4py", "check", "--live", "--lib", str(lib),
                        "--model", os.environ["DS4_TEST_MODEL"]], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "sync:     ok" in r.stdout and "restore:  not verified" not in r.stdout
    checks = Manifest.read(manifest_path(lib)).live_checks
    assert checks["sync_plans"]["ok"] and isinstance(checks["restore"]["ok"], bool)
```

`tests/live/test_raw_parity.py` is M1's raw parity criterion. `DS4_AGENT` must point to a `ds4-agent` built from the same ds4 tree as the library:

```python
PROMPT = "The three primary colors are"

@pytest.mark.live
def test_raw_matches_ds4_agent(live_lib, tmp_path):
    agent = os.environ.get("DS4_AGENT") or pytest.skip("DS4_AGENT is not set")
    backend = Manifest.read(manifest_path(live_lib)).backend
    common = ["-m", os.environ["DS4_TEST_MODEL"], "-c", "4096", "-n", "64", "--temp", "0", "--seed", "1"]
    c = subprocess.run([agent, "--non-interactive", "--raw", "-p", PROMPT, *common, f"--{backend}",
                        "--trace", str(tmp_path / "c.log")], capture_output=True)
    p = subprocess.run([sys.executable, "-m", "ds4py", "raw", PROMPT, *common, "--backend", backend,
                        "--lib", str(live_lib), "--trace", str(tmp_path / "py.log")], capture_output=True)
    assert c.returncode == 0 and p.returncode == 0, (c.stderr, p.stderr)
    assert read_trace_tokens(tmp_path / "py.log") == read_trace_tokens(tmp_path / "c.log")
    assert p.stdout == c.stdout
```

`ds4-agent` accepts `--raw` only with `--non-interactive` and `-p` (`ds4_agent.c:993`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/abi/test_live_checks.py tests/abi/test_check.py -v`
Expected: FAIL with `ImportError` for `run_live_checks`.

- [ ] **Step 3: Implement the live checks**

`LIVE_TEXT` is the opening sentence of *A Tale of Two Cities*, which is in the public domain and varied enough that a drifting state changes the greedy output:

```python
LIVE_TEXT = (
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of "
    "foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, "
    "it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had "
    "everything before us, we had nothing before us, we were all going direct to Heaven, we were all going "
    "direct the other way -- in short, the period was so far like the present period, that some of its "
    "noisiest authorities insisted on its being received, for good or for evil, in the superlative degree "
    "of comparison only. "
)
```

`run_live_checks`:
1. When `prefix + generate + 1` exceeds the session's context, raise `EngineError(f"ds4py check --live needs a context of at least {prefix + generate + 1}")`.
2. Tokenize `LIVE_TEXT` repeated until it gives at least `prefix` tokens, and keep the first `prefix`. `base` is the first `prefix - tail` of them.
3. Sync `base`, then save `LIVE_CHECKPOINT`.
4. **Extend:** sync all `prefix` tokens.
5. **Restore:** invalidate the live state with `worker.call(lambda e, s: s.invalidate())`, sync again, then generate `generate` tokens with `GenerateJob(SamplingPolicy(temperature=0.0), Rng(1), generate, Transcript())`. Call this run `a`.
6. **Rebuild:** drop `LIVE_CHECKPOINT`, invalidate, sync again, then generate run `b` the same way.

Time each sync in steps 4–6 with `clock()` from `submit` to its `Synced` event, and keep the `Synced.plan`. A run is the list of `Token.id`s plus the `Stopped.reason`.

`sync_plans` fails on the first of these:
- A plan other than `extend` with `to_prefill == tail`, `restore` with `to_prefill == tail` and `checkpoint == LIVE_CHECKPOINT`, or `rebuild` with `to_prefill == prefix`, in steps 4, 5 and 6. The message is `the <step> sync planned <kind> of <n> tokens, expected <step> of <m>`.
- An extend or restore that took more than half of the rebuild's time. The message is `<step> of <tail> tokens took <s> s, more than half of the <r> s rebuild; the engine did not reuse its KV`.

On success the message is `extend of <tail> tokens took <s> s, restore of <tail> took <s> s, rebuild of <prefix> took <s> s`. Times are printed with three decimals.

`restore` passes when `a` equals `b`, with the message `<len(a) tokens> greedy tokens after a restore match a rebuild`, and `tokens` is that count. Otherwise the message names the first index `i` where the runs differ: `the runs differ at token <i>: <x> after a restore, <y> after a rebuild`, where `x` and `y` are token ids or `end (<reason>)`.

`check_live`:
1. `library = load_library(lib)`, `native = LibNative(library)`, and `native.restore_enabled = True`, so a restore that failed before is checked again.
2. Start `EngineWorker(native, EngineConfig(Path(model), ctx=ctx, backend=native.backend))`, run `run_live_checks`, and close the worker in `finally`.
3. Set `library.manifest.live_checks` to the result and write it to `manifest_path(library.path)`. The library hash does not change, so the library still loads.
4. Return `check_library(library.path)`. The loader returns the `Library` it already holds, so the report shows the manifest that step 3 updated in memory.

`format_report` adds the `sync:` line before `restore:` when `live_checks` has `sync_plans`, and fills in `restore:` from `live_checks["restore"]`:

```text
sync:     ok (<message>)                     # or: failed (<message>)
restore:  ok (<message>)                     # or: disabled (<message>); "not verified (...)" when never checked
```

`cli.py`: `check --live` calls `check_live` and prints the report. It exits 1 when the report has an error or `sync_plans` failed, and 0 otherwise. A failed restore check still exits 0: ds4py works without restore, and `LibNative` turns it off.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -v`
Expected: PASS, with `build` and `live` tests skipped.

On a machine with a model, also run: `DS4_TEST_MODEL=<model> DS4_AGENT=<tree>/ds4-agent uv run pytest -m live -v`.
Expected: PASS. These are M1's two live criteria; they do not block M2.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py tests
git commit -m "feat(abi): add ds4py check --live with sync-plan and restore checks"
```

### Task 15: CI

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `pins.UPSTREAM`, `pins.FORK`, the `build` and `live` pytest markers, `DS4_SRC_UPSTREAM`, `DS4_SRC_FORK`, `tests/fixtures/ext_site`.
- Produces: two jobs. `unit` runs every test that needs neither a ds4 tree nor a model, and checks that the wheel carries `Makefile.lib`. `build` runs once per pin: it builds the CPU library, runs the `build` tests and runs `ds4py build` and `ds4py check` as a user would.

This task adds no tests: the workflow is configuration, and the tests it runs already exist. It is verified with `actionlint` and by running its commands locally.

- [ ] **Step 1: Write the workflow**

```yaml
name: ci

on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read

defaults:
  run:
    shell: bash          # adds -o pipefail

jobs:
  unit:
    strategy:
      fail-fast: false
      matrix:
        include:
          - { os: ubuntu-latest, python: "3.11" }
          - { os: ubuntu-latest, python: "3.13" }
          - { os: macos-latest, python: "3.11" }
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@v10
        with:
          python-version: ${{ matrix.python }}
      - run: uv run pytest -m "not build and not live" -v
      - name: The wheel carries Makefile.lib
        run: |
          uv build --wheel
          unzip -l dist/*.whl | grep -q 'ds4py/abi/Makefile.lib'

  build:
    strategy:
      fail-fast: false
      matrix:
        include:
          - { pin: upstream, src_var: DS4_SRC_UPSTREAM }
          - { pin: fork, src_var: DS4_SRC_FORK }
    runs-on: ubuntu-latest
    env:
      PIN: ${{ matrix.pin }}
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@v10
        with:
          python-version: "3.11"
      - name: Read the pin
        id: pin
        run: |
          uv run python - "$PIN" >> "$GITHUB_OUTPUT" <<'EOF'
          import sys
          from ds4py.abi import pins
          pin = {"upstream": pins.UPSTREAM, "fork": pins.FORK}[sys.argv[1]]
          print("repo=" + pin.repo.removeprefix("https://github.com/").removesuffix(".git"))
          print("commit=" + pin.commit)
          EOF
      - uses: actions/checkout@v7
        with:
          repository: ${{ steps.pin.outputs.repo }}
          ref: ${{ steps.pin.outputs.commit }}
          path: ds4-src
          persist-credentials: false
      - name: Build tests
        run: env "${{ matrix.src_var }}=$GITHUB_WORKSPACE/ds4-src" uv run pytest -m build -v
      - name: ds4py build and ds4py check
        run: |
          if [ "$PIN" = fork ]; then export PYTHONPATH="$GITHUB_WORKSPACE/tests/fixtures/ext_site"; fi
          uv run ds4py build --src ds4-src --backend cpu --out out
          uv run ds4py check --lib out/libds4.so
```

Each `build` job sets only its own `DS4_SRC_*` variable, so the build tests for the other pin skip. The fork job puts the test extension on `PYTHONPATH`, as `built_fork` does, so `ds4py check` lists its bindings. The last step builds the library a second time, into `out/` instead of a pytest temporary directory. That doubles the compile time, but it runs the installed `ds4py` console script the way a user would, and its report stays in the job log.

- [ ] **Step 2: Lint the workflow**

Run: `nix run nixpkgs#actionlint -- .github/workflows/ci.yml`
Expected: no output, exit 0.

- [ ] **Step 3: Run the jobs' commands locally**

Run:

```bash
uv run pytest -m "not build and not live" -v
uv build --wheel && unzip -l dist/*.whl | grep 'ds4py/abi/Makefile.lib'
uv run python - fork <<'EOF'
import sys
from ds4py.abi import pins
pin = {"upstream": pins.UPSTREAM, "fork": pins.FORK}[sys.argv[1]]
print("repo=" + pin.repo.removeprefix("https://github.com/").removesuffix(".git"))
print("commit=" + pin.commit)
EOF
```

Expected: the tests pass with the `build` and `live` tests deselected; `grep` prints the `Makefile.lib` line; the last command prints `repo=pauleveritt/ds4` and `commit=cfb2faf5c03199790db7c22166c4c71b549f1e66`.

- [ ] **Step 4: Commit and watch the first run**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: run unit tests and CPU builds of both ds4 pins"
```

After the branch is pushed, `gh run watch` must show every job green. In the `build (fork)` log, `ds4py check` reports `warning: unknown header ds4.h b1e0ad190897` and exits 0. These are M1's first done-when criteria.
