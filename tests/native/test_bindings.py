from dataclasses import FrozenInstanceError
from importlib.metadata import EntryPoint

import pytest

from ds4py._native import bindings, typemap


def test_core_bindings_are_the_m1_list():
    names = [b.name for b in bindings.CORE_BINDINGS]
    assert len(names) == len(set(names)) == 52
    assert all(b.required for b in bindings.CORE_BINDINGS)
    assert {"ds4_session_sync", "ds4_session_save_snapshot", "ds4_session_checkpoint_valid",
            "ds4_session_stage_payload", "ds4_session_write_staged_payload",
            "ds4_session_payload_file_free", "ds4_session_load_payload",
            "ds4_think_mode_parse_level", "ds4_chat_begin", "ds4_chat_append_think_prefix",
            "ds4_chat_append_message", "ds4_chat_append_assistant_prefix",
            "ds4_qwen4_reasoning_effort_text"} <= set(names)


def test_every_core_binding_maps_to_ctypes():
    for b in bindings.CORE_BINDINGS:
        typemap.ctypes_signature(typemap.parse_prototype(b.prototype))


def test_binding_name_must_match_prototype():
    with pytest.raises(ValueError):
        bindings.Binding("ds4_a", "int ds4_b(void)")


def test_binding_rejects_unmapped_types_at_declaration():
    from ds4py.errors import TypeMapError

    with pytest.raises(TypeMapError, match="ds4_extra"):
        bindings.Binding("ds4_extra", "mystery_t ds4_extra(void)")


def test_binding_normalization_and_immutability():
    binding = bindings.Binding("ds4_extra", " int ds4_extra( const ds4_tokens* prompt ); ")
    assert binding.normalized == "int ds4_extra(const ds4_tokens *)"
    with pytest.raises(FrozenInstanceError):
        binding.required = False


def test_no_extensions_installed_by_default():
    assert bindings.discover_extension_bindings() == ()
    assert bindings.all_bindings() == bindings.CORE_BINDINGS


def test_extension_discovered_through_entry_point(ext_site):
    found = {b.name: b for b in bindings.discover_extension_bindings()}
    assert found["ds4_engine_layer_count"].required
    assert not found["ds4_engine_sampling_defaults"].required
    assert found["ds4_engine_layer_count"].source == "test_ext"
    combined = bindings.all_bindings()
    assert combined[:52] == bindings.CORE_BINDINGS
    assert [b.name for b in combined][-2:] == list(found)


def test_extension_may_not_redeclare_a_core_binding(ext_site, monkeypatch):
    import ds4py_test_ext
    monkeypatch.setattr(ds4py_test_ext, "BINDINGS",
                        (bindings.Binding("ds4_session_pos", "int ds4_session_pos(ds4_session *s)"),))
    with pytest.raises(ValueError, match="ds4_session_pos"):
        bindings.discover_extension_bindings()


def install_providers(monkeypatch, *providers):
    """Use real EntryPoint loading, isolating only the installed provider list."""
    import sys
    from types import ModuleType

    points = []
    for index, declarations in enumerate(providers):
        name = f"binding_provider_{index}"
        module = ModuleType(name)
        module.BINDINGS = declarations
        monkeypatch.setitem(sys.modules, name, module)
        points.append(EntryPoint(name=name, value=f"{name}:BINDINGS", group="ds4py.bindings"))
    monkeypatch.setattr(bindings.metadata, "entry_points", lambda *, group: points)


@pytest.mark.parametrize("first_required", [False, True])
@pytest.mark.parametrize("same_provider", [False, True])
def test_compatible_duplicates_merge_requirements_keep_first_source_and_position(
    monkeypatch, first_required, same_provider
):
    first = bindings.Binding("ds4_extra", "int ds4_extra(ds4_engine *e)", first_required)
    other = bindings.Binding("ds4_other", "void ds4_other(void)", False)
    duplicate = bindings.Binding("ds4_extra", "int ds4_extra(ds4_engine* engine)", not first_required)
    providers = ((first, other, duplicate),) if same_provider else ((first, other), (duplicate,))
    install_providers(monkeypatch, *providers)
    found = bindings.discover_extension_bindings()
    assert [b.name for b in found] == ["ds4_extra", "ds4_other"]
    assert found[0].required
    assert found[0].source == "binding_provider_0"
    assert found[0].prototype == first.prototype
    assert first.required == first_required


@pytest.mark.parametrize("same_provider", [False, True])
def test_conflicting_extension_prototypes_are_rejected(monkeypatch, same_provider):
    first = bindings.Binding("ds4_extra", "int ds4_extra(void)", False)
    conflict = bindings.Binding("ds4_extra", "long ds4_extra(void)")
    providers = ((first, conflict),) if same_provider else ((first,), (conflict,))
    install_providers(monkeypatch, *providers)
    with pytest.raises(ValueError, match="ds4_extra"):
        bindings.discover_extension_bindings()


def test_provider_must_yield_bindings(monkeypatch):
    install_providers(monkeypatch, ("not a Binding",))
    with pytest.raises(TypeError, match="binding_provider_0"):
        bindings.discover_extension_bindings()
