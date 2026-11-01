from typing import Any
from pathlib import Path
import hashlib
import os
import shutil

import pytest

from ds4py.errors import AbiMismatch, BuildError


from ds4py.abi.shaders import copy_metal_sources as copy_sources, metal_source_env as source_env
from typing import Never


def make_source(root: Any) -> Any:
    (root / "metal").mkdir(parents=True)
    for name, data in {"dense.metal": b"dense shader", "flash_attn.metal": b"attention shader"}.items():
        (root / "metal" / name).write_bytes(data)
    return root


def test_captured_files_survive_changed_and_deleted_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = make_source(tmp_path / "source")
    (src / "metal/ignored.h").write_text("not a shader")
    bundle = tmp_path / "bundle"
    hashes = copy_sources(src, bundle)
    assert hashes == {
        "metal/dense.metal": hashlib.sha256(b"dense shader").hexdigest(),
        "metal/flash_attn.metal": hashlib.sha256(b"attention shader").hexdigest(),
    }
    assert (bundle / "metal/dense.metal").read_bytes() == b"dense shader"
    assert not (bundle / "metal/dense.metal").is_symlink()
    (src / "metal/dense.metal").write_bytes(b"changed checkout")
    shutil.rmtree(src)
    moved = tmp_path / "moved"
    bundle.rename(moved)
    before = dict(os.environ)
    monkeypatch.setenv("DS4_METAL_DENSE_SOURCE", "/ignored/process/environment")
    result = source_env(moved, hashes)
    assert result == {
        "DS4_METAL_DENSE_SOURCE": str(moved / "metal/dense.metal"),
        "DS4_METAL_FLASH_ATTN_SOURCE": str(moved / "metal/flash_attn.metal"),
    }
    # The resolver ignores the real environment and never mutates it.
    assert dict(os.environ) == {**before, "DS4_METAL_DENSE_SOURCE": "/ignored/process/environment"}


@pytest.mark.parametrize("damage", ["missing", "changed", "escape", "loop"])
def test_bad_bundled_files_require_rebuild(tmp_path: Path, damage: Any) -> None:
    src = make_source(tmp_path / "source")
    bundle = tmp_path / "bundle"
    hashes = copy_sources(src, bundle)
    shader = bundle / "metal/dense.metal"
    shader.unlink()
    if damage == "changed":
        shader.write_bytes(b"new version")
    elif damage == "escape":
        shader.symlink_to(src / "metal/dense.metal")  # same bytes, wrong ownership
    elif damage == "loop":
        shader.symlink_to(shader)
    with pytest.raises(AbiMismatch, match="ds4py build"):
        source_env(bundle, hashes)


@pytest.mark.parametrize("hashes", [
    {}, None, [], {"metal/dense.metal": "bad"}, {"metal/dense.metal": None},
    {"/metal/dense.metal": "0" * 64}, {"../dense.metal": "0" * 64},
    {"metal/../dense.metal": "0" * 64}, {"metal/nested/dense.metal": "0" * 64},
    {"metal//dense.metal": "0" * 64}, {"metal/dense.txt": "0" * 64},
    {1: "0" * 64},
])
def test_malformed_records_require_rebuild(tmp_path: Path, hashes: Any) -> None:
    with pytest.raises(AbiMismatch, match="ds4py build"):
        source_env(tmp_path, hashes)


def test_overrides_must_match_and_keep_their_original_strings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = make_source(tmp_path / "source")
    bundle = tmp_path / "bundle"
    hashes = copy_sources(src, bundle)
    monkeypatch.chdir(tmp_path)
    override = tmp_path / "override.metal"
    override.write_bytes(b"dense shader")
    overrides = {"DS4_METAL_DENSE_SOURCE": "./override.metal", "UNRELATED": "kept"}
    assert source_env(bundle, hashes, overrides)["DS4_METAL_DENSE_SOURCE"] == "./override.metal"
    assert overrides == {"DS4_METAL_DENSE_SOURCE": "./override.metal", "UNRELATED": "kept"}
    override.write_bytes(b"wrong shader")
    with pytest.raises(AbiMismatch, match="DS4_METAL_DENSE_SOURCE.*ds4py build"):
        source_env(bundle, hashes, overrides)
    override.unlink()
    with pytest.raises(AbiMismatch, match="DS4_METAL_DENSE_SOURCE.*ds4py build"):
        source_env(bundle, hashes, overrides)


def test_matching_override_does_not_hide_missing_bundle(tmp_path: Path) -> None:
    src = make_source(tmp_path / "source")
    bundle = tmp_path / "bundle"
    hashes = copy_sources(src, bundle)
    (bundle / "metal/dense.metal").unlink()
    with pytest.raises(AbiMismatch, match="ds4py build"):
        source_env(bundle, hashes, {"DS4_METAL_DENSE_SOURCE": str(src / "metal/dense.metal")})


def test_empty_shader_override_is_not_silently_ignored(tmp_path: Path) -> None:
    src = make_source(tmp_path / "source")
    hashes = copy_sources(src, tmp_path / "bundle")
    with pytest.raises(AbiMismatch, match="DS4_METAL_DENSE_SOURCE.*ds4py build"):
        source_env(tmp_path / "bundle", hashes, {"DS4_METAL_DENSE_SOURCE": ""})


@pytest.mark.parametrize("override", [False, True])
def test_nonregular_shader_files_are_refused_without_reading_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: Any) -> None:
    from ds4py.abi import shaders
    src = make_source(tmp_path / "source")
    bundle = tmp_path / "bundle"
    hashes = copy_sources(src, bundle)
    fifo = tmp_path / "override.metal" if override else bundle / "metal/dense.metal"
    fifo.unlink(missing_ok=True)
    os.mkfifo(fifo)
    original_hash = shaders.file_sha256
    def guarded_hash(path: Any) -> Any:
        if not path.is_file():
            pytest.fail("resolver attempted to read a nonregular file; a FIFO would block")
        return original_hash(path)
    monkeypatch.setattr(shaders, "file_sha256", guarded_hash)
    overrides = {"DS4_METAL_DENSE_SOURCE": str(fifo)} if override else None
    with pytest.raises(AbiMismatch, match="regular.*ds4py build"):
        source_env(bundle, hashes, overrides)


def test_copy_reports_missing_sources(tmp_path: Path) -> None:
    with pytest.raises(BuildError, match="Metal.*ds4py build"):
        copy_sources(tmp_path / "absent", tmp_path / "bundle")


def test_copy_reports_io_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = make_source(tmp_path / "source")
    def fail(*args: Any, **kwargs: Any) -> Never:
        raise OSError("disk full")
    monkeypatch.setattr(shutil, "copy2", fail)
    with pytest.raises(BuildError, match="disk full.*ds4py build"):
        copy_sources(src, tmp_path / "bundle")
