"""Artifact receipts locate the loaded descriptor, not a supplied filename."""

from __future__ import annotations

import ctypes
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from generativeqc.automatic_artifacts import (
    _loaded_artifact,
    read_automatic_artifact_diagnostic,
)
from generativeqc_compiler.common.provenance import file_hash

if TYPE_CHECKING:
    from conftest import NativeCxx


@pytest.mark.parametrize("version", (1, 2))
def test_executed_descriptor_binds_real_binary_and_explicit_work(
    tmp_path: Path,
    native_cxx: NativeCxx,
    version: int,
) -> None:
    source = tmp_path / "receipt.cpp"
    source_text = """
#include <cstdint>
#include <cstddef>
extern "C" int generativeqc_ks_batch_automatic_artifact_v1(
    const void*, std::uint32_t index, const char** text, std::size_t nt,
    const void** owners, std::size_t no, std::uint64_t* work, std::size_t nw) {
  if (nt != 7 || no != 2 || nw != 40) return 1;
  static const char* names[]{"automatic", "math", "point-source", "force-profile",
                             "force-plan", "force-source", "none"};
  for (unsigned i=0;i<7;++i) text[i] = names[i];
  owners[0] = names; owners[1] = index ? nullptr : names;
  for (unsigned i=0;i<40;++i) work[i] = 0;
  work[0] = !index; work[1] = 5; work[2] = 1; work[3] = index ? 0 : 7;
  if (!index) { work[4] = 4096; work[34] = 100; }
  else text[6] = "force-not-executed-at-current-token";
  return 0;
}
"""
    if version == 2:
        source_text = (
            source_text.replace("artifact_v1", "artifact_v2")
            .replace(
                "nt != 7 || no != 2 || nw != 40", "nt != 11 || no != 4 || nw != 43"
            )
            .replace(
                '"force-plan", "force-source", "none"',
                '"force-plan", "force-source", "none", "consumer-source", '
                '"shared_012345678901234567890123", "polarized", "restricted"',
            )
            .replace("i<7", "i<11")
            .replace("i<40", "i<43")
            .replace(
                "owners[0] = names;",
                "owners[0] = names; owners[2] = names; owners[3] = index ? nullptr : names;",
            )
            .replace(
                "work[34] = 100;",
                "work[34] = 100; work[40] = 8192; work[41] = 2; work[42] = 100;",
            )
        )
    source.write_text(source_text)
    binary = tmp_path / "executed.so"
    native_cxx.build_shared([source], binary, compile_args=("-std=c++20",))
    loaded = ctypes.CDLL(str(binary))
    query_name = f"generativeqc_ks_batch_automatic_artifact_v{version}"
    batch = SimpleNamespace(
        _ensure_open=lambda: None,
        _batch=None,
        _library=SimpleNamespace(
            _name=str(tmp_path / "unexecuted-decoy.so"),
            **{query_name: getattr(loaded, query_name)},
        ),
    )
    receipt = read_automatic_artifact_diagnostic(batch)
    assert (
        receipt["point_artifact"]
        == receipt["force_artifact"]
        == {
            "path": str(binary),
            "sha256": file_hash(binary),
            "bytes": binary.stat().st_size,
        }
    )
    assert receipt["force_completed"] and receipt["source_count"] == 7
    assert receipt["capacity"]["device_upper_bytes"] == 4096
    assert receipt["grid_work"]["evaluation_points"] == 100
    assert receipt["schema"] == f"generativeqc.native-automatic-artifact.v{version}"
    if version == 2:
        assert (
            receipt["point_executable_artifact"]
            == receipt["force_executable_artifact"]
            == receipt["point_artifact"]
        )
        assert receipt["force_schedule"] == "shared-staged"
        assert receipt["staged_point_work"] == {
            "tile_bytes": 8192,
            "launches": 2,
            "points": 100,
        }
    energy = read_automatic_artifact_diagnostic(batch, 1)
    assert not energy["force_completed"] and energy["force_artifact"] is None
    assert energy["point_artifact"] == receipt["point_artifact"]
    assert not any(energy["stationary_work_v1"])
    if version == 2:
        assert not any(energy["staged_point_work"].values())
        assert energy["force_executable_artifact"] is None
    assert energy["fallback_reason"] == "force-not-executed-at-current-token"
    for index in (-1, 2**32, True):
        with pytest.raises(ValueError, match="uint32"):
            read_automatic_artifact_diagnostic(batch, index)
    if Path("/proc/self/maps").exists():
        replacement = tmp_path / "replacement.so"
        replacement.write_bytes(binary.read_bytes())
        replacement.replace(binary)
        with pytest.raises(RuntimeError, match="replaced"):
            read_automatic_artifact_diagnostic(batch)


def test_bss_descriptor_and_executable_callback_bind_the_same_module(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """A shared registration's BSS page need not carry the module's inode."""
    source = tmp_path / "bss.cpp"
    source.write_text("""
extern "C" void* descriptor() {
  static char storage[65536];
  return storage + 32768;
}
""")
    binary = native_cxx.build_shared([source], tmp_path / "bss.so")
    loaded = ctypes.CDLL(str(binary))
    loaded.descriptor.restype = ctypes.c_void_p
    descriptor = loaded.descriptor()
    callback = ctypes.cast(loaded.descriptor, ctypes.c_void_p).value
    memo = {}
    assert _loaded_artifact(descriptor, memo) == _loaded_artifact(callback, memo)
    replacement = tmp_path / "replacement.so"
    replacement.write_bytes(binary.read_bytes())
    replacement.replace(binary)
    if Path("/proc/self/maps").exists():
        with pytest.raises(RuntimeError, match="replaced"):
            _loaded_artifact(descriptor, memo)
