"""Artifact receipts locate the loaded descriptor, not a supplied filename."""

from __future__ import annotations

import ctypes
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from generativeqc.automatic_artifacts import read_automatic_artifact_diagnostic
from generativeqc_compiler.common.provenance import file_hash

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_executed_descriptor_binds_real_binary_and_explicit_work(
    tmp_path: Path,
    native_cxx: NativeCxx,
) -> None:
    source = tmp_path / "receipt.cpp"
    source.write_text("""
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
""")
    binary = tmp_path / "executed.so"
    native_cxx.build_shared([source], binary, compile_args=("-std=c++20",))
    loaded = ctypes.CDLL(str(binary))
    batch = SimpleNamespace(
        _ensure_open=lambda: None,
        _batch=None,
        _library=SimpleNamespace(
            _name=str(tmp_path / "unexecuted-decoy.so"),
            generativeqc_ks_batch_automatic_artifact_v1=loaded.generativeqc_ks_batch_automatic_artifact_v1,
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
    energy = read_automatic_artifact_diagnostic(batch, 1)
    assert not energy["force_completed"] and energy["force_artifact"] is None
    assert energy["point_artifact"] == receipt["point_artifact"]
    assert not any(energy["stationary_work_v1"])
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
