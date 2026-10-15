"""Execute native schedule preference without a CUDA runtime or device query.

This checks selection only; numerical/output leases and real-device arithmetic
are qualified independently by the admission and force-phase tests.
"""

from __future__ import annotations

import ctypes as ct
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from test_nonlocal_signed_weight_kernels import _definition

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def schedule_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> ct.CDLL:
    source = (ROOT / "src/dft/nonlocal_correlation/vv10_runtime_cuda.cu").read_text()
    folder = tmp_path_factory.mktemp("nonlocal-pair-schedule")
    cpp, library = folder / "probe.cpp", folder / "probe.so"
    cpp.write_text(
        "#include <cstddef>\n#include <cstdlib>\n#include <string_view>\n"
        + _definition(source, "bool prefer_parallel_pairs(")
        + '\nextern "C" bool prefer(std::size_t count) {'
        + " return prefer_parallel_pairs(count); }\n"
    )
    native_cxx.build_shared([cpp], library, compile_args=["-std=c++17", "-O2"])
    probe = ct.CDLL(str(library))
    probe.prefer.argtypes = [ct.c_size_t]
    probe.prefer.restype = ct.c_bool
    return probe


@pytest.mark.parametrize("count", (0, 1, 65535, 65536, 65537, 1 << 32))
@pytest.mark.parametrize(
    "selector",
    (None, "", "parallel", "parallel-packet", "parallel-wide", "ordered", "unknown"),
)
def test_schedule_preference_without_device_dependency(
    schedule_probe: ct.CDLL,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    selector: str | None,
) -> None:
    """Preserve the size boundary, explicit overrides and safe unknown fallback."""
    if selector is None:
        monkeypatch.delenv("GENERATIVEQC_VV10_PAIR_SCHEDULE", raising=False)
    else:
        monkeypatch.setenv("GENERATIVEQC_VV10_PAIR_SCHEDULE", selector)
    expected = (
        count >= 65536
        if not selector
        else selector in ("parallel", "parallel-packet", "parallel-wide")
    )
    assert schedule_probe.prefer(count) == expected
