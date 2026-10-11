"""Bounded panel traversal changes visits, never canonical energy addresses."""

from __future__ import annotations

import ctypes as ct
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def visitor_probe(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Execute the production visitor with no GPU or runtime-library dependency."""
    compiler = shutil.which("c++")
    cache = shutil.which("ccache") or shutil.which("sccache")
    if not compiler or not cache:
        pytest.skip("C++ compiler and compiler cache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    directory = tmp_path_factory.mktemp("triples-traversal")
    source = directory / "visitor.cpp"
    source.write_text("""#include <cstddef>
#include "cc/df_triples_traversal.hpp"
extern "C" std::size_t visit(std::size_t occupied,bool snake,std::size_t* output) {
  std::size_t count=0;
  generativeqc::cc::triples::detail::visit_occupied_tiles(occupied,snake,
    [&](std::size_t first,std::size_t second,std::size_t third,std::size_t canonical) {
      output[4*count]=first; output[4*count+1]=second;
      output[4*count+2]=third; output[4*count+3]=canonical; ++count;
    });
  return count;
}
""")
    obj, library = directory / "visitor.o", directory / "visitor.so"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-I" + str(ROOT / "src"),
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [compiler, "-shared", str(obj), "-o", str(library)],
        check=True,
        capture_output=True,
    )
    dll = ct.CDLL(str(library))
    dll.visit.argtypes = [ct.c_size_t, ct.c_bool, ct.POINTER(ct.c_size_t)]
    dll.visit.restype = ct.c_size_t
    return dll


def _sequence(probe: Any, occupied: int, snake: bool) -> np.ndarray:
    count = occupied * (occupied + 1) * (occupied + 2) // 6
    result = np.zeros((count, 4), dtype=np.uintp)
    assert (
        probe.visit(occupied, snake, result.ctypes.data_as(ct.POINTER(ct.c_size_t)))
        == count
    )
    return result


def _panel_builds(sequence: np.ndarray, capacity: int) -> int:
    """Independent exact LRU matches the native panel lifetime, not GPU timing."""
    residents = [None] * capacity
    ages = [0] * capacity
    epoch = builds = 0
    for first, second, third, _ in sequence.tolist():
        physical = (first, second, third)
        for position, value in enumerate(physical):
            if value in physical[:position]:
                continue
            if value in residents:
                slot = residents.index(value)
            else:
                slot = min(range(capacity), key=ages.__getitem__)
                residents[slot] = value
                builds += 1
            epoch += 1
            ages[slot] = epoch
    return builds


@pytest.mark.parametrize("occupied", [1, 2, 3, 4, 5, 8, 9, 12, 16, 32])
def test_actual_visitor_covers_each_tile_and_preserves_original_flat_energy_index(
    visitor_probe: Any, occupied: int
) -> None:
    expected = [
        (first, second, third)
        for first in range(occupied)
        for second in range(first + 1)
        for third in range(second + 1)
    ]
    old = _sequence(visitor_probe, occupied, False)
    new = _sequence(visitor_probe, occupied, True)
    assert old[:, :3].tolist() == [list(row) for row in expected]
    np.testing.assert_array_equal(old[:, 3], np.arange(len(expected)))
    np.testing.assert_array_equal(new[np.argsort(new[:, 3])], old)
    assert _panel_builds(new, min(3, occupied)) <= _panel_builds(old, min(3, occupied))


def test_ethane_panel_work_is_reduced_without_new_storage(visitor_probe: Any) -> None:
    old = _sequence(visitor_probe, 9, False)
    new = _sequence(visitor_probe, 9, True)
    assert _panel_builds(old, 3) == 152 and _panel_builds(new, 3) == 98
    assert (152 - 98) * 488 * 221**3 == 284439825072


def test_runtime_route_preserves_original_precision_fallbacks_and_response() -> None:
    owner = (ROOT / "src/cc/df_triples_cuda.cu").read_text()
    assert (
        "detail::visit_occupied_tiles(o, distinct_moments && p.panel_capacity == 3, visit_tile);"
        in owner
    )
    assert "energies + canonical_tile" in owner
    response = owner[owner.index("static DFCudaResponseResult pullback_df_cuda_impl") :]
    assert "visit_occupied_tiles" not in response
    assert "std::array<std::size_t, 3> identities" in owner
