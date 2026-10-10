"""Exact occupied-source aliasing leaves the independent energy algebra intact."""

from __future__ import annotations

import ctypes as ct
import hashlib
import itertools
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc_compiler.cc.occupied_triples import PERMUTATIONS
from test_df_occupied_triples import case, reference

from tools.generate_df_occupied_triples import (
    cuda_source,
    fock_cuda_source,
    header,
    response_cuda_source,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def emitted_probe(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Run actual emitted load/finite logic with one host lane, not CUDA timing."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("C++ compiler and ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    directory = tmp_path_factory.mktemp("distinct-occupied-moments")
    generated = cuda_source()
    kernels = generated[
        generated.index("__global__ void tile_kernel") : generated.index(
            "void energy_tile"
        )
    ]
    (directory / "generated_df_occupied_triples.hpp").write_text(header())
    source = directory / "probe.cpp"
    source.write_text(
        """#include <algorithm>
#include <cmath>
#include <cstddef>
#include "generated_df_occupied_triples.hpp"
#define __global__
#define __shared__
struct Index { std::size_t x; };
static Index blockIdx{0},threadIdx{0},blockDim{1},gridDim{1};
inline void __syncthreads() {}
using std::isfinite;
inline int atomicCAS(int* pointer,int compare,int value) {
  const int previous=*pointer; if (previous==compare) *pointer=value; return previous;
}
namespace generativeqc_tensor {
inline double finite(double value,int* error,int code) {
  if (!std::isfinite(value)) atomicCAS(error,0,code); return value;
}
}
namespace generativeqc::cc::triples::generated_df {
"""
        + kernels
        + """}
extern "C" void sources(std::size_t first,std::size_t second,std::size_t third,unsigned* output) {
  const auto plan=generativeqc::cc::triples::generated_df::occupied_moment_sources(first,second,third);
  std::copy(plan.index,plan.index+6,output);
}
extern "C" double energy(bool aliases,std::size_t occupied_count,std::size_t virtual_count,
                        std::size_t first,std::size_t second,std::size_t third,
                        double degeneracy,const double* const* inputs,const double* moments,int* error) {
  using namespace generativeqc::cc::triples::generated_df;
  Inputs values{inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],inputs[6],inputs[7],inputs[8]};
  double partial{};
  if (aliases)
    distinct_tile_kernel(occupied_count,virtual_count,first,second,third,degeneracy,1e-12,
                        values,moments,occupied_moment_sources(first,second,third),&partial,error);
  else
    tile_kernel(occupied_count,virtual_count,first,second,third,degeneracy,1e-12,values,moments,&partial,error);
  return partial;
}
"""
    )
    obj, library = directory / "probe.o", directory / "probe.so"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
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
    pointer = ct.POINTER(ct.c_double)
    dll.sources.argtypes = [ct.c_size_t] * 3 + [ct.POINTER(ct.c_uint)]
    dll.sources.restype = None
    dll.energy.argtypes = (
        [ct.c_bool]
        + [ct.c_size_t] * 5
        + [ct.c_double, ct.POINTER(pointer), pointer, ct.POINTER(ct.c_int)]
    )
    dll.energy.restype = ct.c_double
    return dll


def _tiles(occupied_count: int) -> Iterator[tuple[int, int, int]]:
    for first in range(occupied_count):
        for second in range(first + 1):
            for third in range(second + 1):
                yield first, second, third


def _sources(probe: Any, occupied: tuple[int, int, int]) -> list[int]:
    result = (ct.c_uint * 6)()
    probe.sources(*occupied, result)
    return list(result)


def test_every_equality_pattern_selects_the_first_identical_physical_tuple(
    emitted_probe: Any,
) -> None:
    for occupied in itertools.product(range(4), repeat=3):
        orders = [tuple(occupied[axis] for axis in order) for order in PERMUTATIONS]
        actual = _sources(emitted_probe, occupied)
        assert actual == [orders.index(order) for order in orders]
        assert all(actual[source] == source for source in actual)
    for occupied_count in range(1, 10):
        assert (
            sum(
                len(set(_sources(emitted_probe, tile)))
                for tile in _tiles(occupied_count)
            )
            == occupied_count**3
        )


@pytest.mark.parametrize(
    "occupied_count,virtual_count,auxiliary_count",
    [(1, 3, 4), (2, 3, 4), (3, 2, 1), (3, 4, 4), (4, 3, 5)],
)
def test_actual_emitted_bindings_ignore_poisoned_duplicates_and_match_independent_energy(
    emitted_probe: Any, occupied_count: int, virtual_count: int, auxiliary_count: int
) -> None:
    inputs, ovvv = case(occupied_count, virtual_count, auxiliary_count)
    _, _, ovoo, _, _, _, t2, _, _ = inputs
    pointer = ct.POINTER(ct.c_double)
    input_pointers = (pointer * 9)(*(array.ctypes.data_as(pointer) for array in inputs))
    old_energy = alias_energy = 0.0
    for occupied in _tiles(occupied_count):
        moments = np.ascontiguousarray(
            [
                np.einsum(
                    "afb,cf->abc",
                    ovvv[occupied[order[0]]],
                    t2[occupied[order[2]], occupied[order[1]]],
                )
                - np.einsum(
                    "am,mbc->abc",
                    ovoo[occupied[order[0]], :, occupied[order[1]]],
                    t2[:, occupied[order[2]]],
                )
                for order in PERMUTATIONS
            ]
        )
        sources = _sources(emitted_probe, occupied)
        poisoned = moments.copy()
        for slot, canonical in enumerate(sources):
            if slot != canonical:
                poisoned[slot] = np.nan
        first, second, third = occupied
        degeneracy = (
            6 if first == third else 2 if first == second or second == third else 1
        )
        arguments = (
            occupied_count,
            virtual_count,
            *occupied,
            degeneracy,
            input_pointers,
        )
        error = ct.c_int()
        old = emitted_probe.energy(
            False, *arguments, moments.ctypes.data_as(pointer), ct.byref(error)
        )
        assert error.value == 0 and np.isfinite(old)
        actual = emitted_probe.energy(
            True, *arguments, poisoned.ctypes.data_as(pointer), ct.byref(error)
        )
        assert error.value == 0 and np.isfinite(actual)
        np.testing.assert_allclose(actual, old, atol=2e-12, rtol=2e-12)
        old_energy += old
        alias_energy += actual
        if len(set(sources)) < 6:
            emitted_probe.energy(
                False, *arguments, poisoned.ctypes.data_as(pointer), ct.byref(error)
            )
            assert error.value != 0
    np.testing.assert_allclose(
        [old_energy, alias_energy], reference(inputs, ovvv), atol=3e-12, rtol=3e-12
    )


def test_all_equal_tile_still_checks_nonfinite_moments(emitted_probe: Any) -> None:
    inputs, _ = case(1, 3, 4)
    pointer = ct.POINTER(ct.c_double)
    input_pointers = (pointer * 9)(*(array.ctypes.data_as(pointer) for array in inputs))
    moments = np.full((6, 3, 3, 3), np.nan)
    error = ct.c_int()
    emitted_probe.energy(
        True,
        1,
        3,
        0,
        0,
        0,
        6,
        input_pointers,
        moments.ctypes.data_as(pointer),
        ct.byref(error),
    )
    assert error.value != 0


def test_legacy_energy_types_response_and_fock_sources_are_unchanged() -> None:
    """Hashes pin the pre-alias compiler output, not a second copied formula."""
    source = cuda_source()
    original = source[
        source.index("__global__ void tile_kernel") : source.index(
            "__global__ void distinct_tile_kernel"
        )
    ]
    assert (
        hashlib.sha256(original.encode()).hexdigest()
        == "ec1e4552be2b2f9165dc7109843990b3fd217fafc3699f53bc7cec379457d77e"
    )
    generated_header = header()
    start = generated_header.index("/** Alias only equal physical occupied tuples;")
    end = generated_header.index("struct Inputs", start)
    legacy_header = generated_header[:start] + generated_header[end:]
    assert (
        hashlib.sha256(legacy_header.encode()).hexdigest()
        == "00f7d5f4fd5713857a08df8d23949f2446b2c974ad8a93bf6358a5064319ae42"
    )
    assert (
        hashlib.sha256(response_cuda_source().encode()).hexdigest()
        == "b65ddf83a4f7a3d6f7311cebf9deb5c277bb6e6075c5b3fce748a41bb9d232f8"
    )
    assert (
        hashlib.sha256(fock_cuda_source().encode()).hexdigest()
        == "f78824786a1eaabdb7a4880fe7a30581a763e81941c900f60dca3e3674efdd18"
    )


def test_forces_keep_full_response_materialization() -> None:
    owner = (ROOT / "src/cc/df_triples_cuda.cu").read_text()
    response = owner[owner.index("static DFCudaResponseResult pullback_df_cuda_impl") :]
    assert "MomentSourceMap" not in response and "energy_distinct_tile" not in response
    endpoint = (ROOT / "src/methods/df_ccsdt_force.cu").read_text()
    assert (
        "if (forces) {\n      auto response = cc::triples::pullback_and_fock_df_cuda("
        in endpoint
    )
    assert "} else {\n      result.triples = cc::triples::evaluate_df_cuda(" in endpoint
