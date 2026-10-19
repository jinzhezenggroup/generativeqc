"""Execute the generated suffix contraction against an independent AO orbit sum."""

from __future__ import annotations

import ctypes as ct
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.integral.md_j_masked_cuda import emit_md_j_masked_header

_DRIVER = r"""
#include "generated_md_j_masked.cuh"
#include <array>
using namespace generativeqc::scf::cuda_execution;
struct Pair { unsigned first_ao, second_ao, first_count, second_count; };
template <unsigned KetHermites, bool Reverse>
struct Radial {
  const double* values;
  double operator()(unsigned output, unsigned input) const {
    return values[Reverse ? input * KetHermites + output : output * KetHermites + input];
  }
};

template <unsigned KetHermites>
int contract(const unsigned* descriptors, unsigned nbf, unsigned offset, bool reverse,
             bool unrestricted, double screening, const double* bounds, const double* density,
             const double* beta, unsigned bra_primitives, unsigned ket_primitives,
             const double* bra_transform, const double* ket_transform, const double* radial,
             double* result, unsigned* groups, bool gate) {
  const Pair bra_pair{descriptors[0], descriptors[1], descriptors[2], descriptors[3]};
  const Pair ket_pair{descriptors[4], descriptors[5], descriptors[6], descriptors[7]};
  MdMaskedPlan<18> bra{}, ket{};
  if (!md_masked_pair(bra, bra_pair, nbf, offset, bounds, density, beta, unrestricted) ||
      !md_masked_pair(ket, ket_pair, nbf, offset, bounds, density, beta, unrestricted)) return -1;
  md_masked_cutoffs(bra, ket, screening);
  md_masked_cutoffs(ket, bra, screening);
  groups[0] = bra.groups;
  groups[1] = reverse ? ket.groups : 0;
  const unsigned bra_components = bra_pair.first_count * bra_pair.second_count;
  const unsigned ket_components = ket_pair.first_count * ket_pair.second_count;
  if (gate && !md_masked_work_admitted<10, KetHermites>(
          groups[0] + groups[1], bra_components, ket_components, KetHermites == 4 ? 4 : 8))
    return 0;
  std::array<double, 18> bra_values{}, ket_values{};
  for (unsigned bra_index = 0; bra_index < bra_primitives; ++bra_index) {
    const double* bra_e = bra_transform + bra_index * bra_components * 10;
    std::array<double, 190> bra_suffix{};
    for (unsigned derivative = 0; derivative < 10; ++derivative)
      md_masked_suffix<10>(bra, bra_e, derivative, bra_suffix.data());
    for (unsigned cutoff = 0; cutoff <= bra.count; ++cutoff)
      for (unsigned derivative = 0; derivative < 10; ++derivative)
        if (md_masked_tail<10>(bra, bra_e, derivative, cutoff) != bra_suffix[cutoff * 10 + derivative])
          return -7;
    for (unsigned ket_index = 0; ket_index < ket_primitives; ++ket_index) {
      const double* ket_e = ket_transform + ket_index * ket_components * KetHermites;
      const double* product = radial + (bra_index * ket_primitives + ket_index) * 10 * KetHermites;
      std::array<double, 19 * KetHermites> ket_suffix{};
      std::array<double, 180> bra_projected{};
      std::array<double, 18 * KetHermites> ket_projected{};
      for (unsigned derivative = 0; derivative < KetHermites; ++derivative)
        md_masked_suffix<KetHermites>(ket, ket_e, derivative, ket_suffix.data());
      for (unsigned cutoff = 0; cutoff <= ket.count; ++cutoff)
        for (unsigned derivative = 0; derivative < KetHermites; ++derivative)
          if (md_masked_tail<KetHermites>(ket, ket_e, derivative, cutoff) !=
              ket_suffix[cutoff * KetHermites + derivative]) return -8;
      for (unsigned element = 0; element < bra.groups * 10; ++element)
        bra_projected[element] = md_masked_project<10, KetHermites>(
            bra, ket_suffix.data(), element, Radial<KetHermites, false>{product});
      for (unsigned index = 0; index < bra.count; ++index)
        bra_values[index] += md_masked_output<10>(bra, bra_e, bra_projected.data(), index);
      if (reverse) {
        for (unsigned element = 0; element < ket.groups * KetHermites; ++element)
          ket_projected[element] = md_masked_project<KetHermites, 10>(
              ket, bra_suffix.data(), element, Radial<KetHermites, true>{product});
        for (unsigned index = 0; index < ket.count; ++index)
          ket_values[index] += md_masked_output<KetHermites>(ket, ket_e, ket_projected.data(), index);
      }
    }
  }
  for (unsigned index = 0; index < bra.count; ++index)
    result[offset + bra.entries[index].first * nbf + bra.entries[index].second] += bra_values[index];
  if (reverse)
    for (unsigned index = 0; index < ket.count; ++index)
      result[offset + ket.entries[index].first * nbf + ket.entries[index].second] += ket_values[index];
  return 1;
}

extern "C" int run_masked(unsigned ket_hermites, const unsigned* descriptors,
                          unsigned nbf, unsigned offset, bool reverse, bool unrestricted,
                          double screening, const double* bounds, const double* density,
                          const double* beta, unsigned bra_primitives, unsigned ket_primitives,
                          const double* bra_transform, const double* ket_transform,
                          const double* radial, double* result, unsigned* groups, bool gate) {
  if (ket_hermites == 4)
    return contract<4>(descriptors, nbf, offset, reverse, unrestricted, screening, bounds,
                       density, beta, bra_primitives, ket_primitives, bra_transform,
                       ket_transform, radial, result, groups, gate);
  return contract<10>(descriptors, nbf, offset, reverse, unrestricted, screening, bounds,
                      density, beta, bra_primitives, ket_primitives, bra_transform,
                      ket_transform, radial, result, groups, gate);
}

extern "C" bool check_cutoff(double output_bound, const double* input_bounds,
                             unsigned count, double screening) {
  MdMaskedPlan<18> output{}, input{};
  output.count = 1;
  output.entries[0] = {0, 0, 0, output_bound, 0.0};
  input.count = count;
  for (unsigned index = 0; index < count; ++index)
    input.entries[index] = {index, 0, 0, input_bounds[index], 0.0};
  if (!md_masked_sort(output) || !md_masked_sort(input)) return false;
  md_masked_cutoffs(output, input, screening);
  const unsigned cutoff = output.groups ? output.cutoff[output.group[0]] : count;
  for (unsigned index = 0; index < count; ++index)
    if ((index >= cutoff) != !(output_bound * input.entries[index].bound < screening)) return false;
  return true;
}

extern "C" bool cached_roundtrip(const unsigned* descriptor, const double* bounds,
                                  const double* density, const double* beta, bool unrestricted,
                                  const double* transform) {
  const Pair pair{descriptor[0],descriptor[1],descriptor[2],descriptor[3]};
  MdMaskedResidentPair resident;
  md_masked_prepare_resident(resident,pair,20,400,bounds);
  MdMaskedPlan<18> direct, cached;
  const bool valid=md_masked_pair(direct,pair,20,400,bounds,density,beta,unrestricted);
  if(resident.valid!=valid) return false;
  if(!valid) return true;
  if(!md_masked_cached_pair(cached,resident,pair,20,400,bounds)) return false;
  md_masked_weights(cached,20,400,density,beta,unrestricted);
  if(direct.count!=cached.count) return false;
  for(unsigned index=0;index<direct.count;++index) {
    const auto first=direct.entries[index], second=cached.entries[index];
    if(first.component!=second.component || first.first!=second.first ||
       first.second!=second.second || first.bound!=second.bound || first.weight!=second.weight)
      return false;
  }
  std::array<double,190> first{}, second{};
  for(unsigned derivative=0;derivative<10;++derivative) {
    md_masked_suffix<10>(direct,transform,derivative,first.data());
    md_masked_suffix<10>(cached,transform,derivative,second.data());
  }
  return first==second;
}
"""


@pytest.fixture(scope="module")
def masked_library(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Compile generated production math, not a Python reimplementation."""
    compiler = shutil.which("c++")
    launcher = shutil.which("sccache") or shutil.which("ccache")
    if not compiler or not launcher:
        pytest.skip("masked-Hermite host gates require C++ and a compiler cache")
    subprocess.run([launcher, "--version"], check=True, capture_output=True, timeout=30)
    directory = tmp_path_factory.mktemp("md-masked")
    (directory / "generated_md_j_masked.cuh").write_text(emit_md_j_masked_header())
    source = directory / "probe.cpp"
    source.write_text(_DRIVER)
    library = directory / "probe.so"
    subprocess.run(
        [
            launcher,
            compiler,
            "-std=c++20",
            "-O2",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        timeout=90,
    )
    loaded = ct.CDLL(str(library))
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    unsigneds = np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS")
    loaded.run_masked.argtypes = [
        ct.c_uint,
        unsigneds,
        ct.c_uint,
        ct.c_uint,
        ct.c_bool,
        ct.c_bool,
        ct.c_double,
        doubles,
        doubles,
        doubles,
        ct.c_uint,
        ct.c_uint,
        doubles,
        doubles,
        doubles,
        doubles,
        unsigneds,
        ct.c_bool,
    ]
    loaded.run_masked.restype = ct.c_int
    loaded.check_cutoff.argtypes = [ct.c_double, doubles, ct.c_uint, ct.c_double]
    loaded.check_cutoff.restype = ct.c_bool
    loaded.cached_roundtrip.argtypes = [
        unsigneds,
        doubles,
        doubles,
        doubles,
        ct.c_bool,
        doubles,
    ]
    loaded.cached_roundtrip.restype = ct.c_bool
    return loaded


def _canonical_entries(pair: tuple[int, ...]) -> list[tuple[int, int, int]]:
    first, second, first_count, second_count = pair
    return [
        (
            component,
            first + component // second_count,
            second + component % second_count,
        )
        for component in range(first_count * second_count)
        if first + component // second_count >= second + component % second_count
    ]


def _orbit_oracle(
    bra_pair: tuple[int, ...],
    ket_pair: tuple[int, ...],
    integrals: np.ndarray,
    bounds: np.ndarray,
    total: np.ndarray,
    screening: float,
    same_pair: bool,
) -> np.ndarray:
    """Enumerate unique ERI orbits without sorting, suffixes or Hermite grouping."""
    result = np.zeros_like(total)
    nbf = total.shape[0]
    for bra_component, first, second in _canonical_entries(bra_pair):
        for ket_component, third, fourth in _canonical_entries(ket_pair):
            if same_pair and first * nbf + second < third * nbf + fourth:
                continue
            orbit = {
                (bra_first, bra_second, ket_first, ket_second)
                for bra_first, bra_second in ((first, second), (second, first))
                for ket_first, ket_second in ((third, fourth), (fourth, third))
            }
            orbit |= {
                (third, fourth, first, second) for first, second, third, fourth in orbit
            }
            for out_first, out_second, in_first, in_second in orbit:
                if (
                    bounds[out_first, out_second] * bounds[in_first, in_second]
                    < screening
                ):
                    continue
                result[out_first, out_second] += (
                    integrals[bra_component, ket_component] * total[in_first, in_second]
                )
    return result


_PAIRS = [
    ((9, 4, 3, 3), (15, 1, 3, 1), 4, False),
    ((4, 4, 3, 3), (15, 1, 3, 1), 4, False),
    ((12, 0, 6, 1), (9, 4, 3, 3), 10, False),
    ((9, 4, 3, 3), (9, 4, 3, 3), 10, True),
    ((4, 4, 3, 3), (4, 4, 3, 3), 10, True),
]


@pytest.mark.parametrize("bra_pair,ket_pair,ket_hermites,same_pair", _PAIRS)
@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("screening", [0.0, 0.25, 2.0])
def test_masked_contraction_matches_independent_ao_orbits(
    masked_library: Any,
    bra_pair: tuple[int, ...],
    ket_pair: tuple[int, ...],
    ket_hermites: int,
    same_pair: bool,
    unrestricted: bool,
    screening: float,
) -> None:
    """Cover diagonal deduplication, both directions, UKS and primitive sums."""
    generator = np.random.default_rng(2600)
    nbf = 20
    bra_components, ket_components = np.prod(bra_pair[2:]), np.prod(ket_pair[2:])
    bra_transform = generator.normal(size=(2, bra_components, 10))
    ket_transform = (
        bra_transform.copy()
        if same_pair
        else generator.normal(size=(3, ket_components, ket_hermites))
    )
    radial = generator.normal(size=(2, len(ket_transform), 10, ket_hermites))
    if same_pair:
        radial = 0.5 * (radial + radial.transpose(1, 0, 3, 2))
    bounds = generator.choice([0.0, 0.125, 0.5, 1.0, 2.0], size=(3, nbf, nbf))
    density = generator.normal(size=bounds.shape)
    beta = generator.normal(size=bounds.shape)
    integrals = np.einsum("pat,pqtu,qbu->ab", bra_transform, radial, ket_transform)
    total = density[1] + (beta[1] if unrestricted else 0.0)
    expected = np.zeros_like(density)
    expected[1] = _orbit_oracle(
        bra_pair, ket_pair, integrals, bounds[1], total, screening, same_pair
    )
    actual = np.zeros_like(density)
    groups = np.zeros(2, dtype=np.uint32)
    status = masked_library.run_masked(
        ket_hermites,
        np.array(bra_pair + ket_pair, dtype=np.uint32),
        nbf,
        nbf * nbf,
        not same_pair,
        unrestricted,
        screening,
        bounds,
        density,
        beta,
        len(bra_transform),
        len(ket_transform),
        bra_transform,
        ket_transform,
        np.ascontiguousarray(radial),
        actual,
        groups,
        False,
    )
    assert status == 1
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=3e-13)
    assert not np.any(actual[[0, 2]])


@pytest.mark.parametrize(
    "screening", [0.0, np.nextafter(0.25, 0.0), 0.25, np.nextafter(0.25, 1.0), 1e200]
)
def test_suffix_uses_exact_original_fp64_comparison(
    masked_library: Any, screening: float
) -> None:
    """Ties, zeros, overflow and underflow must not use a divided cutoff."""
    bounds = np.array([1e200, 0.5, 1e-200, 0.0, 0.5, 2.0, np.finfo(float).max])
    for output in bounds:
        assert masked_library.check_cutoff(output, bounds, len(bounds), screening)


@pytest.mark.parametrize("invalid", [-1.0, np.inf, -np.inf, np.nan])
def test_invalid_bounds_reject_the_suffix_proof(
    masked_library: Any, invalid: float
) -> None:
    bounds = np.array([0.5, invalid, 1.0])
    assert not masked_library.check_cutoff(1.0, bounds, len(bounds), 0.25)
    assert not masked_library.check_cutoff(invalid, np.ones(3), 3, 0.25)


def test_complex_masks_fall_back_before_publication(masked_library: Any) -> None:
    """The work gate owns no output and cannot leak a partially contracted J."""
    bra_pair, ket_pair, ket_hermites, _ = _PAIRS[0]
    bounds = np.full((20, 20), 0.125)
    entries = _canonical_entries(bra_pair) + _canonical_entries(ket_pair)
    for component, first, second in entries:
        bounds[first, second] = 2.0 ** (component % 6 - 2)
        bounds[second, first] = 2.0 ** (component % 6 - 3)
    density = np.ones_like(bounds)
    actual = np.zeros_like(bounds)
    groups = np.zeros(2, dtype=np.uint32)
    status = masked_library.run_masked(
        ket_hermites,
        np.array(bra_pair + ket_pair, dtype=np.uint32),
        20,
        0,
        True,
        False,
        1.0,
        bounds,
        density,
        density,
        1,
        1,
        np.ones((9, 10)),
        np.ones((3, 4)),
        np.ones((10, 4)),
        actual,
        groups,
        True,
    )
    assert status == 0
    assert groups.sum() > 4
    assert not np.any(actual)


@pytest.mark.parametrize("pair", [(8, 1, 3, 3), (5, 5, 3, 3), (3, 2, 1, 1)])
@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("invalid", [None, -1.0, float("inf"), float("nan")])
def test_resident_cache_preserves_directional_plan(
    masked_library: Any,
    pair: tuple[int, ...],
    unrestricted: bool,
    invalid: float | None,
) -> None:
    """Stable permutations retain directed bounds/UKS weights and invalid fallback."""
    random = np.random.default_rng(481)
    bounds = random.choice([0.0, 0.25, 1.0, 2.0], size=(2, 20, 20))
    if invalid is not None:
        bounds[1, pair[0], pair[1]] = invalid
    density = random.normal(size=(2, 20, 20))
    beta = random.normal(size=(2, 20, 20))
    transform = random.normal(size=(pair[2] * pair[3], 10))
    assert masked_library.cached_roundtrip(
        np.array(pair, dtype=np.uint32), bounds, density, beta, unrestricted, transform
    )


def test_masked_generation_is_deterministic_and_runtime_independent(
    tmp_path: Path,
) -> None:
    """The standalone build generator may not need an installed native runtime."""
    assert emit_md_j_masked_header() == emit_md_j_masked_header()
    assert emit_md_j_masked_header().startswith("#pragma once\n")
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "generated_md_j_masked.cuh"
    command = [
        "python",
        str(root / "tools/generate_md_j_masked.py"),
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, timeout=30)
    modified = output.stat().st_mtime_ns
    subprocess.run(command, check=True, timeout=30)
    assert output.read_text() == emit_md_j_masked_header()
    assert output.stat().st_mtime_ns == modified
