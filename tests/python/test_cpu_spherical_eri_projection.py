"""Bounds, fallback ownership and real-source coverage for shell-local projection."""

import ctypes as ct
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "src/integrals/s_integrals.cpp").read_text()


@pytest.fixture(scope="module")
def spherical_values(
    tmp_path_factory: pytest.TempPathFactory,
    required_native_cxx: Any,
    policy_library: Any,
) -> Any:
    """Link the actual loaded producer, not an alternate projection algorithm."""
    directory = tmp_path_factory.mktemp("spherical-values")
    core = Path(policy_library._name).resolve()
    library = required_native_cxx.build_shared(
        [ROOT / "tests/native/host_spherical_eri_probe.cpp"],
        directory / "values.so",
        compile_args=[
            "-std=c++20",
            "-O2",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
        ],
        link_args=[str(core), "-Wl,-rpath," + str(core.parent)],
    )
    loaded = ct.CDLL(str(library))
    loaded.spherical_eri_error.restype = ct.c_char_p
    loaded.spherical_eri_values.argtypes = [
        ct.c_uint,
        ct.POINTER(ct.c_double),
        ct.c_size_t,
    ]
    loaded.spherical_eri_values.restype = ct.c_int
    return loaded


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("reversed_order", [False, True])
def test_f_and_mixed_public_tensors_against_independent_libcint(
    spherical_values: Any, mixed: bool, reversed_order: bool
) -> None:
    """Compare every public element, including distinct f centers and contractions."""
    gto = pytest.importorskip("pyscf.gto")
    centers = [(0.1, -0.3, 0.2), (-0.7, 0.8, 0.5), (0.9, 0.2, -0.6), (-0.4, -0.6, 0.7)]
    records = (
        (
            [[0, [0.9, 0.7], [0.25, 0.3]]],
            [[1, [0.65, 0.8], [0.18, -0.15]]],
            [[2, [0.4, 0.9], [0.12, -0.1]]],
            [[3, [0.55, 1.0], [0.16, -0.2]]],
        )
        if mixed
        else tuple([[3, [0.35 + 0.1 * atom, 1.0]]] for atom in range(4))
    )
    atoms = [(f"He{index}", center) for index, center in enumerate(centers)]
    if reversed_order:
        atoms.reverse()
    molecule = gto.M(
        atom=atoms,
        basis={f"He{index}": record for index, record in enumerate(records)},
        unit="Bohr",
        cart=False,
        charge=0,
        spin=0,
        verbose=0,
    )
    expected = molecule.intor("int2e_sph")
    actual = np.full(expected.shape, np.nan)
    status = spherical_values.spherical_eri_values(
        int(mixed) + 2 * int(reversed_order),
        actual.ctypes.data_as(ct.POINTER(ct.c_double)),
        actual.size,
    )
    assert status == 0, spherical_values.spherical_eri_error().decode()
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=2e-12)
    np.testing.assert_array_equal(actual, actual.transpose(1, 0, 2, 3))
    np.testing.assert_array_equal(actual, actual.transpose(2, 3, 0, 1))


def test_native_helper_reports_missing_compiler_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: None if name in ("sccache", "ccache") else "/unused/c++",
    )
    with pytest.raises(pytest.skip.Exception, match="compiler cache is required"):
        test_actual_native_helpers_cover_blocks_and_public_orbits(tmp_path)


def test_generic_transform_requires_explicit_producer_symmetry() -> None:
    generic = SOURCE.split("std::vector<double> transform_eri(", 1)[1].split(
        "std::vector<double> transform_three_center(", 1
    )[0]
    assert "bool eightfold_symmetric = false" in generic
    assert "if (eightfold_symmetric)" in generic
    assert (
        "else\n            transformed[eri_index(p, q, r, s, target_count)] = value"
        in generic
    )
    adapter = SOURCE.split("IntegralData transform_integrals(", 1)[1].split(
        "IntegralData build_integrals(", 1
    )[0]
    assert "target_aos, true" not in adapter
    range_values = SOURCE.split("std::vector<double> build_range_eri(", 1)[1].split(
        "std::array<double, 12> contract_weighted_eri_shell_derivative(", 1
    )[0]
    assert "build_spherical_value_eri_shell_quartets" not in range_values
    assert "spherical_expansions(system)" in range_values


def test_actual_native_helpers_cover_blocks_and_public_orbits(tmp_path: Path) -> None:
    """Compile actual schedule helpers without a library/generator dependency.

    Synthetic nonzero integer Cartesian orbits detect holes, wrong pair order,
    stale scratch and writes beyond each compact block. Identity expansions
    isolate schedule correctness from the independently tested basis algebra.
    Numerical spherical expansions are exercised by the linked native test.
    """
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    cache = shutil.which("sccache") or shutil.which("ccache")
    if cache is None:
        pytest.skip("a compiler cache is required for native source qualification")
    subprocess.run([cache, "--version"], check=True, capture_output=True, text=True)
    launcher = [cache]
    terms = SOURCE.split("struct GlobalExpansionTerm", 1)[1].split(
        "std::vector<GlobalAoExpansion> spherical_expansions(", 1
    )[0]
    components = SOURCE.split("std::size_t eri_index(", 1)[1].split(
        "std::size_t prepare_value_eri_components(", 1
    )[0]
    projection = SOURCE.split("using ValueEriCartesianBlock", 1)[1].split(
        "void build_spherical_value_eri_shell_quartets(", 1
    )[0]
    text = (
        "#include <algorithm>\n#include <array>\n#include <vector>\n#include <span>\n"
        "#include <cmath>\n#include <limits>\n#include <iostream>\n"
        "#include <stdexcept>\n"
        '#include "integrals/eri_symmetry.hpp"\n'
        "using generativeqc::integrals::eri_permutations;\n"
        + "struct GlobalExpansionTerm"
        + terms
        + "std::size_t eri_index("
        + components
        + "using ValueEriCartesianBlock"
        + projection
        + r"""
void require(bool condition) {
  if (!condition) throw std::runtime_error("native projection coverage failure");
}
// Independent orbit identity: first sort each AO pair, then the two pair IDs.
double orbit_value(std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
  if (i < j) std::swap(i, j);
  if (k < l) std::swap(k, l);
  auto a = i * (i + 1) / 2 + j;
  auto b = k * (k + 1) / 2 + l;
  if (a < b) std::swap(a, b);
  return static_cast<double>(1 + a * (a + 1) / 2 + b);
}
int main() {
  for (std::size_t count = 1; count <= 9; ++count) {
    std::vector<unsigned> visits(count * count * count * count);
    std::size_t unique = 0;
    generativeqc::integrals::for_each_unique_eri_quartet(count, [&](const auto& indices) {
      ++unique;
      generativeqc::integrals::for_each_eri_permutation(indices, [&](const auto& incidence) {
        auto slot = ((incidence[0] * count + incidence[1]) * count + incidence[2]) * count +
                    incidence[3];
        require(++visits[slot] == 1);
      });
    });
    const auto pairs = count * (count + 1) / 2;
    require(unique == pairs * (pairs + 1) / 2);
    for (auto visited : visits) require(visited == 1);
  }
  std::size_t blocks = 0, local_values = 0, public_values = 0;
  for (std::size_t nshell = 1; nshell <= 4; ++nshell) {
    std::size_t configurations = 1;
    for (std::size_t s = 0; s < nshell; ++s) configurations *= 3;
    for (std::size_t layout = 0; layout < configurations; ++layout) {
      auto code = layout;
      std::vector<std::size_t> offsets{0};
      for (std::size_t s = 0; s < nshell; ++s) {
        offsets.push_back(offsets.back() + std::array<std::size_t, 3>{1, 3, 6}[code % 3]);
        code /= 3;
      }
      const auto n = offsets.back();
      std::vector<GlobalAoExpansion> target_aos(n);
      for (std::size_t i = 0; i < n; ++i) target_aos[i].push_back({i, 1.0});
      std::vector<double> output(n * n * n * n, std::numeric_limits<double>::quiet_NaN());
      ValueEriCartesianBlock block;
      ValueEriComponents components;
      for (std::size_t si = 0; si < nshell; ++si)
        for (std::size_t sj = 0; sj <= si; ++sj)
          for (std::size_t sk = 0; sk <= si; ++sk)
            for (std::size_t sl = 0; sl <= sk; ++sl) {
              if (si == sk && sj < sl) continue;
              const std::array<std::size_t, 4> shells{si, sj, sk, sl};
              std::array<std::size_t, 4> begins, extents;
              for (unsigned slot = 0; slot < 4; ++slot) {
                begins[slot] = offsets[shells[slot]];
                extents[slot] = offsets[shells[slot] + 1] - begins[slot];
              }
              std::size_t count = 0;
              for (auto i = offsets[si]; i < offsets[si + 1]; ++i)
                for (auto j = offsets[sj]; j < offsets[sj + 1]; ++j)
                  for (auto k = offsets[sk]; k < offsets[sk + 1]; ++k)
                    for (auto l = offsets[sl]; l < offsets[sl + 1]; ++l) {
                      if (si == sj && i < j) continue;
                      if (sk == sl && k < l) continue;
                      if (si == sk && sj == sl &&
                          i * (i + 1) / 2 + j < k * (k + 1) / 2 + l) continue;
                      require(count < components.size());
                      components[count++] = {{i, j, k, l}, 0, 1.0, orbit_value(i, j, k, l)};
                    }
              block.fill(-17.0);
              assemble_value_eri_cartesian_block(components, count, begins, extents, block);
              std::size_t local = 0;
              for (auto i = offsets[si]; i < offsets[si + 1]; ++i)
                for (auto j = offsets[sj]; j < offsets[sj + 1]; ++j)
                  for (auto k = offsets[sk]; k < offsets[sk + 1]; ++k)
                    for (auto l = offsets[sl]; l < offsets[sl + 1]; ++l)
                      require(block[local++] == orbit_value(i, j, k, l));
              for (auto i = local; i < block.size(); ++i) require(block[i] == -17.0);
              project_value_eri_shell_quartet(shells, offsets, target_aos, begins, extents,
                                             block, output);
              ++blocks;
              local_values += local;
            }
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          for (std::size_t k = 0; k < n; ++k)
            for (std::size_t l = 0; l < n; ++l)
              require(output[eri_index(i, j, k, l, n)] == orbit_value(i, j, k, l));
      public_values += output.size();
    }
  }
  std::cout << "blocks=" << blocks << " local_values=" << local_values
            << " public_values=" << public_values << '\n';
}
"""
    )
    source = tmp_path / "projection_coverage.cpp"
    binary = tmp_path / "projection_coverage"
    source.write_text(text)
    subprocess.run(
        [
            *launcher,
            compiler,
            "-std=c++20",
            "-O1",
            "-I" + str(ROOT / "src"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert "blocks=5079 local_values=1167644 public_values=4650300" in result.stdout
