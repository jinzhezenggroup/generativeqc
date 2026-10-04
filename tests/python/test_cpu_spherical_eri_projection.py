"""Bounds, fallback ownership and real-source coverage for shell-local projection."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "src/integrals/s_integrals.cpp").read_text()


def test_native_helper_reports_missing_ccache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        shutil, "which", lambda name: None if name == "ccache" else "/unused/c++"
    )
    with pytest.raises(pytest.skip.Exception, match="ccache is required"):
        test_actual_native_helpers_cover_blocks_and_public_orbits(tmp_path)


def test_shell_local_projection_is_value_only_spherical_spd() -> None:
    build = SOURCE.split("IntegralData build_integrals(", 1)[1].split(
        "std::vector<double> build_range_eri(", 1
    )[0]
    normalized = " ".join(build.split())
    assert "!include_derivatives && std::all_of" in normalized
    assert "shell.angular_momentum <= 2" in normalized
    assert "include_eri && shared_value_geometry && spherical_output" in normalized
    assert "system.basis_representation == GENERATIVEQC_BASIS_SPHERICAL" in build
    assert "else if (!shell_local_spherical)" in build
    assert "out.eri.assign(n4, 0.0)" in build
    assert "spherical_eri.assign(public_n4, 0.0)" in build
    assert "spherical.eri = std::move(spherical_eri)" in build
    assert "build_value_eri_shell_quartets(system, aos, out.eri)" in build
    # Conservative Cartesian admission survives; no allocation failure retry
    # may launch the old full tensor path after expensive source work.
    assert "include_derivatives ? sizeof(Jet) : sizeof(double)" in build
    assert "catch" not in build
    assert "production_eri_cartesian(" in build
    assert "transform_eri(out.eri.data(), out.nbf, target_aos)" in build


def test_generic_transform_and_range_paths_do_not_symmetrize_arbitrary_inputs() -> None:
    generic = SOURCE.split("std::vector<double> transform_eri(", 1)[1].split(
        "std::vector<double> transform_three_center(", 1
    )[0]
    assert "store_eri_symmetry" not in generic
    for index in ("p", "q", "r", "s"):
        assert f"{index} = 0; {index} < target_count; ++{index}" in generic
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
    cache = shutil.which("ccache")
    if cache is None:
        pytest.skip("ccache is required for native source qualification")
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
        "#include <algorithm>\n#include <array>\n#include <vector>\n"
        "#include <cmath>\n#include <limits>\n#include <iostream>\n"
        "#include <stdexcept>\n"
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
        [*launcher, compiler, "-std=c++20", "-O1", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert "blocks=5079 local_values=1167644 public_values=4650300" in result.stdout
