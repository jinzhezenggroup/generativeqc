"""Standalone C++20 CPU oracle for the common-reference RCCSD input adapter."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_canonical_reference_view_cpu_contract(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++20 compiler")
    source = tmp_path / "reference_input.cpp"
    source.write_text(
        r"""
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "cc/reference_input.hpp"

using namespace generativeqc;

int main() {
  const std::array<double, 4> identity{1, 0, 0, 1};
  const std::array<double, 4> c{.8, -.6, .6, .8};
  std::array<double, 4> f{1.1, .2, .2, 2.3};
  const std::array<double, 2> energies{-1, .6};
  core::ElectronicReferenceView ref;
  ref.basis_functions = 2;
  ref.spin_channels = 1;
  ref.overlap = identity;
  ref.hcore = identity;
  ref.channels[0] = {1, c, energies, identity, f, {}};
  ref.energy = -1.234;

  const auto result = cc::prepare_canonical_reference_input(ref);
  if (result.basis_functions != 2 || result.occupied != 1 ||
      result.virtuals != 1 || result.reference_energy != -1.234 ||
      result.orbital_energies.data() != energies.data())
    return 1;

  // Independent four-index loop, not the adapter's two staged products.
  for (std::size_t p = 0; p < 2; ++p)
    for (std::size_t q = 0; q < 2; ++q) {
      double expected = 0.0;
      for (std::size_t mu = 0; mu < 2; ++mu)
        for (std::size_t nu = 0; nu < 2; ++nu)
          expected += c[mu * 2 + p] * f[mu * 2 + nu] * c[nu * 2 + q];
      if (std::abs(expected - result.fock_mo[p * 2 + q]) > 1e-14) return 2;
    }

  // A changed borrowed Fock must produce fresh data, not reuse an old result.
  f[0] += .5;
  const auto fresh = cc::prepare_canonical_reference_input(ref);
  if (std::abs(fresh.fock_mo[0] - result.fock_mo[0]) < .1) return 3;

  try {
    auto bad = ref; bad.spin_channels = 2;
    bad.channels[1] = bad.channels[0];
    (void)cc::prepare_canonical_reference_input(bad);
    return 4;
  } catch (const std::invalid_argument&) {}
  try {
    auto bad = ref; bad.channels[0].occupied = 0;
    (void)cc::prepare_canonical_reference_input(bad);
    return 5;
  } catch (const std::invalid_argument&) {}
  try {
    auto bad = ref; bad.channels[0].occupied = 2;
    (void)cc::prepare_canonical_reference_input(bad);
    return 6;
  } catch (const std::invalid_argument&) {}
  try {
    auto bad = ref; bad.basis_functions = std::numeric_limits<std::size_t>::max();
    (void)cc::prepare_canonical_reference_input(bad);
    return 7;
  } catch (const std::invalid_argument&) {}
  try {
    auto bad = ref; bad.energy = std::numeric_limits<double>::quiet_NaN();
    (void)cc::prepare_canonical_reference_input(bad);
    return 8;
  } catch (const std::invalid_argument&) {}
  try {
    const std::array<double, 4> nonfinite{
        1, 0, std::numeric_limits<double>::infinity(), 1};
    auto bad = ref; bad.channels[0].fock = nonfinite;
    (void)cc::prepare_canonical_reference_input(bad);
    return 9;
  } catch (const std::invalid_argument&) {}
  try {
    auto bad = ref; bad.channels[0].orbital_energies = {};
    (void)cc::prepare_canonical_reference_input(bad);
    return 10;
  } catch (const std::invalid_argument&) {}
}
"""
    )
    executable = tmp_path / "reference_input"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(ROOT / "src"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = subprocess.run(
        [str(executable)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, (
        result.stdout + result.stderr + f"exit status {result.returncode}"
    )
