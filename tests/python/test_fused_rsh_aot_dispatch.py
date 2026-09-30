"""Exercise the production fused radial dispatcher without a CUDA device."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "src/scf/cuda/direct_bounded_contraction.cuh"


def test_fused_dispatch_retains_registered_and_generic_radial_identities() -> None:
    source = HEADER.read_text(encoding="utf-8")
    assert "template <bool Unrestricted, int OmegaMilli>" in source
    assert (
        "contract_bounded_direct_rsh_force_subtile_impl<Unrestricted, 300>(" in source
    )
    assert "contract_bounded_direct_rsh_force_subtile_impl<Unrestricted, 0>(" in source
    assert "if (omega == 0.3)" in source
    assert (
        "OmegaMilli == 0 ? runtime_omega : static_cast<double>(OmegaMilli) / 1000.0"
        in source
    )


@pytest.fixture(scope="module")
def dispatch_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for the exact-source dispatch probe")
    source = HEADER.read_text(encoding="utf-8")
    begin = source.index("template <bool Unrestricted, int OmegaMilli>")
    end = source.index(
        "template <bool Unrestricted>\n"
        "__device__ inline __noinline__ void contract_bounded_direct_force_subtile(",
        begin,
    )
    directory = tmp_path_factory.mktemp("fused-rsh-dispatch")
    cpp = directory / "probe.cpp"
    executable = directory / "probe"
    cpp.write_text(
        r"""
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#define __device__
#define __noinline__
#define __forceinline__ inline
struct DeviceBatch { int identity; };
struct ActiveShellQuartetTile { int identity; };
unsigned calls, expected_order;
bool expected_spin;
double expected_omega;
std::uint32_t count = 7;
ActiveShellQuartetTile task{19};
double bounds[1], density[1], output[1];
std::uint8_t active = 1;
template <bool Unrestricted, unsigned Order>
void contract_two_electron_force_quartet_subtile_rsh_scaled(
    DeviceBatch batch, const std::uint32_t* queue_count,
    const ActiveShellQuartetTile* queued, double screening,
    const double* schwarz, const double* weights, const std::uint8_t* enabled,
    double* forces, double coulomb, double short_exchange, double long_exchange,
    double omega, std::size_t subtile, unsigned lane) {
  ++calls;
  assert(Unrestricted == expected_spin && Order == expected_order);
  assert(batch.identity == 11 && queue_count == &count && queued == &task);
  assert(screening == 1e-14 && schwarz == bounds && weights == density);
  assert(enabled == &active && forces == output);
  assert(coulomb == 0.9 && short_exchange == -0.15 && long_exchange == -0.5);
  assert(omega == expected_omega && subtile == 23 && lane == 5);
}
"""
        + source[begin:end]
        + r"""
int main(int argc, char** argv) {
  assert(argc == 3);
  expected_spin = std::atoi(argv[1]) != 0;
  expected_order = static_cast<unsigned>(std::atoi(argv[2]));
  // Values adjacent to the registered constant must never be rounded into it.
  for (double omega : {0.3, std::nextafter(0.3, 0.0),
                       std::nextafter(0.3, 1.0), 0.2, 0.8}) {
    calls = 0;
    expected_omega = omega;
    if (expected_spin)
      contract_bounded_direct_rsh_force_subtile<true>(
          DeviceBatch{11}, expected_order, &count, &task, 1e-14, bounds, density,
          &active, output, 0.9, -0.15, -0.5, omega, 23, 5);
    else
      contract_bounded_direct_rsh_force_subtile<false>(
          DeviceBatch{11}, expected_order, &count, &task, 1e-14, bounds, density,
          &active, output, 0.9, -0.15, -0.5, omega, 23, 5);
    assert(calls == (expected_order <= 12 ? 1U : 0U));
  }
}
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [compiler, "-std=c++17", "-O2", str(cpp), "-o", str(executable)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return executable


@pytest.mark.parametrize("spin", (0, 1))
@pytest.mark.parametrize("angular_order", range(14))
def test_fused_dispatch_forwards_exact_radial_and_source_arguments(
    dispatch_probe: Path, spin: int, angular_order: int
) -> None:
    completed = subprocess.run(
        [str(dispatch_probe), str(spin), str(angular_order)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
