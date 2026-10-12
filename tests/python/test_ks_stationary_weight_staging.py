"""Run the actual KS staging block against deterministic per-spin launch stubs."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _stage_source() -> str:
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text(encoding="utf-8")
    begin = source.index("    bool staged_stationary_weights = false;")
    end = source.index("    KsPhysicalState physical;", begin)
    from test_stationary_task_work_budget import _block

    stage = _block(source, "  void stage_stationary_weights(")
    stage = stage.replace(
        "void stage_stationary_weights()",
        "const auto stage_stationary_weights = [&]()",
        1,
    )
    return stage + ";\n" + source[begin:end]


@pytest.fixture(scope="module")
def staging_program(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Path:
    directory = tmp_path_factory.mktemp("stationary-weights")
    source = directory / "probe.cpp"
    binary = directory / "probe"
    source.write_text(
        r"""
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

static unsigned builds = 0, sums = 0;
static unsigned expected_spins = 0;
int cudaGetLastError() { return 0; }
void check(int status) { assert(status == 0); }

// The production generated launcher writes separate [spin, AO, AO] blocks.
// Deliberately different alpha/beta entries make an omitted beta block visible.
void build(unsigned spins, int n, const std::int32_t* occupied, double* output) {
  ++builds;
  assert(spins == expected_spins);
  for (unsigned spin = 0; spin < spins; ++spin)
    for (int i = 0; i < n * n; ++i)
      output[spin * n * n + i] = occupied[spin] * (10.0 + 3.0 * spin + i);
}
void launch_sum_uhf_spin_matrices_kernel(
    unsigned grid, unsigned block, std::size_t shared, int stream, int batch, int n,
    const double* input, const std::uint8_t* active, double* output) {
  assert(grid * block >= unsigned(n * n) && shared == 0 && stream == 7);
  assert(batch == 1 && *active == 1);
  ++sums;
  for (int i = 0; i < n * n; ++i) output[i] = input[i] + input[n * n + i];
}
int main(int argc, char** argv) {
  assert(argc == 4);
  const unsigned spins = static_cast<unsigned>(std::atoi(argv[1]));
  const int beta = std::atoi(argv[2]);
  const bool device_validation = std::atoi(argv[3]) != 0;
  expected_spins = spins;
  const std::size_t n = 3, matrix = n * n, elements = spins * matrix;
  const int stream = 7;
  const std::int32_t occupied[2] = {2, beta};
  const std::uint8_t final_enabled[1] = {1};
  const double coefficients[1] = {1.0}, eigenvalues[1] = {-0.5};
  const double *final_coefficients = coefficients, *final_eigenvalues = eigenvalues;
  std::vector<double> density_storage(elements), first(elements, -99), second(elements, -88);
  for (std::size_t i = 0; i < matrix; ++i) {
    density_storage[i] = 2.0 + i;
    if (spins == 2) density_storage[matrix + i] = beta * (5.0 + i);
  }
  auto* density = density_storage.data();
  auto* tmp1 = first.data();
  auto* tmp2 = second.data();
  bool final_stationary_weights_ready = false;
  const bool compute_weighted_density = true;
  // Staging now calls the common structured-density owner. Keep this fixture
  // focused on its per-spin weights and lease, with distinct argument sentinels.
  const auto build_density = [&](const double* frame, const double* energies,
                                 const std::uint8_t* active, double* output) {
    assert(frame == final_coefficients && energies == final_eigenvalues);
    assert(active == final_enabled && *active == 1 && output == tmp1);
    build(spins, static_cast<int>(n), occupied, output);
  };
  for (unsigned attempt = 0; attempt < 2; ++attempt) {
"""
        + _stage_source()
        + r"""
    if (device_validation && compute_weighted_density && !final_stationary_weights_ready) {
      stage_weights();
    }
    assert(staged_stationary_weights == (attempt == 0));
    final_stationary_weights_ready = true;
    for (std::size_t i = 0; i < matrix; ++i) {
      const double expected_w = 2.0 * (10.0 + i) + (spins == 2 ? beta * (13.0 + i) : 0);
      const double expected_d = 2.0 + i + (spins == 2 ? beta * (5.0 + i) : 0);
      assert(std::abs(tmp1[i] - expected_w) < 1e-12);
      assert(std::abs((spins == 1 ? density : tmp2)[i] - expected_d) < 1e-12);
    }
    assert(builds == 1);
    assert(sums == (spins == 2 ? 2U : 0U));
  }
}
""",
        encoding="utf-8",
    )
    return native_cxx.build_executable(
        [source], binary, compile_args=["-std=c++17", "-O0"], compile_timeout=60
    )


@pytest.mark.parametrize("spins,beta", [(1, 0), (2, 0), (2, 1), (2, 2)])
@pytest.mark.parametrize("device_validation", [False, True])
def test_stationary_staging_keeps_both_spin_weights_and_reuses_them(
    staging_program: Path, spins: int, beta: int, device_validation: bool
) -> None:
    completed = subprocess.run(
        [str(staging_program), str(spins), str(beta), str(int(device_validation))],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_stationary_weight_lease_is_published_only_after_validation(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Execute the production acceptance/publication tail with fault controls."""
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text(encoding="utf-8")
    begin = source.index("  VerifiedKsFinalState read_final(")
    begin = source.index("    VerifiedKsFinalState verified;", begin)
    end = source.index("    return verified;", begin) + len("    return verified;")
    program = tmp_path / "publication.cpp"
    program.write_text(
        r"""
#include <cassert>
#include <stdexcept>
#include <string>

struct VerifiedKsFinalState { int diagnostic = 0; };
bool final_state_ready = true, final_stationary_weights_ready = false;
int final_validation_diagnostic = -1, failure = 0, stages = 0, validations = 0;
int stream = 7;
struct { unsigned synchronizations = 0; } movement;
struct { int identity = 1; } current;
struct OneElectron { int overlap = 2, hcore = 3; };
struct Provider { OneElectron one_electron() { return {}; } } provider;
int physical = 4, candidate = 5, limits = 6, operations = 7;
int cudaStreamSynchronize(int value) { assert(value == stream); return failure == 4; }
void check(int status) { if(status) throw std::runtime_error("drain"); }
bool validate_ks_final_state(int, int, int, int, int, int, bool,
                            VerifiedKsFinalState& verified, std::string& detail, int*) {
  ++validations;
  assert(!final_stationary_weights_ready && final_validation_diagnostic == -1);
  if(failure == 1) { detail = "rejected"; return false; }
  if(failure == 2) throw std::invalid_argument("invalid owner");
  verified.diagnostic = 42;
  return true;
}
VerifiedKsFinalState read(bool device_validation, bool compute_weighted_density) {
  std::string detail;
  bool staged_stationary_weights = compute_weighted_density && !device_validation;
  const auto stage_weights = [&] {
    assert(validations && !final_stationary_weights_ready);
    ++stages;
    if(failure == 3) throw std::runtime_error("stage");
    staged_stationary_weights = true;
  };
"""
        + source[begin:end]
        + r"""
}
int main() {
  for(bool device : {false, true}) {
    for(int injected : {1, 2, 3, 4, 0}) {
      if(!device && (injected == 3 || injected == 4)) continue;
      failure = injected;
      final_state_ready = true;
      final_stationary_weights_ready = false;
      final_validation_diagnostic = -1;
      stages = validations = 0;
      bool threw = false;
      try {
        const auto result = read(device, true);
        assert(result.diagnostic == 42);
      } catch(const std::exception&) { threw = true; }
      assert(threw == (failure != 0));
      assert(final_stationary_weights_ready == !threw);
      assert(final_validation_diagnostic == (threw ? -1 : 42));
      if(threw) assert(!final_state_ready);
      assert(validations == 1);
      assert(stages == (device && (failure == 0 || failure >= 3) ? 1 : 0));
    }
    final_stationary_weights_ready = false;
    final_validation_diagnostic = -1;
    stages = validations = 0;
    const auto no_weights = read(device, false);
    assert(no_weights.diagnostic == 42 && !final_stationary_weights_ready);
    assert(final_validation_diagnostic == -1 && stages == 0);
  }
}
""",
        encoding="utf-8",
    )
    binary = native_cxx.build_executable(
        [program], tmp_path / "publication", compile_args=["-std=c++17", "-O0"]
    )
    subprocess.run([str(binary)], check=True, timeout=10)
