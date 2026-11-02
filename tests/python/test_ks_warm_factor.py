"""Execute the real KS warm-factor lifecycle with host-only CUDA test doubles.

These gates establish storage, provenance and fallback behavior, not GPU timing
or endpoint qualification. NumPy independently checks the density and RI-K
algebra; real-device DF E/force qualification remains a separate requirement.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _definition(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def warm_factor_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Path:
    """Compile production cache methods, replacing only CUDA and provider calls."""
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    methods = "\n".join(
        _definition(source, signature)
        for signature in (
            "  bool warm_factor_replay_enabled()",
            "  bool qualified_warm_factor()",
            "  void materialize_warm_density(",
            "  void restore_warm_seed()",
            "  void publish_warm_seed()",
            "  void invalidate_warm_orbitals()",
            "  void clear_warm_state()",
            "  void restore_held_fitted_factor()",
        )
    )
    harness = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
constexpr int cudaMemcpyDeviceToDevice=3;
unsigned copies=0, fail_copy=0, density_builds=0;
int cudaMemcpyAsync(void* target, const void* input, std::size_t bytes, int kind, void*) {
  assert(kind == cudaMemcpyDeviceToDevice);
  if (++copies == fail_copy) return 1;
  std::memcpy(target, input, bytes);
  return 0;
}
int cudaGetLastError() { return 0; }
void check(int status) { if (status) throw std::runtime_error("copy rejected"); }
namespace runtime::cuda_trace {
void trace_counter(const char*,std::uint64_t) {}
}
void launch_build_density_kernel(unsigned, unsigned, unsigned, void*, unsigned batch,
    std::size_t dimension, const std::int32_t* occupied, const double* coefficients,
    const std::uint8_t* enabled, double* target) {
  assert(batch == 1 && *enabled == 1);
  ++density_builds;
  for (std::size_t row=0; row<dimension; ++row)
    for (std::size_t column=0; column<dimension; ++column) {
      double value=0;
      for (std::int32_t orbital=0; orbital<*occupied; ++orbital)
        value += coefficients[row+orbital*dimension]*coefficients[column+orbital*dimension];
      target[row*dimension+column]=2*value;
    }
}
namespace scf {
struct PreparedCudaOccupiedFockBinding {
  int device_id=0;
  void* stream=reinterpret_cast<void*>(1);
  const void* source_identity=reinterpret_cast<void*>(2);
  std::size_t nbf=3;
  bool unrestricted=false;
  explicit operator bool() const { return source_identity && stream && nbf && device_id>=0; }
};
struct PreparedFockPlan { PreparedCudaOccupiedFockBinding binding; };
PreparedCudaOccupiedFockBinding prepared_cuda_occupied_fock_binding(const PreparedFockPlan& owner) {
  return owner.binding;
}
}
struct Owner {
  scf::PreparedFockPlan provider;
  scf::PreparedCudaOccupiedFockBinding occupied_fock_binding{}, warm_factor_binding{};
  bool warm_ready=false, warm_factor_storage=false, fitted_exchange=true;
  bool pending_fitted_occupied=true, final_fitted_projection_ready=true;
  bool warm_orbitals_ready=true, occupied_fitted_factor_ready=false;
  bool warm_orbitals_are_ao=true;
  bool is_pending=false, warm_energy_baseline=false, warm_incumbent_deferred=false;
  std::array<double, 3> warm_xc_totals{};
  unsigned spins=1;
  std::size_t n=3, matrix=9, elements=9, warm_factor_rank=0;
  std::array<std::size_t,2> occupations{1,1};
  std::uint64_t owner=42, solve_epoch=1, generation=7;
  std::uint64_t warm_factor_owner=0, warm_density_generation=0;
  std::uint64_t warm_factor_generation=0, warm_factor_epoch=0;
  double warm_energy=0, warm_energy_correction=0, range_warm_energy=0;
  struct { double energy=-1; } output;
  struct {
    bool mixed=false;
    bool any_lower_precision() const { return mixed; }
  } precision_schedule;
  struct {
    unsigned fitted_warm_factor_publications=0, fitted_warm_factor_restores=0;
    unsigned fitted_warm_factor_fallbacks=0;
    unsigned warm_orbital_frame_invalidations=0;
  } movement;
  double warm[9]{}, density[9]{}, proposal[9]{}, warm_orbitals[9]{}, tmp1[9]{}, effective[9]{};
  std::int32_t occupied[1]{1};
  std::uint8_t enabled=1, *final_enabled=&enabled;
  int device=0;
  void* stream=reinterpret_cast<void*>(1);
  void current_device() {}
  std::vector<double> download(const double* input) { return {input,input+elements}; }
"""
    harness += methods
    harness += r"""
};
struct CudaKsPlan {
  Owner* impl_;
  std::vector<double> warm_density();
};
"""
    harness += _definition(source, "std::vector<double> CudaKsPlan::warm_density()")
    harness += r"""
int main(int argc, char** argv) {
  assert(argc == 2);
  const std::string mode=argv[1];
  setenv("GENERATIVEQC_CUDA_KS_WARM_FACTOR", "1", 1);
  Owner state;
  state.proposal[0]=1/std::sqrt(2.0);
  state.proposal[1]=1/std::sqrt(6.0);
  state.proposal[2]=1/std::sqrt(3.0);
  launch_build_density_kernel(1,128,0,state.stream,1,state.n,state.occupied,state.proposal,
                              state.final_enabled,state.density);
  const auto original=state.download(state.density);
  if (mode.rfind("held-",0)==0) {
    std::copy_n(state.proposal,9,state.warm_orbitals);
    const auto factor=state.download(state.warm_orbitals);
    // The next eigensolve proposes a different determinant. The confirmation
    // deliberately keeps D, so that proposal must never become its K factor.
    std::fill_n(state.tmp1,9,123.0);
    std::fill_n(state.proposal,9,456.0);
    const auto rejected_proposal=state.download(state.proposal);
    if (mode == "held-dense") state.pending_fitted_occupied=false;
    if (mode == "held-spins") state.spins=2;
    if (mode == "held-no-frame") state.warm_orbitals_ready=false;
    if (mode == "held-orthonormal") state.warm_orbitals_are_ao=false;
    if (mode == "held-copy-failure") fail_copy=copies+1;
    bool failed=false;
    try { state.restore_held_fitted_factor(); }
    catch (const std::runtime_error&) { failed=true; }
    assert(failed == (mode == "held-copy-failure"));
    assert(state.download(state.density) == original);
    assert(state.download(state.proposal) == rejected_proposal);
    assert(state.occupied_fitted_factor_ready == (mode == "held-qualified"));
    if (state.occupied_fitted_factor_ready) {
      assert(state.download(state.tmp1) == factor);
      launch_build_density_kernel(1,128,0,state.stream,1,state.n,state.occupied,state.tmp1,
                                  state.final_enabled,state.effective);
      assert(state.download(state.effective) == original);
    }
    return 0;
  }
  if (mode == "default") unsetenv("GENERATIVEQC_CUDA_KS_WARM_FACTOR");
  if (mode == "disabled" || mode == "enable-replay")
    setenv("GENERATIVEQC_CUDA_KS_WARM_FACTOR", "0", 1);
  if (mode == "bad-option") setenv("GENERATIVEQC_CUDA_KS_WARM_FACTOR", "yes", 1);
  if (mode == "empty-option") setenv("GENERATIVEQC_CUDA_KS_WARM_FACTOR", "", 1);
  if (mode == "no-lease") state.final_fitted_projection_ready=false;
  if (mode == "noncanonical") state.pending_fitted_occupied=false;
  if (mode == "no-frame") state.warm_orbitals_ready=false;
  if (mode == "orthonormal-frame") state.warm_orbitals_are_ao=false;
  if (mode == "mixed-publication") state.precision_schedule.mixed=true;
  if (mode == "publish-failure") fail_copy=1;
  state.warm_energy=17;
  try { state.publish_warm_seed(); }
  catch (const std::runtime_error&) {
    assert(mode == "publish-failure" && !state.warm_ready && !state.warm_factor_storage);
    return 0;
  }
  assert(mode != "publish-failure");
  const bool factor_published =
      mode != "disabled" && mode != "enable-replay" && mode != "bad-option" &&
      mode != "empty-option" && mode != "no-lease" && mode != "noncanonical" &&
      mode != "no-frame" && mode != "orthonormal-frame" && mode != "mixed-publication";
  assert(state.warm_factor_storage == factor_published);
  assert(state.movement.fitted_warm_factor_publications == (factor_published ? 1U : 0U));
  if (!factor_published) assert(state.download(state.warm) == original);
  assert(state.warm_energy == 17);
  state.solve_epoch=2;
  const auto frozen=state.download(state.warm);
  if (mode == "owner") ++state.warm_factor_owner;
  if (mode == "generation") ++state.warm_factor_generation;
  if (mode == "zero-generation") state.warm_density_generation=state.warm_factor_generation=0;
  if (mode == "epoch") state.warm_factor_epoch=state.solve_epoch;
  if (mode == "zero-epoch") state.warm_factor_epoch=0;
  if (mode == "rank") ++state.warm_factor_rank;
  if (mode == "zero-rank") state.warm_factor_rank=0;
  if (mode == "spins") state.spins=2;
  if (mode == "mixed") state.precision_schedule.mixed=true;
  if (mode == "provider") state.provider.binding.source_identity=reinterpret_cast<void*>(3);
  if (mode == "device") ++state.provider.binding.device_id;
  if (mode == "stream") state.provider.binding.stream=reinterpret_cast<void*>(3);
  if (mode == "dimension") ++state.provider.binding.nbf;
  if (mode == "unrestricted") state.provider.binding.unrestricted=true;
  if (mode == "no-binding") state.provider.binding.source_identity=nullptr;
  if (mode == "disable-replay") setenv("GENERATIVEQC_CUDA_KS_WARM_FACTOR", "0", 1);
  if (mode == "default-replay" || mode == "enable-replay")
    unsetenv("GENERATIVEQC_CUDA_KS_WARM_FACTOR");
  const bool expected=mode == "qualified" || mode == "default" || mode == "default-replay" ||
                      mode == "frozen" || mode == "export";
  const unsigned repeats=mode == "frozen" ? 5 : 1;
  CudaKsPlan plan{&state};
  for (unsigned replay=0; replay<repeats; ++replay) {
    state.occupied_fitted_factor_ready=state.warm_orbitals_ready=false;
    state.restore_warm_seed();
    assert(state.download(state.density) == original);
    assert(state.occupied_fitted_factor_ready == expected);
    assert(state.warm_orbitals_ready == expected);
    assert(state.download(state.warm) == frozen);
    if (expected) {
      assert(state.download(state.tmp1) == frozen);
      assert(state.download(state.warm_orbitals) == frozen);
    }
    const auto before_proposal=state.download(state.proposal);
    const auto before_factor=state.download(state.tmp1);
    assert(plan.warm_density() == original);
    assert(state.download(state.proposal) == before_proposal);
    assert(state.download(state.tmp1) == before_factor);
    if (mode == "frozen") {
      std::fill_n(state.proposal,9,123.0);
      std::fill_n(state.warm_orbitals,9,456.0);
      state.generation+=3;
      ++state.solve_epoch;
    }
  }
  assert(state.movement.fitted_warm_factor_restores == (expected ? repeats : 0));
  assert(state.movement.fitted_warm_factor_fallbacks == (expected ? 0 : repeats));
  if (mode == "export") {
    state.is_pending=true;
    bool rejected=false;
    try { plan.warm_density(); } catch (const std::logic_error&) { rejected=true; }
    assert(rejected);
    state.is_pending=false;
  }
  std::cout.precision(17);
  std::cout << "[";
  for (std::size_t index=0; index<original.size(); ++index)
    std::cout << (index ? "," : "") << state.density[index];
  std::cout << "]\n";
  state.clear_warm_state();
  state.clear_warm_state();
  assert(!state.warm_ready && !state.warm_factor_storage && !state.qualified_warm_factor());
  assert(plan.warm_density().empty());
}
"""
    folder = tmp_path_factory.mktemp("ks-warm-factor")
    cpp, executable = folder / "probe.cpp", folder / "probe"
    cpp.write_text(harness)
    native_cxx.build_executable(
        [cpp], executable, compile_args=("-std=c++17", "-Wall", "-Werror")
    )
    return executable


@pytest.mark.parametrize(
    "mode",
    [
        "qualified",
        "default",
        "default-replay",
        "frozen",
        "export",
        "disabled",
        "bad-option",
        "empty-option",
        "disable-replay",
        "enable-replay",
        "no-lease",
        "noncanonical",
        "no-frame",
        "orthonormal-frame",
        "mixed-publication",
        "owner",
        "generation",
        "zero-generation",
        "epoch",
        "zero-epoch",
        "rank",
        "zero-rank",
        "spins",
        "mixed",
        "provider",
        "device",
        "stream",
        "dimension",
        "unrestricted",
        "no-binding",
        "publish-failure",
    ],
)
def test_exact_warm_factor_lifecycle(warm_factor_probe: Path, mode: str) -> None:
    result = subprocess.run(
        [str(warm_factor_probe), mode],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if mode == "publish-failure":
        return
    density = np.asarray(json.loads(result.stdout)).reshape(3, 3)
    occupied = 1 / np.sqrt([2.0, 6.0, 3.0])
    expected = 2 * np.outer(occupied, occupied)
    np.testing.assert_allclose(density, expected, atol=2e-16, rtol=2e-16)
    generator = np.random.default_rng(2474)
    fitted = generator.normal(size=(5, 3, 3))
    fitted += fitted.transpose(0, 2, 1)
    dense_k = np.einsum("Pmi,ij,Pnj->mn", fitted, density, fitted)
    projection = fitted @ occupied
    occupied_k = 2 * projection.T @ projection
    np.testing.assert_allclose(occupied_k, dense_k, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize(
    "mode",
    ["qualified", "dense", "spins", "no-frame", "orthonormal", "copy-failure"],
)
def test_held_density_keeps_its_own_factor(warm_factor_probe: Path, mode: str) -> None:
    """An unaccepted proposal cannot replace the held density's exact factor."""
    result = subprocess.run(
        [str(warm_factor_probe), "held-" + mode],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def native_matrix_audit_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Path:
    """Execute the native comparator with finite and poisoned matrix inputs."""
    source = (ROOT / "tests/native/test_ks_cuda.cpp").read_text()
    audit = _definition(source, "void check_fitted_warm_seed_matrices(")
    comparison = audit[
        audit.index("  double exchange_error =") : audit.index("  std::cout")
    ]
    harness = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
void require(bool passed, const char* message) {
  if (!passed) throw std::runtime_error(message);
}
struct Jk { std::vector<double> coulomb{0.25,0.5}, exchange_alpha{0.25,0.5}; };
struct Fock { std::vector<double> alpha{0.25,0.5}; };
int main(int argc, char** argv) {
  assert(argc == 3);
  const auto slot=static_cast<unsigned>(std::atoi(argv[1]));
  const std::string mode=argv[2];
  Jk dense, factored, reference;
  Fock dense_fock, factored_fock;
  std::array<std::vector<double>*,7> inputs{
    &dense.exchange_alpha, &factored.exchange_alpha, &dense_fock.alpha,
    &factored_fock.alpha, &factored.coulomb, &reference.coulomb, &reference.exchange_alpha};
  assert(slot < inputs.size());
  auto& input=*inputs[slot];
  if (mode == "nan") input[1]=std::numeric_limits<double>::quiet_NaN();
  if (mode == "infinity") input[1]=std::numeric_limits<double>::infinity();
  if (mode == "outside") input[1]+=3e-11;
  if (mode == "inside") input[1]+=1e-12;
  const std::size_t matrix=2;
  bool rejected=false;
  try {
"""
    harness += comparison
    harness += r"""
  } catch (const std::runtime_error&) { rejected=true; }
  assert(rejected == (mode == "nan" || mode == "infinity" || mode == "outside"));
}
"""
    folder = tmp_path_factory.mktemp("ks-native-matrix-audit")
    cpp, executable = folder / "probe.cpp", folder / "probe"
    cpp.write_text(harness)
    native_cxx.build_executable(
        [cpp], executable, compile_args=("-std=c++17", "-Wall", "-Werror")
    )
    return executable


@pytest.mark.parametrize("slot", range(7))
@pytest.mark.parametrize("mode", ["equal", "inside", "outside", "nan", "infinity"])
def test_native_warm_matrix_comparison_rejects_nonfinite_inputs(
    native_matrix_audit_probe: Path, slot: int, mode: str
) -> None:
    result = subprocess.run(
        [str(native_matrix_audit_probe), str(slot), mode],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
