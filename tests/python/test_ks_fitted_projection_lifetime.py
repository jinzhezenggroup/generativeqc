"""Execute the real final occupied-projection lease guards without a GPU."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

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
def projection_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    owner = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    facade = (ROOT / "src/scf/cuda_fock_execution.cpp").read_text()
    binding = (ROOT / "src/scf/cuda_fock_execution.hpp").read_text()
    plan = (ROOT / "src/scf/cuda/df_plan_internal.hpp").read_text()
    public = (ROOT / "src/dft/cuda_ks.hpp").read_text()
    legacy = _definition(owner, "  void enqueue_legacy()")
    submission_begin = legacy.index("      pending_fitted_occupied =")
    submission_end = legacy.index("      generativeqc_status jk_status")
    submitted = legacy[legacy.index("\n      check(jk_status, detail);") :]
    capture = _definition(submitted, "      if (use_occupied_fitted) {")
    unit = r"""
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
using cudaStream_t = void*;
using generativeqc_status = int;
constexpr int GENERATIVEQC_STATUS_SUCCESS=0, GENERATIVEQC_STATUS_INVALID_ARGUMENT=1;
constexpr int GENERATIVEQC_STATUS_NOT_IMPLEMENTED=2, GENERATIVEQC_STATUS_NUMERICAL_FAILURE=3;
int cudaGetLastError() { return 0; }
void check(int code) { assert(code == 0); }
int launches = 0;
template<class... T> void launch_spin_matrix_product_kernel(T...) { ++launches; }
namespace scf {
struct Source {
  std::size_t nbf=2, naux=2, batch_size=1, row_tile=2, auxiliary_tile=2;
  std::size_t completed_occupied_projection_rank=0;
  std::uint64_t projection_scratch_generation=0;
  bool streamed=false, resident_exchange_enabled=true;
  double values[4]{};
  double *three_center=values, *auxiliary_tile_values=values;
  std::vector<bool> metric_full_rank{true};
  struct { int pairs=0; std::size_t rank_capacity=2; } value_storage;
  std::optional<int> final_projection_token;
"""
    unit += _definition(plan, "  void revoke_projection_leases()")
    unit += r"""
};
struct PreparedFockPlan {
  mutable Source source;
  Source* cuda_fitted_source() const { return &source; }
};
bool df_packed_pairs(int mode) { return mode != 0; }
"""
    unit += _definition(binding, "struct PreparedCudaOccupiedFockBinding") + ";\n"
    unit += _definition(binding, "struct PreparedCudaOccupiedProjectionBinding") + ";\n"
    unit += r"""
PreparedCudaOccupiedFockBinding prepared_cuda_occupied_fock_binding(const PreparedFockPlan& p) {
  return {0, reinterpret_cast<void*>(1), &p.source, 2, false};
}
"""
    unit += _definition(
        facade,
        "PreparedCudaOccupiedProjectionBinding prepared_cuda_occupied_projection_binding(",
    )
    unit += "\n}\n"
    unit += _definition(public, "struct CudaKsResidentFittedProjectionBinding") + ";\n"
    unit += r"""
struct CudaKsFinalStateToken {
  unsigned version=1;
  std::uint64_t generation=7;
  bool operator!=(const CudaKsFinalStateToken& other) const {
    return version != other.version || generation != other.generation;
  }
};
struct Owner {
  scf::PreparedFockPlan provider;
  scf::PreparedCudaOccupiedFockBinding occupied_fock_binding =
      scf::prepared_cuda_occupied_fock_binding(provider);
  bool pending_fitted_occupied=true, final_fitted_projection_ready=false;
  bool warm_orbitals_ready=true;
  unsigned spins=1;
  std::array<std::size_t,2> occupations{1,0};
  int device=0;
  void* stream=reinterpret_cast<void*>(1);
  std::size_t n=2, matrix=4;
  std::uint64_t owner=1, solve_epoch=1, final_generation=7;
  std::uint64_t pending_fitted_projection_scratch_generation=0;
  std::uint64_t final_fitted_projection_scratch_generation=0;
  double values[4]{};
  double *x=values, *warm_orbitals=values, *proposal=values;
  unsigned char enabled=1, *final_enabled=&enabled;
  struct { unsigned fitted_final_projection_leases=0; } movement;
  CudaKsFinalStateToken token() const { return {1, final_generation}; }
"""
    unit += _definition(owner, "  void retain_final_fitted_projection()")
    unit += "\n  void capture_submission(bool use_occupied_fitted) {\n"
    unit += legacy[submission_begin:submission_end] + capture + "\n}\n"
    unit += r"""
};
struct CudaKsPlan {
  Owner* impl_;
  generativeqc_status resident_final_fitted_projection(const CudaKsFinalStateToken&,
      CudaKsResidentFittedProjectionBinding&, std::string&) const;
};
"""
    unit += _definition(
        owner, "generativeqc_status CudaKsPlan::resident_final_fitted_projection("
    )
    unit += r"""
int main(int argc, char** argv) {
  assert(argc == 2);
  const std::string mode=argv[1];
  Owner owner;
  auto& source=owner.provider.source;
  source.revoke_projection_leases();
  source.completed_occupied_projection_rank=1;
  owner.capture_submission(true);
  if (mode == "interleaved") {
    source.revoke_projection_leases();
    source.completed_occupied_projection_rank=1;
    owner.retain_final_fitted_projection();
    assert(!owner.final_fitted_projection_ready && launches == 0);
    return 0;
  }
  owner.retain_final_fitted_projection();
  assert(owner.final_fitted_projection_ready && launches == 1);
  const auto token=owner.token();
  CudaKsPlan plan{&owner};
  CudaKsResidentFittedProjectionBinding binding;
  std::string detail;
  assert(plan.resident_final_fitted_projection(token,binding,detail) == GENERATIVEQC_STATUS_SUCCESS);
  assert(binding && launches == 1);
  if (mode == "unchanged") return 0;
  if (mode == "new-state") {
    ++owner.final_generation;
  } else {
    source.revoke_projection_leases();
    if (mode == "same-rank") source.completed_occupied_projection_rank=1;
    if (mode == "different-rank") source.completed_occupied_projection_rank=2;
    if (mode == "overflow") {
      source.projection_scratch_generation=std::numeric_limits<std::uint64_t>::max();
      source.revoke_projection_leases();
      source.completed_occupied_projection_rank=1;
    }
  }
  assert(plan.resident_final_fitted_projection(token,binding,detail) == GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  assert(!binding && launches == 1);
  // Only a new method-owned final publication may bind a new resident write.
  if (mode == "same-rank") {
    owner.capture_submission(true);
    owner.retain_final_fitted_projection();
    assert(plan.resident_final_fitted_projection(token,binding,detail) == GENERATIVEQC_STATUS_SUCCESS);
    assert(binding && launches == 2);
  }
  return 0;
}
"""
    directory = tmp_path_factory.mktemp("ks-projection-lifetime")
    cpp, executable = directory / "probe.cpp", directory / "probe"
    cpp.write_text(unit)
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(cpp),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize(
    "mode",
    [
        "unchanged",
        "revoked",
        "same-rank",
        "different-rank",
        "new-state",
        "overflow",
        "interleaved",
    ],
)
def test_final_projection_rejects_replaced_scratch(
    projection_probe: Path, mode: str
) -> None:
    result = subprocess.run(
        [str(projection_probe), mode],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_dft_df_force_consumes_projection_before_revoke() -> None:
    """The method proof is revalidated by the prepared DF owner before K' consumes it."""
    method = (ROOT / "src/methods/dft_method.cpp").read_text()
    prepared = (ROOT / "src/scf/fock_prepared.cpp").read_text()
    response = (ROOT / "src/scf/cuda/df_force_response.cpp").read_text()

    owner = _definition(
        method, "  generativeqc_status density_fitted_integral_gradient("
    )
    assert "resident_final_fitted_projection(expected" in owner
    assert (
        "prepared_cuda_occupied_projection_binding(fock_, resident_projection.rank)"
        in owner
    )
    assert "fitted_projection ? &fitted_projection : nullptr" in owner

    bridge = _definition(
        prepared,
        "FockEnergyDerivativeComponents PreparedFockPlan::energy_derivative_components(\n"
        "    const std::vector<double>& density, const std::vector<double>& beta,\n"
        "    const CudaDensityFittingOccupiedProjectionLease* occupied_projection)",
    )
    assert bridge.index(
        "result.exchange = execute(exchange, occupied_projection)"
    ) < bridge.index("result.coulomb = execute(coulomb, nullptr)")

    consumer = _definition(
        response,
        "generativeqc_status execute_cuda_density_fitting_generated_force_response(",
    )
    proof = consumer.index(
        "external fitted occupied projection differs from the prepared DF owner"
    )
    revoke = consumer.index("plan->revoke_projection_leases();")
    assert proof < revoke
    assert "lease.source_identity != plan" in consumer
    assert "lease.scratch_generation != plan->projection_scratch_generation" in consumer
    assert "streamed_factors.owner_identity ? nullptr" in consumer
