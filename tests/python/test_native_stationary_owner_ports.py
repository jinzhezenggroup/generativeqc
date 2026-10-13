"""Exercise stationary adapters through private owner ports without a CUDA device."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_stationary_owner_port_forwarding_and_refusals(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Preserve exact arguments, typed failures, and non-KS refusal publication."""
    source = tmp_path / "stationary_owner_ports.cpp"
    source.write_text(
        r"""
#include <cstdlib>
#include <stdexcept>
#include "methods/dft_method_owner_port.hpp"

using namespace generativeqc;
using namespace generativeqc::methods;
using namespace generativeqc::methods::detail;
using Token = dft::CudaKsFinalStateToken;
using Matrices = std::vector<scf::reference::Matrix>;
using Work = std::array<std::uint64_t, 9>;

void require(bool condition) {
  if (!condition) throw std::runtime_error("stationary owner-port contract");
}
class PlainBatch : public PreparedBatch {
 public:
  std::size_t size() const noexcept override { return 2; }
 protected:
  std::vector<BatchItemResult> execute_impl(const Coordinates&, bool) override { return {}; }
};
class StationaryOwner final : public PlainBatch, public KsBatchOwnerPort {
 public:
  generativeqc_status result{GENERATIVEQC_STATUS_SUCCESS};
  Token expected;
  mutable unsigned observed{};
  static constexpr std::size_t item = 1, allowance = 927;
  generativeqc_status native_cuda_stationary_gradient(
      std::size_t index, const Token& token, std::size_t bytes, std::vector<double>& output,
      dft::CudaStationaryResources& resources, std::string& detail,
      scf::DirectForcePrecisionRequest* precision) override {
    require(index == item && token == expected && bytes == allowance);
    if (precision) precision->work = {7, 11, 0};
    observed |= 1;
    detail = "gradient owner";
    if (result == GENERATIVEQC_STATUS_SUCCESS) {
      output = {4, 5, 6};
      resources.device_upper_bytes = allowance;
    }
    return result;
  }
  generativeqc_status native_cuda_stationary_capacity(
      std::size_t index, dft::CudaStationaryCapacity& capacity, std::string& detail) const override {
    require(index == item);
    observed |= 2;
    detail = "capacity owner";
    if (result == GENERATIVEQC_STATUS_SUCCESS) capacity.minimum_device_bytes = allowance;
    return result;
  }
  generativeqc_status native_cuda_stationary_limit(
      std::size_t index, std::size_t bytes, std::string& detail) override {
    require(index == item && bytes == allowance);
    observed |= 4;
    detail = "limit owner";
    return result;
  }
  generativeqc_status cuda_prepare_force_memory(std::size_t index, const Token& token,
      std::size_t bytes, std::array<std::uint64_t, 4>& receipt, std::string& detail) override {
    require(index == item && token == expected && bytes == allowance);
    observed |= 8;
    detail = "force memory owner";
    if (result == GENERATIVEQC_STATUS_SUCCESS) receipt = {bytes, 100, 200, 100};
    return result;
  }
  // Any unrelated owner call is a test failure; the production adapters select
  // only the requested port and do not export a detached snapshot first.
  generativeqc_status final_state_token(std::size_t, Token&, std::string&) const override {
    std::abort();
  }
  generativeqc_status read_final_state(std::size_t, const Token&, bool,
      dft::VerifiedKsFinalState&, std::string&) override { std::abort(); }
  generativeqc_status cuda_full_range_integral_derivatives(std::size_t, const Token&,
      std::vector<double>&, std::string&) override { std::abort(); }
  generativeqc_status cuda_integral_gradient_additional_device_bytes(std::size_t, const Token&,
      std::size_t&, std::string&) override { std::abort(); }
  generativeqc_status cuda_integral_gradient(std::size_t, const Token&, std::vector<double>&,
      std::size_t, Work&, std::string&, const Matrices*, const Matrices*, bool) override {
    std::abort();
  }
  generativeqc_status density_fitted_integral_gradient(std::size_t, const Token&,
      const Matrices&, const Matrices&, std::vector<double>&, std::size_t, Work&,
      std::string&) override { std::abort(); }
  generativeqc_status mixed_density_fitted_integral_gradient(std::size_t, const Token&,
      const Matrices&, const Matrices&, std::span<const double>, std::vector<double>&,
      std::size_t, Work&, std::string&) override { std::abort(); }
  generativeqc_status resident_density(std::size_t, const Token&, int&, const double*&,
      const double*&, std::size_t&, unsigned&, void*&, std::string&) override { std::abort(); }
  generativeqc_status fixed_density_profile(std::size_t, const Token&, KsFixedDensityProfile&,
      std::string&) override { std::abort(); }
  generativeqc_status resident_grid(std::size_t, const Token&, int&, const double*&,
      const double*&, const double*&, std::size_t&, std::string&) override { std::abort(); }
  generativeqc_status resident_nonlocal_features(std::size_t, const Token&, int&,
      const double*&, const double*&, std::size_t&, void*&, std::string&) override {
    std::abort();
  }
  generativeqc_status read_derivative_state(std::size_t, const Token&, KsDerivativeSnapshot&,
      std::string&) override { std::abort(); }
  generativeqc_status read_derivative_sources(std::size_t, const Token&, KsDerivativeSnapshot&,
      std::string&) override { std::abort(); }
};
int main() {
  StationaryOwner owner;
  owner.expected.identity.model.owner = 39;
  owner.expected.identity.model.functional = 7;
  owner.expected.identity.model.device = 2;
  owner.expected.identity.model.semilocal_exchange_scale = 0.75;
  PreparedBatch& batch = owner;
  for (auto status : {GENERATIVEQC_STATUS_SUCCESS, GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      GENERATIVEQC_STATUS_OUT_OF_MEMORY, GENERATIVEQC_STATUS_CUDA_ERROR,
                      GENERATIVEQC_STATUS_NUMERICAL_FAILURE, GENERATIVEQC_STATUS_NOT_IMPLEMENTED}) {
    owner.result = status;
    owner.observed = 0;
    std::vector<double> output{99};
    dft::CudaStationaryResources resources;
    resources.device_upper_bytes = 123;
    std::string detail;
    require(dft_cuda_native_stationary_gradient(batch, owner.item, owner.expected,
        owner.allowance, output, resources, detail) == status);
    require(detail == "gradient owner");
    require(output == (status == GENERATIVEQC_STATUS_SUCCESS ? std::vector<double>{4, 5, 6}
                                                            : std::vector<double>{99}));
    require(resources.device_upper_bytes == (status == GENERATIVEQC_STATUS_SUCCESS ? 927 : 123));
    dft::CudaStationaryCapacity capacity;
    capacity.minimum_device_bytes = 321;
    require(dft_cuda_stationary_capacity(batch, owner.item, capacity, detail) == status);
    require(detail == "capacity owner");
    require(capacity.minimum_device_bytes == (status == GENERATIVEQC_STATUS_SUCCESS ? 927 : 321));
    require(dft_cuda_stationary_limit(batch, owner.item, owner.allowance, detail) == status);
    require(detail == "limit owner" && owner.observed == 7);
    std::array<std::uint64_t, 4> receipt{9, 9, 9, 9};
    require(dft_cuda_prepare_force_memory(batch, owner.item, owner.expected,
        owner.allowance, receipt, detail) == status);
    require(detail == "force memory owner" && owner.observed == 15);
    require(receipt == (status == GENERATIVEQC_STATUS_SUCCESS
        ? std::array<std::uint64_t, 4>{927, 100, 200, 100} : std::array<std::uint64_t, 4>{}));
  }
  PlainBatch foreign;
  std::vector<double> output{99};
  dft::CudaStationaryResources resources;
  resources.device_upper_bytes = 123;
  std::string detail;
  owner.result = GENERATIVEQC_STATUS_SUCCESS;
  scf::DirectForcePrecisionRequest precision{true, true};
  require(dft_cuda_native_stationary_gradient(batch, owner.item, owner.expected,
      owner.allowance, output, resources, detail, &precision) == GENERATIVEQC_STATUS_SUCCESS);
  require(precision.mixed_weighted && precision.collect_work &&
          precision.work == std::array<std::uint64_t, 3>{7, 11, 0});
  output = {99};
  resources.device_upper_bytes = 123;
  require(dft_cuda_native_stationary_gradient(foreign, 1, owner.expected, 927,
      output, resources, detail) == GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
  require(output == std::vector<double>{99} && resources.device_upper_bytes == 0 && !detail.empty());
  dft::CudaStationaryCapacity capacity;
  capacity.minimum_device_bytes = 321;
  require(dft_cuda_stationary_capacity(foreign, 1, capacity, detail) ==
      GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
  require(capacity.minimum_device_bytes == 321 && !detail.empty());
  require(dft_cuda_stationary_limit(foreign, 1, 927, detail) == GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
  require(!detail.empty());
  std::array<std::uint64_t, 4> receipt{9, 9, 9, 9};
  require(dft_cuda_prepare_force_memory(foreign, 1, owner.expected, 927, receipt, detail) ==
      GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
  require(receipt == std::array<std::uint64_t, 4>{} && !detail.empty());
}
"""
    )
    executable = native_cxx.build_executable(
        [
            source,
            ROOT / "src/methods/dft_method_derivative_adapter.cpp",
            ROOT / "src/methods/prepared_execution.cpp",
        ],
        tmp_path / "stationary_owner_ports",
        compile_args=[
            "-std=c++20",
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
        ],
    )
    subprocess.run([str(executable)], check=True, timeout=30)
