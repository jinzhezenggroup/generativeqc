"""Execute native XC submission/publication with host runtime failure doubles."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner

ROOT = Path(__file__).resolve().parents[2]


def _definition(source: str, signature: str) -> str:
    begin = source.index(signature)
    end = source.index("{", begin) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


@pytest.fixture(scope="module")
def publication_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host compiler required")
    directory = tmp_path_factory.mktemp("xc-potential-publication")
    owner = (ROOT / "src/dft/cuda_xc.cpp").read_text()
    methods = "\n".join(
        _definition(owner, signature)
        for signature in (
            "void CudaXcPlan::publish_submitted_generation(",
            "void CudaXcPlan::publish_potential_work(",
            "void CudaXcPlan::enqueue_impl(",
            "const CudaXcDensityBinding& CudaXcPlan::density_binding(",
            "const tensor::PreparedPanelProduct* CudaXcPlan::density_execution_provider(",
        )
    )
    source = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <tuple>
#include <stdexcept>
#include <vector>
#include "generativeqc/generativeqc.hpp"
#include "runtime/bounded_workspace.hpp"
#include "runtime/lowering_binding.hpp"
using cudaStream_t = int;
using cudaStreamCaptureStatus = int;
constexpr int cudaStreamCaptureStatusNone = 0;
int capture{}, failure{}, bodies{};
int cudaStreamIsCapturing(int stream, int* out) {
  assert(stream == 7); *out = capture; return failure;
}
void check(int status) { if (status) throw std::runtime_error("capture query failed"); }
void device_pointer(const void* pointer, int device) { assert(pointer && device == 2); }
namespace runtime = generativeqc::runtime;
using runtime::size_mul;
namespace generativeqc_tensor {
struct DeviceAllocationError : std::bad_alloc {};
struct DeviceRuntimeError : std::runtime_error { using std::runtime_error::runtime_error; };
}
using runtime::PrecisionPhase;
struct CudaXcDensityBinding { runtime::NativeLoweringPrecision precision; };
namespace tensor {
struct PreparedPanelProduct { bool enabled() const { return true; } };
}
const void *expected_density_provider{}, *expected_potential_binding{};
struct Layout {
  std::size_t nao{3}, npoint{7}, tile_points{3}, work_jets{4}, spins{2};
  bool local_ao{}, response{};
  std::size_t feature_terms{5}, device_bytes{};
};
struct Capabilities { bool mixed_density_contraction{true}; };
Capabilities cuda_xc_execution_capabilities(const Layout&) { return {}; }
namespace cuda_xc_detail {
template <class... T> void enqueue(T... args) {
  auto values = std::tie(args...);
  assert(std::get<sizeof...(T)-2>(values) == expected_density_provider);
  assert(std::get<sizeof...(T)-1>(values) == expected_potential_binding);
  ++bodies;
}
}
struct CudaXcPlan {
  Layout layout_;
  runtime::AsyncGeneration generations_;
  struct { std::uint64_t evaluations{}, potential_calls{}, potential_summands{}; } transfers_;
  cudaStream_t stream_{7};
  int device_{2}, point_launcher_{};
  bool evaluation_started_{};
  double *basis_{}, *points_{}, *weights_{}, *ao_{}, *work_{}, *features_{};
  double *coefficients_{}, *point_totals_{}, *potential_{}, *totals_{}, *delta_features_{};
  void* arena_{};
  int* error_{};
  std::vector<std::size_t> ao_offsets_{0, 2, 2, 3};
  const std::size_t* ao_ids_{};
  std::unique_ptr<int> potential_binding_;
  std::unique_ptr<tensor::PreparedPanelProduct> density_provider_;
  std::unique_ptr<CudaXcDensityBinding> provider_density_binding_;
  std::array<CudaXcDensityBinding, 2> strict_density_, admitted_density_;
  std::vector<int> local_density_launchers_;
  const CudaXcDensityBinding& density_binding(PrecisionPhase) const;
  const tensor::PreparedPanelProduct* density_execution_provider(PrecisionPhase) const;
  void check_device() const {}
  void publish_submitted_generation(std::uint64_t);
  void publish_potential_work();
  void enqueue_impl(const double*, const double*, std::size_t, std::uint64_t,
                    PrecisionPhase, double*, double*, bool);
};
"""
    driver = r"""
int main(int argc, char** argv) {
  assert(argc == 6);
  const bool replay = std::atoi(argv[1]), local = std::atoi(argv[2]);
  const int mode = std::atoi(argv[3]);
  CudaXcPlan plan;
  plan.layout_.spins = std::atoi(argv[4]);
  plan.layout_.local_ao = local;
  for (auto* table : {&plan.strict_density_, &plan.admitted_density_})
    for (auto& binding : *table) binding.precision.arithmetic = runtime::strict_fp64_precision();
  plan.potential_binding_ = std::make_unique<int>(17);
  expected_potential_binding = plan.potential_binding_.get();
  if (std::atoi(argv[5])) {
    plan.density_provider_ = std::make_unique<tensor::PreparedPanelProduct>();
    plan.provider_density_binding_ = std::make_unique<CudaXcDensityBinding>();
  }
  expected_density_provider = local ? nullptr : plan.density_provider_.get();
  const auto initial = mode == 2 ? std::numeric_limits<std::uint64_t>::max() : 0;
  plan.transfers_.potential_calls = initial;
  failure = mode == 1;
  capture = mode == 3;
  auto submit = [&](std::uint64_t generation) {
    if (replay) plan.publish_submitted_generation(generation);
    else {
      double density[18]{};
      plan.enqueue_impl(density, nullptr, 9 * plan.layout_.spins, generation,
                        PrecisionPhase::StrictAudit, nullptr, nullptr, true);
    }
  };
  bool rejected = false;
  try { submit(1); } catch (const std::exception&) { rejected = true; }
  assert(rejected == (mode == 1 || mode == 2));
  assert(bodies == (replay ? 0 : 1));
  if (rejected) {
    assert(plan.generations_.published() == 0);
    assert(plan.transfers_.evaluations == 0);
    assert(plan.transfers_.potential_calls == initial);
    assert(plan.transfers_.potential_summands == 0);
    // The failed submission consumes its generation but must allow recovery.
    failure = 0;
    plan.transfers_.potential_calls = 0;
    submit(2);
    plan.generations_.require(2);
  } else plan.generations_.require(1);
  assert(plan.transfers_.evaluations == 1);
  const auto calls = plan.layout_.spins * (local ? 2 : 3);
  const auto summands = plan.layout_.spins * (local ? 80 : 336);
  assert(plan.transfers_.potential_calls == (capture ? 0 : calls));
  assert(plan.transfers_.potential_summands == (capture ? 0 : summands));
}
"""
    cpp, binary = directory / "publication.cpp", directory / "publication"
    cpp.write_text(source + methods + driver)
    compile_owner(compiler, directory, [cpp], binary)
    return binary


@pytest.mark.parametrize("density_provider", (False, True))
@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize(
    "mode", range(4), ids=("success", "query-failure", "overflow", "capture")
)
@pytest.mark.parametrize("local", (False, True))
@pytest.mark.parametrize("replay", (False, True))
def test_publication_is_atomic_and_counts_only_submitted_products(
    publication_probe: Path,
    replay: bool,
    local: bool,
    mode: int,
    spins: int,
    density_provider: bool,
) -> None:
    result = subprocess.run(
        [
            str(publication_probe),
            str(int(replay)),
            str(int(local)),
            str(mode),
            str(spins),
            str(int(density_provider)),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
