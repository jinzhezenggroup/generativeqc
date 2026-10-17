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
struct CudaXcPointBatchPlan {
  std::size_t tiles{1}, ao_elements{}, feature_elements{}, total_elements{}, device_bytes{};
  bool operator==(const CudaXcPointBatchPlan&) const = default;
};
using CudaXcPointBatchLauncher = void (*)();
void batch_launcher() {}
void mixed_batch_launcher() {}
CudaXcPointBatchPlan expected_batch_plan;
CudaXcPointBatchLauncher expected_batch_launcher{};
double* expected_batch_arena{};
enum class CudaXcAoPrecision { Fp64, Fp32ComputeFp64Storage };
int expected_point_launcher{};
bool expected_fp32_contractions{}, expected_fp32_point_totals{};
CudaXcAoPrecision expected_ao_precision{CudaXcAoPrecision::Fp64};
struct Layout {
  std::size_t nao{3}, npoint{7}, tile_points{3}, work_jets{4}, spins{2};
  bool local_ao{}, response{}, fp32_contractions{}, fp32_point_totals{};
  CudaXcAoPrecision ao_precision{CudaXcAoPrecision::Fp64};
  std::size_t feature_terms{5}, device_bytes{};
};
struct Capabilities { bool mixed_density_contraction{true}; };
Capabilities cuda_xc_execution_capabilities(const Layout&) { return {}; }
namespace cuda_xc_detail {
int mixed_pbe_point_launcher() { return 19; }
CudaXcPointBatchLauncher mixed_pbe_point_batch_launcher() { return &mixed_batch_launcher; }
template <class... T> void enqueue(T... args) {
  auto values = std::tie(args...);
  assert(std::get<0>(values).fp32_contractions == expected_fp32_contractions);
  assert(std::get<0>(values).fp32_point_totals == expected_fp32_point_totals);
  assert(std::get<0>(values).ao_precision == expected_ao_precision);
  assert(std::get<1>(values) == expected_point_launcher);
  assert(std::get<sizeof...(T)-5>(values) == expected_density_provider);
  assert(std::get<sizeof...(T)-4>(values) == expected_potential_binding);
  assert(std::get<sizeof...(T)-3>(values) == expected_batch_plan);
  assert(std::get<sizeof...(T)-2>(values) == expected_batch_launcher);
  assert(std::get<sizeof...(T)-1>(values) == expected_batch_arena);
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
  bool mixed_pbe_points_{}, mixed_pbe_ao_{}, mixed_matrix_contractions_{};
  CudaXcPointBatchPlan point_batch_plan_;
  CudaXcPointBatchLauncher point_batch_launcher_{};
  double* point_batch_arena_{};
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
  assert(argc == 8);
  const bool replay = std::atoi(argv[1]), local = std::atoi(argv[2]);
  const int mode = std::atoi(argv[3]);
  CudaXcPlan plan;
  // Matrix-only arithmetic must leave point totals strict. Both mixed-points
  // and matrix-only preparations remain disabled in StrictAudit; all modes
  // retain the same publication transaction and physical work accounting.
  const auto arithmetic = std::atoi(argv[7]);
  const auto phase = arithmetic == 0 || arithmetic == 2 || arithmetic == 6
                         ? PrecisionPhase::StrictAudit : PrecisionPhase::Admitted;
  plan.mixed_pbe_points_ = arithmetic >= 2 && arithmetic <= 4;
  plan.mixed_pbe_ao_ = arithmetic == 2 || arithmetic == 4;
  plan.mixed_matrix_contractions_ = arithmetic == 2 || arithmetic >= 4;
  const bool mixed = arithmetic == 3 || arithmetic == 4;
  expected_point_launcher = mixed ? 19 : plan.point_launcher_;
  expected_fp32_contractions = arithmetic == 4 || arithmetic == 5;
  expected_fp32_point_totals = arithmetic == 4;
  expected_ao_precision = arithmetic == 4 ? CudaXcAoPrecision::Fp32ComputeFp64Storage
                                        : CudaXcAoPrecision::Fp64;
  double retained_panel{};
  if (std::atoi(argv[6])) {
    plan.point_batch_plan_ = {4, 19, 23, 29, 752};
    plan.point_batch_launcher_ = &batch_launcher;
    plan.point_batch_arena_ = &retained_panel;
  }
  expected_batch_plan = plan.point_batch_plan_;
  expected_batch_launcher = mixed ? &mixed_batch_launcher : plan.point_batch_launcher_;
  expected_batch_arena = plan.point_batch_arena_;
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
  // Strict physical execution admits both dense and indexed/local providers.
  expected_density_provider = plan.density_provider_.get();
  const auto initial = mode == 2 ? std::numeric_limits<std::uint64_t>::max() : 0;
  plan.transfers_.potential_calls = initial;
  failure = mode == 1;
  capture = mode == 3;
  auto submit = [&](std::uint64_t generation) {
    if (replay) plan.publish_submitted_generation(generation);
    else {
      double density[18]{};
      plan.enqueue_impl(density, nullptr, 9 * plan.layout_.spins, generation,
                        phase, nullptr, nullptr, true);
    }
  };
  bool rejected = false;
  try { submit(1); } catch (const std::exception&) { rejected = true; }
  assert(rejected == (mode == 1 || mode == 2));
  assert(bodies == (replay ? 0 : 1));
  assert(!plan.layout_.fp32_contractions);
  assert(!plan.layout_.fp32_point_totals);
  assert(plan.layout_.ao_precision == CudaXcAoPrecision::Fp64);
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


@pytest.mark.parametrize(
    "arithmetic",
    range(7),
    ids=(
        "strict",
        "admitted-strict",
        "strict-mixed-disabled",
        "mixed-points",
        "mixed-full",
        "mixed-matrix-only",
        "strict-matrix-disabled",
    ),
)
@pytest.mark.parametrize("batched", (False, True))
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
    batched: bool,
    arithmetic: int,
) -> None:
    result = subprocess.run(
        [
            str(publication_probe),
            str(int(replay)),
            str(int(local)),
            str(mode),
            str(spins),
            str(int(density_provider)),
            str(int(batched)),
            str(arithmetic),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
