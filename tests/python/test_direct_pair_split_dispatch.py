"""Compile the real split/fallback dispatcher with fault-injected CUDA seams."""

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from test_direct_force_schedule_composition import _definition

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_materialized_split_preserves_ownership_forwarding_and_errors(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Exercise every admission guard, both source layouts and all failed steps."""
    source = (ROOT / "src/scf/cuda/direct_bounded_fallback.cu").read_text()
    cooperative_source = (ROOT / "src/scf/cuda/direct_order_seven_force.cu").read_text()
    class_source = (ROOT / "src/scf/cuda/direct_force_class_domains.cu").read_text()
    definitions = "\n".join(
        _definition(
            class_source
            if marker == "cudaError_t launch_direct_force_class_domains("
            else cooperative_source
            if marker == "cudaError_t launch_direct_order_seven_force("
            else source,
            marker,
        )
        for marker in (
            "bool materialized_pair_derivative_available(",
            "bool cooperative_pair_derivative_available(",
            "cudaError_t launch_direct_order_seven_force(",
            "cudaError_t launch_direct_force_class_domains(",
            "cudaError_t launch_bounded_direct_shell_quartet_kernel_scaled(",
        )
    )
    production, launches = re.subn(r"<<<([^<>]+)>>>\s*\(", r"(\1, ", definitions)
    assert launches == 6
    probe = tmp_path / "split.cpp"
    probe.write_text(PREFIX + production + DRIVER)
    executable = native_cxx.build_executable(
        [probe],
        tmp_path / "split",
        compile_args=["-std=c++20", "-O0", f"-I{ROOT / 'src'}"],
    )
    result = subprocess.run([executable], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


PREFIX = r"""
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>
#define __host__
#define __device__
#include "scf/cuda/direct_force_class_pages.cuh"
using namespace generativeqc::scf::cuda_execution;
enum cudaError_t { cudaSuccess, cudaErrorUnknown };
using cudaStream_t = unsigned;
struct dim3 {
  unsigned x{}, y{1}, z{1};
  dim3(unsigned width = 1, unsigned height = 1, unsigned depth = 1)
      : x(width), y(height), z(depth) {}
  bool operator==(const dim3&) const = default;
};
enum class DirectRangeOperator { Full = 0, FullSources = 4 };
enum class DirectForceOutputMode { Combined, Separate };
enum class DirectScreeningPurpose { Fock, Force };
struct DeviceBatch {
  bool direct_pair_materialized_derivatives{};
  bool direct_pair_cooperative_derivatives{};
  const void* shell_primitive_pairs{};
  const void* shell_pair_primitive_offsets{};
  unsigned direct_coulomb_reachable{}, direct_hermite_convolution{};
  unsigned direct_maximum_shell_angular{255};
  bool operator==(const DeviceBatch&) const = default;
};
struct ShellPairDensityBounds {};
struct GeneratedShellPairStream {};
struct DeviceShellClassProfileEntry {};
struct MaterializedDirectPairDerivativeRecurrence { unsigned char bytes[513]; };
struct CooperativeDirectPairDerivativeRecurrence { unsigned char bytes[1537]; };
namespace detail {
struct BoundedDirectBlockDomain {
  unsigned identity{};
  bool operator==(const BoundedDirectBlockDomain&) const = default;
};
}
constexpr unsigned kBoundedDirectThreads = 256, kBoundedDirectForceThreads = 128;
enum Kind { Launch, Peek, Reset };
struct Event {
  Kind kind;
  int order{};
  bool pair{};
  bool pure{};
  unsigned width{};
  std::size_t bytes{};
  bool cooperative{};
  bool operator==(const Event&) const = default;
};
std::vector<Event> events;
int fail_event = -1;
cudaError_t last_error = cudaSuccess;
unsigned long long cursor{};
DeviceBatch expected_batch;
GeneratedShellPairStream topology;
const GeneratedShellPairStream* expected_topology;
bool expected_spin, expected_separate;
DirectScreeningPurpose expected_purpose;
double shell_bounds, block_bounds, system_bounds, schwarz, density, output;
ShellPairDensityBounds density_bounds;
DeviceShellClassProfileEntry profile;
std::uint32_t pair_order, class_state;
std::uint64_t mask_pointer;
std::uint8_t active;
constexpr double tolerance = 1e-13, coulomb = 1.7, exchange = -0.23;
constexpr std::uint64_t mask = 919;
constexpr cudaStream_t stream_id = 17;
constexpr detail::BoundedDirectBlockDomain domain{37};

cudaError_t record(Event event) {
  events.push_back(event);
  return static_cast<int>(events.size()) - 1 == fail_event ? cudaErrorUnknown : cudaSuccess;
}
cudaError_t cudaPeekAtLastError() {
  assert(!events.empty() && events.back().kind == Launch);
  const auto injected = record({Peek});
  return injected == cudaSuccess ? last_error : injected;
}
cudaError_t cudaMemsetAsync(void* target, int value, std::size_t bytes, cudaStream_t stream) {
  assert(target == &cursor && value == 0 && bytes == sizeof(cursor) && stream == stream_id);
  const auto error = record({Reset});
  if (error == cudaSuccess) cursor = 0;
  return error;
}
template<bool Unrestricted, DirectForceOutputMode Mode>
void direct_order_seven_force_kernel(
    dim3 grid, dim3 block, std::size_t bytes, cudaStream_t stream, DeviceBatch batch,
    const GeneratedShellPairStream* force_topology, double screening, const double* shell,
    const ShellPairDensityBounds* shell_density, const std::uint64_t* enabled_pointer,
    std::uint64_t enabled_mask, const std::uint32_t* state, const double* schwarz_pointer,
    const double* density_pointer, const std::uint8_t* active_pointer, double* output_pointer,
    unsigned long long* cursor_pointer, DeviceShellClassProfileEntry* profile_pointer,
    double j_coefficient, double k_coefficient) {
  assert(Unrestricted == expected_spin && expected_purpose == DirectScreeningPurpose::Force);
  assert((Mode == DirectForceOutputMode::Separate) == expected_separate);
  assert(batch == expected_batch && grid == dim3{7} && block == dim3{256} && stream == stream_id);
  assert(force_topology == expected_topology && force_topology != nullptr);
  assert(screening == tolerance && shell == &shell_bounds && shell_density == &density_bounds);
  assert(enabled_pointer == &mask_pointer && enabled_mask == mask && state == &class_state);
  assert(schwarz_pointer == &schwarz && density_pointer == &density && active_pointer == &active);
  assert(output_pointer == &output && cursor_pointer == &cursor && profile_pointer == &profile);
  assert(j_coefficient == coulomb && k_coefficient == exchange && cursor == 0);
  last_error = record({Launch,7,false,false,256,bytes,true});
  cursor = 999;
}
template<bool Unrestricted, DirectForceOutputMode Mode, DirectForceClassDomain Domain>
void direct_force_class_domain_kernel(
    dim3 grid, dim3 block, std::size_t bytes, cudaStream_t stream, DeviceBatch batch,
    const GeneratedShellPairStream* force_topology, double screening, const double* shell,
    const ShellPairDensityBounds* shell_density, const std::uint64_t* enabled_pointer,
    std::uint64_t enabled_mask, const std::uint32_t* state, const double* schwarz_pointer,
    const double* density_pointer, const std::uint8_t* active_pointer, double* output_pointer,
    unsigned long long* cursor_pointer, DeviceShellClassProfileEntry* profile_pointer,
    double j_coefficient, double k_coefficient) {
  constexpr bool cooperative = Domain == DirectForceClassDomain::Cooperative;
  constexpr bool materialized = Domain == DirectForceClassDomain::Materialized;
  constexpr unsigned width = cooperative || materialized ? 256U : 128U;
  constexpr int order = Domain == DirectForceClassDomain::LowOrder ? 0
      : Domain == DirectForceClassDomain::WeightedFour ? 4
      : Domain == DirectForceClassDomain::WeightedFive ? 5 : cooperative ? 6 : 8;
  assert(Unrestricted == expected_spin && expected_purpose == DirectScreeningPurpose::Force);
  assert((Mode == DirectForceOutputMode::Separate) == expected_separate);
  assert(batch == expected_batch && grid == dim3{7} && block == dim3{width} && stream == stream_id);
  assert(force_topology == expected_topology && force_topology != nullptr);
  assert(screening == tolerance && shell == &shell_bounds && shell_density == &density_bounds);
  assert(enabled_pointer == &mask_pointer && enabled_mask == mask && state == &class_state);
  assert(schwarz_pointer == &schwarz && density_pointer == &density && active_pointer == &active);
  assert(output_pointer == &output && cursor_pointer == &cursor && profile_pointer == &profile);
  assert(j_coefficient == coulomb && k_coefficient == exchange && cursor == 0);
  last_error = record({Launch,order,materialized,materialized,width,bytes,cooperative});
  cursor = 999;
}
template<bool Unrestricted, DirectScreeningPurpose Purpose, bool Force,
         int AngularOrder = -1, int Radial = -1, bool PairDerivatives = false,
         bool CooperativeDerivatives = false, bool PureMaterializedDerivatives = false>
void bounded_direct_shell_quartet_kernel(
    dim3 grid, dim3 block, std::size_t bytes, cudaStream_t stream, DeviceBatch batch,
    double screening, const double* shell, const ShellPairDensityBounds* shell_density,
    const std::uint32_t* pairs, const double* blocks, const double* systems,
    const std::uint64_t* enabled_pointer, std::uint64_t enabled_mask,
    const std::uint32_t* state, const double* schwarz_pointer, const double* density_pointer,
    const std::uint8_t* active_pointer, double* output_pointer,
    unsigned long long* cursor_pointer, DeviceShellClassProfileEntry* profile_pointer,
    double j_coefficient, double k_coefficient, DirectRangeOperator range,
    double omega, double secondary, bool coulomb_only, bool exchange_only,
    detail::BoundedDirectBlockDomain block_domain) {
  static_assert(Force);
  if constexpr (CooperativeDerivatives)
    assert(!PairDerivatives && !PureMaterializedDerivatives && AngularOrder == 7 &&
           batch.direct_maximum_shell_angular == 2);
  if constexpr (PureMaterializedDerivatives)
    assert(PairDerivatives && AngularOrder == 8 && batch.direct_maximum_shell_angular == 2);
  assert(Unrestricted == expected_spin && Purpose == expected_purpose);
  assert(batch == expected_batch && grid == dim3{7} && stream == stream_id);
  assert(screening == tolerance && shell == &shell_bounds && shell_density == &density_bounds);
  assert(pairs == &pair_order && blocks == &block_bounds && systems == &system_bounds);
  assert(enabled_pointer == &mask_pointer && enabled_mask == mask && state == &class_state);
  assert(schwarz_pointer == &schwarz && density_pointer == &density && active_pointer == &active);
  assert(output_pointer == &output && cursor_pointer == &cursor && profile_pointer == &profile);
  assert(j_coefficient == coulomb && k_coefficient == exchange && omega == 0 && secondary == 0);
  assert(!coulomb_only && !exchange_only && block_domain == domain);
  assert(range == (expected_separate ? DirectRangeOperator::FullSources : DirectRangeOperator::Full));
  assert(Radial == -1 || Radial == static_cast<int>(range));
  if constexpr (AngularOrder == -2 || AngularOrder == -3 || AngularOrder == 7 || AngularOrder == 8) assert(cursor == 0);
  last_error = record({Launch, AngularOrder, PairDerivatives, PureMaterializedDerivatives, block.x, bytes, CooperativeDerivatives});
  cursor = 999;
}
"""

DRIVER = r"""
int main() {
  for (bool spin : {false, true})
    for (bool separate : {false, true})
      for (auto purpose : {DirectScreeningPurpose::Fock, DirectScreeningPurpose::Force})
        for (unsigned maximum : {0U, 1U, 2U, 3U, 255U})
          for (unsigned state_bits = 0; state_bits < 128; ++state_bits)
            for (dim3 block : {dim3{256}, dim3{128}, dim3{64}, dim3{256,2}, dim3{256,1,2}})
              for (std::size_t requested_bytes : {0U, 1024U, 4096U}) {
                expected_spin = spin;
                expected_separate = separate;
                expected_purpose = purpose;
                expected_topology = state_bits & 64 ? &topology : nullptr;
                expected_batch = {bool(state_bits & 1), bool(state_bits & 32), state_bits & 2 ? &profile : nullptr,
                    state_bits & 4 ? &profile : nullptr, state_bits & 8 ? 2U : 0U,
                    state_bits & 16 ? 2U : 0U, maximum};
                const bool pair = expected_batch.direct_pair_materialized_derivatives &&
                    expected_batch.shell_primitive_pairs && expected_batch.shell_pair_primitive_offsets &&
                    !expected_batch.direct_coulomb_reachable && !expected_batch.direct_hermite_convolution;
                const bool split = pair && purpose == DirectScreeningPurpose::Force && maximum == 2 && block == dim3{256};
                const bool cooperative = split && expected_batch.direct_pair_cooperative_derivatives && expected_topology;
                const bool class_domains = pair && purpose == DirectScreeningPurpose::Force && maximum == 3 &&
                    expected_batch.direct_pair_cooperative_derivatives && expected_topology && block == dim3{256};
                const auto workspace = pair ? std::max(requested_bytes, sizeof(MaterializedDirectPairDerivativeRecurrence)) : requested_bytes;
                const auto cooperative_workspace = std::max(requested_bytes, sizeof(CooperativeDirectPairDerivativeRecurrence));
                std::vector<Event> expected;
                if (class_domains)
                  expected = {{Launch,-4,false,false,128,requested_bytes},{Peek},
                              {Reset},{Launch,0,false,false,128,requested_bytes},{Peek},
                              {Reset},{Launch,4,false,false,128,requested_bytes},{Peek},
                              {Reset},{Launch,5,false,false,128,requested_bytes},{Peek},
                              {Reset},{Launch,6,false,false,256,cooperative_workspace,true},{Peek},
                              {Reset},{Launch,8,true,true,256,workspace},{Peek}};
                else if (cooperative)
                  expected = {{Launch,-3,false,false,128,requested_bytes},{Peek},{Reset},
                              {Launch,7,false,false,256,cooperative_workspace,true},{Peek},{Reset},
                              {Launch,8,true,true,256,workspace},{Peek}};
                else if (split)
                  expected = {{Launch,-2,false,false,128,requested_bytes},{Peek},{Reset},
                              {Launch,8,true,true,256,workspace},{Peek}};
                else
                  expected = {{Launch,-1,pair,false,!pair && block == dim3{256} ? 128U : block.x,workspace},{Peek}};
                for (int failed = -1; failed < static_cast<int>(expected.size()); ++failed) {
                  fail_event = failed;
                  events.clear(); last_error = cudaSuccess; cursor = 0;
                  const auto error = launch_bounded_direct_shell_quartet_kernel_scaled(
                      spin,purpose,7,block,requested_bytes,stream_id,expected_batch,tolerance,
                      &shell_bounds,&density_bounds,&pair_order,&block_bounds,&system_bounds,
                      &mask_pointer,mask,&class_state,&schwarz,&density,&active,&output,
                      &cursor,&profile,coulomb,exchange,separate,domain,expected_topology);
                  assert(error == (failed < 0 ? cudaSuccess : cudaErrorUnknown));
                  std::size_t completed = expected.size();
                  if (failed >= 0) {
                    completed = failed + 1;
                    if (expected[failed].kind == Launch) ++completed;
                  }
                  assert(events == std::vector<Event>(expected.begin(), expected.begin() + completed));
                }
              }
}
"""
