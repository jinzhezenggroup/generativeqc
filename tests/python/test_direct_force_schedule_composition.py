"""Compile production host dispatch for the independent psss/dddd force opt-ins.

CUDA submission is recorded, not executed. This checks ownership, forwarding,
workspace and error boundaries; independent numerical qualification stays in
the Libcint GPU campaign.
"""

import ast
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _definition(source: str, marker: str) -> str:
    begin = source.index(marker)
    opening = source.index("{", begin)
    depth = 0
    for end in range(opening, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[begin : end + 1]
    raise AssertionError(f"unterminated definition: {marker}")


@pytest.fixture(scope="module")
def composed_force_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: "NativeCxx"
) -> Path:
    bounded = (ROOT / "src/scf/cuda/direct_bounded_fallback.cu").read_text()
    angular = (ROOT / "src/scf/cuda/direct_angular_force.cu").read_text()
    definitions = [
        _definition(bounded, "bool materialized_pair_derivative_available("),
        _definition(bounded, "bool cooperative_pair_derivative_available("),
        _definition(angular, "bool direct_force_resident_bra_capacity_supported("),
        _definition(angular, "bool direct_force_resident_bra_schedule_available("),
        _definition(
            bounded,
            "template <bool Unrestricted, DirectRangeOperator Range, unsigned Order = 0U>",
        ),
        _definition(bounded, "cudaError_t launch_bounded_direct_angular_force_kernel("),
    ]
    # Preserve the real launch configuration as ordinary arguments to the stub.
    # Only CUDA's submission punctuation changes; branch and recursion code is
    # compiled directly from production, including both admission predicates.
    production, launches = re.subn(
        r"<<<([^<>]+)>>>\s*\(", r"(\1, ", "\n".join(definitions)
    )
    assert launches == 1
    folder = tmp_path_factory.mktemp("direct-force-schedule-composition")
    cpp, binary = folder / "probe.cpp", folder / "probe"
    cpp.write_text(PREFIX + production + DRIVER)
    native_cxx.build_executable(
        [cpp], binary, compile_args=["-std=c++20", "-O0"], compile_timeout=60
    )
    return binary


@pytest.mark.parametrize("unrestricted", (False, True))
@pytest.mark.parametrize("range_operator", (0, 1, 2))
def test_composed_force_owners_and_submission_failures(
    composed_force_probe: Path, unrestricted: bool, range_operator: int
) -> None:
    """Every lease/cache state must own each order once and stop on any error."""
    result = subprocess.run(
        [str(composed_force_probe), str(int(unrestricted)), str(range_operator)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_independent_force_oracle_retains_both_schedule_dimensions() -> None:
    """The resident branch must not erase dddd, spin, AO or disabled-K gates."""
    module = ast.parse((ROOT / "tests/python/test_cuda_hybrid_snapshot.py").read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_separate_full_range_derivatives_match_libcint"
    )
    dimensions = {
        ast.literal_eval(decorator.args[0]): set(ast.literal_eval(decorator.args[1]))
        for decorator in function.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "parametrize"
    }
    for dimension, expected in {
        "method": {"pbe0-rks", "pbe0-uks", "pbe-rks"},
        "representation": {"cartesian", "spherical"},
        "force_schedule": {"bounded", "angular", "resident"},
        "materialized_derivative": {"0", "1"},
        "two_oxygens": {False, True},
    }.items():
        assert expected <= dimensions[dimension]


PREFIX = r"""
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

enum cudaError_t { cudaSuccess, cudaErrorUnknown, cudaErrorInvalidValue };
using cudaStream_t = unsigned;
struct dim3 { unsigned x; dim3(unsigned value) : x(value) {} };
enum class DirectRangeOperator { Full, Short, Long, FullSources };
enum class DirectScreeningPurpose { Fock, Force };
enum class DirectForceOutputMode { Combined, Separate };
struct ShellPairDensityBounds {};
struct DeviceShellClassProfileEntry {};
struct MaterializedDirectPairDerivativeRecurrence { unsigned char bytes[513]; };
struct CooperativeDirectPairDerivativeRecurrence { unsigned char bytes[777]; };
struct DeviceBatch {
  bool direct_pair_materialized_derivatives{};
  const void* shell_primitive_pairs{};
  const void* shell_pair_primitive_offsets{};
  unsigned direct_coulomb_reachable{}, direct_hermite_convolution{};
  bool direct_pair_cooperative_derivatives{};
  unsigned direct_maximum_shell_angular{255};
};
struct DirectForceResidentBraSchedule {
  const void *tasks{}, *ket_pairs{};
  std::size_t task_count{}, bra_primitive_pair_capacity{};
};
namespace detail { struct BoundedDirectBlockDomain { unsigned identity; }; }
constexpr unsigned kBoundedDirectThreads = 128;
constexpr unsigned kResidentPsssMaximumBraPrimitivePairs = 64;
enum Kind { Reset, Bounded, LastError, Resident };
struct Event {
  Kind kind;
  int order;
  bool pair;
  bool cooperative{};
  bool operator==(const Event&) const = default;
};
std::vector<Event> events;
int fail_event = -1;
bool expected_unrestricted;
DirectRangeOperator expected_range;
DeviceBatch expected_batch;
DirectForceResidentBraSchedule expected_resident;
double shell_bounds, block_bounds, system_bounds, schwarz, density, output;
ShellPairDensityBounds density_bounds;
std::uint32_t pair_order, class_state;
std::uint8_t active;
unsigned long long cursor = 919;
constexpr double coulomb = 1.7, exchange = -0.23, tolerance = 1e-14, omega = 0.3;
constexpr cudaStream_t stream_id = 17;

cudaError_t record(Event event) {
  events.push_back(event);
  return static_cast<int>(events.size()) - 1 == fail_event ? cudaErrorUnknown : cudaSuccess;
}
void check_batch(DeviceBatch batch) {
  assert(batch.direct_pair_materialized_derivatives ==
         expected_batch.direct_pair_materialized_derivatives);
  assert(batch.shell_primitive_pairs == expected_batch.shell_primitive_pairs);
  assert(batch.shell_pair_primitive_offsets == expected_batch.shell_pair_primitive_offsets);
  assert(batch.direct_coulomb_reachable == expected_batch.direct_coulomb_reachable);
  assert(batch.direct_hermite_convolution == expected_batch.direct_hermite_convolution);
  assert(batch.direct_pair_cooperative_derivatives == expected_batch.direct_pair_cooperative_derivatives);
  assert(batch.direct_maximum_shell_angular == expected_batch.direct_maximum_shell_angular);
}
cudaError_t cudaMemsetAsync(void* target, int value, std::size_t bytes, cudaStream_t stream) {
  assert(target == &cursor && value == 0 && bytes == sizeof(cursor) && stream == stream_id);
  const auto error = record({Reset, -1, false});
  if (error == cudaSuccess) cursor = 0;
  return error;
}
cudaError_t cudaGetLastError() {
  assert(!events.empty() && events.back().kind == Bounded);
  return record({LastError, events.back().order, events.back().pair, events.back().cooperative});
}
template<bool Unrestricted, DirectScreeningPurpose Purpose, bool Force,
         unsigned Order, int Range, bool PairDerivatives = false, bool CooperativeDerivatives = false>
void bounded_direct_shell_quartet_kernel(
    dim3 grid, dim3 block, std::size_t bytes, cudaStream_t stream, DeviceBatch batch,
    double screening, const double* shell, const ShellPairDensityBounds* shell_density,
    const std::uint32_t* pairs, const double* blocks, const double* systems,
    const std::uint64_t* mask_pointer, std::uint64_t mask, const std::uint32_t* state,
    const double* schwarz_pointer, const double* density_pointer,
    const std::uint8_t* active_pointer, double* output_pointer,
    unsigned long long* cursor_pointer, DeviceShellClassProfileEntry* profile,
    double j, double k, DirectRangeOperator range, double w, double secondary,
    bool coulomb_only, bool exchange_only, detail::BoundedDirectBlockDomain domain) {
  static_assert(Force && Purpose == DirectScreeningPurpose::Force);
  assert(Unrestricted == expected_unrestricted && static_cast<int>(expected_range) == Range);
  assert(grid.x == 7 && block.x == kBoundedDirectThreads && stream == stream_id);
  assert(screening == tolerance);
  assert(bytes == (CooperativeDerivatives ? sizeof(CooperativeDirectPairDerivativeRecurrence)
                     : PairDerivatives ? sizeof(MaterializedDirectPairDerivativeRecurrence) : 0U));
  assert(shell == &shell_bounds && shell_density == &density_bounds && pairs == &pair_order);
  assert(blocks == &block_bounds && systems == &system_bounds && state == &class_state);
  assert(!mask_pointer && mask == 0 && !profile);
  assert(schwarz_pointer == &schwarz && density_pointer == &density && active_pointer == &active);
  assert(output_pointer == &output && cursor_pointer == &cursor && cursor == 0);
  assert(j == coulomb && k == exchange && range == expected_range && w == omega);
  assert(secondary == 0 && !coulomb_only);
  assert(exchange_only == (range == DirectRangeOperator::Long) && domain.identity == 23);
  check_batch(batch);
  assert(!events.empty() && events.back().kind == Reset);
  record({Bounded, Order, PairDerivatives, CooperativeDerivatives});
  cursor = 919;  // Every subsequent bounded pass must reset its own cursor.
}
cudaError_t launch_direct_force_resident_bra(
    DirectForceOutputMode mode, bool unrestricted, cudaStream_t stream, DeviceBatch batch,
    DirectForceResidentBraSchedule resident, double screening, const double* shell,
    const ShellPairDensityBounds* shell_density, bool density_screening,
    const double* schwarz_pointer, const double* density_pointer,
    const std::uint8_t* active_pointer, double* output_pointer, std::uint64_t mask,
    double j, double k) {
  assert(expected_range == DirectRangeOperator::FullSources ||
         expected_range == DirectRangeOperator::Full);
  assert(mode == (expected_range == DirectRangeOperator::FullSources
                     ? DirectForceOutputMode::Separate : DirectForceOutputMode::Combined));
  assert(unrestricted == expected_unrestricted);
  assert(stream == stream_id && screening == tolerance && shell == &shell_bounds);
  assert(shell_density == &density_bounds && density_screening);
  assert(schwarz_pointer == &schwarz && density_pointer == &density && active_pointer == &active);
  assert(output_pointer == &output && mask == 0 && j == coulomb && k == exchange);
  assert(resident.tasks == expected_resident.tasks && resident.ket_pairs == expected_resident.ket_pairs);
  assert(resident.task_count == expected_resident.task_count);
  assert(resident.bra_primitive_pair_capacity == expected_resident.bra_primitive_pair_capacity);
  check_batch(batch);
  return record({Resident, 1, false});
}
"""


DRIVER = r"""
cudaError_t run() {
  events.clear();
  cursor = 919;
  return launch_bounded_direct_angular_force_kernel(
      expected_unrestricted, 7, stream_id, expected_batch, tolerance, &shell_bounds,
      &density_bounds, &pair_order, &block_bounds, &system_bounds, &class_state,
      &schwarz, &density, &active, &output, &cursor, expected_range, omega,
      coulomb, exchange, {23}, expected_resident);
}
int main(int argc, char** argv) {
  assert(argc == 3);
  expected_unrestricted = std::atoi(argv[1]);
  const int selection = std::atoi(argv[2]);
  const bool long_range = selection == 2;
  expected_range = long_range ? DirectRangeOperator::Long
                             : selection == 1 ? DirectRangeOperator::Full
                                              : DirectRangeOperator::FullSources;
  int views;
  // Resident states: complete, empty, missing each view, zero/overflow task
  // count, zero/over-limit bra capacity. Only a complete lease owns order one.
  for (int resident_state = 0; resident_state < 8; ++resident_state) {
    expected_resident = {&views, &views, 2, kResidentPsssMaximumBraPrimitivePairs};
    switch (resident_state) {
      case 1: expected_resident = {}; break;
      case 2: expected_resident.tasks = nullptr; break;
      case 3: expected_resident.ket_pairs = nullptr; break;
      case 4: expected_resident.task_count = 0; break;
      case 5: expected_resident.task_count =
          std::size_t{std::numeric_limits<unsigned>::max()} + 1; break;
      case 6: expected_resident.bra_primitive_pair_capacity = 0; break;
      case 7: ++expected_resident.bra_primitive_pair_capacity; break;
    }
    // Pair states: admitted, disabled, missing each cache view, conflicting
    // derivative schedules, and unrelated value-only bits (still admitted).
    for (int cooperative_state = 0; cooperative_state < 4; ++cooperative_state) {
    for (int pair_state = 0; pair_state < 7; ++pair_state) {
      expected_batch = {true, &views, &views, 0, 0};
      expected_batch.direct_pair_cooperative_derivatives = cooperative_state != 0;
      expected_batch.direct_maximum_shell_angular = cooperative_state < 2 ? 2
                                                 : cooperative_state == 2 ? 3 : 255;
      switch (pair_state) {
        case 1: expected_batch.direct_pair_materialized_derivatives = false; break;
        case 2: expected_batch.shell_primitive_pairs = nullptr; break;
        case 3: expected_batch.shell_pair_primitive_offsets = nullptr; break;
        case 4: expected_batch.direct_coulomb_reachable = 2; break;
        case 5: expected_batch.direct_hermite_convolution = 2; break;
        case 6: expected_batch.direct_coulomb_reachable =
                    expected_batch.direct_hermite_convolution = 1; break;
      }
      std::vector<Event> expected;
      for (int order = 0; order <= 12; ++order) {
        if (!long_range && resident_state == 0 && order == 1) {
          expected.push_back({Resident, order, false});
        } else {
          const bool pair = !long_range && (pair_state == 0 || pair_state == 6) && order == 8;
          expected.push_back({Reset, -1, false});
          const bool cooperative = !long_range && cooperative_state == 1 &&
              (pair_state == 0 || pair_state == 1 || pair_state == 6) &&
              order == 7;
          expected.push_back({Bounded, order, pair, cooperative});
          expected.push_back({LastError, order, pair, cooperative});
        }
      }
      fail_event = -1;
      assert(run() == cudaSuccess);
      assert(events == expected);
      // An admission success followed by a resident submission error must
      // propagate, never retry bounded work after a possibly enqueued kernel.
      for (std::size_t failure = 0; failure < expected.size(); ++failure) {
        if (expected[failure].kind == Bounded) continue;
        fail_event = static_cast<int>(failure);
        assert(run() == cudaErrorUnknown);
        const std::vector<Event> prefix(expected.begin(), expected.begin() + failure + 1);
        assert(events == prefix);
      }
    }
  }
  }
  // The public wrapper rejects other radial operators before any submission.
  for (auto range : {DirectRangeOperator::Short, static_cast<DirectRangeOperator>(99)}) {
    expected_range = range;
    fail_event = -1;
    assert(run() == cudaErrorInvalidValue && events.empty());
  }
}
"""
