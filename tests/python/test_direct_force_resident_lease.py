"""Exercise the actual resident admission and asynchronous rollback boundary."""

import subprocess
from pathlib import Path

from generativeqc_compiler.integral.direct_resident_schedule import (
    emit_direct_resident_psss_schedule_header,
)
from _cpp_source_support import cpp_function_definition, cpp_if_block
from test_coulomb_optional_allocation import compile_cached_probe
from test_direct_jk_optional_allocation import STUBS

ROOT = Path(__file__).resolve().parents[2]


def test_resident_capacity_and_complete_view_admission(tmp_path: Path) -> None:
    """Zero, over-limit, null and complete leases select the intended owner."""
    source = (ROOT / "src/scf/cuda/direct_angular_force.cu").read_text()
    definitions = "\n".join(
        cpp_function_definition(source, name)
        for name in (
            "direct_force_resident_bra_capacity_supported",
            "direct_force_resident_bra_schedule_available",
        )
    )
    cpp, binary = tmp_path / "admission.cpp", tmp_path / "admission"
    cpp.write_text(
        "#include <cassert>\n#include <cstdint>\n#include <limits>\n"
        + emit_direct_resident_psss_schedule_header()
        + "using namespace generativeqc::scf::cuda_execution;\n"
        + "struct DirectForceResidentBraSchedule { const void *tasks, *ket_pairs; "
        "std::size_t task_count, bra_primitive_pair_capacity; };\n"
        + definitions
        + r"""
int main() {
  const auto limit = kResidentPsssMaximumBraPrimitivePairs;
  assert(!direct_force_resident_bra_capacity_supported(0, limit));
  assert(!direct_force_resident_bra_capacity_supported(1, 0));
  assert(!direct_force_resident_bra_capacity_supported(1, limit+1));
  assert(!direct_force_resident_bra_capacity_supported(
      std::size_t{std::numeric_limits<unsigned>::max()}+1, limit));
  assert(direct_force_resident_bra_capacity_supported(1, limit));
  int tasks = 0, pairs = 0;
  assert(!direct_force_resident_bra_schedule_available({nullptr,&pairs,1,limit}));
  assert(!direct_force_resident_bra_schedule_available({&tasks,nullptr,1,limit}));
  assert(direct_force_resident_bra_schedule_available({&tasks,&pairs,1,limit}));
}
"""
    )
    compile_cached_probe(cpp, binary)
    subprocess.run([str(binary)], check=True, timeout=10)


def test_partial_resident_allocation_preserves_bounded_lease(tmp_path: Path) -> None:
    """OOM drains pending uploads and retires only optional views; errors propagate."""
    source = (ROOT / "src/scf/cuda/direct_coulomb.cpp").read_text()
    block = cpp_if_block(source, "retain_resident")
    stubs = STUBS.replace("CudaDirectJkPlan", "GeneratedExchangePlan").replace(
        "  std::size_t device_bytes = 0;",
        "  std::size_t device_bytes = 0;\n"
        "  struct Lease { const void *tasks{}, *ket_pairs{}; "
        "std::size_t count{}, capacity{}; } force_resident_bra;",
    )
    driver = r"""
void check(cudaError_t error) { if (error != cudaSuccess) throw error; }
struct PsssResidentTask { unsigned bra, begin, count; };
void exercise(int fail_at, int kind, bool fail_fence) {
  assert(live.empty());
  sticky = synchronization_error = cudaSuccess;
  {
    auto plan = std::make_unique<GeneratedExchangePlan>();
    auto* required = static_cast<double*>(::allocate(*plan, sizeof(double)));
    *required = 37.0;
    const auto required_bytes = plan->device_bytes;
    std::vector<PsssResidentTask> resident_tasks{{0,0,2}};
    std::vector<std::uint32_t> resident_ket_pairs{1,2};
    const auto resident_bytes = resident_tasks.size()*sizeof(PsssResidentTask)+
                                resident_ket_pairs.size()*sizeof(std::uint32_t);
    auto additional = required_bytes + resident_bytes;
    bool retain_resident = true;
    std::size_t resident_bra_capacity = 4;
    auto stream = plan->stream;
    int call = 0;
    auto allocate = [&](std::size_t count, std::size_t width, const void* values) {
      if (call++ == fail_at) {
        if (fail_fence) synchronization_error = cudaErrorUnknown;
        fault(kind);
      }
      auto* pointer = ::allocate(*plan, count*width);
      pending.push_back({pointer,values,count*width});
      return pointer;
    };
    bool propagated = false;
    try {
__PRODUCTION_BLOCK__
    } catch (cudaError_t error) {
      assert(error == cudaErrorUnknown);
      propagated = true;
    }
    assert(propagated == (fail_at < 2 && (kind == 4 || fail_fence)));
    assert(*required == 37.0 && live.count(required));
    if (!propagated) assert(plan->device_bytes == additional);
    if (fail_at < 2 && !propagated) {
      assert(plan->device_bytes == required_bytes && plan->allocations.size() == 1);
      assert(!plan->force_resident_bra.tasks && !plan->force_resident_bra.ket_pairs);
      assert(pending.empty() && sticky == cudaSuccess);
    } else if (fail_at == 2) {
      assert(plan->device_bytes > required_bytes && plan->allocations.size() == 3);
      assert(plan->force_resident_bra.tasks && plan->force_resident_bra.ket_pairs);
    }
    synchronization_error = cudaSuccess;
  }
  assert(live.empty() && pending.empty());
}
int main() {
  for (int fail_at : {0,1,2})
    for (int kind : {0,2,4}) exercise(fail_at,kind,false);
  for (int fail_at : {0,1}) exercise(fail_at,2,true);
}
""".replace("__PRODUCTION_BLOCK__", block)
    cpp, binary = tmp_path / "rollback.cpp", tmp_path / "rollback"
    cpp.write_text("#include <memory>\n#include <cstdint>\n" + stubs + driver)
    compile_cached_probe(cpp, binary)
    subprocess.run([str(binary)], check=True, timeout=10)
