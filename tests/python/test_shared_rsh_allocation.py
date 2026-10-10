"""Execute joint-RSH admission with budget boundaries and allocation faults.

The real admission and rollback bodies run unchanged. Only CUDA allocation and
the earlier owner metadata are substituted; native gates check device results.
"""

import subprocess
from pathlib import Path

import pytest
from _cpp_source_support import cpp_function_definition, cpp_if_block
from test_coulomb_optional_allocation import compile_cached_probe
from test_direct_jk_optional_allocation import STUBS

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def admission_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source = (ROOT / "src/scf/cuda/direct_jk.cpp").read_text()
    admission = cpp_if_block(
        source, "plan->canonical_cartesian && plan->canonical_pairs"
    )
    definitions = [
        cpp_function_definition(source, "direct_jk_check"),
        cpp_function_definition(
            source, "direct_jk_optional_storage", include_template=True
        ),
    ]
    stubs = STUBS.replace(
        "struct CudaDirectJkPlan {",
        """
struct OptionalOwner { std::size_t device_bytes = 0; };
struct CudaDirectJkPlan {
  bool canonical_cartesian = true, complete_owner = false;
  const int* canonical_pairs = reinterpret_cast<const int*>(1);
  struct { int nbf = 1; } canonical_batch;
  OptionalOwner *generated_exchange = nullptr, *generated_coulomb = nullptr;
  double* canonical_range_exchange = nullptr;
""",
    )
    folder = tmp_path_factory.mktemp("shared-rsh-admission")
    cpp, executable = folder / "probe.cpp", folder / "probe"
    cpp.write_text(
        stubs + "\n".join(definitions) + DRIVER.replace("// ADMISSION", admission)
    )
    compile_cached_probe(cpp, executable)
    return executable


@pytest.mark.parametrize("fault_kind", range(-1, 7))
@pytest.mark.parametrize("after_allocation", (False, True))
def test_joint_storage_preserves_prior_owners_and_budget(
    admission_probe: Path, fault_kind: int, after_allocation: bool
) -> None:
    subprocess.run(
        [str(admission_probe), str(fault_kind), str(int(after_allocation))],
        check=True,
        timeout=10,
    )


DRIVER = r"""
bool requested = true;
bool direct_shared_rsh_values_requested() { return requested; }
bool direct_jk_generated_full_range_value_available(const CudaDirectJkPlan& p) {
  return p.complete_owner;
}
std::size_t direct_jk_product(std::size_t a, std::size_t b) { return a * b; }
void prepare(CudaDirectJkPlan* plan, bool through_f, std::size_t budget,
             std::size_t batch, int fault_kind, bool after_allocation,
             std::string& detail) {
  std::vector<int> systems(batch);
  const auto scratch = [&](std::size_t bytes) {
    if (fault_kind >= 0 && !after_allocation) fault(fault_kind);
    auto* pointer = static_cast<double*>(allocate(*plan, bytes));
    if (fault_kind >= 0 && after_allocation) fault(fault_kind);
    return pointer;
  };
  // ADMISSION
}
int main(int argc, char** argv) {
  assert(argc == 3);
  const int fault_kind = std::atoi(argv[1]);
  const bool after_allocation = std::atoi(argv[2]);
  for (bool through_f : {false, true})
    for (int owner_kind : {0, 1, 2})
      for (std::size_t dimension : {1U, 10U})
        for (std::size_t batch : {1U, 2U})
          for (int eligibility : {0, 1, 2, 3, 4})
            for (int margin : {-100, -1, 0, 1}) {
              assert(live.empty());
              sticky = synchronization_error = cudaSuccess;
              CudaDirectJkPlan plan;
              auto* retained = static_cast<double*>(allocate(plan, sizeof(double)));
              *retained = 37.0;
              OptionalOwner owner{1234};
              if (owner_kind == 1) plan.generated_exchange = &owner;
              if (owner_kind == 2) plan.generated_coulomb = &owner;
              plan.canonical_batch.nbf = static_cast<int>(dimension);
              requested = eligibility != 1;
              if (eligibility == 2) plan.canonical_cartesian = false;
              if (eligibility == 3) plan.canonical_pairs = nullptr;
              if (eligibility == 4) plan.complete_owner = true;
              const auto charge = batch * dimension * dimension * 2 * sizeof(double);
              // Earlier SPD owners already reduced this local allowance;
              // later through-f owners must be reserved here exactly once.
              const auto later_owner = through_f && owner_kind ? owner.device_bytes : 0U;
              const auto signed_budget = static_cast<long>(sizeof(double) + later_owner + charge)
                                         + margin;
              const auto budget = signed_budget > 0 ? static_cast<std::size_t>(signed_budget) : 0U;
              const bool admitted = eligibility == 0 && margin >= 0;
              bool propagated = false;
              std::string detail;
              try {
                prepare(&plan, through_f, budget, batch, fault_kind, after_allocation, detail);
              } catch (...) { propagated = true; }
              assert(propagated == (admitted && fault_kind >= 3));
              assert(*retained == 37.0 && owner.device_bytes == 1234);
              assert(plan.generated_exchange == (owner_kind == 1 ? &owner : nullptr));
              assert(plan.generated_coulomb == (owner_kind == 2 ? &owner : nullptr));
              if (!propagated) {
                const bool stored = admitted && fault_kind == -1;
                assert(bool(plan.canonical_range_exchange) == stored);
                assert(plan.device_bytes == sizeof(double) + (stored ? charge : 0U));
                assert(plan.allocations.size() == (stored ? 2U : 1U));
                assert(detail.empty() && sticky == cudaSuccess);
              }
            }
  assert(live.empty());
}
"""
