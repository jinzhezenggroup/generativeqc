"""Execute KS reservation forwarding and the real packed planner without CUDA.

Method selection, constructor forwarding/validation, matching, tile planning and
allocator arguments execute production definitions; device allocation and
unrelated preparation are stubbed. A disconnected reservation fails the gate.
Actual device projection/response is covered by the public CUDA endpoint test.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def definition(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def reservation_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: object
) -> Path:
    directory = tmp_path_factory.mktemp("ks-projection-reservation")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_df_exchange_schedule.py"),
            "--output",
            str(directory / "generated_df_exchange_schedule.hpp"),
        ],
        check=True,
    )
    method = (ROOT / "src/methods/dft_method.cpp").read_text()
    prepared = (ROOT / "src/scf/fock_prepared.cpp").read_text()
    public = (ROOT / "src/scf/fock_prepared.hpp").read_text()
    occupations = (ROOT / "src/scf/initial_guess/density.cpp").read_text()
    ctor = public[
        public.index("  PreparedFockPlan(") : public.index("  ~PreparedFockPlan")
    ]
    matches = public[public.index("  bool matches(") : public.index("\n private:")]
    validation_start = prepared.index("    const auto reserved_rank =")
    validation_end = prepared.index("    for (const auto* term", validation_start)
    initializer_start = prepared.index(
        "  Impl(const core::System& system, const core::System* aux,"
    )
    initializer_end = prepared.index(
        "    validate_resolved_fock_build", initializer_start
    )
    fock_start = method.index("        fock_(system_")
    fock_end = method.index("\n        basis_(", fock_start)
    call_start = prepared.index(
        "      checked(create_cuda_density_fitting_jk_plan_from_source("
    )
    call_end = prepared.index("      cuda_df.reset", call_start)
    packing_start = prepared.index("      const bool coulomb_only_packing =")
    planning_start = prepared.index("      const auto plan_values =", packing_start)
    unit = r"""
#include <cassert>
#include <iostream>
#include "src/scf/density_fitting.cpp"
#include "src/molecule/basis.cpp"
#include "scf/fock_prepared.hpp"
#include "scf/types.hpp"
namespace generativeqc::scf::initial_guess {
"""
    unit += definition(
        occupations, "std::pair<std::size_t, std::size_t> spin_occupations("
    )
    unit += "\n}\nnamespace generativeqc::scf {\n"
    unit += "struct Capture { struct Impl; std::unique_ptr<Impl> impl_;\n"
    unit += ctor.replace("PreparedFockPlan(", "Capture(") + matches + "};\n"
    unit += r"""
struct Capture::Impl {
  core::System orbital;
  std::optional<core::System> auxiliary;
  int device_id;
  std::size_t requested_budget;
  unsigned retained_fitted_derivative_order;
  FockOccupiedProjectionReservation projection_reservation;
  bool external_integrals{};
  FockPreparationDiagnostic diagnostic;
  CudaDirectJkPlan* cuda_exact = nullptr;
"""
    unit += prepared[initializer_start:initializer_end]
    unit += "(void)aux; (void)retained_direct_derivative_order;\n"
    unit += prepared[validation_start:validation_end]
    unit += "diagnostic.strategy = strategy; }\n};\n"
    unit += definition(prepared, "PreparedFockPlan::PreparedFockPlan(").replace(
        "PreparedFockPlan::PreparedFockPlan(", "Capture::Capture("
    )
    unit += definition(prepared, "bool same_system(")
    unit += definition(prepared, "FockExecutionVariant execution_variant(")
    unit += definition(prepared, "bool PreparedFockPlan::matches(").replace(
        "PreparedFockPlan::matches(", "Capture::matches("
    )
    unit += r"""
DensityFittingTilePlan plan_values_for(FockOccupiedProjectionReservation reservation,
    DfPairStorage storage, std::size_t n, std::size_t a, std::size_t plan_budget,
    bool automatic = false, ResolvedFockBuild strategy = {}) {
  const auto reserved_rank = reservation.restricted_rank;
  FockPreparationDiagnostic diagnostic;
  diagnostic.variant.df_pair_storage_request =
      automatic ? DfPairStorageRequest::Automatic :
      storage == DfPairStorage::Dense ? DfPairStorageRequest::Dense :
      storage == DfPairStorage::SymmetricLower ? DfPairStorageRequest::SymmetricLower :
      DfPairStorageRequest::SymmetricLowerSingle;
  // These probes supply an explicit value allowance. Automatic rebalancing
  // has separate live-resource coverage in test_df_preparation_budget.py.
  DfResolvedBudget resolved;
  resolved.requested_bytes = resolved.total_bytes = resolved.value_bytes = plan_budget;
"""
    # Execute the production eligibility predicate captured by the planner.
    # A constant stand-in would lose the J-only physical/spin boundary.
    unit += prepared[packing_start:planning_start]
    unit += definition(prepared, "      const auto plan_values =") + ";\n"
    unit += "return plan_values(n,a,0); }\n"
    unit += r"""
DfValueStorageOptions allocated_storage;
bool allocated_retained = false;
generativeqc_status create_cuda_density_fitting_jk_plan_from_source(
    int, CudaDensityFittingIntegralSource**, std::size_t, std::size_t, std::size_t,
    const std::vector<double>&, double, std::size_t, std::size_t,
    CudaDensityFittingJkPlan**, std::vector<CudaDensityFittingMetricDiagnostic>&,
    std::string&, bool retain, DfValueStorageOptions storage, std::size_t automatic_rank) {
  assert(automatic_rank == 0); // KS owns Cocc; this is not the RHF factor owner.
  allocated_storage = storage;
  allocated_retained = retain;
  return GENERATIVEQC_STATUS_SUCCESS;
}
void checked(generativeqc_status status, const std::string&) {
  assert(status == GENERATIVEQC_STATUS_SUCCESS);
}
void allocate(const DensityFittingTilePlan& tiles) {
  int device=0;
  CudaDensityFittingIntegralSource* raw_source=nullptr;
  CudaDensityFittingJkPlan* raw_plan=nullptr;
  std::vector<double> metrics;
  std::size_t nbf=24, naux=24;
  ResolvedFockBuild strategy;
  FockPreparationDiagnostic diagnostic;
  std::string detail;
"""
    unit += prepared[call_start:call_end]
    unit += "}\n}\nusing namespace generativeqc;\n"
    unit += r"""
struct NativeKsExecutionPlan {
  unsigned spin_channels=1;
  bool range_exchange=false;
  bool density_fitted_range_exchange=false;
};
"""
    unit += definition(
        method,
        "scf::FockOccupiedProjectionReservation ks_fitted_projection_reservation(",
    )
    unit += definition(method, "std::size_t ks_fitted_provider_budget(")
    unit += r"""
unsigned ks_direct_derivative_order(const scf::ResolvedFockBuild&, generativeqc_backend) {
  assert(false); return 0;
}
std::size_t ks_provider_bytes(const core::System&, generativeqc_backend, unsigned) {
  assert(false); return 0;
}
scf::Capture method_prepare(core::System system_, NativeKsExecutionPlan execution_plan_,
    const scf::ResolvedFockBuild& strategy) {
  scf::ScfOptions options_;
  options_.resolved_fock_build = strategy;
  options_.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_AUTO;
  options_.density_fitting_memory_budget_bytes = 1U << 30;
  const auto backend = GENERATIVEQC_BACKEND_CUDA;
  const int device = 0;
  std::optional<core::System> auxiliary;
"""
    unit += (
        method[fock_start:fock_end]
        .replace("        fock_(", "  scf::Capture result(")
        .rstrip(",")
    )
    unit += ";\nreturn result; }\n"
    unit += r"""
int main(int argc, char** argv) {
  assert(argc == 2);
  const std::string mode=argv[1];
  using namespace scf;
  setenv("GENERATIVEQC_DF_EXCHANGE", "auto", 1);
  core::System system;
  system.electron_count=10;
  system.shells.resize(24); // 24 real s-shell AOs, integer restricted rank 5.
  ResolvedFockBuild strategy;
  strategy.backend=FockBackend::Cuda;
  strategy.spec.coulomb.approximation=FockApproximation::DensityFitted;
  strategy.spec.exchange.approximation=FockApproximation::DensityFitted;
  strategy.spec.spin=FockSpin::Restricted;
  strategy.spec.derivative_order=0;
  NativeKsExecutionPlan execution;
  const auto reserved=ks_fitted_projection_reservation(system,execution,strategy);
  assert(reserved.restricted_rank == 5);
  if (mode == "authorization") {
    auto rsh=execution;
    rsh.range_exchange=true;
    assert(ks_fitted_projection_reservation(system,rsh,strategy) == reserved);
    for (int kind=0; kind<5; ++kind) {
      auto e=execution;
      auto s=strategy;
      if (kind==0) e.spin_channels=2;
      if (kind==1) s.backend=FockBackend::Cpu;
      if (kind==2) s.spec.spin=FockSpin::Unrestricted;
      if (kind==3) s.spec.exchange.present=false;
      if (kind==4) s.spec.exchange.approximation=FockApproximation::Exact;
      assert(!ks_fitted_projection_reservation(system,e,s).restricted_rank);
    }
    for (int electrons : {-2,0,9,50}) {
      auto invalid=system;
      invalid.electron_count=electrons;
      bool rejected=false;
      try { (void)ks_fitted_projection_reservation(invalid,execution,strategy); }
      catch (const std::invalid_argument&) { rejected=true; }
      assert(rejected);
    }
    auto invalid=system;
    invalid.multiplicity=3;
    bool rejected=false;
    try { (void)ks_fitted_projection_reservation(invalid,execution,strategy); }
    catch (const std::invalid_argument&) { rejected=true; }
    assert(rejected);
  } else if (mode == "forwarding") {
    const auto method=method_prepare(system,execution,strategy);
    assert(method.impl_->projection_reservation == reserved);
    auto rsh=execution;
    rsh.range_exchange=true;
    const auto rsh_method=method_prepare(system,rsh,strategy);
    assert(rsh_method.impl_->projection_reservation == reserved);
    assert(rsh_method.impl_->retained_fitted_derivative_order == 1);
    assert(rsh_method.matches(system,nullptr,strategy,0,1U<<30,0,1,reserved));
    assert(!rsh_method.matches(system,nullptr,strategy,0,1U<<30,0,2,reserved));
    rsh.density_fitted_range_exchange=true;
    const auto fitted_rsh=method_prepare(system,rsh,strategy);
    assert(fitted_rsh.impl_->projection_reservation == reserved);
    assert(fitted_rsh.impl_->requested_budget == (1U<<29));
    assert(fitted_rsh.matches(system,nullptr,strategy,0,1U<<29,0,1,reserved));
    assert(!fitted_rsh.matches(system,nullptr,strategy,0,1U<<30,0,1,reserved));
    ScfOptions budgets;
    budgets.resolved_fock_build=strategy;
    for (std::size_t bytes : {0U,2U,3U,1025U}) {
      budgets.density_fitting_memory_budget_bytes=bytes;
      const auto primary=ks_fitted_provider_budget(budgets,rsh,false);
      const auto long_range=ks_fitted_provider_budget(budgets,rsh,true);
      assert(primary+long_range == bytes);
      assert(primary == bytes/2 && long_range == bytes-primary);
    }
    budgets.density_fitting_memory_budget_bytes=1;
    bool small_rejected=false;
    try { (void)ks_fitted_provider_budget(budgets,rsh,false); }
    catch (const std::bad_alloc&) { small_rejected=true; }
    assert(small_rejected); // Never reinterpret a zero half as automatic admission.
    assert(ks_fitted_provider_budget(budgets,execution,false) == 1);
    const Capture generic(system,nullptr,strategy,0);
    assert(!generic.impl_->projection_reservation.restricted_rank);
    assert(generic.matches(system,nullptr,strategy,0,0));
    generic.impl_->external_integrals = true;
    assert(!generic.matches(system,nullptr,strategy,0,0));
    generic.impl_->external_integrals = false;
    assert(!generic.matches(system,nullptr,strategy,0,0,0,0,reserved));
    assert(method.matches(system,nullptr,strategy,0,1U<<30,0,0,reserved));
    assert(!method.matches(system,nullptr,strategy,0,1U<<30));
    auto changed=system;
    changed.shells[0].primitives.push_back({1.0,0.5});
    assert(!method.matches(changed,nullptr,strategy,0,1U<<30,0,0,reserved));
    bool rejected=false;
    try { Capture wrong(system,nullptr,strategy,0,0,0,0,{6}); }
    catch (const std::invalid_argument&) { rejected=true; }
    assert(rejected);
    for (int kind=0; kind<4; ++kind) {
      auto s=strategy;
      auto invalid=system;
      if (kind==0) s.backend=FockBackend::Cpu;
      if (kind==1) s.spec.spin=FockSpin::Unrestricted;
      if (kind==2) s.spec.exchange.present=false;
      if (kind==3) invalid.electron_count=9;
      rejected=false;
      try { Capture wrong(invalid,nullptr,s,0,0,0,0,reserved); }
      catch (const std::invalid_argument&) { rejected=true; }
      assert(rejected);
    }
  } else if (mode == "coulomb_planning") {
    const auto dense = DfPairStorage::Dense;
    auto j_only = strategy;
    j_only.spec.exchange.present = false;
    constexpr std::size_t budget = 13685173124ULL;
    const auto pure = plan_values_for({},dense,768,3712,budget,true,j_only);
    assert(pure.value_storage.pairs == DfPairStorage::SymmetricLowerSingle);
    assert(pure.stores_full_three_center && pure.value_storage.rank_capacity == 0);
    assert(pure.automatic_rhf_rank == 0 && pure.peak_workspace_bytes <= budget);
    allocate(pure);
    assert(allocated_retained && allocated_storage.rank_capacity == 0);
    for (int kind = 0; kind < 4; ++kind) {
      auto excluded = j_only;
      if (kind == 0) excluded.spec.spin = FockSpin::Unrestricted;
      if (kind == 1) excluded.spec.coulomb.present = false;
      if (kind == 2) excluded.spec.coulomb.approximation = FockApproximation::Exact;
      if (kind == 3) excluded.spec.exchange.present = true;
      const auto plan = plan_values_for({},dense,768,3712,budget,true,excluded);
      assert(plan.value_storage.pairs == dense && !plan.stores_full_three_center);
    }
    const auto explicit_dense = plan_values_for({},dense,768,3712,budget,false,j_only);
    assert(explicit_dense.value_storage.pairs == dense);
    const auto roomy = plan_values_for({},dense,96,96,1U<<30,true,j_only);
    assert(roomy.value_storage.pairs == dense && roomy.stores_full_three_center);
    const auto bounded = plan_values_for({},dense,768,3712,4ULL<<30,true,j_only);
    assert(bounded.value_storage.pairs == dense && !bounded.stores_full_three_center);
    assert(bounded.peak_workspace_bytes <= (4ULL<<30));
  } else if (mode == "planning") {
    const auto packed=DfPairStorage::SymmetricLowerSingle;
    const auto generic=plan_values_for({},packed,24,24,1U<<30);
    const auto method=plan_values_for(reserved,packed,24,24,1U<<30);
    assert(generic.value_storage.rank_capacity==0);
    assert(method.value_storage.rank_capacity==5);
    assert(method.automatic_rhf_rank==0);
    allocate(method);
    assert(allocated_retained && allocated_storage.pairs==packed);
    assert(allocated_storage.rank_capacity==5);
    const auto dense=plan_values_for(reserved,DfPairStorage::Dense,24,24,1U<<30);
    assert(dense.value_storage.rank_capacity==0);
    // Packed-raw has no optional-U fallback and is not an admitted borrowed
    // force consumer. A KS request must preserve its old rank-zero budget.
    const auto raw=DfPairStorage::SymmetricLower;
    const auto raw_generic=plan_values_for({},raw,768,3712,0);
    const auto raw_method=plan_values_for({160},raw,768,3712,raw_generic.peak_workspace_bytes);
    assert(raw_method.value_storage.rank_capacity==0);
    assert(raw_method.peak_workspace_bytes==raw_generic.peak_workspace_bytes);
    allocate(raw_method);
    assert(allocated_storage.pairs==raw && allocated_storage.rank_capacity==0);
    // At this real planner boundary, complete U cannot fit but packed B can.
    const auto bounded=plan_values_for({160},packed,768,3712,13685173124ULL);
    assert(bounded.value_storage.rank_capacity==0 && bounded.stores_full_three_center);
    assert(bounded.peak_workspace_bytes<=13685173124ULL);
    allocate(bounded);
    assert(allocated_storage.rank_capacity==0); // Forward the resolved fallback.
    const auto roomy=plan_values_for({160},packed,768,3712,16421977600ULL);
    assert(roomy.value_storage.rank_capacity==160);
    bool rejected=false;
    try { (void)plan_values_for(reserved,packed,24,24,1); }
    catch (const DensityFittingBudgetError&) { rejected=true; }
    assert(rejected); // An infeasible budget is not silently enlarged.
    setenv("GENERATIVEQC_DF_EXCHANGE", "occupied", 1);
    rejected=false;
    try { (void)plan_values_for({160},packed,768,3712,13685173124ULL); }
    catch (const DensityFittingBudgetError&) { rejected=true; }
    assert(rejected); // Preserve explicit diagnostic admission.
  } else return 2;
}
"""
    source, executable = directory / "probe.cpp", directory / "probe"
    source.write_text(unit)
    native_cxx.build_executable(
        [source],
        executable,
        compile_args=(
            "-std=c++20",
            "-O0",
            "-ffunction-sections",
            "-fdata-sections",
            f"-I{ROOT}",
            f"-I{ROOT / 'include'}",
            f"-I{ROOT / 'src'}",
            f"-I{directory}",
        ),
        link_args=("-Wl,--gc-sections",),
    )
    return executable


@pytest.mark.parametrize(
    "mode", ["authorization", "forwarding", "planning", "coulomb_planning"]
)
def test_method_owned_projection_reservation(
    reservation_probe: Path, mode: str
) -> None:
    subprocess.run([str(reservation_probe), mode], check=True)
