"""Host-only boundaries for default-auto screened cold Global-Hybrid reuse."""

import subprocess
from pathlib import Path

from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_unset_selection_reaches_the_bounded_auto_policy() -> None:
    """Unset and explicit auto share admission; explicit 0 skips all setup."""
    solver = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    prepare = solver[solver.index("  void prepare_phase_source(bool cold)") :]
    prepare = prepare[: prepare.index("phase_source = std::make_unique")]
    assert "if (!selection) return;" not in prepare
    assert "const auto requested = direct_jk_source_selection(selection);" in prepare
    assert "if (requested == DirectJkSourceSelection::disabled) return;" in prepare
    assert "requested == DirectJkSourceSelection::automatic" in prepare
    assert "direct_jk_source_auto_refusal(" in prepare


def test_warm_energy_handoff_is_counted_and_preserves_the_final_state() -> None:
    """A resident energy is not an incumbent warm baseline, even at identical D."""
    solver = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    finish = solver[
        solver.index("  bool finish() {") : solver.index("  void rebase_warm_energy()")
    ]
    assert finish.index("retire_phase_source(!is_failed)") < finish.index(
        "rebase_warm_energy()"
    )
    assert "auditing_phase && output.converged && warm_updates" in finish
    assert "clear_warm_state();" in finish
    assert (
        "use_warm && !device_chunk_mode && !phase_source && std::isfinite(warm_energy)"
        in " ".join(solver.split())
    )
    rebase = solver[
        solver.index("  void rebase_warm_energy()") : solver.index(
            "  void enqueue_range_correction("
        )
    ]
    assert "++output.fock_builds;" in rebase
    assert "++output.precision.post_scf_fock_builds;" in rebase
    assert "++movement.warm_energy_rebase_fock_actions;" in rebase
    assert "PrecisionWorkEventKind::PostScfFock" in rebase
    assert "PrecisionWorkPhase::Finalization" in rebase
    assert "warm_energy = rebased.electronic_energy;" in rebase
    assert "warm_energy_correction = rebased.electronic_energy_correction;" in rebase
    for forbidden in (
        "assemble_fock(",
        "stage_xc(",
        "++output.iterations",
        "output.energy =",
        "output.energy_change =",
        "output.converged =",
        "options.energy_tolerance =",
        "options.density_tolerance =",
    ):
        assert forbidden not in rebase


def test_native_cold_seed_provenance_is_not_an_external_density() -> None:
    """Only native preparation, never a warm projection, broadens seed eligibility."""
    header = (ROOT / "src/dft/cuda_ks.hpp").read_text()
    solver = " ".join((ROOT / "src/dft/cuda_ks.cpp").read_text().split())
    owner = " ".join((ROOT / "src/methods/dft_method.cpp").read_text().split())
    assert "CudaKsSeedOrigin seed_origin = CudaKsSeedOrigin::External" in header
    assert (
        "(!input || seed_origin == CudaKsSeedOrigin::NativeColdGuess) && !use_warm"
        in solver
    )
    assert "prepared ? dft::CudaKsSeedOrigin::NativeColdGuess" in owner
    assert (
        "!attempt && !reuse && preliminary_seeds[i] ? dft::CudaKsSeedOrigin::NativeColdGuess"
        in owner
    )
    assert solver.index("direct_jk_source_auto_refusal(") < solver.index(
        "phase_source = std::make_unique<DirectJkSourceLease>"
    )


def test_screened_cold_candidates_preserve_rhf_policy(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Resource eligibility is explicit; neither a cap nor an iteration limit predicts speed."""
    source = tmp_path / "policy.cpp"
    source.write_text(r"""
#include <limits>
#include "scf/cuda/direct_jk_source_policy.hpp"
using namespace generativeqc::scf::cuda_execution;
constexpr auto screened = DirectJkSourceAutoPolicy::screened_cold;
constexpr auto generic = DirectJkSourceAutoPolicy::generic_f;
constexpr auto automatic = DirectJkSourceSelection::automatic;
static_assert(direct_jk_source_selection(nullptr)==automatic);
static_assert(direct_jk_source_selection("auto")==automatic);
static_assert(direct_jk_source_selection("0")==DirectJkSourceSelection::disabled);
static_assert(direct_jk_source_selection("1")==DirectJkSourceSelection::forced);
static_assert(!direct_jk_source_auto_refusal(768,800,100,2,true,false,screened,1e-12));
static_assert(direct_jk_source_auto_refusal(768,800,100,2,true,false));
static_assert(direct_jk_source_auto_refusal(768,800,100,2,false,false,screened,1e-12));
static_assert(direct_jk_source_auto_refusal(768,800,100,2,true,false,screened,0.0));
static_assert(direct_jk_source_auto_refusal(768,800,100,2,true,false,screened,-1e-12));
static_assert(direct_jk_source_auto_refusal(63,128,8,2,true,false,screened,1e-12));
static_assert(direct_jk_source_auto_refusal(64,127,8,2,true,false,screened,1e-12));
static_assert(direct_jk_source_auto_refusal(64,128,7,2,true,false,screened,1e-12));
static_assert(!direct_jk_source_auto_refusal(64,128,8,2,true,false,screened,1e-12));
static_assert(!direct_jk_source_auto_refusal(64,128,8,0,true,false,screened,1e-12));
static_assert(!direct_jk_source_auto_refusal(64,128,8,1,true,false,screened,1e-12));
static_assert(!direct_jk_source_auto_refusal(64,128,8,3,true,true,screened,0.0));
static_assert(direct_jk_source_auto_refusal(64,128,8,3,true,false,screened,1e-12));
static_assert(direct_jk_source_auto_refusal(64,128,8,4,true,true,screened,1e-12));
static_assert(!direct_jk_source_auto_refusal(64,128,8,3,true,true));
static_assert(direct_jk_source_value_ceiling(automatic,screened,2)==(16ULL<<30));
static_assert(direct_jk_source_value_ceiling(automatic,screened,3)==(8ULL<<30));
static_assert(direct_jk_source_value_ceiling(automatic,generic,2)==(8ULL<<30));
static_assert(direct_jk_source_value_ceiling(DirectJkSourceSelection::forced,screened,3)
              ==(16ULL<<30));
constexpr auto maximum = std::numeric_limits<std::size_t>::max();
constexpr std::size_t values = 14985773728ULL;
constexpr std::size_t required = 123456, overhead = 654321;
static_assert(direct_jk_source_allowance(768,100,required,overhead,maximum,16ULL<<30)>=values);
static_assert(direct_jk_source_allowance(768,100,required,overhead,maximum,8ULL<<30)<values);
static_assert(direct_jk_source_allowance(768,100,required,overhead,
                                        required+overhead+values-1,16ULL<<30)<values);
static_assert(direct_jk_source_allowance(768,100,maximum,1,maximum,16ULL<<30)==0);
int main() {
  return !direct_jk_source_auto_refusal(768,800,100,2,true,false,screened,
                                       std::numeric_limits<double>::quiet_NaN());
}
""")
    native_cxx.build_executable(
        [source], tmp_path / "policy", compile_args=("-std=c++20", f"-I{ROOT / 'src'}")
    )
    subprocess.run([str(tmp_path / "policy")], check=True, timeout=10)
