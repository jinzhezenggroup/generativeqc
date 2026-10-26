"""The native publication gate must not certify incomplete failed-stage counts."""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def publication_binary(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: "NativeCxx"
) -> Path:
    source = (ROOT / "src/scf/cuda_rhf.cpp").read_text()
    begin = source.index("    result.precision.operator_work_counters_valid")
    end = source.index("    const std::size_t density_stride", begin)
    publication = source[begin:end]
    folder = tmp_path_factory.mktemp("precision-publication")
    path = folder / "probe.cpp"
    path.write_text(
        r"""#include "scf/types.hpp"
#include <cstdint>
#include <cstdlib>
#include <vector>
int main(int argc, char** argv) {
  if(argc != 3) return 90;
  const int scenario = std::atoi(argv[1]);
  const bool screened_reference_iterations = std::atoi(argv[2]) != 0;
  const bool failed = scenario == 1, audited = scenario == 2;
  const std::size_t system = 0;
  const bool precision_item_mixed = false;
  const bool scf_force_ready_state = false, reuse_converged_fock = false;
  const std::vector<std::uint32_t> host_iterations{5};
  const std::vector<std::uint32_t> host_pre_refinement_iterations{screened_reference_iterations ? 7U : 0U};
  const std::vector<std::uint64_t> host_mixed_item_census{0};
  const std::vector<double> host_mixed_item_threshold{0};
  const std::vector<std::uint8_t> host_failed{static_cast<std::uint8_t>(failed)};
  const std::vector<std::uint8_t> host_converged{static_cast<std::uint8_t>(scenario == 0)};
  const std::vector<std::uint8_t> host_final_audit_mask{static_cast<std::uint8_t>(audited)};
  const std::vector<std::uint8_t> host_final_fock_reuse_mask{0};
  const struct { double item_budget_error; } requested_precision_policy{0};
  const generativeqc::scf::IncrementalDirectJkPolicy incremental_direct_jk_policy{};
  const bool incremental_iteration_enabled = false;
  const std::uint64_t incremental_rebuild_interval = 0;
  const std::vector<double> host_incremental_max_abs_delta_density{0.0};
  const std::vector<std::uint64_t> host_incremental_full_build_count{0};
  const std::vector<std::uint64_t> host_incremental_delta_build_count{0};
  const bool bounded_direct_streaming = false;
  const struct { std::vector<std::size_t> system_shell_quartet_offsets; } host{{0, 0}};
  const std::vector<std::uint64_t> host_incremental_full_admitted_shell_quartets{0};
  const std::vector<std::uint64_t> host_incremental_full_admitted_quartet_tiles{0};
  const std::vector<std::uint64_t> host_incremental_delta_admitted_shell_quartets{0};
  const std::vector<std::uint64_t> host_incremental_delta_admitted_quartet_tiles{0};
  generativeqc::scf::ScfResult result;
"""
        + publication
        + r"""
  // A final eigensolver can reject an item after its physical Fock was built.
  // Final-state iteration/status counters cannot reconstruct that failed work.
  if(bool(result.precision.operator_work_counters_valid) != !failed) return 1;
  // Screened proposals still execute FP64 Focks and belong to strict work.
  if(result.precision.strict_stage_fock_builds != (screened_reference_iterations ? 12U : 5U)) return 2;
  if(result.precision.mixed_stage_fock_builds != 0) return 5;
  if(result.precision.refinement_iterations != (screened_reference_iterations ? 5U : 0U)) return 6;
  if(!failed && result.precision.post_scf_fock_builds != (audited ? 2U : scenario == 0 ? 1U : 0U)) return 3;
  if(result.precision.final_residual_audits != (audited ? 1U : 0U)) return 4;
  return 0;
}
"""
    )
    binary = folder / "probe"
    native_cxx.build_executable(
        [path],
        binary,
        compile_args=[
            "-std=c++20",
            "-O0",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
        ],
        compile_timeout=45,
    )
    return binary


@pytest.mark.parametrize(
    "scenario",
    [0, 1, 2, 3],
    ids=[
        "success",
        "late-numerical-failure",
        "audit-rejection",
        "iteration-exhaustion",
    ],
)
@pytest.mark.parametrize("screened", [False, True])
def test_precision_work_validity_tracks_complete_item_accounting(
    publication_binary: Path, scenario: int, screened: bool
) -> None:
    result = subprocess.run(
        [str(publication_binary), str(scenario), str(int(screened))],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, (scenario, result.returncode, result.stderr)
