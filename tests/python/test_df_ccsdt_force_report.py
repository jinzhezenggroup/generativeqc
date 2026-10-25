"""Execute the benchmark's complete JSON contract with real native diagnostics.

The schema below is an independent public-data oracle, not extracted from the
formatter. Columns are JSON key, native input member, and availability (W means
attempt work; M means metadata or an independently complete measurement).
No scientific endpoint is executed by this CPU-only test.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]

# Real native member types determine sentinel conversion, independently of JSON
# serialization. Unique magnitudes expose swapped member mappings; bool inputs
# are swept separately because two booleans cannot carry unique magnitudes.
SCHEMA = """
total_energy energy M
reference_energy reference_energy M
reference_energy_change reference_energy_change M
reference_density_rms reference_density_rms M
reference_iterations reference_iterations W
correlation_energy correlation_energy M
triples_energy triples_energy M
lambda_residual lambda.independent_residual_norm M
z_residual orbital.orbital_residual M
stationarity orbital.maximum_stationarity M
numeric_capacity_bytes numeric_capacity_bytes W
native_seconds total_seconds M
resident_jk_retry_seconds orbital.resident_jk_retry_seconds M
reference_seconds primal.reference_seconds W
source_seconds primal.problem_seconds W
df_source_scans primal.source_scans W
df_source_reads primal.source_reads W
df_source_values primal.source_values W
df_transform_fmas primal.transform_fmas W
df_transform_stages primal.transform_stages W
df_block_gemms primal.mo_blocks W
df_cuda_transform_calls primal.cuda_transform_calls W
ccsd_seconds primal.solver_seconds W
ccsd_matrix_gemm solver.df_matrix_gemm M
ccsd_mixed_precision solver.df_mixed_precision M
ccsd_mixed_precision_refused solver.df_mixed_precision_refused M
ccsd_mixed_evaluations solver.df_mixed_evaluations W
ccsd_fp32_gemm_calls solver.df_fp32_gemm_calls W
ccsd_fp32_gemm_summands solver.df_fp32_gemm_summands W
ccsd_precision_cast_elements solver.df_precision_cast_elements W
ccsd_precision_switches solver.df_precision_switches W
ccsd_mixed_pair_projection_calls solver.df_mixed_pair_projection_calls W
ccsd_mixed_pair_projection_bytes solver.df_mixed_pair_projection_bytes W
ccsd_strict_batch_size solver.df_strict_auxiliary_batch_size W
ccsd_replay_matrix_gemm solver.df_replay_matrix_gemm M
ccsd_gemm_calls solver.df_gemm_calls W
ccsd_gemm_summands solver.df_gemm_summands W
ccsd_packing_bytes solver.df_packing_bytes W
ccsd_provider_capacity solver.df_provider_capacity_bytes W
ccsd_q_batch_size solver.df_auxiliary_batch_size M
ccsd_q_tiles solver.df_auxiliary_tiles W
ccsd_q_slices solver.df_auxiliary_slices W
ccsd_q_operations solver.df_virtual_operations W
ccsd_accumulation_calls solver.df_accumulation_calls W
ccsd_accumulation_bytes solver.df_accumulation_bytes W
ccsd_contraction_terms solver.df_contraction_terms W
ccsd_occupied_pairs solver.df_occupied_pairs M
ccsd_pair_resource_refused solver.df_pair_resource_refused M
ccsd_pair_initial_symmetry_refused solver.df_pair_initial_symmetry_refused M
ccsd_pair_evaluations solver.df_pair_evaluations W
ccsd_pair_refusals solver.df_pair_refusals W
ccsd_pair_projection_calls solver.df_pair_projection_calls W
ccsd_pair_projection_bytes solver.df_pair_projection_bytes W
ccsd_pair_geometry_elements solver.df_pair_geometry_elements W
ccsd_pair_capacity_bytes solver.df_pair_capacity_bytes W
ccsd_pair_binding_host_bytes solver.df_pair_binding_host_bytes W
ccsd_evaluations solver.iteration_graph_calls W
ccsd_replay_evaluations solver.replay_graph_calls W
ccsd_capacity solver.numeric_capacity_bytes W
ccsd_device_bytes solver.owned_device_bytes W
ccsd_setup_h2d_bytes solver.setup_h2d_bytes W
denominator_identity solver.denominator_identity M
derived_d2_iteration_evaluations solver.derived_d2_iteration_evaluations W
ccsd_iterations solver.iterations W
ccsd_replay_r1_max solver.replay_r1_max M
ccsd_replay_r2_max solver.replay_r2_max M
ccsd_diis_seconds solver.diis_seconds W
ccsd_iteration_seconds solver.iteration_seconds W
ccsd_replay_seconds solver.replay_seconds W
ccsd_update_seconds solver.update_seconds W
ccsd_diis_restarts solver.diis_restarts W
ccsd_packed_diis solver.packed_diis M
ccsd_packed_diis_refused solver.packed_diis_refused M
ccsd_diis_disabled_after_packing_refusal solver.diis_disabled_after_packing_refusal M
ccsd_diis_history_capacity_bytes solver.diis_history_capacity_bytes W
ccsd_diis_conversion_bytes solver.diis_conversion_bytes W
ccsd_diis_metric_weight_terms solver.diis_metric_weight_terms W
ccsd_diis_pack_calls solver.diis_pack_calls W
ccsd_diis_maximum_pair_asymmetry solver.diis_maximum_pair_asymmetry M
ccsd_diis_gram_calls solver.diis_gram_calls W
ccsd_diis_coefficient_calls solver.diis_coefficient_calls W
ccsd_diis_combine_calls solver.diis_combine_calls W
ccsd_diis_insert_bytes solver.diis_history_insert_bytes W
ccsd_diis_shift_bytes solver.diis_history_shift_bytes W
ccsd_diis_dot_terms solver.diis_residual_dot_terms W
ccsd_diis_gram_updates solver.diis_gram_updates W
ccsd_diis_combine_terms solver.diis_combine_terms W
triples_seconds triples_seconds W
lambda_seconds lambda_seconds W
source_response_seconds source_response_seconds W
orbital_seconds orbital_seconds W
jk_actions orbital.jk_actions W
resident_jk_bytes orbital.resident_jk_bytes W
resident_jk_values orbital.resident_jk_values W
resident_jk_actions orbital.resident_jk_actions W
resident_jk_value_reads orbital.resident_jk_value_reads W
resident_jk_setup_seconds orbital.resident_jk_setup_seconds W
host_jk_bytes orbital.host_jk_bytes W
host_jk_capacity_bytes orbital.host_jk_capacity_bytes W
host_jk_actions orbital.host_jk_actions W
host_jk_h2d_bytes orbital.host_jk_h2d_bytes W
host_jk_setup_seconds orbital.host_jk_setup_seconds W
resident_jk_seconds orbital.resident_jk_seconds W
resident_jk_device_seconds orbital.resident_jk_device_seconds W
z_iterations orbital.orbital_response.iterations W
z_operator_actions orbital.orbital_response.operator_actions W
z_preconditioner_actions orbital.orbital_response.preconditioner_actions W
z_df_preconditioned orbital.df_preconditioned M
z_preconditioner_fallback orbital.preconditioner_fallback M
z_preconditioner_setup_seconds orbital.preconditioner_setup_seconds W
z_preconditioner_capacity_bytes orbital.preconditioner_capacity_bytes W
z_preconditioner_planned_contraction_terms orbital.preconditioner_contraction_terms W
z_recycled_guess orbital.recycled_guess M
z_recycle_published orbital.recycle_published M
z_recycle_capacity_bytes orbital.recycle_capacity_bytes W
orbital_setup_seconds orbital.setup_seconds W
orbital_reference_audit_seconds orbital.reference_audit_seconds W
orbital_weights_seconds orbital.weights_seconds W
orbital_solve_seconds orbital.solve_seconds W
orbital_independent_audit_seconds orbital.independent_audit_seconds W
orbital_one_electron_seconds orbital.one_electron_seconds W
orbital_two_electron_seconds orbital.two_electron_seconds W
orbital_bilinear_derivative_used orbital.bilinear_derivative_used M
orbital_symmetric_polarization_used orbital.symmetric_polarization_used M
orbital_derivative_census_measured orbital.derivative_census_measured M
orbital_derivative_quartet_visits orbital.derivative_quartet_visits W
orbital_derivative_jet_evaluations orbital.derivative_jet_evaluations W
orbital_jk_seconds orbital.jk_seconds W
orbital_jk_device_seconds orbital.jk_device_seconds W
orbital_h2d_bytes orbital.h2d_bytes W
orbital_d2h_bytes orbital.d2h_bytes W
orbital_synchronizations orbital.synchronizations W
orbital_screened_jk_seconds orbital.screened_jk_seconds W
orbital_screened_residual orbital.screened_residual M
orbital_requested_screening orbital.requested_screening M
orbital_applied_screening orbital.applied_screening M
orbital_screened_jk_actions orbital.screened_jk_actions W
orbital_jk_census_actions orbital.jk_census_actions W
orbital_exact_refinements orbital.exact_refinements W
orbital_screened_iterations orbital.screened_iterations W
orbital_screened_operator_actions orbital.screened_operator_actions W
orbital_jk_quartet_visits orbital.jk_quartet_visits W
orbital_jk_eri_evaluations orbital.jk_eri_evaluations W
orbital_jk_timing_measured orbital.jk_timing_measured M
orbital_linear_screening_available orbital.linear_screening_available M
orbital_screened_converged orbital.screened_converged M
orbital_derivative_passes orbital.derivative_passes W
orbital_shell_derivative_passes orbital.shell_derivative_passes W
orbital_generic_derivative_passes orbital.generic_derivative_passes W
hessian_elements orbital.explicit_hessian_elements M
source_weight_values source_weight_values W
metric_weight_values metric_weight_values W
triples_work triples.contraction_summands W
triples_occupied_tiles triples.occupied_tiles W
triples_panel_capacity triples.panel_capacity W
triples_panel_calls triples.panel_gemms W
triples_moment_products triples.moment_gemms W
triples_epilogue_kernels triples.epilogue_kernels W
triples_reduction_kernels triples.reduction_kernels W
triples_reserved_bytes triples.workspace_bytes W
triples_arena_bytes triples.arena_bytes W
triples_w_resource_fallback triples.resource_fallback M
triples_w_storage_bits triples.w_contraction_storage_bits W
triples_w_compute_bits triples.w_contraction_compute_bits W
triples_w_accumulation_bits triples.w_contraction_accumulation_bits W
triples_fp64_gemms triples.fp64_gemms W
triples_fp32_gemms triples.fp32_gemms W
triples_precision_cast_elements triples.precision_cast_elements W
triples_gap_parallel triples_gap.parallel M
triples_gap_requested triples_gap.requested M
triples_gap_kernels triples_gap.kernels W
triples_gap_workspace_bytes triples_gap.workspace_bytes W
triples_gap_materialized_elements triples_gap.materialized_elements W
triples_gap_value_reads triples_gap.value_reads W
triples_gap_value_writes triples_gap.value_writes W
triples_gap_reduction_summands triples_gap.reduction_summands W
triples_scalar_fusion_requested triples_scalar_fusion.requested M
triples_scalar_fusion_selected triples_scalar_fusion.selected M
triples_scalar_fusion_resource_fallback triples_scalar_fusion.resource_fallback M
triples_scalar_fusion_tiles triples_scalar_fusion.tiles W
triples_scalar_fusion_kernels triples_scalar_fusion.kernels W
triples_scalar_fusion_avoided_kernels triples_scalar_fusion.avoided_kernels W
triples_scalar_fusion_workspace_bytes triples_scalar_fusion.workspace_bytes W
triples_scalar_fusion_value_reads triples_scalar_fusion.value_reads W
triples_scalar_fusion_value_writes triples_scalar_fusion.value_writes W
triples_scalar_fusion_arithmetic_ops triples_scalar_fusion.arithmetic_ops W
fock_response_work triples_fock.contraction_summands W
fock_response_w_gemms triples_fock.w_gemms W
fock_response_w_copy_calls triples_fock.w_copy_calls W
fock_response_w_copy_bytes triples_fock.w_copy_bytes W
fock_response_panel_gemms triples_fock.panel_gemms W
fock_response_page_builds triples_fock.page_builds W
fock_response_vector_cubes triples_fock.vector_cubes W
lambda_work lambda.df_contraction_terms W
lambda_matrix_gemm lambda.df_matrix_gemm M
lambda_batch_size lambda.df_auxiliary_batch_size M
lambda_batches lambda.df_auxiliary_batches W
lambda_gemm_calls lambda.df_gemm_calls W
lambda_gemm_summands lambda.df_gemm_summands W
lambda_packing_output_bytes lambda.df_packing_output_bytes W
lambda_provider_allowance lambda.df_provider_allowance_bytes W
lambda_shared_scratch_bytes lambda.df_shared_scratch_bytes W
lambda_scratch_avoided_bytes lambda.df_scratch_avoided_bytes W
lambda_available_device_bytes lambda.df_available_device_bytes W
lambda_device_limit_bytes lambda.df_device_limit_bytes W
lambda_core_reuse lambda.df_core_reuse M
lambda_core_reuse_bytes lambda.df_core_reuse_bytes W
lambda_core_reuse_preparations lambda.df_core_reuse_preparations W
lambda_core_reuse_actions lambda.df_core_reuse_actions W
lambda_audit_matrix lambda.df_audit_matrix_gemm M
lambda_primal_matrix lambda.df_primal_matrix_gemm M
lambda_mixed_precision lambda.df_mixed_precision M
lambda_mixed_precision_refused lambda.df_mixed_precision_refused M
lambda_mixed_storage_bytes lambda.df_mixed_storage_bytes W
lambda_mixed_input_offset_bytes lambda.df_mixed_input_offset_bytes W
lambda_mixed_actions lambda.df_mixed_actions W
lambda_strict_actions lambda.df_strict_actions W
lambda_precision_switches lambda.df_precision_switches W
lambda_fp32_gemm_calls lambda.df_fp32_gemm_calls W
lambda_fp32_gemm_summands lambda.df_fp32_gemm_summands W
lambda_precision_cast_elements lambda.df_precision_cast_elements W
lambda_mixed_iterations lambda.df_mixed_iterations W
lambda_strict_iterations lambda.df_strict_iterations W
lambda_audit_arena_bytes lambda.df_audit_arena_bytes W
lambda_reduced lambda.df_auxiliary_reduction M
lambda_preparations lambda.df_preparation_calls W
lambda_reduced_actions lambda.df_reduced_actions W
lambda_actions lambda.operator_actions W
lambda_iterations lambda.iterations W
lambda_auxiliary_visits lambda.df_auxiliary_slices W
lambda_kernels lambda.df_generated_kernels W
lambda_capacity lambda.numeric_capacity_bytes W
lambda_device_bytes lambda.owned_device_bytes W
lambda_h2d_bytes lambda.h2d_bytes W
lambda_d2h_bytes lambda.d2h_bytes W
global_stability_certified orbital.global_stability_certified M
"""
FIELDS = [line.split() for line in SCHEMA.strip().splitlines()]


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        assert key not in result, f"duplicate report key: {key}"
        result[key] = value
    return result


@pytest.fixture(scope="module")
def report_probe(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> Path:
    directory = tmp_path_factory.mktemp("df-force-report")
    # The real df_mo_response.hpp declares callbacks with an opaque CUDA stream.
    # No CUDA API is called or implemented. Keep host-only CI toolkit independent
    # while compiling every actual native diagnostic declaration unchanged.
    (directory / "cuda_runtime_api.h").write_text(
        "#pragma once\ntypedef struct CUstream_st* cudaStream_t;\n"
    )
    source = directory / "probe.cpp"
    assignments = "\n".join(
        f'  seed(result.{member}, {index}, bit, zero, "{key}");'
        for index, (key, member, _) in enumerate(FIELDS)
    )
    source.write_text(
        r"""
#include "df_ccsdt_force_report.hpp"
#include <cstdlib>
#include <iostream>
#include <limits>
#include <type_traits>

// Input receipt uses native scalar types, independent of the report writer.
// It describes only assigned inputs, never report availability or key selection.
template<class T>
void seed(T& target, unsigned index, unsigned bit, bool zero, const char* key) {
  // Measured zero is distinct from an absent phase. Synthetic signed values
  // test transparent serialization, not physical convergence or timing.
  if constexpr (std::is_same_v<T, bool>) target = (index >> bit) & 1;
  else if constexpr (std::is_floating_point_v<T>) {
    const double magnitude = 1000.0 + index + 0.12345678901234567;
    target = zero ? 0.0 : ((index + bit) & 1 ? -magnitude : magnitude);
  } else if constexpr (std::is_integral_v<T> && sizeof(T) >= 8)
    target = zero ? 0 : std::numeric_limits<T>::max() - 1000 - index;
  else target = zero ? 0 : 1000 + index;
  std::cerr << key << ' ' << std::setprecision(17) << target << '\n';
}
int main(int argc, char** argv) {
  using namespace generativeqc::benchmarks;
  if (argc != 8) return 3;
  const int stage = std::atoi(argv[1]), mask = std::atoi(argv[2]);
  const int policy = std::atoi(argv[3]), bit = std::atoi(argv[4]);
  const bool zero = std::atoi(argv[5]);
  const int mode = std::atoi(argv[6]);
  generativeqc::methods::detail::DFCCSDTResult result{};
"""
        + assignments
        + r"""
  result.recycling_discarded_primal_attempt = mask & 1;
  result.orbital.resident_jk_discarded_attempt = mask & 2;
  result.solver.df_mixed_program_hash = zero ? nullptr : "ccsd-mixed-\"quote\"-\\slash";
  result.lambda.shared_program_hash = zero ? nullptr : "lambda-shared-\"quote\"-\\slash";
  result.lambda.independent_program_hash = zero ? nullptr : "lambda-independent-\"quote\"-\\slash";
  result.lambda.mixed_program_hash = zero ? nullptr : "lambda-mixed-\"quote\"-\\slash";
  result.lambda.audit_schedule_hash = zero ? nullptr : "lambda-audit-\"quote\"-\\slash";
  result.lambda.core_reuse_plan_hash = zero ? nullptr : "lambda-core-\"quote\"-\\slash";
  result.triples.w_candidate_identity = zero ? "" : "candidate-\"c\"";
  result.triples.w_codegen_precision_schedule_identity = zero ? "" : "precision-\\p";
  result.triples_gap.schedule = zero ? "" : "gap-\"g\"";
  result.triples_scalar_fusion.schedule = zero ? "" : "fusion-\\f";
  if (!zero) result.forces = {0.12345678901234566, -2.25, 3.5, -4.75, 5.125, -6.0};
  // Separate codewords also distinguish every requested option from the native
  // bool inputs (whose indices are below 236). Shared true/false patterns would
  // let an incorrect option-to-field mapping pass this contract.
  const auto requested = [bit](unsigned code) { return bool((code >> bit) & 1U); };
  DFCCSDTReportOptions options{31, 47, stage, requested(240), requested(241), requested(242),
      1e-12, 1e-11, 37, 41, std::nullopt, 0.00012345678901234567, zero ? 6ULL : 17ULL + bit, 9.875,
      requested(243), requested(244), requested(245), requested(246), requested(247), requested(248)};
  if (policy != 2) options.ccsd_mixed_precision = bool(policy);
  if (mode == 1 || mode == 3) {
    try { publish_df_ccsdt_force_report(argv[7], result, options); }
    catch (const std::runtime_error& error) { std::cout << error.what(); return 0; }
    return mode == 3 ? 0 : 4;
  }
  if (mode == 2) result.energy = std::numeric_limits<double>::quiet_NaN();
  write_df_ccsdt_force_report(std::cout, result, options);
}
"""
    )
    binary = directory / "report"
    required_native_cxx.build_executable(
        [source],
        binary,
        compile_args=(
            "-std=c++20",
            "-I" + str(directory),
            "-I" + str(ROOT / "benchmarks"),
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
        ),
    )
    return binary


def _run(
    probe: Path,
    stage: int = 0,
    mask: int = 0,
    policy: int = 2,
    bit: int = 0,
    zero: int = 0,
    mode: int = 0,
    path: str | Path = "unused",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(probe), *map(str, (stage, mask, policy, bit, zero, mode, path))],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )


@pytest.mark.parametrize("stage", range(3))
@pytest.mark.parametrize("mask", range(4))
@pytest.mark.parametrize("policy", range(3))
def test_complete_report(
    report_probe: Path, stage: int, mask: int, policy: int
) -> None:
    for zero in (0, 1):
        # Eight bit planes distinguish all bool inputs, including false flags.
        for bit in range(8):
            completed = _run(report_probe, stage, mask, policy, bit, zero)
            actual = json.loads(completed.stdout, object_pairs_hook=_unique_object)
            inputs = {
                key: json.loads(value)
                for key, value in (
                    line.split() for line in completed.stderr.splitlines()
                )
            }
            expected = {
                key: None if mask and kind == "W" else inputs[key]
                for key, _, kind in FIELDS
            }
            expected.update(
                lifecycle_stage=("cold", "warm", "moved-warm")[stage],
                nbf=31,
                naux=47,
                forces_requested=(240 >> bit) & 1,
                lambda_reduction_requested=(241 >> bit) & 1,
                matrix_gemm_requested=(242 >> bit) & 1,
                reference_energy_tolerance=1e-12,
                reference_density_tolerance=1e-11,
                lambda_true_residual_interval=None if mask else 37,
                z_true_residual_interval=41,
                ccsd_mixed_precision_requested=None if policy == 2 else policy,
                ccsd_mixed_precision_policy=("fp64", "mixed", "auto")[policy],
                ccsd_mixed_switch_residual=0.00012345678901234567,
                ccsd_diis_history=6 if zero else 17 + bit,
                previous_endpoint_seconds=9.875,
                resident_jk_discarded_attempt=int(bool(mask & 2)),
                recycling_discarded_primal_attempt=int(bool(mask & 1)),
                triples_w_fp32_requested=(243 >> bit) & 1,
                lambda_core_reuse_requested=(244 >> bit) & 1,
                lambda_audit_matrix_requested=(245 >> bit) & 1,
                lambda_primal_matrix_requested=(246 >> bit) & 1,
                lambda_serial_scratch_reuse_requested=(247 >> bit) & 1,
                lambda_mixed_precision_requested=(248 >> bit) & 1,
                triples_w_candidate_identity="" if zero else 'candidate-"c"',
                triples_w_precision_schedule_identity="" if zero else "precision-\\p",
                triples_gap_schedule="" if zero else 'gap-"g"',
                triples_scalar_fusion_schedule="" if zero else "fusion-\\f",
                forces=[]
                if zero
                else [0.12345678901234566, -2.25, 3.5, -4.75, 5.125, -6.0],
            )
            for key, prefix in {
                "ccsd_mixed_program_hash": "ccsd-mixed",
                "lambda_shared_program_hash": "lambda-shared",
                "lambda_independent_program_hash": "lambda-independent",
                "lambda_mixed_program_hash": "lambda-mixed",
                "lambda_audit_schedule_hash": "lambda-audit",
                "lambda_core_reuse_plan_hash": "lambda-core",
            }.items():
                expected[key] = "" if zero else prefix + '-"quote"-\\slash'
            assert actual == expected
            assert all(
                type(value) in (int, float, str, list, type(None))
                for value in actual.values()
            )


def test_publication_failures(report_probe: Path, tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "report.json"
    assert (
        _run(report_probe, mode=1, path=missing).stdout
        == "cannot open completed force output"
    )
    # Linux full device accepts open, but rejects flush/close. No GPU is used.
    if Path("/dev/full").exists():
        assert (
            _run(report_probe, mode=1, path="/dev/full").stdout
            == "failed publishing completed force output"
        )


def test_successful_file_publication(report_probe: Path, tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    _run(report_probe, mode=3, path=path)
    assert json.loads(path.read_text(), object_pairs_hook=_unique_object) == json.loads(
        _run(report_probe).stdout, object_pairs_hook=_unique_object
    )


def test_nonfinite_is_not_silently_replaced(report_probe: Path) -> None:
    raw = _run(report_probe, mode=2).stdout
    # Preserve the pre-existing stream behavior; this is not new JSON admission.
    assert re.search(r'"total_energy"\s*:\s*-?nan\b', raw, re.IGNORECASE)

