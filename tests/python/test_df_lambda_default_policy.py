"""Protect matrix Lambda defaults and explicit scalar fallback selection."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_lambda_matrix_defaults_and_explicit_benchmark_selection(
    tmp_path: Path,
) -> None:
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("requires host C++ and ccache")
    header = (ROOT / "src/methods/df_ccsdt_force.hpp").read_text()
    triples_header = (ROOT / "src/cc/df_triples.hpp").read_text()
    assert (
        "bool parallel_gap_reduction = true, bool include_gap_response = true"
        in triples_header
    )
    declaration = (
        "DFCCSDTResult run_df_ccsdt_native("
        + header.split("DFCCSDTResult run_df_ccsdt_native(", 1)[1].split(");", 1)[0]
        + ");\n"
    )
    # Define the extracted overload itself. Maintaining a second hand-written
    # signature silently turns new trailing selectors into an unresolved call.
    definition = re.sub(r"\s*=\s*[^,)]+", "", declaration).strip().removesuffix(";")
    definition = definition.replace(
        "const generativeqc_method_descriptor&,",
        "const generativeqc_method_descriptor& descriptor,",
    )
    definition += (
        " { return {df_matrix_gemm,forces,lambda_matrix_gemm,frame_options,"
        "descriptor.ccsd_diis_history,df_auxiliary_reduction,lambda_batch_limit,"
        "ccsd_batch_limit,derived_denominators,packed_diis,parallel_gap_reduction,"
        "request_triples_gap_cotangents,descriptor.energy_tolerance,"
        "descriptor.density_tolerance,fused_triples_scalar_response,admitted_triples_w,"
        "lambda_true_residual_interval,lambda_core_reuse,lambda_audit_matrix,lambda_primal_matrix,"
        "reference_experiment}; }\n"
    )
    endpoint = (ROOT / "benchmarks/df_ccsdt_force_endpoint.cpp").read_text()
    selectors = (
        "if (argc <"
        + endpoint.split("if (argc <", 1)[1].split("    std::ifstream input", 1)[0]
    )
    descriptor_controls = "\n".join(
        line
        for line in endpoint.splitlines()
        if any(
            field in line
            for field in (
                "descriptor.ccsd_diis_history =",
                "descriptor.energy_tolerance =",
                "descriptor.density_tolerance =",
            )
        )
    )
    assert 'field("ccsd_diis_history", diis_history);' in endpoint
    endpoint_call = (
        "const auto result ="
        + endpoint.split("const auto result =", 1)[1].split(
            "    std::ofstream output", 1
        )[0]
    )
    reference_benchmark = (ROOT / "benchmarks/df_hf_preconvergence.cpp").read_text()
    reference_call = (
        "const auto result ="
        + reference_benchmark.split("const auto result =", 1)[1].split(";", 1)[0]
        + ";\n"
    )
    source = tmp_path / "defaults.cpp"
    source.write_text(
        r"""
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include "cc/lambda_response.hpp"
#include "hf/rhf_frame_response.hpp"
#include "runtime/execution_precision.hpp"
namespace generativeqc {
namespace runtime { struct ExecutionContext {}; }
namespace hf { struct RHFFrameResponseOptions; }
namespace methods::detail {
struct DFCCSDTReferenceExperiment {};
struct DFCCSDTResult {
  bool primal, forces, lambda;
  hf::RHFFrameResponseOptions frame;
  unsigned diis_history;
  bool reduction;
  std::size_t batch_limit, ccsd_batch_limit;
  bool derived_denominators, packed_diis, parallel_gap, request_gap;
  double reference_energy_tolerance{}, reference_density_tolerance{};
  bool fused_scalar{};
  runtime::PrecisionDirective triples_w;
  std::size_t lambda_interval;
  bool core_reuse;
  bool audit_matrix;
  bool replay_matrix;
  DFCCSDTReferenceExperiment* reference_experiment;
};
"""
        + declaration
        + definition
        + r"""
}}
generativeqc::methods::detail::DFCCSDTResult select(int argc,const char** argv) {
  generativeqc::runtime::ExecutionContext execution;
  generativeqc::core::System orbital, auxiliary;
  generativeqc_method_descriptor descriptor{};
"""
        + selectors
        + descriptor_controls
        + "\n"
        + endpoint_call
        + r"""
  return result;
}
void check_reference_benchmark() {
  generativeqc::runtime::ExecutionContext execution;
  generativeqc::core::System orbital, correlation;
  generativeqc_method_descriptor descriptor{};
  generativeqc::methods::detail::DFCCSDTReferenceExperiment experiment;
  const std::string endpoint = "forces";
"""
        + reference_call
        + r"""
  if(result.reference_experiment != &experiment || result.replay_matrix ||
     result.batch_limit != 8 || result.ccsd_batch_limit != 8)
    throw std::runtime_error("reference benchmark selectors or experiment pointer changed");
}
bool default_frame(const generativeqc::hf::RHFFrameResponseOptions& frame) {
  return frame.orbital_screening_tolerance == 0.0 && !frame.profile_jk &&
         !frame.bilinear_derivative && frame.symmetric_polarization;
}
int main() {
  check_reference_benchmark();
  generativeqc::cc::LambdaOptions options;
  if(!options.df_matrix_gemm || !options.df_auxiliary_reduction ||
     options.gmres.true_residual_every != 1 ||
     options.df_auxiliary_batch_limit != 32 || !options.df_primal_matrix_gemm) return 1;
  options.df_matrix_gemm=false;
  if(options.df_matrix_gemm) return 2;
  generativeqc::runtime::ExecutionContext context;
  generativeqc::core::System system;
  generativeqc_method_descriptor descriptor{};
  using generativeqc::methods::detail::run_df_ccsdt_native;
  auto ordinary=run_df_ccsdt_native(context,system,system,descriptor);
  auto explicit_matrix=run_df_ccsdt_native(context,system,system,descriptor,
                                         true,true,true,true,true,8);
  if(!ordinary.primal || !ordinary.lambda || !explicit_matrix.lambda ||
     !ordinary.derived_denominators || !explicit_matrix.derived_denominators ||
     ordinary.packed_diis || explicit_matrix.packed_diis || !ordinary.parallel_gap ||
     !explicit_matrix.parallel_gap || ordinary.request_gap || explicit_matrix.request_gap) return 3;
  if(!default_frame(ordinary.frame) || !default_frame(explicit_matrix.frame) ||
     ordinary.lambda_interval != 30 || explicit_matrix.lambda_interval != 30 ||
     ordinary.batch_limit != 32 || explicit_matrix.batch_limit != 8 ||
     !ordinary.replay_matrix) return 7;
  generativeqc::hf::RHFFrameResponseOptions explicit_frame;
  explicit_frame.orbital_screening_tolerance = 1e-7;
  explicit_frame.profile_jk = true;
  explicit_frame.bilinear_derivative = true;
  explicit_frame.symmetric_polarization = false;
  auto explicit_scalar=run_df_ccsdt_native(context,system,system,descriptor,
                                         true,true,true,false,false,8,3,explicit_frame);
  if(explicit_scalar.primal || explicit_scalar.lambda ||
     explicit_scalar.frame.orbital_screening_tolerance != 1e-7 ||
     !explicit_scalar.frame.profile_jk || !explicit_scalar.frame.bilinear_derivative ||
     explicit_scalar.frame.symmetric_polarization || explicit_scalar.ccsd_batch_limit != 3 ||
     explicit_scalar.lambda_interval != 30) return 8;
  const auto strict_lambda=run_df_ccsdt_native(
      context,system,system,descriptor,true,true,true,true,true,8,8,{},
      true,false,true,false,false,{},1);
  if(strict_lambda.lambda_interval != 1) return 44;
  const char* missing[]{"endpoint","input","output","1"};
  const char* matrix[]{"endpoint","input","output","1","1","1","1"};
  const char* scalar[]{"endpoint","input","output","1","1","1","0"};
  const char* invalid[]{"endpoint","input","output","1","1","1","x"};
  const auto defaults = select(4,missing);
  if(std::array<bool,3>{defaults.primal,defaults.forces,defaults.lambda} !=
     std::array<bool,3>{true,true,true}) return 4;
  if(!default_frame(defaults.frame) || defaults.diis_history != 6 ||
     defaults.batch_limit != 32 || defaults.ccsd_batch_limit != 8 || !defaults.reduction ||
     !defaults.derived_denominators || !defaults.parallel_gap || defaults.request_gap ||
     defaults.lambda_interval != 30) return 9;
  if(!select(7,matrix).lambda || select(7,scalar).lambda) return 5;
  try { (void)select(7,invalid);return 6; }
  catch(const std::invalid_argument&) {}
  for(const char* schedule : {"0","1","2"}) {
    const char* selected[]{"endpoint","input","output","1","1","1","1","8",
                           "6","3","1e-7","1",schedule};
    const auto frame = select(13,selected).frame;
    if(frame.orbital_screening_tolerance != 1e-7 || !frame.profile_jk ||
       frame.bilinear_derivative != (schedule[0] == '1') ||
       frame.symmetric_polarization != (schedule[0] == '2')) return 10;
  }
  const char* invalid_frame[]{"endpoint","input","output","1","1","1","1","8",
                              "6","8","0","0","x"};
  try { (void)select(13,invalid_frame);return 11; }
  catch(const std::invalid_argument&) {}
  // Every historical master DIIS choice retains its exact argument position.
  for(unsigned history = 0; history <= 20; ++history) {
    if(history == 1) continue;
    const auto token = std::to_string(history);
    const char* selected[]{"endpoint","input","output","0","0","0","0","3",
                           token.c_str()};
    const auto old = select(9,selected);
    if(old.diis_history != history || !default_frame(old.frame) || old.reduction ||
       old.primal || old.forces || old.lambda || old.batch_limit != 3 || old.ccsd_batch_limit != 8 || !old.derived_denominators) return 12;
  }
  // No numeric guessing: even an integer zero in argv[8] always means DIIS zero.
  // Fractional/scientific legacy screening tokens must not partially parse.
  for(const char* token : {"1","21","-2","0.0","0e-12","2e-12","2.0","6junk",""}) {
    const char* bad[]{"endpoint","input","output","1","1","1","1","8",token};
    try { (void)select(9,bad);return 13; }
    catch(const std::invalid_argument&) {}
  }
  const char* partial[]{"endpoint","input","output","1","1","1","1","8",
                        "4","3","1e-7","0"};
  for(int argc = 8; argc <= 12; ++argc) {
    const auto selected = select(argc,partial);
    if(selected.diis_history != (argc > 8 ? 4U : 6U) ||
       selected.ccsd_batch_limit != (argc > 9 ? 3U : 8U) ||
       selected.frame.orbital_screening_tolerance != (argc > 10 ? 1e-7 : 0.0) ||
       selected.frame.profile_jk || !selected.frame.symmetric_polarization ||
       selected.frame.bilinear_derivative || !selected.derived_denominators) return 14;
  }
  for(int index : {3,4,5,6,11,12}) {
    const char* bad[]{"endpoint","input","output","1","1","1","1","8",
                      "6","8","0","0","2"};
    bad[index] = "x";
    try { (void)select(13,bad);return 15; }
    catch(const std::invalid_argument&) {}
  }
  for(const char* batch : {"0","1","3","8","32"}) {
    const char* selected[]{"endpoint","input","output","1","1","1","1","8","6",batch};
    const auto old = select(10,selected);
    if(old.ccsd_batch_limit != std::stoull(batch) || !default_frame(old.frame) ||
       old.diis_history != 6) return 17;
  }
  for(int index : {7,8,9}) {
    for(const char* token : {"-1","+2","2.0","2e-12","8junk",""}) {
      const char* bad[]{"endpoint","input","output","1","1","1","1","8","6","8"};
      bad[index] = token;
      try { (void)select(10,bad);return 18; }
      catch(const std::invalid_argument&) {}
    }
  }
  for(const char* token : {"nan","inf","-1e-7","1e-7junk",""}) {
    const char* bad[]{"endpoint","input","output","1","1","1","1","8","6","8",token};
    try { (void)select(11,bad);return 19; }
    catch(const std::invalid_argument&) {}
  }
    for(bool derived : {false,true}) {
    const auto direct=run_df_ccsdt_native(context,system,system,descriptor,
                                          true,true,true,true,true,5,3,explicit_frame,derived);
    if(direct.derived_denominators!=derived || direct.batch_limit!=5 ||
       direct.ccsd_batch_limit!=3 || direct.frame.orbital_screening_tolerance!=1e-7 ||
       !direct.frame.profile_jk || !direct.frame.bilinear_derivative) return 20;
    for(const char* schedule : {"0","1","2"}) {
      const char* selected[]{"endpoint","input","output","1","1","1","1","5",
                             "4","3","1e-7","1",schedule,derived?"1":"0"};
      const auto result=select(14,selected);
      if(result.derived_denominators!=derived || result.diis_history!=4 ||
         result.batch_limit!=5 || result.ccsd_batch_limit!=3 ||
         result.frame.orbital_screening_tolerance!=1e-7 || !result.frame.profile_jk ||
         result.frame.bilinear_derivative!=(schedule[0]=='1') ||
         result.frame.symmetric_polarization!=(schedule[0]=='2')) return 21;
      if(!select(13,selected).derived_denominators) return 22;
    }
  }
  for(const char* token : {"", "2", "-1", "+1", "true", "0.0", "1junk"}) {
    const char* bad[]{"endpoint","input","output","1","1","1","1","8",
                      "6","8","0","0","2",token};
    try { (void)select(14,bad);return 23; }
    catch(const std::invalid_argument&) {}
  }
  for(const char* inverse : {"0","1"}) for(const char* repeat : {"0","1"}) {
    const char* selected[]{"endpoint","input","output","1","1","1","1","5",
                          "4","3","1e-7","1","2","0","7",inverse,repeat};
    const auto result=select(17,selected);
    if(result.derived_denominators || result.diis_history!=4 || result.ccsd_batch_limit!=3 ||
       result.frame.gmres.true_residual_every!=7 ||
       result.frame.df_preconditioning!=(inverse[0]=='1') ||
       bool(result.frame.recycling)!=(repeat[0]=='1') ||
       result.frame.orbital_screening_tolerance!=1e-7 || !result.frame.profile_jk ||
       !result.frame.symmetric_polarization || result.packed_diis) return 24;
  }
  for(int index : {14,15,16}) for(const char* token : {"", "-1", "1junk", "1.0"}) {
    const char* bad[]{"endpoint","input","output","1","1","1","1","8",
                      "6","8","0","0","2","1","7","0","0"};
    bad[index]=token;
    try { (void)select(17,bad);return 25; } catch(const std::invalid_argument&) {}
  }
  for(bool packed : {false,true})
  for(const char* inverse : {"0","1"}) for(const char* repeat : {"0","1"}) {
    const char* selected[]{"endpoint","input","output","1","1","1","1","5",
                          "4","3","1e-7","1","2","0","7",inverse,repeat,packed?"1":"0"};
    const auto result=select(18,selected);
    if(result.derived_denominators || result.packed_diis!=packed || result.diis_history!=4 ||
       result.ccsd_batch_limit!=3 || result.frame.gmres.true_residual_every!=7 ||
       result.frame.df_preconditioning!=(inverse[0]=='1') ||
       bool(result.frame.recycling)!=(repeat[0]=='1') ||
       result.frame.orbital_screening_tolerance!=1e-7 || !result.frame.profile_jk ||
       !result.frame.symmetric_polarization) return 26;
    if(select(17,selected).packed_diis) return 27;
  }
  for(const char* token : {"", "-1", "2", "1junk", "1.0"}) {
    const char* bad[]{"endpoint","input","output","1","1","1","1","8",
                      "6","8","0","0","2","1","7","0","0",token};
    try { (void)select(18,bad);return 28; } catch(const std::invalid_argument&) {}
  }
  for(const char* ceiling : {"0", "1", "8589934592"}) {
    const char* selected[]{"endpoint","input","output","1","1","1","1","8",
                           "6","8","0","0","2","1","7","0","0","1",ceiling};
    const auto explicit_cache = select(19,selected);
    if(!explicit_cache.frame.resident_jk_maximum_bytes ||
       *explicit_cache.frame.resident_jk_maximum_bytes != std::stoull(ceiling) ||
       !explicit_cache.packed_diis || explicit_cache.frame.gmres.true_residual_every != 7)
      return 29;
    if(select(18,selected).frame.resident_jk_maximum_bytes) return 30;
  }
  for(const char* token : {"", "-1", "+1", "1junk", "1.0"}) {
    const char* bad[]{"endpoint","input","output","1","1","1","1","8",
                      "6","8","0","0","2","1","7","0","0","1",token};
    try { (void)select(19,bad);return 31; } catch(const std::invalid_argument&) {}
  }
  for(const char* parallel : {"0", "1"}) for(const char* requested : {"0", "1"}) {
    const char* selected[]{"endpoint","input","output","1","1","1","1","8",
                           "6","8","0","0","2","1","7","0","0","1","auto",
                           parallel,requested};
    const auto result = select(21,selected);
    if(result.parallel_gap != (parallel[0]=='1') || result.request_gap != (requested[0]=='1') ||
       result.frame.resident_jk_maximum_bytes || !result.packed_diis) return 32;
    const auto partial_gap = select(20,selected);
    if(partial_gap.parallel_gap != (parallel[0]=='1') || partial_gap.request_gap) return 33;
    const auto default_gap = select(19,selected);
    if(!default_gap.parallel_gap || default_gap.request_gap) return 34;
  }
  for(int index : {19,20}) for(const char* token : {"", "-1", "+1", "2", "1junk", "1.0"}) {
    const char* bad[]{"endpoint","input","output","1","1","1","1","8",
                      "6","8","0","0","2","1","7","0","0","1","auto","0","1"};
    bad[index]=token;
    try { (void)select(21,bad);return 35; } catch(const std::invalid_argument&) {}
  }
  for(const char* tolerance : {"1e-12", "1e-13"}) {
    const char* selected[]{"endpoint","input","output","1","1","1","1","8",
                           "6","8","0","0","2","1","7","0","0","1","auto",
                           "1","0",tolerance};
    const auto result = select(22,selected);
    if(result.reference_energy_tolerance != std::stod(tolerance) ||
       result.reference_density_tolerance != std::stod(tolerance) ||
       !result.parallel_gap || result.request_gap) return 36;
    const auto legacy = select(21,selected);
    if(legacy.reference_energy_tolerance != 1e-12 ||
       legacy.reference_density_tolerance != 1e-11) return 37;
  }
  for(const char* tolerance : {"", "0", "-1e-13", "nan", "inf", "1e-11", "1e-13x"}) {
    const char* bad[]{"endpoint","input","output","1","1","1","1","8",
                      "6","8","0","0","2","1","7","0","0","1","auto",
                      "0","1",tolerance};
    try { (void)select(22,bad);return 38; } catch(const std::invalid_argument&) {}
  }
  for(const char* fused : {"0", "1"}) for(const char* tolerance : {"auto", "1e-13"}) {
    const char* selected[]{"endpoint","input","output","1","1","1","1","8",
                           "6","8","0","0","2","1","7","0","0","1","auto",
                           "1","0",tolerance,fused};
    const auto result = select(23,selected);
    if(result.fused_scalar != (fused[0]=='1') || select(22,selected).fused_scalar) return 39;
    if(std::string(tolerance)=="auto" &&
       (result.reference_energy_tolerance != 1e-12 || result.reference_density_tolerance != 1e-11))
      return 40;
  }
  for(const char* token : {"", "2", "true", "1x"}) {
    const char* bad[]{"endpoint","input","output","1","1","1","1","8",
                      "6","8","0","0","2","1","7","0","0","1","auto",
                      "1","0","auto",token};
    try { (void)select(23,bad);return 41; } catch(const std::invalid_argument&) {}
  }
  for(const char* precision : {"0", "1"}) {
    const char* selected[]{"endpoint","input","output","1","1","1","1","8",
                           "6","8","0","0","2","1","7","0","0","1","auto",
                           "1","0","auto","0",precision};
    const auto result = select(24,selected);
    if(result.triples_w.is_strict_fp64() != (precision[0]=='0') ||
       !select(23,selected).triples_w.is_strict_fp64()) return 42;
  }
  for(const char* interval : {"1", "7", "30"}) {
    const char* selected[]{"endpoint","input","output","1","1","1","1","8",
                           "6","8","0","0","2","1","7","0","0","1","auto",
                           "1","0","auto","0","1",interval};
    if(select(25,selected).lambda_interval != std::stoull(interval) ||
       select(24,selected).lambda_interval != 30) return 43;
  }
  for(const char* reuse : {"0", "1"}) {
    const char* selected[]{"endpoint","input","output","1","1","1","1","8",
                           "6","8","0","0","2","1","7","0","0","1","auto",
                           "1","0","auto","0","0","30",reuse};
    if(select(26,selected).core_reuse != (reuse[0]=='1') ||
       !select(25,selected).core_reuse) return 44;
  }
  for(const char* token : {"", "2", "true", "1x"}) {
    const char* selected[]{"endpoint","input","output","1","1","1","1","8",
                           "6","8","0","0","2","1","7","0","0","1","auto",
                           "1","0","auto","0","0","30",token};
    try { (void)select(26,selected);return 45; } catch(const std::invalid_argument&) {}
  }
  for(const char* audit : {"0", "1"}) {
    const char* selected[]{"endpoint","input","output","1","1","1","1","8",
                           "6","8","0","0","2","1","7","0","0","1","auto",
                           "1","0","auto","0","0","30","1",audit};
    if(select(27,selected).audit_matrix != (audit[0]=='1') ||
       !select(26,selected).audit_matrix) return 46;
  }
  for(const char* replay : {"0", "1"}) {
    const char* selected[]{"endpoint","input","output","1","1","1","1","8",
                           "6","8","0","0","2","1","7","0","0","1","auto",
                           "1","0","auto","0","0","30","1","1",replay};
    if(select(28,selected).replay_matrix != (replay[0]=='1') ||
       !select(27,selected).replay_matrix || !ordinary.replay_matrix) return 47;
  }
  for(int argc : {0,1,2,3,29}) {
    try { (void)select(argc,nullptr);return 16; }
    catch(const std::invalid_argument&) {}
  }
}
"""
    )
    executable = tmp_path / "defaults"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=10)


def test_shared_state_probe_keeps_a_scalar_control() -> None:
    source = (ROOT / "benchmarks/df_lambda_shared_state.cpp").read_text()
    first = source.index("generativeqc::cc::LambdaOptions options;")
    matrix = source.index("const auto matrix =", first)
    scalar_option = source.index("options.df_matrix_gemm = false;", matrix)
    scalar = source.index("const auto scalar =", scalar_option)
    assert first < matrix < scalar_option < scalar


def test_force_owner_forwards_denominators_after_reference_and_batch(
    tmp_path: Path,
) -> None:
    """Compile the real middle-owner call against its current RCCSD declaration."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("requires host C++ and ccache")
    header = (ROOT / "src/methods/rccsd_method.hpp").read_text()
    declaration = (
        "RccsdNativeState run_rccsd_native_state("
        + header.split("RccsdNativeState run_rccsd_native_state(", 1)[1].split(");", 1)[
            0
        ]
        + ");\n"
    )
    definition = re.sub(r"\s*=\s*[^,)]+", "", declaration).strip().removesuffix(";")
    definition += (
        " { return {cuda_reference_plan,df_auxiliary_batch_limit,derived_denominators,"
        "retain_df_response,df_matrix_gemm,correlation_auxiliary,packed_diis,"
        "external_reservation_bytes,initial_density,warm_start_fallback}; }\n"
    )
    owner = (ROOT / "src/methods/df_ccsdt_force.cu").read_text()
    call = "auto state =" + owner.split("auto state =", 1)[1].split(";", 1)[0] + ";\n"
    source = tmp_path / "forward.cpp"
    source.write_text(
        r"""
#include <memory>
#include <vector>
#include <cstddef>
struct generativeqc_method_descriptor {};
namespace generativeqc {
namespace runtime { struct ExecutionContext {}; }
namespace core { struct System {}; }
namespace scf { struct PreparedFockPlan {}; struct CudaRhfBucketPlan {}; }
namespace methods::detail {
struct RccsdNativeState {
  scf::CudaRhfBucketPlan** reference_plan;
  std::size_t batch;
  bool derived, retained, matrix;
  const core::System* auxiliary;
  bool packed;
  std::size_t reserved;
  const std::vector<double>* seed;
  bool* seed_fallback;
};
struct DFPhysicalResponseComparison { std::size_t output_bytes; };
std::size_t checked_add(std::size_t left, std::size_t right) { return left + right; }
"""
        + declaration
        + definition
        + r"""
int probe() {
  runtime::ExecutionContext execution;
  core::System system, auxiliary;
  generativeqc_method_descriptor descriptor;
  RccsdNativeState* replay_state=nullptr;
  struct ReferenceExperiment { bool seed_fallback{}; };
  ReferenceExperiment experiment;
  const std::vector<double> seed{1.,2.,3.};
  const std::size_t recycle_bytes=123;
  DFPhysicalResponseComparison comparison{576};
  for(auto* physical_replay : {static_cast<DFPhysicalResponseComparison*>(nullptr), &comparison})
  for(bool packed_diis : {false,true})
  for(bool derived_denominators : {false,true})
  for(bool forces : {false,true})
  for(bool df_matrix_gemm : {false,true})
  for(std::size_t ccsd_batch_limit : {1,3,8}) {
  for(bool seeded : {false,true}) {
    const auto* reference_seed=seeded ? &seed : nullptr;
    auto* reference_experiment=seeded ? &experiment : nullptr;
    const std::size_t seed_bytes=seeded ? seed.capacity()*sizeof(double) : 0;
"""
        + call
        + r"""
    if(state.reference_plan || state.batch!=ccsd_batch_limit ||
       state.derived!=derived_denominators || state.retained!=forces ||
       state.matrix!=df_matrix_gemm || state.auxiliary!=&auxiliary || state.packed!=packed_diis ||
       state.reserved!=seed_bytes+recycle_bytes+(physical_replay ? physical_replay->output_bytes : 0) ||
       state.seed!=reference_seed ||
       state.seed_fallback!=(seeded ? &experiment.seed_fallback : nullptr)) return 1;
  }
  }
  const auto ordinary=run_rccsd_native_state(execution,system,descriptor);
  if(ordinary.reference_plan || ordinary.batch!=8 || !ordinary.derived ||
     ordinary.retained || !ordinary.matrix || ordinary.auxiliary || ordinary.packed || ordinary.reserved) return 2;
  scf::CudaRhfBucketPlan resident;
  auto* reference=&resident;
  const auto explicit_state=run_rccsd_native_state(execution,system,descriptor,
      nullptr,nullptr,nullptr,0,&auxiliary,true,true,&reference,3,false);
  if(explicit_state.reference_plan!=&reference || reference!=&resident ||
     explicit_state.batch!=3 || explicit_state.derived) return 3;
  return 0;
}
}}
int main() { return generativeqc::methods::detail::probe(); }
"""
    )
    executable = tmp_path / "forward"
    subprocess.run(
        [cache, compiler, "-std=c++20", str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=10)
