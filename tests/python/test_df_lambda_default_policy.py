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
        "ccsd_batch_limit}; }\n"
    )
    endpoint = (ROOT / "benchmarks/df_ccsdt_force_endpoint.cpp").read_text()
    selectors = (
        "if (argc <"
        + endpoint.split("if (argc <", 1)[1].split("    std::ifstream input", 1)[0]
    )
    descriptor_diis = next(
        line
        for line in endpoint.splitlines()
        if "descriptor.ccsd_diis_history =" in line
    )
    assert 'field("ccsd_diis_history", diis_history);' in endpoint
    endpoint_call = (
        "const auto result ="
        + endpoint.split("const auto result =", 1)[1].split(
            "    std::ofstream output", 1
        )[0]
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
struct generativeqc_method_descriptor { unsigned ccsd_diis_history = 6; };
namespace generativeqc {
namespace runtime { struct ExecutionContext {}; }
namespace core { struct System {}; }
namespace hf { struct RHFFrameResponseOptions; }
namespace methods::detail {
struct DFCCSDTResult {
  bool primal, forces, lambda;
  hf::RHFFrameResponseOptions frame;
  unsigned diis_history;
  bool reduction;
  std::size_t batch_limit, ccsd_batch_limit;
};
"""
        + declaration
        + definition
        + r"""
}}
generativeqc::methods::detail::DFCCSDTResult select(int argc,const char** argv) {
  generativeqc::runtime::ExecutionContext execution;
  generativeqc::core::System orbital, auxiliary;
  generativeqc_method_descriptor descriptor;
"""
        + selectors
        + descriptor_diis
        + "\n"
        + endpoint_call
        + r"""
  return result;
}
bool default_frame(const generativeqc::hf::RHFFrameResponseOptions& frame) {
  return frame.orbital_screening_tolerance == 0.0 && !frame.profile_jk &&
         !frame.bilinear_derivative && frame.symmetric_polarization;
}
int main() {
  generativeqc::cc::LambdaOptions options;
  if(!options.df_matrix_gemm || !options.df_auxiliary_reduction) return 1;
  options.df_matrix_gemm=false;
  if(options.df_matrix_gemm) return 2;
  generativeqc::runtime::ExecutionContext context;
  generativeqc::core::System system;
  generativeqc_method_descriptor descriptor;
  using generativeqc::methods::detail::run_df_ccsdt_native;
  auto ordinary=run_df_ccsdt_native(context,system,system,descriptor);
  auto explicit_matrix=run_df_ccsdt_native(context,system,system,descriptor,
                                         true,true,true,true,true,8);
  if(!ordinary.primal || !ordinary.lambda || !explicit_matrix.lambda) return 3;
  if(!default_frame(ordinary.frame) || !default_frame(explicit_matrix.frame)) return 7;
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
     explicit_scalar.frame.symmetric_polarization || explicit_scalar.ccsd_batch_limit != 3) return 8;
  const char* missing[]{"endpoint","input","output","1"};
  const char* matrix[]{"endpoint","input","output","1","1","1","1"};
  const char* scalar[]{"endpoint","input","output","1","1","1","0"};
  const char* invalid[]{"endpoint","input","output","1","1","1","x"};
  const auto defaults = select(4,missing);
  if(std::array<bool,3>{defaults.primal,defaults.forces,defaults.lambda} !=
     std::array<bool,3>{true,true,true}) return 4;
  if(!default_frame(defaults.frame) || defaults.diis_history != 6 ||
     defaults.batch_limit != 8 || defaults.ccsd_batch_limit != 8 || !defaults.reduction) return 9;
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
       old.primal || old.forces || old.lambda || old.batch_limit != 3 || old.ccsd_batch_limit != 8) return 12;
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
       selected.frame.bilinear_derivative) return 14;
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
  for(int argc : {0,1,2,3,14}) {
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
