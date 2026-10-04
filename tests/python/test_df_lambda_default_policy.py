"""Protect matrix Lambda defaults and explicit scalar fallback selection."""

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
    endpoint = (ROOT / "benchmarks/df_ccsdt_force_endpoint.cpp").read_text()
    selectors = (
        "const bool reduction ="
        + endpoint.split("const bool reduction =", 1)[1].split(
            "    std::ifstream input", 1
        )[0]
    )
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
#include <stdexcept>
#include <string>
#include "cc/lambda_response.hpp"
#include "hf/rhf_frame_response.hpp"
struct generativeqc_method_descriptor {};
namespace generativeqc {
namespace runtime { struct ExecutionContext {}; }
namespace core { struct System {}; }
namespace methods::detail {
struct DFCCSDTResult {
  bool primal, forces, lambda;
  hf::RHFFrameResponseOptions frame;
};
"""
        + declaration
        + r"""
DFCCSDTResult run_df_ccsdt_native(runtime::ExecutionContext&,const core::System&,
    const core::System&,const generativeqc_method_descriptor&,bool forces,bool,bool,
    bool primal,bool lambda,std::size_t,const hf::RHFFrameResponseOptions& frame) {
  return {primal,forces,lambda,frame};
}
}}
generativeqc::methods::detail::DFCCSDTResult select(int argc,const char** argv) {
  generativeqc::runtime::ExecutionContext execution;
  generativeqc::core::System orbital, auxiliary;
  generativeqc_method_descriptor descriptor;
"""
        + selectors
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
                                         true,true,true,false,false,8,explicit_frame);
  if(explicit_scalar.primal || explicit_scalar.lambda ||
     explicit_scalar.frame.orbital_screening_tolerance != 1e-7 ||
     !explicit_scalar.frame.profile_jk || !explicit_scalar.frame.bilinear_derivative ||
     explicit_scalar.frame.symmetric_polarization) return 8;
  const char* missing[]{"endpoint","input","output","1"};
  const char* matrix[]{"endpoint","input","output","1","1","1","1"};
  const char* scalar[]{"endpoint","input","output","1","1","1","0"};
  const char* invalid[]{"endpoint","input","output","1","1","1","x"};
  const auto defaults = select(4,missing);
  if(std::array<bool,3>{defaults.primal,defaults.forces,defaults.lambda} !=
     std::array<bool,3>{true,true,true}) return 4;
  if(!default_frame(defaults.frame)) return 9;
  if(!select(7,matrix).lambda || select(7,scalar).lambda) return 5;
  try { (void)select(7,invalid);return 6; }
  catch(const std::invalid_argument&) {}
  for(const char* schedule : {"0","1","2"}) {
    const char* selected[]{"endpoint","input","output","1","1","1","1","8",
                           "1e-7","1",schedule};
    const auto frame = select(11,selected).frame;
    if(frame.orbital_screening_tolerance != 1e-7 || !frame.profile_jk ||
       frame.bilinear_derivative != (schedule[0] == '1') ||
       frame.symmetric_polarization != (schedule[0] == '2')) return 10;
  }
  const char* invalid_frame[]{"endpoint","input","output","1","1","1","1","8",
                              "0","0","x"};
  try { (void)select(11,invalid_frame);return 11; }
  catch(const std::invalid_argument&) {}
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
