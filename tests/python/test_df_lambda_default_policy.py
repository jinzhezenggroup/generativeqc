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
    definition += " { return {df_matrix_gemm,lambda_matrix_gemm}; }\n"
    endpoint = (ROOT / "benchmarks/df_ccsdt_force_endpoint.cpp").read_text()
    selectors = (
        "const auto selector ="
        + endpoint.split("const auto selector =", 1)[1].split(
            "    const std::size_t batch_limit", 1
        )[0]
    )
    source = tmp_path / "defaults.cpp"
    source.write_text(
        r"""
#include <array>
#include <stdexcept>
#include <string>
#include "cc/lambda_response.hpp"
struct generativeqc_method_descriptor {};
namespace generativeqc {
namespace runtime { struct ExecutionContext {}; }
namespace core { struct System {}; }
namespace hf { struct RHFFrameResponseOptions; }
namespace methods::detail {
struct DFCCSDTResult { bool primal, lambda; };
"""
        + declaration
        + definition
        + r"""
}}
std::array<bool,3> select(int argc,const char** argv) {
"""
        + selectors
        + r"""
  return {matrix,forces,lambda_matrix};
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
  const char* missing[]{"endpoint","input","output","1"};
  const char* matrix[]{"endpoint","input","output","1","1","1","1"};
  const char* scalar[]{"endpoint","input","output","1","1","1","0"};
  const char* invalid[]{"endpoint","input","output","1","1","1","x"};
  if(select(4,missing)!=std::array<bool,3>{true,true,true}) return 4;
  if(!select(7,matrix)[2] || select(7,scalar)[2]) return 5;
  try { (void)select(7,invalid);return 6; }
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
