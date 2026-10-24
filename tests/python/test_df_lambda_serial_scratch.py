"""Exercise the actual native capacity planner without a CUDA device.

Generated query stubs separate storage/lifetime tests from scientific execution;
real-GPU numerical and provider tests remain in the existing DF Lambda suite.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import typing
from pathlib import Path

import numpy as np
import pytest
from test_df_cc_lambda import case, run
from test_df_cc_lambda import probe as _native_probe

ROOT = Path(__file__).resolve().parents[2]
probe = _native_probe


@pytest.fixture(scope="module")
def planning_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("host C++ compiler and ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    source = (ROOT / "src/cc/df_lambda_cuda.cu").read_text()
    start = source.index("    const std::array<const std::vector<double>*, 16> host")
    end = source.index("    cuda_check(cudaStreamCreateWithFlags", start)
    planner = source[start:end]
    queries = []
    for namespace in ("generated_core", "generated_response", "generated_virtual"):
        definitions = []
        for name in sorted(set(re.findall(namespace + r"::(\w+)\(", planner))):
            elements = 128
            if "staged_" in name:
                elements = 256
            if "contraction_terms" in name:
                elements = (
                    1000
                    if name
                    in (
                        "transpose_contraction_terms",
                        "virtual_amplitude_vjp_contraction_terms",
                    )
                    else 1
                )
            if "host_bytes" in name:
                elements = 64
            if name == "staged_core_reuse_arena_elements":
                elements = 64
            definitions.append(
                f"template<class... Args> std::size_t {name}(Args...){{return {elements}"
                + ("+padding" if "arena_elements" in name else "")
                + ";}"
            )
        queries.append("namespace " + namespace + "{" + "\n".join(definitions) + "}")
    folder = tmp_path_factory.mktemp("df-lambda-scratch")
    code = folder / "planning.cpp"
    code.write_text(
        PREFIX
        + "\n".join(queries)
        + r"""
struct Plan {
  generativeqc::cc::LambdaDiagnostic metrics;
  std::array<std::size_t, 3> offsets;
  std::size_t retained, allocation;
};
Plan plan(bool sharing, bool reduction, bool matrix, bool audit, bool retention,
          bool publication, unsigned rollback, std::size_t budget = 1ULL << 30) {
  generativeqc::cc::Problem p;
  generativeqc::cc::SolverResult cc;
  generativeqc::cc::LambdaOptions options;
  options.serial_scratch_reuse = sharing;
  options.df_auxiliary_reduction = reduction;
  options.df_matrix_gemm = matrix;
  options.df_audit_matrix_gemm = audit;
  options.df_core_reuse = retention;
  options.max_bytes = budget;
  const std::size_t o=2,v=3,q=5,n1=o*v,n2=n1*n1,vv=v*v;
  bool with_source=true, with_parameters=publication, parameters_admitted=false;
  std::array<double*,6> cuts{};
  std::array<std::size_t,6> cut_sizes{};
  generativeqc::cc::LambdaDiagnostic metrics;
"""
        + planner
        + r"""
  if (rollback == 1) drop_core_reuse();
  if (rollback == 2) drop_audit();
  if (rollback == 3) scalar_plan();
  if (rollback == 4) {
    scalar_plan(); metrics.df_auxiliary_reduction=false;
    cursor=fallback_cursor; scratch_elements=scratch; capacity();
  }
  if (metrics.owned_device_bytes != allocation_bytes + metrics.df_provider_allowance_bytes)
    throw std::runtime_error("allocation disagrees with admission");
  if (sharing && (arena != staged_arena || arena != audit_offset))
    throw std::runtime_error("serial scratch was not shared");
  const auto scratch_end=arena+bytes(scratch_elements);
  if (sharing && metrics.df_core_reuse && core_reuse_offset + scratch_offset_shift < scratch_end)
    throw std::runtime_error("invariant frontier overlaps scratch");
  if (sharing && metrics.df_auxiliary_reduction) {
    for (std::size_t index=0;index<cut_sizes.size();++index)
      if (cut_offsets[index]+scratch_offset_shift<scratch_end ||
          seed_offsets[index]+scratch_offset_shift<scratch_end)
        throw std::runtime_error("cuts or seeds overlap scratch");
    if (tau_offset+scratch_offset_shift<scratch_end || tau_seed_offset+scratch_offset_shift<scratch_end)
      throw std::runtime_error("tau overlaps scratch");
  }
  return {metrics,{arena,staged_arena,audit_offset},core_reuse_offset,allocation_bytes};
}
int main(int argc,char** argv) {
  padding=std::stoul(argv[3]);
  const unsigned mask=std::stoul(argv[1]),rollback=std::stoul(argv[2]);
  const auto baseline=plan(false,mask&1,mask&2,mask&4,mask&8,mask&16,rollback);
  const auto candidate=plan(true,mask&1,mask&2,mask&4,mask&8,mask&16,rollback);
  const auto &left=baseline.metrics,&right=candidate.metrics;
  if (right.numeric_capacity_bytes>left.numeric_capacity_bytes ||
      right.owned_device_bytes>left.owned_device_bytes)
    return 1;
  if (left.df_auxiliary_reduction!=right.df_auxiliary_reduction ||
      left.df_matrix_gemm!=right.df_matrix_gemm ||
      left.df_auxiliary_batch_size!=right.df_auxiliary_batch_size ||
      left.df_core_reuse!=right.df_core_reuse ||
      left.df_audit_matrix_gemm!=right.df_audit_matrix_gemm ||
      left.df_primal_matrix_gemm!=right.df_primal_matrix_gemm)
    return 2;
  const auto exact=plan(true,mask&1,mask&2,mask&4,mask&8,mask&16,rollback,
                        right.numeric_capacity_bytes);
  if (exact.metrics.numeric_capacity_bytes>right.numeric_capacity_bytes) return 3;
  std::cout<<baseline.allocation<<" "<<candidate.allocation<<"\n";
}
"""
    )
    output = folder / "planning"
    compiled = subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O0",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            str(code),
            "-o",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert compiled.returncode == 0, compiled.stderr
    return output


@pytest.mark.parametrize("mask", range(32))
@pytest.mark.parametrize("rollback", range(5))
@pytest.mark.parametrize("padding", [0, 1, 31])
def test_serial_plan_admission_and_rollback(
    planning_probe: Path, mask: int, rollback: int, padding: int
) -> None:
    result = subprocess.run(
        [str(planning_probe), str(mask), str(rollback), str(padding)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    baseline, candidate = map(int, result.stdout.split())
    assert candidate <= baseline
    if mask & 1 and rollback < 4:
        assert candidate < baseline


@pytest.mark.parametrize(
    "df,reduction,matrix",
    [
        (True, True, True),
        (True, True, False),
        (True, False, False),
        (False, False, False),
    ],
)
@pytest.mark.parametrize("source", [False, True])
@pytest.mark.parametrize("batch_two", [False, True])
def test_real_gpu_serial_scratch_preserves_values_and_work(
    probe: typing.Any,
    df: bool,
    reduction: bool,
    matrix: bool,
    source: bool,
    batch_two: bool,
) -> None:
    """Run only in the existing finite Slurm/device gate, using a rebuilt library."""
    _, _, arrays = case(2, 3, q=5)
    baseline_status, baseline, baseline_values, baseline_counts, error = run(
        probe,
        arrays,
        df=df,
        reduction=reduction,
        matrix=matrix,
        source=source,
        batch_two=batch_two,
        primal_matrix=True,
    )
    assert baseline_status == 0, error
    status, candidate, values, counts, error = run(
        probe,
        arrays,
        df=df,
        reduction=reduction,
        matrix=matrix,
        source=source,
        batch_two=batch_two,
        primal_matrix=True,
        serial_scratch_reuse=True,
    )
    assert status == 0, error
    for actual, expected in zip(candidate, baseline, strict=True):
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(
        values.view(np.uint64), baseline_values.view(np.uint64)
    )
    work = [index for index in range(len(counts)) if index not in (2, 3, 27, 28)]
    np.testing.assert_array_equal(counts[work], baseline_counts[work])
    assert counts[2] <= baseline_counts[2] and counts[3] <= baseline_counts[3]
    assert np.max(values[1:]) < 1e-9


PREFIX = r"""
#include <algorithm>
#include <array>
#include <iostream>
#include <mutex>
#include <string>
#include "cc/lambda_response.hpp"
#include "posthf/capacity.hpp"
using generativeqc::posthf::checked_add;
using generativeqc::posthf::checked_mul;
std::size_t bytes(std::size_t count){return checked_mul(count,sizeof(double));}
std::size_t reserve(std::size_t& cursor,std::size_t count){
  if(cursor%256)cursor=checked_add(cursor,256-cursor%256);
  const auto offset=cursor;cursor=checked_add(cursor,count);return offset;
}
namespace generativeqc::cc { std::size_t problem_host_bytes(const Problem&){return 1024;} }
namespace response {
struct Workspace{std::size_t workspace_bytes=512;};
template<class Options> Workspace prepare_gmres(std::size_t,const Options&){return {};}
}
namespace runtime{std::mutex allocation_measurement_mutex;}
void cuda_check(int code){if(code)throw std::runtime_error("CUDA failure");}
int cudaMemGetInfo(std::size_t* available,std::size_t* total){*available=*total=1ULL<<30;return 0;}
constexpr std::size_t kProviderAllowance=4096;
std::size_t padding=0;
struct Stage {
  std::size_t elements;
  std::size_t matrix_arena(std::size_t,std::size_t,std::size_t)const{return elements+padding;}
  bool fit(std::size_t,std::size_t,std::size_t)const{return true;}
};
const Stage primal_prepare{512},primal_auxiliary{512},core_stage{512},
  auxiliary_stage{512},prepare_stage{512},factor_stage{512},audit_core{768},
  audit_auxiliary{768},primal_virtual{1024};
struct Parameter {
  Stage stage{512};
  std::size_t arena(std::size_t,std::size_t)const{return 128;}
  std::size_t staged_arena(std::size_t,std::size_t)const{return 256;}
};
const std::array<Parameter,8> parameters{};
"""
