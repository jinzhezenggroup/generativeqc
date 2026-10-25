"""Compiler-level parity for one shared CUDA transport and source reducer.

Device numerical qualification belongs to the existing ten native CLI E+F
fixtures. These host checks prove that every promoted MethodIR graph is an
ordered, complete unit source sum before it can use the shared CUDA kernel.
"""

from __future__ import annotations

import struct
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.method.stationary_cuda import (
    _qualified_aot_profile,
    emit_stationary_reduction_cuda,
    stationary_runtime_sources,
)
from generativeqc_compiler.method.stationary_native_cuda import (
    NATIVE_STATIONARY_CUDA_PROFILES,
    emit_native_stationary_profile_cuda,
)

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("profile", NATIVE_STATIONARY_CUDA_PROFILES)
def test_native_shared_reduction_preserves_ir_source_order(profile: str) -> None:
    plan = _qualified_aot_profile(profile).plan
    sources = stationary_runtime_sources(plan, include_external=True)
    program = plan.reduction_program(atoms=1)
    reduction = program.outputs["gradient"]
    assert plan.source_names == sources
    assert reduction.op == "add"
    assert reduction.spec.shape == (1, 3)
    assert reduction.spec.dtype == "float64"
    assert tuple(node.attrs["name"] for node in reduction.inputs) == sources
    assert tuple(reduction.attrs["coefficients"]) == ((1, 1),) * len(sources)

    # Cover cancellation and underflow in precisely the source order that the
    # original emitted reducer uses, not a reordered parallel tree.
    values = [
        1.0e16,
        1.0,
        -1.0e16,
        0.25,
        -0.125,
        1.0e-200,
        -1.0e-200,
        0.0625,
        0.0,
        -0.0,
    ]
    selected = [values[index % len(values)] for index in range(len(sources))]
    original = 0.0
    for coefficient, value in zip(
        reduction.attrs["coefficients"], selected, strict=True
    ):
        original += float(Fraction(*coefficient)) * value
    shared = 0.0
    for value in selected:
        shared += 1.0 * value
    assert struct.pack("=d", original) == struct.pack("=d", shared)


@pytest.mark.parametrize("profile", NATIVE_STATIONARY_CUDA_PROFILES)
def test_native_emission_uses_one_shared_reducer(profile: str) -> None:
    generated = emit_native_stationary_profile_cuda(profile)
    assert "#define GENERATIVEQC_NATIVE_SHARED_REDUCTION 1" in generated
    # The method graph still supplies its own XC and AO derivatives; only the
    # now-shared integral transport and final reduction are excluded.
    assert "stationary_evaluate_point(" in generated
    assert "__global__ void source_reduce(" not in generated
    assert "__global__ void native_source_transport(" not in generated


@pytest.mark.parametrize(
    "mutation",
    ["coefficient", "shape", "dtype", "order", "computed", "missing", "repeated"],
)
def test_native_shared_reduction_rejects_changed_program(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    from generativeqc_compiler.method.stationary_gradient import StationaryGradientPlan
    from generativeqc_compiler.tensor import Program, add, input_tensor

    original = StationaryGradientPlan.reduction_program

    def changed(self: StationaryGradientPlan, **kwargs: object) -> Program:
        program = original(
            self, **({**kwargs, "atoms": 2} if mutation == "shape" else kwargs)
        )
        root = program.outputs["gradient"]
        inputs = root.inputs
        if mutation == "coefficient":
            root = add(*inputs, coefficients=(*([1] * (len(inputs) - 1)), "1/2"))
        elif mutation == "dtype":
            root = add(
                *(
                    input_tensor(
                        node.attrs["name"], replace(node.spec, dtype="float32")
                    )
                    for node in inputs
                )
            )
        elif mutation == "order":
            root = add(*reversed(inputs))
        elif mutation == "computed":
            root = add(add(inputs[0], coefficients=(-1,)), *inputs[1:])
        elif mutation == "missing":
            root = add(*inputs[:-1])
        elif mutation == "repeated":
            root = add(*inputs[:-1], inputs[0])
        return Program({"gradient": root})

    monkeypatch.setattr(StationaryGradientPlan, "reduction_program", changed)
    with pytest.raises(ValueError, match="exact unit-source sum"):
        emit_native_stationary_profile_cuda("pbe_rks")


def test_shared_kernel_scalar_order_and_transport(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Run the actual kernel bodies as host functions; no CUDA execution claim."""
    production = (ROOT / "src/dft/native_stationary_shared_kernels.cu").read_text()
    kernels = production[
        production.index("__global__ void transport_source(") : production.index(
            "}  // namespace"
        )
    ]
    runtime = (ROOT / "src/tensor/cuda_runtime.cuh").read_text()
    finite = runtime[
        runtime.index(
            "template <class T>\n__device__ inline T finite("
        ) : runtime.index("__device__ inline double quotient(")
    ]
    references = []
    calls = []
    for profile in NATIVE_STATIONARY_CUDA_PROFILES:
        plan = _qualified_aot_profile(profile).plan
        references.append(
            f"namespace {profile} {{\n"
            + emit_stationary_reduction_cuda(plan, include_external=True)
            + "}\n"
        )
        calls.append(
            f"verify_reduction({len(plan.source_names)}, {profile}::generativeqc_stationary_cuda::source_reduce);"
        )
    source = tmp_path / "shared_scalar.cpp"
    source.write_text(
        r"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#include "dft/mixed_rsh_exchange.hpp"
using generativeqc::dft::mixed_rsh_exchange_sources;
#define __global__
#define __device__
using std::isfinite;
struct Dim { std::size_t x{}; };
Dim blockIdx{}, blockDim{64}, threadIdx{};
int atomicCAS(int* p, int expected, int value) { const auto old=*p; if(old==expected) *p=value; return old; }
int atomicExch(int* p, int value) { const auto old=*p; *p=value; return old; }
"""
        + "namespace generativeqc_tensor {\n"
        + finite
        + "}\nusing generativeqc_tensor::finite;\n"
        + kernels
        + "\n".join(references)
        + r"""
void require(bool value) { if (!value) throw std::runtime_error("shared scalar contract mismatch"); }
void coordinate(std::size_t i) { blockIdx.x=i/64; threadIdx.x=i%64; }
void verify_reduction(std::size_t sources, void (*original)(const double*, std::size_t, double*, int*)) {
  const double values[] = {1e16, 1, -1e16, 0.25, -0.125, 1e-200, -1e-200, 0.0625, 0.0, -0.0};
  for (const std::size_t atoms : {1U, 2U, 128U}) {
    const auto count=3*atoms;
    std::vector<double> input(sources*count), expected(count,991), actual(count,991);
    for(std::size_t s=0;s<sources;++s)
      for(std::size_t i=0;i<count;++i) input[s*count+i]=values[(s+i)%10];
    for(const int fault : {0,1,2,3,4}) {
      if(fault==4) std::fill(input.begin(),input.end(),std::numeric_limits<double>::max());
      if(fault==1) input[0]=std::numeric_limits<double>::quiet_NaN();
      if(fault==2) input[0]=std::numeric_limits<double>::infinity();
      std::fill(expected.begin(),expected.end(),991); std::fill(actual.begin(),actual.end(),991);
      int before=fault==3?7:0, after=before;
      for(std::size_t i=0;i<count+64;++i) {
        coordinate(i); original(input.data(),atoms,expected.data(),&before);
        reduce_sources(input.data(),atoms,sources,actual.data(),&after);
      }
      require(before==after);
      for(std::size_t i=0;i<count;++i)
        require((std::isnan(expected[i])&&std::isnan(actual[i])) ||
                std::memcmp(&expected[i],&actual[i],sizeof(double))==0);
    }
  }
}
int main() {
"""
        + "\n".join(calls)
        + r"""
  for(const std::size_t count : {1U,64U,65U,384U})
    for(const double sign : {-1.0,1.0})
      for(const std::size_t channels : {1U,2U}) {
        std::vector<double> input(2*count),first(count+1,991),second(count+1,992);
        for(std::size_t i=0;i<input.size();++i) input[i]=double(i)-double(count);
        int error=0;
        for(std::size_t i=0;i<count+64;++i) { coordinate(i); transport_source(input.data(),count,channels,sign,first.data(),second.data(),&error); }
        require(!error && first.back()==991 && second.back()==992);
        for(std::size_t i=0;i<count;++i) { require(first[i]==sign*input[i]); require(second[i]==(channels==2?sign*input[count+i]:992)); }
        std::fill(input.begin()+count,input.end(),0.0);
        for(std::size_t i=0;i<count;++i) { coordinate(i); transport_source(input.data(),count,2,sign,first.data(),nullptr,&error); }
        require(!error);
        input[count]=1.0; coordinate(0); transport_source(input.data(),count,2,sign,first.data(),nullptr,&error); require(error==1);
        error=0; input[0]=std::numeric_limits<double>::quiet_NaN();
        transport_source(input.data(),count,1,sign,first.data(),nullptr,&error); require(error==1 && std::isnan(first[0]));
      }
}
"""
    )
    executable = native_cxx.build_executable(
        [source],
        tmp_path / "shared_scalar",
        compile_args=("-std=c++20", "-ffp-contract=off", f"-I{ROOT / 'src'}"),
    )
    subprocess.run([str(executable)], check=True, timeout=10)
