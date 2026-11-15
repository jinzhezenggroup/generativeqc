"""Native CUDA scalar reductions and exact emitted-worker sharing contracts."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.tensor import Node, Program
from generativeqc_compiler.tensor.ir import (
    add,
    divide,
    einsum,
    input_tensor,
    multiply,
    reduce_sum,
    runtime_indexed_select,
)
from generativeqc_compiler.tensor.types import Index, IndexSpace, TensorSpec

from tools import generate_df_ccsd_core, generate_df_ccsd_hoisted
from tools import generate_rccsd_native as codegen

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_kernel_pool_preserves_executable_contracts() -> None:
    """Names may share; arithmetic, layout, precision and reduction policy may not."""
    pool = codegen.CudaKernelPool("probe")
    occupied = IndexSpace("occupied", "occupied", 3)
    virtual = IndexSpace("virtual", "virtual", 3)

    def source(name: str, space: IndexSpace, dtype: str = "float64") -> Node:
        return input_tensor(
            name, TensorSpec((Index("index", space),), role="input", dtype=dtype)
        )

    left, right = source("left", occupied), source("right", occupied)
    symbol, definition = pool.emit(add(left, right))
    renamed = add(source("renamed_left", occupied), source("renamed_right", occupied))
    assert pool.emit(renamed) == (symbol, "")
    assert definition
    variants = (
        add(left, right, coefficients=(1, -1)),
        add(source("left", virtual), source("right", virtual)),
        add(source("left", occupied, "float32"), source("right", occupied, "float32")),
        multiply(left, right),
    )
    symbols = {symbol}
    for variant in variants:
        other, emitted = pool.emit(variant)
        assert other not in symbols and emitted
        symbols.add(other)
    scalar = reduce_sum(left, (0,))
    serial, _ = pool.emit(scalar)
    parallel, emitted = pool.emit(scalar, parallel_scalar_reductions=True)
    assert serial != parallel and emitted
    batched, emitted = pool.emit(add(left, right), batch_dim=True)
    assert batched != symbol and emitted
    for ownership in ({"emit_kernels": False}, {"kernel_prefix": "external"}):
        with pytest.raises(ValueError, match="ownership"):
            codegen._cuda_program(
                Program({"out": add(left, right)}),
                "runner",
                "Output",
                kernel_pool=pool,
                **ownership,
            )


@pytest.mark.parametrize(
    "owner", ("", "0owner", "unit-name", "unit::name", "unit name")
)
def test_kernel_pool_requires_a_valid_owner(owner: str) -> None:
    with pytest.raises(ValueError, match="owner"):
        codegen.CudaKernelPool(owner)


def test_identical_workers_in_distinct_units_link_and_execute(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Same arithmetic in one namespace must not create duplicate global symbols."""
    occupied = Index("occupied", IndexSpace("occupied", "occupied", 3))
    source = input_tensor("source", TensorSpec((occupied,), role="input"))
    node = add(source, source)
    units = []
    for owner in ("left_unit", "right_unit"):
        symbol, kernel = codegen.CudaKernelPool(owner).emit(node)
        unit = tmp_path / f"{owner}.cpp"
        unit.write_text(
            """
#include <cassert>
#include <cstddef>
#define __global__
struct Dim {unsigned x;};
constexpr Dim threadIdx{0},blockIdx{0},blockDim{1},gridDim{1};
namespace generativeqc_tensor {inline double finite(double value,int*,int){return value;}}
namespace common_namespace {
"""
            + kernel
            + f"""
}}
extern "C" int run_{owner}(){{
  const double input[3]={{1.,-2.,.125}};
  double output[5]={{54321.,-1.,-1.,-1.,54321.}};
  int error=0;
  common_namespace::{symbol}(input,input,output+1,3,1,&error,37);
  assert(error==0 && output[0]==54321. && output[4]==54321.);
  for(std::size_t index=0;index<3;++index)assert(output[index+1]==2.*input[index]);
  return 1;
}}
"""
        )
        units.append(unit)
    main = tmp_path / "main.cpp"
    main.write_text(
        '#include <cassert>\nextern "C" int run_left_unit();\n'
        'extern "C" int run_right_unit();\n'
        "int main(){assert(run_left_unit()==1 && run_right_unit()==1);}\n"
    )
    binary = required_native_cxx.build_executable(
        [*units, main], tmp_path / "linked", compile_args=("-std=c++17", "-O2")
    )
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=30)


def test_shared_runners_preserve_values_arenas_and_node_errors(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Execute real emitted runners with a one-thread host CUDA launch shim.

    Handwritten matrix/scalar formulas are the independent finite-value oracle.
    Unshared runners additionally check bitwise parity and original node errors
    under NaN, infinity and overflow injection. This is not GPU qualification.
    """
    occupied = IndexSpace("occupied", "occupied", 3)
    virtual = IndexSpace("virtual", "virtual", 3)
    spec = TensorSpec((Index("row", occupied), Index("column", virtual)), role="input")
    left, right = input_tensor("left", spec), input_tensor("right", spec)
    squared = multiply(left, left)
    programs = (
        Program({"matrix": add(left, right), "energy": einsum("ij,ij->", left, right)}),
        Program({"matrix": add(squared, right), "energy": reduce_sum(squared, (0, 1))}),
    )
    pool = codegen.CudaKernelPool("probe")
    definitions = []
    cases = []
    for index, program in enumerate(programs):
        definitions.append(codegen._required_function(program, f"required_{index}"))
        for variant in ("legacy", "shared"):
            emitted = codegen._cuda_program(
                program,
                f"{variant}_{index}",
                "Outputs",
                state_type="State",
                arena_field="arena",
                input_overrides={name: f"s.{name}" for name in ("left", "right")},
                output_fields=("matrix", "energy"),
                kernel_pool=pool if variant == "shared" else None,
            )
            definitions.append(re.sub(r"<<<.*?>>>", "", emitted))
        matrix = (
            "left[flat]+right[flat]"
            if index == 0
            else "left[flat]*left[flat]+right[flat]"
        )
        energy = "left[flat]*right[flat]" if index == 0 else "left[flat]*left[flat]"
        cases.append(
            f"""{{
  const auto capacity=required_{index}(occupied,virtuals);
  std::vector<double> legacy_arena(capacity+2,54321.), shared_arena=legacy_arena;
  int legacy_error=0, shared_error=0;
  State legacy{{occupied,virtuals,legacy_arena.data()+1,&legacy_error,0,left.data(),right.data()}};
  State shared{{occupied,virtuals,shared_arena.data()+1,&shared_error,0,left.data(),right.data()}};
  auto expected=run_legacy_{index}(legacy), actual=run_shared_{index}(shared);
  assert(legacy_error==shared_error);
  assert(mode==0 || !count || shared_error!=0);
  double expected_energy=0.0;
  for(std::size_t flat=0;flat<count;++flat){{
    assert(same_bits(expected.matrix[flat],actual.matrix[flat]));
    if(mode==0) assert(actual.matrix[flat]=={matrix});
    expected_energy+={energy};
  }}
  assert(same_bits(*expected.energy,*actual.energy));
  if(mode==0) assert(shared_error==0 && *actual.energy==expected_energy);
  assert(legacy_arena.front()==54321. && legacy_arena.back()==54321.);
  assert(shared_arena.front()==54321. && shared_arena.back()==54321.);
}}"""
        )
    unit = tmp_path / "shared.cpp"
    unit.write_text(
        """
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <vector>
#define __global__
struct Dim { unsigned x; };
constexpr Dim threadIdx{0},blockIdx{0},blockDim{1},gridDim{1};
double __dadd_rn(double left,double right){return left+right;}
double __dmul_rn(double left,double right){return left*right;}
int cudaMemsetAsync(void* target,int value,std::size_t bytes,int){std::memset(target,value,bytes);return 0;}
int cudaGetLastError(){return 0;}
std::size_t checked_add(std::size_t left,std::size_t right){assert(right<=SIZE_MAX-left);return left+right;}
std::size_t checked_product(std::initializer_list<std::size_t> values){std::size_t result=1;for(auto value:values)result*=value;return result;}
namespace generativeqc_tensor {
void cuda_check(int status){assert(status==0);}
double finite(double value,int* error,int node){if(!std::isfinite(value) && !*error)*error=node+1;return value;}
}
struct State {std::size_t o,v;double* arena;int* error;int stream;const double* left;const double* right;};
struct Outputs {const double* matrix;const double* energy;};
bool same_bits(double left,double right){return std::memcmp(&left,&right,sizeof(double))==0;}
"""
        + "\n".join(definitions)
        + """
int main(){
  for(std::size_t occupied:{0,1,3})for(std::size_t virtuals:{0,1,2,7}){
    const auto count=occupied*virtuals;
    for(int mode=0;mode<4;++mode){
      std::vector<double> left(count),right(count);
      for(std::size_t flat=0;flat<count;++flat){left[flat]=(double(flat)+1)*.125;right[flat]=(double(flat)-2)*.25;}
      if(count && mode==1)left[0]=std::numeric_limits<double>::quiet_NaN();
      if(count && mode==2)right[0]=std::numeric_limits<double>::infinity();
      if(count && mode==3){left[0]=1.e308;right[0]=1.e308;}
"""
        + "\n".join(cases)
        + "\n    }\n  }\n}\n"
    )
    binary = required_native_cxx.build_executable(
        [unit],
        tmp_path / "shared",
        compile_args=("-std=c++17", "-O2", "-ffp-contract=off"),
    )
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=30)


def test_shared_kernels_keep_signed_singular_and_invalid_index_errors(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """A shared entry must report the caller's node, not its first registration."""
    scalar = TensorSpec((), role="input")
    quotient = divide(input_tensor("left", scalar), input_tensor("right", scalar))
    occupied = Index("occupied", IndexSpace("occupied", "occupied", 3))
    batch = Index("batch", IndexSpace("batch", "batch", 3))
    values = input_tensor("values", TensorSpec((occupied,), role="input"))
    mapping = input_tensor("map", TensorSpec((batch,), dtype="int64", role="input"))
    selection = runtime_indexed_select(values, ((0, mapping),), batch)
    pool = codegen.CudaKernelPool("probe")
    quotient_symbol, quotient_kernel = pool.emit(quotient)
    selection_symbol, selection_kernel = pool.emit(selection, batch_dim=True)
    unit = tmp_path / "errors.cpp"
    unit.write_text(
        """
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#define __global__
struct Dim {unsigned x;};
constexpr Dim threadIdx{0},blockIdx{0},blockDim{1},gridDim{1};
int atomicCAS(int* target,int expected,int desired){const int previous=*target;if(previous==expected)*target=desired;return previous;}
namespace generativeqc_tensor {
double finite(double value,int* error,int node){if(!std::isfinite(value))atomicCAS(error,0,node+1);return value;}
double quotient(double left,double right,int* error,int node){if(right==0.){atomicCAS(error,0,-(node+1));return 0.;}return finite(left/right,error,node);}
}
"""
        + quotient_kernel
        + selection_kernel
        + f"""
int main(){{
  for(int node:{{7,37,81}}){{
    const double numerator=4.;
    double output[3]={{54321.,-1.,54321.}};
    for(double denominator:{{2.,0.}}){{
      int error=0;
      {quotient_symbol}(&numerator,&denominator,output+1,3,1,&error,node);
      assert(error==(denominator==0. ? -(node+1) : 0));
      assert(output[1]==(denominator==0. ? 0. : 2.));
      assert(output[0]==54321. && output[2]==54321.);
    }}
    const double input[3]={{4.,5.,6.}};
    double selected[5]={{54321.,-1.,-1.,-1.,54321.}};
    std::int64_t indices[3]={{2,0,1}};
    int error=0;
    {selection_symbol}(input,indices,selected+1,3,1,3,&error,node);
    assert(error==0 && selected[1]==6. && selected[2]==4. && selected[3]==5.);
    indices[1]=-1;
    {selection_symbol}(input,indices,selected+1,3,1,3,&error,node);
    assert(error==node+1 && selected[1]==6. && selected[2]==0. && selected[3]==5.);
    assert(selected[0]==54321. && selected[4]==54321.);
  }}
}}
"""
    )
    binary = required_native_cxx.build_executable(
        [unit], tmp_path / "errors", compile_args=("-std=c++17", "-O2")
    )
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=30)


def _kernel_paths(source: str, entry: str) -> tuple[tuple[str, ...], ...]:
    """Follow the actual native ABI to workers, independent of private names."""
    functions = {}
    for match in re.finditer(
        r"(?m)^(?:static\s+)?(?:__global__\s+)?\w+\s+(\w+)\([^;{}]*\)\s*\{", source
    ):
        begin = match.end()
        depth, end = 1, begin
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        functions[match[1]] = source[begin : end - 1]

    def paths(name: str) -> tuple[tuple[str, ...], ...]:
        body = functions[name]
        launches = tuple(functions[symbol] for symbol in re.findall(r"(\w+)<<<", body))
        if launches:
            return (launches,)
        return tuple(
            branch
            for called in re.findall(r"\b(\w+)\s*\(", body)
            if called in functions
            for branch in paths(called)
        )

    result = paths(entry)
    assert result
    return result


def _iteration_scalar_reductions() -> tuple[tuple[int, object], ...]:
    program = codegen._prepare_production(
        codegen.iteration_program(*codegen.REPRESENTATIVE), "cuda"
    )
    return tuple(
        (number, node)
        for number, node in enumerate(codegen._execution_nodes(program))
        if not node.spec.indices and node.op in ("reduce", "einsum")
    )


def test_scalar_iteration_reduction_has_fixed_parallel_tree_and_serial_fallback() -> (
    None
):
    reductions = _iteration_scalar_reductions()
    assert reductions
    for number, node in reductions:
        parallel = codegen._cuda_kernel(
            node,
            number,
            "probe",
            {},
            parallel_scalar_reductions=True,
        )
        serial = codegen._cuda_kernel(node, number, "probe_serial", {})

        assert "const std::size_t reduction_count=" in parallel
        assert "if(reduction_count<32)" in parallel
        assert "#if GENERATIVEQC_TENSOR_HAS_STRICT_FP64_BLOCK_REDUCE" in parallel
        assert "for(std::size_t r=0;r<reduction_count;++r)" in parallel
        assert (
            "for(std::size_t r=threadIdx.x;r<reduction_count;r+=blockDim.x)" in parallel
        )
        assert "generativeqc::tensor::StrictFp64BlockReduce<256>" in parallel
        assert "BlockReduce::sum(sum,temp_storage)" in parallel
        assert "__dadd_rn" in parallel
        assert "StrictFp64BlockReduce" not in serial


def test_production_iteration_uses_parallel_scalar_reduction_but_replay_stays_serial() -> (
    None
):
    source = codegen.cuda_source()
    assert '#include "tensor/cuda_reduction.cuh"' in source
    for branch in _kernel_paths(source, "run_iteration_cuda"):
        assert any("StrictFp64BlockReduce<256>" in kernel for kernel in branch)
    for branch in _kernel_paths(source, "run_replay_cuda"):
        assert all("StrictFp64BlockReduce" not in kernel for kernel in branch)


def test_df_iteration_paths_enable_parallel_scalar_reduction() -> None:
    for source in (
        generate_df_ccsd_core.cuda_source(),
        generate_df_ccsd_hoisted.cuda_source(),
    ):
        for branch in _kernel_paths(source, "run_iteration_cuda"):
            assert any("StrictFp64BlockReduce<256>" in kernel for kernel in branch)


@pytest.mark.parametrize("operation", ("reduce", "einsum"))
@pytest.mark.parametrize("shared", (False, True))
def test_complete_orbital_scalar_kernel_declares_runtime_extent(
    tmp_path: Path, native_cxx: NativeCxx, operation: str, shared: bool
) -> None:
    """Compile the CuMetal branch and execute its real full-domain serial fallback."""
    space = IndexSpace("complete", "orbital", codegen.REPRESENTATIVE_ORBITALS)
    source = input_tensor("x", TensorSpec((Index("p", space),), role="input"))
    node = (
        reduce_sum(source, (0,))
        if operation == "reduce"
        else einsum("p,p->", source, source)
    )
    if shared:
        symbol, kernel = codegen.CudaKernelPool("complete").emit(
            node, parallel_scalar_reductions=True
        )
    else:
        symbol = "complete_node_0"
        kernel = codegen._cuda_kernel(
            node, 0, "complete", {}, parallel_scalar_reductions=True
        )
    arguments = ",".join(
        [
            *("values" for _ in node.inputs),
            "&out",
            "o",
            "v",
            "&error",
            *(["19"] if shared else []),
        ]
    )
    unit = tmp_path / "complete.cpp"
    unit.write_text(
        """
#include <cassert>
#include <cstddef>
#include <initializer_list>
#define __global__
#define __shared__
#define __device__
#define __forceinline__ inline
struct Dim { unsigned x; };
Dim threadIdx{0}, blockDim{256};
double __dadd_rn(double a, double b) { return a+b; }
double __dmul_rn(double a, double b) { return a*b; }
#define GENERATIVEQC_CUDA_PROVIDER_CUMETAL 1
#include "tensor/cuda_reduction.cuh"
static_assert(!GENERATIVEQC_TENSOR_HAS_STRICT_FP64_BLOCK_REDUCE);
namespace generativeqc_tensor {
double finite(double value, int*, int) { return value; }
}
"""
        + kernel
        + f"""
int main() {{
  double values[300];
  for (int i=0;i<300;++i) values[i]=(i%7)-3;
  for (std::size_t o : {{0,1,2}}) {{
    for (std::size_t v : {{0,1,5,31,32,33,255,256,257}}) {{
      const auto n=o+v;
      double expected=0.0;
      for (std::size_t i=0;i<n;++i)
        expected += {"values[i]" if operation == "reduce" else "values[i]*values[i]"};
      double out=-1.0;
      int error=0;
      {symbol}({arguments});
      assert(error==0 && out==expected);
    }}
  }}
}}
"""
    )
    executable = native_cxx.build_executable(
        [unit],
        tmp_path / "complete",
        compile_args=(
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(Path(__file__).resolve().parents[2] / "src"),
        ),
    )
    subprocess.run([str(executable)], check=True, timeout=10)
