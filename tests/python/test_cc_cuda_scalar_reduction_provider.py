"""Host compile contracts for provider selection, not device qualification."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tools import generate_df_ccsd_core, generate_df_ccsd_hoisted
from tools import generate_rccsd_native as codegen

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
CAPABILITY = "GENERATIVEQC_TENSOR_HAS_STRICT_FP64_BLOCK_REDUCE"


@pytest.mark.parametrize("provider", (None, "0"))
def test_nvidia_provider_preserves_cub_and_strict_add_contract(
    tmp_path: Path, required_native_cxx: NativeCxx, provider: str | None
) -> None:
    """Use a CUB API double to check dispatch and the real strict-add functor."""
    cub = tmp_path / "cub/block/block_reduce.cuh"
    cub.parent.mkdir(parents=True)
    cub.write_text(
        """
namespace cub {
enum BlockReduceAlgorithm { BLOCK_REDUCE_WARP_REDUCTIONS };
template <class T, int Threads, BlockReduceAlgorithm Algorithm>
struct BlockReduce {
  static_assert(std::is_same_v<T, double> && Threads == 256);
  static_assert(Algorithm == BLOCK_REDUCE_WARP_REDUCTIONS);
  struct TempStorage { int calls = 0; };
  TempStorage& storage;
  explicit BlockReduce(TempStorage& value) : storage(value) {}
  template <class Op> T Reduce(T value, Op op) {
    ++storage.calls;
    return op(value, 0.25);
  }
};
}
"""
    )
    source = tmp_path / "provider.cpp"
    source.write_text(
        (
            f"#define GENERATIVEQC_CUDA_PROVIDER_CUMETAL {provider}\n"
            if provider is not None
            else ""
        )
        + f"""
#include <cassert>
#include <type_traits>
#define __device__
#define __forceinline__ inline
int additions=0;
double __dadd_rn(double a, double b) {{ ++additions; return a+b; }}
#include "tensor/cuda_reduction.cuh"
static_assert({CAPABILITY});
int main() {{
  using Provider = generativeqc::tensor::StrictFp64BlockReduce<256>;
  Provider::TempStorage storage;
  assert(Provider::sum(1.0, storage)==1.25);
  assert(storage.calls==1 && additions==1);
}}
"""
    )
    binary = required_native_cxx.build_executable(
        [source],
        tmp_path / "provider",
        compile_args=(
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(tmp_path),
            "-I",
            str(ROOT / "src"),
        ),
    )
    subprocess.run([str(binary)], check=True, timeout=10)


@pytest.mark.parametrize("family", ("rccsd", "df_core", "df_scalar", "df_packed"))
def test_cumetal_iteration_scalars_match_legacy_serial_lowering(
    tmp_path: Path, required_native_cxx: NativeCxx, family: str
) -> None:
    """Execute every emitted scalar with poisoned CUB and no block primitives."""
    if family == "rccsd":
        program = codegen._prepare_production(
            codegen.iteration_program(*codegen.REPRESENTATIVE), "cuda"
        )
    elif family == "df_core":
        program = generate_df_ccsd_core.programs()["iteration"]
    elif family == "df_scalar":
        program = generate_df_ccsd_hoisted.programs()["iteration"]
    else:
        program = generate_df_ccsd_hoisted.packed_programs()["iteration"]

    cub = tmp_path / "cub/block/block_reduce.cuh"
    cub.parent.mkdir(parents=True)
    cub.write_text('#error "CuMetal must not include NVIDIA CUB"\n')
    kernels = []
    cases = []
    for number, node in enumerate(codegen._execution_nodes(program)):
        if node.spec.indices or node.op not in ("reduce", "einsum"):
            continue
        current = codegen._cuda_kernel(
            node, number, "current", {}, parallel_scalar_reductions=True
        )
        assert CAPABILITY in current
        kernels.extend((current, codegen._cuda_kernel(node, number, "legacy", {})))
        arrays = ",".join(
            f"make_data({codegen._device_size(source.spec)},{i + 1},mode)"
            for i, source in enumerate(node.inputs)
        )
        arguments = ",".join(f"data[{i}].data()+1" for i in range(len(node.inputs)))
        cases.append(
            f"""{{
  std::vector<std::vector<double>> data{{{arrays}}};
  std::array<double,3> current{{12345.,-99.,12345.}}, legacy=current;
  int current_error=0, legacy_error=0;
  // No non-owner lane may publish or enter a block collective.
  for (unsigned lane=255;lane>0;--lane) {{
    threadIdx.x=lane;
    current_node_{number}({arguments},current.data()+1,o,v,&current_error);
    assert(current[1]==-99. && current_error==0);
  }}
  threadIdx.x=0;
  current_node_{number}({arguments},current.data()+1,o,v,&current_error);
  legacy_node_{number}({arguments},legacy.data()+1,o,v,&legacy_error);
  assert(current_error==legacy_error);
  for (int i=0;i<3;++i) assert(same_bits(current[i],legacy[i]));
  for (const auto& values : data)
    assert(values.front()==54321. && values.back()==54321.);
}}"""
        )
    assert cases
    source = tmp_path / "cumetal.cpp"
    source.write_text(
        f"""
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>
#define __global__
#define GENERATIVEQC_CUDA_PROVIDER_CUMETAL 1
#include "tensor/cuda_reduction.cuh"
static_assert(!{CAPABILITY});
struct Dim {{ unsigned x; }};
Dim threadIdx{{0}};
constexpr Dim blockIdx{{0}},blockDim{{256}},gridDim{{1}};
double __dadd_rn(double a,double b) {{ return a+b; }}
double __dmul_rn(double a,double b) {{ return a*b; }}
namespace generativeqc_tensor {{
double finite(double value,int* error,int node) {{
  if (!std::isfinite(value) && !*error) *error=node+1;
  return value;
}}
}}
bool same_bits(double a,double b) {{ return std::memcmp(&a,&b,sizeof(a))==0; }}
std::vector<double> make_data(std::size_t n,unsigned seed,int mode) {{
  std::vector<double> values(n+2,54321.);
  for(std::size_t i=0;i<n;++i)
    values[i+1]=std::sin(double((i+1)*(seed+1))*.17)/(1+i%7);
  if(n && mode==1) values[1]=std::numeric_limits<double>::infinity();
  if(n && mode==2) values[1]=std::numeric_limits<double>::quiet_NaN();
  return values;
}}
"""
        + "\n".join(kernels)
        + """
int main() {
  for (const auto& shape : std::array<std::array<std::size_t,2>,10>{{
      {0,3},{1,0},{1,1},{1,31},{1,32},{1,33},{1,255},{1,256},{1,257},{3,7}}}) {
    const auto o=shape[0],v=shape[1];
    for(int mode=0;mode<3;++mode) {
"""
        + "\n".join(cases)
        + "\n    }\n  }\n}\n"
    )
    binary = required_native_cxx.build_executable(
        [source],
        tmp_path / "cumetal",
        compile_args=(
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-ffp-contract=off",
            "-fsanitize=undefined",
            "-I",
            str(tmp_path),
            "-I",
            str(ROOT / "src"),
        ),
        link_args=("-fsanitize=undefined",),
    )
    subprocess.run([str(binary)], check=True, timeout=30)
