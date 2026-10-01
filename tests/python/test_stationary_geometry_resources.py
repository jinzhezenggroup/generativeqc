"""Compiler geometry planning agrees with native allocation without a GPU."""

import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.method.stationary_resources import (
    GEOMETRY_MAX_SCRATCH_BYTES,
    plan_stationary_cuda_resources,
    stationary_cuda_allocation_bytes,
)

ROOT = Path(__file__).resolve().parents[2]
TARGET = cuda_target_info("sm_120")
SHAPE = {
    "atoms": 12,
    "aos": 96,
    "primitives": 240,
    "points": 4096,
    "tasks": 256,
    "spins": 2,
    "sources": 8,
}


def test_geometry_schedule_is_bounded_by_points_scratch_budget_and_target() -> None:
    for atoms in (1, 3, 12, 128):
        for points in (1, 17, 32, 33, 255, 256, 257, 2048, 4096):
            shape = {**SHAPE, "atoms": atoms, "points": points}
            full = plan_stationary_cuda_resources(
                **shape, target=TARGET, budget_bytes=1 << 30
            )
            assert full.geometry_lanes == min(
                points, 2048, GEOMETRY_MAX_SCRATCH_BYTES // (144 * atoms)
            )
            assert full.geometry_threads == min(full.geometry_lanes, 32)
            if points == 256:
                assert (
                    full.geometry_lanes + full.geometry_threads - 1
                ) // full.geometry_threads == 8
            assert full.geometry_scratch_bytes == 144 * atoms * full.geometry_lanes
            assert full.geometry_scratch_bytes <= GEOMETRY_MAX_SCRATCH_BYTES
            assert full.allocation_bytes == stationary_cuda_allocation_bytes(
                **shape, geometry_lanes=full.geometry_lanes
            )
            for lanes in (1, min(32, full.geometry_lanes), full.geometry_lanes):
                budget = stationary_cuda_allocation_bytes(**shape, geometry_lanes=lanes)
                tight = plan_stationary_cuda_resources(
                    **shape, target=TARGET, budget_bytes=budget
                )
                assert tight.geometry_lanes == lanes
                assert tight.allocation_bytes == budget
            with pytest.raises(ValueError, match="byte budget exceeded"):
                plan_stationary_cuda_resources(
                    **shape,
                    target=TARGET,
                    budget_bytes=stationary_cuda_allocation_bytes(
                        **shape, geometry_lanes=1
                    )
                    - 1,
                )
    limited = replace(TARGET, warp_size=16, maximum_threads_per_block=16)
    plan = plan_stationary_cuda_resources(**SHAPE, target=limited, budget_bytes=1 << 30)
    assert plan.geometry_lanes == 2048
    assert plan.geometry_threads == 16


@pytest.mark.parametrize(
    "field", ("atoms", "aos", "primitives", "points", "tasks", "spins", "sources")
)
@pytest.mark.parametrize("value", (0, -1, 1 << 64, True))
def test_geometry_resource_shapes_fail_closed(field: str, value: int) -> None:
    with pytest.raises(ValueError, match="resource caps"):
        plan_stationary_cuda_resources(
            **{**SHAPE, field: value}, target=TARGET, budget_bytes=1 << 30
        )


@pytest.mark.parametrize("budget", (-1, 1 << 64, True))
def test_geometry_budget_overflow_is_rejected(budget: int) -> None:
    with pytest.raises(ValueError, match="not representable"):
        plan_stationary_cuda_resources(**SHAPE, target=TARGET, budget_bytes=budget)


def test_native_allocation_matches_compiler_plan_and_rejects_oversized_lanes(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    header = (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text()
    function = re.search(r"size_t allocation\([^)]*\) \{.*?\n\}", header, re.DOTALL)
    assert function is not None
    checks = []
    for atoms in (1, 12, 128):
        for points in (1, 17, 256, 4096):
            shape = {**SHAPE, "atoms": atoms, "points": points}
            plan = plan_stationary_cuda_resources(
                **shape, target=TARGET, budget_bytes=1 << 30
            )
            checks.append(
                f"if(allocation({atoms},96,240,{points},256,2,{plan.geometry_lanes}) != {plan.allocation_bytes}) return 1;"
            )
    source = tmp_path / "allocation.cpp"
    source.write_text(
        """#include <cstddef>
#include <stdexcept>
#include <limits>
constexpr size_t stationary_spin_blocks=2, stationary_source_count=8;
constexpr size_t stationary_geometry_max_lanes=2048, stationary_geometry_max_scratch_bytes=8<<20;
"""
        + function[0]
        + "\nint main() {\n"
        + "\n".join(checks)
        + """
for (size_t lanes : {size_t(0),size_t(4097),std::numeric_limits<size_t>::max()}) {
  try { allocation(12,96,240,4096,256,2,lanes); return 2; }
  catch (const std::invalid_argument&) {}
}
try { allocation(128,96,240,4096,256,2,2048); return 3; }
catch (const std::invalid_argument&) {}
return 0;
}
"""
    )
    executable = tmp_path / "allocation"
    subprocess.run(
        [compiler, "-std=c++17", str(source), "-o", str(executable)],
        check=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_native_create_allocates_exact_panels_and_cleans_up_on_failure(
    tmp_path: Path,
) -> None:
    """Execute production allocation/create with explicit CUDA resource stubs."""
    from test_stationary_task_work_budget import _block

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    header = (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text()
    body = "\n".join(
        (
            _block(header, "struct Owner {") + ";",
            _block(header, "size_t allocation("),
            "template <class F>\n" + _block(header, "int guarded("),
            _block(header, "int stationary_create("),
        )
    )
    source = tmp_path / "create.cpp"
    source.write_text(
        r"""
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
using cudaEvent_t=void*;
using cudaStream_t=void*;
namespace generativeqc_stationary_cuda {}
constexpr size_t stationary_spin_blocks=2, stationary_source_count=8;
constexpr size_t stationary_geometry_max_lanes=2048, stationary_geometry_max_threads=32;
constexpr size_t stationary_geometry_max_scratch_bytes=8<<20, task_stride=9;
int allocations=0, owners=0, max_threads=1024;
bool oom=false;
struct Context {
  unsigned char* arena{};
  Context() { ++owners; }
  ~Context() { delete[] arena; --owners; }
  void prepare(int,int,int,size_t bytes,size_t error_offset,size_t,size_t,size_t,bool) {
    ++allocations;
    if(error_offset != bytes-256) throw std::runtime_error("bad error boundary");
    if(oom) throw std::bad_alloc();
    arena=new unsigned char[bytes];
  }
};
struct cudaDeviceProp { int maxThreadsPerBlock{}, maxThreadsDim[3]{}, maxGridSize[3]{}; };
int cudaGetDeviceCount(int* count) { *count=1; return 0; }
int cudaGetDeviceProperties(cudaDeviceProp* p,int) {
  p->maxThreadsPerBlock=max_threads; p->maxThreadsDim[0]=max_threads;
  p->maxGridSize[0]=65535; return 0;
}
void cuda_check(int status) { if(status) throw std::runtime_error("cuda failure"); }
void error_text(char* out,size_t size,const char* message) { std::snprintf(out,size,"%s",message); }
"""
        + body
        + r"""
int main() {
  const size_t bytes=allocation(12,96,240,4096,256,2,256);
  char error[256]{};
  void* result=reinterpret_cast<void*>(1);
  auto create=[&](size_t budget) {
    return stationary_create(0,12,0,12,96,240,4096,256,2,1000000,budget,256,32,
                             &result,error,sizeof(error));
  };
  if(!create(bytes-1) || result || allocations || owners) return 1;
  max_threads=16;
  if(!create(bytes) || result || allocations || owners) return 2;
  max_threads=1024; oom=true;
  if(!create(bytes) || result || allocations!=1 || owners) return 3;
  oom=false;
  if(create(bytes) || !result || allocations!=2 || owners!=1) return 4;
  auto* p=static_cast<Owner*>(result);
  if(p->bytes!=bytes || p->geometry_lanes!=256 || p->geometry_threads!=32) return 5;
  if(p->scratch-p->partial!=256*9*12 || p->sources-p->scratch!=256*9*12) return 6;
  if(reinterpret_cast<unsigned char*>(p->weighted_density+2*96*96)-p->context.arena != bytes-256)
    return 7;
  delete p;
  return owners ? 8 : 0;
}
"""
    )
    executable = tmp_path / "create"
    subprocess.run(
        [compiler, "-std=c++17", str(source), "-o", str(executable)],
        check=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_optional_lanes_preserve_existing_native_admission_budget() -> None:
    from generativeqc_compiler.method.stationary_resources import (
        stationary_native_pair_reserve,
    )

    shape = {
        **SHAPE,
        "atoms": 3,
        "aos": 7,
        "primitives": 21,
        "points": 256,
        "tasks": 64,
        "spins": 1,
        "sources": 7,
    }
    minimum = stationary_cuda_allocation_bytes(**shape, geometry_lanes=32)
    needed = stationary_native_pair_reserve(atoms=3, aos=7, primitives=21)
    for available in (1, 143, 144, 432, 433, needed - 1, needed, needed + 432, 1 << 20):
        reserve = min(available, needed)
        budget = minimum + available
        plan = plan_stationary_cuda_resources(
            **shape, target=TARGET, budget_bytes=budget - reserve
        )
        assert budget - plan.allocation_bytes >= reserve
        if available <= needed:
            assert plan.geometry_lanes == 32
            assert budget - plan.allocation_bytes == available
        else:
            assert plan.geometry_lanes > 32


def test_native_pair_reserve_matches_actual_native_admission(tmp_path: Path) -> None:
    from generativeqc_compiler.method.stationary_resources import (
        stationary_native_pair_reserve,
    )

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    native = (ROOT / "src/scf/cuda/one_electron_gradient_bridge.cu").read_text()
    pair = native.split(
        "generativeqc_status execute_cuda_stationary_one_electron_pair(", 1
    )[1]
    expression = pair[
        pair.index("  constexpr long double per_ao =") : pair.index(
            "  if (host_bound > maximum_bytes)"
        )
    ]
    basis = (ROOT / "src/molecule/basis.hpp").read_text()
    terms = re.search(
        r"inline constexpr std::size_t kMaximumAoExpansionTerms = \d+;", basis
    )
    assert terms is not None
    checks = []
    for atoms, aos, primitives in (
        (1, 1, 3),
        (3, 7, 21),
        (12, 96, 240),
        (128, 1024, 16384),
    ):
        checks.append(
            f"if(bound({atoms},{aos},{primitives}) != {stationary_native_pair_reserve(atoms=atoms, aos=aos, primitives=primitives)}) return 1;"
        )
    source = tmp_path / "native-reserve.cpp"
    source.write_text(
        "#include <cstddef>\n#include <cstdint>\nnamespace molecule {"
        + terms[0]
        + "}\nsize_t bound(size_t atoms,size_t n,size_t primitives) {\n"
        + expression
        + "return size_t(host_bound);\n}\nint main() {\n"
        + "\n".join(checks)
        + "\n}\n"
    )
    executable = tmp_path / "native-reserve"
    subprocess.run(
        [compiler, "-std=c++17", str(source), "-o", str(executable)],
        check=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)
