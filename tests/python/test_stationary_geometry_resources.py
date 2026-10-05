"""Compiler geometry planning agrees with native allocation without a GPU."""

import ast
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import CodeType, FunctionType, SimpleNamespace

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


@pytest.mark.parametrize("architecture", ["sm_80", "sm_89", "sm_120"])
@pytest.mark.parametrize("atoms", [1, 3, 6, 11, 12, 24, 32, 33, 48, 96, 128])
def test_automatic_cooperation_is_limited_to_qualified_target_and_shape(
    architecture: str, atoms: int
) -> None:
    target = cuda_target_info(architecture)
    shape = {**SHAPE, "atoms": atoms, "points": 256}
    automatic = plan_stationary_cuda_resources(
        **shape, target=target, budget_bytes=1 << 30
    )
    generic = plan_stationary_cuda_resources(
        **shape, target=target, budget_bytes=1 << 30, cooperative_becke=False
    )
    expected_threads = 32 if architecture == "sm_120" and atoms >= 12 else 1
    assert automatic.becke_threads_per_point == expected_threads
    assert generic.becke_threads_per_point == 1
    assert automatic.allocation_bytes == generic.allocation_bytes
    assert automatic.geometry_lanes == generic.geometry_lanes
    for limited in (
        replace(target, warp_size=16, maximum_threads_per_block=16),
        replace(target, tuning_maximum_shared_bytes=16),
    ):
        fallback = plan_stationary_cuda_resources(
            **shape, target=limited, budget_bytes=1 << 30
        )
        assert fallback.becke_threads_per_point == 1
        assert fallback.allocation_bytes == generic.allocation_bytes


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
                **shape, geometry_lanes=full.geometry_lanes, cache_center_geometry=True
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
    for atoms, aos, primitives in (
        (1, 96, 240),
        (12, 96, 240),
        (24, 192, 176),
        (48, 384, 352),
        (96, 768, 704),
        (128, 1024, 16384),
    ):
        for points in (1, 17, 256, 4096):
            shape = {
                **SHAPE,
                "atoms": atoms,
                "aos": aos,
                "primitives": primitives,
                "points": points,
            }
            plan = plan_stationary_cuda_resources(
                **shape, target=TARGET, budget_bytes=1 << 30
            )
            checks.append(
                f"if(allocation({atoms},{aos},{primitives},{points},256,2,{plan.geometry_lanes},true) != {plan.allocation_bytes}) return 1;"
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
namespace generativeqc_grid_adjoint { struct CenterPair { double values[6]; }; }
namespace generativeqc::runtime {
template <class Element> struct OwnedCudaBuffer {
  explicit operator bool() const { return false; }
};
}
constexpr size_t stationary_spin_blocks=2, stationary_source_count=8;
constexpr size_t stationary_becke_retained_max_atoms=32;
constexpr size_t stationary_geometry_max_lanes=2048, stationary_geometry_max_threads=32;
constexpr size_t stationary_geometry_max_scratch_bytes=8<<20, task_stride=9;
int allocations=0, owners=0, arenas=0, max_threads=1024;
bool oom=false, runtime_failure=false;
size_t oom_above=std::numeric_limits<size_t>::max();
struct DeviceAllocationError : std::runtime_error { using std::runtime_error::runtime_error; };
int cudaGetLastError() { return 0; }
struct Context {
  unsigned char* arena{};
  Context() { ++owners; }
  ~Context() { if(arena) --arenas; delete[] arena; --owners; }
  void prepare(int,int,int,size_t bytes,size_t error_offset,size_t,size_t,size_t,bool) {
    ++allocations;
    if(error_offset != bytes-256) throw std::runtime_error("bad error boundary");
    if(oom) throw std::bad_alloc();
    if(runtime_failure) throw std::runtime_error("injected non-allocation failure");
    arena=new unsigned char[bytes]; ++arenas;
    if(bytes>oom_above) throw DeviceAllocationError("injected post-allocation cache OOM");
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
  if(p->bytes!=bytes || p->byte_budget!=bytes || p->geometry_lanes!=256 || p->geometry_threads!=32) return 5;
  if(p->scratch-p->partial!=256*9*12 || p->sources-p->scratch!=256*9*12) return 6;
  if(reinterpret_cast<unsigned char*>(p->weighted_density+2*96*96)-p->context.arena != bytes-256)
    return 7;
  if(p->center_pairs || p->center_geometry_bytes) return 8;
  delete p;
  constexpr size_t cache_bytes=48*(12*11/2);
  if(create(bytes+cache_bytes-1) || !result) return 9;
  p=static_cast<Owner*>(result);
  if(p->bytes!=bytes || p->center_pairs || p->center_geometry_bytes) return 10;
  delete p;
  if(create(bytes+cache_bytes) || !result) return 11;
  p=static_cast<Owner*>(result);
  if(p->bytes!=bytes+cache_bytes || !p->center_pairs || p->center_geometry_bytes!=cache_bytes) return 12;
  if(reinterpret_cast<unsigned char*>(p->weighted_density+2*96*96)-p->context.arena != p->bytes-256)
    return 13;
  if(reinterpret_cast<double*>(p->center_pairs)!=p->centers+3*12 ||
     reinterpret_cast<double*>(p->center_pairs)+cache_bytes/8!=p->weights) return 14;
  delete p;
  oom_above=bytes;
  if(create(bytes+cache_bytes) || !result) return 15;
  p=static_cast<Owner*>(result);
  if(p->bytes!=bytes || p->center_pairs || p->center_geometry_bytes || owners!=1) return 16;
  delete p;
  if(arenas) return 17;
  runtime_failure=true;
  const int prior_allocations=allocations;
  if(!create(bytes+cache_bytes) || result || owners || arenas || allocations!=prior_allocations+1) return 18;
  runtime_failure=false;
  oom_above=std::numeric_limits<size_t>::max();
  const size_t large_bytes=allocation(96,768,704,256,4096,2,256);
  auto large=[&](size_t budget) {
    return stationary_create(0,12,0,96,768,704,256,4096,2,16000000,budget,256,32,
                             &result,error,sizeof(error));
  };
  const int before_large=allocations;
  if(!large(large_bytes-1) || result || allocations!=before_large || owners) return 19;
  if(large(large_bytes) || !result || allocations!=before_large+1 || owners!=1) return 20;
  p=static_cast<Owner*>(result);
  if(p->bytes!=large_bytes || p->atoms!=96 || p->aos!=768) return 21;
  if(reinterpret_cast<unsigned char*>(p->weighted_density+2*768*768)-p->context.arena != large_bytes-256)
    return 22;
  delete p;
  if(arenas || owners) return 23;
  return 0;
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


@pytest.mark.parametrize("cooperative", [False, True])
@pytest.mark.parametrize("direct_available", [False, True])
@pytest.mark.parametrize("allowance", [432, 433, 1 << 20])
def test_fitted_provider_retains_its_full_admitted_allowance(
    direct_available: bool, allowance: int, cooperative: bool
) -> None:
    from generativeqc_compiler.method.stationary_resources import (
        stationary_native_pair_reserve,
    )

    path = ROOT / "python/generativeqc/_stationary_cuda.py"
    owner = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_plan_stationary_cuda_tile"
    )
    reservation = next(
        node
        for node in owner.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "native_geometry_reserve"
            for target in node.targets
        )
    )
    source = SimpleNamespace(
        density_fitted=True, density_fitted_integral_derivatives=lambda *_: None
    )
    if direct_available:
        source.cuda_integral_derivatives = lambda *_: None
    scope = {
        "state": SimpleNamespace(_source=source),
        "available": allowance + 1024,
        "tensor_plans": {"other": SimpleNamespace(peak_bytes=1024)},
        "ecp": False,
        "na": SHAPE["atoms"],
        "n": SHAPE["aos"],
        "basis": SimpleNamespace(nprimitive=SHAPE["primitives"]),
        "stationary_native_pair_reserve": stationary_native_pair_reserve,
    }
    function = ast.parse("def reserve():\n    pass").body[0]
    function.body = [ast.Return(value=reservation.value)]
    code = compile(
        ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
        str(path),
        "exec",
    )
    compiled = next(
        item
        for item in code.co_consts
        if isinstance(item, CodeType) and item.co_name == "reserve"
    )
    reserve = FunctionType(compiled, scope)()
    assert reserve == allowance
    minimum = stationary_cuda_allocation_bytes(**SHAPE, geometry_lanes=32)
    plan = plan_stationary_cuda_resources(
        **SHAPE,
        target=TARGET,
        budget_bytes=minimum + allowance - reserve,
        cooperative_becke=cooperative,
    )
    assert plan.geometry_lanes == 32
    assert plan.becke_threads_per_point == (32 if cooperative else 1)
    assert plan.becke_shared_bytes == (4240 if cooperative else 0)
    assert minimum + allowance - plan.allocation_bytes == allowance


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
        (24, 192, 176),
        (48, 384, 352),
        (96, 768, 704),
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


def test_center_cache_uses_spare_budget_without_reducing_lanes() -> None:
    shape = {**SHAPE, "points": 17, "atoms": 12}
    base = stationary_cuda_allocation_bytes(**shape, geometry_lanes=17)
    cache = 48 * 66
    for extra, expected in ((0, 0), (cache - 1, 0), (cache, cache), (cache + 1, cache)):
        plan = plan_stationary_cuda_resources(
            **shape, target=TARGET, budget_bytes=base + extra
        )
        assert plan.geometry_lanes == 17
        assert plan.center_geometry_bytes == expected
        assert plan.allocation_bytes == base + expected
    with pytest.raises(ValueError, match="boolean"):
        stationary_cuda_allocation_bytes(
            **shape, geometry_lanes=17, cache_center_geometry=1
        )
