"""Execute current reset and typed dispatch contracts without CUDA execution.

The production host bodies are unchanged. CUDA launches become typed recording
functions so the checks cover decisions, buffer ownership and failure ordering,
not device arithmetic or endpoint qualification.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from test_stationary_task_work_budget import _block

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def assert_dispatch_contract(
    header: str, directory: Path, native_cxx: NativeCxx
) -> None:
    body = _block(header, "void launch_geometry(")
    # Only the CUDA launch syntax is replaced; arguments and template types
    # still reach the recording consumer, including scratch and seed pointers.
    body = re.sub(r"<<<.*?>>>", "", body, flags=re.DOTALL)
    source = directory / "dispatch.cpp"
    source.write_text(DISPATCH_PREFIX + body + DISPATCH_DRIVER)
    binary = native_cxx.build_executable(
        [source], directory / "dispatch", compile_args=["-std=c++20", "-O2"]
    )
    subprocess.run([binary], check=True, capture_output=True, text=True, timeout=10)


def test_current_typed_dispatch_preserves_capacity_and_failure_contracts(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    assert_dispatch_contract(
        (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text(),
        tmp_path,
        native_cxx,
    )


def assert_reset_contract(header: str, directory: Path, native_cxx: NativeCxx) -> None:
    from generativeqc_compiler.method import resolve_method
    from generativeqc_compiler.method.stationary_cuda import _runtime_layout_cuda
    from generativeqc_compiler.method.stationary_gradient import (
        SCF_POINT_MODEL,
        StationaryGradientPlan,
        StationaryMeanField,
    )
    from test_stationary_task_work_budget import PREAMBLE, _host_becke_dependencies

    preamble = (
        PREAMBLE.replace(
            "void cuda_check(int) {}",
            'void cuda_check(int e) { if(e) throw DeviceRuntimeError("driver failure"); }',
        )
        .replace(
            "int cudaMemsetAsync(void*,int,size_t,void*) { return 0; }",
            "int fences=0, sync_failure=0; void* borrowed;\n"
            "int cudaMemsetAsync(void* p,int,size_t n,void*) {\n"
            "  assert(fences==1); if(p) std::memset(p,0,n); return 0; }",
        )
        .replace(
            "int cudaStreamSynchronize(void*) { return 0; }",
            "int cudaStreamSynchronize(void* stream) {\n"
            "  assert(stream==borrowed); ++fences; return sync_failure; }",
        )
    )
    pieces = [
        _block(header, "struct Owner {") + ";",
        "template <class F>\n" + _block(header, "int guarded("),
        _block(header, "void discard_geometry("),
    ]
    for name in ("stationary_reset", "stationary_geometry_reset"):
        pieces.append(
            re.sub(
                r"\b\w+<<<.*?>>>\(.*?\);",
                "/* Device center validation is outside this host reset probe. */",
                _block(header, f"int {name}("),
                flags=re.DOTALL,
            )
        )
    plan = StationaryGradientPlan(
        resolve_method("PBE"), StationaryMeanField(SCF_POINT_MODEL)
    )
    source = directory / "reset.cpp"
    source.write_text(
        "#include <cassert>\n#include <cstring>\n"
        + _host_becke_dependencies()
        + preamble
        + _runtime_layout_cuda(plan).replace("__host__ __device__", "")
        + "using namespace generativeqc_stationary_cuda;\n"
        + "\n".join(pieces)
        + RESET_DRIVER
    )
    binary = native_cxx.build_executable(
        [source], directory / "reset", compile_args=["-std=c++20", "-O2"]
    )
    subprocess.run([binary], check=True, capture_output=True, text=True, timeout=10)


def test_current_resets_preserve_tolerance_and_fence_before_discard(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    assert_reset_contract(
        (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text(),
        tmp_path,
        native_cxx,
    )


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "geometry_cooperative_kernel<false, float>",
            "geometry_cooperative_kernel<false, double>",
        ),
        ("geometry_kernel<float>", "geometry_kernel<double>"),
        (
            "geometry_cooperative_kernel<true, double>",
            "geometry_cooperative_kernel<true, float>",
        ),
        (
            "geometry_cooperative_kernel<false, double>",
            "geometry_cooperative_kernel<false, float>",
        ),
        ("geometry_kernel<double>", "geometry_kernel<float>"),
    ],
)
def test_dispatch_rejects_precision_substitution(
    tmp_path: Path, native_cxx: NativeCxx, old: str, new: str
) -> None:
    header = (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text()
    with pytest.raises(subprocess.CalledProcessError):
        assert_dispatch_contract(header.replace(old, new, 1), tmp_path, native_cxx)


@pytest.mark.parametrize(
    "replacement",
    ["(void)stream;", "cuda_check(cudaStreamSynchronize(p.context.stream));"],
)
def test_reset_rejects_missing_or_wrong_borrowed_stream_fence(
    tmp_path: Path, native_cxx: NativeCxx, replacement: str
) -> None:
    header = (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text()
    original = _block(header, "void discard_geometry(")
    mutated = original.replace(
        "cuda_check(cudaStreamSynchronize(stream));", replacement
    )
    with pytest.raises(subprocess.CalledProcessError):
        assert_reset_contract(header.replace(original, mutated), tmp_path, native_cxx)


RESET_DRIVER = r"""
int main() {
  for(bool geometry_only:{false,true}) for(int failure:{0,71})
  for(double tolerance:{0.0,3e-10,1e-5}) {
    Owner owner; owner.topology_ready=true; owner.failed=true;
    owner.geometry_peak_lanes=17; owner.geometry_tolerance=99;
    owner.geometry_pending=true; owner.geometry_stream=reinterpret_cast<void*>(1);
    owner.context.stream=reinterpret_cast<void*>(2);
    int numerical_status=17; owner.context.error=&numerical_status;
    borrowed=owner.geometry_stream; sync_failure=failure; fences=0;
    char message[256]{};
    const int status=geometry_only
        ? stationary_geometry_reset(&owner,nullptr,tolerance,message,sizeof(message))
        : stationary_reset(&owner,nullptr,nullptr,nullptr,tolerance,message,sizeof(message));
    assert(fences==1 && !owner.geometry_pending && !owner.geometry_stream);
    if(failure) {
      assert(status==4 && owner.failed && numerical_status==17);
      assert(owner.geometry_tolerance==99 && owner.synchronizations==0);
    } else {
      assert(status==0 && !owner.failed && numerical_status==0);
      assert(owner.geometry_tolerance==tolerance && owner.geometry_peak_lanes==0);
      assert(owner.synchronizations==1);
    }
  }
}
"""


DISPATCH_PREFIX = r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <type_traits>
using std::size_t;
using cudaStream_t = void*;
struct dim3 { dim3(size_t,size_t=1,size_t=1) {} };
size_t blocks(size_t n,size_t t) { return (n+t-1)/t; }
namespace generativeqc::dft { struct GridTaskView { size_t npoint; const double* points; }; }
namespace generativeqc_grid_adjoint { struct CenterPair {}; }
struct StationaryPointValue { double words[10]; };
constexpr size_t stationary_becke_control_bytes=16;
constexpr bool stationary_restricted_point_capable=true;
bool stationary_fp32_point_capable=true, restricted_capable=true;
bool stationary_restricted_point_selected(bool requested) { return requested && restricted_capable; }
struct Owner {
  bool phased_storage{}, becke_primitive{}, becke_normalize_cooperative{}, profile{}, scaled{};
  int becke_primitive_mode{}, becke_threads_per_point=32, geometry_threads=128;
  size_t becke_shared_bytes=256, launches{}, restricted_point_batches{}, restricted_point_count{};
  size_t general_point_batches{}, general_point_count{}, becke_primitive_batches{};
  size_t becke_primitive_reverse_pair_visits{}, phased_batches{}, phased_points{};
  size_t becke_binary_scaled_batches{}, becke_binary_scaled_points{}, synchronizations{}, becke_profile_batches{};
  int becke_events[8]{}; double becke_phase_ms[7]{};
};
struct PhasedBeckeInput {
  bool normalized_adjoints{}, binary_scaled_products{};
  const double *points{}, *centers{}; const int64_t* owners{};
  size_t owner_offset{}, points_per_atom{};
  const generativeqc_grid_adjoint::CenterPair* center_pairs{};
  double *partial{}, *seeds{}; int* error{};
};
double seed_storage[16];
PhasedBeckeInput phased_input(Owner& owner,size_t) {
  PhasedBeckeInput result; result.seeds=seed_storage; result.binary_scaled_products=owner.scaled; return result;
}
int sticky{}, producer_error{}, consumers{}, producers{}, normalizations{};
bool selected_float{}, selected_cooperative{}, selected_precomputed{}, selected_bound{};
double* expected_scratch{};
int cudaPeekAtLastError() { return sticky; }
void cuda_check(int status) { if(status) throw std::runtime_error("launch failure"); }
int cudaEventSynchronize(int) { return 0; }
template<class... Args> void profile_record(Args...) {}
template<class... Args> void profile_elapsed(Args...) {}
template<class Arithmetic,class... Args> void geometry_kernel(Args... args) {
  ++consumers; selected_float=std::is_same_v<Arithmetic,float>; selected_cooperative=false;
  selected_precomputed=false;
  if constexpr(std::is_same_v<Arithmetic,double>)
    selected_bound=std::get<sizeof...(Args)-1>(std::tuple(args...));
  assert(std::get<15>(std::tuple(args...))==expected_scratch);
}
template<bool Precomputed,class Arithmetic,class... Args> void geometry_cooperative_kernel(Args... args) {
  ++consumers; selected_float=std::is_same_v<Arithmetic,float>; selected_cooperative=true;
  selected_precomputed=Precomputed;
  if constexpr(!Precomputed && std::is_same_v<Arithmetic,double>)
    selected_bound=std::get<sizeof...(Args)-1>(std::tuple(args...));
  assert(std::get<15>(std::tuple(args...))==expected_scratch);
}
template<bool Bound,class... Args> void geometry_point_kernel(Args... args) {
  ++producers; selected_bound=Bound; sticky=producer_error;
  assert(std::get<10>(std::tuple(args...))==expected_scratch);
  assert(std::get<11>(std::tuple(args...))==seed_storage);
}
template<int Phase,bool... Modes> void phased_becke_atom(PhasedBeckeInput) {}
template<bool... Modes> void phased_becke_pair(PhasedBeckeInput) {}
template<bool Scaled> void launch_phased_becke_normalize(PhasedBeckeInput input,bool,cudaStream_t) {
  ++normalizations; assert(Scaled==input.binary_scaled_products);
}
"""

DISPATCH_DRIVER = r"""
int main() {
  double scratch[32]{}, partial[32]{}, external_value=0;
  expected_scratch=scratch;
  generativeqc_grid_adjoint::CenterPair pair;
  const generativeqc::dft::GridTaskView view{7,nullptr};
  for(bool mixed:{false,true}) for(bool phased:{false,true})
  for(int threads:{1,32}) for(size_t atoms:{size_t(1),size_t(3),size_t(4),size_t(48)})
  for(bool seeded:{false,true}) for(bool lanes_match:{false,true})
  for(bool pairs:{false,true}) for(bool requested:{false,true})
  for(bool capable:{false,true}) for(bool scaled:{false,true}) {
    Owner owner; owner.phased_storage=phased; owner.becke_threads_per_point=threads; owner.scaled=scaled;
    restricted_capable=capable; stationary_fp32_point_capable=capable;
    sticky=producer_error=consumers=producers=normalizations=0;
    selected_float=selected_cooperative=selected_precomputed=selected_bound=false;
    const bool reject=mixed ? (seeded || !capable) : (phased && (!lanes_match || !pairs || threads==1));
    bool failed=false;
    try {
      launch_geometry(owner,nullptr,view,nullptr,nullptr,nullptr,0,0,nullptr,atoms,nullptr,nullptr,
          seeded?&external_value:nullptr,7,0,lanes_match?7:3,partial,scratch,pairs?&pair:nullptr,nullptr,
          requested,mixed);
    } catch(const std::invalid_argument&) { failed=true; }
    assert(failed==reject);
    if(reject) { assert(consumers==0 && producers==0); continue; }
    const bool precomputed=!mixed && phased && atoms>=4;
    assert(consumers==1 && selected_float==mixed && selected_cooperative==(threads>1));
    assert(selected_precomputed==precomputed && producers==int(precomputed));
    assert(normalizations==int(!mixed && phased));
    assert(owner.launches==(mixed?0U:7U*phased+precomputed));
    if(!mixed) assert(selected_bound==(!seeded && requested && capable));
    const bool bound=!mixed && !seeded && requested && capable;
    assert(owner.restricted_point_batches==size_t(bound));
    assert(owner.restricted_point_count==view.npoint*size_t(bound));
    assert(owner.general_point_count==view.npoint*size_t(!mixed && !bound));
  }
  // An incoming/producer error must not consume borrowed scratch or disappear.
  for(bool before:{false,true}) {
    Owner owner; owner.phased_storage=true;
    restricted_capable=true; sticky=before?11:0; producer_error=before?0:12;
    consumers=producers=0; bool failed=false;
    try { launch_geometry(owner,nullptr,view,nullptr,nullptr,nullptr,0,0,nullptr,48,nullptr,nullptr,
        nullptr,7,0,7,partial,scratch,&pair,nullptr,true,false); }
    catch(const std::runtime_error&) { failed=true; }
    assert(failed && consumers==0 && producers==int(!before) && sticky==(before?11:12));
    assert(owner.launches==0);
  }
}
"""
