"""Host admission coverage, not GPU numerics or generic default profitability."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from _cpp_source_support import cpp_function_definition
from generativeqc_compiler.dft.ao_cuda import emit_native_xc_point_dispatch

from tools.generate_xc_split_hybrid_registry import emit_registry

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _definition(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def admission_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: object
) -> Path:
    """Compile real point capabilities, XC layout/resource logic and SCF guard."""
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    start = source.index("    const char* ao_selection = std::getenv(")
    end = source.index("    if (host_unfused &&", start)
    guard = source[start:end]
    # No method/provider/schedule stubs: restoring a whitelist is a compile error.
    for unrelated in (
        "SemilocalFamily",
        "precision_schedule",
        "provider",
        "exchange_coefficient",
    ):
        assert unrelated not in guard
    header = (ROOT / "src/dft/cuda_xc.hpp").read_text()
    declarations = "\n".join(
        _definition(header, signature) + ";"
        for signature in (
            "enum class CudaXcAoPrecision",
            "struct CudaXcPointCapabilities",
            "struct CudaXcLayout",
            "struct CudaXcExecutionCapabilities",
            "struct CudaXcAoTiles",
            "struct CudaXcAoSelectionResources",
        )
    )
    # Keep the storage constant with the real layout body instead of duplicating
    # its value in the host-only extraction fixture.
    totals_start = header.index(
        "inline constexpr std::size_t kCudaXcTotalStorageElements ="
    )
    declarations += header[totals_start : header.index(";", totals_start) + 1]
    # Keep the exact production launcher signature in the host-storage census.
    # The CUDA stream is opaque here; this probe never calls a launcher.
    launcher_start = header.index("using CudaXcDensityLauncher =")
    launcher_end = header.index(";", launcher_start) + 1
    declarations += header[launcher_start:launcher_end]
    xc_source = (ROOT / "src/dft/cuda_xc.cpp").read_text()
    declarations += "\n" + _definition(xc_source, "struct CudaXcProgramTraits") + ";"
    definitions = "\n".join(
        _definition(xc_source, signature)
        for signature in (
            "CudaXcProgramTraits cuda_xc_program_traits(",
            "CudaXcLayout cuda_xc_layout_shape(",
            "CudaXcExecutionCapabilities cuda_xc_execution_capabilities(",
            "CudaXcLayout cuda_xc_local_ao_layout_shape(",
            "CudaXcLayout cuda_xc_local_ao_layout(",
            "CudaXcAoSelectionResources cuda_xc_ao_selection_resources(",
        )
    )
    directory = tmp_path_factory.mktemp("ks-local-ao-admission")
    # The real potential-binding header needs only an opaque stream in this
    # host admission probe; keep its production host reservation authoritative.
    (directory / "cuda_runtime_api.h").write_text(
        "#pragma once\nstruct CUstream_st; using cudaStream_t = CUstream_st*;\n"
    )
    (directory / "generated_split_hybrid_registry.cuh").write_text(emit_registry())
    unit, executable = directory / "probe.cpp", directory / "probe"
    unit.write_text(
        r"""
#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "dft/cuda_ks_precision.hpp"
#include "dft/semilocal_family.hpp"
#include "runtime/bounded_workspace.hpp"
#include "runtime/resource_ledger.hpp"
#include "tensor/cuda_symmetric_product.hpp"
#include "generated_split_hybrid_registry.cuh"
using namespace generativeqc;
using namespace generativeqc::dft;
using runtime::size_add;
using runtime::size_mul;
namespace generated = generativeqc::dft::generated;
"""
        + declarations
        + r"""
namespace cuda_xc_detail {
using CudaXcPointLauncher = void (*)();
using CudaXcPointBatchLauncher = void (*)();
template <unsigned F, bool R, bool Specialized = false> void launch_points() {}
template <unsigned Mask> void launch_split_hybrid_points() {}
template <unsigned Terms, unsigned Threads, bool Specialized = false> void launch_point_batches() {}
void launch_mixed_pbe_points() {}
void launch_mixed_pbe_point_batches() {}
}
namespace generativeqc::dft::generated {
struct AutomaticPointStub {
  void (*launch)(){};
  unsigned ingredient_mask{1};
  generativeqc::dft::CudaXcFastPathCapabilities fast_paths{};
};
const AutomaticPointStub* automatic_cuda_point_entry(std::uint32_t) { return nullptr; }
}
namespace cuda_xc_detail {
struct CudaXcPointBinding {
  CudaXcPointLauncher launch{};
  CudaXcPointBatchLauncher launch_batch{};
  const generated::AutomaticPointStub* automatic_program{};
};
"""
        + emit_native_xc_point_dispatch()
        + "\n}\n"
        + definitions
        + r"""
int main(int argc, char** argv) {
  if (argc != 7) return 2;
  if (std::string(argv[1]) == "unset") unsetenv("GENERATIVEQC_CUDA_KS_ACTIVE_AO");
  else setenv("GENERATIVEQC_CUDA_KS_ACTIVE_AO", argv[1], 1);
  const std::string family = argv[2], mode = argv[4];
  const auto functional = family == "m062x" ? generated::kM062XFunctionalCode
                          : family == "mn15" ? generated::kMN15FunctionalCode
                          : static_cast<unsigned>(std::stoul(family));
  const bool unrestricted = std::string(argv[3]) == "uks";
  const bool host_unfused = mode == "host";
  const bool automatic = std::string(argv[5]) == "auto";
  const bool nonlocal = std::string(argv[6]) == "nonlocal";
  const int device = 0;
  try {
    auto xc_layout = cuda_xc_layout_shape(
        3, 9, mode == "budget" ? 4096 : (mode == "large" ? 1664 : 12),
        mode == "budget" ? 134217728 : (mode == "large" ? 2359296 : 64),
        functional, unrestricted, 16, false, CudaXcAoPrecision::Fp64,
        mode == "scaled" ? 0.37 : 1.0, mode == "scaled" ? 0.81 : 1.0, false);
    if (mode == "response") xc_layout.response = true;
    if (mode == "fp32-ao") xc_layout.ao_precision = CudaXcAoPrecision::Fp32ComputeFp64Storage;
    if (mode == "already-local") xc_layout.local_ao = true;
    if (mode == "zero-nao") xc_layout.nao = 0;
    if (mode == "zero-points") xc_layout.npoint = 0;
    if (mode == "zero-tile") xc_layout.tile_points = 0;
    if (mode == "ledger")
      runtime::active_device_resource_ledger=std::make_shared<runtime::DeviceResourceLedger>();
    std::unique_ptr<runtime::DeviceCapacityReservation> reservation;
    std::unique_ptr<runtime::DeviceCapacityScope> capacity_scope;
    if (mode == "promise-headroom" || mode == "promise-full") {
      auto ledger = std::make_shared<runtime::DeviceResourceLedger>();
      ledger->limit = 128U << 20;
      reservation = std::make_unique<runtime::DeviceCapacityReservation>(
          ledger, mode == "promise-full" ? ledger->limit : 1U << 20);
      capacity_scope = std::make_unique<runtime::DeviceCapacityScope>(*reservation);
    }

"""
        + guard
        + r"""
    if (reservation) {
      const auto ledger = runtime::active_device_resource_ledger;
      assert(runtime::active_device_capacity_promise);
      assert(ledger->capacity_reserved ==
             (mode == "promise-full" ? ledger->limit : 1U << 20));
      assert(ledger->live == 0 && ledger->rejected == 0);
    }
    // Emulate a successful owner's selected layout without any CUDA execution.
    // Empty maps are legal; this tests the arithmetic contract, not the cutoff.
    if (admit_ao) {
      CudaXcAoTiles maps;
      maps.derivative_order = 1;
      maps.offsets.resize(ao_selection_bound.tiles + 1);
      xc_layout = cuda_xc_local_ao_layout(xc_layout, maps);
    }
    const auto schedule = resolve_cuda_ks_precision_schedule(
        automatic ? GENERATIVEQC_PRECISION_AUTO : GENERATIVEQC_PRECISION_FP64,
        xc_layout.fast_paths, false, nonlocal);
    const auto iteration = resolve_cuda_ks_iteration_precision(
        schedule, false, cuda_xc_execution_capabilities(xc_layout).mixed_density_contraction);
    std::cout << (admit_ao ? "local" : "dense") << ":"
              << iteration.uses_lower_precision(cuda_ks_precision_region::kCoulombJ) << ":"
              << iteration.uses_lower_precision(cuda_ks_precision_region::kDensityContraction);
  } catch (const std::invalid_argument&) { std::cout << "rejected"; }
}
"""
    )
    native_cxx.build_executable(
        [unit],
        executable,
        compile_args=[
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined",
            "-I",
            str(directory),
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
        ],
        link_args=["-fsanitize=undefined", "-fno-sanitize-recover=undefined"],
        compile_timeout=60,
        link_timeout=60,
    )
    return executable


def _run(
    probe: Path,
    selection: str,
    family: str = "1",
    spin: str = "rks",
    mode: str = "physical",
    precision: str = "fp64",
    nonlocal_: str = "local",
) -> str:
    return subprocess.run(
        [str(probe), selection, family, spin, mode, precision, nonlocal_],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout


@pytest.mark.parametrize("family", ["0", "1", "2", "3", "4"])
@pytest.mark.parametrize("spin", ["rks", "uks"])
@pytest.mark.parametrize("precision", ["fp64", "auto"])
@pytest.mark.parametrize("selection", ["unset", "0", "1"])
def test_capable_family_spin_matrix(
    admission_probe: Path, family: str, spin: str, precision: str, selection: str
) -> None:
    automatic = precision == "auto"
    dense = selection == "0"
    expected = f"{'dense' if dense else 'local'}:{int(automatic)}:{int(dense and automatic and int(family) < 3)}"
    assert (
        _run(admission_probe, selection, family, spin, precision=precision) == expected
    )


@pytest.mark.parametrize("spin", ["rks", "uks"])
@pytest.mark.parametrize("precision", ["fp64", "auto"])
@pytest.mark.parametrize("nonlocal_", ["local", "nonlocal"])
@pytest.mark.parametrize("selection", ["unset", "1"])
def test_scaled_pbe_is_not_an_exact_composition_gate(
    admission_probe: Path, spin: str, precision: str, nonlocal_: str, selection: str
) -> None:
    assert _run(
        admission_probe, selection, "1", spin, "scaled", precision, nonlocal_
    ) == (f"local:{int(precision == 'auto')}:0")


@pytest.mark.parametrize(
    "mode",
    [
        "host",
        "response",
        "fp32-ao",
        "already-local",
        "zero-nao",
        "zero-points",
        "zero-tile",
    ],
)
@pytest.mark.parametrize("selection", ["unset", "0", "1"])
def test_actual_layout_incompatibilities_fail_closed(
    admission_probe: Path, mode: str, selection: str
) -> None:
    assert _run(admission_probe, selection, mode=mode) == (
        "rejected" if selection == "1" else "dense:0:0"
    )


@pytest.mark.parametrize("family", ["m062x", "mn15"])
@pytest.mark.parametrize("spin", ["rks", "uks"])
@pytest.mark.parametrize("precision", ["fp64", "auto"])
@pytest.mark.parametrize("selection", ["unset", "0", "1"])
def test_split_hybrid_local_ao_does_not_admit_mixed_density(
    admission_probe: Path, family: str, spin: str, precision: str, selection: str
) -> None:
    """Qualified split consumers retain FP64 density contractions on local maps."""
    assert _run(admission_probe, selection, family, spin, precision=precision) == (
        f"{'dense' if selection == '0' else 'local'}:{int(precision == 'auto')}:0"
    )


@pytest.mark.parametrize("mode", ["budget", "ledger"])
def test_m06_map_capacity_fallback_stays_fp64(admission_probe: Path, mode: str) -> None:
    assert _run(admission_probe, "unset", "m062x", mode=mode) == "dense:0:0"


@pytest.mark.parametrize("selection", ["unset", "1"])
def test_host_budget_decline_keeps_dense_arithmetic_capability(
    admission_probe: Path, selection: str
) -> None:
    assert (
        _run(admission_probe, selection, mode="budget", precision="auto") == "dense:1:1"
    )


def test_large_grid_map_needs_only_one_host_tile(admission_probe: Path) -> None:
    """The complete device map may exceed 64 MiB while host discovery fits."""
    assert _run(admission_probe, "unset", "2", mode="large") == "local:0:0"


@pytest.mark.parametrize("selection", ["", "yes", "2", "-1", "01"])
def test_unknown_selection_fails_closed(admission_probe: Path, selection: str) -> None:
    assert _run(admission_probe, selection) == "rejected"


@pytest.mark.parametrize("selection", ["unset", "1"])
@pytest.mark.parametrize("family", ["1", "3"])
@pytest.mark.parametrize("spin", ["rks", "uks"])
@pytest.mark.parametrize("precision", ["fp64", "auto"])
def test_public_ledger_fallback_keeps_dense_arithmetic_capability(
    admission_probe: Path, selection: str, family: str, spin: str, precision: str
) -> None:
    automatic = precision == "auto"
    assert _run(admission_probe, selection, family, spin, "ledger", precision) == (
        f"dense:{int(automatic)}:{int(automatic and family == '1')}"
    )


def _independent_grid_allocation_block(source: str, signature: str) -> str:
    begin = source.index(signature)
    opening = source.index("{", begin)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


def test_independent_grid_keeps_dense_fallback_and_failure_cleanup(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Compile real arena admission and cleanup with bounded host doubles."""
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    # Isolate numeric arena admission after mandatory solver preparation and
    # before device uploads. Neither allocator outcomes nor call order is mocked.
    anchor = "sum(eigensolver->host_bytes(), eigensolver->metadata_bytes()));"
    start = source.index(anchor) + len(anchor)
    end = source.index("      check(cudaMemsetAsync(arena", start)
    early = source[start:end]
    constructor_start = source.index("        xc = std::make_unique<CudaXcPlan>(")
    constructor_end = source.index(";", constructor_start) + 1
    primary = source[constructor_start:constructor_end]
    secondary = _independent_grid_allocation_block(
        source[constructor_end:], "        if (options.nonlocal_grid) {"
    )
    cleanup = cpp_function_definition(source, "cleanup")
    unit, executable = tmp_path / "allocation.cpp", tmp_path / "allocation"
    unit.write_text(
        _INDEPENDENT_GRID_ALLOCATION_HARNESS.replace("EARLY", early)
        .replace("PRIMARY_CONSTRUCTOR", primary)
        .replace("SECONDARY", secondary)
        .replace("CLEANUP", cleanup)
    )
    native_cxx.build_executable(
        [unit], executable, compile_args=("-std=c++20", "-O0", "-I" + str(ROOT / "src"))
    )
    result = subprocess.run(
        [str(executable)], check=False, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stdout + result.stderr


_INDEPENDENT_GRID_ALLOCATION_HARNESS = r"""
#include <cassert>
#include <cstddef>
#include <iostream>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <vector>
#include "runtime/resource_ledger.hpp"
constexpr int cudaSuccess=0,cudaErrorMemoryAllocation=2,cudaErrorUnknown=9;
int cudaGetLastError(){return 0;}int cudaGetDevice(int* p){*p=0;return 0;}int cudaSetDevice(int){return 0;}
int cudaStreamSynchronize(void*){return 0;}
void check(int s){if(s==2)throw std::bad_alloc();if(s)throw s;}
std::size_t sum(std::size_t a,std::size_t b){return a+b;}
namespace runtime {
using generativeqc::runtime::DeviceUnpromisedCapacityScope;
using generativeqc::runtime::DeviceProviderAllowance;
std::size_t free_bytes;std::vector<std::size_t> requests;std::map<void*,std::size_t> owned;int failure=0;bool host_fail=false;std::size_t failing_bytes=0;
int resource_cuda_malloc(void** pointer,std::size_t bytes,bool* host_oom=nullptr){
 requests.push_back(bytes);if(host_oom)*host_oom=false;
 if(bytes==failing_bytes&&failure){if(host_oom)*host_oom=host_fail;return failure;}
 if(bytes>free_bytes)return cudaErrorMemoryAllocation;
 free_bytes-=bytes;*pointer=reinterpret_cast<void*>(requests.size()*16);owned[*pointer]=bytes;return cudaSuccess;
}
void resource_cuda_free(void* pointer){assert(owned.contains(pointer));free_bytes+=owned.at(pointer);owned.erase(pointer);}
void reset(std::size_t cap){assert(owned.empty());free_bytes=cap;requests.clear();failure=0;host_fail=false;failing_bytes=0;}
}
enum class CudaXcAoPrecision{Fp64};enum class SemilocalFamily{Pbe};int semilocal_family_code(SemilocalFamily){return 1;}
unsigned rejected_constructor=0, constructor_calls=0;
struct CudaXcPlan{template<class... Args>CudaXcPlan(Args&&...){if(++constructor_calls==rejected_constructor)throw 7;}};
struct CudaXcLayout{static constexpr std::size_t lowering_host_bytes=256;std::size_t device_bytes=0;};
struct Grid{struct View{std::size_t device_bytes=1024;explicit operator bool()const{return true;}};View cuda_view()const{return {};}};
struct Resettable{void reset(){}};
struct Fixture {
 struct {std::size_t xc_device_bytes,grid_device_bytes=0,retained_host_numeric_bytes=512;} resource;
 CudaXcLayout xc_layout{64*1024},nonlocal_xc_layout{128*1024};
 struct {std::size_t host_peak_bytes=256;} ao_selection_bound;
 bool admit_ao;bool host_unfused=false;bool borrow_resident_grid=true;
 struct{bool nonlocal_grid;double semilocal_exchange_scale=1,semilocal_correlation_scale=1;}options;
 void *xc_arena=nullptr,*nonlocal_xc_arena=nullptr,*fast_anchor_arena=nullptr,*nonlocal_arena=nullptr,*arena=nullptr;
 std::unique_ptr<CudaXcPlan> xc,nonlocal_xc,eigensolver;Resettable matrix_products;
 std::optional<runtime::DeviceProviderAllowance> matrix_provider_allowance;
 struct{void* stream=nullptr;}range_fock_binding;void retire_phase_source(bool){}
 int basis=0,spins=1,tile=64,device=0,functional=1;void* stream=nullptr;Grid grid,nonlocal_grid;
 Fixture(bool maps,bool split):resource{(maps?96u:64u)*1024},admit_ao(maps),options{split}{}
 void prepare(){try{
EARLY
PRIMARY_CONSTRUCTOR
SECONDARY
 }catch(...){cleanup();throw;}}
CLEANUP
 ~Fixture(){cleanup();}
};
int main(){

 for(bool maps:{false,true})for(bool split:{false,true}){
  runtime::reset(208*1024);bool ok=true;{
   Fixture f(maps,split);try{f.prepare();}catch(...){ok=false;}
   assert(ok && "independent grid must preserve the dense primary fallback");
   if(ok&&split){assert(f.resource.xc_device_bytes==192*1024);assert(f.resource.grid_device_bytes==1024);}
   if(maps&&split)assert(!f.admit_ao);
  }assert(runtime::owned.empty()&&runtime::free_bytes==208*1024);
  std::cout<<"success maps="<<maps<<" split="<<split<<" result="<<ok<<" cleanup=complete\n";
 }
 for(auto bytes:{64u*1024,96u*1024,128u*1024})for(int kind:{2,9})for(bool host:{false,true}){
  runtime::reset(256*1024);runtime::failure=kind;runtime::host_fail=host;runtime::failing_bytes=bytes;
  bool failed=false;{Fixture f(bytes!=64*1024,true);try{f.prepare();}catch(...){failed=true;}
   const bool optional_device_miss=bytes==96*1024&&kind==2&&!host;
   assert(failed!=optional_device_miss);
  }assert(runtime::owned.empty()&&runtime::free_bytes==256*1024);
 }
 for(unsigned target:{1u,2u}){
  runtime::reset(256*1024);rejected_constructor=target;constructor_calls=0;bool failed=false;
  {Fixture f(true,true);try{f.prepare();}catch(...){failed=true;}}
  assert(failed&&runtime::owned.empty()&&runtime::free_bytes==256*1024);
 }
 std::cout<<"12 allocator failure cases and both constructor failures preserve cleanup; host/runtime failures are not retried\n";
}
"""


@pytest.mark.parametrize(
    "mode, expected", [("promise-headroom", "local:0:0"), ("promise-full", "dense:0:0")]
)
def test_optional_ao_spends_only_unpromised_capacity(
    admission_probe: Path, mode: str, expected: str
) -> None:
    assert _run(admission_probe, "1", mode=mode) == expected
