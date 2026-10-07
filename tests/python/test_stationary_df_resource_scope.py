"""Execute DF publication admission and scope without claiming device measurements."""

from __future__ import annotations

import ast
import builtins
import ctypes as ct
import shutil
import subprocess
from pathlib import Path
from types import CodeType, FunctionType, SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc._ks_snapshot import NativeKsSnapshot

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def publication_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    source = (ROOT / "src/methods/dft_method.cpp").read_text()
    method = source[
        source.index("  generativeqc_status density_fitted_integral_gradient(") :
    ]
    admission = method[
        method.index("    if (!maximum_bytes") : method.index(
            "    std::vector<double> hcore"
        )
    ]
    capacity = method[
        method.index("    std::size_t publication_remaining_bytes") : method.index(
            "    candidate.insert("
        )
    ]
    metrics = method[
        method.index("    work[0] =") : method.index(
            "    output = std::move(candidate);"
        )
    ]
    unit = (
        r"""
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
constexpr int GENERATIVEQC_STATUS_OUT_OF_MEMORY=1;
struct Owner {
  struct { struct { std::size_t device_bytes=123; } data;
    auto diagnostic() const { return data; }
  } fock_;
  int run(std::size_t maximum_bytes, bool extra_capacity, std::array<std::uint64_t,9>& work) {
    const std::size_t coordinates=6;
    std::string detail;
"""
        + admission
        + r"""
    std::vector<double> hcore(coordinates), pulay(coordinates), candidate;
    struct { std::vector<double> coulomb, exchange; } two;
    std::size_t one_electron_device_bytes=0;
    std::size_t one_electron_h2d_bytes=0;
    std::size_t one_electron_d2h_bytes=0;
    two.coulomb.resize(coordinates); two.exchange.resize(coordinates);
    candidate.reserve(4*coordinates);
    if (extra_capacity) two.exchange.reserve(40);
"""
        + capacity
        + metrics
        + r"""
    return 0;
  }
};
int main(int argc,char** argv) {
  assert(argc==3);
  const auto budget=static_cast<std::size_t>(std::strtoull(argv[1],nullptr,10));
  const bool extra=std::atoi(argv[2]);
  std::array<std::uint64_t,9> work{};
  Owner owner;
  const auto result=owner.run(budget,extra,work);
  const auto expected=(extra ? 82U : 48U)*sizeof(double);
  assert(result == (budget < expected ? GENERATIVEQC_STATUS_OUT_OF_MEMORY : 0));
  if (!result) {
    assert(work[0]==123 && work[2]==0 && work[3]==expected);
    assert(work[4]==0 && work[5]==0); // Host fallback has no CUDA movement.
  }
}
"""
    )
    directory = tmp_path_factory.mktemp("df-publication-budget")
    cpp, executable = directory / "probe.cpp", directory / "probe"
    cpp.write_text(unit)
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(cpp),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize(
    "budget,extra",
    [(0, False), (383, False), (384, False), (384, True), (655, True), (656, True)],
)
def test_publication_budget_counts_all_live_capacities(
    publication_probe: Path, budget: int, extra: bool
) -> None:
    result = subprocess.run(
        [str(publication_probe), str(budget), str(int(extra))],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_df_bridge_keeps_nine_slot_wire_and_marks_partial_scope() -> None:
    def evaluate(
        batch: object,
        handle: object,
        output: object,
        count: int,
        budget: int,
        usage: object,
        slots: int,
    ) -> int:
        assert slots == 9 and count == 24
        np.ctypeslib.as_array(ct.cast(output, ct.POINTER(ct.c_double)), shape=(count,))[
            :
        ] = 0
        values = np.ctypeslib.as_array(
            ct.cast(usage, ct.POINTER(ct.c_uint64)), shape=(slots,)
        )
        values[:] = 0
        values[3] = 384
        return 0

    source = SimpleNamespace(
        density_fitted=True,
        check_current=lambda: None,
        _library=SimpleNamespace(
            generativeqc_ks_snapshot_density_fitted_integral_gradient_v1=evaluate
        ),
        _batch=SimpleNamespace(_batch=None, _context=None),
        _handle=None,
    )
    output, work = NativeKsSnapshot.density_fitted_integral_derivatives(source, 2, 384)
    assert output.shape == (4, 2, 3)
    assert work["compact_source_publication_host_peak_bytes"] == 384
    assert work["one_electron_h2d_bytes"] == work["one_electron_d2h_bytes"] == 0
    assert work["density_fitted_one_electron_resident_cuda"] == 0
    assert work["density_fitted_one_electron_host_contraction"] == 1
    assert work["density_fitted_response_resources_included"] == 0


def test_cuda_df_one_electron_borrows_final_stationary_weights() -> None:
    source = (ROOT / "src/methods/dft_method.cpp").read_text()
    begin = source.index("  generativeqc_status density_fitted_integral_gradient(")
    end = source.index(
        "  generativeqc_status cuda_full_range_integral_derivatives(", begin
    )
    method = source[begin:end]

    assert (
        "resident_final_stationary_weights(expected, resident_weights, detail)"
        in method
    )
    assert "execute_cuda_stationary_one_electron_pair(" in method
    assert "system_, {}, {}, 0, maximum_bytes" in method
    assert "resident_weights.density, resident_weights.weighted_density" in method
    assert "if (!resident_one_electron)" in method
    assert method.index("resident_final_stationary_weights") < method.index(
        "resident_final_fitted_projection"
    )


def test_df_python_metadata_distinguishes_resident_one_electron() -> None:
    def evaluate(
        batch: object,
        handle: object,
        output: object,
        count: int,
        budget: int,
        usage: object,
        slots: int,
    ) -> int:
        assert slots == 9 and count == 24
        np.ctypeslib.as_array(ct.cast(output, ct.POINTER(ct.c_double)), shape=(count,))[
            :
        ] = 0
        values = np.ctypeslib.as_array(
            ct.cast(usage, ct.POINTER(ct.c_uint64)), shape=(slots,)
        )
        values[:] = 0
        values[2] = 4096
        values[4] = 1024
        values[5] = 96
        return 0

    source = SimpleNamespace(
        density_fitted=True,
        check_current=lambda: None,
        _library=SimpleNamespace(
            generativeqc_ks_snapshot_density_fitted_integral_gradient_v1=evaluate
        ),
        _batch=SimpleNamespace(_batch=None, _context=None),
        _handle=None,
    )
    _, work = NativeKsSnapshot.density_fitted_integral_derivatives(source, 2, 4096)
    assert work["density_fitted_one_electron_resident_cuda"] == 1
    assert work["density_fitted_one_electron_host_contraction"] == 0
    assert work["one_electron_device_peak_bytes"] == 4096
    assert work["one_electron_h2d_bytes"] == 1024
    assert work["one_electron_d2h_bytes"] == 96


def test_cuda_outward_record_discloses_unmeasured_df_transfers() -> None:
    path = ROOT / "python/generativeqc/_stationary_cuda.py"
    source = ast.parse(path.read_text())
    owner = next(
        n
        for n in source.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_complete_rks_cuda_gradient_diagnostic"
    )
    block = next(
        n
        for n in owner.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "use_fitted_integrals"
    )
    function = ast.parse("def scope():\n    pass").body[0]
    function.body = [block]
    module = compile(
        ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
        str(path),
        "exec",
    )
    code = next(
        c for c in module.co_consts if isinstance(c, CodeType) and c.co_name == "scope"
    )
    work = {"transfer_work": {}, "host_scope": "snapshot validation"}
    FunctionType(
        code,
        {
            "__builtins__": builtins.__dict__,
            "work": work,
            "use_fitted_integrals": True,
            "native_integral_resources": {
                "density_fitted_one_electron_resident_cuda": 0
            },
        },
    )()
    assert work["density_fitted_response_resources_included"] is False
    assert work["transfer_work"]["density_fitted_response_included"] is False
    assert "excludes DF-provider" in work["additional_device_peak_bound_scope"]
    assert work["native_integral_resource_scope"].endswith("host-one-electron-fallback")
    assert "H'/S'" in work["host_scope"]

    resident = {"transfer_work": {}, "host_scope": "snapshot validation"}
    FunctionType(
        code,
        {
            "__builtins__": builtins.__dict__,
            "work": resident,
            "use_fitted_integrals": True,
            "native_integral_resources": {
                "density_fitted_one_electron_resident_cuda": 1
            },
        },
    )()
    assert resident["native_integral_resource_scope"].endswith(
        "resident-cuda-one-electron"
    )
    assert "final D/W remain resident" in resident["host_scope"]


def test_real_cuda_response_terms_require_nonzero_distinct_density_uploads(
    tmp_path: Path,
    native_cxx: NativeCxx,
) -> None:
    provider = (ROOT / "src/scf/cuda_fock_provider.cpp").read_text()
    selection = provider[
        provider.index("  const double cj = spec.coulomb.present") : provider.index(
            "  const auto staging = total.size()"
        )
    ]
    bridge = (ROOT / "src/scf/cuda/df_gradient_bridge.cu").read_text()
    start = bridge.index("      const double* densities = nullptr;")
    uploads = bridge[
        start : bridge.index("      const auto workspace_elements =", start)
    ]
    unit = (
        r"""
#include <cassert>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>
enum class FockSpin { Restricted, Unrestricted };
struct FockBuildSpec {
  FockSpin spin;
  struct { bool present; double coefficient; } coulomb, exchange;
};
struct DensityFittingDensityResponse {
  std::vector<double> density;
  double coulomb_coefficient, exchange_coefficient;
};
struct Arena {
  struct {
    std::size_t host_to_device_bytes{}, density_host_to_device_bytes{}, uploads{},
                borrowed_device_bytes{};
  } stats;
  void* stream{};
  double buffer[16]{};
  std::size_t allocations{};
  void* allocate(std::size_t bytes) {
    assert(bytes <= sizeof(buffer)); ++allocations; return buffer;
  }
};
struct BorrowedDensity { const double* density; };
namespace runtime::cuda_trace {
void trace_counter(const char*, std::size_t) {}
}
constexpr int cudaMemcpyHostToDevice=1;
int cudaMemcpyAsync(void* destination,const void* source,std::size_t bytes,int,void*) {
  std::memcpy(destination,source,bytes); return 0;
}
void check(int code) { assert(code==0); }
std::vector<double> density_upload(bool unrestricted, bool coulomb, bool borrow=false) {
  const FockBuildSpec spec{unrestricted ? FockSpin::Unrestricted : FockSpin::Restricted,
                          {coulomb,1.0},{!coulomb,-0.25}};
  const std::size_t n=2;
  std::vector<double> density(n*n,1.0),beta(n*n,2.0),out(1);
  const auto resident=density;
  const BorrowedDensity lease{resident.data()};
  const auto* borrowed_response_density=borrow ? &lease : nullptr;
"""
        + selection
        + r"""
  Arena arena;
  const auto* borrowed_density=borrowed_response_density;
"""
        + uploads
        + r"""
  assert(arena.stats.uploads==(borrow ? 0 : terms.size()));
  assert(arena.allocations==(borrow ? 0 : 1));
  assert(arena.stats.borrowed_device_bytes==(borrow ? n*n*sizeof(double) : 0));
  assert(arena.stats.host_to_device_bytes==arena.stats.density_host_to_device_bytes);
  assert(densities==(borrow ? resident.data() : arena.buffer));
  for(std::size_t term=0;term<terms.size();++term)
    for(std::size_t item=0;item<n*n;++item)
      assert(densities[term*n*n+item]==terms[term].density[item]);
  return {static_cast<double>(arena.stats.density_host_to_device_bytes)};
}
int main() {
  const auto rks_j=density_upload(false,true)[0], rks_k=density_upload(false,false)[0];
  const auto uks_j=density_upload(true,true)[0], uks_k=density_upload(true,false)[0];
  assert(rks_j==32 && rks_k==32 && uks_j==32 && uks_k==64);
  assert(rks_j+rks_k==64 && uks_j+uks_k==96);
  assert(density_upload(false,true,true)[0]==0);
  for(const auto mode : {0,1,2}) {
    bool rejected=false;
    try { density_upload(mode!=0,mode==1,true); }
    catch(const std::invalid_argument&) { rejected=true; }
    assert(rejected);
  }
}
"""
    )
    cpp, executable = tmp_path / "terms.cpp", tmp_path / "terms"
    cpp.write_text(unit)
    native_cxx.build_executable(
        [cpp],
        executable,
        compile_args=(
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
        ),
        compile_timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)
