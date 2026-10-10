"""Exercise the actual bound-point lowering and its optional runtime/ABI gates."""

from __future__ import annotations

import ctypes as ct
import re
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest
from generativeqc_compiler.dft.cuda import GridTaskView
from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.method.stationary_cuda import _STATIONARY_SCIENTIFIC_KERNELS
from generativeqc_compiler.xc.geometry_cuda import (
    _emit_restricted_point_capability,
    _emit_stationary_point,
)
from test_stationary_task_work_budget import _block

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("method", "spin", "capable"),
    [
        ("PBE0", "unpolarized", True),
        ("PBE0", "polarized", False),
        ("PBE", "unpolarized", False),
    ],
)
def test_capability_requires_exact_components_not_a_functional_label(
    method: str, spin: str, capable: bool
) -> None:
    semilocal = resolve_method(method, spin=spin).primitives[0].functional
    source = _emit_restricted_point_capability(1, semilocal)
    assert (
        f"stationary_pbe0_restricted_point_capable = {str(capable).lower()}" in source
    )
    assert ("evaluate_pbe0_restricted_bound" in source) == capable
    renamed = replace(semilocal, identifier="not-a-named-functional")
    assert source == _emit_restricted_point_capability(1, renamed)
    half_exchange = replace(
        semilocal,
        components=(("GGA_X_PBE", Fraction(1, 2)), ("GGA_C_PBE", Fraction(1))),
    )
    assert "evaluate_pbe0_restricted_bound" not in _emit_restricted_point_capability(
        1, half_exchange
    )
    assert "evaluate_pbe0_restricted_bound" not in _emit_restricted_point_capability(
        0, semilocal
    )
    assert "evaluate_pbe0_restricted_bound" not in _emit_restricted_point_capability(
        1, None
    )


def test_restricted_point_policy_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    from generativeqc import _stationary_cuda as runtime

    variable = "GENERATIVEQC_STATIONARY_PBE0_RESTRICTED_POINT"
    monkeypatch.delenv(variable, raising=False)
    assert not runtime._resolve_restricted_point_policy()
    monkeypatch.setenv(variable, "on")
    assert runtime._resolve_restricted_point_policy()
    assert not runtime._resolve_restricted_point_policy(False)
    monkeypatch.setenv(variable, "off")
    assert not runtime._resolve_restricted_point_policy()
    assert runtime._resolve_restricted_point_policy(True)
    monkeypatch.setenv(variable, "yes")
    with pytest.raises(ValueError, match="'off' or 'on'"):
        runtime._resolve_restricted_point_policy()
    with pytest.raises(TypeError, match="boolean"):
        runtime._resolve_restricted_point_policy(1)


@pytest.mark.parametrize(
    ("requested", "consumer", "producer", "flags"),
    [
        (False, True, True, 1),
        (True, False, True, 1),
        (True, True, False, 1),
        (True, True, True, 0),
        (True, True, True, 1),
    ],
)
def test_runtime_passes_only_current_producer_proof_or_uses_v1(
    requested: bool, consumer: bool, producer: bool, flags: int
) -> None:
    from generativeqc import _stationary_cuda as runtime

    owner = object.__new__(runtime._CudaSources)
    owner.handle = ct.c_void_p()
    owner.device = 0
    owner.profile_device = False
    owner.borrowed_streams = set()
    owner.restricted_point_requested = requested
    owner.library = SimpleNamespace()
    if consumer:
        owner.library.stationary_geometry_molecular_resident_weights_enqueue_v2 = Mock()
    owner._call = Mock()
    view = GridTaskView(generation=17, npoint=2, stream=123)
    work = ct.pointer(ct.c_double())
    task = SimpleNamespace(
        view=view,
        _owner=SimpleNamespace(device_id=0),
        density_jets=Mock(return_value=work),
    )
    if producer:
        task.density_jets_binding = Mock(return_value=(work, flags))
    owner.geometry_molecular_resident_weights(
        task, 0, 4, 256, None, 512, None, functional=1
    )
    arguments = owner._call.call_args.args
    bound_api = requested and consumer and producer
    assert arguments[0] == (
        "stationary_geometry_molecular_resident_weights_enqueue_v2"
        if bound_api
        else "stationary_geometry_molecular_resident_weights_enqueue"
    )
    assert arguments[2]._obj is view
    if bound_api:
        assert arguments[-2:] == (17, flags)
        task.density_jets_binding.assert_called_once_with(4)
        task.density_jets.assert_not_called()
    else:
        task.density_jets.assert_called_once_with(4)
        if producer:
            task.density_jets_binding.assert_not_called()
    assert owner.borrowed_streams == {123}


def test_point_execution_counts_are_differenced_but_capabilities_are_not() -> None:
    from generativeqc import _stationary_cuda as runtime

    required = dict.fromkeys(
        (
            "h2d_bytes",
            "d2h_bytes",
            "launches",
            "primitive_records",
            "xc_points",
            "grid_pair_visits",
            "task_descriptors",
            "task_batches",
        ),
        0,
    )
    counters = (
        "restricted_point_batches",
        "restricted_point_count",
        "general_point_batches",
        "general_point_count",
    )
    before = {**required, **dict.fromkeys(counters, 3)}
    after = {
        **required,
        **dict.fromkeys(counters, 11),
        "restricted_point_requested": 1,
        "pbe0_restricted_point_capable": 1,
    }
    delta = runtime._metric_delta(after, before)
    assert {name: delta[name] for name in counters} == dict.fromkeys(counters, 8)
    assert delta["restricted_point_requested"] == 1
    assert delta["pbe0_restricted_point_capable"] == 1
    assert not set(counters).intersection(runtime._metric_delta(required, required))


def test_native_consumer_rejects_stale_or_unknown_proofs(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    header = (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text()
    native = "\n".join(
        _block(header, marker)
        for marker in (
            "static int stationary_geometry_molecular_resident_weights_enqueue_impl(",
            "int stationary_geometry_molecular_resident_weights_enqueue(",
            "int stationary_geometry_molecular_resident_weights_enqueue_v2(",
        )
    )
    native, count = re.subn(
        r"geometry_reduce\s*<<<.*?>>>\(.*?\);", "++reductions;", native, flags=re.DOTALL
    )
    assert count == 1
    source = tmp_path / "binding.cpp"
    source.write_text(_NATIVE_PREFIX + native + _NATIVE_DRIVER)
    executable = native_cxx.build_executable(
        [source], tmp_path / "binding", compile_args=["-std=c++17", "-O2"]
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_emitted_bound_point_preserves_all_independent_references(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Run the emitted point kernel, not just the scalar helper, at the strict gate."""
    semilocal = resolve_method("PBE0", spin="unpolarized").primitives[0].functional
    setup = _block(
        _STATIONARY_SCIENTIFIC_KERNELS, "__device__ bool geometry_point_setup("
    )
    kernel = _block(
        _STATIONARY_SCIENTIFIC_KERNELS, "__global__ void geometry_point_kernel("
    )
    source = tmp_path / "point.cpp"
    source.write_text(
        _POINT_PREFIX
        + _emit_stationary_point(1, semilocal=semilocal)
        + _emit_restricted_point_capability(1, semilocal)
        + "template <bool restricted_point=false>\n"
        + setup
        + "\ntemplate <bool restricted_point=false>\n"
        + kernel
        + _POINT_DRIVER
    )
    library = ct.CDLL(
        str(
            native_cxx.build_shared(
                [source],
                tmp_path / "point.so",
                compile_args=[
                    "-std=c++17",
                    "-O2",
                    "-ffp-contract=off",
                    "-I",
                    str(ROOT / "src"),
                ],
            )
        )
    )
    pointer = ct.POINTER(ct.c_double)
    library.emitted_point.argtypes = [ct.c_bool, pointer, pointer]
    fixture = np.loadtxt(ROOT / "tests/data/xc/pbe0_restricted_point.tsv")
    assert fixture.shape == (87, 17)
    failures = []
    for index, row in enumerate(fixture):
        inputs = np.ascontiguousarray(row[:8])
        expected = row[8:]
        bound = np.empty(9)
        general = np.empty(9)
        assert (
            library.emitted_point(
                True, inputs.ctypes.data_as(pointer), bound.ctypes.data_as(pointer)
            )
            == 1
        )
        assert (
            library.emitted_point(
                False, inputs.ctypes.data_as(pointer), general.ctypes.data_as(pointer)
            )
            == 1
        )
        tolerance = 5e-10 * np.abs(expected) + 1e-322
        assert np.all(np.isfinite(bound))
        assert np.all(np.abs(bound - expected) <= tolerance), index
        if not np.all(np.abs(general - expected) <= tolerance):
            failures.append(index)
    assert failures == [8, 9, 65, 66, 67, 68]


_NATIVE_PREFIX = r"""
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
namespace generativeqc::dft {
struct GridTaskView {
  uint64_t version=1,generation=17;
  size_t npoint=2,nao=4,jets=10;
  const double *features{},*ao{},*points{};
  int stream=3;
};
}
int launches,reductions; bool proof_selected;
namespace generativeqc_stationary_cuda {
constexpr size_t stationary_ao_jets=10,stationary_xc_source=0;
struct Owner {
  size_t aos=4,atoms=12,points=8,geometry_lanes=8,geometry_peak_lanes{};
  bool geometry_pending{};
  int geometry_stream{};
  double *centers{},*partial{},*scratch{},*sources{};
  int* ao_atoms{};
  void* center_pairs{};
  struct { int* error{}; } context;
  size_t launches{},point_count{},pair_visits{},becke_pair_state_evaluations{},
      center_distance_evaluations{},geometry_batches{};
  bool retains_becke_pair_state() const { return false; }
};
template<class Function>
int guarded(Owner*,char*,size_t,Function function) {
  try { function(); return 0; } catch(const std::invalid_argument&) { return 1; }
}
bool valid_geometry_ao_map(const generativeqc::dft::GridTaskView&,size_t) { return true; }
void check(Owner&) {}
void launch_geometry(Owner&,int,generativeqc::dft::GridTaskView,const double*,
    const int*,const int*,size_t,size_t,const double*,size_t,const double*,const double*,
    const double*,size_t,size_t,size_t,double*,double*,const void*,int*,bool bound) {
  ++launches; proof_selected=bound;
}
void cuda_check(int) {}
int cudaGetLastError() { return 0; }
}
"""

_NATIVE_DRIVER = r"""
int main() {
  using namespace generativeqc_stationary_cuda;
  double value{}; int error{};
  for(int mode=0;mode<7;++mode) {
    Owner owner; owner.context.error=&error;
    generativeqc::dft::GridTaskView view;
    view.features=view.ao=view.points=&value;
    launches=reductions=0;
    const uint64_t generation=mode==2?16:17;
    const uint64_t flags=mode==3?2:mode==1?0:1;
    if(mode==4) view.features=nullptr;
    if(mode==5) view.npoint=0;
    const int status=mode==6
        ?stationary_geometry_molecular_resident_weights_enqueue(
            &owner,&view,&value,0,4,&value,&value,nullptr,0)
        :stationary_geometry_molecular_resident_weights_enqueue_v2(
            &owner,&view,&value,0,4,&value,&value,generation,flags,nullptr,0);
    assert(status==int(mode==2 || mode==3 || mode==4));
    const bool submitted=mode==0 || mode==1 || mode==6;
    assert(launches==int(submitted) && reductions==int(submitted));
    if(submitted) assert(proof_selected==(mode==0));
  }
}
"""

_POINT_PREFIX = r"""
#include <cmath>
#include <cstddef>
#include <cstdint>
#define __device__
#define __global__
#include "dft/xc_point.hpp"
using std::isfinite;
struct { size_t x{}; } threadIdx,blockIdx;
struct { size_t x=1; } blockDim;
int atomicExch(int* pointer,int value) { const int old=*pointer; *pointer=value; return old; }
constexpr unsigned stationary_functional=1,stationary_coefficients=4;
namespace generativeqc::dft {
struct GridTaskView { size_t npoint=1; const double* features{}; const int* error{}; };
}
"""

_POINT_DRIVER = r"""
extern "C" int emitted_point(bool bound,const double* input,double* output) {
  double features[10]{},scratch[36]{},seed{},weight=1,raw=1;
  int error{};
  features[0]=input[0]; features[5]=input[1];
  for(size_t axis=0;axis<3;++axis) {
    features[1+axis]=input[2+axis]; features[6+axis]=input[5+axis];
  }
  generativeqc::dft::GridTaskView view{1,features,&error};
  if(bound) geometry_point_kernel<true>(view,nullptr,0,1,4,&weight,&raw,nullptr,0,0,scratch,&seed,&error);
  else geometry_point_kernel<false>(view,nullptr,0,1,4,&weight,&raw,nullptr,0,0,scratch,&seed,&error);
  if(error) return 0;
  const auto& value=*reinterpret_cast<const StationaryPointValue*>(scratch);
  output[0]=value.energy;
  for(size_t spin=0;spin<2;++spin) {
    output[1+spin]=value.rho[spin];
    for(size_t axis=0;axis<3;++axis) output[3+3*spin+axis]=value.gradient[spin][axis];
  }
  return value.valid;
}
"""
