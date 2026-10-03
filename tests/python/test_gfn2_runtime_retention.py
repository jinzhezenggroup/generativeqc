"""Retained runtime identity, fresh SCC, error recovery and deterministic cleanup."""

from __future__ import annotations

import ctypes
import gc
import os
import shutil
import subprocess
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import Calculator, Result, _native
from generativeqc._context_owner import ContextOwner

ROOT = Path(__file__).resolve().parents[2]
WATER = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 1.43, 1.1)), ("H", (0.0, -1.43, 1.1))]


def _calculator(device: str) -> Calculator:
    if device == "cuda":
        if os.environ.get("GENERATIVEQC_TEST_GFN2_CUDA") != "1":
            pytest.skip("explicit GFN2 CUDA qualification is disabled")
        if not os.environ.get("SLURM_JOB_ID"):
            pytest.fail("GFN2 CUDA qualification requires Slurm")
    return Calculator(method="gfn2-xtb", device=device, max_iterations=300)


def _same(actual: Result, expected: Result) -> None:
    assert actual.converged and expected.converged
    assert actual.iterations == expected.iterations
    assert actual.energy == pytest.approx(expected.energy, rel=0, abs=2e-12)
    if expected.forces is None:
        assert actual.forces is None
    else:
        np.testing.assert_allclose(actual.forces, expected.forces, rtol=0, atol=2e-11)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cached_singlepoints_match_cold_requests_across_identity_changes(
    device: str,
) -> None:
    calculator = _calculator(device)
    first = calculator.singlepoint(WATER)
    handle = calculator._singlepoint_context._context.value
    # Exercise geometry, property selection, element order, charge and spin,
    # then return to the original topology after every incompatible request.
    requests = [
        (WATER, {}),
        ([WATER[0], ("H", (0.02, 1.44, 1.09)), WATER[2]], {}),
        (WATER, {"properties": ("energy",)}),
        (list(reversed(WATER)), {}),
        (WATER, {"charge": 2}),
        (WATER, {"multiplicity": 3}),
        ([("O", (0.1, 0.0, 0.0)), ("H", (0.0, 0.1, 1.834))], {"multiplicity": 2}),
        (
            [("H", (-0.47, 0.815, 0)), ("H", (-0.47, -0.815, 0)), ("H", (0.94, 0, 0))],
            {"charge": 1},
        ),
    ]
    for atoms, options in requests:
        cold = _calculator(device)
        try:
            expected = cold.singlepoint(atoms, **options)
            actual = calculator.singlepoint(atoms, **options)
            _same(actual, expected)
            assert calculator._singlepoint_context._context.value == handle
            _same(calculator.singlepoint(WATER), first)
        finally:
            cold.clear_cache()
    calculator.clear_cache()
    assert not calculator._singlepoint_context._context.value
    _same(calculator.singlepoint(WATER), first)
    calculator.clear_cache()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_failed_cached_requests_do_not_poison_later_fresh_scc(device: str) -> None:
    calculator = _calculator(device)
    reference = calculator.singlepoint(WATER)
    handle = calculator._singlepoint_context._context.value
    for bad in [float("nan"), float("inf")]:
        with pytest.raises((ValueError, RuntimeError)):
            calculator.singlepoint([("O", (bad, 0, 0)), *WATER[1:]])
        _same(calculator.singlepoint(WATER), reference)
    with pytest.raises(RuntimeError, match="integral spin occupations"):
        calculator.singlepoint(WATER, multiplicity=2)
    _same(calculator.singlepoint(WATER), reference)
    # Scientific controls are normally immutable; alter the private test value
    # to exercise the native cache's options identity and nonconvergence path.
    maximum = calculator._max_iterations
    calculator._max_iterations = 1
    try:
        with pytest.raises(RuntimeError, match="SCF did not converge"):
            calculator.singlepoint(WATER)
    finally:
        calculator._max_iterations = maximum
    _same(calculator.singlepoint(WATER), reference)
    assert calculator._singlepoint_context._context.value == handle
    calculator.clear_cache()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shared_calculator_serializes_requests_and_cache_clear(device: str) -> None:
    calculator = _calculator(device)
    probes = [[WATER[0], ("H", (0.01 * i, 1.43, 1.1)), WATER[2]] for i in range(4)]
    expected = []
    for atoms in probes:
        cold = _calculator(device)
        expected.append(cold.singlepoint(atoms))
        cold.clear_cache()
    start = threading.Barrier(5)

    def execute(index: int) -> Result:
        start.wait(timeout=10)
        return calculator.singlepoint(probes[index])

    def clear() -> None:
        start.wait(timeout=10)
        calculator.clear_cache()

    with ThreadPoolExecutor(max_workers=5) as workers:
        futures = [workers.submit(execute, i) for i in range(4)]
        cleared = workers.submit(clear)
        actual = [future.result(timeout=60) for future in futures]
        cleared.result(timeout=60)
    for result, reference in zip(actual, expected, strict=True):
        _same(result, reference)
    calculator.clear_cache()


def test_python_owner_releases_exactly_once_and_rebuilds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calculator = _calculator("cpu")
    library = calculator._library
    destroy = library.generativeqc_context_destroy
    released = []

    def release(handle: ctypes.c_void_p) -> None:
        released.append(handle.value)
        destroy(handle)

    monkeypatch.setattr(library, "generativeqc_context_destroy", release)
    calculator.singlepoint(WATER)
    assert released == []
    calculator.clear_cache()
    calculator.clear_cache()
    assert len(released) == 1
    calculator.singlepoint(WATER)
    owner = weakref.ref(calculator._singlepoint_context)
    del calculator
    gc.collect()
    assert owner() is None
    assert len(released) == 2


def test_owner_failed_creation_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calculator = _calculator("cpu")
    library = calculator._library
    create = library.generativeqc_context_create
    owner = ContextOwner(library)
    descriptor = calculator._context_descriptor()
    monkeypatch.setattr(
        library,
        "generativeqc_context_create",
        lambda *args: _native.STATUS_OUT_OF_MEMORY,
    )
    with owner.lock, pytest.raises((MemoryError, RuntimeError)):
        owner.get(descriptor)
    assert not owner._context.value
    monkeypatch.setattr(library, "generativeqc_context_create", create)
    with owner.lock:
        first = owner.get(descriptor).value
        assert owner.get(descriptor).value == first
    owner.clear()


def test_owner_replaces_backend_and_device_identity() -> None:
    """Check resource retirement without opening an unscheduled CUDA device."""
    events: list[tuple[str, int]] = []
    created = 0

    def create(descriptor: object, output: object) -> int:
        nonlocal created
        created += 1
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = created
        events.append(("create", created))
        return _native.STATUS_SUCCESS

    def destroy(handle: ctypes.c_void_p) -> None:
        events.append(("destroy", handle.value))

    owner = ContextOwner(
        SimpleNamespace(
            generativeqc_context_create=create, generativeqc_context_destroy=destroy
        )
    )
    descriptor = _native.ContextDescriptor()
    descriptor.backend = _native.BACKEND_CPU_REFERENCE
    with owner.lock:
        assert owner.get(descriptor).value == 1
        assert owner.get(descriptor).value == 1
        descriptor.backend = _native.BACKEND_CUDA
        assert owner.get(descriptor).value == 2
        descriptor.device_id = 1
        assert owner.get(descriptor).value == 3
    owner.clear()
    assert events == [
        ("create", 1),
        ("destroy", 1),
        ("create", 2),
        ("destroy", 2),
        ("create", 3),
        ("destroy", 3),
    ]


def test_context_workspace_lifetime_without_runtime_oracle(tmp_path: Path) -> None:
    """Compile the real method adapter with a counted, non-scientific bridge.

    This proves the workspace survives destruction of individual calculations,
    stays private to its context, and outlives replacement while a prepared
    owner still references it. Endpoint tests independently qualify the math.
    """
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    source = tmp_path / "lifetime.cpp"
    source.write_text(r"""
#include "methods/xtb_method.hpp"
#include "methods/gfn2_runtime_bridge.hpp"
using namespace generativeqc;
namespace generativeqc::methods::detail {
static int live=0, created=0;
static std::vector<double> requests;
struct Gfn2RuntimeBridge::Impl {};
Gfn2RuntimeBridge::Gfn2RuntimeBridge(Gfn2RuntimeBackend,int):impl_(std::make_unique<Impl>()){++live;++created;}
Gfn2RuntimeBridge::~Gfn2RuntimeBridge(){--live;}
Gfn2RuntimeResult Gfn2RuntimeBridge::execute(const Gfn2RuntimeRequest& request){
  requests.insert(requests.end(),{request.positions[0],double(request.charge),
                                 double(request.maximum_iterations)});
  Gfn2RuntimeResult result;result.status=Gfn2RuntimeStatus::kSuccess;return result;
}
const char* gfn2_runtime_status_name(Gfn2RuntimeStatus) noexcept {return "stub";}
}
int main(){
  using namespace methods::detail;
  core::ContextState first,second;
  core::System system;system.atoms.push_back({1,{0.0,0.0,0.0}});
  methods::Capabilities capabilities{};
  generativeqc_method_descriptor descriptor{};
  descriptor.max_iterations=100;descriptor.diis_history=8;
  descriptor.energy_tolerance=1e-10;descriptor.density_tolerance=1e-8;
  descriptor.precision_mode=GENERATIVEQC_PRECISION_FP64;
  auto a=prepare_xtb_calculation(capabilities,first,system,descriptor);
  system.atoms[0].position[0]=2.0;system.charge=1;descriptor.max_iterations=200;
  auto b=prepare_xtb_calculation(capabilities,first,system,descriptor);
  if(created!=1 || live!=1)return 1;
  // A shared bridge must not replace the immutable request owned by either calculation.
  a->execute(false);b->execute(false);a->execute(false);
  if(requests!=std::vector<double>{0,0,100,2,1,200,0,0,100})return 9;
  a.reset();b.reset();if(live!=1)return 2;
  auto c=prepare_xtb_calculation(capabilities,first,system,descriptor);
  auto d=prepare_xtb_calculation(capabilities,second,system,descriptor);
  if(created!=2 || live!=2)return 3;
  first.workspace.reset();if(live!=2)return 4;
  c.reset();if(live!=1)return 5;
  d.reset();second.workspace.reset();if(live!=0)return 6;
  descriptor.max_iterations=0;
  try{auto invalid=prepare_xtb_calculation(capabilities,first,system,descriptor);return 7;}
  catch(const methods::MethodError&){}
  if(first.workspace || live!=0)return 8;
}
""")
    binary = tmp_path / "lifetime"
    command = [
        compiler,
        "-std=c++20",
        "-O0",
        "-I",
        str(ROOT / "src"),
        "-I",
        str(ROOT / "include"),
        str(source),
        str(ROOT / "src/methods/xtb_method.cpp"),
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    subprocess.run([str(binary)], check=True, timeout=10)
