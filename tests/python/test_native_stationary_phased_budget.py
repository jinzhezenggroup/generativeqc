"""Host admission contracts and allocation-gated forces across compiled profiles."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.method.stationary_cuda import _qualified_aot_profile
from generativeqc_compiler.method.stationary_native_cuda import (
    NATIVE_STATIONARY_CUDA_PROFILES,
    emit_native_stationary_profile_cuda,
)

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


def test_pair_state_accounting_tracks_executed_schedule(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Reserved FP64 phase panels must not undercount bounded FP32 pair work."""
    body = _definition(
        (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text(),
        "bool retains_becke_pair_state(",
    )
    source = tmp_path / "pair_work.cpp"
    source.write_text(
        "#include <cassert>\n#include <cstddef>\n#include <initializer_list>\n"
        "constexpr size_t stationary_becke_retained_max_atoms=32;\n"
        "struct Owner { bool phased_storage; size_t atoms,becke_threads_per_point;\n"
        + body
        + "\n};\n"
        + r"""
int main() {
  for(size_t atoms:{size_t(3),size_t(32),size_t(33),size_t(48),size_t(128)})
    for(size_t threads:{size_t(1),size_t(32)})
      for(bool phased:{false,true}) for(bool fp32:{false,true}) {
        const Owner owner{phased,atoms,threads};
        const size_t expected = (threads>1 && atoms<=32) || (phased && !fp32) ? 1 : 2;
        assert((owner.retains_becke_pair_state(fp32)?1U:2U)==expected);
        if(!fp32) assert(owner.retains_becke_pair_state()==owner.retains_becke_pair_state(false));
      }
}
"""
    )
    binary = native_cxx.build_executable(
        [source], tmp_path / "pair_work", compile_args=["-std=c++20"]
    )
    subprocess.run([binary], check=True, timeout=10)


@pytest.mark.parametrize("profile", NATIVE_STATIONARY_CUDA_PROFILES)
def test_cached_phase_reservation_and_bounded_fallback(
    profile: str, tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Qualify actual generated policy and allocation arithmetic, not source hashes."""
    generated = emit_native_stationary_profile_cuda(profile)
    plan = _qualified_aot_profile(profile).plan
    policy_begin = generated.index("constexpr size_t native_phased_becke_min_atoms =")
    policy = generated[policy_begin : generated.index(";", policy_begin) + 1]
    runtime = (ROOT / "src/dft/native_stationary_runtime.cuh").read_text()
    executor = (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text()
    definitions = "\n".join(
        (
            _definition(executor, "size_t phased_allocation("),
            _definition(executor, "size_t allocation("),
            _definition(runtime, "size_t native_phased_becke_bytes("),
            _definition(runtime, "size_t native_numeric_bytes("),
        )
    )
    source = tmp_path / "phase_capacity.cpp"
    source.write_text(
        """
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <limits>
#include "runtime/bounded_workspace.hpp"
using std::size_t;
struct uint2 { unsigned first, second; };
constexpr size_t stationary_becke_retained_max_atoms = 32;
constexpr size_t stationary_becke_max_atoms = 128;
constexpr size_t stationary_geometry_max_lanes = 1024;
constexpr size_t stationary_geometry_max_scratch_bytes = 64U << 20;
"""
        + f"constexpr size_t stationary_spin_blocks = {plan.spin_blocks}, "
        + f"stationary_source_count = {len(plan.source_names)};\n"
        + policy
        + "\n"
        + definitions
        + """
int main() {
  for (size_t atoms : {2U, 24U, 32U, 47U, 48U, 64U, 96U, 128U})
    for (size_t points : {64U, 128U, 256U, 512U})
      for (size_t lanes : {size_t{1}, points}) {
        const size_t pairs = atoms * (atoms - 1) / 2;
        const size_t phase = atoms >= 48 && lanes == points
            ? 8 * (4 * pairs * points + (12 * atoms + 2) * points + pairs) : 0;
        const size_t minimum = allocation(atoms, 224, 672, points, 1,
                                          stationary_spin_blocks, lanes, false);
        const size_t centers = 48 * pairs;
        assert(native_phased_becke_bytes(atoms, points, lanes) == phase);
        assert(native_numeric_bytes(atoms, 224, 672, points, 1,
                                    stationary_spin_blocks, lanes, false) == minimum);
        assert(native_numeric_bytes(atoms, 224, 672, points, 1,
                                    stationary_spin_blocks, lanes, true) ==
               minimum + centers + phase);
        const size_t bounded_budget = minimum + centers;
        const size_t cached_budget = bounded_budget + phase;
        assert(cached_budget >= bounded_budget);
        if (phase) assert(cached_budget - 1 < minimum + centers + phase);
      }
  assert(native_phased_becke_bytes(129, 256, 256) == 0);
  assert(native_numeric_bytes(129, 224, 672, 256, 1,
                              stationary_spin_blocks, 256, true) > 0);
  bool rejected = false;
  try { (void)native_numeric_bytes(std::numeric_limits<size_t>::max(), 224, 672,
                                  256, 1, stationary_spin_blocks, 256, true); }
  catch (const std::overflow_error&) { rejected = true; }
  assert(rejected);
}
"""
    )
    executable = native_cxx.build_executable(
        [source],
        tmp_path / "phase_capacity",
        compile_args=["-std=c++20", "-O0", f"-I{ROOT / 'src'}"],
    )
    subprocess.run([str(executable)], check=True, timeout=10)


@pytest.mark.parametrize(
    ("profile", "native_admitted", "force_tile_points"),
    [
        pytest.param(
            profile,
            True,
            None,
            id=profile,
        )
        for profile in NATIVE_STATIONARY_CUDA_PROFILES
    ]
    + [
        pytest.param(profile, True, 1024, id=profile + "-force1024")
        for profile in ("pbe_rks", "pbe0_rks", "wb97mv_rks")
    ]
    + [
        # The public upper bound exceeds the geometry lane/scratch cap and
        # must still execute the complete force through its bounded fallback.
        pytest.param("pbe0_rks", True, 4096, id="pbe0_rks-force4096")
    ],
)
@pytest.mark.parametrize("atom_count", (48, 96, 128))
def test_phase_force_matches_energy_derivatives_and_existing_fallback(
    profile: str, native_admitted: bool, force_tile_points: int | None, atom_count: int
) -> None:
    """Every emitted program keeps moved forces at cached and minimum capacity."""
    if os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly Slurm- or LSF-allocated GPU")
    assert os.environ.get("SLURM_JOB_ID") or os.environ.get("LSB_JOBID")
    import ctypes

    import numpy as np
    from generativeqc import Calculator, GridSpec, KsOptions, _native

    atoms = []
    for pair in range(atom_count // 2):
        center = (8.0 * (pair % 4), 8.0 * ((pair // 4) % 4), 8.0 * (pair // 16))
        for offset in (-0.7, 0.7):
            atoms.append(("H", (center[0], center[1], center[2] + offset)))
    multiplicity = 2 if profile.endswith("_uks") else 1
    if multiplicity == 2:
        # A single HeH radical breaks equal-spin symmetry without making all
        # separated H2 fragments degenerate open-shell triplets.
        atoms[-1] = ("He", atoms[-1][1])
    scf_tile_points = 128 if force_tile_points else 256
    calculator = Calculator(
        # Compiler profile identifiers are not public method aliases (e.g.
        # wb97mv_rks). Exercise the exact scientific graph being qualified.
        method=_qualified_aot_profile(profile).plan.method,
        basis="sto-3g",
        device="cuda",
        device_id=int(os.environ.get("GENERATIVEQC_TEST_DEVICE_ID", "0")),
        ks_options=KsOptions(
            grid=GridSpec(12, 4, 8),
            tile_points=scf_tile_points,
            force_tile_points=force_tile_points,
        ),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        max_iterations=200,
    )
    positions = np.asarray([coordinates for _, coordinates in atoms])
    with calculator.prepare_batch(
        [atoms], multiplicities=[multiplicity], warm_start=True
    ) as batch:
        assert batch._native_dft_force_eligible(0) is native_admitted
        original = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        positions[-1] += (0.002, -0.001, 0.003)
        moved = batch.execute(
            [positions], strict=True, properties=("energy", "forces")
        ).items[0]
        assert original.converged and moved.converged
        assert (
            original.ks_diagnostic.tile_points
            == moved.ks_diagnostic.tile_points
            == scf_tile_points
        )
        # Omitted coordinates select the original prepared geometry.
        warm = batch.execute(
            [positions], strict=True, properties=("energy", "forces")
        ).items[0]
        np.testing.assert_allclose(warm.forces, moved.forces, atol=1e-7, rtol=0)
        if not native_admitted:
            work = batch._stationary_cuda_execution.sources.metrics()
            assert work["phased_becke_bytes"] > 0
            assert work["becke_phase_batches"] > 0
            assert work["becke_phase_points"] > 0
        direction = np.zeros_like(positions)
        direction[-1] = (0.3, -0.2, 0.4)
        analytic = -float(np.sum(moved.forces * direction))
        for step in (2e-4, 1e-4):
            plus = batch.execute(
                [positions + step * direction], strict=True, properties=("energy",)
            ).items[0]
            minus = batch.execute(
                [positions - step * direction], strict=True, properties=("energy",)
            ).items[0]
            np.testing.assert_allclose(
                (plus.energy - minus.energy) / (2 * step), analytic, atol=1e-7, rtol=0
            )
    displaced = [
        (symbol, tuple(position))
        for (symbol, _), position in zip(atoms, positions, strict=True)
    ]
    with calculator.prepare_batch(
        [displaced], multiplicities=[multiplicity], warm_start=False
    ) as batch:
        if native_admitted:
            query = batch._library.generativeqc_ks_batch_stationary_cuda_capacity_v1
            query.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
            ]
            query.restype = ctypes.c_int
            capacity = (ctypes.c_uint64 * 9)()
            _native.check(
                batch._library, query(batch._batch, 0, capacity, len(capacity))
            )
            assert capacity[5] > capacity[4]
            assert capacity[6] == (force_tile_points or scf_tile_points)
            # A dry plan and the prepared owner must describe the same force
            # arena even when SCF uses a different tile. These STO-3G fixtures
            # have one s AO and three primitives per H/He center.
            shape = (ctypes.c_uint64 * 7)(
                atom_count, atom_count, 3 * atom_count, atom_count * 12 * 4 * 8, 0, 0, 0
            )
            dry = batch._library.generativeqc_resource_ks_stationary_cuda_v1
            dry.argtypes = [
                ctypes.POINTER(_native.MethodDescriptor),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
            ]
            dry.restype = ctypes.c_int
            descriptor = calculator._method_descriptor(systems=(displaced,))
            planned = (ctypes.c_uint64 * 9)()
            _native.check(
                batch._library,
                dry(ctypes.byref(descriptor), shape, len(shape), planned, len(planned)),
            )
            assert tuple(planned) == tuple(capacity)
            descriptor.ks_options.contents.force_tile_points = 0
            _native.check(
                batch._library,
                dry(ctypes.byref(descriptor), shape, len(shape), planned, len(planned)),
            )
            assert planned[6] == scf_tile_points
            limit = batch._library.generativeqc_ks_batch_stationary_cuda_limit_v1
            limit.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64]
            limit.restype = ctypes.c_int
            assert (
                limit(batch._batch, 0, capacity[4] - 1) == _native.STATUS_OUT_OF_MEMORY
            )
            _native.check(batch._library, limit(batch._batch, 0, capacity[4]))
        assert batch._native_dft_force_eligible(0) is native_admitted
        bounded = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        assert bounded.converged
        np.testing.assert_allclose(bounded.energy, moved.energy, atol=1e-9, rtol=0)
        np.testing.assert_allclose(bounded.forces, moved.forces, atol=1e-7, rtol=0)


def test_phase_creation_respects_budget_and_preserves_failure_cleanup(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Execute real creation/admission with only CUDA allocation/configuration doubled."""
    runtime = (ROOT / "src/dft/native_stationary_runtime.cuh").read_text()
    executor = (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text()
    definitions = "\n".join(
        (
            _definition(executor, "size_t phased_allocation("),
            _definition(runtime, "size_t native_phased_becke_bytes("),
            _definition(executor, "template <class F>\nint guarded("),
            _definition(runtime, "void native_require_status("),
            _definition(runtime, "int native_create("),
        )
    )
    source = tmp_path / "phase_creation.cpp"
    source.write_text(
        """
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
using std::size_t;
struct uint2 { unsigned first, second; };
struct DeviceAllocationError : std::runtime_error { using std::runtime_error::runtime_error; };
struct DeviceRuntimeError : std::runtime_error { using std::runtime_error::runtime_error; };
struct Context { void check_device() {} };
struct Owner { bool failed{}; size_t bytes{}, byte_budget{}; Context context; };
struct NativeOwner { Owner* scientific{}; };
constexpr size_t native_phased_becke_min_atoms = 48;
constexpr size_t stationary_spin_blocks = 1;
constexpr size_t stationary_becke_retained_max_atoms = 32, stationary_becke_max_atoms = 128;
constexpr size_t stationary_becke_pair_tile_rows = 4, stationary_becke_control_bytes = 16;
constexpr size_t stationary_becke_threads = 32;
namespace generativeqc_grid_adjoint { struct PointPair { double values[8]; }; }
int live_scientific{}, phase_calls{}, phase_status{};
void error_text(char* output, size_t capacity, const char* text) {
  if (output && capacity) std::snprintf(output, capacity, "%s", text);
}
int stationary_create(int, int, int, size_t atoms, size_t, size_t, size_t, size_t,
                      size_t, size_t, size_t budget, size_t, size_t,
                      void** output, char*, size_t) {
  const size_t center_bytes = 48 * atoms * (atoms - 1) / 2;
  auto owner = std::make_unique<Owner>();
  owner->bytes = 100 + (budget - 100 >= center_bytes ? center_bytes : 0);
  owner->byte_budget = budget;
  *output = owner.release();
  ++live_scientific;
  return 0;
}
template <class... Arguments> int stationary_configure_becke(Arguments&&...) { return 0; }
int stationary_configure_phased_becke_v1(void* pointer, size_t bytes, char*, size_t) {
  auto& owner = *static_cast<Owner*>(pointer);
  assert(bytes <= owner.byte_budget - owner.bytes);
  ++phase_calls;
  if (!phase_status) owner.bytes += bytes;
  return phase_status;
}
void stationary_destroy(void* pointer) {
  delete static_cast<Owner*>(pointer);
  --live_scientific;
}
"""
        + definitions
        + """
int main() {
  char error[256]{};
  for (size_t atoms : {24U, 48U, 96U, 128U})
    for (size_t lanes : {1U, 256U}) {
      const size_t centers = 48 * atoms * (atoms - 1) / 2;
      const size_t phase = native_phased_becke_bytes(atoms, 256, lanes);
      for (size_t budget : {size_t{100}, 100 + centers,
                            100 + centers + phase - (phase ? 1 : 0),
                            100 + centers + phase}) {
        void* output = nullptr;
        phase_calls = phase_status = 0;
        assert(native_create(0, 12, 0, atoms, 224, 672, 256, 1, 1, 1,
                             budget, lanes, 1, &output, error, sizeof(error)) == 0);
        assert(output && live_scientific == 1);
        assert(phase_calls == (phase && budget >= 100 + centers + phase ? 1 : 0));
        auto* owner = static_cast<NativeOwner*>(output);
        assert(owner->scientific->bytes <= budget);
        stationary_destroy(owner->scientific);
        delete owner;
        assert(live_scientific == 0);
      }
    }
  for (int status : {1, 2, 3, 4}) {
    phase_status = status;
    phase_calls = 0;
    void* output = nullptr;
    const size_t budget = 100 + 48 * (96 * 95 / 2) + phased_allocation(96, 256);
    assert(native_create(0, 12, 0, 96, 224, 672, 256, 1, 1, 1,
                         budget, 256, 1, &output, error, sizeof(error)) == status);
    assert(!output && phase_calls == 1 && live_scientific == 0);
  }
}
"""
    )
    executable = native_cxx.build_executable(
        [source], tmp_path / "phase_creation", compile_args=["-std=c++20", "-O0"]
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_phase_allocation_driver_fault_is_not_a_seed_retry(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Execute real allocation/error boundaries against inert CUDA operations.

    CUDA launch syntax becomes ordinary no-op calls. Allocation admission,
    native cleanup, method status mapping and the retry predicate remain real.
    OOM keeps the bounded path; a CUDA runtime failure must not request a new SCF.
    """
    runtime = (ROOT / "src/dft/native_stationary_runtime.cuh").read_text()
    executor = (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text()
    owner = (ROOT / "src/runtime/cuda_resources.cuh").read_text()
    stub = r"""

#pragma once
#include <cstddef>
using cudaError_t = int;
using cudaStream_t = void *;
constexpr int cudaSuccess = 0, cudaErrorMemoryAllocation = 2, cudaErrorInvalidDevice = 3;
inline int alloc_status = 0, property_status = 0, scope_status = 0, select_status = 0,
           allocations = 0, live_allocations = 0;
inline int cudaGetDevice(int *d) {
  *d = 0;
  return scope_status;
}
inline int cudaSetDevice(int) { return select_status; }
inline const char *cudaGetErrorString(int) { return "injected CUDA failure"; }
inline int cudaGetLastError() { return 0; }
inline int cudaStreamSynchronize(void *) { return 0; }
"""
    (tmp_path / "cuda_runtime_api.h").write_text(stub)
    prefix = r"""

#include "cuda_runtime_api.h"
#include "runtime/bounded_workspace.hpp"
#include "runtime/cuda_device_scope.hpp"
#include "tensor/cuda_error.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
using namespace generativeqc_tensor;
using std::size_t;
namespace generativeqc::runtime {
template <class T> int resource_cuda_malloc(T **p, size_t) {
  ++allocations;
  if (alloc_status)
    return alloc_status;
  *p = reinterpret_cast<T *>(4096);
  ++live_allocations;
  return 0;
}
inline int resource_cuda_free(void *) {
  --live_allocations;
  return 0;
}
"""
    for sig in ("struct BorrowedCudaBuffer", "class OwnedCudaBuffer"):
        prefix += "template<class T>\n" + _definition(owner, sig) + ";\n"
    prefix += "}\n"
    prefix += r"""

struct uint2 {
  unsigned a, b;
};
constexpr size_t native_phased_becke_min_atoms = 48, stationary_spin_blocks = 1,
                 stationary_source_count = 8;
constexpr size_t stationary_becke_retained_max_atoms = 32, stationary_becke_max_atoms = 128;
constexpr size_t stationary_becke_pair_tile_rows = 4, stationary_becke_control_bytes = 16,
                 stationary_becke_threads = 32;
constexpr bool stationary_becke_cooperative_normalize = false;
constexpr size_t stationary_becke_normalize_max_atoms = 128,
                 stationary_becke_normalize_point_lanes = 1;
constexpr size_t stationary_geometry_max_lanes = 1024,
                 stationary_geometry_max_scratch_bytes = 64U << 20;
namespace generativeqc_grid_adjoint {
struct PointPair {
  double values[8];
};
} // namespace generativeqc_grid_adjoint
struct Context {
  int device = 0;
  void *stream = nullptr;
  void check_device() {}
};
struct Owner {
  Context context;
  generativeqc::runtime::OwnedCudaBuffer<unsigned char> phased_storage;
  size_t atoms{}, points{}, geometry_lanes{}, bytes{}, byte_budget{},
      becke_threads_per_point = 1, becke_shared_bytes{}, phased_bytes{}, launches{},
      synchronizations{};
  bool failed{}, topology_ready{}, becke_normalize_supported{}, becke_normalize_cooperative{},
      becke_binary_scaled_requested{};
  void *center_pairs{};
};
struct NativeOwner {
  Owner *scientific{};
};
int live_owners = 0;
bool center_miss = false;
void error_text(char *out, size_t n, const char *s) {
  if (out && n)
    std::snprintf(out, n, "%s", s);
}
struct cudaDeviceProp {
  int major = 12, minor = 0, maxThreadsPerBlock = 1024, maxThreadsDim[3]{1024, 1024, 64},
      maxGridSize[3]{65535, 65535, 65535};
  size_t sharedMemPerBlock = 65536;
};
inline cudaDeviceProp props;
int cudaGetDeviceProperties(cudaDeviceProp *p, int) {
  *p = props;
  return property_status;
}
struct cudaFuncAttributes {
  int maxThreadsPerBlock = 1024;
  size_t sharedSizeBytes = 0;
};
template <class F> int cudaFuncGetAttributes(cudaFuncAttributes *, F) { return 0; }
size_t blocks(size_t x, size_t n) { return (x + n - 1) / n; }
struct Input {
  const uint2 *indices;
};
Input phased_input(Owner &o, size_t) { return {reinterpret_cast<uint2 *>(o.phased_storage.get())}; }
void phased_becke_indices(size_t, uint2 *) {}
template <bool> void phased_becke_normalize_cooperative() {}
namespace generativeqc_stationary_cuda {}
"""
    source = prefix + "\n".join(
        _definition(executor, s)
        for s in (
            "size_t phased_allocation(",
            "size_t allocation(",
            "template <class F>\nint guarded(",
        )
    )
    source += (
        "\n"
        + r"""

int stationary_create(int, int, int, size_t atoms, size_t aos, size_t primitives, size_t points,
                      size_t tasks, size_t spins, size_t, size_t budget, size_t lanes, size_t,
                      void **out, char *err, size_t n) {
  return guarded(nullptr, err, n, [&] {
    size_t bytes = allocation(atoms, aos, primitives, points, tasks, spins, lanes);
    if (budget < bytes)
      throw std::invalid_argument("short");
    size_t centers = 48 * atoms * (atoms - 1) / 2;
    auto p = std::make_unique<Owner>();
    p->bytes = bytes;
    if (!center_miss && centers <= budget - bytes) {
      p->bytes += centers;
      p->center_pairs = reinterpret_cast<void *>(1);
    }
    p->atoms = atoms;
    p->points = points;
    p->geometry_lanes = lanes;
    p->byte_budget = budget;
    *out = p.release();
    ++live_owners;
  });
}
void stationary_destroy(void *p) {
  delete static_cast<Owner *>(p);
  --live_owners;
}
"""
    )
    source += "\n" + _definition(executor, "int stationary_configure_becke(")
    source += "\n" + re.sub(
        r"<<<.*?>>>",
        "",
        _definition(executor, "int stationary_configure_phased_becke_v1("),
        flags=re.DOTALL,
    )
    source += "\n" + "\n".join(
        _definition(runtime, s)
        for s in (
            "size_t native_phased_becke_bytes(",
            "size_t native_numeric_bytes(",
            "void native_require_status(",
            "int native_create(",
        )
    )
    controller = (ROOT / "src/dft/cuda_stationary_controller.cpp").read_text()
    method = (ROOT / "src/methods/dft_method.cpp").read_text()
    failure = (ROOT / "src/methods/execution_failure.hpp").read_text()
    method_header = (ROOT / "src/methods/method.hpp").read_text()
    public = '\n#include <array>\n#include <exception>\n#include "generativeqc/generativeqc.hpp"\nusing generativeqc::Error;\n'
    public += _definition(method_header, "class MethodError final") + ";\n"
    public += _definition(failure, "inline bool retryable_seed_failure(") + "\n"
    public += _definition(controller, "void require_stationary_call(") + "\n"
    # Exercise the allocation boundary inside the complete-attempt callee; the
    # public wrapper only adds a bounded retry for optional automatic storage.
    f = _definition(method, "generativeqc_status native_cuda_stationary_gradient_once(")
    catch = f[f.rindex("} catch (const std::bad_alloc&)") : f.index("#else")]
    public += (
        "generativeqc_status map_public_status(int status) { std::string detail; auto stationary_controller_=std::make_unique<int>(); std::array<char,512> error{}; try { require_stationary_call(status,error); return GENERATIVEQC_STATUS_SUCCESS; "
        + catch
        + "}\n"
    )
    public += 'bool retry_status(int status) { const auto mapped=map_public_status(status); return retryable_seed_failure(mapped,std::make_exception_ptr(MethodError(mapped,"force failure"))); }\n'
    # Run the public retry wrapper too: an allocation failure may retire an
    # automatic tile, but numerical/driver failures must retain their status.
    public += r"""
#include <vector>
namespace dft {
struct CudaKsFinalStateToken {};
struct CudaStationaryResources { size_t bytes{}; };
}
namespace scf { struct DirectForcePrecisionRequest {}; }
struct ForceBoundary {
  int private_status;
  bool automatic, retired = false;
  generativeqc_status native_cuda_stationary_gradient_once(
      const dft::CudaKsFinalStateToken&, size_t, std::vector<double>& output,
      dft::CudaStationaryResources& resources, std::string&,
      scf::DirectForcePrecisionRequest*) {
    if (retired) {
      assert(output.empty() && resources.bytes == 0);
      output = {1.0};
      return GENERATIVEQC_STATUS_SUCCESS;
    }
    output = {-1.0};
    resources.bytes = 1024;
    return map_public_status(private_status);
  }
  bool retire_automatic_force_tile() {
    if (!automatic) return false;
    assert(!retired);
    retired = true;
    return true;
  }
"""
    public += _definition(
        method, "generativeqc_status native_cuda_stationary_gradient("
    )
    public += "\n};\n"
    source += public
    source += r"""

int main() {
  std::setbuf(stdout, nullptr);
  for (int private_status : {0, 1, 2, 3, 4}) {
    for (bool automatic : {false, true}) {
      ForceBoundary boundary{private_status, automatic};
      std::vector<double> output;
      dft::CudaStationaryResources resources;
      std::string detail;
      const auto status = boundary.native_cuda_stationary_gradient(
          {}, 4096, output, resources, detail);
      const bool optional_oom = automatic && (private_status == 2 || private_status == 3);
      assert(boundary.retired == optional_oom);
      assert(status == (optional_oom ? GENERATIVEQC_STATUS_SUCCESS : map_public_status(private_status)));
      if (optional_oom) assert(output == std::vector<double>{1.0});
    }
  }
  const size_t atoms = 96, points = 256;
  const size_t min = allocation(atoms, 224, 672, points, 1, 1, points, false);
  const size_t center = 48 * atoms * (atoms - 1) / 2, phase = phased_allocation(atoms, points);
  auto run = [&](size_t budget, int expected, size_t expected_phase) {
    char error[256]{};
    void *out = nullptr;
    const int before = allocations;
    int status = native_create(0, 12, 0, atoms, 224, 672, points, 1, 1, 1, budget, points, 32, &out,
                               error, sizeof(error));
    std::printf(
        "allocation_status=%d property_status=%d status=%d expected=%d phase=%zu allocations=%d\n",
        alloc_status, property_status, status, expected, expected_phase, allocations - before);
    if (status != expected) {
      if (out) {
        auto *n = static_cast<NativeOwner *>(out);
        stationary_destroy(n->scientific);
        delete n;
      }
      return false;
    }
    if (!status) {
      auto *n = static_cast<NativeOwner *>(out);
      assert(n && n->scientific->phased_bytes == expected_phase);
      assert(n->scientific->bytes <= budget);
      stationary_destroy(n->scientific);
      delete n;
    } else
      assert(!out);
    assert(!live_owners && !live_allocations);
    return true;
  };
  assert(run(min, 0, 0));
  assert(run(min + center + phase - 1, 0, 0));
  assert(run(min + center + phase, 0, phase));
  center_miss = true;
  assert(run(min + center + phase, 0, 0));
  center_miss = false;
  props.major = 8;
  assert(run(min + center + phase, 0, phase));
  props.major = 9;
  assert(run(min + center + phase, 0, phase));
  props.major = 12;
  props.maxThreadsPerBlock = 64;
  assert(run(min + center + phase, 0, 0));
  props.maxThreadsPerBlock = 1024;
  props.maxThreadsDim[0] = 127;
  assert(run(min + center + phase, 0, 0));
  props.maxThreadsDim[0] = 1024;
  props.maxGridSize[1] = int(atoms * (atoms - 1) / 2) - 1;
  assert(run(min + center + phase, 0, 0));
  props.maxGridSize[1] = 65535;
  props.maxGridSize[0] = 1;
  assert(run(min + center + phase, 0, 0));
  props.maxGridSize[0] = 65535;
  assert(run(min + center + phase, 0, phase));
  alloc_status = cudaErrorMemoryAllocation;
  assert(run(min + center + phase, 0, 0));
  alloc_status = 0;
  property_status = 71;
  assert(run(min + center + phase, 4, 0));
  property_status = 0;
  scope_status = 71;
  assert(run(min + center + phase, 4, 0));
  scope_status = 0;
  select_status = 71;
  assert(run(min + center + phase, 4, 0));
  select_status = 0;
  alloc_status = 71;
  const int expect = 4;
  const auto mapped = map_public_status(expect);
  std::printf("private=%d public=%d numerical=%d CUDA=%d seed_retry=%d\n", expect, int(mapped),
              int(GENERATIVEQC_STATUS_NUMERICAL_FAILURE), int(GENERATIVEQC_STATUS_CUDA_ERROR),
              int(retry_status(expect)));
  assert(map_public_status(1) == GENERATIVEQC_STATUS_NUMERICAL_FAILURE && retry_status(1));
  assert(map_public_status(4) == GENERATIVEQC_STATUS_CUDA_ERROR && !retry_status(4));
  return run(min + center + phase, expect, 0) ? 0 : 1;
}
"""
    (tmp_path / "phase_driver_status.cpp").write_text(source)
    executable = native_cxx.build_executable(
        [tmp_path / "phase_driver_status.cpp"],
        tmp_path / "phase_driver_status",
        compile_args=[
            "-std=c++20",
            "-O0",
            f"-I{tmp_path}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ],
    )
    subprocess.run([str(executable)], check=True, timeout=10)
