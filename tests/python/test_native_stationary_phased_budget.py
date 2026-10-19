"""Host admission contracts and allocation-gated forces across compiled profiles."""

from __future__ import annotations

import os
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
  bool rejected = false;
  try { (void)native_numeric_bytes(129, 224, 672, 256, 1,
                                  stationary_spin_blocks, 256, true); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
}
"""
    )
    executable = native_cxx.build_executable(
        [source], tmp_path / "phase_capacity", compile_args=["-std=c++20", "-O0"]
    )
    subprocess.run([str(executable)], check=True, timeout=10)


@pytest.mark.parametrize(
    ("method", "native_admitted"),
    [
        pytest.param(
            profile.replace("_", "-"),
            not profile.startswith("m06_2x"),
            id=profile,
        )
        for profile in NATIVE_STATIONARY_CUDA_PROFILES
    ],
)
@pytest.mark.parametrize("atom_count", (48, 96, 128))
def test_phase_force_matches_energy_derivatives_and_existing_fallback(
    method: str, native_admitted: bool, atom_count: int
) -> None:
    """Check moved forces without promoting compiled-only M06-2X native graphs.

    Large split-hybrid contexts retain their existing packaged Python owner;
    its already-generic phased geometry is checked rather than bypassing native
    admission. Admitted native profiles additionally exercise minimum capacity.
    """
    if os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly Slurm-allocated GPU")
    assert os.environ.get("SLURM_JOB_ID")
    import ctypes

    import numpy as np
    from generativeqc import Calculator, GridSpec, KsOptions, _native

    atoms = []
    for pair in range(atom_count // 2):
        center = (8.0 * (pair % 4), 8.0 * ((pair // 4) % 4), 8.0 * (pair // 16))
        for offset in (-0.7, 0.7):
            atoms.append(("H", (center[0], center[1], center[2] + offset)))
    multiplicity = 2 if method.endswith("-uks") else 1
    if multiplicity == 2:
        # A single HeH radical breaks equal-spin symmetry without making all
        # separated H2 fragments degenerate open-shell triplets.
        atoms[-1] = ("He", atoms[-1][1])
    calculator = Calculator(
        method=method,
        basis="sto-3g",
        device="cuda",
        device_id=int(os.environ.get("GENERATIVEQC_TEST_DEVICE_ID", "0")),
        ks_options=KsOptions(grid=GridSpec(12, 4, 8)),
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
            limit = batch._library.generativeqc_ks_batch_stationary_cuda_limit_v1
            limit.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64]
            limit.restype = ctypes.c_int
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
