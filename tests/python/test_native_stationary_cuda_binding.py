"""Native ownership adaptation must retain the existing stationary mathematics."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.method.stationary_cuda import (
    _qualified_aot_profile,
    emit_stationary_profile_aot_wrapper_cuda,
    qualified_sp_requests,
)
from generativeqc_compiler.method.stationary_native_cuda import (
    NATIVE_STATIONARY_CUDA_PROFILES,
    emit_native_stationary_profile_cuda,
    emit_native_stationary_registry_cpp,
)

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("profile", NATIVE_STATIONARY_CUDA_PROFILES)
def test_native_binding_retains_scientific_wrapper(profile: str) -> None:
    """Token ownership changes cannot rewrite equations, literals or IR hashes."""
    wrapper = emit_stationary_profile_aot_wrapper_cuda(profile)
    native = emit_native_stationary_profile_cuda(profile)
    assert wrapper in native
    aliases = dict(re.findall(r"^#define (\w+) (\w+)$", native, re.MULTILINE))
    assert "first_derivative" not in aliases
    for symbol in (
        "generativeqc_stationary_cuda",
        "generativeqc_grid_adjoint",
        "generativeqc_grid_phased",
        "generativeqc_grid_coefficients",
        "generativeqc_grid_normalized",
        "StationaryPointValue",
        "ao_pullback",
        "local_ratio_geometry",
        "local_ratio_prepared",
        "local_norm",
        "local_ratio",
        "local_log",
        "local_becke",
        "stationary_create",
        "stationary_finish_reduced",
        "stationary_destroy",
    ):
        assert aliases[symbol] == f"generativeqc_native_{profile}_{symbol}"
    assert native == emit_native_stationary_profile_cuda(profile)


@pytest.mark.parametrize("profile", NATIVE_STATIONARY_CUDA_PROFILES)
def test_native_transport_metadata_comes_from_plan(profile: str) -> None:
    plan = _qualified_aot_profile(profile).plan
    source = emit_native_stationary_profile_cuda(profile)
    assert plan.identity in source
    assert plan.method.identity in source
    integral_sources = ("one_electron", "coulomb", "overlap_pulay", "exact_exchange")
    mask = sum(
        1 << plan.source_names.index(name)
        for name in integral_sources
        if name in plan.source_names
    )
    assert f"native_required_integral_mask = {mask};" in source
    assert (
        f"native_nuclear_kind = {qualified_sp_requests().index(('nuclear', ()))};"
        in source
    )
    assert f"sources, {len(plan.source_names)}, {plan.spin_blocks}," in source
    exchange_slot = plan.source_names.index("exact_exchange") if plan.exchange else -1
    assert f"native_exchange_slot = {exchange_slot};" in source


def test_native_registry_contains_only_built_profiles() -> None:
    selected = ("pbe0_uks", "pbe_rks")
    registry = emit_native_stationary_registry_cpp(selected)
    assert registry == emit_native_stationary_registry_cpp(tuple(reversed(selected)))
    assert 'profile == "pbe0_uks"' in registry
    assert 'profile == "pbe_rks"' in registry
    assert 'profile == "pbe_uks"' not in registry
    assert "return nullptr;" in emit_native_stationary_registry_cpp(())
    for invalid in (("pbe_rks", "pbe_rks"), ("lda_rks",)):
        with pytest.raises(ValueError):
            emit_native_stationary_registry_cpp(invalid)
    with pytest.raises(ValueError):
        emit_native_stationary_profile_cuda("b3lyp_rks")


def test_complete_controller_uses_resident_providers_and_emitted_reduction() -> None:
    """Structural guard complements, not substitutes for, the real-device oracles."""
    controller = (ROOT / "src/dft/cuda_stationary_controller.cpp").read_text()
    transport = (ROOT / "src/dft/native_stationary_runtime.cuh").read_text()
    for entry in (
        "grid_cuda_density_device_v1(",
        "grid_cuda_run_selected_device_deferred_v1(",
        "grid_cuda_density_jets_v2(",
        "consume_prepared_cuda_stationary_one_electron_pair(",
        "consume_prepared_cuda_direct_shell_full_range_derivatives_device(",
        "binding_->finish_gradient(",
    ):
        assert entry in controller
    assert controller.count("token_current()") == 2
    assert "output.swap(candidate)" in controller
    assert "read_final_state(" not in controller
    assert "stationary_finish_reduced(" in transport
    assert "owner->grid_points != expected_points" in transport
    assert "owner->integral_mask != native_required_integral_mask" in transport
    assert "owner_offset != owner->grid_points" in transport
    assert "duplicate native stationary source" in transport


def test_native_operations_preserve_inner_typed_status(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Execute every nested error boundary with only CUDA side effects doubled."""
    transport = (ROOT / "src/dft/native_stationary_runtime.cuh").read_text()
    executor = (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text()
    guarded = executor[
        executor.index("template <class F>\nint guarded(") : executor.index(
            "bool valid_geometry_ao_map("
        )
    ]
    status_policy = transport[
        transport.index("void native_require_status(") : transport.index(
            "int native_create("
        )
    ]
    operations = (
        ("native_create", "native_topology"),
        ("native_reset_geometry", "native_consume_integrals"),
        ("native_nuclear", "native_geometry"),
        ("native_geometry", "native_finish_gradient"),
        ("native_finish_gradient", "native_source_span"),
        ("native_source_span", "native_metrics"),
    )
    wrappers = "\n".join(
        transport[transport.index(f"int {begin}(") : transport.index(f"int {end}(")]
        for begin, end in operations
    )
    source = tmp_path / "typed_creation.cpp"
    source.write_text(
        """
#include <cstdio>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
struct DeviceAllocationError : std::runtime_error { using std::runtime_error::runtime_error; };
struct DeviceRuntimeError : std::runtime_error { using std::runtime_error::runtime_error; };
struct Context { void check_device() {} };
struct Owner { bool failed{}; Context context; uint64_t synchronizations{}; };
int live_owners{}, inner_status{};
struct NativeOwner {
  Owner* scientific{};
  bool integral_pending{}, nuclear_ready{};
  void* integral_stream{};
  uint64_t integral_mask{}, grid_points{};
  NativeOwner() { ++live_owners; }
  ~NativeOwner() { --live_owners; }
};
constexpr size_t stationary_spin_blocks = 1;
constexpr uint64_t native_required_integral_mask = 1, native_nuclear_kind = 0;
namespace generativeqc::dft { struct GridTaskView { size_t npoint{}; }; }
void native_drain_integrals(NativeOwner&) {}
void cuda_check(int) {}
int cudaStreamSynchronize(void*) { return 0; }
void error_text(char* output, size_t capacity, const char* text) {
  if (output && capacity) std::snprintf(output, capacity, "%s", text);
}
template <class... Arguments> int stationary_create(Arguments&&...) { return inner_status; }
template <class... Arguments> int stationary_geometry_reset(Arguments&&...) { return inner_status; }
template <class... Arguments> int stationary_nuclear_all(Arguments&&...) { return inner_status; }
template <class... Arguments>
int stationary_geometry_molecular_resident_weights_enqueue_v2(Arguments&&...) { return inner_status; }
template <class... Arguments> int stationary_finish_reduced(Arguments&&...) { return inner_status; }
template <class... Arguments> int stationary_finish_span(Arguments&&...) { return inner_status; }
"""
        + guarded
        + status_policy
        + wrappers
        + """
int main() {
  for (const int expected : {1, 2, 3, 4}) {
    inner_status = expected;
    void* output = reinterpret_cast<void*>(1);
    char error[128] = "injected creation failure";
    const int actual = native_create(0, 12, 0, 2, 2, 2, 64, 1, 1, 1,
                                     1 << 20, 64, 16, &output, error, sizeof(error));
    if (actual != expected || output || live_owners) return expected;
    Owner scientific;
    NativeOwner owner;
    owner.scientific = &scientific;
    if (native_reset_geometry(&owner, nullptr, 1e-12, error, sizeof(error)) != expected ||
        !scientific.failed) return 5;
    scientific.failed = false;
    if (native_nuclear(&owner, nullptr, 2, error, sizeof(error)) != expected ||
        !scientific.failed || owner.nuclear_ready) return 6;
    scientific.failed = false;
    generativeqc::dft::GridTaskView view{1};
    if (native_geometry(&owner, &view, nullptr, 0, 1, nullptr, nullptr, 0, 0,
                        error, sizeof(error)) != expected ||
        !scientific.failed || owner.grid_points) return 7;
    owner.grid_points = 1;
    owner.integral_mask = native_required_integral_mask;
    owner.nuclear_ready = true;
    scientific.failed = false;
    if (native_finish_gradient(&owner, 1, nullptr, 0, error, sizeof(error)) != expected ||
        !scientific.failed) return 8;
    scientific.failed = false;
    if (native_source_span(&owner, 0, 1, nullptr, 0, error, sizeof(error)) != expected ||
        !scientific.failed) return 9;
  }
}
"""
    )
    binary = tmp_path / "typed_creation"
    native_cxx.build_executable([source], binary, compile_args=["-std=c++17", "-O2"])
    result = subprocess.run(
        [str(binary)], check=False, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stdout + result.stderr
