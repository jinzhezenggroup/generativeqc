"""MethodIR-driven native stationary graph packaging: no second method list."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from _cpp_source_support import cpp_function_definition

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "tools/generate_native_stationary_cuda.py"


def generate(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GENERATOR), *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def qualified(*arguments: str) -> tuple[str, ...]:
    result = generate(*arguments)
    assert result.returncode == 0, result.stderr
    return tuple(result.stdout.strip().split(";"))


def test_inventory_is_generated_from_compiler_method_ir() -> None:
    native = qualified("--list-profiles")
    aot = qualified("--list-aot-profiles")
    assert native and len(native) == len(set(native))
    assert set(native) <= set(aot)
    assert native == qualified("--list-profiles")


def test_single_graph_generation_matches_independent_codegen(tmp_path: Path) -> None:
    names = qualified("--list-profiles")
    assert len(names) >= 2
    selected = (names[0], names[-1])
    output = tmp_path / "group"
    actual = generate(
        "--all-profiles", "--output-dir", str(output), "--profiles", *selected
    )
    assert actual.returncode == 0, actual.stderr
    snapshots: dict[str, tuple[bytes, int]] = {}
    for name in selected:
        candidate = output / f"native_{name}.cu"
        assert candidate.is_file()
        old = tmp_path / f"single_{name}.cu"
        separate = generate("--output", str(old), "--profile", name)
        assert separate.returncode == 0, separate.stderr
        assert candidate.read_bytes() == old.read_bytes()
        snapshots[name] = (candidate.read_bytes(), candidate.stat().st_mtime_ns)

    second = generate(
        "--all-profiles", "--output-dir", str(output), "--profiles", *selected
    )
    assert second.returncode == 0, second.stderr
    for name in selected:
        candidate = output / f"native_{name}.cu"
        assert (candidate.read_bytes(), candidate.stat().st_mtime_ns) == snapshots[name]


@pytest.mark.parametrize("profiles", [("invalid_profile",), ("pbe_rks", "pbe_rks")])
def test_invalid_or_duplicate_compiler_graph_selection_is_rejected(
    tmp_path: Path, profiles: tuple[str, ...]
) -> None:
    rejected = generate(
        "--all-profiles", "--output-dir", str(tmp_path), "--profiles", *profiles
    )
    assert rejected.returncode != 0
    assert not tuple(tmp_path.glob("*.cu"))


def test_prepared_admission_retains_graph_and_capacity(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Execute the real admission guard with colliding diagnostic labels."""
    from generativeqc_compiler.method.stationary_native_cuda import (
        emit_native_stationary_registry_cpp,
    )

    production = (ROOT / "src/methods/dft_method.cpp").read_text()
    admission = cpp_function_definition(production, "native_cuda_stationary_binding")
    descriptor_capacity = cpp_function_definition(
        production, "dft_cuda_stationary_capacity"
    )
    source = tmp_path / "graph_admission.cpp"
    source.write_text(
        '#include "scf/types.hpp"\n'
        '#include "scf/fock_precision.hpp"\n'
        '#include "dft/grid.hpp"\n'
        '#include "dft/application_precision.hpp"\n'
        '#include "dft/semilocal_family.hpp"\n'
        '#include "dft/cuda_stationary_capacity.hpp"\n'
        + emit_native_stationary_registry_cpp(("pbe_rks", "pbe_uks", "m06_2x_rks"))
        + r"""
namespace generativeqc::dft {
std::size_t numeric(std::size_t, std::size_t, std::size_t, std::size_t,
                    std::size_t, std::size_t spin, std::size_t, bool cached) {
  return 1000 + 10 * spin + (cached ? 1000 : 0);
}
CudaStationaryBinding make_binding(unsigned spin) {
  CudaStationaryBinding value;
  value.profile = "same diagnostic label";
  value.spin_blocks = spin;
  value.geometry_max_lanes = value.geometry_max_threads = 32;
  value.geometry_max_scratch_bytes = 4096;
  value.ao_jets = 4;
  value.numeric_bytes = numeric;
  value.geometry_fp32 = +[](void*, const GridTaskView*, const double*, std::size_t,
                            std::size_t, const double*, const double*, std::uint64_t,
                            std::uint64_t, char*, std::size_t) { return 0; };
  value.model = {1, 4, spin, 3, 0.75, 1.0, 0.25};
  return value;
}
const CudaStationaryBinding& native_stationary_profile_pbe_rks() {
  static const auto binding = make_binding(1);
  return binding;
}
const CudaStationaryBinding& native_stationary_profile_pbe_uks() {
  static const auto binding = make_binding(2);
  return binding;
}
const CudaStationaryBinding& native_stationary_profile_m06_2x_rks() {
  static const auto binding = [] {
    auto value = make_binding(1);
    value.model = {131522, 4, 1, 3, 1.0, 1.0, 0.54};
    return value;
  }();
  return binding;
}
}
namespace generativeqc::methods::detail {
struct NativeKsExecutionPlan {
  unsigned spin_channels{1}, functional{1}, domain{4};
  bool range_exchange{}, nonlocal_correlation{}, d4_correction{}, generated_split_hybrid{};
};
unsigned ks_xc_functional_code(const NativeKsExecutionPlan& plan) { return plan.functional; }
unsigned ks_scf_domain_version(const NativeKsExecutionPlan& plan) { return plan.domain; }
"""
        + admission
        + r"""
// Descriptor parsing is outside this host-only test. Both front doors consume
// the same resolved scientific options, but execute their production gates.
scf::ScfOptions capacity_options;
NativeKsExecutionPlan capacity_plan;
dft::GridSpec capacity_grid;
scf::ScfOptions dft_options(const generativeqc_method_descriptor&,
                            generativeqc_backend, NativeKsExecutionPlan& plan) {
  plan = capacity_plan;
  return capacity_options;
}
dft::GridSpec ks_grid_options(const generativeqc_method_descriptor&,
                              const scf::ScfOptions&, const NativeKsExecutionPlan&) {
  return capacity_grid;
}
struct MethodError : std::runtime_error {
  using std::runtime_error::runtime_error;
  generativeqc_status status() const { return GENERATIVEQC_STATUS_INVALID_ARGUMENT; }
};
"""
        + descriptor_capacity
        + r"""
}
int main() {
  using namespace generativeqc;
  using namespace generativeqc::methods::detail;
  for (const unsigned spin : {1U, 2U}) {
    NativeKsExecutionPlan plan;
    plan.spin_channels = spin;
    scf::ScfOptions options;
    options.precision_mode = GENERATIVEQC_PRECISION_FP64;
    options.semilocal_exchange_scale = 0.75;
    options.resolved_fock_build.emplace();
    auto& provider = *options.resolved_fock_build;
    provider.backend = scf::FockBackend::Cuda;
    provider.schedule = scf::FockSchedule::CudaIndependent;
    provider.spec.derivative_order = 0;
    provider.spec.spin = spin == 1 ? scf::FockSpin::Restricted : scf::FockSpin::Unrestricted;
    provider.spec.exchange.coefficient = spin == 1 ? -0.125 : -0.25;
    dft::GridSpec grid;
    const auto select = [&] { return native_cuda_stationary_binding(plan, options, grid, true, 1, 96, 12); };
    const auto* selected = select();
    const auto* expected = spin == 1 ? &dft::native_stationary_profile_pbe_rks()
                                     : &dft::native_stationary_profile_pbe_uks();
    if (selected != expected) return 1;
    const auto capacity = dft::cuda_stationary_capacity(*selected, 2, 2, 2, 4, 2);
    if (capacity.stationary_minimum_bytes != 1000 + 10 * spin ||
        capacity.stationary_cached_bytes != 2000 + 10 * spin) return 2;
    if (native_cuda_stationary_binding(plan, options, grid, false, 1, 96, 12) ||
        bool(native_cuda_stationary_binding(plan, options, grid, true, 2, 96, 12)) != (spin == 1)) return 3;
    if (native_cuda_stationary_binding(plan, options, grid, true, 2, 768, 96)) return 14;
    if (native_cuda_stationary_binding(plan, options, grid, true, 2, 384, 48)) return 15;
    const auto mixed = dft::cuda_stationary_capacity(*selected, 2, 2, 2, 4, 2, true);
    const auto extra = 4 * (8 * 2 * 2 + 2 * 2) + dft::CudaGridStorage::provider_allowance;
    if (mixed.minimum_device_bytes != capacity.minimum_device_bytes ||
        mixed.stationary_cached_bytes != capacity.stationary_cached_bytes ||
        mixed.cached_grid_extra_bytes != extra ||
        mixed.cached_device_bytes != capacity.cached_device_bytes + extra) return 10;
    const auto receipt = dft::cuda_stationary_capacity_values(mixed);
    if(receipt[4] != receipt[0]+receipt[1]+receipt[2] ||
       receipt[5] != receipt[0]+receipt[1]+receipt[3]) return 11;
    ++plan.domain;
    if (select()) return 4;
    --plan.domain;
    provider.spec.exchange.coefficient *= 2;
    if (select()) return 5;
    provider.spec.exchange.coefficient /= 2;
    options.semilocal_exchange_scale = 0.5;
    if (select()) return 6;
    options.semilocal_exchange_scale = 0.75;
    grid.partition_iterations = 2;
    if (select()) return 7;
    grid.partition_iterations = 3;
    provider.backend = scf::FockBackend::Cpu;
    if (select()) return 8;
    provider.backend = scf::FockBackend::Cuda;
    if (select() != expected) return 9;
    options.precision_mode = GENERATIVEQC_PRECISION_AUTO;
    options.application_precision_budget = {1e-5, 1e-4, 3e-5};
    if (bool(select()) != (spin == 1)) return 12;
    if (bool(native_cuda_stationary_binding(plan, options, grid, true, 2, 384, 48)) != (spin == 1)) return 16;
    options.energy_tolerance = 1e-6;
    options.density_tolerance = 1e-4;
    options.screening_tolerance = 1e-10;
    if (bool(select()) != (spin == 1)) return 23;
    provider.schedule = scf::FockSchedule::CpuReference;
    if (select()) return 24;
    provider.schedule = scf::FockSchedule::CudaIndependent;
    options.energy_tolerance = 1.01e-6;
    if (select()) return 25;
    options.energy_tolerance = 1e-10;
    options.density_tolerance = 1e-8;
    options.screening_tolerance = 1e-12;
    options.precision_mode = GENERATIVEQC_PRECISION_FP64;
    if (select() != expected) return 17;
    options.precision_mode = GENERATIVEQC_PRECISION_AUTO;
    options.application_precision_budget[1] = 1e-7;
    if (select()) return 13;
  }
  NativeKsExecutionPlan split;
  split.generated_split_hybrid = true;
  split.functional = 131522;
  scf::ScfOptions options;
  options.precision_mode = GENERATIVEQC_PRECISION_FP64;
  options.resolved_fock_build.emplace();
  auto& provider = *options.resolved_fock_build;
  provider.backend = scf::FockBackend::Cuda;
  provider.spec.derivative_order = 0;
  provider.spec.spin = scf::FockSpin::Restricted;
  provider.spec.exchange.coefficient = -0.27;
  dft::GridSpec grid;
  const auto select = [&](std::size_t aos, std::size_t atoms, std::size_t angular) {
    return native_cuda_stationary_binding(split, options, grid, true, angular, aos, atoms);
  };
  const auto* expected = &dft::native_stationary_profile_m06_2x_rks();
  if (select(24, 3, 2) != expected || select(96, 12, 2) != expected) return 18;
  if (select(23, 3, 1) || select(97, 12, 1) || select(24, 1, 1) ||
      select(96, 33, 1) || select(96, 12, 3)) return 19;
  split.spin_channels = 2;
  provider.spec.spin = scf::FockSpin::Unrestricted;
  provider.spec.exchange.coefficient = -0.54;
  if (select(24, 3, 1)) return 20;
  split.spin_channels = 1;
  provider.spec.spin = scf::FockSpin::Restricted;
  provider.spec.exchange.coefficient = -0.27;
  options.precision_mode = GENERATIVEQC_PRECISION_AUTO;
  if (select(24, 3, 2)) return 21;
  options.application_precision_budget = {1e-5, 1e-4, 3e-5};
  if (select(24, 3, 2) != expected) return 22;
  // Expanded AUTO candidates are allowed without changing the strict-FP64
  // comparator assertions above. They must retain audit-required provenance.
  for (const auto aos : {97U, 384U, 385U, 2048U}) {
    if (select(aos, 12, 1) != expected || select(aos, 12, 2) != expected) return 26;
    if (dft::application_gradient_qualification(
            *expected, options.application_precision_budget, aos, 12, 2) != 106) return 27;
  }
  for (const auto atoms : {33U, 49U, 96U})
    if (select(96, atoms, 2) != expected) return 28;

  capacity_options.precision_mode = GENERATIVEQC_PRECISION_AUTO;
  capacity_options.semilocal_exchange_scale = 0.75;
  capacity_options.application_precision_budget = {1e-5, 1e-4, 3e-5};
  capacity_options.xc_tile_points = 32;
  capacity_options.resolved_fock_build.emplace();
  auto& capacity_provider = *capacity_options.resolved_fock_build;
  capacity_provider.backend = scf::FockBackend::Cuda;
  capacity_provider.schedule = scf::FockSchedule::CudaIndependent;
  capacity_provider.spec.derivative_order = 0;
  capacity_provider.spec.spin = scf::FockSpin::Restricted;
  capacity_provider.spec.exchange.coefficient = -0.125;
  capacity_grid.radial_points = 4;
  capacity_grid.angular_polar = 2;
  capacity_grid.angular_azimuth = 4;
  const std::array<std::uint64_t, 7> shape{12, 96, 96, 384, 2, 0, 0};
  const auto& capacity_binding = dft::native_stationary_profile_pbe_rks();
  const auto prepared_mixed = dft::cuda_stationary_capacity(
      capacity_binding, shape[0], shape[1], shape[2], shape[3], 32, true);
  const auto prepared_strict = dft::cuda_stationary_capacity(
      capacity_binding, shape[0], shape[1], shape[2], shape[3], 32, false);
  if (!prepared_mixed.cached_grid_extra_bytes || prepared_strict.cached_grid_extra_bytes) return 29;
  generativeqc_method_descriptor descriptor{};
  for (const auto relaxed : {false, true}) {
    capacity_options.energy_tolerance = relaxed ? 1e-6 : 1e-10;
    capacity_options.density_tolerance = relaxed ? 1e-4 : 1e-8;
    capacity_options.screening_tolerance = relaxed ? 1e-10 : 1e-12;
    dft::CudaStationaryCapacity capacity;
    std::string detail;
    if (dft_cuda_stationary_capacity(descriptor, shape, capacity, detail) !=
        GENERATIVEQC_STATUS_SUCCESS) return 30;
    if (dft::cuda_stationary_capacity_values(capacity) !=
        dft::cuda_stationary_capacity_values(prepared_mixed)) return 31;
  }
  capacity_options.precision_mode = GENERATIVEQC_PRECISION_FP64;
  capacity_options.application_precision_budget = {};
  capacity_options.energy_tolerance = 1e-10;
  capacity_options.density_tolerance = 1e-8;
  capacity_options.screening_tolerance = 1e-12;
  dft::CudaStationaryCapacity strict_capacity;
  std::string detail;
  if (dft_cuda_stationary_capacity(descriptor, shape, strict_capacity, detail) !=
      GENERATIVEQC_STATUS_SUCCESS) return 32;
  if (dft::cuda_stationary_capacity_values(strict_capacity) !=
      dft::cuda_stationary_capacity_values(prepared_strict)) return 33;
}
"""
    )
    executable = native_cxx.build_executable(
        [source],
        tmp_path / "graph_admission",
        compile_args=("-std=c++20", f"-I{ROOT / 'src'}", f"-I{ROOT / 'include'}"),
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_native_controller_qualification_fixture_compiles(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Compile the real controller qualification body without executing CUDA."""
    fixture = (
        ROOT / "tests/native/test_native_stationary_cuda_binding.cpp"
    ).read_text()
    preamble = fixture[
        fixture.index("#include <algorithm>") : fixture.index("/** These transport/")
    ]
    controller = fixture[
        fixture.index("void controller_admission(") : fixture.index("\nint main(")
    ]
    source = tmp_path / "controller_fixture.cpp"
    source.write_text(preamble + controller)
    native_cxx.compile_object(
        source,
        tmp_path / "controller_fixture.o",
        args=("-std=c++20", f"-I{ROOT / 'src'}", f"-I{ROOT / 'include'}"),
    )
