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
    source = tmp_path / "graph_admission.cpp"
    source.write_text(
        '#include "scf/types.hpp"\n'
        '#include "dft/grid.hpp"\n'
        '#include "dft/cuda_stationary_capacity.hpp"\n'
        + emit_native_stationary_registry_cpp(("pbe_rks", "pbe_uks"))
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
}
namespace generativeqc::methods::detail {
struct NativeKsExecutionPlan {
  unsigned spin_channels{1}, functional{1}, domain{4};
  bool range_exchange{}, nonlocal_correlation{}, d4_correction{};
};
unsigned ks_xc_functional_code(const NativeKsExecutionPlan& plan) { return plan.functional; }
unsigned ks_scf_domain_version(const NativeKsExecutionPlan& plan) { return plan.domain; }
"""
        + admission
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
    provider.spec.derivative_order = 0;
    provider.spec.spin = spin == 1 ? scf::FockSpin::Restricted : scf::FockSpin::Unrestricted;
    provider.spec.exchange.coefficient = spin == 1 ? -0.125 : -0.25;
    dft::GridSpec grid;
    const auto select = [&] { return native_cuda_stationary_binding(plan, options, grid, true, 1); };
    const auto* selected = select();
    const auto* expected = spin == 1 ? &dft::native_stationary_profile_pbe_rks()
                                     : &dft::native_stationary_profile_pbe_uks();
    if (selected != expected) return 1;
    const auto capacity = dft::cuda_stationary_capacity(*selected, 2, 2, 2, 4, 2);
    if (capacity.stationary_minimum_bytes != 1000 + 10 * spin ||
        capacity.stationary_cached_bytes != 2000 + 10 * spin) return 2;
    if (native_cuda_stationary_binding(plan, options, grid, false, 1) ||
        native_cuda_stationary_binding(plan, options, grid, true, 2)) return 3;
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
  }
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
