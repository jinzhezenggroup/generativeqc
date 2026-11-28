"""AOT inventory admission and generation without the public runtime."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.integral.hessian_aot import (
    IntegralRequest,
    integral_request,
)

from tools.generate_hessian_aot import parse_inventory

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_factorized_inventory_groups_host_metadata_without_changing_selectors() -> None:
    """Finite package selection is independent of CUDA source duplication."""
    from generativeqc_compiler.integral.hessian_aot import integral_program_identity

    from tools.complete_hessian_aot_inventory import inventory_payload
    from tools.generate_hessian_aot import compilation_units, emit_program

    expanded = parse_inventory(json.loads(INVENTORY.read_text()))
    factorized = replace(expanded, integral_schedule="factorized")
    assert parse_inventory(inventory_payload(factorized)) == factorized
    assert factorized.names == expanded.names
    units = compilation_units(factorized)
    assert "factorized_integrals.cu" in units
    assert "integral_metadata_0000.cpp" in units
    for request in factorized.integrals:
        source = emit_program(factorized, request.name)
        if request.family in ("eri", "coulomb_metric", "three_center_eri"):
            assert request.name + ".cu" not in units
            assert "factorized_integral_callback<" in source
            assert "factorized_integral_batch_callback<" in source
            is_parametric = request.radial != "full_range"
            assert (integral_program_identity(request) not in source) == is_parametric
            assert request.name in source
    with pytest.raises(ValueError, match="inventory"):
        parse_inventory({**inventory_payload(factorized), "integral_schedule": "jit"})


def test_integral_omega_admission_without_device(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Test the production dispatch with fake callbacks that record the radial parameter."""
    source = tmp_path / "radial.cpp"
    source.write_text(
        """
#include <cassert>
#include <limits>
#include "dft/hessian_aot.hpp"
namespace generativeqc::dft::hessian_aot {
namespace {
int runtime_eri(double omega, const double*, std::size_t, const long long*, const double*,
                double* output, int*, cudaStream_t) {
  *output = omega;
  return cudaSuccess;
}
int fixed_eri(const double* records, std::size_t count, const long long* atoms,
              const double* direction, double* output, int* status, cudaStream_t stream) {
  return runtime_eri(0.3, records, count, atoms, direction, output, status, stream);
}
}
std::span<const Program> programs() noexcept {
  static const Program inventory[] = {
      {"fixed", "fixed-id", Kind::integral, {}, 25, {}, 4, 17,
       nullptr, nullptr, fixed_eri, {"eri", {}, "long_range", 0.3}},
      {"runtime", "runtime-id", Kind::integral, {}, 25, {}, 4, 17,
       nullptr, nullptr, fixed_eri, {"eri", {}, "long_range", 0.3}, nullptr, runtime_eri},
  };
  return inventory;
}
}
int main() {
  namespace aot = generativeqc::dft::hessian_aot;
  double records[17]{}, direction[3]{}, output{};
  long long atoms[4]{};
  int status{};
  const auto at = [&](const char* leaf, double omega) {
    return aot::launch_integral_at_omega(*aot::find(leaf), omega, records, 1, atoms,
                                         direction, &output, &status, nullptr);
  };
  assert(at("fixed", 0.33) == cudaErrorNotSupported && output == 0);
  assert(at("runtime", -0.33) == cudaErrorNotSupported && output == 0);
  assert(at("runtime", std::numeric_limits<double>::infinity()) == cudaErrorNotSupported);
  assert(at("runtime", 0.33) == cudaSuccess && output == 0.33);
  assert(aot::launch_integral("runtime", records, 1, atoms, direction, &output,
                              &status, nullptr) == cudaSuccess && output == 0.3);
  assert(at("fixed", 0.3) == cudaSuccess && output == 0.3);
}
"""
    )
    probe = native_cxx.build_executable(
        (source, ROOT / "src/dft/hessian_aot.cpp"),
        tmp_path / "radial",
        compile_args=(
            "-std=c++20",
            f"-I{ROOT / 'tests/native/fixtures/hessian_budget'}",
            f"-I{ROOT / 'src'}",
        ),
    )
    subprocess.run([str(probe)], check=True, timeout=20)


def test_complete_factorized_f_g_inventory_has_bounded_compilation_units() -> None:
    """Do not regress to one NVCC invocation per high-angular component tile."""
    from generativeqc_compiler.integral.hessian_aot import FAMILIES

    from tools.complete_hessian_aot_inventory import complete_inventory
    from tools.generate_hessian_aot import (
        compilation_units,
        emit_compilation_unit,
        emit_registry,
    )

    seed = replace(
        parse_inventory(json.loads(INVENTORY.read_text())),
        integral_schedule="factorized",
    )
    complete = complete_inventory(seed, FAMILIES, 3, max_auxiliary_angular=4)
    assert len(complete.integrals) == 31469
    units = compilation_units(complete)
    assert sum(unit == "factorized_integrals.cu" for unit in units) == 1
    assert len(units) < 300
    assert sum(unit.startswith("integral_metadata_") for unit in units) < 128
    source = emit_compilation_unit(complete, "integral_metadata_0000.cpp")
    assert source.count("const Program& program_") == 256
    registry = emit_registry(complete)
    assert "entries[index]=getters[index]();" in registry
    assert "static const Inventory inventory;" in registry
    assert "static const std::array<Program," not in registry


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "manifests/hessian_aot_development.json"


def test_development_inventory_is_explicit_not_a_complete_endpoint_claim() -> None:
    inventory = parse_inventory(json.loads(INVENTORY.read_text()))
    assert inventory.ao
    assert {"xc_polarized", "xc_unpolarized", "vv10_pair"} <= set(inventory.names)
    assert {item.family for item in inventory.integrals} == {
        "overlap",
        "kinetic",
        "nuclear_attraction",
        "eri",
        "coulomb_metric",
        "three_center_eri",
    }
    assert {item.angular[0] for item in inventory.integrals} == {0, 1, 2, 3}
    assert "eri_full_range_3_3_3_3_c0" not in inventory.names
    assert inventory.names == tuple(sorted(set(inventory.names)))


@pytest.mark.parametrize(
    "change",
    [
        {"runtime_python": True},
        {"ao": 1},
        {"arrays": ["unknown"]},
        {"arrays": ["xc_polarized", "xc_polarized"]},
        {
            "integrals": [
                {
                    "family": "overlap",
                    "angular": [0, 0],
                    "components": [0],
                    "omega": 0.4,
                }
            ]
        },
        {"integrals": [{"family": "overlap", "angular": [True, 0], "components": [0]}]},
    ],
)
def test_inventory_rejects_ignored_or_ambiguous_controls(change: dict) -> None:
    payload = {
        "schema": "generativeqc.hessian-aot-inventory/v1",
        "ao": True,
        "arrays": [],
        "integrals": [],
    }
    with pytest.raises(ValueError):
        parse_inventory({**payload, **change})


@pytest.mark.parametrize(
    "specification",
    [
        ("eri", (0, 0), (0,), "full_range"),
        ("overlap", (4, 0), (0,), "full_range"),
        ("three_center_eri", (4, 0, 0), (0,), "full_range"),
        ("three_center_eri", (0, 4, 0), (0,), "full_range"),
        ("coulomb_metric", (5, 0), (0,), "full_range"),
        ("overlap", (0, 0), (0, 0), "full_range"),
        ("overlap", (1, 0), (1, 0), "full_range"),
        ("overlap", (0, 0), (1,), "full_range"),
        ("coulomb_metric", (0, 0), (0,), "short_range"),
    ],
)
def test_raw_component_requests_have_no_wildcard_fallback(specification: tuple) -> None:
    with pytest.raises(ValueError):
        IntegralRequest(*specification)


def test_attraction_retains_external_mathematical_center_and_unit_charge() -> None:
    request = integral_request(
        {"family": "nuclear_attraction", "angular": [1, 0], "components": [0, 2]}
    )
    assert request.centers == 3
    assert request.record_stride == 12
    assert request.payload()["charge"] == 1
    assert request.components == (0, 2)


def test_fitted_long_range_closure_round_trips_with_distinct_leaf_identity() -> None:
    from generativeqc_compiler.integral.hessian_aot import integral_program_identity

    from tools.complete_hessian_aot_inventory import (
        complete_inventory,
        inventory_payload,
    )

    families = ("coulomb_metric", "three_center_eri")
    seed = replace(parse_inventory(json.loads(INVENTORY.read_text())), integrals=())
    closure = complete_inventory(seed, families, 1, df_long_range=True)
    assert parse_inventory(inventory_payload(closure)) == closure
    for family in families:
        angular = (0,) * (2 if family == "coulomb_metric" else 3)
        full = IntegralRequest(family, angular, (0,))
        ranged = IntegralRequest(family, angular, (0,), "long_range")
        assert full in closure.integrals and ranged in closure.integrals
        assert full.payload()["omega"] == 0 and ranged.payload()["omega"] == 0.3
        assert integral_program_identity(full) != integral_program_identity(ranged)


@pytest.mark.parametrize(
    "family,angular", (("three_center_eri", (3, 3, 4)), ("coulomb_metric", (4, 4)))
)
def test_auxiliary_g_has_explicit_df_role_admission(
    family: str, angular: tuple
) -> None:
    request = IntegralRequest(family, angular, (0,))
    assert request.centers == len(angular)
    assert request.record_stride == 4 * len(angular) + 1
    assert (
        integral_request(
            {"family": family, "angular": list(angular), "components": [0]}
        )
        == request
    )


def test_build_generation_never_imports_runtime_or_gpu_packages(tmp_path: Path) -> None:
    output = tmp_path / "aot"
    script = """
import importlib.abc
import runpy
import sys
class ForbiddenRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'generativeqc', 'pyscf', 'cupy', 'torch'}:
            raise AssertionError('build-time generation imported ' + fullname)
sys.meta_path.insert(0, ForbiddenRuntime())
sys.argv = ['generate_hessian_aot.py', '--inventory', sys.argv[1], '--all', '--output-dir', sys.argv[2]]
runpy.run_path('tools/generate_hessian_aot.py', run_name='__main__')
"""
    subprocess.run(
        [sys.executable, "-c", script, str(INVENTORY), str(output)],
        check=True,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "python")},
        capture_output=True,
        text=True,
        timeout=120,
    )
    inventory = parse_inventory(json.loads(INVENTORY.read_text()))
    assert {item.stem for item in output.glob("*.cu")} == set(inventory.names)
    assert (output / "registry.cpp").is_file()
    assert all(item.stat().st_size for item in output.iterdir())


@pytest.mark.parametrize("pool", ["", "bounded_aot"])
def test_native_leaf_target_preserves_configured_compile_pool(
    tmp_path: Path, native_cxx: object, pool: str
) -> None:
    """Exercise real target wiring with inert CUDA sources, without NVCC."""
    import shutil

    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("AOT target configuration requires CMake")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(HessianTarget LANGUAGES CXX)\n"
        f'set(PROJECT_SOURCE_DIR "{ROOT}")\n'
        f'set(Python3_EXECUTABLE "{sys.executable}")\n'
        "add_library(CUDA::cudart INTERFACE IMPORTED)\n"
        "add_library(CUDA::cublas INTERFACE IMPORTED)\n"
        "function(generativeqc_register_generated_sources)\n"
        '  cmake_parse_arguments(G "" "GENERATOR" "OUTPUTS;DEPENDS;ARGS;COMMENT" ${ARGN})\n'
        "  set_source_files_properties(${G_OUTPUTS} PROPERTIES GENERATED TRUE LANGUAGE CXX)\n"
        "endfunction()\n"
        # The parent build registers these shared codegen inputs before
        # adding the Hessian archive. Leave their generators inert here.
        'generativeqc_register_generated_sources(OUTPUTS "${CMAKE_BINARY_DIR}/generated/'
        'generated_symmetric_matrix_function.cuh")\n'
        'generativeqc_register_generated_sources(OUTPUTS "${CMAKE_BINARY_DIR}/generated/'
        'generated_split_hybrid_registry.cuh")\n'
        "add_custom_target(generativeqc_d3_codegen)\n"
        "add_custom_target(generativeqc_dispersion_scalar_codegen)\n"
        "add_custom_target(generativeqc_method_parameters_codegen)\n"
        "add_custom_target(generativeqc_nonlocal_pair_codegen)\n"
        f'include("{ROOT}/cmake/GenerativeQCCuda.cmake")\n'
        f'include("{ROOT}/cmake/GenerativeQCHessianAot.cmake")\n'
        f'set(_generativeqc_aot_compile_pool "{pool}")\n'
        "set_property(GLOBAL PROPERTY JOB_POOLS bounded_aot=1)\n"
        f'generativeqc_add_hessian_aot(leaves "{INVENTORY}")\n'
        "get_target_property(pool leaves JOB_POOL_COMPILE)\n"
        'file(WRITE "${CMAKE_BINARY_DIR}/pool.txt" "${pool}")\n'
    )
    build = tmp_path / "build"
    result = subprocess.run(
        [
            cmake,
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            f"-DCMAKE_CXX_COMPILER={native_cxx.compiler}",
            f"-DCMAKE_CXX_COMPILER_LAUNCHER={native_cxx.cache}",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (build / "pool.txt").read_text() == (pool or "pool-NOTFOUND")


@pytest.mark.parametrize("requested", [False, True])
def test_requested_native_leaves_cannot_silently_use_cpu_fallback(
    tmp_path: Path, native_cxx: object, requested: bool
) -> None:
    """A missing CUDA compiler must reject an explicit AOT leaf request."""
    import shutil

    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("AOT package admission requires CMake")
    result = subprocess.run(
        [
            cmake,
            "-S",
            str(ROOT),
            "-B",
            str(tmp_path / "build"),
            f"-DCMAKE_CXX_COMPILER={native_cxx.compiler}",
            f"-DCMAKE_CXX_COMPILER_LAUNCHER={native_cxx.cache}",
            "-DCMAKE_CUDA_COMPILER=NOTFOUND",
            "-DGENERATIVEQC_ENABLE_CUDA=ON",
            f"-DGENERATIVEQC_ENABLE_HESSIAN_AOT_LEAVES={'ON' if requested else 'OFF'}",
            "-DGENERATIVEQC_BUILD_CLI=OFF",
            "-DGENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT=OFF",
            "-DGENERATIVEQC_CPU_LINALG_PROVIDER=scalar",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (result.returncode != 0) is requested, result.stdout + result.stderr
    if requested:
        assert "Hessian AOT leaves require CUDA" in result.stderr
