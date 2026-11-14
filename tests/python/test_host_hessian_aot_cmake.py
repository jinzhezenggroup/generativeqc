"""Build and execute the optional host scalar package without enabling CUDA."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from generativeqc_compiler.method.host_hessian_aot import parse_inventory, programs
from generativeqc_compiler.method.spec import resolve_method
from generativeqc_compiler.xc.atomic_grid_native import emit_atomic_grid_header

ROOT = Path(__file__).resolve().parents[2]


def test_cpu_only_cmake_target_and_inventory_reconfiguration(
    tmp_path: Path, required_native_cxx: Any
) -> None:
    """A linked consumer sees only the inventory built into its static library."""
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required to qualify the build-time host leaf package")
    payload = json.loads((ROOT / "manifests/host_hessian_aot_b3lyp.json").read_text())
    inventory = tmp_path / "selected inventory.json"
    inventory.write_text(json.dumps(payload))
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(host_leaf_consumer LANGUAGES CXX)\n"
        f'set(PROJECT_SOURCE_DIR "{ROOT.as_posix()}")\n'
        f'set(Python3_EXECUTABLE "{sys.executable}")\n'
        f'include("{ROOT.as_posix()}/cmake/GenerativeQCGenerated.cmake")\n'
        f'include("{ROOT.as_posix()}/cmake/GenerativeQCHostHessianAot.cmake")\n'
        f'generativeqc_add_host_hessian_aot(host_leaves "{inventory.as_posix()}")\n'
        "add_executable(consumer consumer.cpp)\n"
        "target_link_libraries(consumer PRIVATE host_leaves)\n"
    )
    (tmp_path / "consumer.cpp").write_text(
        '#include "dft/host_hessian_aot.hpp"\n'
        '#include "dft/host_xc_response.hpp"\n'
        "#include <iostream>\n"
        "int main() {\n"
        "  using namespace generativeqc::dft::host_hessian_aot;\n"
        "  if (!generativeqc::dft::HostGgaXcResponse::workspace_bytes({2, 7, 1, 3})) return 3;\n"
        "  for (const auto& entry : programs()) {\n"
        "    if (find(entry.name, entry.identity) != &entry) return 1;\n"
        '    std::cout << entry.name << " " << entry.identity << " " << entry.outputs << "\\n";\n'
        "  }\n"
        "  for (const auto& binding : method_bindings()) {\n"
        "    if (find_method(binding.identity) != &binding ||\n"
        '        !find(binding.spins == 1 ? "xc_unpolarized" : "xc_polarized", binding.xc_identity) ||\n'
        '        !find(binding.spins == 1 ? "coefficient_response_unpolarized" : "coefficient_response_polarized",\n'
        "              binding.coefficient_response_identity)) return 4;\n"
        "    if (!binding.field_response_identity.empty() &&\n"
        '        !find(binding.spins == 1 ? "gga_field_response_unpolarized" : "gga_field_response_polarized",\n'
        "              binding.field_response_identity)) return 6;\n"
        "    if (!binding.energy_mixed_identity.empty() &&\n"
        '        !find(binding.spins == 1 ? "gga_energy_mixed_unpolarized" : "gga_energy_mixed_polarized",\n'
        "              binding.energy_mixed_identity)) return 7;\n"
        '    std::cout << "method " << binding.identity << "\\n";\n'
        "  }\n"
        '  if (find_method("B3LYP") || find_method("absent")) return 5;\n'
        '  return find("absent", "absent") ? 2 : 0;\n}\n'
    )
    build = tmp_path / "build"
    subprocess.run(
        [
            cmake,
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CXX_COMPILER=" + required_native_cxx.compiler,
            "-DCMAKE_CXX_COMPILER_LAUNCHER=" + required_native_cxx.cache,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    full_arrays = payload["arrays"]
    for spins, arrays in (
        (payload["spins"], full_arrays),
        (["unpolarized"], full_arrays),
        (["unpolarized"], ["xc", "coefficient_response"]),
        (["unpolarized"], ["xc"]),
        (["unpolarized"], ["coefficient_response"]),
    ):
        payload["spins"] = spins
        payload["arrays"] = arrays
        inventory.write_text(json.dumps(payload))
        subprocess.run(
            [cmake, "--build", str(build), "--parallel", "2", "--verbose"],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        result = subprocess.run(
            [str(build / "consumer")],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        expected = programs(parse_inventory(payload))
        expected_methods = (
            [
                "method " + resolve_method(payload["method"], spin=spin).identity
                for spin in sorted(spins)
            ]
            if {"xc", "coefficient_response"} <= set(arrays)
            else []
        )
        assert (
            result.stdout.splitlines()
            == [
                f"{entry.name} {entry.identity} {len(entry.roots)}"
                for entry in expected
            ]
            + expected_methods
        )


@pytest.mark.parametrize(
    "python,inventory", [("", "unused.json"), (sys.executable, "absent.json")]
)
def test_cmake_prerequisite_refusals(
    tmp_path: Path, python: str, inventory: str
) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is unavailable")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(host_leaf_refusal NONE)\n"
        f'set(Python3_EXECUTABLE "{python}")\n'
        f'include("{ROOT.as_posix()}/cmake/GenerativeQCHostHessianAot.cmake")\n'
        f'generativeqc_add_host_hessian_aot(host_leaves "{inventory}")\n'
    )
    result = subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(tmp_path / "build")],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert (
        "requires build-time Python" if not python else "inventory is missing"
    ) in result.stderr


def test_cpu_coupled_consumer_links_and_runs_without_python_on_path(
    tmp_path: Path,
    required_native_cxx: Any,
    policy_library: Any,
) -> None:
    """Exercise the optional static consumer with the ordinary CPU core target."""
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required to qualify the native coupled consumer")
    core = Path(policy_library._name).resolve()
    inventory = ROOT / "manifests/host_hessian_aot_b3lyp.json"
    (tmp_path / "generated_atomic_grid.hpp").write_text(emit_atomic_grid_header())
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(host_cpks_consumer LANGUAGES CXX)\n"
        f'set(PROJECT_SOURCE_DIR "{ROOT.as_posix()}")\n'
        f'set(Python3_EXECUTABLE "{sys.executable}")\n'
        "set(GENERATIVEQC_ENABLE_CUDA OFF)\n"
        f'include("{ROOT.as_posix()}/cmake/GenerativeQCGenerated.cmake")\n'
        f'include("{ROOT.as_posix()}/cmake/GenerativeQCHostHessianAot.cmake")\n'
        f'generativeqc_add_host_hessian_aot(host_leaves "{inventory.as_posix()}")\n'
        "add_library(native_core SHARED IMPORTED)\n"
        f'set_target_properties(native_core PROPERTIES IMPORTED_LOCATION "{core.as_posix()}"\n'
        f'  INTERFACE_INCLUDE_DIRECTORIES "{ROOT.as_posix()}/include")\n'
        "generativeqc_add_host_ks_response(host_cpks host_leaves native_core)\n"
        f'add_executable(consumer consumer.cpp "{ROOT.as_posix()}/tests/native/host_ks_response_probe.cpp")\n'
        "target_link_libraries(consumer PRIVATE host_cpks)\n"
        f'add_executable(nuclear_consumer nuclear.cpp "{ROOT.as_posix()}/tests/native/host_xc_nuclear_probe.cpp"\n'
        f'  "{ROOT.as_posix()}/src/dft/grid.cpp")\n'
        f'target_include_directories(nuclear_consumer PRIVATE "{tmp_path.as_posix()}")\n'
        "target_link_libraries(nuclear_consumer PRIVATE host_cpks)\n"
        f'add_executable(integral_consumer integral.cpp "{ROOT.as_posix()}/tests/native/host_nuclear_integrals_probe.cpp")\n'
        "target_link_libraries(integral_consumer PRIVATE host_cpks)\n"
        f'add_executable(mixed_consumer "{ROOT.as_posix()}/tests/native/host_mixed_primitives_probe.cpp")\n'
        "target_compile_definitions(mixed_consumer PRIVATE GENERATIVEQC_HOST_MIXED_STANDALONE)\n"
        "target_link_libraries(mixed_consumer PRIVATE host_cpks)\n"
        f'add_executable(grid_column_consumer "{ROOT.as_posix()}/tests/native/host_grid_mixed_geometry_probe.cpp"\n'
        f'  "{ROOT.as_posix()}/src/dft/grid.cpp")\n'
        "target_compile_definitions(grid_column_consumer PRIVATE GENERATIVEQC_MIXED_GRID_STANDALONE=1)\n"
        f'target_include_directories(grid_column_consumer PRIVATE "{tmp_path.as_posix()}")\n'
        "target_link_libraries(grid_column_consumer PRIVATE host_cpks)\n"
        f'add_executable(xc_column_consumer "{ROOT.as_posix()}/tests/native/host_xc_column_probe.cpp"\n'
        f'  "{ROOT.as_posix()}/src/dft/grid.cpp")\n'
        "target_compile_definitions(xc_column_consumer PRIVATE GENERATIVEQC_XC_COLUMN_STANDALONE=1)\n"
        f'target_include_directories(xc_column_consumer PRIVATE "{tmp_path.as_posix()}")\n'
        "target_link_libraries(xc_column_consumer PRIVATE host_cpks)\n"
    )
    (tmp_path / "consumer.cpp").write_text(
        r"""
#include "dft/host_hessian_aot.hpp"
#include <cmath>
#include <iostream>
#include <vector>
extern "C" const char* host_ks_error();
extern "C" void* host_ks_create(unsigned, std::size_t, std::size_t, const char*, int);
extern "C" void host_ks_destroy(void*);
extern "C" void host_ks_sizes(void*, std::size_t*);
extern "C" int host_ks_apply(void*, const double*, std::size_t, double*, std::size_t);
extern "C" int host_ks_solve(void*, const double*, double*, std::size_t, std::size_t,
                              double*, std::size_t*);
extern "C" int host_ks_perturbation_bind(void*, std::size_t, std::size_t);
extern "C" int host_ks_perturbation_solve(void*, const double*, const double*, double*, double*,
    std::size_t, std::size_t, double*, std::size_t*);
int main() {
  using namespace generativeqc::dft::host_hessian_aot;
  for (const auto& binding : method_bindings()) {
    void* handle = host_ks_create(binding.spins, 256, 4 << 20, binding.identity.data(), 0);
    if (!handle) { std::cerr << host_ks_error(); return 1; }
    std::size_t sizes[6]{}, work[12]{};
    double diagnostic[3]{};
    host_ks_sizes(handle, sizes);
    std::vector<double> rhs(sizes[3], .02), solution(sizes[3]), action(sizes[3]);
    const auto status = host_ks_solve(handle, rhs.data(), solution.data(), 150, 1 << 20,
                                     diagnostic, work);
    if (status != 0 && status != 1) { std::cerr << host_ks_error(); return 2; }
    if (host_ks_apply(handle, solution.data(), sizes[3], action.data(), sizes[3])) return 3;
    for (std::size_t index = 0; index < sizes[3]; ++index)
      if (!std::isfinite(action[index]) || std::abs(action[index] - rhs[index]) > 2e-11) return 4;
    if (!(diagnostic[1] < 2e-11) || !work[4] || work[4] > work[3]) return 5;
    const auto matrix = sizes[0] * sizes[0], matrices = matrix * binding.spins;
    std::vector<double> metric(matrix), frozen(matrices), density(matrices), weights(matrices);
    for (std::size_t row = 0; row < sizes[0]; ++row) {
      metric[row * sizes[0] + row] = .01;
      for (std::size_t spin = 0; spin < binding.spins; ++spin)
        frozen[spin * matrix + row * sizes[0] + row] = .02 * (1 + row + spin);
    }
    if (host_ks_perturbation_bind(handle, 8 << 20, 1 << 30)) return 6;
    if (host_ks_perturbation_solve(handle, metric.data(), frozen.data(), density.data(),
        weights.data(), 150, 1 << 20, diagnostic, work)) { std::cerr << host_ks_error(); return 7; }
    for (std::size_t index = 0; index < matrices; ++index)
      if (!std::isfinite(density[index]) || !std::isfinite(weights[index])) return 8;
    if (!(diagnostic[1] < 2e-11) || !work[5] || work[5] > work[3]) return 9;
    std::cout << binding.spins << " " << sizes[3] << "\n";
    host_ks_destroy(handle);
  }
}
"""
    )
    (tmp_path / "nuclear.cpp").write_text(
        r"""
#include "dft/host_hessian_aot.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
extern "C" const char* host_nuclear_error();
extern "C" void* host_nuclear_create(const double*, unsigned, unsigned, std::size_t,
                                     std::size_t, const double*, const char*, const char*);
extern "C" void host_nuclear_destroy(void*);
extern "C" int host_nuclear_apply(void*, const double*, std::size_t, double*, std::size_t);
extern "C" void host_nuclear_sizes(void*, std::size_t*);
extern "C" void host_nuclear_work(void*, std::size_t*);
int main() {
  using namespace generativeqc::dft::host_hessian_aot;
  const Program* motion = nullptr;
  for (const auto& entry : programs()) if (entry.name == "ao_first_motion") motion = &entry;
  if (!motion) return 1;
  const double centers[]{.1, -.2, .3, 1.5, .3, -.2};
  const double direction[]{.17, -.08, .12, -.09, .11, -.15};
  const double translation[]{.2, -.1, .3, .2, -.1, .3};
  for (const auto& binding : method_bindings()) {
    std::vector<double> density(4 * binding.spins), result(density.size());
    for (std::size_t spin = 0; spin < binding.spins; ++spin) {
      density[4 * spin] = 1;
      density[4 * spin + 3] = .7;
    }
    void* handle = host_nuclear_create(centers, 0, 2, binding.spins, 17, density.data(),
                                       motion->identity.data(), binding.identity.data());
    if (!handle) { std::cerr << host_nuclear_error(); return 2; }
    if (host_nuclear_apply(handle, direction, 6, result.data(), result.size())) {
      std::cerr << host_nuclear_error(); return 3;
    }
    double maximum = 0;
    for (double value : result) {
      if (!std::isfinite(value)) return 4;
      maximum = std::max(maximum, std::abs(value));
    }
    if (maximum < 1e-10) return 5;
    std::size_t sizes[10]{}, work[11]{};
    host_nuclear_sizes(handle, sizes);
    host_nuclear_work(handle, work);
    if (work[2] != sizes[1] || work[6] != 1 || work[8] != sizes[1] || work[10] != sizes[7]) return 6;
    if (host_nuclear_apply(handle, translation, 6, result.data(), result.size())) return 7;
    for (double value : result) if (std::abs(value) > 1e-12) return 8;
    host_nuclear_destroy(handle);
    std::cout << "nuclear " << binding.spins << "\n";
  }
}
"""
    )
    (tmp_path / "integral.cpp").write_text(
        r"""
#include "dft/host_hessian_aot.hpp"
#include <cmath>
#include <iostream>
#include <vector>
extern "C" const char* host_integral_error();
extern "C" void* host_integral_create(const double*, unsigned, int, int, int, int,
                                      std::size_t, const double*, const char*, std::size_t);
extern "C" void host_integral_destroy(void*);
extern "C" int host_integral_apply(void*, const double*, std::size_t, double*, std::size_t,
                                   double*, std::size_t);
extern "C" void host_integral_sizes(void*, std::size_t*);
extern "C" void host_integral_work(void*, std::size_t*);
extern "C" int host_curvature_bind(void*, const double*, std::size_t, const char*,
                                    const std::size_t*, std::size_t);
extern "C" int host_curvature_apply(void*, const double*, std::size_t, const double*,
                                     std::size_t, double*, std::size_t);
extern "C" void host_curvature_work(void*, std::size_t*);
extern "C" void host_curvature_resources(void*, std::size_t*);
extern "C" int host_column_bind(void*, const double*, std::size_t, const char*,
                                 const std::size_t*, std::size_t);
extern "C" int host_column_apply(void*, const double*, std::size_t, double*, std::size_t);
extern "C" int host_column_response(void*, const double*, std::size_t, const double*, std::size_t,
                                     const double*, std::size_t, double*, std::size_t);
int main() {
  using namespace generativeqc::dft::host_hessian_aot;
  const double centers[]{.13, -.21, .34, 1.7, .28, -.42};
  const double direction[]{.16, -.09, .13, -.12, .07, -.11};
  const double translation[]{.13, -.07, .11, .13, -.07, .11};
  for (const auto& binding : method_bindings()) {
    std::vector<double> density(binding.spins * 16), overlap(16), fock(density.size());
    for (std::size_t spin = 0; spin < binding.spins; ++spin)
      for (std::size_t index = 0; index < 4; ++index)
        density[spin * 16 + index * 5] = .1 * (1 + index + spin);
    void* handle = host_integral_create(centers, 1, 1, 1, 1, 0, binding.spins,
                                       density.data(), binding.identity.data(), 64);
    if (!handle) { std::cerr << host_integral_error(); return 1; }
    if (host_integral_apply(handle, direction, 6, overlap.data(), overlap.size(),
                            fock.data(), fock.size())) {
      std::cerr << host_integral_error(); return 2;
    }
    double overlap_norm{}, fock_norm{};
    for (const auto value : overlap) {
      if (!std::isfinite(value)) return 3;
      overlap_norm += value * value;
    }
    for (const auto value : fock) {
      if (!std::isfinite(value)) return 4;
      fock_norm += value * value;
    }
    if (!(overlap_norm > 1e-8) || !(fock_norm > 1e-8)) return 5;
    std::size_t sizes[7]{}, work[8]{};
    host_integral_sizes(handle, sizes);
    host_integral_work(handle, work);
    // Resource admission is conservative; completed semantic work can be smaller.
    if (work[0] != 1 || work[1] != 1 || work[2] != 16 || work[3] != 256 ||
        !work[4] || work[4] > sizes[3] || !work[5] || work[5] > sizes[4] ||
        !work[6] || !work[7] || work[7] > sizes[5]) return 6;
    if (host_integral_apply(handle, translation, 6, overlap.data(), overlap.size(),
                            fock.data(), fock.size())) return 7;
    for (const auto value : overlap) if (std::abs(value) > 2e-12) return 8;
    for (const auto value : fock) if (std::abs(value) > 2e-12) return 9;
    std::vector<double> weighted_density(density.size());
    for (std::size_t index = 0; index < density.size(); ++index)
      weighted_density[index] = .3 * density[index];
    if (host_curvature_bind(handle, weighted_density.data(), weighted_density.size(),
                            binding.identity.data(), nullptr, 64)) return 10;
    double curvature[4]{};
    if (host_curvature_apply(handle, direction, 6, direction, 6, curvature, 4)) return 11;
    const double mixed = curvature[3];
    for (const auto value : curvature) if (!std::isfinite(value)) return 12;
    if (!(std::abs(curvature[3]) > 1e-8)) return 13;
    std::size_t planned[4]{};
    host_curvature_work(handle, work);
    host_curvature_resources(handle, planned);
    if (work[0] != 1 || work[1] != 1 || work[2] != 16 || work[3] != 256 ||
        work[4] != planned[1] || work[5] != planned[2] || work[7] != planned[3]) return 14;
    if (host_curvature_apply(handle, translation, 6, direction, 6, curvature, 4)) return 15;
    if (curvature[1] != 0 || curvature[3] != 0) return 16;
    if (host_column_bind(handle, weighted_density.data(), weighted_density.size(),
                         binding.identity.data(), nullptr, 64)) return 17;
    double column[14]{};
    if (host_column_apply(handle, direction, 6, column, 14)) return 18;
    double projected{};
    for (std::size_t coordinate = 0; coordinate < 6; ++coordinate)
      projected += direction[coordinate] * column[8 + coordinate];
    if (std::abs(projected - mixed) > 3e-12 + 3e-11 * std::abs(mixed)) return 19;
    host_curvature_work(handle, work);
    host_curvature_resources(handle, planned);
    if (work[0] != 1 || work[1] != 1 || work[2] != 16 || work[3] != 256 ||
        work[4] != planned[1] || work[5] != planned[2] || work[7] != planned[3]) return 20;
    std::vector<double> density_right(density.size()), weighted_right(density.size());
    for (std::size_t index = 0; index < density.size(); ++index) {
      density_right[index] = .07 * density[index];
      weighted_right[index] = -.04 * weighted_density[index];
    }
    double relaxed[14]{}, density_only[14]{}, zero[6]{};
    if (host_column_response(handle, direction, 6, density_right.data(), density_right.size(),
                              weighted_right.data(), weighted_right.size(), relaxed, 14) ||
        host_column_response(handle, zero, 6, density_right.data(), density_right.size(),
                              weighted_right.data(), weighted_right.size(), density_only, 14)) return 24;
    double response_norm{};
    for (std::size_t coordinate = 0; coordinate < 6; ++coordinate) {
      if (relaxed[2 + coordinate] != column[2 + coordinate] ||
          std::abs(relaxed[8 + coordinate] - column[8 + coordinate] - density_only[8 + coordinate]) > 3e-12) return 25;
      response_norm += std::abs(density_only[8 + coordinate]);
    }
    if (!(response_norm > 1e-8) || std::abs(relaxed[1] - column[1] - density_only[1]) > 3e-12) return 26;
    if (host_column_apply(handle, translation, 6, column, 14)) return 21;
    if (column[1] != 0) return 22;
    for (std::size_t coordinate = 0; coordinate < 6; ++coordinate)
      if (column[8 + coordinate] != 0) return 23;
    host_integral_destroy(handle);
    std::cout << "integral " << binding.spins << "\n";
  }
}
"""
    )
    build = tmp_path / "build"
    subprocess.run(
        [
            cmake,
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CXX_COMPILER=" + required_native_cxx.compiler,
            "-DCMAKE_CXX_COMPILER_LAUNCHER=" + required_native_cxx.cache,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    built = subprocess.run(
        [cmake, "--build", str(build), "--parallel", "2", "--verbose"],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert required_native_cxx.cache in built.stdout
    empty_path = tmp_path / "no-python-on-path"
    empty_path.mkdir()
    result = subprocess.run(
        [str(build / "consumer")],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
        env={
            **os.environ,
            "PATH": str(empty_path),
            "PYTHONPATH": "",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
    )
    assert result.stdout.splitlines() == ["2 13", "1 3"]
    result = subprocess.run(
        [str(build / "nuclear_consumer")],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **os.environ,
            "PATH": str(empty_path),
            "PYTHONPATH": "",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
    )
    assert result.stdout.splitlines() == ["nuclear 2", "nuclear 1"]
    result = subprocess.run(
        [str(build / "integral_consumer")],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "PATH": str(empty_path),
            "PYTHONPATH": "",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
    )
    assert result.stdout.splitlines() == ["integral 2", "integral 1"]
    subprocess.run(
        [str(build / "mixed_consumer")],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PATH": str(empty_path), "PYTHONPATH": ""},
    )
    subprocess.run(
        [str(build / "grid_column_consumer")],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PATH": str(empty_path), "PYTHONPATH": ""},
    )
    result = subprocess.run(
        [str(build / "xc_column_consumer")],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PATH": str(empty_path), "PYTHONPATH": ""},
    )
    assert (
        result.stdout.strip()
        == "XC columns: frozen/supplied-density, Cartesian/spherical RKS/UKS"
    )


@pytest.mark.parametrize(
    "missing,cuda", [("leaves", False), ("core", False), ("", True)]
)
def test_coupled_cmake_requires_explicit_cpu_core_and_leaf_targets(
    tmp_path: Path,
    missing: str,
    cuda: bool,
) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is unavailable")
    targets = "\n".join(
        "add_library(" + name + " INTERFACE)"
        for name in ("leaves", "core")
        if name != missing
    )
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\nproject(host_cpks_refusal NONE)\n"
        f"set(GENERATIVEQC_ENABLE_CUDA {'ON' if cuda else 'OFF'})\n"
        f'include("{ROOT.as_posix()}/cmake/GenerativeQCHostHessianAot.cmake")\n'
        + targets
        + "\ngenerativeqc_add_host_ks_response(host_cpks leaves core)\n"
    )
    result = subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(tmp_path / "build")],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert (
        "CPU-only native core" if cuda else "requires explicit linked leaves"
    ) in result.stderr


@pytest.mark.parametrize("primitive_aot", [False, True])
def test_coupled_cmake_requires_primitive_ad_and_exact_leaf_inventory(
    tmp_path: Path, primitive_aot: bool
) -> None:
    """Missing integral authorities fail at configure, not at a later action."""
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is unavailable")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\nproject(host_integral_refusal NONE)\n"
        "set(GENERATIVEQC_ENABLE_CUDA OFF)\n"
        f"set(GENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT {'ON' if primitive_aot else 'OFF'})\n"
        f'set(Python3_EXECUTABLE "{sys.executable}")\n'
        f'include("{ROOT.as_posix()}/cmake/GenerativeQCHostHessianAot.cmake")\n'
        "add_library(leaves INTERFACE)\nadd_library(core INTERFACE)\n"
        "generativeqc_add_host_ks_response(host_cpks leaves core)\n"
    )
    result = subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(tmp_path / "build")],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert (
        "exact leaf inventory"
        if primitive_aot
        else "linked stationary CPU primitive AD"
    ) in result.stderr
