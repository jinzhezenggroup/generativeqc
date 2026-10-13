"""Structural build tests for modular CMake ownership."""

import json
import shutil
import subprocess
import sys
import typing
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text()


def test_generated_commands_are_centralized() -> None:
    """Project declarations should use the shared generated-source helper."""
    helper = _read("cmake/GenerativeQCGenerated.cmake")
    assert helper.count("add_custom_command(") == 1
    for relative in (
        "CMakeLists.txt",
        "cmake/GenerativeQCGeneratedSources.cmake",
        "cmake/GenerativeQCCuda.cmake",
    ):
        assert "add_custom_command(" not in _read(relative)


def test_cuda_implib_generation_runs_at_build_time() -> None:
    """CUDA import trampolines belong to the incremental build graph."""
    implib = _read("cmake/GenerativeQCCudaImplib.cmake")
    assert "execute_process(" not in implib
    assert "generativeqc_register_generated_sources(" in implib
    assert "file(GENERATE" in implib
    assert "if(NOT TARGET ${codegen_target})" in implib


def test_generated_command_preserves_list_valued_arguments(
    tmp_path: typing.Any,
) -> None:
    """A multi-SM generator request must arrive as one semicolon-delimited value."""
    generator = tmp_path / "record_args.py"
    generator.write_text(
        "import json, pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))\n"
    )
    output = tmp_path / "arguments.json"
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(GeneratorArguments LANGUAGES NONE)\n"
        f'include("{ROOT / "cmake/GenerativeQCGenerated.cmake"}")\n'
        f'set(Python3_EXECUTABLE "{sys.executable}")\n'
        'set(architectures "90;120")\n'
        "generativeqc_register_generated_sources(\n"
        "  NAME generate\n"
        f'  GENERATOR "{generator}"\n'
        f'  OUTPUTS "{output}"\n'
        f'  ARGS "{output}" --architectures "${{architectures}}" "with spaces")\n'
    )
    build = tmp_path / "build"
    subprocess.run(["cmake", "-S", str(tmp_path), "-B", str(build)], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "generate"], check=True)
    assert json.loads(output.read_text()) == [
        "--architectures",
        "90;120",
        "with spaces",
    ]


def test_generated_command_depfile_tracks_loaded_python_modules(
    tmp_path: typing.Any,
) -> None:
    """Only Python modules loaded by a generator should invalidate its output."""
    dependency = tmp_path / "used_dep.py"
    dependency.write_text("VALUE = 1\n")
    unused = tmp_path / "unused_dep.py"
    unused.write_text("VALUE = 1\n")
    generator = tmp_path / "generate.py"
    output = tmp_path / "generated.txt"
    counter = tmp_path / "runs.txt"
    generator.write_text(
        "import pathlib, sys\n"
        "import used_dep\n"
        "output, counter = map(pathlib.Path, sys.argv[1:3])\n"
        "runs = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "counter.write_text(str(runs))\n"
        "output.write_text(str(used_dep.VALUE))\n"
    )
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(GeneratorDepfile LANGUAGES NONE)\n"
        f'include("{ROOT / "cmake/GenerativeQCGenerated.cmake"}")\n'
        f'set(Python3_EXECUTABLE "{sys.executable}")\n'
        "generativeqc_register_generated_sources(\n"
        "  NAME generate\n"
        f'  GENERATOR "{generator}"\n'
        f'  OUTPUTS "{output}"\n'
        f'  ARGS "{output}" "{counter}")\n'
    )
    build = tmp_path / "build"
    subprocess.run(["cmake", "-S", str(tmp_path), "-B", str(build)], check=True)
    command = ["cmake", "--build", str(build), "--target", "generate"]
    subprocess.run(command, check=True)
    assert counter.read_text() == "1"

    unused.write_text("VALUE = 2\n")
    subprocess.run(command, check=True)
    assert counter.read_text() == "1"

    dependency.write_text("VALUE = 200\n")
    subprocess.run(command, check=True)
    assert counter.read_text() == "2"
    assert output.read_text() == "200"
    depfile = (tmp_path / "generated.txt.d").read_text()
    assert str(dependency) in depfile
    assert str(unused) not in depfile


def test_project_codegen_does_not_depend_on_every_compiler_module() -> None:
    """Keep generator invalidation narrower than the complete source identity."""
    assert "DEPFILE" in _read("cmake/GenerativeQCGenerated.cmake")
    assert "run_codegen.py" in _read("cmake/GenerativeQCGenerated.cmake")
    for relative in (
        "CMakeLists.txt",
        "cmake/GenerativeQCGeneratedSources.cmake",
        "cmake/GenerativeQCCuda.cmake",
    ):
        assert "GENERATIVEQC_SCIENTIFIC_COMPILER_INPUTS" not in _read(relative)


def test_production_sources_are_explicit_and_component_owned() -> None:
    """Do not trade a monolithic list for implicit recursive source globbing."""
    sources = _read("cmake/GenerativeQCSources.cmake")
    assert "file(GLOB" not in sources
    for owner in (
        "generativeqc_add_runtime_sources",
        "generativeqc_add_integrals_scf_sources",
        "generativeqc_add_dft_sources",
        "generativeqc_add_posthf_cc_sources",
    ):
        assert f"function({owner}" in sources


def test_top_level_is_composition_only_for_sources_and_codegen() -> None:
    cmake = _read("CMakeLists.txt")
    for module in (
        "GenerativeQCGeneratedSources.cmake",
        "GenerativeQCSources.cmake",
        "GenerativeQCCuda.cmake",
    ):
        assert module in cmake
    assert "generativeqc_add_component_sources(generativeqc)" in cmake
    assert "generativeqc_register_host_generated_sources(generativeqc)" in cmake
    assert "generativeqc_register_cuda_generated_sources(generativeqc)" in cmake


def test_source_identity_hashing_runs_at_build_time(tmp_path: typing.Any) -> None:
    """Source edits should rebuild the identity header without reconfiguring CMake."""
    cmake = _read("CMakeLists.txt")
    identity_cmake = _read("cmake/GenerativeQCSourceIdentity.cmake")
    manifest = json.loads(_read("cmake/GenerativeQCSourceIdentity.json"))

    assert "file(SHA256" not in cmake
    assert "CMAKE_CONFIGURE_DEPENDS" not in cmake
    assert "tools/generate_build_identity.py" in cmake
    assert "CONFIGURE_DEPENDS" in identity_cmake
    assert (
        'set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_manifest_path}")'
        in identity_cmake
    )
    assert "tools/generate_build_identity.py" in manifest["files"]

    output = tmp_path / "build_identity.hpp"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_build_identity.py"),
            "--source-root",
            str(ROOT),
            "--manifest",
            str(ROOT / "cmake/GenerativeQCSourceIdentity.json"),
            "--template",
            str(ROOT / "src/api/build_identity.hpp.in"),
            "--output",
            str(output),
            "--cuda-fast-compile",
            "OFF",
            "--release-build",
            "ON",
        ],
        check=True,
    )
    rendered = output.read_text()
    assert 'kGenerativeQCSourceIdentity = "' in rendered
    assert "#define GENERATIVEQC_CUDA_FAST_COMPILE 0" in rendered
    assert "#define GENERATIVEQC_TUNING_RELEASE_BUILD 1" in rendered


def test_generated_commands_use_explicit_dependency_boundary_when_supported() -> None:
    """Newer CMake/Ninja should not infer unrelated transitive custom-command edges."""
    helper = _read("cmake/GenerativeQCGenerated.cmake")
    assert 'CMAKE_VERSION VERSION_GREATER_EQUAL "3.27"' in helper
    assert "DEPENDS_EXPLICIT_ONLY" in helper
    assert "${_generativeqc_explicit_dependency_boundary}" in helper


@pytest.mark.parametrize("library_type", ["SHARED", "STATIC"])
@pytest.mark.parametrize("pch", [False, True])
def test_host_object_component_codegen_compile_and_link(
    tmp_path: typing.Any, library_type: str, pch: bool
) -> None:
    """A host OBJECT target waits for generated headers and is linked into the SDK."""
    launcher = shutil.which("sccache") or shutil.which("ccache")
    if not launcher:
        pytest.skip("C++ compiler cache required for local build qualification")

    subprocess.run([launcher, "--version"], check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src/pch.hpp").write_text("#pragma once\n#include <cstddef>\n")
    (tmp_path / "toolkit").mkdir()
    (tmp_path / "toolkit/cuda_fixture.hpp").write_text("#pragma once\n")
    (tmp_path / "provider").mkdir()
    (tmp_path / "provider/provider_fixture.hpp").write_text("#pragma once\n")
    export = (
        "#ifdef _WIN32\n#define API __declspec(dllexport)\n"
        '#else\n#define API __attribute__((visibility("default")))\n#endif\n'
    )
    (tmp_path / "seed.hpp").write_text("#pragma once\n#define GENERATED_VALUE 7\n")
    (tmp_path / "host.cpp").write_text(
        export + '#include "fixture_generated.hpp"\n'
        '#include "cuda_fixture.hpp"\n'
        '#include "provider_fixture.hpp"\n'
        "#ifndef HOST_PARENT_DEFINE\n#error missing parent compilation definitions\n#endif\n"
        'extern "C" API int host_value() { return HOST_PARENT_DEFINE + GENERATED_VALUE'
        " + HOST_TRANSITIVE_DEFINE + HOST_OPTION_DEFINE; }\n"
    )
    (tmp_path / "core.cpp").write_text(
        export + 'extern "C" API int api_value() { return 5; }\n'
    )
    (tmp_path / "caller.cpp").write_text(
        'extern "C" int api_value();\n'
        'extern "C" int host_value();\n'
        "int main() { return api_value() == 5 && host_value() == 55 ? 0 : 1; }\n"
    )
    fixture = """cmake_minimum_required(VERSION 3.24)
project(HostObjectBoundary LANGUAGES CXX)
find_package(Threads REQUIRED)
include("{helper}")
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(GENERATIVEQC_ENABLE_CXX_PCH {pch})
set(GENERATIVEQC_ENABLE_CUDA ON)
set(GENERATIVEQC_CUDA_TOOLKIT_INCLUDE_DIRS "${{CMAKE_CURRENT_SOURCE_DIR}}/toolkit")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_CURRENT_BINARY_DIR}}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${{CMAKE_CURRENT_BINARY_DIR}}/bin")
add_library(fixture_provider INTERFACE)
target_include_directories(fixture_provider INTERFACE "${{CMAKE_CURRENT_SOURCE_DIR}}/provider")
target_compile_definitions(fixture_provider INTERFACE HOST_TRANSITIVE_DEFINE=12)
add_library(fixture_core {library_type} core.cpp)
target_link_libraries(fixture_core PRIVATE fixture_provider)
target_include_directories(fixture_core PRIVATE "${{CMAKE_CURRENT_BINARY_DIR}}")
target_compile_definitions(fixture_core PRIVATE HOST_PARENT_DEFINE=33)
target_compile_options(fixture_core PRIVATE -DHOST_OPTION_DEFINE=3)
add_custom_command(
  OUTPUT "${{CMAKE_CURRENT_BINARY_DIR}}/fixture_generated.hpp"
  COMMAND "${{CMAKE_COMMAND}}" -E copy_if_different
          "${{CMAKE_CURRENT_SOURCE_DIR}}/seed.hpp"
          "${{CMAKE_CURRENT_BINARY_DIR}}/fixture_generated.hpp"
  DEPENDS "${{CMAKE_CURRENT_SOURCE_DIR}}/seed.hpp"
  VERBATIM)
add_custom_target(fixture_codegen DEPENDS
  "${{CMAKE_CURRENT_BINARY_DIR}}/fixture_generated.hpp")
generativeqc_add_host_object_component(fixture_core fixture_host
  SOURCES host.cpp CODEGEN fixture_codegen)
add_executable(fixture_cli caller.cpp)
target_link_libraries(fixture_cli PRIVATE fixture_core)
if(APPLE)
  set_target_properties(fixture_cli PROPERTIES INSTALL_RPATH "@loader_path")
elseif(UNIX)
  set_target_properties(fixture_cli PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
install(TARGETS fixture_core fixture_cli
  RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)
""".format(
        helper=(ROOT / "cmake/GenerativeQCSources.cmake").as_posix(),
        library_type=library_type,
        pch="ON" if pch else "OFF",
    )
    (tmp_path / "CMakeLists.txt").write_text(fixture)
    build = tmp_path / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            f"-DCMAKE_CXX_COMPILER_LAUNCHER={launcher}",
        ],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build),
            "--config",
            "Release",
            "--target",
            "fixture_host",
            "--parallel",
        ],
        check=True,
    )
    assert (build / "fixture_generated.hpp").exists()
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build),
            "--config",
            "Release",
            "--target",
            "fixture_cli",
            "--parallel",
        ],
        check=True,
    )
    executable = "fixture_cli.exe" if sys.platform == "win32" else "fixture_cli"
    subprocess.run([str(build / "bin" / executable)], check=True)
    installed = tmp_path / "installed"
    subprocess.run(
        [
            "cmake",
            "--install",
            str(build),
            "--config",
            "Release",
            "--prefix",
            str(installed),
        ],
        check=True,
    )
    subprocess.run([str(installed / "bin" / executable)], check=True)
    assert not list(installed.rglob("*fixture_host*"))


@pytest.mark.parametrize(
    ("component", "headers"),
    [
        ("generativeqc_host_dft", {"generated_method_parameters.hpp"}),
        (
            "generativeqc_host_posthf_cc",
            {
                "generated_df_ccsd_cpu.hpp",
                "generated_df_ccsd_core_cpu.hpp",
                "generated_df_ccsd_hoisted_cpu.hpp",
            },
        ),
    ],
)
def test_host_components_own_transitive_generated_headers(
    tmp_path: typing.Any, component: str, headers: set[str]
) -> None:
    """Selective production builds must schedule indirect generated includes too."""
    ninja = shutil.which("ninja")
    if not ninja:
        pytest.skip("Ninja is required to inspect the executable build graph")
    build = tmp_path / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(ROOT),
            "-B",
            str(build),
            "-G",
            "Ninja",
            f"-DCMAKE_MAKE_PROGRAM={ninja}",
            "-DGENERATIVEQC_ENABLE_CUDA=OFF",
            "-DGENERATIVEQC_BUILD_CLI=OFF",
            "-DGENERATIVEQC_BUILD_TESTS=OFF",
            "-DGENERATIVEQC_CPU_LINALG_PROVIDER=scalar",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ],
        check=True,
    )
    # Ask Ninja for the commands in the actual component build plan without
    # running generators or compiling the library during this focused test.
    commands = subprocess.run(
        [ninja, "-C", str(build), "-t", "commands", component],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for header in headers:
        assert header in commands, f"{component} does not generate {header}"
