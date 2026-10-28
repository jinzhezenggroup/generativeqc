"""Build-graph regressions for optional CUDA XC, without compiling CUDA."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _target_block(source: str, target: str) -> str:
    start = source.index(f"add_executable({target}")
    end = source.index(f"set_target_properties({target}", start)
    return source[start:end]


def _assert_grid_registry_closure(source: str) -> None:
    consumers = []
    for target in re.findall(r"add_executable\((\w+)", source):
        start = source.index(f"add_executable({target}")
        declaration = source[start : source.index(")", start) + 1]
        if '"${GENERATIVEQC_GRID_SOURCE}"' not in declaration:
            continue
        consumers.append(target)
        block = _target_block(source, target)
        assert "$<TARGET_OBJECTS:generativeqc_libxc_semilocal_cuda>" in block
        assert '"${GENERATIVEQC_LIBXC_SEMILOCAL_CUDA_DIRECTORY}"' in block
        dependencies = re.findall(r"add_dependencies\([^)]*\)", block)
        assert any(
            "generativeqc_libxc_semilocal_cuda_codegen" in value
            for value in dependencies
        )
    assert consumers, "the generated grid must have standalone native consumers"


def test_every_standalone_grid_consumer_has_registry_compile_and_link_closure() -> None:
    source = (ROOT / "cmake/GenerativeQCTests.cmake").read_text()
    _assert_grid_registry_closure(source)


@pytest.mark.parametrize(
    "removed",
    (
        "$<TARGET_OBJECTS:generativeqc_libxc_semilocal_cuda>",
        '"${GENERATIVEQC_LIBXC_SEMILOCAL_CUDA_DIRECTORY}"',
        "generativeqc_libxc_semilocal_cuda_codegen",
    ),
)
def test_grid_closure_guard_rejects_each_missing_cosx_dependency(removed: str) -> None:
    source = (ROOT / "cmake/GenerativeQCTests.cmake").read_text()
    block = _target_block(source, "generativeqc_cosx_cuda_tests")
    with pytest.raises(AssertionError):
        _assert_grid_registry_closure(source.replace(block, block.replace(removed, "")))


@pytest.mark.parametrize("wheel", (False, True), ids=("native", "wheel"))
@pytest.mark.parametrize("enabled", (False, True), ids=("off", "on"))
def test_registry_object_keeps_wheel_headers_separate_from_provider_linkage(
    tmp_path: Path, wheel: bool, enabled: bool
) -> None:
    """Configure the actual object wiring without a CUDA toolkit or GPU build."""
    cmake = shutil.which("cmake")
    launcher = shutil.which("sccache") or shutil.which("ccache")
    if cmake is None or launcher is None:
        pytest.skip("object configuration probe requires CMake and a compiler cache")
    assert cmake is not None and launcher is not None
    subprocess.run([launcher, "--version"], check=True, capture_output=True, timeout=10)
    source = (ROOT / "cmake/GenerativeQCGeneratedSources.cmake").read_text()
    start = source.index("  # Reuse the exact AOT objects")
    end = source.index("  generativeqc_register_generated_sources(", start)
    wiring = source[start:end]
    registry = tmp_path / "registry"
    registry.mkdir()
    toolkit = tmp_path / "toolkit-headers"
    toolkit.mkdir()
    (toolkit / "cuda_runtime_api.h").write_text("using cudaStream_t = void*;\n")
    (registry / "registry.cpp").write_text("#include <cuda_runtime_api.h>\n")
    sources = [registry / "registry.cpp"]
    if enabled:
        for index in range(32):
            shard = registry / f"shard_{index}.cu"
            shard.write_text("// inert configuration probe\n")
            sources.append(shard)
    (tmp_path / "dummy.cpp").write_text("int fixture;\n")
    source_list = " ".join(f'"{path}"' for path in sources)
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(AutomaticCudaObject LANGUAGES CXX)\n"
        "set(CMAKE_EXPORT_COMPILE_COMMANDS ON)\n"
        f'set(PROJECT_SOURCE_DIR "{ROOT}")\n'
        f"set(GENERATIVEQC_PYTHON_WHEEL {'ON' if wheel else 'OFF'})\n"
        f'set(GENERATIVEQC_CUDA_TOOLKIT_INCLUDE_DIRS "{toolkit}")\n'
        f'set(GENERATIVEQC_LIBXC_SEMILOCAL_CUDA_DIRECTORY "{registry}")\n'
        f"set(GENERATIVEQC_LIBXC_SEMILOCAL_CUDA_SOURCES {source_list})\n"
        # CUDA shards only exercise target wiring here; never invoke NVCC.
        f"set_source_files_properties({source_list} PROPERTIES LANGUAGE CXX)\n"
        "set(_generativeqc_cuda_compile_architectures 120)\n"
        "set(target generativeqc)\n"
        "add_library(generativeqc STATIC dummy.cpp)\n"
        "add_custom_target(generativeqc_libxc_semilocal_cuda_codegen)\n"
        "if(GENERATIVEQC_PYTHON_WHEEL)\n"
        "  set(CMAKE_CUDA_RUNTIME_LIBRARY None)\n"
        "endif()\n"
        "if(NOT GENERATIVEQC_PYTHON_WHEEL)\n"
        "  add_library(CUDA::cudart INTERFACE IMPORTED)\n"
        "  set_target_properties(CUDA::cudart PROPERTIES\n"
        f'    INTERFACE_INCLUDE_DIRECTORIES "{toolkit}")\n'
        "endif()\n"
        f'include("{ROOT / "cmake/GenerativeQCCuda.cmake"}")\n'
        f'include("{ROOT / "cmake/GenerativeQCCudaImplib.cmake"}")\n'
        + wiring
        + "get_target_property(links generativeqc_libxc_semilocal_cuda LINK_LIBRARIES)\n"
        'file(WRITE "${CMAKE_BINARY_DIR}/links.txt" "${links}")\n'
        "get_target_property(runtime generativeqc_libxc_semilocal_cuda CUDA_RUNTIME_LIBRARY)\n"
        'file(WRITE "${CMAKE_BINARY_DIR}/runtime.txt" "${runtime}")\n'
    )
    build = tmp_path / "build"
    result = subprocess.run(
        [
            cmake,
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            f"-DCMAKE_CXX_COMPILER_LAUNCHER={launcher}",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    links = (build / "links.txt").read_text()
    assert ("CUDA::cudart" in links) is not wheel
    assert ("generativeqc_cuda_wheel_imports" in links) is wheel
    assert (build / "runtime.txt").read_text() == (
        "None" if wheel else "runtime-NOTFOUND"
    )
    commands = json.loads((build / "compile_commands.json").read_text())
    registry_commands = [row for row in commands if row["file"] == str(sources[0])]
    assert len(registry_commands) == 1
    assert str(toolkit) in registry_commands[0]["command"]
    shim = str(ROOT / "src/runtime/nvidia_host_api")
    assert (shim in registry_commands[0]["command"]) is wheel
    assert len(commands) == (34 if enabled else 2)


_SCIENCE_INPUTS = (
    "python/generativeqc_compiler/xc/libxc_bulk_catalog.json",
    "python/generativeqc_compiler/common/identity_only.py",
    "python/generativeqc_compiler/integral/production_df_values.json",
    "upstream/libxc-fulltree/7.0.0/maple/input.mpl",
    "upstream/libxc-fulltree/7.0.0/src/owner.c",
    "upstream/libxc/7.0.0/functionals.c",
    "upstream/libxc/7.0.0/work_mgga_inc.c",
)


@pytest.mark.parametrize(
    ("backend", "enabled", "functionals", "share_parameters"),
    (
        ("cuda", False, (), False),
        ("cuda", True, (), False),
        ("cuda", True, ("GGA_X_APBE", "MGGA_X_LTA"), False),
        ("cuda", True, ("MGGA_X_KTBM_0", "MGGA_X_KTBM_1"), True),
        ("cpu", True, (), False),
    ),
    ids=("cuda-off", "cuda-full", "cuda-selected", "cuda-shared", "cpu"),
)
def test_registry_incremental_build_tracks_science_only_when_enabled(
    tmp_path: Path,
    backend: str,
    enabled: bool,
    functionals: tuple[str, ...],
    share_parameters: bool,
) -> None:
    """Both registries must invalidate on hashed bytes absent from import depfiles."""
    cmake, ninja = shutil.which("cmake"), shutil.which("ninja")
    if cmake is None or ninja is None:
        pytest.skip("incremental graph probe requires CMake and Ninja")
    assert cmake is not None
    source = (ROOT / "cmake/GenerativeQCGeneratedSources.cmake").read_text()
    start = source.index(
        f"  set(GENERATIVEQC_LIBXC_SEMILOCAL_{backend.upper()}_DIRECTORY"
    )
    end = source.index(
        "  # Reuse the exact AOT objects"
        if backend == "cuda"
        else "  generativeqc_register_generated_sources(\n    NAME generativeqc_weighted_gram_cpu_codegen",
        start,
    )
    declaration = source[start:end]
    inputs = set(re.findall(r'"\$\{CMAKE_CURRENT_SOURCE_DIR\}/([^"*$]+)"', declaration))
    for relative in inputs | set(_SCIENCE_INPUTS):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source input\n")
    generator = tmp_path / f"tools/generate_libxc_semilocal_{backend}_registry.py"
    generator.parent.mkdir(parents=True, exist_ok=True)
    generator.write_text(
        "import argparse, pathlib\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--output-directory', type=pathlib.Path)\n"
        "p.add_argument('--disabled', action='store_true')\n"
        "p.add_argument('--functional', action='append')\n"
        "p.add_argument('--share-parameters', action=argparse.BooleanOptionalAction, default=False)\n"
        "p.add_argument('--parameter-policy', choices=('qualified', 'discover'), default='qualified')\n"
        "a = p.parse_args()\n"
        "a.output_directory.mkdir(parents=True, exist_ok=True)\n"
        "counter = a.output_directory / 'runs.txt'\n"
        "counter.write_text(str(int(counter.read_text()) + 1 if counter.exists() else 1))\n"
        "(a.output_directory / 'functionals.txt').write_text(';'.join(a.functional or []))\n"
        "(a.output_directory / 'sharing.txt').write_text(str(a.share_parameters))\n"
        "(a.output_directory / 'policy.txt').write_text(a.parameter_policy)\n"
        "names = ['registry.hpp', 'registry.cpp']\n"
        f"if not a.disabled: names += [str(i) + '{'.cu' if backend == 'cuda' else '.cpp'}' for i in range({32 if backend == 'cuda' else 8})]\n"
        "for name in names:\n"
        f"    (a.output_directory / ('generated_libxc_semilocal_{'cuda_' if backend == 'cuda' else ''}' + name)).write_text('// inert probe\\n')\n"
    )
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(AutomaticCudaGraph LANGUAGES NONE)\n"
        "add_custom_target(probe)\nset(target probe)\n"
        f'include("{ROOT / "cmake/GenerativeQCGenerated.cmake"}")\n'
        f'set(Python3_EXECUTABLE "{sys.executable}")\n'
        f"set(GENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT {'ON' if enabled else 'OFF'})\n"
        f'set(GENERATIVEQC_AUTOMATIC_XC_CUDA_FUNCTIONALS "{";".join(functionals)}")\n'
        f"set(GENERATIVEQC_SHARE_AUTOMATIC_XC_CUDA_PARAMETERS {'ON' if share_parameters else 'OFF'})\n"
        "set(GENERATIVEQC_AUTOMATIC_XC_CUDA_PARAMETER_POLICY discover)\n" + declaration
    )
    build = tmp_path / "build"
    subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(build), "-G", "Ninja"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    command = [
        cmake,
        "--build",
        str(build),
        "--target",
        f"generativeqc_libxc_semilocal_{backend}_codegen",
    ]

    def run() -> int:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        return int(
            (build / f"generated/libxc_semilocal_{backend}/runs.txt").read_text()
        )

    assert run() == 1
    assert run() == 1
    assert (
        build / f"generated/libxc_semilocal_{backend}/functionals.txt"
    ).read_text() == ";".join(functionals)
    assert (
        build / f"generated/libxc_semilocal_{backend}/sharing.txt"
    ).read_text() == str(share_parameters)
    assert (build / f"generated/libxc_semilocal_{backend}/policy.txt").read_text() == (
        "discover" if share_parameters else "qualified"
    )
    expected = 1
    identity_only = (
        ("python/generativeqc_compiler/xc/automatic_cuda.py",)
        if backend == "cpu"
        else ()
    )
    for relative in (
        *_SCIENCE_INPUTS,
        *identity_only,
        "python/generativeqc_compiler/common/added.py",
    ):
        path = tmp_path / relative
        path.write_text("# changed scientific or identity input\n")
        expected += int(enabled)
        assert run() == expected, relative
    directory = build / f"generated/libxc_semilocal_{backend}"
    shards = list(directory.glob("*.cu" if backend == "cuda" else "*.cpp"))
    assert len(shards) == (9 if backend == "cpu" else 32 if enabled else 0)


@pytest.mark.parametrize("cuda_enabled", (False, True))
@pytest.mark.parametrize("functional_name", ("GGA_X_APBE", "OFF", "0"))
def test_explicit_package_requires_cuda_and_opt_in(
    tmp_path: Path, cuda_enabled: bool, functional_name: str
) -> None:
    """Exercise the actual option guard without configuring a compiler/GPU."""
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("option guard requires CMake")
    assert cmake is not None
    source = (ROOT / "CMakeLists.txt").read_text()
    start = source.index('set(GENERATIVEQC_AUTOMATIC_XC_CUDA_FUNCTIONALS ""')
    end = source.index("option(GENERATIVEQC_CUDA_AO_RADIAL_REUSE", start)
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(AutomaticCudaOptions LANGUAGES NONE)\n"
        f"set(GENERATIVEQC_ENABLE_CUDA {'ON' if cuda_enabled else 'OFF'})\n"
        f"set(GENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT {'OFF' if cuda_enabled else 'ON'})\n"
        + source[start:end]
    )
    result = subprocess.run(
        [
            cmake,
            "-S",
            str(tmp_path),
            "-B",
            str(tmp_path / "build"),
            f"-DGENERATIVEQC_AUTOMATIC_XC_CUDA_FUNCTIONALS={functional_name}",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode != 0
    assert "requires CUDA and automatic CUDA AOT" in " ".join(result.stderr.split())


@pytest.mark.parametrize("share_parameters", (False, True))
def test_explicit_package_rejects_cuda_detection_fallback(
    tmp_path: Path, share_parameters: bool
) -> None:
    """A requested package cannot quietly disappear if nvcc is unavailable."""
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("option guard requires CMake")
    assert cmake is not None
    source = (ROOT / "CMakeLists.txt").read_text()
    start = source.rindex(
        'if(NOT GENERATIVEQC_AUTOMATIC_XC_CUDA_FUNCTIONALS STREQUAL "" AND'
    )
    end = source.index("# Scoped GenerativeQC-owned GFN2", start)
    functionals = '""' if share_parameters else "GGA_X_APBE"
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(AutomaticCudaFallback LANGUAGES NONE)\n"
        f"set(GENERATIVEQC_AUTOMATIC_XC_CUDA_FUNCTIONALS {functionals})\n"
        f"set(GENERATIVEQC_SHARE_AUTOMATIC_XC_CUDA_PARAMETERS {'ON' if share_parameters else 'OFF'})\n"
        "set(GENERATIVEQC_ENABLE_CUDA OFF)\n" + source[start:end]
    )
    result = subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(tmp_path / "build")],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode != 0
    assert "cannot use the CPU-only compiler fallback" in " ".join(
        result.stderr.split()
    )


@pytest.mark.parametrize(("cuda", "aot"), ((False, True), (True, False), (True, True)))
def test_parameter_sharing_defaults_to_cuda_aot_and_remains_optional(
    tmp_path: Path, cuda: bool, aot: bool
) -> None:
    """Default sharing requires CUDA AOT; explicit OFF retains specialization."""
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("option guard requires CMake")
    assert cmake is not None
    source = (ROOT / "CMakeLists.txt").read_text()
    start = source.index("set(_generativeqc_parameter_sharing_default OFF)")
    end = source.index('set(GENERATIVEQC_AUTOMATIC_XC_CUDA_FUNCTIONALS ""', start)
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\nproject(ParameterSharing LANGUAGES NONE)\n"
        f"set(GENERATIVEQC_ENABLE_CUDA {'ON' if cuda else 'OFF'})\n"
        f"set(GENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT {'ON' if aot else 'OFF'})\n"
        + source[start:end]
        + 'file(WRITE "${CMAKE_BINARY_DIR}/sharing.txt" "${GENERATIVEQC_SHARE_AUTOMATIC_XC_CUDA_PARAMETERS}")\n'
        + 'file(WRITE "${CMAKE_BINARY_DIR}/policy.txt" "${GENERATIVEQC_AUTOMATIC_XC_CUDA_PARAMETER_POLICY}")\n'
    )
    command = [cmake, "-S", str(tmp_path), "-B", str(tmp_path / "build")]
    subprocess.run(command, check=True, capture_output=True, timeout=30)
    assert (tmp_path / "build/sharing.txt").read_text() == (
        "ON" if cuda and aot else "OFF"
    )
    assert (tmp_path / "build/policy.txt").read_text() == "discover"
    subprocess.run(
        [
            *command,
            "-DGENERATIVEQC_SHARE_AUTOMATIC_XC_CUDA_PARAMETERS=OFF",
            "-DGENERATIVEQC_AUTOMATIC_XC_CUDA_PARAMETER_POLICY=qualified",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert (tmp_path / "build/sharing.txt").read_text() == "OFF"
    assert (tmp_path / "build/policy.txt").read_text() == "qualified"
    result = subprocess.run(
        [*command, "-DGENERATIVEQC_SHARE_AUTOMATIC_XC_CUDA_PARAMETERS=ON"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (result.returncode == 0) == (cuda and aot)
    if result.returncode:
        assert "requires CUDA and automatic CUDA AOT" in " ".join(result.stderr.split())
