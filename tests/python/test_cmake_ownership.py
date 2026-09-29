"""Structural build tests for modular CMake ownership."""

import json
import subprocess
import sys
import typing
from pathlib import Path

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
