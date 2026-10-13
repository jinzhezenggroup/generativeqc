"""Changing the compiler catalog refreshes CMake's actual selected inventory."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import venv
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def inventory_python(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Keep an editable install's redirect finder out of the copied checkout."""
    directory = tmp_path_factory.mktemp("inventory-python")
    builder = venv.EnvBuilder(with_pip=False)
    builder.create(directory)
    return Path(builder.ensure_directories(directory).env_exec_cmd)


@pytest.mark.parametrize("backend", ["Ninja", "Unix Makefiles"])
@pytest.mark.parametrize("selection", ["auto", "pbe_rks;pbe_rks", ""])
def test_build_refreshes_compiler_profile_inventory(
    tmp_path: Path, inventory_python: Path, backend: str, selection: str
) -> None:
    cmake = shutil.which("cmake")
    builder = "ninja" if backend == "Ninja" else "make"
    if cmake is None or shutil.which(builder) is None:
        pytest.skip(f"CMake and {builder} are required for inventory evaluation")

    # Exercise the real compiler/catalog and policy in isolation. No CUDA
    # compiler, native runtime, or handwritten substitute inventory is used.
    source = tmp_path / "source"
    source.mkdir()
    shutil.copytree(
        ROOT / "python/generativeqc_compiler",
        source / "python/generativeqc_compiler",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for relative in (
        "cmake/GenerativeQCStationaryProfiles.cmake",
        "tools/__init__.py",
        "tools/generate_native_stationary_cuda.py",
        "tools/generate_df_kernels.py",
        "tools/run_codegen.py",
    ):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(stationary_inventory NONE)\n"
        f'set(Python3_EXECUTABLE "{inventory_python.as_posix()}")\n'
        # Isolate discovery, not the caller's CMake launcher (which can itself
        # need PYTHONPATH to load a pip-installed CMake wrapper).
        f'set(ENV{{PYTHONPATH}} "{Path(np.__file__).resolve().parent.parent.as_posix()}")\n'
        'include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerativeQCStationaryProfiles.cmake")\n'
        # Match the source-identity membership watcher: it does not detect
        # content edits in existing files and cannot replace discovery edges.
        "file(GLOB_RECURSE identity_inputs CONFIGURE_DEPENDS\n"
        '  "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/*.py")\n'
        "generativeqc_select_stationary_aot_profiles(selected)\n"
        'file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/selected.txt"\n'
        '  CONTENT "${selected}")\n'
        "add_custom_target(inventory ALL)\n"
    )
    build = tmp_path / "build"
    # The clean interpreter reuses NumPy's installed files without executing the
    # parent environment's .pth hooks, which can redirect compiler imports away
    # from this fixture even when the generator prepends its source directory.
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }

    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [cmake, *arguments],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=60,
        )

    configured = run(
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        backend,
        f"-DGENERATIVEQC_STATIONARY_AOT_PROFILES={selection}",
    )
    assert configured.returncode == 0, configured.stdout + configured.stderr
    dependencies = {
        Path(line)
        for line in (build / "CMakeFiles/generativeqc_stationary_profile_inputs.txt")
        .read_text()
        .splitlines()
    }
    assert all(path.is_relative_to(source) for path in dependencies)
    assert (
        source / "python/generativeqc_compiler/method/stationary_cuda.py"
        in dependencies
    )
    assert (
        source / "python/generativeqc_compiler/xc/libxc_bulk_catalog.json"
        in dependencies
    )
    original = (build / "selected.txt").read_text()
    if selection == "auto":
        assert "pbe_rks" in original.split(";")
    else:
        assert original == ("pbe_rks" if selection else "")

    catalog = source / "python/generativeqc_compiler/method/stationary_cuda.py"
    with catalog.open("a") as stream:
        stream.write(
            "\nQUALIFIED_STATIONARY_AOT_PROFILES += (\n"
            '    StationaryAotProfile("probe", 1, "PBE", "unpolarized"),\n'
            ')\nQUALIFIED_STATIONARY_AOT_PROFILE_NAMES += ("probe_rks",)\n'
        )
    rebuilt = run("--build", str(build))
    assert rebuilt.returncode == 0, rebuilt.stdout + rebuilt.stderr
    selected = (build / "selected.txt").read_text()
    if selection == "auto":
        assert set(selected.split(";")) == {*original.split(";"), "probe_rks"}
    else:
        assert selected == original

    # Removing a formerly qualified graph must remove auto targets and reject
    # an explicit stale selection instead of leaving the previous graph alive.
    with catalog.open("a") as stream:
        stream.write(
            "\nQUALIFIED_STATIONARY_AOT_PROFILES = tuple(\n"
            '    p for p in QUALIFIED_STATIONARY_AOT_PROFILES if p.name != "pbe"\n'
            ")\nQUALIFIED_STATIONARY_AOT_PROFILE_NAMES = tuple(\n"
            "    n for n in QUALIFIED_STATIONARY_AOT_PROFILE_NAMES\n"
            '    if n not in ("pbe_rks", "pbe_uks")\n)\n'
        )
    rebuilt = run("--build", str(build))
    if selection and selection != "auto":
        assert rebuilt.returncode != 0
        assert "Unknown compiler-qualified stationary CUDA AOT profile: pbe_rks" in (
            rebuilt.stdout + rebuilt.stderr
        )
    else:
        assert rebuilt.returncode == 0, rebuilt.stdout + rebuilt.stderr
        expected = (
            set(selected.split(";")) - {"pbe_rks", "pbe_uks"}
            if selection == "auto"
            else {""}
        )
        assert set((build / "selected.txt").read_text().split(";")) == expected

        # Import-time XC catalog validation is a discovery prerequisite too.
        # A stale configured inventory must not hide incompatible metadata.
        metadata = source / "python/generativeqc_compiler/xc/libxc_bulk_catalog.json"
        valid = metadata.read_text()
        incompatible = json.loads(valid)
        incompatible["schema_version"] += 1
        metadata.write_text(json.dumps(incompatible))
        rebuilt = run("--build", str(build))
        assert rebuilt.returncode != 0
        assert "bulk catalog needs regeneration for this importer" in (
            rebuilt.stdout + rebuilt.stderr
        )
        metadata.write_text(valid)
        restored = run("--build", str(build))
        assert restored.returncode == 0, restored.stdout + restored.stderr
        assert set((build / "selected.txt").read_text().split(";")) == expected
