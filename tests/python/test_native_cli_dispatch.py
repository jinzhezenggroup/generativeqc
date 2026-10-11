"""Native wheel CLI entrypoint is a C++ launcher, not a Python dispatch shim."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_native_basis_header_is_self_contained(
    native_cxx: object, tmp_path: Path
) -> None:
    source = tmp_path / "native_basis_header.cpp"
    source.write_text(
        '#include "cli/native_basis.hpp"\n'
        "static_assert(sizeof(std::int32_t) == 4);\n"
        "generativeqc::cli::NativeBasisData basis;\n"
    )
    native_cxx.compile_object(
        source,
        tmp_path / "native_basis_header.o",
        args=("-std=c++20", "-I", str(ROOT / "src"), "-I", str(ROOT / "include")),
    )


def test_native_json_is_build_only_not_runtime_python_dependency() -> None:
    tomllib = pytest.importorskip("tomllib")
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "nlohmann_json==3.11.3" in config["build-system"]["requires"]
    assert not any(
        requirement.startswith("nlohmann")
        for requirement in config["project"]["dependencies"]
    )


def test_only_python_entrypoint_is_explicit_tuner() -> None:
    tomllib = pytest.importorskip("tomllib")
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    scripts = config["project"]["scripts"]
    assert "generativeqc" not in scripts
    assert scripts["generativeqc-tune"] == "generativeqc.tune_cli:main"
    assert not (ROOT / "python/generativeqc/__main__.py").exists()
    assert not (ROOT / "python/generativeqc/_native_cli.py").exists()
    cmake = (ROOT / "CMakeLists.txt").read_text()
    assert "SKBUILD_SCRIPTS_DIR" in cmake
    assert "generativeqc_wheel_launcher" in cmake


@pytest.fixture(scope="module")
def compiled_launcher(
    native_cxx: object, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    if os.name != "posix":
        pytest.skip("native wheel entrypoint is a POSIX executable")
    output = tmp_path_factory.mktemp("native-wheel-launcher") / "generativeqc"
    native_cxx.build_executable(
        [ROOT / "src/cli/wheel_launcher.cpp"],
        output,
        compile_args=(
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
        ),
        compile_timeout=90,
    )
    return output


def _install_fake_wheel(
    root: Path, compiled_launcher: Path, version: str = "python3.11"
) -> Path:
    launcher = root / "bin/generativeqc"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(compiled_launcher, launcher)
    launcher.chmod(0o755)
    native = root / "lib" / version / "site-packages/generativeqc/bin/generativeqc"
    native.parent.mkdir(parents=True)
    native.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\nexit 23\n')
    native.chmod(0o755)
    return launcher


def test_wheel_launcher_executes_only_adjacent_native_binary(
    compiled_launcher: Path, tmp_path: Path
) -> None:
    launcher = _install_fake_wheel(tmp_path / "relocated prefix", compiled_launcher)
    # No Python, package import, PATH lookup or implicit scientific fallback.
    run = subprocess.run(
        [str(launcher), "resources", "Input With Spaces.xyz", "--host-bytes", "0"],
        env={"PATH": ""},
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert run.returncode == 23
    assert run.stdout.splitlines() == [
        "resources",
        "Input With Spaces.xyz",
        "--host-bytes",
        "0",
    ]


def test_wheel_launcher_fails_closed_without_packaged_binary(
    compiled_launcher: Path, tmp_path: Path
) -> None:
    launcher = _install_fake_wheel(tmp_path, compiled_launcher)
    (tmp_path / "lib/python3.11/site-packages/generativeqc/bin/generativeqc").unlink()
    result = subprocess.run(
        [str(launcher), "--version"], text=True, capture_output=True, check=False
    )
    assert result.returncode == 127
    assert "expected exactly one packaged native CLI" in result.stderr


def test_wheel_launcher_rejects_ambiguous_interpreter_installations(
    compiled_launcher: Path, tmp_path: Path
) -> None:
    launcher = _install_fake_wheel(tmp_path, compiled_launcher)
    _install_fake_wheel(tmp_path, compiled_launcher, "python3.12")
    result = subprocess.run(
        [str(launcher), "methods"], text=True, capture_output=True, check=False
    )
    assert result.returncode == 127
    assert "found 2" in result.stderr


def test_wheel_launcher_relocates_target_installation(
    compiled_launcher: Path, tmp_path: Path
) -> None:
    original = tmp_path / "original target"
    _install_fake_wheel(original, compiled_launcher)
    packaged = original / "generativeqc/bin/generativeqc"
    packaged.parent.mkdir(parents=True)
    (original / "lib/python3.11/site-packages/generativeqc/bin/generativeqc").rename(
        packaged
    )
    relocated = tmp_path / "relocated target"
    original.rename(relocated)
    result = subprocess.run(
        [str(relocated / "bin/generativeqc"), "resources", "Input With Spaces.xyz"],
        env={"PATH": ""},
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 23
    assert result.stdout.splitlines() == ["resources", "Input With Spaces.xyz"]


@pytest.mark.parametrize("alias", [False, True])
def test_wheel_launcher_checks_target_and_site_packages_together(
    compiled_launcher: Path, tmp_path: Path, alias: bool
) -> None:
    launcher = _install_fake_wheel(tmp_path, compiled_launcher)
    package = tmp_path / "lib/python3.11/site-packages/generativeqc"
    target_package = tmp_path / "generativeqc"
    if alias:
        target_package.symlink_to(package, target_is_directory=True)
    else:
        shutil.copytree(package, target_package)
    result = subprocess.run(
        [str(launcher), "methods"],
        env={"PATH": ""},
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if alias:
        assert result.returncode == 23
        assert result.stdout.splitlines() == ["methods"]
    else:
        assert result.returncode == 127
        assert "found 2" in result.stderr
