"""The fixed GCC closure manifest must be path-independent and byte-sensitive."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "rank_k_host_manifest", ROOT / "tools/generate_rank_k_host_toolchain_manifest.py"
)
assert SPEC is not None and SPEC.loader is not None
manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest)


def _fake_toolchain(root: Path) -> Path:
    compiler = root / "bin/g++"
    for path in (
        compiler,
        *(root / f"bin/{name}" for name in manifest.PROGRAMS),
        *(root / f"lib/{name}" for name in manifest.LINK_INPUTS),
        root / "lib/liblto_plugin.so",
        root / "include/vector",
        root / "include/detail/config.hpp",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name)
    return compiler


def _fake_run(
    arguments: list[str], *, stdin: str | None = None
) -> subprocess.CompletedProcess:
    del stdin
    executable = Path(arguments[0])
    root = executable.parents[1]
    option = arguments[1]
    stdout, stderr = "", ""
    if option.startswith("-print-prog-name="):
        stdout = str(root / "bin" / option.split("=", 1)[1]) + "\n"
    elif option.startswith("-print-file-name="):
        name = option.split("=", 1)[1]
        candidate = root / "lib" / name
        stdout = str(candidate) + "\n" if candidate.is_file() else name + "\n"
    elif option == "-E":
        stderr = (
            "#include <...> search starts here:\n"
            f" {root / 'include'}\n"
            "End of search list.\n"
        )
    elif option == "-dumpspecs":
        stdout = "fixed specs\n"
    elif option == "-dumpmachine":
        stdout = "x86_64-linux-gnu\n"
    elif option == "-dumpfullversion":
        stdout = "11.4.0\n"
    elif option == "--verbose" and executable.name == "ld":
        stdout = "fixed linker script\n"
    else:
        raise AssertionError(arguments)
    return subprocess.CompletedProcess(arguments, 0, stdout, stderr)


def test_manifest_is_relocation_independent_and_byte_sensitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(manifest, "_run", _fake_run)
    monkeypatch.setattr(manifest, "_ldd", lambda path: ())
    compiler = _fake_toolchain(tmp_path / "first")
    before = manifest.inventory(compiler)
    assert before["schema"] == manifest.SCHEMA
    roles = {entry["role"] for entry in before["entries"]}
    assert {
        "program:driver",
        "program:cc1plus",
        "program:as",
        "program:collect2",
        "program:ld",
        "header:0:vector",
        "link-input:libstdc++.so",
        "config:gcc-specs",
        "config:ld-default-script",
    }.issubset(roles)

    relocated = tmp_path / "relocated"
    shutil.copytree(tmp_path / "first", relocated)
    assert manifest.inventory(relocated / "bin/g++") == before

    (relocated / "include/vector").write_text("changed header")
    assert manifest.inventory(relocated / "bin/g++") != before


def test_linker_plugin_wrapper_and_dependencies_are_identity_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "host"
    compiler = _fake_toolchain(root)
    dependency = root / "lib/plugin-runtime.so"
    dependency.write_text("plugin runtime")

    monkeypatch.setattr(manifest, "_run", _fake_run)
    monkeypatch.setattr(
        manifest,
        "_ldd",
        lambda path: (dependency,) if path.name == "liblto_plugin.so" else (),
    )
    before = manifest.inventory(compiler)
    roles = {entry["role"] for entry in before["entries"]}
    assert "program:lto-wrapper" in roles
    assert "linker-plugin:liblto_plugin.so" in roles
    assert "dependency:linker-plugin:liblto_plugin.so:plugin-runtime.so" in roles

    plugin = root / "lib/liblto_plugin.so"
    plugin.write_text("changed plugin")
    assert manifest.inventory(compiler) != before
    plugin.write_text("liblto_plugin.so")

    wrapper = root / "bin/lto-wrapper"
    wrapper.write_text("changed wrapper")
    assert manifest.inventory(compiler) != before
    wrapper.write_text("lto-wrapper")

    dependency.write_text("changed plugin runtime")
    assert manifest.inventory(compiler) != before


@pytest.mark.parametrize("missing", ["liblto_plugin.so", "lto-wrapper"])
def test_missing_linker_plugin_inputs_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    compiler = _fake_toolchain(tmp_path / "host")
    monkeypatch.setattr(manifest, "_run", _fake_run)
    monkeypatch.setattr(manifest, "_ldd", lambda path: ())
    location = "lib" if missing.endswith(".so") else "bin"
    (compiler.parents[1] / location / missing).unlink()
    with pytest.raises(FileNotFoundError, match="missing GCC host"):
        manifest.inventory(compiler)


@pytest.mark.parametrize("name", ["as", "ld"])
def test_bare_program_name_uses_path_not_adjacent_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    compiler = _fake_toolchain(tmp_path / "host")
    selected = tmp_path / "selected-bin" / name
    selected.parent.mkdir()
    selected.write_text("PATH-selected tool")

    def bare(
        arguments: list[str], *, stdin: str | None = None
    ) -> subprocess.CompletedProcess:
        del stdin
        assert arguments == [str(compiler), f"-print-prog-name={name}"]
        return subprocess.CompletedProcess(arguments, 0, name + "\n", "")

    monkeypatch.setattr(manifest, "_run", bare)
    monkeypatch.setattr(
        manifest.shutil, "which", lambda value: str(selected) if value == name else None
    )
    assert manifest._program(compiler, name) == selected.resolve()
    selected.unlink()
    with pytest.raises(FileNotFoundError, match="missing GCC host program"):
        manifest._program(compiler, name)
