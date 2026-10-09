"""The fixed GCC closure manifest must be path-independent and byte-sensitive."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
