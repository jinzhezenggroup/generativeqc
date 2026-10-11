"""Installed console scripts delegate runtime ownership without PATH recursion."""

import typing
from pathlib import Path

import pytest
from generativeqc import _native_cli


def test_native_basis_header_is_self_contained(
    native_cxx: typing.Any, tmp_path: Path
) -> None:
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "native_basis_header.cpp"
    source.write_text(
        '#include "cli/native_basis.hpp"\n'
        "static_assert(sizeof(std::int32_t) == 4);\n"
        "generativeqc::cli::NativeBasisData basis;\n"
    )
    native_cxx.compile_object(
        source,
        tmp_path / "native_basis_header.o",
        args=("-std=c++20", "-I", str(root / "src"), "-I", str(root / "include")),
    )


def test_native_json_is_a_build_dependency_not_a_runtime_python_dependency() -> None:
    tomllib = pytest.importorskip("tomllib")
    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    assert "nlohmann_json==3.11.3" in config["build-system"]["requires"]
    assert not any(
        requirement.startswith("nlohmann")
        for requirement in config["project"]["dependencies"]
    )


def test_packaged_runtime_dispatch_preserves_arguments_and_status(
    monkeypatch: typing.Any, tmp_path: typing.Any
) -> None:
    package = tmp_path / "generativeqc"
    binary = package / "bin/generativeqc"
    binary.parent.mkdir(parents=True)
    binary.write_text("fixture")
    binary.chmod(0o755)
    monkeypatch.setattr(_native_cli, "__file__", str(package / "_native_cli.py"))
    monkeypatch.delenv("GENERATIVEQC_NATIVE_CLI", raising=False)
    calls = []

    def invoke(arguments: list[str]) -> int:
        calls.append(arguments)
        return 2

    monkeypatch.setattr(_native_cli.subprocess, "call", invoke)
    arguments = ["resources", "Input With Spaces.xyz", "--host-bytes", "0"]
    assert _native_cli.dispatch(arguments) == 2
    assert calls == [[str(binary), *arguments]]


def test_compiler_search_never_recurses_into_native_cli(
    monkeypatch: typing.Any,
) -> None:
    monkeypatch.setenv("GENERATIVEQC_NATIVE_CLI", "/invalid/override")
    assert _native_cli.dispatch(["autotune", "input.xyz", "--quick"]) is None


def test_libxc_discovery_is_native_when_an_executable_is_configured(
    monkeypatch: typing.Any, tmp_path: typing.Any
) -> None:
    binary = tmp_path / "native"
    binary.write_text("fixture")
    binary.chmod(0o755)
    monkeypatch.setenv("GENERATIVEQC_NATIVE_CLI", str(binary))
    calls = []
    monkeypatch.setattr(
        _native_cli.subprocess, "call", lambda arguments: calls.append(arguments) or 0
    )
    assert _native_cli.dispatch(["methods", "--libxc", "--json"]) == 0
    assert calls == [[str(binary), "methods", "--libxc", "--json"]]


def test_invalid_override_is_not_silently_replaced(monkeypatch: typing.Any) -> None:
    monkeypatch.setenv("GENERATIVEQC_NATIVE_CLI", "/invalid/override")
    with pytest.raises(RuntimeError, match="native CLI was not found"):
        _native_cli.dispatch(["profile", "show"])


def test_relative_override_never_searches_path(
    monkeypatch: typing.Any, tmp_path: Path
) -> None:
    binary = tmp_path / "generativeqc"
    binary.write_text("fixture")
    binary.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GENERATIVEQC_NATIVE_CLI", "./generativeqc")
    calls = []
    monkeypatch.setattr(
        _native_cli.subprocess, "call", lambda arguments: calls.append(arguments) or 0
    )
    assert _native_cli.dispatch(["profile", "show"]) == 0
    assert calls == [[str(binary), "profile", "show"]]


def test_source_only_discovery_remains_available(
    monkeypatch: typing.Any, tmp_path: typing.Any
) -> None:
    monkeypatch.setattr(_native_cli, "__file__", str(tmp_path / "_native_cli.py"))
    monkeypatch.delenv("GENERATIVEQC_NATIVE_CLI", raising=False)
    assert _native_cli.dispatch(["methods"]) is None
    assert _native_cli.dispatch(["profile", "show"]) is None
