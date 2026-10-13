"""Regression tests for the native core / C API dependency boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from tools.check_native_api_boundaries import audit_native_api_boundaries


def test_current_native_abi_boundaries() -> None:
    assert audit_native_api_boundaries() == []


def test_core_cannot_import_public_c_abi(tmp_path: Path) -> None:
    core = tmp_path / "src/core"
    methods = tmp_path / "src/methods"
    core.mkdir(parents=True)
    methods.mkdir()
    (core / "types.hpp").write_text('#include "generativeqc/generativeqc.h"\n')
    assert "public C ABI" in audit_native_api_boundaries(tmp_path)[0]


def test_methods_cannot_import_handles_or_dereference_them(tmp_path: Path) -> None:
    core = tmp_path / "src/core"
    methods = tmp_path / "src/methods"
    core.mkdir(parents=True)
    methods.mkdir()
    (core / "types.hpp").write_text("#include <cstdint>\n")
    (methods / "bad.cpp").write_text(
        '#include "api/handles.hpp"\n'
        "auto system = descriptor.density_fitting_auxiliary_basis->data;\n"
    )
    errors = audit_native_api_boundaries(tmp_path)
    assert len(errors) == 2
    assert "must not include API handles" in errors[0]
    assert "native auxiliary system" in errors[1]


def test_comments_do_not_trigger_include_guard(tmp_path: Path) -> None:
    core = tmp_path / "src/core"
    methods = tmp_path / "src/methods"
    core.mkdir(parents=True)
    methods.mkdir()
    (core / "types.hpp").write_text("#include <cstdint>\n")
    (methods / "clean.hpp").write_text('// #include "api/handles.hpp"\n')
    assert audit_native_api_boundaries(tmp_path) == []
