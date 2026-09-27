from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from tools import generate_libxc_public_cpu_registry as registry

if TYPE_CHECKING:
    import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_empty_public_registry_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "PUBLIC_EVIDENCE", {})
    rendered = registry.render_registry()
    assert "namespace vibeqc::dft::bulk_public {" in rendered
    assert "SemilocalPointProgram* find(std::string_view name) noexcept" in rendered
    assert "return nullptr;" in rendered
    assert "point_program()" not in rendered


def test_registry_import_preserves_compiler_packages() -> None:
    """Importing a renderer must not replace package objects or their lazy exports."""
    code = """
import importlib
import sys
from pathlib import Path

root = Path.cwd()
sys.path[:0] = [str(root), str(root / "python")]
names = (
    "vibeqc_compiler",
    "vibeqc_compiler.common",
    "vibeqc_compiler.integral",
    "vibeqc_compiler.xc",
    "vibeqc_compiler.dft",
    "vibeqc_compiler.method",
)
before = {name: importlib.import_module(name) for name in names}
from vibeqc_compiler.method import MethodIR
importlib.import_module("tools.generate_libxc_public_cpu_registry")
for name, package in before.items():
    assert sys.modules[name] is package, name
from vibeqc_compiler.method import MethodIR as after
assert after is MethodIR
"""
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
