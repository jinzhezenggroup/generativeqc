"""MethodIR-driven native stationary graph packaging: no second method list."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "tools/generate_native_stationary_cuda.py"


def generate(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GENERATOR), *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def qualified(*arguments: str) -> tuple[str, ...]:
    result = generate(*arguments)
    assert result.returncode == 0, result.stderr
    return tuple(result.stdout.strip().split(";"))


def test_inventory_is_generated_from_compiler_method_ir() -> None:
    native = qualified("--list-profiles")
    aot = qualified("--list-aot-profiles")
    assert native and len(native) == len(set(native))
    assert set(native) <= set(aot)
    assert native == qualified("--list-profiles")
    cmake = (ROOT / "cmake/GenerativeQCStationaryProfiles.cmake").read_text()
    assert "--list-aot-profiles" in cmake
    assert "GENERATIVEQC_STATIONARY_AOT_PROFILES" in cmake
    assert "lda_rks lda_uks" not in cmake


def test_single_graph_generation_matches_independent_codegen(tmp_path: Path) -> None:
    names = qualified("--list-profiles")
    assert len(names) >= 2
    selected = (names[0], names[-1])
    output = tmp_path / "group"
    actual = generate(
        "--all-profiles", "--output-dir", str(output), "--profiles", *selected
    )
    assert actual.returncode == 0, actual.stderr
    snapshots: dict[str, tuple[bytes, int]] = {}
    for name in selected:
        candidate = output / f"native_{name}.cu"
        assert candidate.is_file()
        old = tmp_path / f"single_{name}.cu"
        separate = generate("--output", str(old), "--profile", name)
        assert separate.returncode == 0, separate.stderr
        assert candidate.read_bytes() == old.read_bytes()
        snapshots[name] = (candidate.read_bytes(), candidate.stat().st_mtime_ns)

    second = generate(
        "--all-profiles", "--output-dir", str(output), "--profiles", *selected
    )
    assert second.returncode == 0, second.stderr
    for name in selected:
        candidate = output / f"native_{name}.cu"
        assert (candidate.read_bytes(), candidate.stat().st_mtime_ns) == snapshots[name]


@pytest.mark.parametrize("profiles", [("invalid_profile",), ("pbe_rks", "pbe_rks")])
def test_invalid_or_duplicate_compiler_graph_selection_is_rejected(
    tmp_path: Path, profiles: tuple[str, ...]
) -> None:
    rejected = generate(
        "--all-profiles", "--output-dir", str(tmp_path), "--profiles", *profiles
    )
    assert rejected.returncode != 0
    assert not tuple(tmp_path.glob("*.cu"))


def test_prepared_cpp_admits_graph_without_label_round_trip() -> None:
    source = (ROOT / "src/methods/dft_method.cpp").read_text()
    controller = (ROOT / "src/dft/cuda_stationary_controller.cpp").read_text()
    assert "const dft::CudaStationaryBinding* native_cuda_stationary_binding(" in source
    assert "dft::cuda_stationary_binding(model)" in source
    assert "cuda_stationary_binding(profile)" not in source
    assert "binding_(&binding)" in controller
    cmake = (ROOT / "cmake/GenerativeQCCuda.cmake").read_text()
    assert "NAME generativeqc_native_stationary_codegen" in cmake
    assert (
        'NAME "generativeqc_native_stationary_${_native_profile}_codegen"' not in cmake
    )
