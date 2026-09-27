"""Ownership tests for compiler-generated Direct-Fock scatter math."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from vibeqc_compiler.integral.lowering.fock_accumulation import (
    emit_direct_fock_accumulation_header,
    emit_generated_shell_fock_accumulation,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_direct_fock_scatter_has_one_compiler_equation_owner() -> None:
    """Keep RHF/UHF coefficients out of retained handwritten CUDA."""

    shared = (
        REPOSITORY_ROOT
        / "python/vibeqc_compiler/integral/lowering/fock_accumulation.py"
    ).read_text(encoding="utf-8")
    native = (REPOSITORY_ROOT / "src/scf/cuda/direct_fock_accumulation.cuh").read_text(
        encoding="utf-8"
    )
    shell_lowering = (
        REPOSITORY_ROOT / "python/vibeqc_compiler/integral/lowering/fock.py"
    ).read_text(encoding="utf-8")

    assert "restricted_exchange_scale" in shared
    assert "unrestricted_exchange_scale" in shared
    assert "j_scale * total_cd * integral" in shared
    assert "k_scale * density_bd * integral" in shared
    assert "j_scale * total_cd * integral" not in native
    assert "k_scale * density_bd * integral" not in native
    assert "j_scale * total_cd * integral" not in shell_lowering
    assert "k_scale * density_bd * integral" not in shell_lowering
    assert '#include "generated_direct_fock_accumulation.cuh"' in native


def test_generated_shell_and_native_scatter_share_spin_semantics() -> None:
    """Render both adapters from the same compiler-owned contraction."""

    native = emit_direct_fock_accumulation_header()
    generated = emit_generated_shell_fock_accumulation()
    for equation in (
        "const double total_cd = alpha_cd + beta_cd;",
        "j_scale * total_cd * integral",
        "k_scale * alpha_bd * integral",
        "k_scale * beta_bd * integral",
        "k_scale * density_bd * integral",
    ):
        assert equation in native
        assert equation in generated
    for coefficient in (
        "exchange_only ? 1.0 : -0.5",
        "exchange_only ? 1.0 : -1.0",
    ):
        assert coefficient in native
    assert "? 1.0 : -0.5" in generated
    assert "? 1.0 : -1.0" in generated


def test_direct_fock_scatter_cli_is_deterministic(tmp_path: Path) -> None:
    """Make the build-time compatibility header reproducible from checkout."""

    output = tmp_path / "generated_direct_fock_accumulation.cuh"
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "tools/generate_shell_kernels.py"),
        "--direct-fock-accumulation-output",
        str(output),
    ]
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
    first = output.read_text(encoding="utf-8")
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
    assert output.read_text(encoding="utf-8") == first
    assert first == emit_direct_fock_accumulation_header()
