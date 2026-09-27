"""Ownership tests for compiler-generated Direct-Fock scatter math."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from vibeqc_compiler.integral.lowering.fock_accumulation import (
    emit_direct_fock_accumulation_header,
    emit_direct_force_density_coefficient,
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
    assert 'contribution_name = f"{function_name}_contribution"' in shared
    assert "static_cast<float>(density_value) * static_cast<float>(integral)" in shared
    assert "return scale * density_value * static_cast<double>(integral);" in shared
    assert (
        "static_cast<float>(density_value) * static_cast<float>(integral)" not in native
    )
    assert "return scale * density_value * static_cast<double>(integral);" not in native
    assert (
        "static_cast<float>(density_value) * static_cast<float>(integral)"
        not in shell_lowering
    )
    assert '#include "generated_direct_fock_accumulation.cuh"' in native


def test_direct_force_density_has_one_compiler_equation_owner() -> None:
    """Keep the exact RHF/UHF force density contraction out of native CUDA."""

    generated = emit_direct_force_density_coefficient()
    native = (REPOSITORY_ROOT / "src/scf/cuda/direct_force_density.cuh").read_text(
        encoding="utf-8"
    )
    for equation in (
        "0.5 * coulomb_coefficient * total_ab * total_cd",
        "0.5 * exchange_coefficient *",
        "unique_eri_symmetry_permutation",
    ):
        assert equation in generated
        assert equation not in native
    assert '#include "generated_direct_fock_accumulation.cuh"' in native
    assert "direct_force_density_coefficient" in emit_direct_fock_accumulation_header()


def test_direct_force_density_exposes_method_neutral_coefficients() -> None:
    """Let DFT reuse Direct force contraction without changing HF defaults."""

    generated = emit_direct_force_density_coefficient()
    assert "direct_force_density_coefficient_scaled" in generated
    assert "double coulomb_coefficient, double exchange_coefficient" in generated
    assert "if (coulomb_coefficient != 0.0)" in generated
    assert "if (exchange_coefficient != 0.0)" in generated
    assert "constexpr double exchange_coefficient = Unrestricted ? -1.0 : -0.5;" in generated
    assert (
        "n, physical_offset, spin_offset, density, i, j, k, l, 1.0, "
        "exchange_coefficient"
        in generated
    )


def test_generated_shell_and_native_scatter_share_spin_semantics() -> None:
    """Render both adapters from the same compiler-owned contraction."""

    native = emit_direct_fock_accumulation_header()
    generated = emit_generated_shell_fock_accumulation()
    for equation in (
        "const double total_cd = alpha_cd + beta_cd;",
        "<MixedProduct>(j_scale, total_cd, integral)",
        "<MixedProduct>(k_scale, alpha_bd, integral)",
        "<MixedProduct>(k_scale, beta_bd, integral)",
        "<MixedProduct>(k_scale, density_bd, integral)",
        "static_cast<float>(density_value) * static_cast<float>(integral)",
        "return scale * density_value * static_cast<double>(integral);",
    ):
        assert equation in native
        assert equation in generated
    assert (
        "template <bool Unrestricted, bool MixedProduct = false, typename Integral = double>"
        in native
    )
    assert (
        "template <bool Unrestricted, bool MixedProduct = false, typename Integral = double>"
        in generated
    )
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
