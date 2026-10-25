"""Compile the real KS adapter with bounded host resource/provider doubles.

This checks owner admission and cleanup, not GPU or molecular numerics.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/native/fixtures/one_electron_budget"


def test_one_electron_owner_reserves_mandatory_charges(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Optional arena capacity cannot make an affordable owner unaffordable."""
    for name in (
        "cuda_runtime_api.h",
        "dft/cuda_ks_final_state.hpp",
        "dft/cuda_ks_grid_reference.hpp",
        "dft/cuda_ks_resident_fock.hpp",
        "methods/dft_method.hpp",
        "scf/fock_prepared.hpp",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('#pragma once\n#include "providers.hpp"\n')
    executable = native_cxx.build_executable(
        [FIXTURE / "probe.cpp"],
        tmp_path / "probe",
        compile_args=(
            "-std=c++20",
            "-O0",
            f"-I{tmp_path}",
            f"-I{FIXTURE}",
            f"-I{ROOT / 'src'}",
        ),
    )
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS mandatory charge reservation" in result.stdout
