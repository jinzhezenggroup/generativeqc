"""Method-free prepared spectra: real owner/provider, no generated or xtb inputs."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def prepared_spectral_probe(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> Path:
    folder = tmp_path_factory.mktemp("prepared-spectral-owner")
    # Deliberately neither generate method headers nor provide method include paths.
    # Linking the real owner and LP64 capability factory is part of this gate.
    return required_native_cxx.build_executable(
        [
            ROOT / "tests/native/test_prepared_spectral_owner.cpp",
            ROOT / "src/solver/cpu/prepared_spectral.cpp",
            ROOT / "src/tensor/cpu/lp64_provider.cpp",
        ],
        folder / "probe",
        compile_args=[
            "-std=c++17",
            "-O2",
            "-ffunction-sections",
            "-fdata-sections",
            "-I" + str(ROOT / "src"),
        ],
        link_args=["-Wl,--gc-sections", "-ldl", "-pthread"],
    )


@pytest.mark.parametrize(
    "case",
    [
        "plan",
        "binding",
        "factor_admission",
        "solve_admission",
        "numerics",
        "failures",
        "token_admission",
        "token_execution",
    ],
)
def test_prepared_spectral_owner_contract(
    prepared_spectral_probe: Path, case: str
) -> None:
    result = subprocess.run(
        [str(prepared_spectral_probe), case],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, (case, result.stdout, result.stderr)
    assert f"{case} passed" in result.stdout
