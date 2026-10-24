"""Exercise synchronous dispatch and FP64 atomic contracts without a GPU.

The CUDA double validates retry/NaN semantics, not device atomic ordering.
Real CUDA compilation and Slurm execution remain separate qualification gates.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "architecture,forced", [(500, False), (600, False), (1200, True)]
)
def test_shared_runtime_contracts(
    tmp_path: Path, native_cxx: NativeCxx, architecture: int, forced: bool
) -> None:
    executable = native_cxx.build_executable(
        [ROOT / "tests/native/test_shared_runtime_primitives.cpp"],
        tmp_path / "probe",
        compile_args=[
            "-std=c++20",
            "-fno-strict-aliasing",
            f"-D__CUDA_ARCH__={architecture}",
            *(["-DGENERATIVEQC_FORCE_FP64_ATOMIC_CAS"] if forced else []),
            "-I" + str(ROOT / "tests/native/fixtures/cuda_atomic"),
            "-I" + str(ROOT / "src"),
        ],
    )
    subprocess.run([str(executable)], check=True, timeout=10)
