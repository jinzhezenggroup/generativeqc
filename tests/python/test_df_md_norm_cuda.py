"""Independent high-precision Coulomb derivatives bound the actual CUDA helper."""

from __future__ import annotations

import math
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.integral.df_md_j_cuda import emit_df_md_j_header
from test_md_j_engine_math import _closed_radial, _states

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TEST_DF_MD_J") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires explicit finite Slurm GPU qualification",
)


@pytest.fixture(scope="module")
def norm_probe(tmp_path_factory: Any, required_native_cxx: Any) -> Path:
    """Compile through the verified launcher; this does not use runtime JIT."""
    folder = tmp_path_factory.mktemp("df-md-norm")
    (folder / "generated_df_md_j.cuh").write_text(emit_df_md_j_header())
    compiler = replace(
        required_native_cxx, compiler=str(Path(os.environ["CUDA_PATH"]) / "bin/nvcc")
    )
    return compiler.build_executable(
        [ROOT / "tests/native/df_md_norm_probe.cu"],
        folder / "probe",
        compile_args=(
            "-std=c++20",
            "-O2",
            "-arch=" + os.environ.get("GENERATIVEQC_TEST_CUDA_ARCH", "sm_120"),
            "-I" + str(folder),
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
        ),
        compile_timeout=180,
        link_timeout=60,
    )


@pytest.mark.parametrize("exponent", [1e-20, 0.03, 1.0, 100.0, 1e20])
def test_device_norm_covers_independent_derivative(
    norm_probe: Path, exponent: float
) -> None:
    """All mixed Cartesian Hermites through f/f, using independent radial polynomials."""
    run = subprocess.run(
        [str(norm_probe), str(exponent)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    actual = np.fromstring(run.stdout, sep="\n")
    assert len(actual) == 210
    offset = 0
    for angular in range(7):
        radial, _ = _closed_radial(2 * angular, exponent / 2, (0.0, 0.0, 0.0))
        positions = {state: i for i, state in enumerate(_states(2 * angular))}
        for state in _states(angular):
            square = (
                2
                * math.pi**2.5
                / (exponent**2 * math.sqrt(2 * exponent))
                * (-1) ** sum(state)
                * radial[positions[tuple(2 * n for n in state)]]
            )
            expected = math.sqrt(square)
            assert expected <= actual[offset] <= expected * 1.001
            offset += 1


@pytest.mark.parametrize("exponent", [0.0, -1.0, 1e-200, 1e200, math.inf, math.nan])
def test_device_unrepresentable_norm_retains_source(
    norm_probe: Path, exponent: float
) -> None:
    run = subprocess.run(
        [str(norm_probe), str(exponent)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    values = np.fromstring(run.stdout, sep="\n")
    # Downward products can saturate to DBL_MAX and still produce a valid,
    # very loose upper bound for low orders. The largest order cannot certify
    # a finite norm at these extremes and must retain the source.
    assert math.isinf(values[-1])
    if not exponent > 0 or not math.isfinite(exponent):
        assert np.isinf(values).all()
