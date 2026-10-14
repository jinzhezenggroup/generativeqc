"""Scheduled exact-PBE0 force qualification at an unchanged FP64 SCF state."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, _native

from benchmarks._cases import benchmark_cases
from benchmarks.pbe0_mixed_gradient import GradientProbe, options
from benchmarks.pbe0_precision import reference

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires finite srun on a real GPU",
)


@pytest.fixture(scope="module")
def probe(native_cxx: Any, tmp_path_factory: Any) -> GradientProbe:
    """Link against the source-matched library rather than reconstruct its state."""
    library = Path(_native.load_library()._name).resolve()
    folder = tmp_path_factory.mktemp("mixed-gradient-endpoint")
    target = folder / "probe.so"
    cuda = Path(os.environ.get("CUDA_HOME", "/group/software/cuda-12.9.1"))
    native_cxx.build_shared(
        [ROOT / "tests/native/mixed_force_endpoint_probe.cpp"],
        target,
        compile_args=(
            "-std=c++20",
            "-O2",
            "-DGENERATIVEQC_HAS_CUDA=1",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
            f"-I{cuda / 'include'}",
        ),
        link_args=(
            str(library),
            f"-L{cuda / 'lib64'}",
            "-lcudart",
            f"-Wl,-rpath,{library.parent}",
        ),
    )
    return GradientProbe(target)


@pytest.mark.parametrize(
    ("case_name", "precision_mask"),
    [
        (case_name, precision_mask)
        for case_name in (
            "water-def2-svp-spherical",
            "water-tetramer-def2-svp-spherical",
        )
        for precision_mask in (1, 2, 4, 7)
    ]
    + [("ammonia-def2-svp-spherical", 7), ("water-def2-svp", 7)],
)
def test_same_state_independent_moving_grid_oracle(
    probe: GradientProbe, case_name: str, precision_mask: int
) -> None:
    pytest.importorskip("pyscf")
    case = benchmark_cases()[case_name]
    grid = GridSpec(48, 16, 32)
    oracle = reference(case_name, grid, 4)
    calculator = Calculator(**options(case_name, grid))
    try:
        with calculator.prepare_batch(
            [case.atoms], charges=[case.charge], multiplicities=[case.multiplicity]
        ) as batch:
            result = batch.execute(properties=("energy",), strict=True).items[0]
            strict, _ = probe(batch, len(case.atoms), False)
            mixed, work = probe(batch, len(case.atoms), precision_mask, True)
            replay, _ = probe(batch, len(case.atoms), False)
            if precision_mask & 1:
                assert work[0] > 0 and work[1] > 0 and work[2] == 0
            assert (work[3] > 0) == bool(precision_mask & 2)
            assert (work[4] > 0) == bool(precision_mask & 4)
            assert work[5] == 0
            assert result.precision["mixed_stage_fock_builds"] == 0
            assert abs(result.energy - oracle["energy"]) < 1e-7
            np.testing.assert_allclose(strict, oracle["forces"], atol=1e-6, rtol=0)
            np.testing.assert_allclose(mixed, oracle["forces"], atol=1e-5, rtol=0)
            np.testing.assert_allclose(mixed, strict, atol=1e-5, rtol=0)
            np.testing.assert_allclose(replay, strict, atol=1e-10, rtol=0)
            assert np.sqrt(np.mean((mixed - strict) ** 2)) < 3e-6
            np.testing.assert_allclose(mixed.sum(axis=0), 0, atol=2e-8, rtol=0)
    finally:
        calculator.clear_cache()


def test_nonfinite_mixed_task_retries_before_any_scatter(probe: GradientProbe) -> None:
    function = probe.library.mixed_force_retry_probe
    function.argtypes = (
        ctypes.POINTER(ctypes.c_double),
        np.ctypeslib.ndpointer(dtype=np.uint64),
        ctypes.c_char_p,
        ctypes.c_size_t,
    )
    function.restype = ctypes.c_int
    difference = ctypes.c_double()
    work = np.zeros(3, dtype=np.uint64)
    error = ctypes.create_string_buffer(512)
    assert function(ctypes.byref(difference), work, error, len(error)) == 0, (
        error.value.decode()
    )
    assert work[0] > 0 and work[1] > 0 and work[2] > 0
    assert difference.value < 1e-12


def test_pending_geometry_failure_can_reset_and_replay_strict(
    probe: GradientProbe,
) -> None:
    """Exercise real device sticky status, pending stream and discarded sources."""
    function = probe.library.mixed_geometry_reset_probe
    function.argtypes = (ctypes.c_char_p, ctypes.c_size_t)
    function.restype = ctypes.c_int
    error = ctypes.create_string_buffer(512)
    assert function(error, len(error)) == 0, error.value.decode()
