"""FP32 arithmetic and indexed/tail parity against independent NumPy products."""

import ctypes
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.dft.xc_contraction_cuda import (
    DEFAULT_XC_MATRIX_SCHEDULE,
    _emit_tiled,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def contraction_probe(native_cxx: Any, tmp_path_factory: Any) -> Any:
    """Compile the production bodies, not a parallel test implementation."""
    if os.environ.get("GENERATIVEQC_RUN_CUDA_TESTS") != "1":
        pytest.skip("scheduled CUDA qualification is opt-in")
    if not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("CUDA qualification requires finite srun")
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.fail("scheduled CUDA qualification requires nvcc")
    directory = tmp_path_factory.mktemp("mixed-contractions")
    generated = _emit_tiled(DEFAULT_XC_MATRIX_SCHEDULE)
    generated = generated[: generated.index("inline bool tiled_xc_admitted(")]
    (directory / "generated_xc_mixed_contractions.cuh").write_text(generated)
    path = directory / "probe.so"
    replace(native_cxx, compiler=nvcc).build_shared(
        [ROOT / "tests/native/xc_mixed_contraction_probe.cu"],
        path,
        compile_args=(
            "-forward-unknown-to-host-compiler",
            "-std=c++20",
            "-O2",
            "--fmad=false",
            "--ftz=false",
            "-arch=native",
            f"-I{directory}",
        ),
    )
    library = ctypes.CDLL(str(path))
    probe = library.mixed_contractions
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    indices = np.ctypeslib.ndpointer(dtype=np.uintp, flags="C_CONTIGUOUS")
    probe.argtypes = (pointer,) * 5 + (ctypes.c_int64,) * 4 + (indices, ctypes.c_int)
    probe.restype = ctypes.c_int
    return probe


@pytest.mark.parametrize(
    "active,count,jets", [(3, 5, 1), (19, 33, 1), (47, 257, 4), (96, 256, 4)]
)
@pytest.mark.parametrize("mapped", [False, True])
def test_actual_fp32_products_preserve_domain_and_fp64_scatter(
    contraction_probe: Any,
    active: int,
    count: int,
    jets: int,
    mapped: bool,
) -> None:
    generator = np.random.default_rng(20261013)
    full = 2 * active + 3 if mapped else active
    ids = np.sort(generator.choice(full, size=active, replace=False)).astype(np.uintp)
    density = generator.normal(size=(full, full))
    ao = generator.normal(size=(jets, count, active))
    work = generator.normal(size=(jets, count, active))
    symmetric = (density + density.T) / 2
    expected_products = ao @ symmetric[np.ix_(ids, ids)]
    cross = sum(panel.T @ weights for panel, weights in zip(ao, work, strict=True))
    increment = cross + cross.T
    outputs = []
    for mixed in (False, True):
        products = np.empty_like(ao)
        potential = np.full((full, full), 1e8)
        assert (
            contraction_probe(
                density,
                ao,
                work,
                products,
                potential,
                count,
                active,
                full,
                jets,
                ids,
                mixed,
            )
            == 0
        )
        expected = np.full_like(potential, 1e8)
        expected[np.ix_(ids, ids)] += increment
        tolerance = 5e-4 if mixed else 2e-7
        np.testing.assert_allclose(potential, expected, rtol=0, atol=tolerance)
        np.testing.assert_array_equal(potential, potential.T)
        np.testing.assert_allclose(
            products, expected_products, rtol=0, atol=5e-5 if mixed else 1e-12
        )
        outputs.append(products)
    assert np.any(outputs[0] != outputs[1])
    np.testing.assert_array_equal(
        outputs[1], outputs[1].astype(np.float32).astype(np.float64)
    )


def test_fp32_overflow_reports_rejection_instead_of_clipping(
    contraction_probe: Any,
) -> None:
    density = np.ones((3, 3))
    ao = np.full((1, 5, 3), 1e40)
    work = np.ones_like(ao)
    products = np.empty_like(ao)
    potential = np.zeros((3, 3))
    ids = np.arange(3, dtype=np.uintp)
    assert (
        contraction_probe(density, ao, work, products, potential, 5, 3, 3, 1, ids, 0)
        == 0
    )
    assert (
        contraction_probe(density, ao, work, products, potential, 5, 3, 3, 1, ids, 1)
        > 0
    )
