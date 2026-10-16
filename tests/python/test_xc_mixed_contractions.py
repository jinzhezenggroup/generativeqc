"""FP32 arithmetic and indexed/tail parity against independent NumPy products."""

import ctypes
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.common.array_graph import evaluate_array_graph
from generativeqc_compiler.dft.xc_contraction_cuda import (
    DEFAULT_XC_MATRIX_SCHEDULE,
    _emit_panels,
    _emit_tiled,
    compact_panel_program,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", params=["false", "true"])
def contraction_probe(native_cxx: Any, tmp_path_factory: Any, request: Any) -> Any:
    """Compile the production bodies, not a parallel test implementation."""
    if os.environ.get("GENERATIVEQC_RUN_CUDA_TESTS") != "1":
        pytest.skip("scheduled CUDA qualification is opt-in")
    if not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("CUDA qualification requires finite srun")
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.fail("scheduled CUDA qualification requires nvcc")
    directory = tmp_path_factory.mktemp("mixed-contractions")
    generated = _emit_panels(diagonal_one_sided=True) + _emit_tiled(
        DEFAULT_XC_MATRIX_SCHEDULE
    )
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
            f"--fmad={request.param}",
            "--ftz=false",
            "-arch=native",
            f"-I{directory}",
        ),
    )
    library = ctypes.CDLL(str(path))
    probe = library.mixed_contractions
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    indices = np.ctypeslib.ndpointer(dtype=np.uintp, flags="C_CONTIGUOUS")
    probe.argtypes = (
        (pointer,) * 5
        + (ctypes.c_int64,) * 4
        + (indices, ctypes.c_int, ctypes.c_int64, pointer)
    )
    probe.restype = ctypes.c_int
    panels = library.diagonal_panels
    panels.argtypes = (pointer,) * 4 + (ctypes.c_int64,) * 3
    panels.restype = ctypes.c_int

    def run_probe(*args: Any, spins: int = 1, weights: Any = None) -> int:
        if weights is None:
            weights = np.ones(args[5])
        return probe(*args, spins, weights)

    run_probe.diagonal_panels = panels
    return run_probe


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


@pytest.mark.parametrize("active,count", [(1, 1), (19, 33), (47, 257), (96, 256)])
@pytest.mark.parametrize("mapped", [False, True])
def test_fp64_diagonal_bilinear_uses_one_authoritative_triangle(
    contraction_probe: Any, active: int, count: int, mapped: bool
) -> None:
    """Independent full bilinear, signed weights, ragged tails and global scatter."""
    generator = np.random.default_rng(20261016)
    full = 2 * active + 3 if mapped else active
    ids = np.sort(generator.choice(full, size=active, replace=False)).astype(np.uintp)
    density = generator.normal(size=(full, full))
    ao = generator.normal(size=(4, count, active))
    coefficients = generator.normal(size=(5, count))
    weights = generator.normal(size=count)
    graph, roots = compact_panel_program("mgga")
    bindings = {f"x{jet}": ao[jet] for jet in range(4)}
    bindings.update({f"c{term}": coefficients[term, :, None] for term in range(5)})
    work = np.ascontiguousarray(evaluate_array_graph(graph, roots, bindings))
    work *= weights[None, :, None]
    expected = ao[0].T @ ((weights * coefficients[0])[:, None] * ao[0])
    for jet in range(1, 4):
        cross = ao[jet].T @ ((weights * coefficients[jet])[:, None] * ao[0])
        expected += cross + cross.T
        expected += ao[jet].T @ ((weights * coefficients[4])[:, None] * ao[jet])
    products = np.empty_like(ao)
    initial = generator.normal(size=(full, full))
    initial = 0.5 * (initial + initial.T)
    potential = initial.copy()
    assert (
        contraction_probe(
            density, ao, work, products, potential, count, active, full, 4, ids, 2
        )
        == 0
    )
    actual = initial.copy()
    actual[np.ix_(ids, ids)] += expected
    np.testing.assert_allclose(potential, actual, rtol=3e-14, atol=2e-12)
    np.testing.assert_array_equal(potential, potential.T)


@pytest.mark.parametrize("spins", [1, 2])
def test_generated_diagonal_panel_producer_preserves_spin_and_weights(
    contraction_probe: Any, spins: int
) -> None:
    generator = np.random.default_rng(20261016)
    active, count = 19, 33
    ao = generator.normal(size=(4, count, active))
    coefficients = generator.normal(size=(spins, 5, count))
    weights = generator.normal(size=count)
    output = np.empty((spins, 4, count, active))
    assert (
        contraction_probe.diagonal_panels(
            ao, coefficients, weights, output, active, count, spins
        )
        == 0
    )
    density = generator.normal(size=(spins, active, active))
    for mode in (2, 3):
        potential = np.zeros((spins, active, active))
        products = np.empty_like(output)
        assert (
            contraction_probe(
                density,
                ao,
                output,
                products,
                potential,
                count,
                active,
                active,
                4,
                np.arange(active, dtype=np.uintp),
                mode,
                spins=spins,
            )
            == 0
        )
        for spin in range(spins):
            matrix = ao[0].T @ ((weights * coefficients[spin, 0])[:, None] * ao[0])
            for jet in range(1, 4):
                cross = ao[jet].T @ (
                    (weights * coefficients[spin, jet])[:, None] * ao[0]
                )
                matrix += cross + cross.T
                matrix += ao[jet].T @ (
                    (weights * coefficients[spin, 4])[:, None] * ao[jet]
                )
            np.testing.assert_allclose(potential[spin], matrix, rtol=3e-14, atol=2e-12)
    expected = np.empty_like(output)
    for spin in range(spins):
        expected[spin, 0] = 0.5 * coefficients[spin, 0, :, None] * ao[0]
        for jet in range(1, 4):
            expected[spin, 0] += coefficients[spin, jet, :, None] * ao[jet]
            expected[spin, jet] = 0.5 * coefficients[spin, 4, :, None] * ao[jet]
    expected *= weights[None, None, :, None]
    np.testing.assert_allclose(output, expected, rtol=3e-15, atol=3e-15)
    ao[2, 0, 0] = np.nan
    assert (
        contraction_probe.diagonal_panels(
            ao, coefficients, weights, output, active, count, spins
        )
        > 0
    )


@pytest.mark.parametrize(
    "case",
    ["finite-overflow", "underflow-cancellation", "hidden-preparation-underflow"],
)
def test_diagonal_range_fallback_preserves_original_finite_arithmetic(
    contraction_probe: Any, case: str
) -> None:
    """Unsafe finite inputs execute both canonical legs, not a repaired full panel."""
    active = count = 16
    ao = np.zeros((4, count, active))
    coefficients = np.zeros((1, 5, count))
    weights = np.zeros(count)
    if case == "finite-overflow":
        ao[1, 0] = 1e-100
        coefficients[0, 4, 0] = 5e307
        weights[0] = 4e100
    elif case == "underflow-cancellation":
        ao[1, 0, :2] = [2.0**300, 2.0**-600]
        ao[1, 1, :2] = [2.0**-600, 2.0**300]
        coefficients[0, 4, :2] = [2.0**-475, -(2.0**-475)]
        weights[:2] = 2.0**800
    else:
        ao[1, 0, :2] = [2.0**-60, 2.0**100]
        coefficients[0, 4, 0] = 2.0**-1020
        weights[0] = 2.0**1000
    work = np.empty_like(ao)
    assert (
        contraction_probe.diagonal_panels(
            ao, coefficients, weights, work, active, count, 1
        )
        == 0
    )
    density = np.zeros((active, active))
    ids = np.arange(active, dtype=np.uintp)
    matrices = []
    for mode in (0, 2, 3):
        potential = np.zeros_like(density)
        assert (
            contraction_probe(
                density,
                ao,
                work,
                np.empty_like(ao),
                potential,
                count,
                active,
                active,
                4,
                ids,
                mode,
                weights=weights,
            )
            == 0
        )
        assert np.isfinite(potential).all()
        matrices.append(potential)
    for matrix in matrices[1:]:
        np.testing.assert_array_equal(matrix, matrices[0])
    if case == "finite-overflow":
        np.testing.assert_allclose(matrices[0], 2e208, rtol=2e-15, atol=0)
    elif case == "underflow-cancellation":
        assert matrices[0][0, 1] == 0
    else:
        assert matrices[0][0, 1] == 2.0**19
