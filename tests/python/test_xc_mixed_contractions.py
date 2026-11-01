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
    XcMatrixSchedule,
    _emit_panels,
    _emit_tiled,
    compact_panel_program,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(
    scope="module",
    params=[(tile, fmad) for tile in (8, 16, 32) for fmad in ("false", "true")],
    ids=lambda value: f"tile{value[0]}-fmad{value[1]}",
)
def contraction_probe(native_cxx: Any, tmp_path_factory: Any, request: Any) -> Any:
    """Compile the production bodies, not a parallel test implementation."""
    if os.environ.get("GENERATIVEQC_RUN_CUDA_TESTS") != "1":
        pytest.skip("scheduled CUDA qualification is opt-in")
    # The ECNU A100 queue uses LSF; retain its real allocation identity rather
    # than requiring a fabricated Slurm ID for the same device qualification.
    if not (os.environ.get("SLURM_JOB_ID") or os.environ.get("LSB_JOBID")):
        pytest.fail("CUDA qualification requires a finite scheduler allocation")
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.fail("scheduled CUDA qualification requires nvcc")
    directory = tmp_path_factory.mktemp("mixed-contractions")
    # Launch every supported schedule: tile32 previously exceeded the per-block
    # register limit despite passing the host thread/shared-memory admission.
    tile, fmad = request.param
    generated = _emit_panels(diagonal_one_sided=True) + _emit_tiled(
        XcMatrixSchedule(tile)
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
            f"--fmad={fmad}",
            f"-DGQC_TEST_XC_MATRIX_TILE={tile}",
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


@pytest.mark.parametrize("second", [1, 33])
@pytest.mark.parametrize(
    "sign,swapped", [(-1, False), (-1, True), (1, False), (1, True)]
)
def test_fp64_cross_cancellation_preserves_large_incoming_sum(
    contraction_probe: Any, second: int, sign: int, swapped: bool
) -> None:
    """Opposite cross legs must cancel before an otherwise overflowing addition."""
    active, count = 16, 48
    large, leg = sign * float.fromhex("0x1.8p1023"), sign * 2.0**1022
    ao = np.zeros((1, count, active))
    work = np.zeros_like(ao)
    ao[0, 0, 0] = 1.0
    work[0, 0, 1] = large
    ao[0, second, :2] = 1.0
    work[0, second, :2] = [leg, -leg] if swapped else [-leg, leg]
    expected = np.zeros((active, active))
    expected[0, 1] = expected[1, 0] = large
    expected[0, 0] = 2 * leg if swapped else -2 * leg
    expected[1, 1] = -expected[0, 0]
    density = np.zeros_like(expected)
    ids = np.arange(active, dtype=np.uintp)
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
                1,
                ids,
                mode,
            )
            == 0
        )
        np.testing.assert_array_equal(potential, expected)


@pytest.mark.parametrize("exponent", [-500, 128, 129])
def test_fp64_cross_exact_dyadic_operands_at_range_boundary(
    contraction_probe: Any, exponent: int
) -> None:
    """Tiny, boundary and unbounded operands share the exact bilinear contract."""
    active, count = 19, 33
    value = 2.0**exponent
    ao = np.zeros((1, count, active))
    work = np.zeros_like(ao)
    ao[0, 0, :2] = value
    work[0, 0, :2] = [value, -value]
    # A ragged final tile contributes an independent exactly representable term.
    ao[0, -1, -1] = work[0, -1, -1] = value
    expected = np.zeros((active, active))
    expected[0, 0] = expected[-1, -1] = 2.0 ** (2 * exponent + 1)
    expected[1, 1] = -expected[0, 0]
    density = np.zeros_like(expected)
    ids = np.arange(active, dtype=np.uintp)
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
                1,
                ids,
                mode,
            )
            == 0
        )
        np.testing.assert_array_equal(potential, expected)


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_fp64_cross_nonfinite_operands_are_rejected(
    contraction_probe: Any, bad: float
) -> None:
    """A failed finite-range proof must not suppress arithmetic rejection."""
    active, count = 16, 16
    ao = np.ones((1, count, active))
    work = np.zeros_like(ao)
    work[0, 0, 0] = bad
    density = np.zeros((active, active))
    assert (
        contraction_probe(
            density,
            ao,
            work,
            np.empty_like(ao),
            np.zeros_like(density),
            count,
            active,
            active,
            1,
            np.arange(active, dtype=np.uintp),
            0,
        )
        > 0
    )


@pytest.mark.parametrize(
    "active,count,jets",
    [
        (1, 1, 1),
        (1, 17, 4),
        (17, 1, 4),
        (3, 5, 1),
        (19, 33, 1),
        (47, 257, 4),
        (96, 256, 4),
    ],
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


@pytest.mark.parametrize("exponent", [-320, -200, -120, -50, 0])
def test_fp32_density_products_preserve_representable_tiny_tails(
    contraction_probe: Any,
    exponent: int,
) -> None:
    """FP64 AO storage can represent tails far below FP32's smallest value."""
    active, count, jets = 19, 33, 4
    rng = np.random.default_rng(721)
    factor = rng.normal(size=(active, active)) / active
    density = np.ascontiguousarray(factor @ factor.T + np.eye(active))
    ao = np.ascontiguousarray(rng.normal(size=(jets, count, active)) * 10.0**exponent)
    work = np.ones_like(ao)
    products = np.empty_like(ao)
    potential = np.zeros((active, active))
    ids = np.arange(active, dtype=np.uintp)
    assert (
        contraction_probe(
            density, ao, work, products, potential, count, active, active, jets, ids, 1
        )
        == 0
    )
    expected = ao @ density.T
    tolerance = 3e-6 * np.max(np.abs(expected)) + active * np.nextafter(0.0, 1.0)
    assert np.max(np.abs(products - expected)) <= tolerance
    assert np.any(products != 0.0)
    rho = np.sum(ao[0] * products[0], axis=1)
    assert np.all(rho >= 0.0)


@pytest.mark.parametrize("mode", [1, 4, 5])
def test_fp32_overflow_reports_rejection_instead_of_clipping(
    contraction_probe: Any,
    mode: int,
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
        contraction_probe(density, ao, work, products, potential, 5, 3, 3, 1, ids, mode)
        > 0
    )


@pytest.mark.parametrize("point", [0, 14, 15, 16, 30, 31])
@pytest.mark.parametrize("mode", [0, 2, 3])
def test_fp64_cross_cancellation_survives_tile_boundaries(
    contraction_probe: Any, point: int, mode: int
) -> None:
    """A finite bilinear must survive an overflowing intermediate FMA.

    Use two spin channels, noncontiguous AO labels, a partial AO block and a
    partial final point tile. The independent answer follows from exactly
    cancelling opposite legs, including a large accumulator from an earlier
    tile. Reversing both spin/sign orientations exercises either nested leg.
    """
    active, count, full, jets, spins = 19, 33, 41, 4, 2
    ids = np.arange(1, 2 * active + 1, 2, dtype=np.uintp)
    density = np.zeros((spins, full, full))
    ao = np.zeros((jets, count, active))
    work = np.zeros((spins, jets, count, active))
    products = np.empty_like(work)
    potential = np.zeros((spins, full, full))
    expected = np.zeros_like(potential)
    ao[0, point, 0] = 1.0
    ao[0, point + 1, :2] = 1.0
    for spin, sign in enumerate((-1, 1)):
        large = sign * float.fromhex("0x1.8p1023")
        leg = sign * float.fromhex("0x1p1022")
        first = leg if spin else -leg
        work[spin, 0, point, 1] = large
        work[spin, 0, point + 1, :2] = (first, -first)
        expected[spin, ids[0], ids[1]] = large
        expected[spin, ids[1], ids[0]] = large
        expected[spin, ids[0], ids[0]] = 2 * first
        expected[spin, ids[1], ids[1]] = -2 * first
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
            mode,
            spins=spins,
        )
        == 0
    )
    np.testing.assert_array_equal(potential, expected)
    np.testing.assert_array_equal(products, 0.0)


@pytest.mark.parametrize("active,count", [(1, 1), (19, 33), (47, 257), (96, 256)])
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("mode", [2, 4, 5])
def test_diagonal_bilinear_uses_one_authoritative_triangle(
    contraction_probe: Any, active: int, count: int, mapped: bool, mode: int
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
            density, ao, work, products, potential, count, active, full, 4, ids, mode
        )
        == 0
    )
    actual = initial.copy()
    actual[np.ix_(ids, ids)] += expected
    if mode in (4, 5):
        np.testing.assert_allclose(potential, actual, rtol=0, atol=5e-4)
    else:
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
    for mode in (2, 3, 4, 5):
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
            if mode in (4, 5):
                np.testing.assert_allclose(potential[spin], matrix, rtol=0, atol=5e-4)
                np.testing.assert_array_equal(
                    potential[spin],
                    potential[spin].astype(np.float32).astype(np.float64),
                )
            else:
                np.testing.assert_allclose(
                    potential[spin], matrix, rtol=3e-14, atol=2e-12
                )
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
    [
        "finite-doubling-overflow",
        "underflow-cancellation",
        "hidden-preparation-underflow",
    ],
)
def test_fp32_diagonal_range_fallback_preserves_original_arithmetic(
    contraction_probe: Any,
    case: str,
) -> None:
    """FP32-specific guards retain finite canonical products and cancellation."""
    active = count = 16
    ao = np.zeros((4, count, active))
    coefficients = np.zeros((1, 5, count))
    weights = np.ones(count)
    if case == "finite-doubling-overflow":
        ao[1, 0] = 2.0**-100
        coefficients[0, 4, 0] = 2.0**228
    elif case == "underflow-cancellation":
        ao[1, 0, :2] = [2.0**40, 2.0**-80]
        ao[1, 1, :2] = [2.0**-80, 2.0**40]
        coefficients[0, 4, :2] = [2.0**-120, -(2.0**-120)]
        weights[:2] = 2.0**100
    else:
        ao[1, 0, :2] = [2.0**-60, 2.0**60]
        coefficients[0, 4, 0] = 2.0**-160
        weights[0] = 2.0**140
    work = np.empty_like(ao)
    assert (
        contraction_probe.diagonal_panels(
            ao, coefficients, weights, work, active, count, 1
        )
        == 0
    )
    density = np.zeros((active, active))
    indices = np.arange(active, dtype=np.uintp)
    matrices = []
    for mode in (1, 4, 5):
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
                indices,
                mode,
                weights=weights,
            )
            == 0
        )
        assert np.isfinite(potential).all()
        matrices.append(potential)
    for matrix in matrices[1:]:
        np.testing.assert_array_equal(matrix, matrices[0])
    if case == "finite-doubling-overflow":
        np.testing.assert_array_equal(matrices[0], np.full_like(density, 2.0**28))
    elif case == "underflow-cancellation":
        assert matrices[0][0, 1] == 0
    else:
        assert matrices[0][0, 1] == 2.0**-20


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
