"""Finite-Slurm real-device gates for the opt-in masked residual prototype."""

from __future__ import annotations

import ctypes as ct
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.integral.md_j_masked_cuda import emit_md_j_masked_header
from scipy.special import hyp1f1
from test_md_j_masked_codegen import _PAIRS, _canonical_entries, _orbit_oracle

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TEST_MD_J_MASKED") != "1",
    reason="requires explicit finite Slurm masked-Hermite qualification",
)


@pytest.fixture(scope="module")
def gpu_probe(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Compile the actual residual owner; keep the test ABI out of production."""
    assert os.environ.get("SLURM_JOB_ID")
    compiler = shutil.which("nvcc")
    launcher = shutil.which("sccache") or shutil.which("ccache")
    assert compiler and launcher, "CUDA and a verified compiler cache are prerequisites"
    subprocess.run([launcher, "--version"], check=True, capture_output=True, timeout=30)
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path_factory.mktemp("masked-cuda")
    (directory / "generated_md_j_masked.cuh").write_text(emit_md_j_masked_header())
    objects = []
    for source in (
        root / "src/scf/cuda/direct_md_jk.cu",
        root / "tests/native/md_j_masked_probe.cu",
    ):
        output = directory / (source.stem + ".o")
        subprocess.run(
            [
                launcher,
                compiler,
                "-std=c++20",
                "-arch=sm_80",
                "-lineinfo",
                "-Xcompiler=-fPIC",
                "-c",
                "-I" + str(root / "src"),
                "-I" + str(root / "include"),
                "-I" + str(directory),
                str(source),
                "-o",
                str(output),
            ],
            check=True,
            capture_output=True,
            timeout=300,
        )
        objects.append(str(output))
    library = directory / "probe.so"
    subprocess.run(
        [compiler, "-shared", *objects, "-o", str(library)], check=True, timeout=120
    )
    loaded = ct.CDLL(str(library))
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    unsigneds = np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS")
    counters = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
    loaded.run_masked_gpu.argtypes = [
        ct.c_uint,
        ct.c_uint,
        ct.c_uint,
        unsigneds,
        ct.c_bool,
        ct.c_bool,
        ct.c_double,
        doubles,
        doubles,
        doubles,
        ct.c_uint,
        ct.c_uint,
        doubles,
        doubles,
        doubles,
        doubles,
        ct.c_bool,
        ct.c_bool,
        ct.c_uint,
        doubles,
        counters,
    ]
    loaded.run_masked_gpu.restype = ct.c_int
    return loaded


def _angular(order: int) -> list[tuple[int, int, int]]:
    return [
        (first, second, third)
        for first in range(order + 1)
        for second in range(order - first + 1)
        for third in range(order - first - second + 1)
    ]


def _radial(
    bra: np.ndarray, ket: np.ndarray, bra_order: int, ket_order: int
) -> np.ndarray:
    """Independent closed Hermite derivative formula, with hypergeometric Boys.

    Unlike native fill_coulomb this uses no spatial or Boys recurrence. It is
    a scientific oracle for arbitrary finite contracted Hermite distributions.
    """
    rho = bra[0] * ket[0] / (bra[0] + ket[0])
    displacement = bra[1:] - ket[1:]
    argument = rho * np.dot(displacement, displacement)
    factor = 2 * math.pi**2.5 / (bra[0] * ket[0] * math.sqrt(bra[0] + ket[0]))
    result = np.zeros((len(_angular(bra_order)), len(_angular(ket_order))))
    for row, bra_angular in enumerate(_angular(bra_order)):
        for column, ket_angular in enumerate(_angular(ket_order)):
            powers = tuple(
                first + second
                for first, second in zip(bra_angular, ket_angular, strict=True)
            )
            value = 0.0
            for first in range(powers[0] // 2 + 1):
                for second in range(powers[1] // 2 + 1):
                    for third in range(powers[2] // 2 + 1):
                        reductions = (first, second, third)
                        order = sum(powers) - sum(reductions)
                        term = (
                            (-2 * rho) ** order
                            * hyp1f1(order + 0.5, order + 1.5, -argument)
                            / (2 * order + 1)
                        )
                        for axis, (power, reduction) in enumerate(
                            zip(powers, reductions, strict=True)
                        ):
                            term *= (
                                math.factorial(power)
                                * displacement[axis] ** (power - 2 * reduction)
                                / (
                                    2**reduction
                                    * math.factorial(reduction)
                                    * math.factorial(power - 2 * reduction)
                                )
                            )
                        value += term
            result[row, column] = factor * (-1) ** sum(ket_angular) * value
    return result


def _expected_residual(
    pairs: list[tuple[tuple[int, ...], np.ndarray, np.ndarray, int]],
    bounds: np.ndarray,
    total: np.ndarray,
    screening: float,
) -> np.ndarray:
    """Enumerate all retained AO orbit contributions without native tile logic."""
    expected = np.full_like(bounds, 17.0)
    expected[1] = 0.0
    for bra_index, (
        output_pair,
        output_e,
        output_parameters,
        output_order,
    ) in enumerate(pairs):
        for ket_index, (
            input_pair,
            input_e,
            input_parameters,
            input_order,
        ) in enumerate(pairs):
            if output_order < input_order or (
                output_order == input_order and bra_index < ket_index
            ):
                continue
            pair_bounds = [
                np.array(
                    [
                        bounds[1, row, column]
                        for _, first, second in _canonical_entries(pair)
                        for row, column in ((first, second), (second, first))
                    ]
                )
                for pair in (output_pair, input_pair)
            ]
            if (
                pair_bounds[0].max() * pair_bounds[1].max() < screening
                or pair_bounds[0].min() * pair_bounds[1].min() >= screening
            ):
                continue
            radial = np.array(
                [
                    [
                        _radial(output, source, output_order, input_order)
                        for source in input_parameters
                    ]
                    for output in output_parameters
                ]
            )
            integrals = np.einsum("pat,pqtu,qbu->ab", output_e, radial, input_e)
            expected[1] += _orbit_oracle(
                output_pair,
                input_pair,
                integrals,
                bounds[1],
                total,
                screening,
                bra_index == ket_index,
            )
    return expected


@pytest.mark.parametrize("bra_pair,ket_pair,ket_hermites,same_pair", _PAIRS)
@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("screening", [0.0, 0.25, 2.0])
@pytest.mark.parametrize("complex_mask", [False, True])
def test_resident_masked_j_matches_closed_hermite_oracle(
    gpu_probe: Any,
    bra_pair: tuple[int, ...],
    ket_pair: tuple[int, ...],
    ket_hermites: int,
    same_pair: bool,
    unrestricted: bool,
    screening: float,
    complex_mask: bool,
) -> None:
    """Compare actual kernels, fallback and all directed AO masks on one GPU."""
    generator = np.random.default_rng(2600)
    bra_transform = generator.normal(size=(3, np.prod(bra_pair[2:]), 10))
    ket_transform = (
        bra_transform.copy()
        if same_pair
        else generator.normal(size=(3, np.prod(ket_pair[2:]), ket_hermites))
    )
    bra_parameters = generator.uniform(0.2, 2.0, size=(3, 4))
    ket_parameters = (
        bra_parameters.copy() if same_pair else generator.uniform(0.2, 2.0, size=(3, 4))
    )
    bounds = generator.choice([0.125, 2.0], size=(3, 20, 20))
    if complex_mask:
        bounds = np.exp2(generator.uniform(-4.0, 4.0, size=bounds.shape))
    density = generator.normal(size=bounds.shape)
    beta = generator.normal(size=bounds.shape)
    total = density[1] + (beta[1] if unrestricted else 0.0)
    pairs = [(bra_pair, bra_transform, bra_parameters, 2)]
    if not same_pair:
        pairs.append(
            (ket_pair, ket_transform, ket_parameters, 1 if ket_hermites == 4 else 2)
        )
    expected = _expected_residual(pairs, bounds, total, screening)
    results, work = [], []
    for masked, counted, preprojected in (
        (False, True, 0),
        (True, True, 0),
        (True, False, 0),
        (False, True, 1),
        (False, False, 1),
        (False, True, 2),
        (False, True, 3),
        (False, False, 3),
        (False, True, 4),
        (False, False, 4),
    ):
        actual = np.full_like(bounds, 17.0)
        counts = np.zeros((8, 25), dtype=np.uint64)
        assert (
            gpu_probe.run_masked_gpu(
                ket_hermites,
                1,
                1,
                np.array(bra_pair + ket_pair, dtype=np.uint32),
                same_pair,
                unrestricted,
                screening,
                bounds,
                density,
                beta,
                len(bra_transform),
                len(ket_transform),
                bra_transform,
                ket_transform,
                bra_parameters,
                ket_parameters,
                masked,
                counted,
                preprojected,
                actual,
                counts,
            )
            == 0
        )
        np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=3e-12)
        results.append(actual)
        work.append(counts)
    np.testing.assert_allclose(results[1], results[0], atol=3e-10, rtol=3e-12)
    np.testing.assert_allclose(results[2], results[1], atol=3e-10, rtol=3e-12)
    np.testing.assert_array_equal(work[0][0], work[1][0])
    assert not np.any(work[0][1:6])
    assert not np.any(work[2])
    np.testing.assert_array_equal(work[0][0], work[5][0])
    assert not np.any(work[5][1:6])
    np.testing.assert_array_equal(work[0][0], work[6][0])
    assert not np.any(work[7])
    np.testing.assert_array_equal(work[0][0], work[8][0])
    np.testing.assert_array_equal(work[0][6] + work[0][7], work[8][6] + work[8][7])
    assert not np.any(work[9])
    assert work[6][4, 11] * 3 == work[6][5, 11] * 2
    if screening and not complex_mask:
        assert work[1][1].sum() > 0
    if complex_mask and screening and ket_hermites == 4:
        assert work[1][1, 11] < work[1][0, 11]


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("complex_mask", [False, True])
@pytest.mark.parametrize("primitive_count", [1, 3, 5])
def test_masked_tiles_reuse_ragged_source_grid(
    gpu_probe: Any, unrestricted: bool, complex_mask: bool, primitive_count: int
) -> None:
    """Exercise both tile axes, partial edge tiles and odd primitive chunks."""
    generator = np.random.default_rng(731)
    bra_pairs = [(5, 0, 3, 3), (9, 1, 3, 3), (13, 2, 3, 3)]
    ket_pairs = [(16, index, 3, 1) for index in range(5)]
    bra_transform = generator.normal(size=(3, primitive_count, 9, 10))
    ket_transform = generator.normal(size=(5, primitive_count, 3, 4))
    bra_parameters = generator.uniform(0.2, 2.0, size=(3, primitive_count, 4))
    ket_parameters = generator.uniform(0.2, 2.0, size=(5, primitive_count, 4))
    bounds = generator.choice([0.125, 2.0], size=(3, 20, 20))
    if complex_mask:
        bounds = np.exp2(generator.uniform(-4.0, 4.0, size=bounds.shape))
    density = generator.normal(size=bounds.shape)
    beta = generator.normal(size=bounds.shape)
    pairs = [
        (pair, bra_transform[index], bra_parameters[index], 2)
        for index, pair in enumerate(bra_pairs)
    ] + [
        (pair, ket_transform[index], ket_parameters[index], 1)
        for index, pair in enumerate(ket_pairs)
    ]
    expected = _expected_residual(
        pairs, bounds, density[1] + (beta[1] if unrestricted else 0.0), 0.25
    )
    work = []
    for policy in (0, 3, 4):
        actual = np.full_like(bounds, 17.0)
        counts = np.zeros((8, 25), dtype=np.uint64)
        assert (
            gpu_probe.run_masked_gpu(
                4,
                len(bra_pairs),
                len(ket_pairs),
                np.array(bra_pairs + ket_pairs, dtype=np.uint32),
                False,
                unrestricted,
                0.25,
                bounds,
                density,
                beta,
                primitive_count,
                primitive_count,
                bra_transform,
                ket_transform,
                bra_parameters,
                ket_parameters,
                False,
                True,
                policy,
                actual,
                counts,
            )
            == 0
        )
        np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=3e-12)
        work.append(counts)
    np.testing.assert_array_equal(work[0][0], work[1][0])
    np.testing.assert_array_equal(work[0][0], work[2][0])
    assert work[2][4, 11] <= work[2][5, 11]
    assert work[1][4, 11] * primitive_count == work[1][5, 11] * (
        (primitive_count + 1) // 2
    )
    if not complex_mask:
        assert work[1][1, 11] > 0
        output_summands = 0
        packed_output_summands = 0
        for bra_pair in bra_pairs:
            for ket_pair in ket_pairs:
                directed_bounds = [
                    np.array(
                        [
                            bounds[1, row, column]
                            for _, first, second in _canonical_entries(pair)
                            for row, column in (
                                ((first, second),)
                                if first == second
                                else ((first, second), (second, first))
                            )
                        ]
                    )
                    for pair in (bra_pair, ket_pair)
                ]
                mask = np.multiply.outer(*directed_bounds) >= 0.25
                if mask.all() or not mask.any():
                    continue
                output_summands += (
                    primitive_count
                    * ((primitive_count + 1) // 2)
                    * (10 * mask.any(axis=1).sum() + 4 * mask.any(axis=0).sum())
                )
                packed_output_summands += (
                    primitive_count * 10 * mask.any(axis=1).sum()
                    + primitive_count**2 * 4 * mask.any(axis=0).sum()
                )
        assert work[1][4, 11] == output_summands
        assert work[2][4, 11] == packed_output_summands


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("zero_density", [False, True])
def test_masked_packed_queue_preserves_refills_and_leftovers(
    gpu_probe: Any, unrestricted: bool, zero_density: bool
) -> None:
    """Filtered candidates cross a full queue and retain its partial remainder."""
    generator = np.random.default_rng(128256)
    bra_pairs = [(5, 0, 3, 3), (9, 1, 3, 3), (13, 2, 3, 3)] * 20
    ket_pairs = [(16, index, 3, 1) for index in range(5)]
    bra_transform = generator.normal(size=(60, 1, 9, 10))
    ket_transform = generator.normal(size=(5, 1, 3, 4))
    bra_parameters = generator.uniform(0.2, 2.0, size=(60, 1, 4))
    ket_parameters = generator.uniform(0.2, 2.0, size=(5, 1, 4))
    bounds = np.ones((3, 20, 20))
    for index, pair in enumerate(ket_pairs):
        for component, first, second in _canonical_entries(pair):
            bounds[1, first, second] = 0.125 if index < 2 or component == 0 else 2.0
            bounds[1, second, first] = 0.125 if index < 2 or component == 1 else 2.0
    density = generator.normal(size=bounds.shape)
    beta = generator.normal(size=bounds.shape)
    if zero_density:
        density.fill(0.0)
        beta.fill(0.0)
    pairs = [
        (pair, bra_transform[index], bra_parameters[index], 2)
        for index, pair in enumerate(bra_pairs)
    ] + [
        (pair, ket_transform[index], ket_parameters[index], 1)
        for index, pair in enumerate(ket_pairs)
    ]
    expected = _expected_residual(
        pairs, bounds, density[1] + (beta[1] if unrestricted else 0.0), 1.0
    )
    for policy in (0, 4, 5):
        actual = np.full_like(bounds, 17.0)
        counts = np.zeros((8, 25), dtype=np.uint64)
        assert (
            gpu_probe.run_masked_gpu(
                4,
                60,
                5,
                np.array(bra_pairs + ket_pairs, dtype=np.uint32),
                False,
                unrestricted,
                1.0,
                bounds,
                density,
                beta,
                1,
                1,
                bra_transform,
                ket_transform,
                bra_parameters,
                ket_parameters,
                False,
                True,
                policy,
                actual,
                counts,
            )
            == 0
        )
        np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=3e-12)
        assert counts[0, 11] == (0 if zero_density else 180)
        assert counts[6, 11] + counts[7, 11] == 300
        if zero_density and policy == 4:
            assert counts[6, 11] == 0
        if policy == 5:
            assert counts[7, 11] == 0
