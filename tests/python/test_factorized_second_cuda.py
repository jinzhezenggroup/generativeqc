"""Slurm-only physical gates for every linked ordered f/g shell class.

Each case checks one explicit component tile, not every Cartesian component.
The complete finite inventory gate separately requires every component selector
to be linked. Independent libcint gradients supply Richardson-extrapolated JVPs.
"""

from __future__ import annotations

import os
from itertools import pairwise, product
from pathlib import Path

import numpy as np
import pytest
from test_factorized_second import DIRECTION, EXPONENTS, POSITIONS, reference
from test_hessian_aot_cuda import _run

CASES = (
    *(
        (0, angular, radial)
        for angular in product(range(4), repeat=4)
        for radial in range(3)
    ),
    *((1, angular, 0) for angular in product(range(4), range(4), range(5))),
    *((2, angular, 0) for angular in product(range(5), repeat=2)),
)
pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TEST_FACTORIZED_CUDA") != "1",
    reason="requires a complete factorized native package and a Slurm GPU",
)


@pytest.fixture(scope="module")
def factorized_package() -> tuple[Path, dict]:
    """Inspect the actual linked selectors without changing device visibility."""
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require finite srun"
    pytest.importorskip("pyscf")
    executable = Path(os.environ["GENERATIVEQC_HESSIAN_AOT_PROBE"])
    metadata = _run(executable, "list")
    programs = {}
    for item in metadata:
        if item.get("family") in ("eri", "three_center_eri", "coulomb_metric"):
            key = (item["family"], tuple(item["angular"]), item["radial"])
            programs.setdefault(key, []).append(item)
    return executable, programs


@pytest.mark.parametrize(
    "family,angular,radial",
    CASES,
    ids=tuple(
        f"{('eri', 'df3', 'metric')[family]}-{''.join('spdfg'[power] for power in angular)}-{('full', 'long', 'short')[radial]}"
        for family, angular, radial in CASES
    ),
)
def test_every_linked_ordered_class_against_libcint(
    factorized_package: tuple, family: int, angular: tuple, radial: int
) -> None:
    executable, programs = factorized_package
    key = (
        ("eri", "three_center_eri", "coulomb_metric")[family],
        angular,
        ("full_range", "long_range", "short_range")[radial],
    )
    selected = programs[key]
    flattened = [component for item in selected for component in item["components"]]
    size = np.prod([(order + 1) * (order + 2) // 2 for order in angular])
    assert len(flattened) == size and sorted(flattened) == list(range(size))
    # Mid-class tiles exercise mixed Cartesian axes rather than only x...x.
    program = next(item for item in selected if int(size) // 2 in item["components"])
    count = len(angular)
    record = np.r_[EXPONENTS[:count], POSITIONS[:count].ravel(), 1.0]
    actual = np.array(
        _run(
            executable,
            "integral",
            program["name"],
            1,
            count,
            values=np.r_[record, np.arange(count), DIRECTION[:count].ravel()],
        )
    ).reshape(len(program["components"]), 1 + 6 * count)
    value, gradient = reference(family, angular, POSITIONS[:count], radial)
    estimates = []
    # libcint's short-range gradient has small quadrature noise that a 1e-4
    # displacement amplifies. Two independently extrapolated, larger stencils
    # must both meet the unchanged gate, rather than tuning the gate to noise.
    for step in (2e-2, 1e-2, 5e-3):
        displaced = [
            reference(
                family,
                angular,
                POSITIONS[:count] + sign * step * DIRECTION[:count],
                radial,
            )[1]
            for sign in (-1, 1)
        ]
        estimates.append((displaced[1] - displaced[0]) / (2 * step))
    for coarse, fine in pairwise(estimates):
        jvp = (4 * fine - coarse) / 3
        for packed, component in enumerate(program["components"]):
            expected = np.r_[
                value[component], gradient[:, component], jvp[:, component]
            ]
            np.testing.assert_allclose(actual[packed], expected, atol=3e-10, rtol=2e-9)
