"""Slurm-only high-angular ordering and late-failure scheduling controls."""

from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest
from test_factorized_second import DIRECTION, EXPONENTS, POSITIONS
from test_hessian_aot_cuda import _run

pytest_plugins = ("test_factorized_second_cuda",)

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TEST_FACTORIZED_CUDA") != "1",
    reason="requires the complete factorized package and a finite Slurm GPU job",
)
CASES = (
    ("eri", (3, 3, 3, 3), "full_range"),
    ("eri", (3, 3, 3, 3), "long_range"),
    ("eri", (3, 3, 3, 3), "short_range"),
    ("three_center_eri", (3, 3, 4), "full_range"),
    ("coulomb_metric", (4, 4), "full_range"),
)


@pytest.mark.parametrize("selection", CASES)
def test_high_angular_signed_radial_contraction_order(
    factorized_package: tuple, selection: tuple
) -> None:
    """Different radial records cancel across warp/block tails, without reassociation."""
    executable, programs = factorized_package
    program = programs[selection][len(programs[selection]) // 2]
    count = len(selection[1])
    suffix = np.r_[np.arange(count), DIRECTION[:count].ravel()]
    uncontracted = {}
    for scale in (0.75, 1.0, 1.2):
        record = np.r_[EXPONENTS[:count] * scale, POSITIONS[:count].ravel(), 1.0]
        uncontracted[scale] = np.asarray(
            _run(
                executable,
                "integral",
                program["name"],
                1,
                count,
                values=np.r_[record, suffix],
            )
        )
    pattern = ((0.75, 1e16), (1.0, 1.0), (0.75, -1e16), (1.2, -0.17), (1.0, 0.73))
    for records_count in (1, 31, 32, 63, 64, 65, 129):
        records = []
        expected = np.zeros_like(uncontracted[1.0])
        for radial in range(records_count):
            scale, weight = pattern[radial % len(pattern)]
            records.append(
                np.r_[EXPONENTS[:count] * scale, POSITIONS[:count].ravel(), weight]
            )
            expected += weight * uncontracted[scale]
        actual = np.asarray(
            _run(
                executable,
                "integral",
                program["name"],
                records_count,
                count,
                values=np.r_[np.asarray(records).ravel(), suffix],
            )
        )
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("selection", CASES)
@pytest.mark.parametrize("invalid_record", (1, 63, 64, 128))
def test_high_angular_late_failure_preserves_whole_publication(
    factorized_package: tuple, selection: tuple, invalid_record: int
) -> None:
    """An invalid physical exponent must not strand other lanes at a barrier."""
    executable, programs = factorized_package
    program = programs[selection][len(programs[selection]) // 2]
    count = len(selection[1])
    records = np.tile(
        np.r_[EXPONENTS[:count], POSITIONS[:count].ravel(), 1.0], (129, 1)
    )
    records[invalid_record, 0] = 0.0
    with pytest.raises(subprocess.CalledProcessError) as failure:
        _run(
            executable,
            "integral",
            program["name"],
            129,
            count,
            values=np.r_[records.ravel(), np.arange(count), DIRECTION[:count].ravel()],
        )
    assert failure.value.stdout == ""
    assert "no output published" in failure.value.stderr
