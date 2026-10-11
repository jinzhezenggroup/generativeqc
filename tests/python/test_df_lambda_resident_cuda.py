"""Real-device DF Lambda residency, physical factor oracle and refusal recovery."""

from __future__ import annotations

import os
import typing

import numpy as np
import pytest
from test_df_cc_lambda import PARAMETERS, case, feeds, run
from test_df_cc_lambda import probe as _native_probe

from tools.generativeqc_cc.oracle import DeterminantOracle, dense_feeds

probe = _native_probe


def check_residency(counts: np.ndarray, o: int, v: int, budget: int) -> None:
    """Charge both providers and require action-count-independent seed staging."""
    singles = o * v
    doubles = singles * singles
    pairs = (singles + doubles) // 2
    dimension = singles + pairs
    assert counts[41] == 1 and counts[42] == 0
    assert 0 < counts[43] < counts[44] <= budget
    assert 0 < counts[46] <= counts[45]
    assert counts[47] > 0 and counts[48] > 0
    assert counts[49] == 2 * pairs * np.dtype(np.uintp).itemsize + 3 * dimension * 8
    assert counts[50] > 0 and counts[51] > 0
    assert counts[2] <= budget


@pytest.mark.parametrize("budget", [256 << 20, 512 << 20])
@pytest.mark.parametrize("o,v", [(1, 3), (2, 3), (3, 2)])
@pytest.mark.parametrize(
    "reduction,matrix,audit,sharing",
    [
        (False, False, False, False),
        (True, False, False, True),
        (True, True, True, False),
    ],
)
def test_resident_df_transpose_matches_existing_physical_schedules(
    probe: typing.Any,
    budget: int,
    o: int,
    v: int,
    reduction: bool,
    matrix: bool,
    audit: bool,
    sharing: bool,
) -> None:
    """Exercise expanded, scalar staged and matrix tail actions without new equations."""
    assert os.environ.get("SLURM_JOB_ID")
    _, _, arrays = case(o, v, q=5)
    options = {
        "budget": budget,
        "reduction": reduction,
        "matrix": matrix,
        "audit_matrix": audit,
        "primal_matrix": audit,
        "batch_two": True,
        "serial_scratch_reuse": sharing,
        "source": True,
    }
    status, expected, expected_values, _, error = run(probe, arrays, **options)
    assert status == 0, error
    status, actual, values, counts, error = run(
        probe, arrays, resident_krylov=True, **options
    )
    assert status == 0, error
    check_residency(counts, o, v, budget)
    assert np.max(values[1:]) < 1e-9
    np.testing.assert_allclose(values[0], expected_values[0], atol=1e-13, rtol=0)
    for candidate, baseline in zip(actual, expected, strict=True):
        np.testing.assert_allclose(candidate, baseline, atol=3e-10, rtol=2e-9)


@pytest.mark.parametrize("budget", [256 << 20, 512 << 20])
def test_resident_df_factor_response_against_determinant_multistep_oracle(
    probe: typing.Any, budget: int
) -> None:
    """Two-electron CCSD is exact: fermionic operators supply a separate oracle."""
    assert os.environ.get("SLURM_JOB_ID")
    occupied, virtuals = 1, 3
    factors, fock, _arrays = case(occupied, virtuals, q=5)
    direction = np.random.default_rng(1765).normal(scale=0.02, size=factors.shape)
    direction = (direction + direction.transpose(0, 2, 1)) / 2
    for change in (0.0, 0.003):
        changed = factors + change * direction
        status, result, values, counts, error = run(
            probe,
            feeds(changed, fock, occupied),
            budget=budget,
            resident_krylov=True,
            native_defaults=True,
        )
        assert status == 0, error
        check_residency(counts, occupied, virtuals, budget)
        integrals = np.einsum("Qpq,Qrs->pqrs", changed, changed)
        exact = np.linalg.eigvalsh(
            DeterminantOracle(fock, integrals, occupied).hnormal
        )[0]
        np.testing.assert_allclose(values[0], exact, atol=3e-11, rtol=0)
        derivative_integrals = np.einsum(
            "Qpq,Qrs->pqrs", direction, changed
        ) + np.einsum("Qpq,Qrs->pqrs", changed, direction)
        derivative = dense_feeds(
            np.zeros_like(fock),
            derivative_integrals,
            np.zeros((occupied, virtuals)),
            np.zeros((occupied, occupied, virtuals, virtuals)),
        )
        analytic = sum(
            float(np.sum(result[index + 4] * derivative[name]))
            for index, name in enumerate(PARAMETERS)
        )
        analytic += float(
            np.sum(result[12] * direction[:, :occupied, occupied:])
            + np.sum(result[13] * direction[:, occupied:, occupied:])
        )
        for step in (2e-4, 1e-4, 5e-5):
            energies = []
            for sign in (-1, 1):
                displaced = changed + sign * step * direction
                gram = np.einsum("Qpq,Qrs->pqrs", displaced, displaced)
                energies.append(
                    np.linalg.eigvalsh(DeterminantOracle(fock, gram, occupied).hnormal)[
                        0
                    ]
                )
            np.testing.assert_allclose(
                analytic,
                (energies[1] - energies[0]) / (2 * step),
                atol=3e-10,
                rtol=3e-6,
            )


def test_resident_df_refusal_and_failure_do_not_poison_next_endpoint(
    probe: typing.Any,
) -> None:
    """Optional refusal keeps the physical path; failed calls publish no arrays."""
    assert os.environ.get("SLURM_JOB_ID")
    _, _, arrays = case(2, 3, q=5)
    status, expected, _, baseline, error = run(probe, arrays)
    assert status == 0, error
    for options in (
        {"budget": int(baseline[2])},
        {"device_budget": int(baseline[3])},
    ):
        status, actual, values, counts, error = run(
            probe, arrays, resident_krylov=True, **options
        )
        assert status == 0, error
        assert counts[41] == 0 and counts[42] == 1
        assert np.max(values[1:]) < 1e-9
        for candidate, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(candidate, reference, atol=3e-10, rtol=2e-9)
    for options in (
        {"budget": 1},
        {"overflow": True},
        {"overflow": True, "overflow_audit": True},
    ):
        status, actual, values, counts, error = run(
            probe, arrays, resident_krylov=True, **options
        )
        assert status != 0 and error
        assert all(np.isnan(output).all() for output in actual)
        assert np.isnan(values).all() and not counts.any()
        status, actual, values, counts, error = run(
            probe, arrays, resident_krylov=True, budget=256 << 20
        )
        assert status == 0, error
        check_residency(counts, 2, 3, 256 << 20)
        assert np.max(values[1:]) < 1e-9
        for candidate, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(candidate, reference, atol=3e-10, rtol=2e-9)
