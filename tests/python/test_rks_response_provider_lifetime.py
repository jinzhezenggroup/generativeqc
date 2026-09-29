"""A zero RHS must not make a closed exact-J provider look usable."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc.response_solver import solve, solve_many
from generativeqc.rks_response import _PreparedRKSJBackend


class _Plan:
    def __init__(self) -> None:
        self.closed = False

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("prepared RKS J plan is closed")

    def close(self) -> None:
        self.closed = True


class _ZeroRHSOperator:
    dimension = 1

    def __init__(self) -> None:
        self.backend = _PreparedRKSJBackend.__new__(_PreparedRKSJBackend)
        self.backend._plan = _Plan()
        self.backend.geometry_hash = "geometry"
        self.backend.basis_hash = "basis"
        self.backend.basis = SimpleNamespace(nao=2, representation="cartesian")
        self.reference = SimpleNamespace(
            geometry_hash="geometry",
            basis_hash="basis",
            representation="cartesian",
            hamiltonian_id="conventional-unscreened",
            algorithm="KS",
            nmo=2,
        )
        self.problem = SimpleNamespace(
            reference=self.reference,
            dimension=self.dimension,
            compatibility_identity="live-rks-test",
            validate_rhs=lambda value: np.asarray(value, dtype=np.float64),
        )
        self.actions = 0

    def validate_current(self) -> None:
        self.backend.validate_reference(self.reference)

    def apply(self, value: np.ndarray) -> np.ndarray:
        self.actions += 1
        return np.asarray(value, dtype=np.float64)


def _solve_zero(operator: _ZeroRHSOperator, strategy: str):
    if strategy == "scalar":
        return solve(operator, np.zeros(1))
    return solve_many(operator, np.zeros((1, 2)), strategy=strategy)


@pytest.mark.parametrize("strategy", ["scalar", "sequential", "blocked", "recycled"])
def test_zero_rhs_checks_provider_lifetime(strategy: str) -> None:
    operator = _ZeroRHSOperator()
    assert _solve_zero(operator, strategy).converged
    before_close = operator.actions

    operator.backend.close()
    with pytest.raises(RuntimeError, match="RKS J plan is closed"):
        _solve_zero(operator, strategy)
    assert operator.actions == before_close


def test_open_provider_still_checks_reference_identity() -> None:
    operator = _ZeroRHSOperator()
    operator.reference.geometry_hash = "different-geometry"
    with pytest.raises(ValueError, match="geometry_hash mismatch"):
        operator.validate_current()
