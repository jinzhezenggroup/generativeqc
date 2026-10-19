"""Root-control acceptance with an independent analytic two-site Hamiltonian.

These model controls are not native molecular qualification. Existing Hubbard
SCF/energy/J/K oracles are imported only by tests; production imports no tools.
"""

import typing
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import Atom
from generativeqc._spin_root_control import (
    _certify,
    _recover_spin_root,
    _root_distance,
    _rotated_density,
    _spin_diagnostics,
    _Work,
)
from generativeqc.spin_roots import (
    SpinRootPolicy,
    SpinRootReference,
    SpinRootUnqualified,
)
from test_uhf_recovery import T, U, _energy, _focks, _solve, _symmetric


class Model:
    def __init__(self, operator: typing.Any) -> None:
        self.operator = operator
        self.closed = False
        r = operator.problem.reference
        coefficients = np.stack([r.coefficients_alpha, r.coefficients_beta])
        occupations = np.stack([r.occupations_alpha, r.occupations_beta])
        density = np.stack(
            [(c * o) @ c.T for c, o in zip(coefficients, occupations, strict=True)]
        )
        self.reference = SpinRootReference(
            (Atom(1, (0.0, 0.0, 0.0)), Atom(1, (0.0, 0.0, 2.0))),
            r.overlap,
            density,
            coefficients,
            np.stack([r.orbital_energies_alpha, r.orbital_energies_beta]),
            occupations,
            np.stack([r.fock_alpha, r.fock_beta]),
            r.reference_energy,
            r.scf_residual,
            "independent-hubbard-sites",
            r.identity,
        )
        self.workspace_bytes = 1024

    def validate_current(self) -> None:
        if self.closed:
            raise SpinRootUnqualified("closed test model")
        self.operator.validate_current()

    def evaluate(self, density: typing.Any) -> typing.Any:
        self.validate_current()
        return SimpleNamespace(
            energy=_energy(density), fock=np.asarray(_focks(density))
        )


def solve(seed: typing.Any, energy: typing.Any) -> typing.Any:
    np.testing.assert_allclose(energy, _energy(seed), rtol=0, atol=1e-14)
    return Model(_solve(seed))


def broken() -> typing.Any:
    return Model(_solve(np.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]])))


def test_shared_control_recovers_analytic_broken_root() -> None:
    initial = Model(_symmetric())
    winner, first, final, work, distinct, equivalents = _recover_spin_root(
        initial, solve, SpinRootPolicy()
    )
    assert first.status == "UNSTABLE" and final.status == "STABLE"
    np.testing.assert_allclose(
        winner.reference.energy, -2 * T * T / U, rtol=0, atol=2e-10
    )
    assert winner.reference.energy < initial.reference.energy - 0.25
    assert distinct == 1 and equivalents > 0
    assert work.scf_restarts > 0
    squared, alpha, beta = _spin_diagnostics(winner.reference)
    np.testing.assert_allclose(squared, 1 - (2 * T / U) ** 2, rtol=0, atol=2e-9)
    np.testing.assert_allclose([alpha.sum(), beta.sum()], [1, 1], rtol=0, atol=1e-14)
    assert np.linalg.norm(alpha - beta) > 0.5


def test_stable_state_has_no_restart_and_same_reference() -> None:
    model = broken()

    def forbidden(*args: typing.Any) -> None:
        raise AssertionError("stable state must not ask another SCF owner")

    winner, first, final, work, distinct, equivalent = _recover_spin_root(
        model, forbidden, SpinRootPolicy()
    )
    assert winner is model and first is final and final.status == "STABLE"
    assert work.scf_restarts == 0
    assert distinct == 1 and equivalent == 0


@pytest.mark.parametrize(
    "options",
    [
        {"memory_budget_bytes": 1},
        {"maximum_ao": 1},
        {"maximum_dimension": 1},
        {"maximum_actions": 1},
    ],
)
def test_rejected_resource_domain_does_no_model_work(
    options: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = Model(_symmetric())

    def forbidden(*args: typing.Any) -> None:
        raise AssertionError("resource rejection must precede response/model work")

    monkeypatch.setattr(model, "evaluate", forbidden)
    monkeypatch.setattr(model.operator, "apply", forbidden)
    with pytest.raises(SpinRootUnqualified):
        _recover_spin_root(model, forbidden, SpinRootPolicy(**options))


def test_check_only_rejects_saddle() -> None:
    with pytest.raises(SpinRootUnqualified, match="UNSTABLE"):
        _recover_spin_root(Model(_symmetric()), solve, SpinRootPolicy(recover=False))


def test_unresolved_curvature_cannot_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    model = Model(_symmetric())
    monkeypatch.setattr(model.operator, "apply", lambda vector: np.zeros_like(vector))
    with pytest.raises(SpinRootUnqualified, match="NEAR_SINGULAR"):
        _recover_spin_root(model, solve, SpinRootPolicy())


def test_energy_decrease_without_physical_stationarity_is_rejected() -> None:
    def dishonest(seed: typing.Any, energy: typing.Any) -> typing.Any:
        model = broken()
        model.reference = replace(model.reference, energy=-100.0)
        return model

    with pytest.raises(SpinRootUnqualified, match="physical nonlinear model"):
        _recover_spin_root(Model(_symmetric()), dishonest, SpinRootPolicy())


def test_linear_fock_response_is_not_a_nonlinear_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = broken()

    def wrong(density: typing.Any) -> typing.Any:
        correction = model.operator._fock_response(density[0], density[1])
        return SimpleNamespace(
            energy=model.reference.energy, fock=np.asarray(correction)
        )

    monkeypatch.setattr(model, "evaluate", wrong)
    with pytest.raises(SpinRootUnqualified, match="actual model"):
        _certify(model, SpinRootPolicy(), _Work(100))


def test_failed_restart_cannot_revoke_original_owner() -> None:
    model = Model(_symmetric())

    def failed(seed: typing.Any, energy: typing.Any) -> typing.Any:
        model.closed = True
        raise RuntimeError("SCF failed after source invalidation")

    with pytest.raises(SpinRootUnqualified, match="closed test model"):
        _recover_spin_root(model, failed, SpinRootPolicy())


def test_signed_rotation_matches_closed_form_and_preserves_spin_projectors() -> None:
    reference = Model(_symmetric()).reference
    direction = np.array([1.0, -1.0]) / np.sqrt(2.0)
    for angle in [-0.4, 0.4]:
        actual = _rotated_density(reference, direction, angle)
        expected = []
        for spin in range(2):
            c = reference.coefficients[spin]
            orbital = c[:, 0] * np.cos(angle * direction[spin]) + c[:, 1] * np.sin(
                angle * direction[spin]
            )
            expected.append(np.outer(orbital, orbital))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=4e-16)
        np.testing.assert_allclose(actual @ actual, actual, rtol=0, atol=5e-16)
        np.testing.assert_allclose(
            np.trace(actual, axis1=1, axis2=2), [1, 1], atol=1e-15
        )


def test_root_metric_is_ao_covariant_and_spin_equivalence_is_explicit() -> None:
    reference = broken().reference
    q = np.array([[2.0, 0.4], [-0.2, 0.7]])
    inverse = np.linalg.inv(q)
    transformed = replace(
        reference,
        overlap=q.T @ reference.overlap @ q,
        density=np.asarray([inverse @ d @ inverse.T for d in reference.density]),
        coefficients=np.asarray([inverse @ c for c in reference.coefficients]),
        fock=np.asarray([q.T @ f @ q for f in reference.fock]),
    )
    assert np.linalg.norm(reference.density - transformed.density) > 0.1
    assert _root_distance(reference, transformed, reference.overlap @ q) < 1e-7
    flipped = replace(
        reference,
        density=reference.density[::-1],
        coefficients=reference.coefficients[::-1],
        fock=reference.fock[::-1],
        orbital_energies=reference.orbital_energies[::-1],
        occupations=reference.occupations[::-1],
    )
    assert _root_distance(reference, flipped, reference.overlap) > 1.0
    assert _root_distance(reference, flipped, reference.overlap, flip=True) < 1e-7


def test_reference_arrays_are_immutable_and_model_budget_does_not_alias() -> None:
    reference = broken().reference
    for value in (
        reference.density,
        reference.coefficients,
        reference.fock,
        reference.overlap,
    ):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(ValueError):
        SpinRootPolicy(angles=(0.4, 0.2))
    with pytest.raises(TypeError):
        SpinRootPolicy(recover=1)


def test_closed_continuation_staging_buffers_remain_in_controller_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Model(_symmetric())

    def forbidden(*args: typing.Any) -> None:
        raise AssertionError("retained-buffer rejection must precede model actions")

    monkeypatch.setattr(model, "evaluate", forbidden)
    with pytest.raises(SpinRootUnqualified, match="budget"):
        _recover_spin_root(
            model, solve, SpinRootPolicy(), retained_numeric_bytes=1 << 40
        )
