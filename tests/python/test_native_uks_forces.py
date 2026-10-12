"""Native UKS force ownership, independent science and transactional lifecycle."""

import typing
from fractions import Fraction

import numpy as np
import pytest
from generativeqc import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc._dft_gradient import StationaryDerivativeContract
from generativeqc.profiles import canonical_hash
from generativeqc_compiler.method import MethodSpec, resolve_method
from test_dft_complete_cpu import (
    ATOMS,
    GRID,
    KsOptions,
    NativeAO,
    StationaryKsState,
    calculator,
    independent_global_hybrid_gradient,
)

H2 = [("H", (0.12, -0.09, -0.71)), ("H", (-0.08, 0.13, 0.79))]


def assert_reference_state(
    basis: typing.Any, state: typing.Any, reference: typing.Any
) -> None:
    """Map independent PySCF D/W through the measured AO change of basis."""
    points = np.random.default_rng(2227).normal(size=(3 * basis.nao + 20, 3))
    native_ao = basis.evaluate(points, order=0)[0]
    mode = "GTOval_cart" if basis.representation == "cartesian" else "GTOval_sph"
    reference_ao = reference.mol.eval_gto(mode, points)
    transform, _, rank, _ = np.linalg.lstsq(native_ao, reference_ao, rcond=1e-13)
    assert rank == basis.nao
    np.testing.assert_allclose(
        native_ao @ transform, reference_ao, atol=2e-11, rtol=2e-11
    )
    reference_density = reference.make_rdm1()
    reference_weighted = np.array(
        [
            (coefficients * (energies * occupations)) @ coefficients.T
            for coefficients, energies, occupations in zip(
                reference.mo_coeff, reference.mo_energy, reference.mo_occ, strict=True
            )
        ]
    )
    for spin in range(2):
        np.testing.assert_allclose(
            state.density[spin],
            transform @ reference_density[spin] @ transform.T,
            atol=3e-7,
            rtol=0,
        )
        np.testing.assert_allclose(
            state.weighted_density[spin],
            transform @ reference_weighted[spin] @ transform.T,
            atol=3e-7,
            rtol=0,
        )


@pytest.mark.parametrize("method,xc", [("pbe-uks", "PBE"), ("pbe0-uks", "PBE0")])
@pytest.mark.parametrize("fitting", ["none", "cpu"])
@pytest.mark.parametrize("atoms", [H2, ATOMS], ids=["empty-beta", "two-populations"])
def test_public_native_uks_matches_independent_moving_grid_and_final_state(
    method: str,
    xc: str,
    fitting: str,
    atoms: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public effect path consumes both spins, never the Python force driver."""
    pytest.importorskip("pyscf", reason="independent native UKS acceptance")
    calc = calculator(
        method,
        max_iterations=200,
        density_fitting=fitting,
        auxiliary_basis="sto-3g" if fitting == "cpu" else None,
    )
    with (
        calc.prepare_batch([atoms], charges=[1], multiplicities=[2]) as batch,
        NativeAO(atoms, charge=1, multiplicity=2) as basis,
    ):
        assert batch._native_dft_force_eligible()

        def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
            pytest.fail("admitted UKS forces invoked Python scientific orchestration")

        monkeypatch.setattr(batch, "_public_dft_cpu_force", forbidden)
        item = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        state = StationaryKsState.from_native(batch, basis)
        energy, gradient, reference = independent_global_hybrid_gradient(
            basis,
            state,
            method,
            xc=xc,
            density_fitting=fitting == "cpu",
            return_mean_field=True,
        )
        assert item.energy == pytest.approx(energy, abs=2e-9)
        np.testing.assert_allclose(item.forces, -gradient, atol=2e-7, rtol=0)
        np.testing.assert_allclose(item.forces.sum(axis=0), 0, atol=3e-10, rtol=0)
        assert_reference_state(basis, state, reference)


@pytest.mark.parametrize("method", ["pbe-uks", "pbe0-uks"])
@pytest.mark.parametrize("fitting", ["none", "cpu"])
def test_native_uks_ragged_replay_move_failed_neighbor_and_stale_state(
    method: str, fitting: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two spin populations share one owner without sharing stale final leases."""
    calc = calculator(method, max_iterations=200, density_fitting=fitting)
    with (
        calc.prepare_batch([H2, ATOMS], charges=[1, 1], multiplicities=[2, 2]) as batch,
        calc.prepare_batch(
            [H2, ATOMS], charges=[1, 1], multiplicities=[2, 2]
        ) as portable,
        NativeAO(H2, charge=1, multiplicity=2) as basis,
    ):

        def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
            pytest.fail("native UKS replay invoked Python stationary science")

        monkeypatch.setattr(batch, "_public_dft_cpu_force", forbidden)
        monkeypatch.setattr(portable, "_native_dft_force_eligible", lambda: False)
        old_state = None
        moved = [np.array([position for _, position in atoms]) for atoms in [H2, ATOMS]]
        moved[0][1] += [0.02, -0.01, 0.03]
        moved[1][2] += [-0.01, 0.02, 0.01]
        for coordinates in (None, None, moved):
            result = batch.execute(
                coordinates, strict=True, properties=("energy", "forces")
            )
            expected = portable.execute(
                coordinates, strict=True, properties=("energy", "forces")
            )
            for item, reference in zip(result.items, expected.items, strict=True):
                assert item.converged and item.forces is not None
                np.testing.assert_allclose(
                    item.energy, reference.energy, atol=2e-9, rtol=0
                )
                np.testing.assert_allclose(
                    item.forces, reference.forces, atol=2e-7, rtol=0
                )
            if old_state is not None:
                with pytest.raises(ValueError, match="stale"):
                    StationaryDerivativeContract(old_state.identity).validate(old_state)
            if coordinates is None:
                old_state = StationaryKsState.from_native(batch, basis)
        failed = batch.execute(
            [np.zeros((1, 3)), moved[1]], properties=("energy", "forces")
        )
        assert failed.failure_indices == (0,)
        assert failed.items[0].forces is None
        assert failed.items[1].converged and failed.items[1].forces is not None
        recovered = batch.execute(moved, strict=True, properties=("energy", "forces"))
        for item, reference in zip(recovered.items, result.items, strict=True):
            np.testing.assert_allclose(item.energy, reference.energy, atol=2e-9, rtol=0)
            np.testing.assert_allclose(item.forces, reference.forces, atol=2e-7, rtol=0)
        current_atoms = [
            (element, tuple(position))
            for (element, _), position in zip(H2, moved[0], strict=True)
        ]
        with NativeAO(current_atoms, charge=1, multiplicity=2) as moved_basis:
            final = StationaryKsState.from_native(batch, moved_basis)
            StationaryDerivativeContract(final.identity).validate(final)
    with pytest.raises(RuntimeError, match="closed"):
        StationaryDerivativeContract(final.identity).validate(final)


@pytest.mark.parametrize("method", ["pbe-uks", "pbe0-uks"])
@pytest.mark.parametrize("fitting", ["none", "cpu"])
def test_native_uks_reconverged_finite_difference(method: str, fitting: str) -> None:
    """Moving-grid total forces differentiate fully reconverged molecular energies."""
    calc = calculator(method, max_iterations=200, density_fitting=fitting)
    step = 2e-4
    with calc.prepare_batch([H2], charges=[1], multiplicities=[2]) as batch:
        item = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        coordinates = np.array([position for _, position in H2])
        for axis in range(3):
            energies = []
            for sign in (1, -1):
                displaced = coordinates.copy()
                displaced[1, axis] += sign * step
                energy = calc.singlepoint(
                    [
                        (element, tuple(position))
                        for (element, _), position in zip(H2, displaced, strict=True)
                    ],
                    charge=1,
                    multiplicity=2,
                    properties=("energy",),
                ).energy
                energies.append(energy)
            derivative = -(energies[0] - energies[1]) / (2 * step)
            assert item.forces[1, axis] == pytest.approx(derivative, abs=2e-6)


@pytest.mark.parametrize("method", ["pbe-uks", "pbe0-uks"])
@pytest.mark.parametrize("fitting", ["none", "cpu"])
def test_native_uks_nonconvergence_grants_no_force_or_final_state(
    method: str, fitting: str
) -> None:
    """A failed determinant is not a stationary source even when the model qualifies."""
    calc = calculator(method, max_iterations=1, density_fitting=fitting)
    with calc.prepare_batch([ATOMS], charges=[1], multiplicities=[2]) as batch:
        assert batch._native_dft_force_eligible()
        failed = batch.execute(properties=("energy", "forces"))
        assert failed.failure_indices == (0,)
        assert not failed.items[0].converged and failed.items[0].forces is None
        with (
            NativeAO(ATOMS, charge=1, multiplicity=2) as basis,
            pytest.raises(RuntimeError, match="invalid argument"),
        ):
            StationaryKsState.from_native(batch, basis)


@pytest.mark.parametrize("fitting", ["none", "cpu"])
def test_native_uks_scaled_composition_is_not_canonical_pbe0(fitting: str) -> None:
    """A family label and two spin blocks do not authenticate a scaled hybrid."""
    graph = resolve_method(
        MethodSpec(
            "PBE50-UKS-negative-control",
            (("GGA_X_PBE", Fraction(1, 2)), ("GGA_C_PBE", Fraction(1))),
            exact_exchange=Fraction(1, 2),
        ),
        spin="polarized",
    )
    calc = calculator(
        "pbe-uks",
        density_fitting=fitting,
        ks_options=KsOptions(grid=GRID, composition=graph),
    )
    with calc.prepare_batch([H2], charges=[1], multiplicities=[2]) as batch:
        assert not batch._native_dft_force_eligible()


@pytest.mark.parametrize("method", ["pbe-uks", "pbe0-uks"])
def test_native_uks_orbital_g_keeps_existing_ao_jet_boundary(method: str) -> None:
    """Integral through-g support is not an unsupported orbital-grid capability."""
    calc = calculator(
        method, basis=angular_basis(4, "cartesian"), density_fitting="cpu"
    )
    with pytest.raises(
        NotImplementedError, match="cpu/ao derivative 1 supports only l<=3"
    ):
        calc.prepare_batch([H2], charges=[1], multiplicities=[2])


@pytest.mark.parametrize("method", ["pbe-uks", "pbe0-uks"])
@pytest.mark.parametrize("fitting", ["none", "cpu"])
def test_native_uks_ecp_keeps_separate_force_owner(method: str, fitting: str) -> None:
    """Removing the spin restriction does not remove the Hamiltonian restriction."""
    from test_ecp import fixture

    atoms, record, reference = fixture(spin=1)
    calc = calculator(method, basis=record, density_fitting=fitting)
    if fitting == "cpu":
        with pytest.raises(NotImplementedError, match="ECP density-fitting"):
            calc.prepare_batch(
                [atoms], charges=[reference.charge], multiplicities=[reference.spin + 1]
            )
        return
    with calc.prepare_batch(
        [atoms], charges=[reference.charge], multiplicities=[reference.spin + 1]
    ) as batch:
        assert not batch._native_dft_force_eligible()


def angular_basis(angular: int, representation: str) -> BasisSet:
    """Keep occupied s orbitals while exposing high-angular source/axis paths."""
    payload = ((0, ("1.2",), (("1",),)), (angular, ("0.7",), (("1",),)))
    return BasisSet(
        "native-uks-angular",
        (ElementBasis(1, tuple(BasisShell(*shell) for shell in payload)),),
        BasisProvenance(
            "inline UKS angular acceptance", "1", "CC0-1.0", canonical_hash(payload)
        ),
        representation=representation,
    )


@pytest.mark.parametrize("method,xc", [("pbe-uks", "PBE"), ("pbe0-uks", "PBE0")])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize(
    "angular,fitting,auxiliary_angular", [(2, "none", 2), (3, "cpu", 3), (3, "cpu", 4)]
)
def test_native_uks_angular_source_oracle(
    method: str,
    xc: str,
    representation: str,
    angular: int,
    fitting: str,
    auxiliary_angular: int,
) -> None:
    """Direct d, DF orbital f and auxiliary g preserve their actual AO spaces."""
    pytest.importorskip("pyscf", reason="independent UKS angular acceptance")
    basis_set = angular_basis(angular, representation)
    auxiliary_basis = angular_basis(auxiliary_angular, representation)
    calc = calculator(
        method,
        basis=basis_set,
        max_iterations=200,
        density_fitting=fitting,
        auxiliary_basis=auxiliary_basis if fitting == "cpu" else None,
    )
    with (
        calc.prepare_batch([H2], charges=[1], multiplicities=[2]) as batch,
        NativeAO(
            H2, basis=basis_set, representation=representation, charge=1, multiplicity=2
        ) as basis,
    ):
        assert batch._native_dft_force_eligible()
        item = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        state = StationaryKsState.from_native(batch, basis)
        energy, gradient, reference = independent_global_hybrid_gradient(
            basis,
            state,
            method,
            xc=xc,
            density_fitting=fitting == "cpu",
            auxiliary_shells=auxiliary_basis.shells_for(basis.atoms),
            return_mean_field=True,
        )
        assert item.energy == pytest.approx(energy, abs=3e-9)
        np.testing.assert_allclose(item.forces, -gradient, atol=4e-7, rtol=0)
        assert_reference_state(basis, state, reference)
