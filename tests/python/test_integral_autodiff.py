"""Independent molecular/integral acceptance of the complete Gaussian AD chain."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import Calculator, Primitive, Shell
from generativeqc.experimental.integral_autodiff import IntegralRHF, IntegralValues

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def hydrogen(tmp_path_factory: pytest.TempPathFactory) -> IntegralRHF:
    # Signed and zero coefficients exercise the normalization and product-rule
    # edge cases that coordinate-only force adapters cannot observe.
    basis = (
        Shell(0, 0, (Primitive(1.2, 0.7), Primitive(0.3, 0.4))),
        Shell(1, 0, (Primitive(0.9, 1.0), Primitive(0.2, 0.0))),
    )
    owner = IntegralRHF(
        ("H", "H"), basis=basis, cache=tmp_path_factory.mktemp("integral-ad")
    )
    yield owner
    owner.close()


XYZ = np.array([[-0.2, 0.1, -0.6], [0.1, -0.15, 0.8]])


def _oracle(
    model: IntegralRHF, xyz: np.ndarray, alpha: np.ndarray, coefficients: np.ndarray
) -> object:
    """Build fresh PySCF integrals/SCF, independent of generated derivatives."""
    from pyscf import gto
    from pyscf.data.elements import ELEMENTS

    labels = [f"{ELEMENTS[number]}{i}" for i, number in enumerate(model.numbers)]
    basis = {label: [] for label in labels}
    begin = 0
    for shell in model.shells:
        count = len(shell.primitives)
        basis[labels[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *[
                    [float(alpha[p]), float(coefficients[p])]
                    for p in range(begin, begin + count)
                ],
            ]
        )
        begin += count
    return gto.M(
        atom=list(zip(labels, xyz, strict=True)),
        unit="Bohr",
        basis=basis,
        cart=True,
        charge=model.charge,
        verbose=0,
    )


def _oracle_energy(
    model: IntegralRHF, xyz: np.ndarray, alpha: np.ndarray, coefficients: np.ndarray
) -> float:
    from pyscf import scf

    mean_field = scf.RHF(_oracle(model, xyz, alpha, coefficients))
    mean_field.conv_tol = 1e-13
    mean_field.conv_tol_grad = 1e-10
    mean_field.max_cycle = 200
    energy = mean_field.kernel()
    assert mean_field.converged
    return energy


def test_primal_integrals_and_energy_match_independent_oracle(
    hydrogen: IntegralRHF,
) -> None:
    molecule = _oracle(hydrogen, XYZ, hydrogen.exponents, hydrogen.coefficients)
    actual = hydrogen.integrals(XYZ)
    np.testing.assert_allclose(
        actual.overlap, molecule.intor("int1e_ovlp"), atol=3e-13, rtol=3e-13
    )
    np.testing.assert_allclose(
        actual.hcore,
        molecule.intor("int1e_kin") + molecule.intor("int1e_nuc"),
        atol=3e-12,
        rtol=3e-13,
    )
    np.testing.assert_allclose(
        actual.eri, molecule.intor("int2e"), atol=3e-13, rtol=3e-13
    )
    assert actual.nuclear == pytest.approx(molecule.energy_nuc(), abs=1e-14)
    value, _ = hydrogen.value_and_grad(XYZ)
    assert value == pytest.approx(
        _oracle_energy(hydrogen, XYZ, hydrogen.exponents, hydrogen.coefficients),
        abs=3e-11,
    )


@pytest.mark.parametrize("parameter", ["coordinates", "exponents", "coefficients"])
def test_complete_gradient_matches_reconverged_independent_energy(
    hydrogen: IntegralRHF, parameter: str
) -> None:
    _, gradient = hydrogen.value_and_grad(XYZ)
    inputs = [XYZ.copy(), hydrogen.exponents.copy(), hydrogen.coefficients.copy()]
    index = ("coordinates", "exponents", "coefficients").index(parameter)
    rng = np.random.default_rng(20261023)
    direction = rng.normal(size=inputs[index].shape)
    direction /= np.linalg.norm(direction)
    expected = float(np.sum(getattr(gradient, parameter) * direction))
    errors = []
    for step in (2e-3, 6e-4, 2e-4):
        displaced = []
        for sign in (1, -1):
            values = [array.copy() for array in inputs]
            values[index] += sign * step * direction
            displaced.append(_oracle_energy(hydrogen, *values))
        errors.append(abs((displaced[0] - displaced[1]) / (2 * step) - expected))
    assert errors[-1] < 2e-7, errors
    assert errors[-1] < max(0.1 * errors[0], 2e-9), errors


def test_parameter_chain_keeps_translation_and_coefficient_scale_null(
    hydrogen: IntegralRHF,
) -> None:
    _, gradient = hydrogen.value_and_grad(XYZ)
    np.testing.assert_allclose(gradient.coordinates.sum(axis=0), 0, atol=2e-12)
    begin = 0
    for shell in hydrogen.shells:
        count = len(shell.primitives)
        assert (
            abs(
                gradient.coefficients[begin : begin + count]
                @ hydrogen.coefficients[begin : begin + count]
            )
            < 2e-11
        )
        begin += count
    assert abs(gradient.coefficients[-1]) > 1e-3  # zero input is not a detached input
    for array in (gradient.coordinates, gradient.exponents, gradient.coefficients):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_arbitrary_integral_cotangents_use_all_source_paths(
    hydrogen: IntegralRHF,
) -> None:
    rng = np.random.default_rng(321)
    seeds = IntegralValues(
        rng.normal(size=(2, 2)), rng.normal(size=(2, 2)), rng.normal(size=(2,) * 4), 0.7
    )
    actual = hydrogen.integral_vjp(XYZ, seeds)
    directions = [
        rng.normal(size=XYZ.shape),
        rng.normal(size=(4,)),
        rng.normal(size=(4,)),
    ]
    expected = sum(
        np.sum(g * d)
        for g, d in zip(
            (actual.coordinates, actual.exponents, actual.coefficients),
            directions,
            strict=True,
        )
    )

    def objective(sign: int, step: float) -> float:
        molecule = _oracle(
            hydrogen,
            XYZ + sign * step * directions[0],
            hydrogen.exponents + sign * step * directions[1],
            hydrogen.coefficients + sign * step * directions[2],
        )
        return float(
            np.sum(seeds.overlap * molecule.intor("int1e_ovlp"))
            + np.sum(
                seeds.hcore
                * (molecule.intor("int1e_kin") + molecule.intor("int1e_nuc"))
            )
            + np.sum(seeds.eri * molecule.intor("int2e"))
            + seeds.nuclear * molecule.energy_nuc()
        )

    step = 1e-5
    assert (objective(1, step) - objective(-1, step)) / (2 * step) == pytest.approx(
        expected, abs=2e-7, rel=2e-8
    )


def test_end_to_end_never_calls_existing_force_or_integral_endpoints(
    hydrogen: IntegralRHF, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("end-to-end AD delegated to a molecular endpoint")

    monkeypatch.setattr(Calculator, "singlepoint", forbidden)
    monkeypatch.setattr(Calculator, "batch_singlepoint", forbidden)
    value, gradient = hydrogen.value_and_grad(XYZ)
    assert np.isfinite(value) and np.isfinite(gradient.coordinates).all()
    assert hydrogen.diagnostics()["physical_residual"] < 1e-9


@pytest.mark.parametrize(
    "field,value",
    [
        ("exponents", np.zeros(4)),
        ("exponents", np.ones(3)),
        ("coefficients", np.full(4, np.nan)),
    ],
)
def test_invalid_inputs_fail_without_publishing(
    hydrogen: IntegralRHF, field: str, value: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        hydrogen.value_and_grad(XYZ, **{field: value})


def test_zero_contraction_and_coincident_nuclei_fail(hydrogen: IntegralRHF) -> None:
    with pytest.raises(ArithmeticError, match="normalization"):
        hydrogen.value_and_grad(XYZ, coefficients=np.zeros(4))
    with pytest.raises(ArithmeticError, match="geometry"):
        hydrogen.value_and_grad(np.zeros_like(XYZ))


def test_resource_failure_precedes_compilation(tmp_path: Path) -> None:
    for options in ({"max_bytes": 1}, {"max_work": 1}, {"compiler_work": 1}):
        with pytest.raises((ValueError, RuntimeError), match="budget|exceed"):
            IntegralRHF(("H", "H"), cache=tmp_path, **options)
    assert not list(tmp_path.iterdir())


def test_native_nonconvergence_is_not_a_gradient(tmp_path: Path) -> None:
    model = IntegralRHF(("H", "H"), cache=tmp_path, max_iterations=1)
    try:
        with pytest.raises(RuntimeError, match="did not converge"):
            model.value_and_grad(XYZ)
        assert model.diagnostics() == {}
    finally:
        model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.energy(XYZ)


def test_multicenter_p_shell_chain_and_basis_response(tmp_path: Path) -> None:
    basis = (
        Shell(0, 0, (Primitive(24.0, 0.8), Primitive(4.5, 0.4))),
        Shell(0, 0, (Primitive(0.7, 1.0),)),
        Shell(0, 1, (Primitive(1.1, 1.0),)),
        Shell(1, 0, (Primitive(0.8, 1.0),)),
        Shell(2, 0, (Primitive(0.8, 1.0),)),
    )
    model = IntegralRHF(
        ("O", "H", "H"), basis=basis, cache=tmp_path, compile_timeout=600
    )
    xyz = np.array([[0.0, 0.0, 0.0], [0.0, -1.43, 1.11], [0.0, 1.43, 1.11]])
    try:
        value, gradient = model.value_and_grad(xyz)
        assert value == pytest.approx(
            _oracle_energy(model, xyz, model.exponents, model.coefficients), abs=2e-9
        )
        rng = np.random.default_rng(892)
        directions = [
            rng.normal(size=xyz.shape),
            rng.normal(size=model.exponents.shape),
            rng.normal(size=model.coefficients.shape),
        ]
        expected = sum(
            np.sum(g * d)
            for g, d in zip(
                (gradient.coordinates, gradient.exponents, gradient.coefficients),
                directions,
                strict=True,
            )
        )
        step = 1e-4
        displaced = [
            _oracle_energy(
                model,
                xyz + sign * step * directions[0],
                model.exponents + sign * step * directions[1],
                model.coefficients + sign * step * directions[2],
            )
            for sign in (1, -1)
        ]
        assert (displaced[0] - displaced[1]) / (2 * step) == pytest.approx(
            expected, abs=3e-6, rel=3e-6
        )
        np.testing.assert_allclose(gradient.coordinates.sum(axis=0), 0, atol=2e-10)
    finally:
        model.close()


def test_signed_coefficients_and_immutable_topology(hydrogen: IntegralRHF) -> None:
    from dataclasses import FrozenInstanceError

    coefficients = hydrogen.coefficients.copy()
    coefficients[1] = -0.2
    value, gradient = hydrogen.value_and_grad(XYZ, coefficients=coefficients)
    assert value == pytest.approx(
        _oracle_energy(hydrogen, XYZ, hydrogen.exponents, coefficients), abs=3e-11
    )
    step = 2e-5
    direction = np.array([0.1, -0.2, 0.3, 0.4])
    energies = [
        _oracle_energy(
            hydrogen, XYZ, hydrogen.exponents, coefficients + sign * step * direction
        )
        for sign in (1, -1)
    ]
    assert (energies[0] - energies[1]) / (2 * step) == pytest.approx(
        gradient.coefficients @ direction, abs=2e-8
    )
    with pytest.raises(FrozenInstanceError):
        hydrogen.nao = 1


@pytest.mark.parametrize("domain", ["spherical", "ecp"])
def test_basis_metadata_rejects_before_shell_expansion(
    domain: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacing raw primitives must not discard Hamiltonian/representation metadata."""
    import json

    from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
    from generativeqc.profiles import canonical_hash

    potential = [
        {
            "ecp_type": "scalar_ecp",
            "angular_momentum": [0],
            "r_exponents": [2],
            "gaussian_exponents": ["1.0"],
            "coefficients": [["1.0"]],
        }
    ]
    shell = BasisShell(0, ("1.0",), (("1.0",),))
    element = ElementBasis(
        3,
        (shell,),
        ecp_core_electrons=2 if domain == "ecp" else 0,
        ecp_data=json.dumps(potential) if domain == "ecp" else None,
    )
    basis = BasisSet(
        "unsupported integral AD test domain",
        (element,),
        BasisProvenance("synthetic domain test", "1", "MIT", canonical_hash(potential)),
        representation="spherical" if domain == "spherical" else "cartesian",
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("unsupported basis reached code generation")

    monkeypatch.setattr(
        "generativeqc.experimental.integral_autodiff.emit_integral_program", forbidden
    )
    with pytest.raises(ValueError, match="Cartesian|all-electron"):
        IntegralRHF(("Li",), charge=1, basis=basis, cache=tmp_path)
    assert not list(tmp_path.iterdir())
