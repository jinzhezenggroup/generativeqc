"""Independent valence-Hamiltonian qualification of native ECP correlation/CBS."""

from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
import pytest
from generativeqc import Calculator, CBSBasis, CCSDTCBSCalculator
from test_cbs_open_shell_native import independent_components

ATOMS = (
    ("Na", (0.13, -0.21, 0.17)),
    ("H", (0.43, 0.19, 3.2)),
    ("H", (1.8, 0.29, 3.3)),
)
OPTIONS = {
    "initial_guess": None,
    "max_iterations": 300,
    "energy_tolerance": 1e-13,
    "density_tolerance": 1e-11,
    "ccsd_max_iterations": 200,
    "ccsd_energy_tolerance": 1e-13,
    "ccsd_residual_tolerance": 1e-11,
    "correlation_memory_budget_bytes": 1 << 30,
}


@pytest.fixture(scope="module")
def ecp_bases() -> tuple[CBSBasis, CBSBasis]:
    """Physical LANL2DZ Na potential with compact test-only valence AO records."""
    gto = pytest.importorskip("pyscf.gto")
    from test_ecp import _record_from_resolved_basis

    ecp = {"Na": gto.basis.load_ecp("lanl2dz", "Na")}
    records = []
    for cardinal, extra in ((2, []), (3, [[0, [0.04, 1.0]]])):
        basis = {
            "Na": [[0, [0.15, 1.0]], [1, [0.08, 1.0]], *extra],
            "H": [[0, [0.7, 1.0]]],
        }
        records.append(
            CBSBasis(_record_from_resolved_basis(basis, ecp, "spherical"), cardinal)
        )
    return tuple(records)


@pytest.fixture(params=("cpu", "cuda"))
def ecp_device(request: pytest.FixtureRequest) -> str:
    """Real-device tests require an explicit Slurm allocation and opt-in."""
    if request.param == "cuda":
        if os.environ.get("GENERATIVEQC_CBS_ECP_CUDA_TEST") != "1":
            pytest.skip("explicit Slurm ECP correlation qualification")
        assert os.environ.get("SLURM_JOB_ID"), "real GPU work must run through srun"
    return request.param


def _displaced(coordinate: int, step: float) -> tuple:
    """Move one physical center, including the center carrying the ECP."""
    atom_index, axis = divmod(coordinate, 3)
    atoms = [(symbol, list(position)) for symbol, position in ATOMS]
    atoms[atom_index][1][axis] += step
    return tuple(atoms)


@pytest.mark.parametrize("charge,multiplicity", ((0, 2), (1, 1), (1, 3)))
def test_ecp_analytic_components(
    ecp_bases: tuple[CBSBasis, CBSBasis],
    ecp_device: str,
    charge: int,
    multiplicity: int,
) -> None:
    """Conventional spin/reference/response sources match displaced libcint energies."""
    restricted = multiplicity == 1
    method = "ccsd(t)" if restricted else "uccsd(t)"
    hf, mp2 = ("rhf", "mp2") if restricted else ("uhf", "ump2")
    basis = ecp_bases[0].basis
    expected = independent_components(
        basis, charge, multiplicity, ATOMS, restricted=restricted
    )
    step = 2e-4
    finite_difference = {key: [] for key in (hf, mp2, method)}
    for coordinate in range(9):
        displaced = [
            independent_components(
                basis,
                charge,
                multiplicity,
                _displaced(coordinate, shift),
                restricted=restricted,
            )
            for shift in (-step, step)
        ]
        for key, values in finite_difference.items():
            values.append((displaced[0][key] - displaced[1][key]) / (2 * step))
    calculator = Calculator(method=method, basis=basis, device=ecp_device, **OPTIONS)
    try:
        companions = calculator.singlepoint_components(
            ATOMS,
            companions=(hf, mp2),
            charge=charge,
            multiplicity=multiplicity,
            properties=("energy", "forces"),
        )
        result = companions[method]
        energy_only = calculator.singlepoint(
            ATOMS, charge=charge, multiplicity=multiplicity, properties=("energy",)
        )
    finally:
        calculator.clear_cache()
    for key, value in ((method, result), (hf, companions[hf]), (mp2, companions[mp2])):
        assert value.energy == pytest.approx(expected[key], abs=3e-8)
        np.testing.assert_allclose(
            value.forces,
            np.array(finite_difference[key]).reshape(3, 3),
            atol=3e-7,
            rtol=0,
        )
        np.testing.assert_allclose(value.forces.sum(axis=0), 0, atol=3e-9, rtol=0)
        assert value.executed_backend == (
            "cuda" if ecp_device == "cuda" else "cpu_reference"
        )
    assert energy_only.forces is None
    assert energy_only.energy == pytest.approx(result.energy, abs=1e-11)
    calculator = Calculator(method=mp2, basis=basis, device=ecp_device, **OPTIONS)
    try:
        standalone = calculator.singlepoint(
            ATOMS,
            charge=charge,
            multiplicity=multiplicity,
            properties=("energy", "forces"),
        )
    finally:
        calculator.clear_cache()
    assert standalone.energy == pytest.approx(companions[mp2].energy, abs=2e-11)
    np.testing.assert_allclose(
        standalone.forces, companions[mp2].forces, atol=3e-9, rtol=0
    )


@pytest.mark.parametrize("multiplicity,charge", ((2, 0), (1, 1)))
@pytest.mark.parametrize("fitting", ("none", "correlation", "full"))
def test_complete_ecp_cbs(
    ecp_bases: tuple[CBSBasis, CBSBasis],
    ecp_device: str,
    multiplicity: int,
    charge: int,
    fitting: str,
) -> None:
    """Complete CBS uses the same ECP, valence counts and weights in every role."""
    restricted = multiplicity == 1
    if fitting != "none" and ecp_device != "cuda":
        pytest.skip("explicit fitted CBS auxiliary bases require CUDA")
    hf, mp2, cc = (
        ("rhf", "mp2", "ccsd(t)") if restricted else ("uhf", "ump2", "uccsd(t)")
    )
    auxiliary = ecp_bases[1].basis if fitting != "none" else None
    workflow = CCSDTCBSCalculator(
        ecp_bases,
        device=ecp_device,
        policy="fp64",
        calculator_options=OPTIONS,
        auxiliary_bases=(auxiliary, auxiliary) if auxiliary else None,
        hf_reference="df" if fitting == "full" else "exact",
    )

    def oracle(atoms: tuple) -> float:
        values = [
            independent_components(
                item.basis,
                charge,
                multiplicity,
                atoms,
                auxiliary,
                fitting == "full",
                restricted=restricted,
            )
            for item in ecp_bases
        ]
        return (
            sum(
                weight * value[hf]
                for weight, value in zip(workflow.hf_weights, values, strict=True)
            )
            + sum(
                weight * (value[mp2] - value[hf])
                for weight, value in zip(workflow.mp2_weights, values, strict=True)
            )
            + values[0][cc]
            - values[0][mp2]
        )

    result = workflow.singlepoint(ATOMS, charge=charge, multiplicity=multiplicity)
    assert result.energy == pytest.approx(oracle(ATOMS), abs=4e-8)
    step = 2e-4
    expected = [
        (oracle(_displaced(index, -step)) - oracle(_displaced(index, step)))
        / (2 * step)
        for index in range(9)
    ]
    np.testing.assert_allclose(
        result.forces, np.array(expected).reshape(3, 3), atol=4e-7, rtol=0
    )
    assert result.basis_metadata["ecp_core_electrons"] == (10, 0, 0)
    assert len(result.execution_groups) == 2


def test_cbs_rejects_changed_ecp_before_execution(
    ecp_bases: tuple[CBSBasis, CBSBasis],
) -> None:
    """Same valence count does not imply the same scalar ECP Hamiltonian."""
    low, high = ecp_bases
    element = high.basis.by_element[11]
    potentials = json.loads(element.ecp_data)
    potentials[0]["coefficients"][0][0] = "0.123"
    changed = replace(
        high.basis,
        elements=(
            replace(element, ecp_data=json.dumps(potentials)),
            high.basis.by_element[1],
        ),
    )
    workflow = CCSDTCBSCalculator((low, CBSBasis(changed, 3)))
    with pytest.raises(ValueError, match="same ECP Hamiltonian"):
        workflow.singlepoint(ATOMS, multiplicity=2)


@pytest.mark.parametrize("full_reference", (False, True))
@pytest.mark.parametrize("restricted", (False, True))
def test_fitted_ecp_analytic_components(
    ecp_bases: tuple[CBSBasis, CBSBasis],
    ecp_device: str,
    full_reference: bool,
    restricted: bool,
) -> None:
    """Differentiate the selected fitted interaction without changing the ECP."""
    if ecp_device != "cuda" and restricted:
        pytest.skip("restricted fitted CC requires CUDA")
    basis, auxiliary = (item.basis for item in ecp_bases)
    charge, multiplicity = (1, 1) if restricted else (0, 2)
    method = "df-ccsd(t)" if restricted else "uccsd(t)"
    hf, mp2, oracle_method = (
        ("rhf", "mp2", "ccsd(t)") if restricted else ("uhf", "ump2", "uccsd(t)")
    )
    values = independent_components(
        basis, charge, multiplicity, ATOMS, auxiliary, full_reference, restricted
    )
    step = 2e-4
    expected = {key: [] for key in (hf, mp2, oracle_method)}
    for coordinate in range(9):
        displaced = [
            independent_components(
                basis,
                charge,
                multiplicity,
                _displaced(coordinate, shift),
                auxiliary,
                full_reference,
                restricted,
            )
            for shift in (-step, step)
        ]
        for key, forces in expected.items():
            forces.append((displaced[0][key] - displaced[1][key]) / (2 * step))
    fitting_mode = "full_cuda" if full_reference else "correlation_cuda"
    if restricted and not full_reference:
        # Restricted DF-CC fits correlation intrinsically; CUDA keeps RHF exact.
        fitting_mode = "cuda"
    calculator = Calculator(
        method=method,
        basis=basis,
        auxiliary_basis=auxiliary,
        device=ecp_device,
        density_fitting=fitting_mode,
        **OPTIONS,
    )
    try:
        actual = calculator.singlepoint_components(
            ATOMS,
            companions=(hf, mp2),
            charge=charge,
            multiplicity=multiplicity,
            properties=("energy", "forces"),
        )
    finally:
        calculator.clear_cache()
    for selected, key in ((method, oracle_method), (hf, hf), (mp2, mp2)):
        assert actual[selected].energy == pytest.approx(values[key], abs=4e-8)
        np.testing.assert_allclose(
            actual[selected].forces,
            np.array(expected[key]).reshape(3, 3),
            atol=4e-7,
            rtol=0,
        )
        np.testing.assert_allclose(
            actual[selected].forces.sum(axis=0), 0, atol=3e-9, rtol=0
        )
        assert actual[selected].executed_backend == (
            "cuda" if ecp_device == "cuda" else "cpu_reference"
        )
    calculator = Calculator(
        method=mp2,
        basis=basis,
        auxiliary_basis=auxiliary,
        device=ecp_device,
        density_fitting="full_cuda" if full_reference else "correlation_cuda",
        **OPTIONS,
    )
    try:
        standalone = calculator.singlepoint(
            ATOMS,
            charge=charge,
            multiplicity=multiplicity,
            properties=("energy", "forces"),
        )
    finally:
        calculator.clear_cache()
    assert standalone.energy == pytest.approx(actual[mp2].energy, abs=3e-10)
    np.testing.assert_allclose(standalone.forces, actual[mp2].forces, atol=3e-9, rtol=0)


def test_ecp_resource_refusal_and_recovery(
    ecp_bases: tuple[CBSBasis, CBSBasis], ecp_device: str
) -> None:
    """Checked ECP staging belongs to the endpoint budget and refuses atomically."""
    options = {**OPTIONS, "correlation_memory_budget_bytes": 1 << 20}
    calculator = Calculator(
        method="uccsd(t)", basis=ecp_bases[0].basis, device=ecp_device, **options
    )
    try:
        with pytest.raises(RuntimeError, match="memory budget"):
            calculator.singlepoint_components(
                ATOMS,
                companions=("uhf", "ump2"),
                multiplicity=2,
                properties=("energy", "forces"),
            )
    finally:
        calculator.clear_cache()
    calculator = Calculator(
        method="uccsd(t)", basis=ecp_bases[0].basis, device=ecp_device, **OPTIONS
    )
    try:
        result = calculator.singlepoint(ATOMS, multiplicity=2, properties=("energy",))
    finally:
        calculator.clear_cache()
    assert result.converged and np.isfinite(result.energy)


@pytest.mark.parametrize("representation", ("spherical", "cartesian"))
def test_ecp_polarized_ccsd_force(
    ecp_bases: tuple[CBSBasis, CBSBasis], ecp_device: str, representation: str
) -> None:
    """ECP and spherical/cartesian d-shell transport also qualify the CCSD parent."""
    from generativeqc import BasisShell

    basis = ecp_bases[0].basis
    sodium = basis.by_element[11]
    basis = replace(
        basis,
        representation=representation,
        elements=(
            replace(
                sodium, shells=(*sodium.shells, BasisShell(2, ("0.1",), (("1",),)))
            ),
            basis.by_element[1],
        ),
    )
    expected = independent_components(basis, 0, 2, ATOMS)
    assert abs(expected["triples"]) > 1e-8
    calculator = Calculator(method="uccsd", basis=basis, device=ecp_device, **OPTIONS)
    try:
        result = calculator.singlepoint(
            ATOMS, multiplicity=2, properties=("energy", "forces")
        )
    finally:
        calculator.clear_cache()
    assert result.energy == pytest.approx(expected["uccsd"], abs=4e-8)
    step = 2e-4
    for coordinate in (0, 4, 8):
        displaced = [
            independent_components(basis, 0, 2, _displaced(coordinate, shift))["uccsd"]
            for shift in (-step, step)
        ]
        assert result.forces.flat[coordinate] == pytest.approx(
            (displaced[0] - displaced[1]) / (2 * step), abs=4e-7
        )
    np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=3e-9, rtol=0)
