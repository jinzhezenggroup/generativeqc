"""Complete open-shell CBS energies against source-matched independent PySCF."""

from __future__ import annotations

import builtins
import ctypes
import math
import os
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc import Atom, Calculator, CBSBasis, CCSDTCBSCalculator, _native
from generativeqc.basis import BasisShell, ElementBasis

if TYPE_CHECKING:
    from generativeqc import BasisSet


ATOMS = (
    ("H", (0.0, 0.0, 0.0)),
    ("H", (1.4, 0.0, 0.2)),
    ("H", (-0.4, 1.8, 0.3)),
)


@pytest.fixture(scope="module")
def compact_bases() -> tuple[CBSBasis, CBSBasis]:
    """Small s-only records qualify the algebra, not a physical CBS family."""
    template = CBSBasis("sto-3g", 2).basis
    records = []
    for cardinal, exponents in ((2, (0.9, 0.25)), (3, (0.9, 0.25, 0.07))):
        element = replace(
            template.by_element[1],
            shells=tuple(
                BasisShell(0, (str(exponent),), (("1",),)) for exponent in exponents
            ),
        )
        records.append(
            CBSBasis(
                replace(
                    template, name=f"cbs-open-shell-{cardinal}", elements=(element,)
                ),
                cardinal,
            )
        )
    return tuple(records)


def independent_components(
    basis: BasisSet,
    charge: int,
    multiplicity: int,
    atoms: tuple = ATOMS,
    auxiliary: BasisSet | None = None,
    full_reference: bool = False,
) -> dict:
    """Feed canonical input shells, never native orbitals/intermediates, to PySCF."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import cc, mp

    labels = [f"{atom[0]}{index}" for index, atom in enumerate(atoms)]
    matched = {label: [] for label in labels}
    for shell in basis.shells_for(tuple(Atom.from_value(atom) for atom in atoms)):
        matched[labels[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *[
                    [primitive.exponent, primitive.coefficient]
                    for primitive in shell.primitives
                ],
            ]
        )
    molecule = pyscf.gto.M(
        atom=[(label, atom[1]) for label, atom in zip(labels, atoms, strict=True)],
        basis=matched,
        charge=charge,
        spin=multiplicity - 1,
        unit="Bohr",
        cart=basis.representation == "cartesian",
        verbose=0,
    )
    reference = pyscf.scf.UHF(molecule)
    fitted_eri = None
    if auxiliary is not None:
        from pyscf import ao2mo, df

        auxiliary_shells = {label: [] for label in labels}
        for shell in auxiliary.shells_for(
            tuple(Atom.from_value(atom) for atom in atoms)
        ):
            auxiliary_shells[labels[shell.atom_index]].append(
                [
                    shell.angular_momentum,
                    *[
                        [primitive.exponent, primitive.coefficient]
                        for primitive in shell.primitives
                    ],
                ]
            )
        auxmol = df.addons.make_auxmol(molecule, auxiliary_shells)
        metric = auxmol.intor("int2c2e")
        raw = df.incore.aux_e2(molecule, auxmol, aosym="s1")
        values, vectors = np.linalg.eigh(metric)
        retained = values > values[-1] * 1e-10
        inverse = (vectors[:, retained] / values[retained]) @ vectors[:, retained].T
        fitted_eri = np.einsum("mnP,PQ,klQ->mnkl", raw, inverse, raw)
        if full_reference:
            reference._eri = ao2mo.restore(8, fitted_eri, molecule.nao)
    reference.conv_tol, reference.conv_tol_grad, reference.max_cycle = 1e-13, 1e-11, 300
    reference.kernel()
    assert reference.converged
    if fitted_eri is not None:
        if not full_reference:
            from pyscf.scf.hf import dot_eri_dm

            exact_eri = reference._eri.copy()

            def exact_jk(
                mol: Any = None, dm: Any = None, hermi: int = 1, **kwargs: Any
            ) -> tuple[np.ndarray, np.ndarray]:
                return dot_eri_dm(exact_eri, dm, hermi)

            reference.get_jk = exact_jk
        reference._eri = ao2mo.restore(8, fitted_eri, molecule.nao)
    perturbation = mp.UMP2(reference).run()
    coupled = cc.UCCSD(reference)
    coupled.conv_tol, coupled.conv_tol_normt, coupled.max_cycle = 1e-13, 1e-11, 200
    coupled.kernel()
    assert coupled.converged
    triples = coupled.ccsd_t()
    return {
        "uhf": reference.e_tot,
        "ump2": perturbation.e_tot,
        "uccsd": coupled.e_tot,
        "uccsd(t)": coupled.e_tot + triples,
        "triples": triples,
    }


@pytest.mark.parametrize("full_reference", (False, True))
@pytest.mark.parametrize("charge,multiplicity", ((0, 2), (1, 3)))
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_open_shell_df_analytic_components(
    compact_bases: tuple[CBSBasis, CBSBasis],
    full_reference: bool,
    charge: int,
    multiplicity: int,
    device: str,
) -> None:
    """Independently fit raw Coulomb integrals and reconverge displaced energies."""
    if device == "cuda" and os.environ.get("GENERATIVEQC_OPEN_SHELL_CUDA_TEST") != "1":
        pytest.skip("real CUDA endpoint qualification must run under Slurm")
    basis, auxiliary = compact_bases[0].basis, compact_bases[1].basis
    options = {
        "method": "uccsd(t)",
        "device": device,
        "basis": basis,
        "auxiliary_basis": auxiliary,
        "density_fitting": "full_cuda" if full_reference else "correlation_cuda",
        "initial_guess": None,
        "max_iterations": 300,
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "ccsd_max_iterations": 200,
        "ccsd_energy_tolerance": 1e-13,
        "ccsd_residual_tolerance": 1e-11,
        "correlation_memory_budget_bytes": 1 << 30,
    }
    calculator = Calculator(**options)
    try:
        actual = calculator.singlepoint_components(
            ATOMS,
            companions=("uhf", "ump2"),
            charge=charge,
            multiplicity=multiplicity,
            properties=("energy", "forces"),
        )
    finally:
        calculator.clear_cache()
    expected = independent_components(
        basis, charge, multiplicity, auxiliary=auxiliary, full_reference=full_reference
    )
    for method, result in actual.items():
        assert result.energy == pytest.approx(expected[method], abs=1e-8)
    standalone = Calculator(**{**options, "method": "ump2"})
    try:
        perturbation = standalone.singlepoint(
            ATOMS,
            charge=charge,
            multiplicity=multiplicity,
            properties=("energy", "forces"),
        )
    finally:
        standalone.clear_cache()
    assert perturbation.energy == pytest.approx(actual["ump2"].energy, abs=1e-10)
    np.testing.assert_allclose(
        perturbation.forces, actual["ump2"].forces, atol=2e-9, rtol=0
    )
    step = 2e-4
    for atom_index, axis in ((0, 0), (1, 2), (2, 1)):
        displaced = []
        for direction in (1, -1):
            atoms = [(symbol, list(position)) for symbol, position in ATOMS]
            atoms[atom_index][1][axis] += direction * step
            displaced.append(
                independent_components(
                    basis, charge, multiplicity, tuple(atoms), auxiliary, full_reference
                )
            )
        for method, result in actual.items():
            force = -(displaced[0][method] - displaced[1][method]) / (2 * step)
            assert result.forces[atom_index][axis] == pytest.approx(force, abs=3e-7)
            np.testing.assert_allclose(
                np.sum(result.forces, axis=0), 0, atol=2e-9, rtol=0
            )


@pytest.mark.parametrize("method", ("uccsd", "uccsd(t)"))
@pytest.mark.parametrize("charge,multiplicity", ((0, 2), (1, 3)))
def test_open_shell_analytic_force_against_independent_energy_differences(
    compact_bases: tuple[CBSBasis, CBSBasis],
    method: str,
    charge: int,
    multiplicity: int,
) -> None:
    """Re-converge the independent reference/CC at each displaced geometry.

    The doublet has a nonzero (T) correction; the triplet exercises an empty
    beta occupied space. These are analytic production forces, not a Python
    finite-difference execution path.
    """
    basis = compact_bases[0].basis
    calculator = Calculator(
        method=method,
        device="cpu",
        basis=basis,
        initial_guess=None,
        max_iterations=300,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        ccsd_max_iterations=200,
        ccsd_energy_tolerance=1e-13,
        ccsd_residual_tolerance=1e-11,
    )
    actual = calculator.singlepoint(
        ATOMS, charge=charge, multiplicity=multiplicity, properties=("energy", "forces")
    )
    np.testing.assert_allclose(np.sum(actual.forces, axis=0), 0, atol=2e-9, rtol=0)
    step = 2e-4
    for atom, axis in ((0, 0), (1, 1), (2, 2)):
        energies = []
        for sign in (-1, 1):
            displaced = [(symbol, list(position)) for symbol, position in ATOMS]
            displaced[atom][1][axis] += sign * step
            energies.append(
                independent_components(basis, charge, multiplicity, tuple(displaced))[
                    method
                ]
            )
        expected = -(energies[1] - energies[0]) / (2 * step)
        np.testing.assert_allclose(
            actual.forces[atom, axis], expected, atol=2e-7, rtol=0
        )
    assert actual.correlation.force_provenance_flags == 7


def test_complete_open_shell_cbs_analytic_force(
    compact_bases: tuple[CBSBasis, CBSBasis],
) -> None:
    """Complete shared-reference force groups obey the independently weighted energy."""
    calculator = CCSDTCBSCalculator(
        compact_bases,
        device="cpu",
        calculator_options={
            "initial_guess": None,
            "max_iterations": 300,
            "energy_tolerance": 1e-13,
            "density_tolerance": 1e-11,
            "ccsd_max_iterations": 200,
            "ccsd_energy_tolerance": 1e-13,
            "ccsd_residual_tolerance": 1e-11,
        },
    )
    actual = calculator.singlepoint(ATOMS, multiplicity=2)
    step = 2e-4
    energies = []
    for sign in (-1, 1):
        displaced = [(symbol, list(position)) for symbol, position in ATOMS]
        displaced[1][1][0] += sign * step
        components = [
            independent_components(record.basis, 0, 2, tuple(displaced))
            for record in compact_bases
        ]
        hf_ratio = math.exp(1.63)
        hf = (hf_ratio * components[1]["uhf"] - components[0]["uhf"]) / (hf_ratio - 1)
        mp2 = sum(
            weight * (values["ump2"] - values["uhf"])
            for weight, values in zip((-8 / 19, 27 / 19), components, strict=True)
        )
        energies.append(hf + mp2 + components[0]["uccsd(t)"] - components[0]["ump2"])
    np.testing.assert_allclose(
        actual.forces[1, 0],
        -(energies[1] - energies[0]) / (2 * step),
        atol=3e-7,
        rtol=0,
    )
    np.testing.assert_allclose(actual.forces.sum(axis=0), 0, atol=2e-9, rtol=0)


@pytest.mark.parametrize("density_fitting", ("none", "correlation_cuda", "full_cuda"))
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_open_shell_force_above_twelve_aos(density_fitting: str, device: str) -> None:
    """A charged 14-AO LiH doublet tests nonzero forces and d-shell response."""
    if device == "cuda" and os.environ.get("GENERATIVEQC_OPEN_SHELL_CUDA_TEST") != "1":
        pytest.skip("real CUDA endpoint qualification must run under Slurm")
    atoms = (("Li", (0.0, 0.0, 0.0)), ("H", (0.2, 0.0, 3.0)))
    template = CBSBasis("sto-3g", 2).basis
    lithium = replace(
        template.by_element[3],
        shells=tuple(
            BasisShell(angular, (str(exponent),), (("1",),))
            for angular, exponent in ((0, 2.3), (0, 0.4), (1, 0.8), (2, 0.7), (1, 0.16))
        ),
    )
    hydrogen = replace(
        template.by_element[1], shells=(BasisShell(0, ("0.5",), (("1",),)),)
    )
    basis = replace(
        template,
        name="open-shell-force-14-ao",
        elements=(hydrogen, lithium),
        representation="spherical",
    )
    calculator = Calculator(
        method="uccsd(t)",
        device=device,
        basis=basis,
        initial_guess=None,
        density_fitting=density_fitting,
        auxiliary_basis=basis if density_fitting != "none" else None,
        correlation_memory_budget_bytes=1 << 30,
        max_iterations=300,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        ccsd_max_iterations=200,
        ccsd_energy_tolerance=1e-13,
        ccsd_residual_tolerance=1e-11,
    )
    try:
        actual = calculator.singlepoint(
            atoms, charge=1, multiplicity=2, properties=("energy", "forces")
        )
    finally:
        calculator.clear_cache()
    step = 2e-4
    energies = []
    for sign in (-1, 1):
        displaced = [(symbol, list(position)) for symbol, position in atoms]
        displaced[1][1][2] += sign * step
        energies.append(
            independent_components(
                basis,
                1,
                2,
                tuple(displaced),
                auxiliary=basis if density_fitting != "none" else None,
                full_reference=density_fitting == "full_cuda",
            )["uccsd(t)"]
        )
    assert abs(actual.forces[1, 2]) > 1e-4
    np.testing.assert_allclose(
        actual.forces[1, 2],
        -(energies[1] - energies[0]) / (2 * step),
        atol=3e-7,
        rtol=0,
    )
    np.testing.assert_allclose(actual.forces.sum(axis=0), 0, atol=2e-9, rtol=0)


@pytest.mark.parametrize("charge,multiplicity", ((0, 2), (1, 3)))
@pytest.mark.parametrize("policy", ("standard", "tight"))
def test_complete_open_shell_cbs_against_independent_oracle(
    compact_bases: tuple[CBSBasis, CBSBasis],
    monkeypatch: pytest.MonkeyPatch,
    charge: int,
    multiplicity: int,
    policy: str,
) -> None:
    """Qualify neutral doublets and ionic, empty-beta triplets without fallback."""
    references = {
        record.basis.identity: independent_components(
            record.basis, charge, multiplicity
        )
        for record in compact_bases
    }
    original_import = builtins.__import__

    def guarded_import(name: str, *arguments: Any, **keywords: Any) -> Any:
        if name.startswith(
            ("pyscf", "tools.generativeqc_cc", "generativeqc_compiler.cc")
        ):
            raise AssertionError(f"production CBS imported a reference owner: {name}")
        return original_import(name, *arguments, **keywords)

    workflow = CCSDTCBSCalculator(compact_bases, policy=policy)
    with monkeypatch.context() as guarded:
        guarded.setattr(builtins, "__import__", guarded_import)
        result = workflow.singlepoint(
            ATOMS, charge=charge, multiplicity=multiplicity, properties=("energy",)
        )
    low, high = [references[record.basis.identity] for record in compact_bases]
    if multiplicity == 2:
        assert abs(low["triples"]) > 1e-6
    ratio = math.exp(-1.63)
    hf_weights = (-ratio / (1 - ratio), 1 / (1 - ratio))
    mp2_weights = (-8 / 19, 27 / 19)
    expected_hf = sum(
        weight * reference["uhf"]
        for weight, reference in zip(hf_weights, (low, high), strict=True)
    )
    expected_mp2 = sum(
        weight * (reference["ump2"] - reference["uhf"])
        for weight, reference in zip(mp2_weights, (low, high), strict=True)
    )
    expected_delta = low["uccsd(t)"] - low["ump2"]
    tolerance = 1e-7 if policy == "standard" else 1e-8
    assert result.hf_energy == pytest.approx(expected_hf, abs=tolerance)
    assert result.mp2_correlation_energy == pytest.approx(expected_mp2, abs=tolerance)
    assert result.delta_ccsdt_energy == pytest.approx(expected_delta, abs=tolerance)
    assert result.energy == pytest.approx(
        expected_hf + expected_mp2 + expected_delta, abs=tolerance
    )
    assert result.converged and result.executed_backend == "cpu_reference"
    assert result.forces is None
    for component in result.components:
        assert component.result.energy == pytest.approx(
            references[component.basis_identity][component.method], abs=tolerance
        )
        assert component.result.forces is None
        if component.method != "uhf":
            assert component.result.correlation.reference_energy == next(
                other.result.energy
                for other in result.components
                if other.basis_identity == component.basis_identity
                and other.method == "uhf"
            )


@pytest.mark.parametrize("method", ("ump2", "uccsd(t)"))
def test_unrestricted_default_companion_and_spin_identity(method: str) -> None:
    """Default companions follow the owner's spin; restricted names are refused."""
    calculator = Calculator(method=method, basis="sto-3g", initial_guess=None)
    try:
        results = calculator.singlepoint_components(
            [("Li", (0.0, 0.0, 0.0))], multiplicity=2, properties=("energy",)
        )
        assert set(results) == {method, "uhf"}
        assert results[method].correlation.reference_energy == results["uhf"].energy
        with pytest.raises(ValueError, match="companion"):
            calculator.singlepoint_components(
                ATOMS, companions=("rhf",), multiplicity=2
            )
    finally:
        calculator.clear_cache()


@pytest.mark.parametrize(
    "options,error",
    (
        ({"ccsd_max_iterations": 1}, RuntimeError),
        ({"mp2_denominator_threshold": 100.0}, RuntimeError),
    ),
)
def test_unrestricted_failed_group_publishes_nothing(
    compact_bases: tuple[CBSBasis, CBSBasis],
    monkeypatch: pytest.MonkeyPatch,
    options: dict,
    error: type[Exception],
) -> None:
    """A failed CC or MP2 companion must not export even a successful UHF subset."""
    calculator = Calculator(method="uccsd(t)", basis=compact_bases[0].basis, **options)
    library = calculator._library
    native_execute = library.generativeqc_calculation_execute_components_v1
    observed = []

    def execute(owner: object, *outputs: object) -> int:
        descriptors = [
            ctypes.cast(value, ctypes.POINTER(_native.ResultDescriptor)).contents
            for value in outputs
        ]
        for descriptor in descriptors:
            descriptor.energy = 123.0
        status = native_execute(owner, *outputs)
        assert status != _native.STATUS_SUCCESS
        assert all(descriptor.energy == 123.0 for descriptor in descriptors)
        assert (
            library.generativeqc_calculation_get_mp2_component_diagnostic_v1(
                owner, None
            )
            == _native.STATUS_NOT_IMPLEMENTED
        )
        observed.append(status)
        return status

    with monkeypatch.context() as wrapped:
        wrapped.setattr(
            library, "generativeqc_calculation_execute_components_v1", execute
        )
        try:
            with pytest.raises(error):
                calculator.singlepoint_components(
                    ATOMS,
                    companions=("uhf", "ump2"),
                    multiplicity=2,
                    properties=("energy",),
                )
        finally:
            calculator.clear_cache()
    assert observed
    recovered = CCSDTCBSCalculator(compact_bases).singlepoint(
        ATOMS, multiplicity=2, properties=("energy",)
    )
    assert recovered.converged and np.isfinite(recovered.energy)


def test_ump2_companions_publish_matching_uhf_forces() -> None:
    """Companion forces complete on the same physical unrestricted reference."""
    calculator = Calculator(method="ump2", basis="sto-3g", initial_guess=None)
    try:
        forced = calculator.singlepoint_components(
            [("Li", (0.0, 0.0, 0.0))],
            multiplicity=2,
            properties=("energy", "forces"),
        )
        assert set(forced) == {"ump2", "uhf"}
        for result in forced.values():
            np.testing.assert_allclose(result.forces, 0, atol=1e-10, rtol=0)
        results = calculator.singlepoint_components(
            [("Li", (0.0, 0.0, 0.0))], multiplicity=2, properties=("energy",)
        )
        assert all(
            result.converged and result.forces is None for result in results.values()
        )
    finally:
        calculator.clear_cache()


def test_open_shell_force_admission_failure_preserves_energy_replay(
    compact_bases: tuple[CBSBasis, CBSBasis],
) -> None:
    """An energy-sized reservation cannot publish an unadmitted force endpoint."""
    options = {
        "method": "uccsd(t)",
        "basis": compact_bases[0].basis,
        "initial_guess": None,
        "ccsd_max_iterations": 200,
    }
    baseline = Calculator(**options)
    try:
        expected = baseline.singlepoint(ATOMS, multiplicity=2)
    finally:
        baseline.clear_cache()
    bounded = Calculator(
        **options,
        correlation_memory_budget_bytes=expected.correlation.numeric_capacity_bytes,
    )
    try:
        before = bounded.singlepoint(ATOMS, multiplicity=2)
        with pytest.raises(RuntimeError, match="budget"):
            bounded.singlepoint(ATOMS, multiplicity=2, properties=("energy", "forces"))
        recovered = bounded.singlepoint(ATOMS, multiplicity=2)
        assert before.energy == pytest.approx(recovered.energy, abs=1e-13)
        assert recovered.forces is None and recovered.converged
    finally:
        bounded.clear_cache()


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_OPEN_SHELL_CUDA_TEST") != "1",
    reason="real CUDA endpoint qualification must run under Slurm",
)
@pytest.mark.parametrize("charge,multiplicity", ((0, 2), (1, 3)))
def test_open_shell_cuda_complete_cbs_force_parity(
    compact_bases: tuple[CBSBasis, CBSBasis],
    charge: int,
    multiplicity: int,
) -> None:
    """Qualify every device component, unequal populations and empty-beta paths."""
    options = {
        "initial_guess": None,
        "max_iterations": 300,
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "ccsd_max_iterations": 200,
        "ccsd_energy_tolerance": 1e-13,
        "ccsd_residual_tolerance": 1e-11,
        "correlation_memory_budget_bytes": 1 << 30,
    }
    expected = CCSDTCBSCalculator(
        compact_bases, device="cpu", calculator_options=options
    ).singlepoint(ATOMS, charge=charge, multiplicity=multiplicity)
    actual = CCSDTCBSCalculator(
        compact_bases, device="cuda", calculator_options=options
    ).singlepoint(ATOMS, charge=charge, multiplicity=multiplicity)
    np.testing.assert_allclose(actual.energy, expected.energy, atol=1e-9, rtol=0)
    np.testing.assert_allclose(actual.forces, expected.forces, atol=2e-8, rtol=0)
    for component in actual.components:
        assert component.result.executed_backend == "cuda"
        assert component.result.forces is not None


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_OPEN_SHELL_CUDA_TEST") != "1",
    reason="real CUDA endpoint qualification must run under Slurm",
)
def test_open_shell_cuda_convergence_failure_recovery(
    compact_bases: tuple[CBSBasis, CBSBasis],
) -> None:
    """Release device graph owners after an unpublished, unconverged CC parent."""
    options = {
        "method": "uccsd(t)",
        "device": "cuda",
        "basis": compact_bases[0].basis,
        "initial_guess": None,
        "correlation_memory_budget_bytes": 1 << 30,
        "ccsd_max_iterations": 1,
    }
    failing = Calculator(**options)
    try:
        for _attempt in range(2):
            with pytest.raises(RuntimeError, match="converg"):
                failing.singlepoint(
                    ATOMS, multiplicity=2, properties=("energy", "forces")
                )
    finally:
        failing.clear_cache()
    recovered = Calculator(**{**options, "ccsd_max_iterations": 200})
    try:
        result = recovered.singlepoint(
            ATOMS, multiplicity=2, properties=("energy", "forces")
        )
        assert result.converged and np.isfinite(result.forces).all()
        assert result.executed_backend == "cuda"
    finally:
        recovered.clear_cache()


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_OPEN_SHELL_CUDA_TEST") != "1",
    reason="real CUDA endpoint qualification must run under Slurm",
)
@pytest.mark.parametrize("hf_reference", ("exact", "df"))
def test_open_shell_cuda_df_complete_cbs_force(
    compact_bases: tuple[CBSBasis, CBSBasis],
    hf_reference: str,
) -> None:
    """Qualify the complete fitted composite against independently fitted energies."""
    auxiliary = compact_bases[1].basis
    workflow = CCSDTCBSCalculator(
        compact_bases,
        device="cuda",
        auxiliary_bases=(auxiliary, auxiliary),
        hf_reference=hf_reference,
        policy="tight",
        calculator_options={
            "initial_guess": None,
            "max_iterations": 300,
            "energy_tolerance": 1e-13,
            "density_tolerance": 1e-11,
            "ccsd_max_iterations": 200,
            "ccsd_energy_tolerance": 1e-13,
            "ccsd_residual_tolerance": 1e-11,
            "correlation_memory_budget_bytes": 1 << 30,
        },
    )
    actual = workflow.singlepoint(ATOMS, multiplicity=2)

    def composite(atoms: tuple) -> float:
        values = [
            independent_components(
                record.basis, 0, 2, atoms, auxiliary, hf_reference == "df"
            )
            for record in compact_bases
        ]
        ratio = math.exp(-1.63)
        return (
            (values[1]["uhf"] - ratio * values[0]["uhf"]) / (1 - ratio)
            + sum(
                weight * (entry["ump2"] - entry["uhf"])
                for weight, entry in zip((-8 / 19, 27 / 19), values, strict=True)
            )
            + values[0]["uccsd(t)"]
            - values[0]["ump2"]
        )

    assert actual.energy == pytest.approx(composite(ATOMS), abs=1e-8)
    step = 2e-4
    energies = []
    for sign in (-1, 1):
        atoms = [(symbol, list(position)) for symbol, position in ATOMS]
        atoms[1][1][0] += sign * step
        energies.append(composite(tuple(atoms)))
    assert actual.forces[1, 0] == pytest.approx(
        -(energies[1] - energies[0]) / (2 * step), abs=3e-7
    )
    assert all(
        component.result.executed_backend == "cuda" for component in actual.components
    )


def test_open_shell_cbs_correction_above_twelve_aos() -> None:
    """Exercise a nonzero-correlation Li doublet through the complete public CBS."""
    template = CBSBasis("sto-3g", 2).basis
    shells = tuple(
        BasisShell(angular, (str(exponent),), (("1",),))
        for angular, exponent in ((0, 2.3), (0, 0.4), (1, 0.8), (2, 0.7), (1, 0.16))
    )
    low = replace(template, name="li-13-ao", elements=(ElementBasis(3, shells),))
    high = replace(
        low,
        name="li-14-ao",
        elements=(ElementBasis(3, (*shells, BasisShell(0, ("0.08",), (("1",),)))),),
    )
    atoms = (("Li", (0.0, 0.0, 0.0)),)
    low_reference = independent_components(low, 0, 2, atoms)
    high_reference = independent_components(high, 0, 2, atoms)
    assert abs(low_reference["uccsd(t)"] - low_reference["uhf"]) > 1e-5
    workflow = CCSDTCBSCalculator((CBSBasis(low, 2), CBSBasis(high, 3)), policy="tight")
    result = workflow.singlepoint(atoms, multiplicity=2, properties=("energy",))
    ratio = math.exp(-1.63)
    expected = (
        (-ratio * low_reference["uhf"] + high_reference["uhf"]) / (1 - ratio)
        - (8 / 19) * (low_reference["ump2"] - low_reference["uhf"])
        + (27 / 19) * (high_reference["ump2"] - high_reference["uhf"])
        + low_reference["uccsd(t)"]
        - low_reference["ump2"]
    )
    assert result.converged and result.executed_backend == "cpu_reference"
    assert result.energy == pytest.approx(expected, abs=1e-8)
