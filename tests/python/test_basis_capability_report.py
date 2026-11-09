"""Whole-route basis diagnostics without native execution or basis truncation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import (
    Atom,
    Calculator,
    Primitive,
    Shell,
    _native,
    basis_capability,
    import_bse,
)
from generativeqc.elements import SYMBOLS

ATOMS = (("C", (0.0, 0.0, 0.0)), ("N", (0.1, 0.2, 2.0)))


def configured_calculator(
    orbital: Any,
    auxiliary: Any = None,
    *,
    method: int = _native.METHOD_MP2,
    fitting: int = _native.DENSITY_FITTING_FULL_CUDA,
    backend: str = "cuda",
    linked_scf_auxiliary: bool = True,
) -> Calculator:
    """Provide a metadata-only execution context, never an executed GPU result.

    Bypass the native loader so the report and early singlepoint rejection can
    be tested even on hosts without a native library or a schedulable GPU.
    """
    calculator = Calculator.__new__(Calculator)
    calculator._basis = orbital
    calculator._auxiliary_basis = auxiliary
    calculator._method = method
    calculator._method_name = "metadata-only test"
    calculator._density_fitting_mode = fitting
    calculator._device_name = backend
    calculator._representation_name = "spherical"
    calculator._ks_options = None
    calculator._dispersion_method_ir = None
    calculator._singlepoint_context = None
    calculator._capabilities = SimpleNamespace(
        family="hartree_fock", supported_properties=frozenset(("energy", "forces"))
    )
    calculator._library = SimpleNamespace()
    if linked_scf_auxiliary:
        calculator._library.generativeqc_scf_auxiliary_system_create_v1 = object()
    return calculator


def shells(angular: int) -> tuple[Shell, ...]:
    """Retain a signed contraction on each atom, including its zero coefficient."""
    return tuple(
        Shell(
            index,
            angular,
            (Primitive(0.8, 0.7), Primitive(0.3, -0.2), Primitive(0.1, 0.0)),
        )
        for index in range(len(ATOMS))
    )


@pytest.mark.parametrize("order", (0, 1))
@pytest.mark.parametrize("operator", ("df_metric", "df_three_center"))
@pytest.mark.parametrize(
    "consumer,maximum", ((None, 3), ("correlation_df", 4), ("scf_df", 5))
)
def test_selected_auxiliary_boundary_is_explicit(
    order: int, operator: str, consumer: str | None, maximum: int
) -> None:
    report = basis_capability(
        shells(5),
        ATOMS,
        backend="cuda",
        role="auxiliary",
        operator=operator,
        derivative_order=order,
        consumer=consumer,
        representation="spherical",
    )
    assert report["consumer"] == consumer
    assert report["max_angular_momentum"] == 5
    assert report["max_supported_angular_momentum"] == maximum
    assert report["shells_checked"] == len(ATOMS)
    assert report["eligible"] == (maximum == 5)


def test_absent_route_and_absent_shells_have_distinct_bounds() -> None:
    report = basis_capability(shells(4), ATOMS, backend="cuda", derivative_order=2)
    assert not report["eligible"]
    assert report["max_supported_angular_momentum"] is None
    assert report["max_angular_momentum"] == 4
    empty = basis_capability((), ATOMS, backend="cuda")
    assert not empty["eligible"]
    assert empty["max_angular_momentum"] is None
    assert empty["max_supported_angular_momentum"] == 3


@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
def test_canonical_records_count_all_contractions_on_all_atoms(
    representation: str,
) -> None:
    path = Path(__file__).resolve().parents[1] / "data/external_basis/cc-pvtz-fe.json"
    basis = import_bse(
        path,
        source="pinned test data",
        source_version="4adaf1372c7101620ca1a9f3130be9ae97fb8f30",
        license="BSD-3-Clause",
        representation=representation,
    )
    original = basis.to_payload()
    atoms = (("Fe", (0.0, 0.0, 0.0)), ("Fe", (0.1, 0.2, 3.0)))
    report = configured_calculator(basis).basis_capability_report(atoms)
    assert not report["eligible"]
    for check in report["checks"]:
        assert check["shells_checked"] == 40
        assert check["max_angular_momentum"] == 4
        assert check["representation"] == representation
        assert any(
            "atom 0 Z=26" in reason and "l=4" in reason for reason in check["reasons"]
        )
        assert any(
            "atom 1 Z=26" in reason and "l=4" in reason for reason in check["reasons"]
        )
    assert basis.to_payload() == original


@pytest.mark.parametrize("compute_forces", (False, True))
@pytest.mark.parametrize("method", (_native.METHOD_MP2, _native.METHOD_DF_RCCSD_T))
def test_report_and_singlepoint_rejection_cover_the_same_complete_closure(
    compute_forces: bool,
    method: int,
) -> None:
    orbital, auxiliary = shells(4), shells(5)
    calculator = configured_calculator(orbital, auxiliary, method=method)
    report = calculator.basis_capability_report(
        iter(ATOMS), compute_forces=compute_forces
    )
    expected = {
        ("orbital", operator, order)
        for operator in (
            "overlap",
            "kinetic",
            "nuclear_attraction",
            "eri",
            "df_three_center",
        )
        for order in ((0, 1) if compute_forces else (0,))
    } | {
        ("auxiliary", operator, order)
        for operator in ("df_metric", "df_three_center")
        for order in ((0, 1) if compute_forces else (0,))
    }
    assert {
        (check["role"], check["operator"], check["derivative_order"])
        for check in report["checks"]
    } == expected
    assert not report["eligible"]
    for check in report["checks"]:
        assert not check["eligible"]
        assert check["shells_checked"] == len(ATOMS)
        if check["role"] == "auxiliary":
            assert check["consumer"] == "correlation_df"
            assert check["max_supported_angular_momentum"] == 4
        else:
            assert check["consumer"] is None
            assert check["max_supported_angular_momentum"] == 3
    properties = ("energy", "forces") if compute_forces else ("energy",)
    with pytest.raises(NotImplementedError) as failure:
        calculator.singlepoint(ATOMS, charge=-1, properties=properties)
    assert str(failure.value) == "; ".join(report["reasons"])
    assert calculator._basis == orbital
    assert calculator._auxiliary_basis == auxiliary


@pytest.mark.parametrize("linked,maximum", ((False, 3), (True, 5)))
def test_scf_auxiliary_closure_does_not_promote_orbitals_or_correlation(
    linked: bool,
    maximum: int,
) -> None:
    calculator = configured_calculator(
        shells(4),
        shells(5),
        method=_native.METHOD_RHF,
        fitting=_native.DENSITY_FITTING_CUDA,
        linked_scf_auxiliary=linked,
    )
    report = calculator.basis_capability_report(ATOMS)
    assert not report["eligible"]
    for check in report["checks"]:
        if check["role"] == "orbital":
            assert check["max_supported_angular_momentum"] == 3
            assert not check["eligible"]
        else:
            assert check["max_supported_angular_momentum"] == maximum
            assert check["eligible"] == linked
    correlation = configured_calculator(shells(3), shells(5)).basis_capability_report(
        ATOMS
    )
    assert not correlation["eligible"]
    assert all(
        check["eligible"]
        for check in correlation["checks"]
        if check["role"] == "orbital"
    )


def test_auxiliary_backend_and_energy_only_ao_jets_are_preserved() -> None:
    calculator = configured_calculator(
        shells(3),
        shells(4),
        fitting=_native.DENSITY_FITTING_CPU_REFERENCE,
    )
    calculator._capabilities.family = "density_functional"
    calculator._ks_options = SimpleNamespace(ao_order=1)
    report = calculator.basis_capability_report(ATOMS, compute_forces=False)
    assert report["eligible"]
    for check in report["checks"]:
        assert check["derivative_order"] == (1 if check["operator"] == "ao" else 0)
        assert check["backend"] == (
            "cpu" if check["operator"] in ("df_metric", "df_three_center") else "cuda"
        )
        assert check["consumer"] is None


def test_value_admission_does_not_infer_derivative_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc.basis_capabilities import LIMITS

    monkeypatch.setitem(LIMITS["cpu"]["overlap"], 1, 3)
    calculator = configured_calculator(
        shells(4),
        fitting=_native.DENSITY_FITTING_NONE,
        backend="cpu",
    )
    assert calculator.basis_capability_report(ATOMS, compute_forces=False)["eligible"]
    report = calculator.basis_capability_report(ATOMS, compute_forces=True)
    assert not report["eligible"]
    assert all("overlap derivative 1" in reason for reason in report["reasons"])


def test_reports_do_not_mutate_calculator_admission() -> None:
    calculator = configured_calculator(shells(3), shells(4))
    report = calculator.basis_capability_report(ATOMS)
    assert report["eligible"]
    report["checks"][0]["eligible"] = False
    report["checks"][0]["reasons"].append("caller-owned diagnostic")
    report["reasons"].append("caller-owned diagnostic")
    again = calculator.basis_capability_report(ATOMS)
    assert again["eligible"]
    assert not again["reasons"]
    assert all(check["eligible"] and not check["reasons"] for check in again["checks"])


def test_intrinsic_basis_is_not_reported_as_gaussian_qualification() -> None:
    calculator = configured_calculator(None, method=_native.METHOD_GFN2_XTB)
    report = calculator.basis_capability_report(ATOMS)
    assert not report["applicable"]
    assert report["eligible"]
    assert not report["checks"] and not report["reasons"]
    with pytest.raises(ValueError, match="at least one atom"):
        calculator.basis_capability_report(())


def standard_shells(name: str, atoms: Any) -> tuple[Shell, ...]:
    """Read the entire installed PySCF basis for metadata-only regression checks.

    This does not use PySCF in a production route or assert numerical endpoint
    parity. Every primitive and contraction is retained, including exact zeros.
    """
    gto = pytest.importorskip("pyscf.gto")
    result = []
    for index, atom in enumerate(atoms):
        for angular, *primitives in gto.basis.load(
            name, SYMBOLS[Atom.from_value(atom).atomic_number - 1]
        ):
            for column in range(1, len(primitives[0])):
                result.append(
                    Shell(
                        index,
                        angular,
                        tuple(
                            Primitive(float(primitive[0]), float(primitive[column]))
                            for primitive in primitives
                        ),
                    )
                )
    return tuple(result)


@pytest.mark.parametrize(
    "cardinal,orbital_max,auxiliary_max", (("t", 3, 4), ("q", 4, 5), ("5", 5, 6))
)
@pytest.mark.parametrize("method", (_native.METHOD_MP2, _native.METHOD_DF_RCCSD_T))
def test_complete_augmented_cn_families_are_inspected_without_truncation(
    cardinal: str,
    orbital_max: int,
    auxiliary_max: int,
    method: int,
) -> None:
    orbital = standard_shells(f"aug-cc-pv{cardinal}z", ATOMS)
    auxiliary = standard_shells(f"aug-cc-pv{cardinal}z-jkfit", ATOMS)
    report = configured_calculator(
        orbital, auxiliary, method=method
    ).basis_capability_report(ATOMS)
    assert report["eligible"] == (cardinal == "t")
    for check in report["checks"]:
        assert check["shells_checked"] == len(
            orbital if check["role"] == "orbital" else auxiliary
        )
        assert check["max_angular_momentum"] == (
            orbital_max if check["role"] == "orbital" else auxiliary_max
        )
    if cardinal == "q":
        assert sum(2 * shell.angular_momentum + 1 for shell in orbital) == 160
    if cardinal != "t":
        assert any(
            f"l={orbital_max}" in reason and "orbital" in reason
            for reason in report["reasons"]
        )
        assert any(
            f"l={auxiliary_max}" in reason and "auxiliary" in reason
            for reason in report["reasons"]
        )


@pytest.mark.parametrize("element", ("H", "C", "N", "O"))
@pytest.mark.parametrize("cardinal,heavy_max", (("q", 4), ("5", 5)))
def test_q_and_five_z_requirements_distinguish_hydrogen_and_heavy_atoms(
    element: str,
    cardinal: str,
    heavy_max: int,
) -> None:
    atoms = ((element, (0.0, 0.0, 0.0)),)
    orbital = standard_shells(f"aug-cc-pv{cardinal}z", atoms)
    auxiliary = standard_shells(f"aug-cc-pv{cardinal}z-jkfit", atoms)
    expected = heavy_max - (element == "H")
    assert max(shell.angular_momentum for shell in orbital) == expected
    assert max(shell.angular_momentum for shell in auxiliary) == expected + 1
