"""Public ECP dispatch boundaries, not numerical or installed qualification."""

from dataclasses import replace
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
import test_global_hybrid_ecp_basis as canonical_basis
from generativeqc import Calculator
from generativeqc.saddle import SaddleOptions, optimize_saddle
from generativeqc_compiler.method import resolve_method
from test_integral_autodiff_second import require_native_library  # noqa: F401


@pytest.fixture(scope="module")
def canonical_records() -> tuple:
    return canonical_basis.canonical_records.__wrapped__()


def _calculator(records: tuple, number: int, method: Any, **options: Any) -> Calculator:
    """Retain the canonical ECP and only the elements present in this case."""
    table = records[2]
    table = replace(
        table,
        elements=tuple(
            element
            for element in table.elements
            if element.atomic_number in (1, number)
        ),
    )
    return Calculator(
        method=method,
        basis=table,
        basis_representation="spherical",
        device="cpu",
        **options,
    )


@pytest.mark.parametrize("number", (53, 55))
@pytest.mark.parametrize("multiplicity", (1, 2))
@pytest.mark.parametrize("bj", (False, True))
@pytest.mark.parametrize("endpoint", ("hessian", "hvp"))
def test_canonical_ecp_dispatches_only_to_the_native_owner(
    canonical_records: tuple,
    monkeypatch: pytest.MonkeyPatch,
    number: int,
    multiplicity: int,
    bj: bool,
    endpoint: str,
) -> None:
    method = resolve_method(
        "B3LYP-D3(BJ)" if bj else "B3LYP",
        spin="unpolarized" if multiplicity == 1 else "polarized",
    )
    calculator = _calculator(canonical_records, number, method)
    assert calculator.second_order_capabilities == {"hessian", "hvp"}
    atoms = ((number, (0.1, -0.2, 0.3)), (1, (0.2, 0.1, 3.2)))
    direction = np.full((2, 3), 0.03)
    published = object()

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("dispatch control entered a Python response owner or SCF")

    def native(owner: Calculator, actual_atoms: tuple, **request: Any) -> object:
        assert owner is calculator
        assert tuple(atom.atomic_number for atom in actual_atoms) == (number, 1)
        assert request["charge"] == 1
        assert request["multiplicity"] == multiplicity
        assert request["max_eri_primitives"] == 300_000_000
        if endpoint == "hvp":
            np.testing.assert_array_equal(request["direction"], direction)
        else:
            assert request.get("direction") is None
        return published

    monkeypatch.setattr(calculator, "prepare_batch", forbidden)
    monkeypatch.setattr(calculator, "_run_public_rks_second_order", forbidden)
    monkeypatch.setattr("generativeqc._native_cpu_hessian.run_native_cpu", native)
    context = {
        "charge": 1,
        "multiplicity": multiplicity,
        "max_eri_primitives": 300_000_000,
    }
    if endpoint == "hessian":
        assert calculator.hessian(atoms, **context) is published
    else:
        assert (
            calculator.hessian_vector_product(atoms, direction, **context) is published
        )


@pytest.mark.parametrize("multiplicity", (1, 2))
@pytest.mark.parametrize("variant", ("pbe0", "changed-exchange", "df"))
def test_other_ecp_compositions_remain_refused_before_scf(
    canonical_records: tuple,
    monkeypatch: pytest.MonkeyPatch,
    multiplicity: int,
    variant: str,
) -> None:
    method = resolve_method(
        "PBE0" if variant == "pbe0" else "B3LYP",
        spin="unpolarized" if multiplicity == 1 else "polarized",
    )
    if variant == "changed-exchange":
        method = replace(
            method,
            primitives=(
                method.primitives[0],
                replace(method.primitives[1], coefficient=Fraction(37, 100)),
            ),
        )
    calculator = _calculator(
        canonical_records,
        53,
        method,
        density_fitting="cpu" if variant == "df" else "none",
    )
    assert not calculator.second_order_capabilities

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("unqualified ECP request started an SCF")

    monkeypatch.setattr(calculator, "prepare_batch", forbidden)
    atoms = ((53, (0, 0, 0)), (1, (0.1, 0.2, 3.1)))
    for request in (
        lambda: calculator.hessian(atoms, multiplicity=multiplicity),
        lambda: calculator.hessian_vector_product(
            atoms, np.ones((2, 3)), multiplicity=multiplicity
        ),
    ):
        with pytest.raises(NotImplementedError, match="does not expose public"):
            request()


@pytest.mark.parametrize("multiplicity", (1, 2))
def test_ecp_hessian_admission_does_not_promote_molecular_saddles(
    canonical_records: tuple, monkeypatch: pytest.MonkeyPatch, multiplicity: int
) -> None:
    calculator = _calculator(
        canonical_records,
        53,
        resolve_method(
            "B3LYP-D3(BJ)", spin="unpolarized" if multiplicity == 1 else "polarized"
        ),
    )
    assert calculator.second_order_capabilities == {"hessian", "hvp"}

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("ECP saddle promotion started an SCF")

    monkeypatch.setattr(calculator, "prepare_batch", forbidden)
    with pytest.raises(NotImplementedError, match="all-electron"):
        optimize_saddle(
            calculator,
            ((53, (0, 0, 0)), (1, (0.1, 0.2, 3.1))),
            masses=(126.90447, 1.00782503223),
            multiplicity=multiplicity,
            options=SaddleOptions(project_rigid_motions=True),
        )
