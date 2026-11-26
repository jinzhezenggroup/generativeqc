"""Public ECP dispatch boundaries, not numerical or installed qualification."""

import json
from dataclasses import replace
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
import test_global_hybrid_ecp_basis as canonical_basis
from generativeqc import Calculator, GridSpec, HostResources, KsOptions
from generativeqc._native_cpu_hessian import linked_method
from generativeqc.saddle import RksColdFockDamping, SaddleOptions, optimize_saddle
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
        assert request["max_map_terms"] == 2_000_000_000_000
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
        "max_map_terms": 2_000_000_000_000,
    }
    if endpoint == "hessian":
        assert calculator.hessian(atoms, **context) is published
    else:
        assert (
            calculator.hessian_vector_product(atoms, direction, **context) is published
        )


def test_canonical_ecp_hvp_shared_host_lease(canonical_records: tuple) -> None:
    calculator = _calculator(
        canonical_records,
        55,
        resolve_method("B3LYP-D3(BJ)"),
        ks_options=KsOptions(grid=GridSpec(12, 4, 8)),
        initial_guess=None,
    )
    if not linked_method(calculator._library, calculator.method_ir.identity):
        pytest.skip("linked CPU Hessian is unavailable")
    atoms = ((55, (0.03, -0.02, 0.01)), (1, (0.17, 0.11, 3.4)))
    directions = np.array([[[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]]])
    with calculator.prepare_batch([atoms]) as reference_batch:
        reference_batch.execute(strict=True, properties=("energy",))
        reference = reference_batch.hessian_vector_products(0, directions)
    floor = reference.diagnostics["minimum_numeric_bytes"]

    def prepare(pool: HostResources) -> Any:
        return calculator.prepare_batch(
            [atoms], host_resources=pool, host_hessian_workspace_bytes=floor
        )

    with HostResources(1 << 31) as pool:
        with prepare(pool):
            reserved = pool.snapshot()
            assert reserved.owner_count == 1
        assert pool.snapshot().owner_count == 0
    exact = reserved.retained_bytes + reserved.workspace_bytes
    with HostResources(exact - 1) as pool:
        with pytest.raises(MemoryError, match="shared host capacity"):
            prepare(pool)
        assert pool.snapshot().owner_count == 0
    with HostResources(exact) as pool:
        with prepare(pool) as batch:
            before = pool.snapshot()
            with pytest.raises(MemoryError, match="shared host capacity"):
                prepare(pool)
            assert pool.snapshot() == before
            batch.execute(strict=True)
            value = batch.hessian_vector_products(0, directions)
            np.testing.assert_array_equal(value.value, reference.value)
            assert value.diagnostics["scf_fock_builds"] == 0
            assert (
                value.diagnostics["d3_pair_visits"]
                == reference.diagnostics["d3_pair_visits"]
            )
        assert pool.snapshot().owner_count == 0


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
    with pytest.raises(NotImplementedError, match="Cs46 def2-ECP"):
        optimize_saddle(
            calculator,
            ((53, (0, 0, 0)), (1, (0.1, 0.2, 3.1))),
            masses=(126.90447, 1.00782503223),
            multiplicity=multiplicity,
            options=SaddleOptions(project_rigid_motions=True),
            accepted_frame_warm_start=True,
            cold_fock_damping=RksColdFockDamping(8, 0.5, 1 << 20),
        )


@pytest.mark.parametrize(
    "case",
    (
        "admitted",
        "equivalent-local-label",
        "no-warm",
        "no-damping",
        "no-d3",
        "uks",
        "cartesian",
        "changed-potential",
    ),
)
def test_cs46_def2_ecp_saddle_gate(
    canonical_records: tuple, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Admit the actual native SCF, but never treat an unconverged frame as a saddle."""
    basis = canonical_records[2]
    cs = basis.by_element[55]
    if case in ("equivalent-local-label", "changed-potential"):
        potentials = json.loads(cs.ecp_data)
        local = max(potentials, key=lambda value: value["angular_momentum"][0])
        if case == "equivalent-local-label":
            local["angular_momentum"] = [4]
            potentials.reverse()
        else:
            local["coefficients"][0][0] = "-28.88"
        cs = replace(cs, ecp_data=json.dumps(potentials))
    representation = "cartesian" if case == "cartesian" else "spherical"
    calculator = Calculator(
        method=resolve_method(
            "B3LYP" if case == "no-d3" else "B3LYP-D3(BJ)",
            spin="polarized" if case == "uks" else "unpolarized",
        ),
        basis=replace(
            basis, elements=(basis.by_element[1], cs), representation=representation
        ),
        basis_representation=representation,
        device="cpu",
        initial_guess=None,
        max_iterations=3,
    )
    args = (
        calculator,
        ((55, (0, 0, 0)), (1, (0.1, 0.2, 5.5))),
    )
    controls = {
        "masses": (132.90545196, 1.00782503223),
        "accepted_frame_warm_start": case != "no-warm",
        "cold_fock_damping": None
        if case == "no-damping"
        else RksColdFockDamping(2, 0.5, 1 << 20),
    }
    if case in ("admitted", "equivalent-local-label"):
        result = optimize_saddle(*args, **controls)
        assert result.optimizer.receipt["status"] == "evaluation_failed"
        assert not result.optimizer.receipt["has_valid_state"]
        assert result.harmonics is None
        assert any(
            "SCF/forces did not converge" in str(failure)
            for failure in result.optimizer.failures
        )
    else:

        def forbidden(*args: Any, **kwargs: Any) -> None:
            pytest.fail("unqualified Cs46 saddle reached native batch preparation")

        monkeypatch.setattr(calculator, "prepare_batch", forbidden)
        with pytest.raises(NotImplementedError, match="Cs46 def2-ECP"):
            optimize_saddle(*args, **controls)


@pytest.mark.parametrize(
    "case",
    (
        "admitted",
        "tighter",
        "standard-guess",
        "no-warm",
        "damping",
        "no-d3",
        "uks",
        "cartesian",
        "changed-potential",
        "wrong-core",
        "mixed-operators",
        "initial-radius",
        "maximum-radius",
        "occupied-overlap",
    ),
)
def test_i28_def2_ecp_bounded_saddle_gate(
    canonical_records: tuple, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Only canonical, cold-start I28 with bounded steps enters the native SCF."""
    basis = canonical_records[2]
    iodine = basis.by_element[53]
    if case == "changed-potential":
        potentials = json.loads(iodine.ecp_data)
        local = max(potentials, key=lambda value: value["angular_momentum"][0])
        local["coefficients"][0][0] = "-21.84"
        iodine = replace(iodine, ecp_data=json.dumps(potentials))
    if case == "wrong-core":
        iodine = replace(iodine, ecp_core_electrons=30)
    atoms = ((53, (0, 0, 0)), (1, (0.1, 0.2, 3.1)))
    masses = (126.90447, 1.00782503223)
    elements = (basis.by_element[1], iodine)
    if case == "mixed-operators":
        atoms += ((55, (3.2, 0.3, 0.5)),)
        masses += (132.90545196,)
        elements += (basis.by_element[55],)
    representation = "cartesian" if case == "cartesian" else "spherical"
    calculator = Calculator(
        method=resolve_method(
            "B3LYP" if case == "no-d3" else "B3LYP-D3(BJ)",
            spin="polarized" if case == "uks" else "unpolarized",
        ),
        basis=replace(basis, elements=elements, representation=representation),
        basis_representation=representation,
        initial_guess="auto" if case == "standard-guess" else None,
        max_iterations=3,
        device="cpu",
    )
    controls = {
        "masses": masses,
        "charge": 1 if case == "mixed-operators" else 0,
        "multiplicity": 1,
        "options": SaddleOptions(
            project_rigid_motions=True,
            initial_radius={"initial-radius": 0.016, "tighter": 0.01}.get(case, 0.015),
            maximum_radius={"maximum-radius": 0.031, "tighter": 0.025}.get(case, 0.03),
        ),
        "minimum_occupied_overlap": {"occupied-overlap": 0.989, "tighter": 0.995}.get(
            case, 0.99
        ),
        "accepted_frame_warm_start": case != "no-warm",
        "cold_fock_damping": RksColdFockDamping(2, 0.5, 1 << 20)
        if case == "damping"
        else None,
    }
    if case in ("admitted", "tighter"):
        result = optimize_saddle(calculator, atoms, **controls)
        assert result.optimizer.receipt["status"] == "evaluation_failed"
        assert not result.optimizer.receipt["has_valid_state"]
        assert result.harmonics is None
        assert any(
            "SCF/forces did not converge" in str(failure)
            for failure in result.optimizer.failures
        )
    else:

        def forbidden(*args: Any, **kwargs: Any) -> None:
            pytest.fail("unqualified I28 saddle reached native batch preparation")

        monkeypatch.setattr(calculator, "prepare_batch", forbidden)
        with pytest.raises(NotImplementedError, match="I28 def2-ECP"):
            optimize_saddle(calculator, atoms, **controls)
