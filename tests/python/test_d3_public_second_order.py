"""Public additive curvature with independent electronic and dispersion oracles."""

import json
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc import Atom, Calculator, GridSpec, KsOptions
from generativeqc.composite_rks_hessian import CompositeRKSResponse
from generativeqc.d3_hessian import NativeD3SecondOrder, compose_second_order
from generativeqc.response_solver import GMRESOptions
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.dft.grid import MolecularGrid
from generativeqc_compiler.method import DispersionCorrectionPrimitive, resolve_method
from generativeqc_compiler.method._generated_parameters import d3_parameters
from generativeqc_compiler.method.dispersion import D3Spec
from test_dft_complete_cpu import independent_global_hybrid_gradient
from test_integral_autodiff_second import require_native_library  # noqa: F401

MANIFEST = json.loads(
    (Path(__file__).parents[1] / "data/d3_second_order/manifest.json").read_text()
)
ATOMS = MANIFEST["atoms_bohr"]
GRID = GridSpec(**MANIFEST["grid"])
GATES = MANIFEST["gates"]
DIRECTION = np.array([[0.02, -0.03, 0.1], [-0.04, 0.01, -0.02], [0.02, 0.02, -0.08]])
SOLVER = GMRESOptions(atol=1e-12, rtol=1e-11)
SPEC = D3Spec(**d3_parameters("B3LYP-D3(BJ)"))
METHOD = replace(
    resolve_method("B3LYP"),
    identifier="B3LYP-D3(BJ)",
    primitives=(
        *resolve_method("B3LYP").primitives,
        DispersionCorrectionPrimitive(SPEC),
    ),
)


def make_calculator(method: Any = METHOD, **options: Any) -> Calculator:
    """Preserve one exact physical method and explicit discretized-grid contract."""
    return Calculator(
        method=method,
        device="cpu",
        basis_representation="spherical",
        ks_options=KsOptions(grid=GRID),
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=240,
        **options,
    )


def independent_total(atoms: Any) -> tuple[float, np.ndarray]:
    """Reconverge only the independent SCF, including its moving-grid gradient.

    The isolated-atom quadrature supplies unpartitioned atomic weights, so no
    production SCF or production gradient is needed at displaced geometries.
    Both engines consume identical raw AO definitions and VWN-RPA coefficients.
    """
    interface = pytest.importorskip("dftd3.interface")
    pytest.importorskip("pyscf")
    native_atoms = tuple(Atom.from_value(atom) for atom in atoms)
    grid = MolecularGrid(native_atoms, GRID, 0, 1).explicit()
    atomic_weights = np.concatenate(
        [MolecularGrid((atom,), GRID, 0, 1).explicit().weights for atom in native_atoms]
    )
    state = SimpleNamespace(
        grid=grid,
        _source=SimpleNamespace(grid_spec=GRID, atomic_weights=atomic_weights),
    )
    functional = METHOD.primitives[0].functional
    code = "+".join(f"{float(weight)}*{name}" for name, weight in functional.components)
    code += f"+{float(METHOD.primitives[1].coefficient)}*HF"
    with NativeAO(native_atoms, representation="spherical") as basis:
        energy, gradient = independent_global_hybrid_gradient(
            basis, state, "b3lyp-rks", xc=code
        )
    dispersion = interface.DispersionModel(
        np.array([atom.atomic_number for atom in native_atoms]),
        np.array([atom.position for atom in native_atoms]),
    ).get_dispersion(
        interface.RationalDampingParam(
            s6=SPEC.s6, s8=SPEC.s8, a1=SPEC.a1, a2=SPEC.a2, s9=SPEC.s9
        ),
        grad=True,
    )
    return energy + float(dispersion["energy"]), gradient + dispersion["gradient"]


def displaced(direction: np.ndarray, step: float) -> list:
    """Retain atom ordering while moving all Cartesian components in Bohr."""
    return [
        (element, np.asarray(center) + step * delta)
        for (element, center), delta in zip(ATOMS, direction, strict=True)
    ]


def independent_action(direction: np.ndarray, step: float) -> np.ndarray:
    """Reduce reference truncation, not the frozen physical acceptance threshold."""
    gradients = {
        offset: independent_total(displaced(direction, offset * step))[1]
        for offset in (-2, -1, 1, 2)
    }
    return (8 * (gradients[1] - gradients[-1]) - (gradients[2] - gradients[-2])) / (
        12 * step
    )


@pytest.fixture(scope="module")
def public_results(tmp_path_factory: pytest.TempPathFactory) -> tuple:
    """Keep successful detached results, never live electronic response owners."""
    cache = tmp_path_factory.mktemp("d3-public-curvature")
    calculator = make_calculator()
    options = {"cache": cache, "solver_options": SOLVER}
    try:
        matrix = calculator.hessian(ATOMS, **options)
        action = calculator.hessian_vector_product(ATOMS, DIRECTION, **options)
        plain = make_calculator(resolve_method("B3LYP"))
        try:
            electronic = plain.hessian(ATOMS, **options)
        finally:
            plain.clear_cache()
        yield calculator, matrix, action, electronic, cache
    finally:
        calculator.clear_cache()


def test_public_total_matches_independent_multistep_gradients(
    public_results: tuple, record_property: Any
) -> None:
    """Freeze total-method gates before recording complete independent raw columns."""
    calculator, matrix, action, _, _ = public_results
    point = calculator.singlepoint(ATOMS, properties=("energy", "forces"))
    energy, gradient = independent_total(ATOMS)
    assert abs(point.energy - energy) < GATES["energy_abs_hartree"]
    np.testing.assert_allclose(
        -point.forces, gradient, atol=GATES["gradient_abs_hartree_per_bohr"], rtol=0
    )
    for step in MANIFEST["difference_steps_bohr"]:
        reference = np.empty_like(matrix.matrix)
        for column in range(DIRECTION.size):
            direction = np.zeros_like(DIRECTION)
            direction.reshape(-1)[column] = 1
            reference[:, column] = independent_action(direction, step).reshape(-1)
        error = float(np.max(abs(matrix.matrix - reference)))
        record_property(f"total_raw_hessian_error_{step}", error)
        assert error < GATES["total_hessian_abs_hartree_per_bohr2"]
        np.testing.assert_allclose(
            action.value,
            independent_action(DIRECTION, step),
            atol=GATES["total_hvp_abs_hartree_per_bohr2"],
            rtol=0,
        )
    record_property("public_matrix_diagnostics", json.dumps(matrix.diagnostics))
    record_property("public_hvp_diagnostics", json.dumps(action.diagnostics))


def test_public_composition_is_once_complete_and_raw(public_results: tuple) -> None:
    """Diagnose omission, force-sign reversal and duplicate correction separately."""
    _, matrix, action, electronic, cache = public_results
    atoms = tuple(Atom.from_value(atom) for atom in ATOMS)
    correction = NativeD3SecondOrder(
        SPEC,
        [atom.atomic_number for atom in atoms],
        [atom.position for atom in atoms],
        cache=cache,
    ).hessian()
    delta = matrix.matrix - electronic.matrix
    np.testing.assert_allclose(
        delta, correction.matrix, atol=GATES["additive_source_abs"], rtol=0
    )
    assert np.max(abs(delta)) > 1e-7
    for wrong in (
        electronic.matrix,
        electronic.matrix - correction.matrix,
        electronic.matrix + 2 * correction.matrix,
    ):
        assert np.max(abs(matrix.matrix - wrong)) > 1e-7
    np.testing.assert_allclose(
        matrix.matrix, matrix.matrix.T, atol=GATES["raw_invariants_abs"], rtol=0
    )
    np.testing.assert_allclose(
        matrix.matrix @ np.tile(np.eye(3), (len(ATOMS), 1)),
        0,
        atol=GATES["raw_invariants_abs"],
        rtol=0,
    )
    np.testing.assert_allclose(
        matrix.matrix @ DIRECTION.reshape(-1),
        action.value.reshape(-1),
        atol=GATES["raw_invariants_abs"],
        rtol=0,
    )
    assert matrix.diagnostics["nuclear_response_solves"] == DIRECTION.size
    assert matrix.diagnostics["response_iterations"] == sum(
        matrix.diagnostics["response_iterations_per_coordinate"]
    )
    assert action.diagnostics["nuclear_response_solves"] == 1
    for result in (matrix, action):
        diagnostics = result.diagnostics
        assert diagnostics["source_names"].count("d3_bj_two_body") == 1
        assert diagnostics["source_order"] == diagnostics["source_names"]
        assert diagnostics["complete_source_coverage"]
        assert diagnostics["displaced_scf_evaluations"] == 0
        assert diagnostics["dispersion"]["electronic_response_solves"] == 0
        assert diagnostics["dispersion"]["total_scalar_work"] > 0
        assert (
            diagnostics["endpoint_seconds"]
            >= diagnostics["dispersion"]["preparation_seconds"]
        )
    np.testing.assert_allclose(
        sum(action.components.values()),
        action.value,
        atol=GATES["additive_source_abs"],
        rtol=0,
    )
    with pytest.raises(ValueError, match="already present"):
        compose_second_order(
            action, METHOD, atoms, cache=cache, max_workspace_bytes=64 << 20
        )
    with pytest.raises(ValueError, match="same electronic geometry"):
        compose_second_order(
            electronic,
            METHOD,
            tuple(Atom.from_value(atom) for atom in displaced(DIRECTION, 0.1)),
            cache=cache,
            max_workspace_bytes=64 << 20,
            output_budget_bytes=3 * electronic.matrix.nbytes,
        )


def test_failure_and_joint_budgets_never_publish_partial_total(
    public_results: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful electronic result is not a successful corrected result."""
    _, matrix, _, electronic, cache = public_results
    atoms = tuple(Atom.from_value(atom) for atom in ATOMS)
    options = {
        "cache": cache,
        "max_workspace_bytes": 64 << 20,
        "output_budget_bytes": 3 * electronic.matrix.nbytes,
    }
    with pytest.raises(ValueError, match="output_budget_bytes"):
        compose_second_order(
            electronic,
            METHOD,
            atoms,
            **{**options, "output_budget_bytes": 2 * electronic.matrix.nbytes},
        )
    with pytest.raises(MemoryError, match="simultaneous"):
        compose_second_order(
            electronic,
            METHOD,
            atoms,
            **{**options, "max_workspace_bytes": electronic.matrix.nbytes},
        )
    with monkeypatch.context() as scoped:

        def fail(*args: Any, **kwargs: Any) -> None:
            raise ArithmeticError("injected D3 action failure")

        scoped.setattr(NativeD3SecondOrder, "hvp", fail)
        with pytest.raises(ArithmeticError, match="injected D3"):
            compose_second_order(electronic, METHOD, atoms, **options)
    recovered = compose_second_order(electronic, METHOD, atoms, **options)
    np.testing.assert_array_equal(recovered.matrix, matrix.matrix)
    with pytest.raises(ValueError):
        recovered.matrix.setflags(write=True)
    wrong_method = replace(
        METHOD,
        primitives=(
            resolve_method("PBE0").primitives[0],
            *METHOD.primitives[1:],
        ),
    )
    with pytest.raises(ValueError, match="same electronic MethodIR"):
        compose_second_order(electronic, wrong_method, atoms, **options)
    changed = replace(
        METHOD,
        primitives=(
            *METHOD.primitives[:-1],
            DispersionCorrectionPrimitive(replace(SPEC, a1=1.1 * SPEC.a1)),
        ),
    )
    changed_result = compose_second_order(electronic, changed, atoms, **options)
    assert changed_result.identity != recovered.identity
    assert changed_result.diagnostics["total_method_identity"] == changed.identity
    assert np.max(abs(changed_result.matrix - recovered.matrix)) > 1e-7


def test_public_correction_failure_closes_electronic_owners_and_recovers(
    public_results: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not expose a successful electronic-only result after correction failure."""
    calculator, _, action, _, cache = public_results
    factory = CompositeRKSResponse.from_native.__func__
    grid_arrays = []
    initialize_correction = NativeD3SecondOrder.__init__
    with monkeypatch.context() as scoped:

        def capture(cls: Any, *args: Any, **kwargs: Any) -> Any:
            owner = factory(cls, *args, **kwargs)
            grid_arrays.extend(
                weakref.ref(array)
                for array in (owner.xc_kernel.jets, owner.xc_kernel.fields)
            )
            return owner

        def check_released(self: Any, *args: Any, **kwargs: Any) -> None:
            assert grid_arrays and all(reference() is None for reference in grid_arrays)
            initialize_correction(self, *args, **kwargs)

        def fail(*args: Any, **kwargs: Any) -> None:
            raise ArithmeticError("injected public correction failure")

        scoped.setattr(CompositeRKSResponse, "from_native", classmethod(capture))
        scoped.setattr(NativeD3SecondOrder, "__init__", check_released)
        scoped.setattr(NativeD3SecondOrder, "hvp", fail)
        with pytest.raises(ArithmeticError, match="injected public correction"):
            calculator.hessian_vector_product(
                ATOMS, DIRECTION, cache=cache, solver_options=SOLVER
            )
    recovered = calculator.hessian_vector_product(
        ATOMS, DIRECTION, cache=cache, solver_options=SOLVER
    )
    np.testing.assert_array_equal(recovered.value, action.value)


@pytest.mark.parametrize("variant", ("uks", "df", "ecp", "atm", "cutoff", "zero", "d4"))
def test_unqualified_domains_do_not_inherit_corrected_admission(variant: str) -> None:
    """Correction availability never grants an unqualified electronic capability."""
    method = METHOD
    options = {}
    if variant == "uks":
        method = replace(
            resolve_method("B3LYP", spin="polarized"),
            primitives=(
                *resolve_method("B3LYP", spin="polarized").primitives,
                DispersionCorrectionPrimitive(SPEC),
            ),
        )
    elif variant == "df":
        options["density_fitting"] = "cpu"
    elif variant == "ecp":
        from test_wb97mv_ecp_hessian import synthetic_basis

        options["basis"] = synthetic_basis()
    elif variant == "d4":
        method = replace(
            METHOD,
            primitives=(
                *METHOD.primitives[:-1],
                resolve_method("PBE-D4(BJ-EEQ-ATM)").primitives[-1],
            ),
        )
    else:
        correction = method.primitives[-1]
        if variant == "zero":
            correction = resolve_method("PBE0-D3(0)").primitives[-1]
        else:
            changed = (
                {"s9": 1.0, "version": "d3-bj-atm-spec-v1"}
                if variant == "atm"
                else {"pair_cutoff": 20.0}
            )
            correction = replace(correction, specification=replace(SPEC, **changed))
        method = replace(method, primitives=(*method.primitives[:-1], correction))
    calculator = make_calculator(method, **options)
    assert not calculator.second_order_capabilities
    with pytest.raises(NotImplementedError, match="does not expose public hessian"):
        calculator.hessian(ATOMS)


def test_corrected_output_admission_precedes_scf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retained electronic matrix is part of corrected publication's peak."""
    calculator = make_calculator()

    def fail(*args: Any, **kwargs: Any) -> None:
        pytest.fail("insufficient output budget must be rejected before SCF")

    monkeypatch.setattr(calculator, "prepare_batch", fail)
    with pytest.raises(ValueError, match="output_budget_bytes"):
        calculator.hessian(ATOMS, output_budget_bytes=2 * DIRECTION.size**2 * 8)


def test_errors_name_the_selected_physical_method_not_its_carrier() -> None:
    """A B3LYP MethodIR executed by a PBE carrier must not be described as PBE."""
    calculator = make_calculator()
    with pytest.raises(ValueError) as caught:
        calculator.singlepoint(ATOMS, properties=("energy", "unqualified-derivative"))
    message = str(caught.value)
    assert "method 'B3LYP-D3(BJ)'" in message
    assert calculator.method_ir.manifest_identity in message
    assert "carrier 'pbe-rks'" in message
    assert "cpu/spherical derivative domain" in message
