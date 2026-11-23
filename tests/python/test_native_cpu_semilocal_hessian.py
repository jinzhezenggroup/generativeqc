"""Independent moving-gradient checks for the linked CPU LDA/PBE endpoint."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import Atom, Calculator, GridSpec, KsOptions
from generativeqc._dft_gradient import StationaryKsState
from generativeqc._native_cpu_hessian import linked_method
from generativeqc.response_solver import GMRESOptions
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.dft.grid import MolecularGrid
from test_dft_complete_cpu import independent_global_hybrid_gradient

if TYPE_CHECKING:
    from collections.abc import Callable

ATOMS = (
    ("O", (0.01, -0.04, 0.02)),
    ("H", (1.42, 0.13, 1.06)),
    ("H", (-1.37, -0.09, 1.10)),
)
MOTION = np.array(((0.12, -0.07, 0.04), (-0.08, 0.06, 0.15), (-0.04, 0.01, -0.19)))


@pytest.mark.parametrize(
    ("representation", "basis_name"),
    (("cartesian", "sto-3g"), ("spherical", "def2-svp")),
)
def test_linked_pbe_bj_matches_independent_total_curvature(
    representation: str,
    basis_name: str,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Differentiate independent PySCF and s-dftd3 moving gradients."""
    grid = GridSpec(radial_points=12, angular_polar=6, angular_azimuth=12)
    bare, corrected = (
        Calculator(
            method=method,
            device="cpu",
            basis=basis_name,
            basis_representation=representation,
            ks_options=KsOptions(grid=grid),
            max_iterations=240,
            energy_tolerance=1e-13,
            density_tolerance=1e-11,
        )
        for method in ("pbe-rks", "pbe-d3(bj)-rks")
    )
    if not linked_method(corrected._library, corrected.method_ir.identity):
        pytest.skip("requires linked native PBE-D3(BJ) Hessian support")
    dispersion = pytest.importorskip("dftd3.interface")
    pytest.importorskip("pyscf")
    atoms = tuple(Atom.from_value(atom) for atom in ATOMS)
    numbers = np.array([atom.atomic_number for atom in atoms])
    positions = np.array([atom.position for atom in atoms])
    parameters = dispersion.RationalDampingParam(method="pbe", atm=False)

    def reference_d3(displacement: np.ndarray) -> tuple[float, np.ndarray]:
        result = dispersion.DispersionModel(
            numbers, positions + displacement
        ).get_dispersion(parameters, grad=True)
        return float(result["energy"]), result["gradient"]

    def reference(displacement: np.ndarray) -> tuple[float, np.ndarray]:
        displaced = tuple(
            Atom.from_value((atom.atomic_number, tuple(position)))
            for atom, position in zip(atoms, positions + displacement, strict=True)
        )
        # Share only the discrete quadrature; neither oracle reads native SCF state.
        state = SimpleNamespace(
            grid=MolecularGrid(displaced, grid, 0, 1).explicit(),
            _source=SimpleNamespace(
                grid_spec=grid,
                atomic_weights=np.concatenate(
                    [
                        MolecularGrid((atom,), grid, 0, 1).explicit().weights
                        for atom in displaced
                    ]
                ),
            ),
        )
        with NativeAO(
            displaced, basis=basis_name, representation=representation
        ) as basis:
            energy, gradient = independent_global_hybrid_gradient(
                basis, state, "pbe-rks", xc="PBE"
            )
        dispersion_energy, dispersion_gradient = reference_d3(displacement)
        return energy + dispersion_energy, gradient + dispersion_gradient

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("linked corrected Hessian entered the Python scientific owner")

    monkeypatch.setattr(Calculator, "_run_public_rks_second_order", forbidden)
    with pytest.raises(NotImplementedError, match="requires native scalar or recycled"):
        corrected.hessian(atoms, strategy="block", block_size=2)
    solver = GMRESOptions(atol=1e-12, rtol=1e-11)
    dense = corrected.hessian(atoms, solver_options=solver)
    action = corrected.hessian_vector_product(atoms, MOTION, solver_options=solver)
    bare_action = bare.hessian_vector_product(atoms, MOTION, solver_options=solver)
    for result in (dense, action):
        assert result.diagnostics["numerical_runtime"] == "native-cpp-cpu"
        assert result.diagnostics["d3_pair_visits"] > 0
        assert result.diagnostics["dispersion"]["electronic_response_solves"] == 0
        assert result.diagnostics["displaced_scf_evaluations"] == 0
    for field in ("response_iterations", "induced_fock_actions"):
        assert action.diagnostics[field] == bare_action.diagnostics[field]
    energy, gradient = reference(np.zeros_like(MOTION))
    point = corrected.singlepoint(atoms, properties=("energy", "forces"))
    assert point.converged
    assert point.energy == pytest.approx(energy, abs=2e-9, rel=0)
    np.testing.assert_allclose(point.forces, -gradient, atol=2e-8, rtol=0)
    np.testing.assert_allclose(dense.matrix, dense.matrix.T, atol=2e-8, rtol=0)
    np.testing.assert_allclose(
        dense.matrix @ MOTION.ravel(), action.value.ravel(), atol=2e-8, rtol=0
    )
    step = 3e-4
    axes = np.eye(MOTION.size).reshape((-1, *MOTION.shape))
    reference_matrix = np.column_stack(
        [
            (reference(step * axis)[1] - reference(-step * axis)[1]).ravel()
            / (2 * step)
            for axis in axes
        ]
    )
    error = np.max(np.abs(dense.matrix - reference_matrix))
    record_property("max_total_hessian_error", float(error))
    np.testing.assert_allclose(dense.matrix, reference_matrix, atol=3e-7, rtol=0)
    dispersion_action = (
        reference_d3(step * MOTION)[1] - reference_d3(-step * MOTION)[1]
    ) / (2 * step)
    assert np.max(np.abs(dispersion_action)) > 1e-7
    d3_error = np.max(np.abs(action.value - bare_action.value - dispersion_action))
    record_property("max_d3_hvp_error", float(d3_error))
    np.testing.assert_allclose(
        action.value - bare_action.value, dispersion_action, atol=2e-9, rtol=0
    )


@pytest.mark.parametrize("method", ("pbe-d3(0)-rks", "pbe-d4(bj-eeq-atm)-rks"))
def test_unlinked_semilocal_correction_does_not_publish_electronic_hessian(
    method: str,
) -> None:
    calculator = Calculator(method=method, device="cpu")
    assert not calculator.capabilities.supported_second_order
    with pytest.raises(NotImplementedError, match="does not expose public hessian"):
        calculator.hessian(ATOMS)


@pytest.mark.parametrize(
    ("method", "xc"), (("lda-rks", "LDA_X,LDA_C_PW"), ("pbe-rks", "PBE"))
)
@pytest.mark.parametrize(
    ("representation", "basis_name"),
    (("cartesian", "sto-3g"), ("spherical", "def2-svp")),
)
def test_linked_hessian_against_reconverged_gradient(
    method: str,
    xc: str,
    representation: str,
    basis_name: str,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Use the same discrete quadrature but independent PySCF SCF and gradients."""
    calculator = Calculator(
        method=method,
        device="cpu",
        basis=basis_name,
        basis_representation=representation,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=6, angular_azimuth=12)
        ),
        max_iterations=240,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
    )
    if not linked_method(calculator._library, calculator.method_ir.identity):
        pytest.skip("requires a linked native CPU semilocal Hessian inventory")
    pytest.importorskip("pyscf", reason="independent moving-grid gradient")

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("linked semilocal Hessian entered the Python scientific owner")

    monkeypatch.setattr(Calculator, "_run_public_rks_second_order", forbidden)
    if representation == "spherical":
        with pytest.raises(
            NotImplementedError, match="requires native scalar or recycled"
        ):
            calculator.hessian(ATOMS, strategy="block", block_size=2)
    solver = GMRESOptions(atol=1e-12, rtol=1e-11)
    dense = calculator.hessian(ATOMS, solver_options=solver)
    action = calculator.hessian_vector_product(ATOMS, MOTION, solver_options=solver)
    for result in (dense, action):
        assert result.diagnostics["numerical_runtime"] == "native-cpp-cpu"
        assert not result.diagnostics["runtime_compilation"]
        assert result.diagnostics["displaced_scf_evaluations"] == 0
        assert result.diagnostics["response_residual_norm"] < 2e-10
    np.testing.assert_allclose(dense.matrix, dense.matrix.T, atol=2e-8, rtol=0)
    np.testing.assert_allclose(
        dense.matrix @ np.tile(np.eye(3), (3, 1)), 0, atol=2e-8, rtol=0
    )
    np.testing.assert_allclose(
        dense.matrix @ MOTION.ravel(), action.value.ravel(), atol=2e-8, rtol=0
    )

    def gradient(offset: np.ndarray) -> np.ndarray:
        displaced = [
            (element, np.asarray(center) + motion)
            for (element, center), motion in zip(ATOMS, offset, strict=True)
        ]
        with (
            calculator.prepare_batch([displaced], warm_start=False) as batch,
            NativeAO(
                displaced, basis=basis_name, representation=representation
            ) as basis,
        ):
            if representation == "spherical":
                assert any(shell.angular_momentum == 2 for shell in basis.shells)
            point = batch.execute(strict=True, properties=("energy", "forces")).items[0]
            state = StationaryKsState.from_native(batch, basis)
            try:
                energy, reference = independent_global_hybrid_gradient(
                    basis, state, method, xc=xc
                )
                assert point.energy == pytest.approx(energy, abs=2e-9, rel=0)
                np.testing.assert_allclose(point.forces, -reference, atol=2e-8, rtol=0)
                return reference.ravel()
            finally:
                state._source.close()

    errors = []
    for step in (1e-3, 3e-4):
        directions = np.eye(MOTION.size).reshape((-1, *MOTION.shape))
        reference = np.column_stack(
            [
                (gradient(step * axis) - gradient(-step * axis)) / (2 * step)
                for axis in directions
            ]
        )
        error = float(np.max(np.abs(dense.matrix - reference)))
        errors.append(error)
        record_property(f"raw_matrix_max_error_at_{step}", error)
    assert errors[1] < 3e-7, errors
    assert errors[1] < 0.4 * errors[0], errors

    if hasattr(calculator._library, "generativeqc_ks_hessian_cpu_recycled_v1"):
        widths = (2, 3, 4) if representation == "cartesian" else (4,)
        for width in widths:
            recycled = calculator.hessian(
                ATOMS, strategy="recycled", block_size=width, solver_options=solver
            )
            assert recycled.diagnostics["response_strategy"] == "recycled"
            assert recycled.diagnostics["block_size"] == width
            assert (
                recycled.diagnostics["block_count"]
                == (MOTION.size + width - 1) // width
            )
            record_property(
                f"recycled_width_{width}_raw_matrix_max_error_at_0.0003",
                float(np.max(np.abs(recycled.matrix - reference))),
            )
            np.testing.assert_allclose(recycled.matrix, reference, atol=3e-7, rtol=0)
            np.testing.assert_allclose(recycled.matrix, dense.matrix, atol=2e-8, rtol=0)


@pytest.mark.parametrize("method", ("lda-rks", "pbe-rks"))
def test_linked_symmetric_diatomic_default_response(method: str) -> None:
    """Default physical residual gates admit symmetry-zero motion components."""
    calculator = Calculator(
        method=method,
        device="cpu",
        basis="sto-3g",
        basis_representation="cartesian",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=6, angular_azimuth=12)
        ),
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
    )
    if not linked_method(calculator._library, calculator.method_ir.identity):
        pytest.skip("requires a linked native CPU semilocal Hessian inventory")

    atoms = (("H", (0.0, 0.0, -0.72)), ("H", (0.0, 0.0, 0.72)))
    stretch = np.array(((0.0, 0.0, -0.1), (0.0, 0.0, 0.1)))
    dense = calculator.hessian(atoms)
    action = calculator.hessian_vector_product(atoms, stretch)
    for result in (dense, action):
        assert result.diagnostics["numerical_runtime"] == "native-cpp-cpu"
        assert result.diagnostics["response_residual_norm"] < 2e-10
    np.testing.assert_allclose(dense.matrix, dense.matrix.T, atol=2e-8, rtol=0)
    np.testing.assert_allclose(
        dense.matrix @ stretch.ravel(), action.value.ravel(), atol=2e-8, rtol=0
    )
