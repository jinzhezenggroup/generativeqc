"""Independent native uncut two-body BJ curvature, without electronic promotion."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from generativeqc.d3_hessian import NativeD3SecondOrder
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.geometry import compile_d3_bj
from generativeqc_compiler.method._generated_parameters import d3_parameters
from generativeqc_compiler.method.dispersion import D3Spec

NUMBERS = np.array([28, 6, 8, 1])
POSITIONS = np.array([[0, 0, 0], [2.2, 0.3, 0.1], [-0.4, 2.5, 0.2], [0.1, -0.2, 2.1]])
SPEC = D3Spec(**d3_parameters("B3LYP-D3(BJ)"))


def _gradient(positions: np.ndarray) -> np.ndarray:
    """Use independent simple-dftd3 with the identical canonical BJ parameters."""
    interface = pytest.importorskip("dftd3.interface")
    damping = interface.RationalDampingParam(
        s6=SPEC.s6, s8=SPEC.s8, a1=SPEC.a1, a2=SPEC.a2, s9=SPEC.s9
    )
    model = interface.DispersionModel(NUMBERS, positions)
    return model.get_dispersion(damping, grad=True)["gradient"]


@pytest.fixture
def native(tmp_path: Path) -> NativeD3SecondOrder:
    """Keep all correction curvature on the compiled CPU TensorIR owner."""
    return NativeD3SecondOrder(
        SPEC,
        NUMBERS,
        POSITIONS,
        cache=tmp_path,
        compiler=CppCompilerAdapter(Path("c++")),
    )


def test_native_d3_hvp_and_raw_hessian_match_independent_multistep_gradients(
    native: NativeD3SecondOrder, record_property: object
) -> None:
    """Freeze 2e-8 Eh/Bohr² reference and 2e-11 raw-invariant gates in advance."""
    matrix = native.hessian()
    direction = np.random.default_rng(2845).normal(size=POSITIONS.shape) * 0.03
    result = native.hvp(direction)
    assert np.linalg.norm(result.value) > 1e-7
    assert not result.diagnostics["public_calculator_endpoint"]
    assert result.diagnostics["electronic_response_solves"] == 0
    assert result.diagnostics["displaced_scf_evaluations"] == 0
    assert result.diagnostics["component"] == "d3_bj_two_body"
    np.testing.assert_allclose(matrix.matrix, matrix.matrix.T, atol=2e-11, rtol=0)
    np.testing.assert_allclose(
        matrix.matrix @ np.tile(np.eye(3), (len(NUMBERS), 1)), 0, atol=2e-11, rtol=0
    )
    np.testing.assert_allclose(
        matrix.matrix @ direction.reshape(-1),
        result.value.reshape(-1),
        atol=2e-11,
        rtol=0,
    )
    second = np.random.default_rng(2945).normal(size=POSITIONS.shape) * 0.03
    assert np.sum(second * result.value) == pytest.approx(
        np.sum(direction * native.hvp(second).value), abs=2e-11
    )
    for step in (3e-4, 1e-4, 3e-5):
        reference = np.empty_like(matrix.matrix)
        for column in range(POSITIONS.size):
            offset = np.zeros_like(POSITIONS)
            offset.reshape(-1)[column] = step
            reference[:, column] = (
                (_gradient(POSITIONS + offset) - _gradient(POSITIONS - offset))
                / (2 * step)
            ).reshape(-1)
        error = float(np.max(abs(matrix.matrix - reference)))
        record_property(f"independent_raw_error_{step}", error)
        assert error < 2e-8
    record_property("hvp_diagnostics", str(dict(result.diagnostics)))
    record_property("hessian_diagnostics", str(dict(matrix.diagnostics)))


def test_native_d3_rotation_covariance_and_geometry_identity(
    native: NativeD3SecondOrder, tmp_path: Path
) -> None:
    """A fresh rotated owner changes geometry identity, not the physical model."""
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    rotated = NativeD3SecondOrder(SPEC, NUMBERS, POSITIONS @ rotation.T, cache=tmp_path)
    assert rotated.identity != native.identity
    direction = np.random.default_rng(2945).normal(size=POSITIONS.shape) * 0.03
    np.testing.assert_allclose(
        rotated.hvp(direction @ rotation.T).value,
        native.hvp(direction).value @ rotation.T,
        atol=2e-11,
        rtol=0,
    )


def test_d3_second_order_budgets_and_invalid_directions_recover(
    native: NativeD3SecondOrder, tmp_path: Path
) -> None:
    """No failed action returns a partial correction matrix or poisons replay."""
    with pytest.raises(ValueError, match="output_budget_bytes"):
        native.hessian(output_budget_bytes=1)
    with pytest.raises(ValueError, match="direction"):
        native.hvp(np.full(POSITIONS.shape, np.nan))
    with pytest.raises(ValueError, match="direction"):
        native.hvp(np.zeros(POSITIONS.size))
    with pytest.raises(MemoryError, match="workspace"):
        NativeD3SecondOrder(
            SPEC, NUMBERS, POSITIONS, cache=tmp_path, max_workspace_bytes=1
        )
    with pytest.raises(ValueError, match="work budget"):
        NativeD3SecondOrder(SPEC, NUMBERS, POSITIONS, cache=tmp_path, max_work=1)
    with pytest.raises(MemoryError, match="max_generation_bytes"):
        NativeD3SecondOrder(
            SPEC, NUMBERS, POSITIONS, cache=tmp_path, max_generation_bytes=1
        )
    workspace = native.hvp(np.zeros_like(POSITIONS)).diagnostics["workspace_bytes"]
    bounded = NativeD3SecondOrder(
        SPEC,
        NUMBERS,
        POSITIONS,
        cache=tmp_path,
        max_workspace_bytes=workspace + POSITIONS.size**2 * 8,
    )
    assert np.isfinite(bounded.hvp(np.ones_like(POSITIONS)).value).all()
    with pytest.raises(MemoryError, match="simultaneous"):
        bounded.hessian()
    np.testing.assert_array_equal(native.hvp(np.zeros_like(POSITIONS)).value, 0)
    assert np.isfinite(native.hessian().matrix).all()


def test_native_d3_geometry_snapshot_and_parameter_identity(
    native: NativeD3SecondOrder, tmp_path: Path
) -> None:
    """Borrowed input mutation cannot silently replay another geometry or fit."""
    positions = POSITIONS.copy()
    bound = NativeD3SecondOrder(SPEC, NUMBERS, positions, cache=tmp_path)
    direction = np.ones_like(positions) * np.arange(len(NUMBERS))[:, None]
    expected = bound.hvp(direction).value
    positions[1, 0] += 0.1
    np.testing.assert_array_equal(bound.hvp(direction).value, expected)
    changed_geometry = NativeD3SecondOrder(SPEC, NUMBERS, positions, cache=tmp_path)
    changed_parameters = NativeD3SecondOrder(
        replace(SPEC, a1=1.1 * SPEC.a1), NUMBERS, POSITIONS, cache=tmp_path
    )
    assert changed_geometry.identity != bound.identity
    assert changed_parameters.identity != bound.identity
    assert (
        np.max(abs(changed_parameters.hessian().matrix - native.hessian().matrix))
        > 1e-7
    )
    with pytest.raises(ValueError, match="ATM"):
        NativeD3SecondOrder(replace(SPEC, s9=1), NUMBERS, POSITIONS, cache=tmp_path)


def test_native_d3_isolated_atom_has_exact_zero_curvature(tmp_path: Path) -> None:
    """An empty pair graph still executes the native constant-zero derivative."""
    isolated = NativeD3SecondOrder(SPEC, [28], [[0, 0, 0]], cache=tmp_path)
    np.testing.assert_array_equal(isolated.hvp(np.ones((1, 3))).value, 0)
    np.testing.assert_array_equal(isolated.hessian().matrix, 0)


@pytest.mark.parametrize("field", ("cn_cutoff", "pair_cutoff"))
def test_finite_cutoff_domains_do_not_inherit_uncut_second_order(field: str) -> None:
    """First-order cutoff admission does not silently qualify a curvature branch."""
    compiled = compile_d3_bj(
        replace(SPEC, **{field: 20.0}), NUMBERS.tolist(), POSITIONS
    )
    with pytest.raises(NotImplementedError, match="uncut two-body BJ"):
        compiled.coordinate_hvp()
