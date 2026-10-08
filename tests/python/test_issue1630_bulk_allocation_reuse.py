"""CPU-only allocation guards for a consolidated #1630 production cleanup.

These tests observe the real production call sites, not an AST-only proxy.
Independent scientific endpoint tests remain responsible for full accuracy.
"""

from __future__ import annotations

import inspect
import typing
from types import SimpleNamespace

import generativeqc.rks_hessian_integrals as second_hvp
import numpy as np
import pytest
from generativeqc import Primitive, Shell
from generativeqc.response_operator import _BaseResponseOperator
from generativeqc.response_solver import GMRESOptions, solve_many
from generativeqc_compiler.common.evidence import finite_difference
from generativeqc_compiler.dft import ExplicitGrid, NativeAO, partition_weights
from generativeqc_compiler.dft.nonlocal_reference import (
    nonlocal_explicit_geometry_derivatives_reference,
)
from generativeqc_compiler.dft.spatial import SpatialPolicy, build_spatial_tasks
from generativeqc_compiler.method import original_nonlocal_correlation
from generativeqc_compiler.tensor.autodiff import _vjp_einsum
from generativeqc_compiler.xc.reference import exchange_reference


def _count_numpy_calls(
    monkeypatch: pytest.MonkeyPatch,
    member: str,
    function: str,
) -> list[typing.Any]:
    """Observe only direct NumPy calls from one named production function."""
    original = getattr(np, member)
    shapes: list[typing.Any] = []

    def wrapped(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        if caller is not None and caller.f_code.co_name == function:
            shapes.append(args[0] if args else kwargs.get("shape"))
        del frame, caller
        return original(*args, **kwargs)

    monkeypatch.setattr(np, member, wrapped)
    return shapes


def test_dense_response_oracle_reuses_input_even_for_mutating_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = np.diag([1.0, 2.0, 3.0, 4.0])

    class ActingOperator:
        dimension = 4
        to_dense = _BaseResponseOperator.to_dense

        def apply(self, vector: np.ndarray) -> np.ndarray:
            result = matrix @ vector
            vector[:] = 37.0
            return result

    with monkeypatch.context() as patch:
        allocations = _count_numpy_calls(patch, "zeros", "to_dense")
        actual = ActingOperator().to_dense()
    assert allocations == [4]
    np.testing.assert_array_equal(actual, matrix)


def test_central_difference_reuses_step_workspace_and_detaches_each_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinates = np.array([[0.2, -0.4, 0.3], [0.8, 0.1, -0.2]])
    expected = 2.0 * coordinates

    def energy(value: np.ndarray, settings: dict) -> float:
        assert settings == {"fixed": True}
        return float(np.sum(value * value))

    with monkeypatch.context() as patch:
        allocations = _count_numpy_calls(patch, "empty_like", "finite_difference")
        result = finite_difference(
            energy,
            coordinates,
            expected,
            settings={"fixed": True},
            steps=(1e-2, 3e-3, 1e-3),
        )
    assert len(allocations) == 1
    assert all(row["error"]["passed"] for row in result["samples"])
    for row in result["samples"]:
        np.testing.assert_allclose(row["gradient"], expected, atol=1e-12, rtol=0)
    assert result["samples"][0]["gradient"] is not result["samples"][1]["gradient"]


def test_becke_coincident_pairs_share_zero_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = np.array([[0.3, 0.1, 0.7], [0.5, -0.2, 0.3]])
    centers = np.zeros((3, 3))
    with monkeypatch.context() as patch:
        allocations = _count_numpy_calls(patch, "zeros", "partition_weights")
        weights = partition_weights(points, centers)
    assert allocations == [len(points)]
    np.testing.assert_allclose(weights, np.full((2, 3), 1 / 3), atol=1e-15, rtol=0)


@pytest.mark.parametrize("variant", ("vv10", "rvv10"))
def test_nonlocal_reference_geometry_reuses_coordinate_accumulator(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    points = np.array(
        [[0.1, 0.2, 0.3], [0.5, -0.2, 0.4], [0.3, 0.6, -0.2]],
        dtype=np.float64,
    )
    weights = np.array([0.4, 0.3, 0.2])
    density = np.array([0.5, 0.4, 0.3])
    gradient = np.full((3, 3), 0.02)
    spec = original_nonlocal_correlation(variant)
    reference = nonlocal_explicit_geometry_derivatives_reference(
        points, weights, density, gradient, spec, tile_size=3
    )
    with monkeypatch.context() as patch:
        allocations = _count_numpy_calls(
            patch, "zeros", "nonlocal_explicit_geometry_derivatives_reference"
        )
        tiled = nonlocal_explicit_geometry_derivatives_reference(
            points, weights, density, gradient, spec, tile_size=2
        )
    assert allocations == [3]
    for result, expected in zip(tiled, reference, strict=True):
        np.testing.assert_allclose(result, expected, atol=2e-13, rtol=1e-12)


def test_spatial_builder_reuses_bounds_and_off_masks_without_aliasing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atoms = [("H", (-1.5, 0.0, 0.0)), ("H", (1.5, 0.0, 0.0))]
    shells = tuple(Shell(atom, 0, (Primitive(1.1, 1.0),)) for atom in range(2))
    points = np.array(
        [
            [-1.5, 0.0, 0.0],
            [-1.4, 0.1, 0.0],
            [1.4, -0.1, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
    )
    grid = ExplicitGrid(
        points,
        np.ones(len(points)),
        (0, 0, 1, 1, 0),
        {"scope": "allocation-ownership-fixture"},
    )
    with NativeAO(atoms, basis=shells) as basis:
        with monkeypatch.context() as patch:
            stacks = _count_numpy_calls(patch, "stack", "build_spatial_tasks")
            ones = _count_numpy_calls(patch, "ones", "build_spatial_tasks")
            tasks = build_spatial_tasks(
                basis, grid, policy=SpatialPolicy(region_points=2)
            )
        assert stacks == []
        assert ones == [basis.nao]
        assert len(tasks.tasks) >= 3
        tasks.validate(basis, grid)
        for task in tasks.tasks:
            np.testing.assert_array_equal(
                task.bounds,
                np.array(
                    [
                        grid.points[task.point_ids].min(axis=0),
                        grid.points[task.point_ids].max(axis=0),
                    ]
                ),
            )
            assert not task.bounds.flags.writeable


def test_lda_exchange_reference_shares_constant_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = np.array(
        [
            [0.4, 0.3],
            [0.6, 0.5],
            [0.01, 0.02],
            [0.02, 0.03],
            [0.04, 0.02],
        ]
    )
    with monkeypatch.context() as patch:
        ones = _count_numpy_calls(patch, "ones", "exchange_reference")
        zeros = _count_numpy_calls(patch, "zeros", "exchange_reference")
        energy, gradient, hessian = exchange_reference(features, spin=True, gga=False)
    assert ones == [2]
    # One output energy + one shared LDA zero derivative buffer.
    assert zeros == [2, 2]
    q = 4 / 3
    cx = 0.75 * (6 / np.pi) ** (1 / 3)
    np.testing.assert_allclose(
        energy,
        -cx * (features[0] ** q + features[1] ** q),
        atol=1e-14,
        rtol=0,
    )
    assert np.isfinite(gradient).all()
    assert np.isfinite(hessian).all()


def test_tensor_einsum_vjp_uses_one_max_extent_ones_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = np.arange(6, dtype=np.float64).reshape(2, 3) + 0.3
    right = np.arange(12, dtype=np.float64).reshape(3, 4) * 0.1 + 1
    cotangent = np.arange(8, dtype=np.float64).reshape(2, 4) * 0.2
    dtype = SimpleNamespace(dtype="float64")
    node = SimpleNamespace(
        attrs={
            "labels": (("i", "k"), ("k", "j")),
            "output": ("i", "j"),
            "coefficient": (1, 1),
        },
        spec=dtype,
        inputs=(
            SimpleNamespace(spec=SimpleNamespace(shape=left.shape)),
            SimpleNamespace(spec=SimpleNamespace(shape=right.shape)),
        ),
    )
    with monkeypatch.context() as patch:
        allocations = _count_numpy_calls(patch, "ones", "_vjp_einsum")
        actual = _vjp_einsum(node, (left, right), cotangent)
    assert allocations == [12]
    np.testing.assert_allclose(actual[0], cotangent @ right.T, rtol=0, atol=1e-13)
    np.testing.assert_allclose(actual[1], left.T @ cotangent, rtol=0, atol=1e-13)


def test_block_gmres_reuses_one_bounded_projection_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = np.diag(np.linspace(1.0, 3.0, 6))
    rhs = np.column_stack((np.linspace(0.1, 0.9, 6), np.linspace(-0.8, 0.2, 6)))

    class Operator:
        dimension = 6

        def __init__(self) -> None:
            self.problem = SimpleNamespace(
                compatibility_identity="block-hoist",
                validate_rhs=self.validate_rhs,
            )

        def validate_rhs(self, values: typing.Any) -> np.ndarray:
            value = np.asarray(values, dtype=np.float64)
            if value.ndim == 1:
                value = value[:, None]
            return value

        def apply(self, values: np.ndarray) -> np.ndarray:
            return matrix @ values

    options = GMRESOptions(rtol=1e-12, restart=6, max_iterations=10)
    with monkeypatch.context() as patch:
        allocations = _count_numpy_calls(patch, "empty", "_block_solve")
        result = solve_many(Operator(), rhs, strategy="blocked", options=options)
    assert allocations.count((6, 6)) == 1
    assert result.converged
    np.testing.assert_allclose(matrix @ result.solution, rhs, atol=1e-10, rtol=0)


def test_second_hvp_weight_buffer_is_frozen_per_ao_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The consuming stream owns weights before the next scratch overwrite."""
    signature = SimpleNamespace(tensor_indices=("component",), component_shape=(65,))
    ir = SimpleNamespace(requested_derivative_centers=(0,), signature=signature)
    retained: list[np.ndarray] = []

    class AtomMap:
        atom_indices = (0,)

        def __init__(self, *args: typing.Any) -> None:
            pass

        def expand_direction(self, direction: np.ndarray) -> np.ndarray:
            return direction

    class Plan:
        def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
            pass

        def __enter__(self) -> Plan:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def contract(self, stream: typing.Any, *, profile: bool) -> typing.Any:
            return SimpleNamespace(
                values=np.zeros((1, 3)), diagnostics={"profile": profile}
            )

    def consume(
        artifact: typing.Any,
        primitives: typing.Any,
        centers: typing.Any,
        tile: typing.Any,
        **kwargs: typing.Any,
    ) -> object:
        retained.append(np.array(tile.values, copy=True))
        return object()

    with monkeypatch.context() as patch:
        patch.setattr(second_hvp, "SecondAtomMap", AtomMap)
        patch.setattr(
            second_hvp, "second_coordinate_tiles", lambda *a, **k: [(0, 1, 2)]
        )
        patch.setattr(second_hvp, "_compile_cached", lambda *a, **k: object())
        patch.setattr(second_hvp, "prepare_second_shell_stream", consume)
        patch.setattr(second_hvp, "PreparedSecondDerivative", Plan)
        patch.setattr(second_hvp, "_scatter_hvp", lambda full, *a: full)
        allocations = _count_numpy_calls(patch, "zeros", "_run_kernel_hvp")
        second_hvp._run_kernel_hvp(
            {
                "adapter": object(),
                "cache": "synthetic",
                "resource_budget": None,
                "device_id": 0,
                "second_executions": [],
                "state": SimpleNamespace(nat=1),
            },
            "synthetic",
            lambda **kw: ir,
            {"_center_atoms": (0,)},
            (),
            np.zeros((1, 3)),
            np.arange(1, 66, dtype=np.float64),
            65,
            np.zeros((1, 3)),
        )
    assert allocations == [3, 65]
    assert len(retained) == 2
    np.testing.assert_array_equal(retained[0][:64], np.arange(1, 65))
    assert retained[0][64] == 0
    np.testing.assert_array_equal(retained[1][:64], np.zeros(64))
    assert retained[1][64] == 65


def test_spatial_validation_reuses_one_omitted_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atoms = [("H", (-2.0, 0.0, 0.0)), ("H", (2.0, 0.0, 0.0))]
    shells = tuple(Shell(atom, 0, (Primitive(2.0, 1.0),)) for atom in range(2))
    points = np.array(
        [
            [-2.0, 0.0, 0.0],
            [-1.9, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.1, 0.0, 0.0],
            [25.0, 0.0, 0.0],
        ],
    )
    grid = ExplicitGrid(
        points,
        np.ones(len(points)),
        (0, 0, 1, 1, 0),
        {"scope": "screened-validation-hoist"},
    )
    policy = SpatialPolicy(region_points=2, screening="absolute_ao_jet", cutoff=1e-8)
    with NativeAO(atoms, basis=shells) as basis:
        tasks = build_spatial_tasks(basis, grid, policy=policy)
        assert len(tasks.tasks) > 1
        with monkeypatch.context() as patch:
            allocations = _count_numpy_calls(patch, "ones", "validate")
            tasks.validate(basis, grid)
    assert allocations == [2]
