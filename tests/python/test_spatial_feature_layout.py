"""Detached spatial layouts describe published jets, not private CUDA scratch."""

from threading import RLock
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.dft.ao import jet_indices
from generativeqc_compiler.dft.spatial import SpatialTask
from generativeqc_compiler.dft.spatial_prepared import PreparedSpatialGrid
from generativeqc_compiler.method.indexed_grid import AoGridBlockProgram
from generativeqc_compiler.tensor import execute


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_detached_layout_matches_requested_jets_without_downgrading_map(
    backend: str, order: int
) -> None:
    """Exercise the CUDA adapter on host arrays without constructing a device."""
    full_jets = np.arange(40, dtype=np.float64).reshape(10, 2, 2) / 17
    ao_ids = np.array([0, 2], dtype=np.int64)
    density = np.stack((np.eye(3), 2 * np.eye(3)))
    task = SpatialTask(
        np.arange(2),
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        np.array([0]),
        ao_ids,
        jet_indices(2),
        1,
        np.zeros(10),
        "0" * 64,
    )

    class HostCuda:
        ingredients = ("rho",)
        basis_generation = 7
        geometry_generation = 11

        def set_density(self, value: np.ndarray) -> None:
            np.testing.assert_array_equal(value, density)

        def evaluate(self, points: np.ndarray, **kwargs: Any) -> dict:
            layout = kwargs["block_layout"]
            assert layout.derivative_order == layout.map_derivative_order == 2
            assert (layout.basis_generation, layout.geometry_generation) == (7, 11)
            np.testing.assert_array_equal(kwargs["ao_ids"], ao_ids)
            assert kwargs["download_jets"]
            return {"ao_jets": full_jets.copy(), "rho": np.zeros((2, len(points)))}

        def close(self) -> None:
            pass

    def evaluate(points: np.ndarray, requested: int, **kwargs: Any) -> np.ndarray:
        return full_jets[: len(jet_indices(requested))].copy()

    # Host stand-ins replace native owners; iteration, point slicing, layout
    # publication and the downstream TensorIR consumer remain real.
    owner = object.__new__(PreparedSpatialGrid)
    owner._lock = RLock()
    owner._closed = owner._leased = False
    owner._execution = 0
    owner._cuda = HostCuda() if backend == "cuda" else None
    owner.basis = SimpleNamespace(nao=3, identity="basis", evaluate=evaluate)
    owner.grid = SimpleNamespace(
        points=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        weights=np.ones(2),
        owners=(0, 0),
    )
    owner.tasks = SimpleNamespace(tasks=(task,), generation_id=task.generation_id)
    owner.tile_plan = SimpleNamespace(order=2, tile_points=2)
    owner.ingredients = ("rho",)
    owner.timings = {"ao_seconds": 0.0, "density_seconds": 0.0, "tiles": 0}
    with owner:
        (tile,) = owner.iter_features(density, include_jets=True, order=order)

    layout = tile.layout
    assert layout is not None and tile.ao_jets is not None
    assert layout.derivative_order == order
    assert layout.map_derivative_order == 2
    assert tile.ao_jets.shape == (layout.jet_count, layout.npoint, layout.nactive)
    layout.require_derivative_order(order)
    if order < 2:
        with pytest.raises(ValueError, match="requested derivative jets"):
            layout.require_derivative_order(order + 1)
    jets = 1 if order == 0 else 4
    projected = execute(
        AoGridBlockProgram(layout).projection_program(jets),
        {"density": density, "ao_ids": ao_ids, "ao_jets": tile.ao_jets},
    ).outputs["projected"]
    expected = np.einsum(
        "jpm,smn->sjpn",
        full_jets[:jets],
        density[:, ao_ids[:, None], ao_ids[None, :]],
    )
    np.testing.assert_allclose(projected, expected, rtol=1e-13, atol=1e-13)
