"""Real-GPU E/V gates for source-derived Libxc programs on shared spatial execution."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.provenance import find_nvcc
from generativeqc_compiler.dft import DensitySource
from generativeqc_compiler.dft.cuda import CudaGrid, compile_cuda
from generativeqc_compiler.dft.spatial import SpatialPolicy
from generativeqc_compiler.dft.spatial_prepared import PreparedSpatialGrid
from generativeqc_compiler.xc.prepared import PreparedXCContractions
from generativeqc_compiler.xc.spatial_program import SpatialContractionProgram
from test_spatial_libxc_codegen import LIBXC_CASES
from test_spatial_xc_codegen import independent_xc, spatial_case

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_GRID_CUDA_TEST") != "1", reason="finite Slurm GPU gate"
)
CACHE = Path(".artifacts/spatial-libxc-cuda-cache")


@pytest.fixture(
    scope="module", params=[(n, r) for n in LIBXC_CASES for r in (False, True)]
)
def program(request: Any) -> SpatialContractionProgram:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU execution requires Slurm"
    nvcc = find_nvcc()
    assert nvcc is not None
    return SpatialContractionProgram(
        request.param[0],
        spin="unpolarized" if request.param[1] else "polarized",
        compiler=CudaCompilerAdapter(nvcc, cuda_target_info("sm_120")),
        cache=CACHE,
    )


@pytest.mark.parametrize("nao,tile", ((1, 1), (3, 5)))
def test_libxc_resident_endpoint(
    program: SpatialContractionProgram, nao: int, tile: int, monkeypatch: Any
) -> None:
    """Full E/V, both density routes, bounded storage, work counts and recovery."""
    restricted = program.spec.spin == "unpolarized"
    basis, grid, density = spatial_case(nao, restricted=restricted)
    expected, _, _ = independent_xc(
        program.spec, basis.evaluate(grid.points), density, grid.weights
    )
    if restricted:
        expected["potential"] = expected["potential"][:1]
    source = DensitySource(density, basis_identity=basis.identity)
    source = source.with_orbitals(
        tuple(np.linalg.cholesky(d) for d in density),
        (np.ones(nao), np.ones(nao)),
        stamp=source.stamp,
    )
    with (
        PreparedSpatialGrid(
            basis,
            grid,
            backend="cuda",
            artifact=program.artifact,
            tile_points=tile,
            policy=SpatialPolicy(region_points=7),
            orbital_capacity=(nao, nao),
            orbital_tile=2,
            ingredients=program.consumer.ingredients,
        ) as spatial,
        PreparedXCContractions(program, basis, grid, spatial=spatial) as prepared,
    ):

        def forbidden(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("device endpoint entered host scientific work")

        with monkeypatch.context() as patch:
            patch.setattr(basis, "evaluate", forbidden)
            patch.setattr("generativeqc_compiler.xc.prepared._tiles", forbidden)
            for route in ("density_matrix", "orbitals", "density_matrix"):
                actual = prepared.execute(source, stamp=source.stamp, route=route)
                for key in expected:
                    np.testing.assert_allclose(
                        actual[key], expected[key], atol=5e-10, rtol=5e-10
                    )
                stats = prepared.statistics
                assert stats["xc_backend"] == "native_cuda"
                assert stats["cpu_contraction_seconds"] == 0
                assert stats["xc_point_evaluations"] == len(grid.points)
                assert stats["xc_point_cache_bytes"] == 88 * tile
                assert (
                    stats["native_metrics"]["owned_device_bytes"]
                    == spatial.tile_plan.allocation_bytes
                )
        # Work-domain vxc is an energy derivative on the unclipped interior;
        # boundary tests separately check Libxc's documented worker outputs.
        direction = 0.03 * density
        derivative = np.sum(
            actual["potential"]
            * (direction.sum(axis=0)[None] if restricted else direction)
        )
        for step in (1e-4, 2e-5):
            energies = []
            for sign in (1, -1):
                shifted = DensitySource(
                    density + sign * step * direction, basis_identity=basis.identity
                )
                energies.append(
                    prepared.execute(shifted, stamp=shifted.stamp)["energy"]
                )
            assert (energies[0] - energies[1]) / (2 * step) == pytest.approx(
                derivative, abs=3e-8
            )
        vacuum = DensitySource(np.zeros_like(density), basis_identity=basis.identity)
        empty = prepared.execute(vacuum, stamp=vacuum.stamp)
        assert empty["energy"] == 0.0
        np.testing.assert_array_equal(empty["potential"], 0.0)
        again = prepared.execute(source, stamp=source.stamp)
        np.testing.assert_allclose(
            again["potential"], expected["potential"], atol=5e-10, rtol=5e-10
        )
        with pytest.raises(ValueError, match="matching device_fused"):
            PreparedXCContractions(
                program, basis, grid, spatial=spatial, schedule="host_unfused"
            )


def test_libxc_local_maps_and_lease(program: SpatialContractionProgram) -> None:
    basis, grid, density = spatial_case(
        5, restricted=program.spec.spin == "unpolarized"
    )
    with CudaGrid(
        basis,
        program.artifact,
        tile_points=5,
        active_ao_capacity=3,
        ingredients=program.consumer.ingredients,
    ) as cuda:
        cuda.set_density(density)
        expected = np.zeros_like(density)
        for index, ids in enumerate(
            (np.array([0, 2, 4]), np.array([1]), np.array([], dtype=int))
        ):
            with cuda.xc_task(grid.points[:5], ids, program.consumer) as lease:
                if len(ids):
                    reference, _, _ = independent_xc(
                        program.spec,
                        basis.evaluate(grid.points[:5], ao_ids=ids),
                        density[:, ids[:, None], ids[None, :]],
                        grid.weights[:5],
                    )
                    expected[:, ids[:, None], ids[None, :]] += reference["potential"]
                    _, actual = lease.xc(
                        grid.weights[:5],
                        program.consumer,
                        reset=index == 0,
                        download=True,
                    )
                    np.testing.assert_allclose(actual, expected, atol=5e-10, rtol=5e-10)
        with pytest.raises(RuntimeError, match="expired"):
            lease.xc(grid.weights[:5], program.consumer)


def test_libxc_mismatch_rejected(program: SpatialContractionProgram) -> None:
    basis, grid, _ = spatial_case()
    compiler = CudaCompilerAdapter(find_nvcc(), cuda_target_info("sm_120"))
    wrong_artifact = compile_cuda(compiler, CACHE)
    with (
        PreparedSpatialGrid(
            basis, grid, backend="cuda", artifact=wrong_artifact, tile_points=5
        ) as spatial,
        pytest.raises(ValueError, match="matching device_fused"),
    ):
        PreparedXCContractions(program, basis, grid, spatial=spatial)
