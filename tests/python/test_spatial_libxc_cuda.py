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


def test_batch_overlapping_maps_and_lifetime(
    program: SpatialContractionProgram,
) -> None:
    """Heterogeneous mapped tiles retain order and own copied lease storage."""
    from generativeqc_compiler.dft.spatial_xc_batch import (
        SpatialXcBatch,
        SpatialXcOptions,
        batch_request,
    )

    basis, grid, density = spatial_case(21, 43)
    cuda = CudaGrid(
        basis,
        program.artifact,
        tile_points=32,
        active_ao_capacity=19,
        ingredients=program.consumer.ingredients,
    )
    options = SpatialXcOptions(batch_tiles=2)
    resources = batch_request(cuda, options)
    batch = SpatialXcBatch(cuda, 2, resources.candidates[0].estimates[0].bytes, options)
    try:
        cuda.set_density(density)
        expected = np.zeros_like(density)
        expected_totals = np.zeros(3)
        for index, (ids, count) in enumerate(
            ((np.arange(19), 19), (np.arange(1, 21, 2), 3), (np.arange(2, 21), 7))
        ):
            with cuda.xc_task(grid.points[:count], ids, program.consumer) as lease:
                reference, _, _ = independent_xc(
                    program.spec,
                    basis.evaluate(grid.points[:count], ao_ids=ids),
                    density[:, ids[:, None], ids[None, :]],
                    grid.weights[:count],
                )
                expected[:, ids[:, None], ids[None, :]] += reference["potential"]
                expected_totals += np.r_[reference["energy"], reference["electrons"]]
                totals, actual = batch.enqueue(
                    lease,
                    grid.weights[:count],
                    program.consumer,
                    restricted=False,
                    reset=index == 0,
                    download=index == 2,
                )
                if index < 2:
                    assert actual is None
                    np.testing.assert_array_equal(totals, 0)
        np.testing.assert_allclose(actual, expected, atol=5e-10, rtol=5e-10)
        np.testing.assert_allclose(totals, expected_totals, atol=5e-10, rtol=5e-10)
        with pytest.raises(RuntimeError, match="expired"):
            batch.enqueue(
                lease,
                grid.weights[:7],
                program.consumer,
                restricted=False,
                reset=True,
                download=True,
            )
        cuda.set_density(-density)
        with (
            pytest.raises(RuntimeError, match="invalid or nonfinite"),
            cuda.xc_task(grid.points[:19], np.arange(19), program.consumer) as live,
        ):
            batch.enqueue(
                live,
                grid.weights[:19],
                program.consumer,
                restricted=False,
                reset=True,
                download=True,
            )
        cuda.set_density(density)
        with cuda.xc_task(grid.points[:19], np.arange(19), program.consumer) as live:
            _, recovered = batch.enqueue(
                live,
                grid.weights[:19],
                program.consumer,
                restricted=False,
                reset=True,
                download=True,
            )
        reference, _, _ = independent_xc(
            program.spec,
            basis.evaluate(grid.points[:19], ao_ids=np.arange(19)),
            density[:, :19, :19],
            grid.weights[:19],
        )
        np.testing.assert_allclose(
            recovered[:, :19, :19], reference["potential"], atol=5e-10, rtol=5e-10
        )
    finally:
        # The native owner must free a still-live queue before its stream.
        cuda.close()
        batch.close()


def test_optional_batch_allocation_fallback(
    program: SpatialContractionProgram, monkeypatch: Any
) -> None:
    basis, grid, density = spatial_case(19, 67)
    source = DensitySource(density, basis_identity=basis.identity)
    if program.spec.spin == "unpolarized":
        density[1] = density[0]
        source = DensitySource(density, basis_identity=basis.identity)
    expected, _, _ = independent_xc(
        program.spec, basis.evaluate(grid.points), density, grid.weights
    )
    with PreparedSpatialGrid(
        basis,
        grid,
        backend="cuda",
        artifact=program.artifact,
        tile_points=32,
        ingredients=program.consumer.ingredients,
    ) as spatial:

        def fail_allocation(*args: Any, **kwargs: Any) -> None:
            raise MemoryError("optional queue allocation")

        monkeypatch.setattr(
            "generativeqc_compiler.xc.prepared.SpatialXcBatch", fail_allocation
        )
        with PreparedXCContractions(program, basis, grid, spatial=spatial) as prepared:
            actual = prepared.execute(source, stamp=source.stamp)
            expected_potential = (
                expected["potential"][:1]
                if program.spec.spin == "unpolarized"
                else expected["potential"]
            )
            np.testing.assert_allclose(
                actual["potential"], expected_potential, atol=5e-10, rtol=5e-10
            )
            assert prepared.statistics["xc_batch_tiles"] == 1
            assert prepared.statistics["xc_batch_fallback"] == "allocation_failure"
            assert (
                dict(prepared.resource_plan.selections)["spatial_xc_batch"] == "single"
            )


def test_shared_spatial_executable_isolates_parameter_owners() -> None:
    from contextlib import ExitStack

    compiler = CudaCompilerAdapter(find_nvcc(), cuda_target_info("sm_120"))
    programs = [
        SpatialContractionProgram(name, compiler=compiler, cache=CACHE)
        for name in ("GGA_X_APBE", "GGA_X_PBE_SOL")
    ]
    assert programs[0].artifact.library == programs[1].artifact.library
    assert programs[0].metadata != programs[1].metadata
    basis, grid, density = spatial_case(19, 67)
    source = DensitySource(density, basis_identity=basis.identity)
    with ExitStack() as owners:
        prepared = []
        for program in programs:
            spatial = owners.enter_context(
                PreparedSpatialGrid(
                    basis,
                    grid,
                    backend="cuda",
                    artifact=program.artifact,
                    tile_points=32,
                    ingredients=program.consumer.ingredients,
                )
            )
            prepared.append(
                owners.enter_context(
                    PreparedXCContractions(program, basis, grid, spatial=spatial)
                )
            )
        for index in (0, 1, 0):
            expected, _, _ = independent_xc(
                programs[index].spec, basis.evaluate(grid.points), density, grid.weights
            )
            actual = prepared[index].execute(source, stamp=source.stamp)
            for key in expected:
                np.testing.assert_allclose(
                    actual[key], expected[key], atol=5e-10, rtol=5e-10
                )


@pytest.mark.parametrize(
    "precision,tiles", (("fp64", 1), ("fp64", 3), ("fp32", 3), ("fp32", 1))
)
def test_compact_spatial_assembly(
    program: SpatialContractionProgram, precision: str, tiles: int
) -> None:
    """Large mapped tiles and signed weights exercise the shared matrix path."""
    from generativeqc_compiler.dft import ExplicitGrid
    from generativeqc_compiler.dft.spatial_xc_batch import SpatialXcOptions

    restricted = program.spec.spin == "unpolarized"
    basis, original, density = spatial_case(19, 67, restricted=restricted)
    if restricted:
        # Separate BLAS calls can differ by an ulp with differently aligned
        # operands. This fixture must supply the exact paired-spin witness.
        density[1] = density[0]
    weights = original.weights.copy()
    weights[::3] *= -0.7
    grid = ExplicitGrid(original.points, weights, (0,) * len(weights), {})
    expected, _, _ = independent_xc(
        program.spec, basis.evaluate(grid.points), density, grid.weights
    )
    if restricted:
        expected["potential"] = expected["potential"][:1]
    source = DensitySource(density, basis_identity=basis.identity)
    with (
        PreparedSpatialGrid(
            basis,
            grid,
            backend="cuda",
            artifact=program.artifact,
            tile_points=32,
            policy=SpatialPolicy(region_points=43),
            ingredients=program.consumer.ingredients,
        ) as spatial,
        PreparedXCContractions(
            program,
            basis,
            grid,
            spatial=spatial,
            device_options=SpatialXcOptions(
                batch_tiles=tiles, matrix_precision=precision
            ),
        ) as prepared,
    ):
        for _ in range(2):
            actual = prepared.execute(source, stamp=source.stamp)
            for key in expected:
                np.testing.assert_allclose(
                    actual[key],
                    expected[key],
                    atol=2e-6 if precision == "fp32" and key == "potential" else 5e-10,
                    rtol=2e-6 if precision == "fp32" and key == "potential" else 5e-10,
                )
            assert (
                prepared.statistics["native_metrics"]["owned_device_bytes"]
                == spatial.tile_plan.allocation_bytes
            )
            assert prepared.statistics["xc_batch_tiles"] == tiles
            assert prepared.statistics["xc_point_evaluations"] == len(grid.points)
            assert prepared.statistics["owned_device_bytes"] == (
                spatial.tile_plan.allocation_bytes
                + prepared.statistics["xc_batch_device_bytes"]
            )
