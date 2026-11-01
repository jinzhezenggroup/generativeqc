"""Allocated GPU gates for generic resident spatial energy/potential endpoints."""

import os
import shutil
import typing
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.provenance import find_nvcc
from generativeqc_compiler.dft import DensitySource
from generativeqc_compiler.dft.cuda import CudaGrid, compile_cuda
from generativeqc_compiler.dft.spatial import SpatialPolicy
from generativeqc_compiler.dft.spatial_prepared import PreparedSpatialGrid
from generativeqc_compiler.xc.native import NativeContractionProgram
from generativeqc_compiler.xc.prepared import PreparedXCContractions
from generativeqc_compiler.xc.spatial_cuda import (
    compile_spatial_cuda,
    compiled_spatial_xc,
)
from generativeqc_compiler.xc.spec import UnsupportedXC
from test_spatial_xc_codegen import (
    SPATIAL_CASES,
    independent_xc,
    spatial_case,
    spatial_spec,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_GRID_CUDA_TEST") != "1", reason="finite Slurm GPU gate"
)
CACHE = Path(".artifacts/spatial-xc-cuda-cache")


@pytest.fixture(scope="module", params=SPATIAL_CASES)
def case_artifact(request: typing.Any) -> typing.Any:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU execution requires Slurm"
    nvcc = find_nvcc()
    assert nvcc is not None
    compiler = CudaCompilerAdapter(nvcc, cuda_target_info("sm_120"))
    spec = spatial_spec(request.param)
    artifact = (
        compile_cuda(compiler, CACHE)
        if request.param in ("LDA_XC_PW", "PBE")
        else compile_spatial_cuda(spec, compiler, CACHE)
    )
    return spec, artifact


def native_program(spec: typing.Any) -> NativeContractionProgram:
    return NativeContractionProgram(
        spec, compiler=CppCompilerAdapter(Path(shutil.which("c++"))), cache=CACHE
    )


@pytest.mark.parametrize("restricted", [False, True])
@pytest.mark.parametrize("nao,tile", [(1, 1), (3, 5)])
def test_resident_endpoint_independent_oracle_and_no_host_fallback(
    case_artifact: typing.Any,
    restricted: bool,
    nao: int,
    tile: int,
    monkeypatch: typing.Any,
) -> None:
    spec, artifact = case_artifact
    spec = replace(spec, spin="unpolarized" if restricted else "polarized")
    basis, grid, density = spatial_case(nao, restricted=restricted)
    expected, _, _ = independent_xc(
        spec, basis.evaluate(grid.points), density, grid.weights
    )
    if restricted:
        expected["potential"] = expected["potential"][:1]
    native = native_program(spec)
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
            artifact=artifact,
            tile_points=tile,
            policy=SpatialPolicy(region_points=7),
            orbital_capacity=(nao, nao),
            orbital_tile=2,
            ingredients=compiled_spatial_xc(spec).ingredients,
        ) as spatial,
        PreparedXCContractions(native, basis, grid, spatial=spatial) as prepared,
    ):

        def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
            raise AssertionError("resident XC entered a CPU scientific consumer")

        with monkeypatch.context() as patch:
            patch.setattr(native, "potential_tile", forbidden)
            patch.setattr(native, "evaluate", forbidden)
            patch.setattr(basis, "evaluate", forbidden)
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
                assert stats["xc_point_cache_bytes"] == 11 * 8 * tile
                assert (
                    stats["native_metrics"]["owned_device_bytes"]
                    == spatial.tile_plan.allocation_bytes
                )
        # Density derivatives cover both gradient legs and the meta-GGA tau
        # coefficient. Use two steps without loosening the independent gate.
        direction = 0.03 * density
        derivative = np.sum(
            actual["potential"]
            * (direction.sum(axis=0)[None] if restricted else direction)
        )
        for step in (1e-4, 2e-5):
            values = []
            for sign in (1, -1):
                displaced = DensitySource(
                    density + sign * step * direction, basis_identity=basis.identity
                )
                values.append(
                    prepared.execute(displaced, stamp=displaced.stamp)["energy"]
                )
            assert (values[0] - values[1]) / (2 * step) == pytest.approx(
                derivative, abs=3e-8
            )
        # An invalid evaluation must not poison the next valid execution.
        vacuum = DensitySource(np.zeros_like(density), basis_identity=basis.identity)
        with pytest.raises(UnsupportedXC, match="CUDA XC output"):
            prepared.execute(vacuum, stamp=vacuum.stamp)
        again = prepared.execute(source, stamp=source.stamp)
        np.testing.assert_allclose(
            again["potential"], expected["potential"], atol=5e-10, rtol=5e-10
        )


def test_artifact_mismatch_rejects_before_borrowing(case_artifact: typing.Any) -> None:
    spec, artifact = case_artifact
    basis, grid, density = spatial_case()
    wrong = spatial_spec("BLEND" if spec.identifier != "mixed-gga-mgga" else "R2SCAN")
    native = native_program(wrong)
    with PreparedSpatialGrid(
        basis, grid, backend="cuda", artifact=artifact, tile_points=5
    ) as spatial:
        assert spatial._cuda.xc_capability(compiled_spatial_xc(wrong)) is None
        with pytest.raises(ValueError, match="incompatible"):
            PreparedXCContractions(
                native, basis, grid, spatial=spatial, schedule="device_fused"
            )
        # The existing explicit host route remains executable when its full
        # feature contract is available; mismatches never select another XC.
        with PreparedXCContractions(native, basis, grid, spatial=spatial) as prepared:
            assert prepared.schedule.name == "host_unfused"
            source = DensitySource(density, basis_identity=basis.identity)
            actual = prepared.execute(source, stamp=source.stamp)
            expected, _, _ = independent_xc(
                wrong, basis.evaluate(grid.points), density, grid.weights
            )
            np.testing.assert_allclose(
                actual["potential"], expected["potential"], atol=5e-10, rtol=5e-10
            )


def test_noncontiguous_local_maps_and_expired_lease(case_artifact: typing.Any) -> None:
    spec, artifact = case_artifact
    basis, grid, density = spatial_case(5)
    capability = compiled_spatial_xc(spec)
    # Built-ins use their established entry even in an ordinary grid artifact.
    from generativeqc_compiler.xc.spatial_cuda import spatial_xc_capability

    with CudaGrid(
        basis,
        artifact,
        tile_points=5,
        active_ao_capacity=3,
        ingredients=capability.ingredients,
    ) as cuda:
        cuda.set_density(density)
        consumer = spatial_xc_capability(spec, cuda.compiled_xc_identity)
        expected = np.zeros_like(density)
        for index, ids in enumerate(
            (np.array([0, 2, 4]), np.array([1]), np.array([], dtype=int))
        ):
            with cuda.xc_task(grid.points[:5], ids, consumer) as lease:
                if len(ids):
                    jets = basis.evaluate(grid.points[:5], ao_ids=ids)
                    reference, _, _ = independent_xc(
                        spec,
                        jets,
                        density[:, ids[:, None], ids[None, :]],
                        grid.weights[:5],
                    )
                    expected[:, ids[:, None], ids[None, :]] += reference["potential"]
                    _, actual = lease.xc(
                        grid.weights[:5], consumer, reset=index == 0, download=True
                    )
                    np.testing.assert_allclose(actual, expected, atol=5e-10, rtol=5e-10)
        with pytest.raises(RuntimeError, match="expired"):
            lease.xc(grid.weights[:5], consumer)


def test_feature_only_r2scan_lease_does_not_require_point_artifact(
    case_artifact: typing.Any,
) -> None:
    """Preserve the external stationary geometry consumer's feature lease."""
    _, artifact = case_artifact
    basis, grid, density = spatial_case()
    with CudaGrid(basis, artifact, tile_points=5, active_ao_capacity=3) as cuda:
        cuda.set_density(density)
        with cuda.xc_task(grid.points[:5], np.arange(3), "R2SCAN") as lease:
            assert lease.view.jets == 4 and bool(lease.view.features)
            # Borrowing features does not admit a name-only XC evaluation.
            with pytest.raises(ValueError, match="does not support"):
                lease.xc(grid.weights[:5], "R2SCAN")


def test_missing_ingredients_fail_before_device_evaluation(
    case_artifact: typing.Any,
) -> None:
    spec, artifact = case_artifact
    from generativeqc_compiler.xc.spatial_cuda import spatial_xc_capability

    if "tau" not in spec.ingredients:
        return
    basis, grid, density = spatial_case()
    with CudaGrid(
        basis,
        artifact,
        tile_points=5,
        active_ao_capacity=3,
        ingredients=("rho", "gradient"),
    ) as cuda:
        cuda.set_density(density)
        consumer = spatial_xc_capability(spec, cuda.compiled_xc_identity)
        with (
            pytest.raises(ValueError, match="features"),
            cuda.xc_task(grid.points[:5], np.arange(3), consumer),
        ):
            pytest.fail("missing tau unexpectedly admitted")
