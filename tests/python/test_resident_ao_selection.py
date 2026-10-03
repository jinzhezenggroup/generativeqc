"""Qualify explicit resident AO discovery against independent CPU jet fixtures."""

import os
import typing
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_GRID_CUDA_TEST") != "1",
    reason="opt-in finite Slurm CUDA gate",
)


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> typing.Any:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require finite Slurm"
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info
    from generativeqc_compiler.dft.cuda import compile_cuda

    return compile_cuda(
        CudaCompilerAdapter(
            Path(os.environ["GENERATIVEQC_NVCC"]), cuda_target_info("sm_120")
        ),
        tmp_path_factory.mktemp("resident-ao-selection"),
    )


@pytest.mark.parametrize(
    "name", ("water", "f_cartesian", "f_spherical", "diffuse", "tight")
)
@pytest.mark.parametrize("order", (0, 1, 2, 3))
def test_discovery_covers_all_jets_without_density(
    artifact: typing.Any, name: str, order: int
) -> None:
    import cupy as cp
    from generativeqc_compiler.dft import NativeAO
    from generativeqc_compiler.dft.ao import jet_indices
    from generativeqc_compiler.dft.cuda import CudaGrid
    from generativeqc_compiler.dft.fixtures import basis_arguments, load_fixture

    meta, arrays = load_fixture(name)
    with (
        NativeAO(**basis_arguments(meta)) as basis,
        CudaGrid(
            basis,
            artifact,
            order=order,
            tile_points=129,
            active_ao_capacity=basis.nao,
            ingredients=("rho",),
        ) as grid,
    ):
        before = grid.metrics()["owned_device_bytes"]
        for count in (1, 7, min(129, len(arrays["points"]))):
            points = cp.asarray(arrays["points"][:count])
            cp.cuda.Stream.null.synchronize()
            maximum = np.max(
                np.abs(arrays["ao_jets"][: len(jet_indices(order)), :count]),
                axis=(0, 1),
            )
            for cutoff in (1e-16, 1e-4, 0.1, 1e100):
                # Exclude threshold ambiguity from a rounding comparison between
                # independent CPU and GPU collocation implementations.
                assert np.all(
                    np.abs(maximum - cutoff) > 1e-11 * np.maximum(maximum, cutoff)
                )
                selected = grid.select_ao_device_points(
                    points.data.ptr, count, cutoff=cutoff
                )
                np.testing.assert_array_equal(
                    selected, np.flatnonzero(maximum > cutoff)
                )
                assert not selected.flags.writeable
                with pytest.raises(ValueError):
                    selected.setflags(write=True)
        assert (
            grid.metrics()["owned_device_bytes"] == before == grid.plan.allocation_bytes
        )


def test_discovery_invalidates_views_and_rejects_bad_inputs(
    artifact: typing.Any,
) -> None:
    import cupy as cp
    from generativeqc_compiler.dft import NativeAO
    from generativeqc_compiler.dft.cuda import CudaGrid
    from generativeqc_compiler.dft.fixtures import basis_arguments, load_fixture

    meta, arrays = load_fixture("water")
    with (
        NativeAO(**basis_arguments(meta)) as basis,
        CudaGrid(
            basis,
            artifact,
            order=2,
            tile_points=7,
            active_ao_capacity=basis.nao,
            ingredients=("rho", "gradient", "tau"),
        ) as grid,
    ):
        points = cp.asarray(arrays["points"][:7])
        cp.cuda.Stream.null.synchronize()
        grid.set_density(arrays["density"])
        with (
            grid.feature_task_device_points(points.data.ptr, 7, None, ("rho",)),
            pytest.raises(RuntimeError, match="leased"),
        ):
            grid.select_ao_device_points(points.data.ptr, 7, cutoff=1e-16)
        grid.select_ao_device_points(points.data.ptr, 7, cutoff=1e-16)
        with pytest.raises(RuntimeError), grid._borrow_current_task():
            pass
        for cutoff in (0, -1, float("nan"), float("inf"), True):
            with pytest.raises(ValueError, match="cutoff"):
                grid.select_ao_device_points(points.data.ptr, 7, cutoff=cutoff)
        for count in (0, 8):
            with pytest.raises(ValueError):
                grid.select_ao_device_points(points.data.ptr, count, cutoff=1e-16)
        bad = points.copy()
        bad[0, 0] = cp.nan
        cp.cuda.Stream.null.synchronize()
        with pytest.raises(RuntimeError, match="nonfinite"):
            grid.select_ao_device_points(bad.data.ptr, 7, cutoff=1e-16)
        # A new valid discovery clears the previous device error and preserves D.
        ids = grid.select_ao_device_points(points.data.ptr, 7, cutoff=1e-16)
        with grid.feature_task_device_points(points.data.ptr, 7, ids, ("rho",)) as task:
            assert task.view.nactive == len(ids)
