"""Opt-in Slurm gates for pre-AO envelopes and their native CSR consumer."""

import ctypes as ct
import os
import typing

import numpy as np
import pytest
import test_grid_cuda as grid_cuda_tests
from generativeqc._resident_ao_maps import ResidentAoMapDomain, ResidentDeviceAoMapOwner
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.dft.ao import jet_indices
from generativeqc_compiler.dft.cuda import CudaGrid
from generativeqc_compiler.dft.envelopes import ao_region_envelopes
from generativeqc_compiler.dft.fixtures import NAMES, basis_arguments, load_fixture

artifact = grid_cuda_tests.artifact

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_GRID_CUDA_TEST") != "1",
    reason="opt-in finite Slurm CUDA gate",
)


def download(
    pointer: typing.Any,
    shape: tuple[int, ...],
    dtype: typing.Any,
    owner: typing.Any,
) -> np.ndarray:
    """Testing only: copy a borrowed native panel after its owner stream gate."""
    import cupy as cp

    elements = int(np.prod(shape))
    if not elements:
        return np.empty(shape, dtype=dtype)
    pointer = ct.cast(pointer, ct.c_void_p).value
    memory = cp.cuda.UnownedMemory(pointer, elements * np.dtype(dtype).itemsize, owner)
    panel = cp.ndarray(shape, dtype=dtype, memptr=cp.cuda.MemoryPointer(memory, 0))
    return cp.asnumpy(panel)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("order", range(4))
def test_preao_csr_matches_independent_box_and_full_ao_oracles(
    artifact: typing.Any,
    name: str,
    order: int,
) -> None:
    """Cover signed contractions, spherical/cartesian jets, nodes and empty tails."""
    import cupy as cp

    meta, arrays = load_fixture(name)
    with NativeAO(**basis_arguments(meta)) as basis:
        centers = basis.packed[: 3 * basis.natom].reshape(-1, 3)
        points = np.ascontiguousarray(
            np.concatenate(
                (
                    arrays["points"][:7],
                    np.resize(centers, (7, 3)),
                    np.array([[100.0, 101.0, 102.0], [101.0, 102.0, 103.0]]),
                )
            )
        )
        cutoff = 1e-16
        ingredients = ("rho",) if order == 0 else ("rho", "gradient", "tau")
        full = basis.evaluate(points, order=order)
        with CudaGrid(
            basis,
            artifact,
            order=order,
            tile_points=7,
            active_ao_capacity=basis.nao,
            ingredients=ingredients,
        ) as cuda:
            cuda.set_density(arrays["density"])
            device_points = cp.asarray(points)
            cp.cuda.get_current_stream().synchronize()
            domain = ResidentAoMapDomain(
                basis.identity,
                "fixture-centers",
                "exact-fixture-point-order",
                cuda.device_id,
                device_points.data.ptr,
                len(points),
                7,
                order,
            )
            maps = ResidentDeviceAoMapOwner(
                cuda, domain, cutoff=cutoff, budget_bytes=1 << 20
            )
            assert maps.work["native_csr_ready"]
            assert cuda.metrics()["ao_grid_work"]["discovery_ao_jet_values"] == 0
            expected_visits = 0
            for begin in range(0, len(points), 7):
                end = min(begin + 7, len(points))
                tile = points[begin:end]
                bounds = np.stack((tile.min(axis=0), tile.max(axis=0)))
                envelope = ao_region_envelopes(basis, bounds, jet_indices(order))
                required = np.flatnonzero(np.any(envelope > cutoff, axis=0))
                with maps.feature_task(
                    cuda, domain, begin, end - begin, ingredients
                ) as task:
                    view = task.view
                    task.layout.require_derivative_order(order)
                    # The normal consumer error gate precedes any test-only D2H.
                    task.scatter(
                        np.zeros((2, view.nactive, view.nactive)),
                        reset=True,
                        download=True,
                    )
                    cp.cuda.ExternalStream(view.stream).synchronize()
                    ids = (
                        download(view.ao_ids, (view.nactive,), np.uintp, task)
                        if task.layout.indexed
                        else np.arange(basis.nao, dtype=np.uintp)
                    )
                    jets = download(
                        view.ao,
                        (len(jet_indices(order)), end - begin, len(ids)),
                        np.float64,
                        task,
                    )
                assert np.all(ids[:-1] < ids[1:])
                assert set(required) <= set(ids)
                omitted = np.setdiff1d(np.arange(basis.nao), ids)
                if len(omitted):
                    assert np.max(np.abs(full[:, begin:end, omitted])) <= cutoff
                np.testing.assert_allclose(
                    jets, full[:, begin:end][:, :, ids], atol=1e-11, rtol=1e-10
                )
                expected_visits += len(tile) * len(ids)
            work = cuda.metrics()["ao_grid_work"]
            assert work["active_point_ao"] == expected_visits
            assert work["ao_jet_values"] == expected_visits * len(jet_indices(order))
            assert work["ao_map_h2d_bytes"] == 0
            maps.reset_work()
            with maps.feature_task(cuda, domain, 0, 7, ingredients) as task:
                task.scatter(
                    np.zeros((2, task.view.nactive, task.view.nactive)), download=True
                )
            assert maps.work["discoveries"] == 0 and maps.work["cache_hits"] == 1
            assert maps.work["discovery_d2h_bytes"] == 0
            cuda._rebind_centers(np.ascontiguousarray(centers + 0.1))
            cuda.set_density(arrays["density"])
            with (
                pytest.raises(RuntimeError, match="stale resident AO map"),
                cuda.feature_task_resident_ao_map(
                    domain.point_pointer,
                    7,
                    0,
                    ingredients,
                    identity=maps.identity,
                ),
            ):
                pass


def test_preao_occupancy_guard_executes_the_complete_dense_domain(
    artifact: typing.Any,
) -> None:
    """Declining sparsity changes scheduling, never the evaluated AO inventory."""
    import cupy as cp

    meta, arrays = load_fixture("water")
    points = np.ascontiguousarray(arrays["points"])
    with (
        NativeAO(**basis_arguments(meta)) as basis,
        CudaGrid(
            basis,
            artifact,
            order=2,
            tile_points=7,
            active_ao_capacity=basis.nao,
            ingredients=("rho", "gradient"),
        ) as cuda,
    ):
        expected = basis.evaluate(points, 2)
        cuda.set_density(arrays["density"])
        device_points = cp.asarray(points)
        cp.cuda.get_current_stream().synchronize()
        domain = ResidentAoMapDomain(
            basis.identity,
            "fixture-centers",
            "fixture-point-order",
            cuda.device_id,
            device_points.data.ptr,
            len(points),
            7,
            2,
        )
        maps = ResidentDeviceAoMapOwner(
            cuda,
            domain,
            cutoff=1e-16,
            budget_bytes=1 << 20,
            max_active_fraction=1e-6,
        )
        assert maps.work["occupancy_declined"]
        assert not maps.work["native_csr_ready"]
        with maps.feature_task(
            cuda, domain, 0, min(7, len(points)), ("rho", "sigma")
        ) as task:
            view = task.view
            assert not task.layout.indexed and view.nactive == basis.nao
            task.scatter(np.zeros((2, basis.nao, basis.nao)), download=True)
            cp.cuda.ExternalStream(view.stream).synchronize()
            actual = download(view.ao, (10, view.npoint, basis.nao), np.float64, task)
        np.testing.assert_allclose(
            actual, expected[:, : view.npoint], atol=1e-11, rtol=1e-10
        )
        assert maps.work["dense_occupancy_tiles"] == 1
        assert maps.work["point_ao_visits"] == view.npoint * basis.nao


def test_preao_overflow_retains_and_bad_points_fail_closed(
    artifact: typing.Any,
) -> None:
    """The producer must not use underflow or invalid coordinates to omit AOs."""
    import cupy as cp

    meta, _ = load_fixture("f_spherical")
    with (
        NativeAO(**basis_arguments(meta)) as basis,
        CudaGrid(
            basis,
            artifact,
            order=3,
            tile_points=1,
            active_ao_capacity=basis.nao,
        ) as cuda,
    ):
        points = cp.asarray([[1e154, -1e154, 1e154]])
        cp.cuda.get_current_stream().synchronize()
        selected = cuda.select_ao_device_points(
            points.data.ptr,
            1,
            cutoff=1e-16,
            producer="pre-ao-envelope",
        )
        assert len(selected) == basis.nao
        points = cp.asarray([[np.nan, 0.0, 0.0]])
        cp.cuda.get_current_stream().synchronize()
        with pytest.raises(RuntimeError, match="nonfinite"):
            cuda.prepare_ao_map_device_points(
                points.data.ptr,
                1,
                cutoff=1e-16,
                budget_bytes=8192,
                identity="bad-points",
            )
