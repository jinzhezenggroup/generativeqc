"""Slurm-only gates at the strict cutoff and its adjacent FP64 values."""

import os
import typing

import numpy as np
import pytest
import test_grid_cuda as grid_cuda_tests
from generativeqc._resident_ao_maps import ResidentAoMapDomain, ResidentDeviceAoMapOwner
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.dft.ao import jet_indices
from generativeqc_compiler.dft.cuda import CudaGrid
from generativeqc_compiler.dft.fixtures import basis_arguments, load_fixture
from test_preao_region_cuda import download

artifact = grid_cuda_tests.artifact

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_GRID_CUDA_TEST") != "1",
    reason="opt-in finite Slurm CUDA gate",
)


@pytest.mark.parametrize("order", range(4))
def test_exact_map_matches_the_strict_materialized_predicate_at_cutoff(
    artifact: typing.Any, order: int
) -> None:
    """Use the incumbent GPU panel itself, not a tolerant host approximation."""
    import cupy as cp

    meta, arrays = load_fixture("water")
    points = np.ascontiguousarray(arrays["points"][:7])
    ingredients = ("rho",) if order == 0 else ("rho", "gradient", "tau")
    with (
        NativeAO(**basis_arguments(meta)) as basis,
        CudaGrid(
            basis,
            artifact,
            order=order,
            tile_points=7,
            active_ao_capacity=basis.nao,
            ingredients=ingredients,
        ) as cuda,
    ):
        cuda.set_density(arrays["density"])
        device_points = cp.asarray(points)
        cp.cuda.get_current_stream().synchronize()
        with cuda.feature_task_device_points(
            device_points.data.ptr, len(points), None, ingredients
        ) as task:
            task.scatter(np.zeros((2, basis.nao, basis.nao)), reset=True, download=True)
            cp.cuda.ExternalStream(task.view.stream).synchronize()
            panel = download(
                task.view.ao,
                (len(jet_indices(order)), len(points), basis.nao),
                np.float64,
                task,
            )
        maxima = np.max(np.abs(panel), axis=(0, 1))
        positive = np.sort(maxima[maxima > 0])
        thresholds = (float(positive[-1]), float(positive[len(positive) // 2]))
        domain = ResidentAoMapDomain(
            basis.identity,
            "threshold-centers",
            "threshold-point-order",
            cuda.device_id,
            device_points.data.ptr,
            len(points),
            7,
            order,
        )
        for threshold in thresholds:
            for cutoff in (
                float(np.nextafter(threshold, -np.inf)),
                threshold,
                float(np.nextafter(threshold, np.inf)),
            ):
                expected = np.flatnonzero(maxima > cutoff)
                sampled = cuda.select_ao_device_points(
                    device_points.data.ptr, len(points), cutoff=cutoff
                )
                np.testing.assert_array_equal(sampled, expected)
                maps = ResidentDeviceAoMapOwner(
                    cuda,
                    domain,
                    cutoff=cutoff,
                    budget_bytes=1 << 20,
                    producer="exact-jets-native-bitmask",
                )
                assert maps.work["native_csr_ready"]
                with maps.feature_task(
                    cuda, domain, 0, len(points), ingredients
                ) as task:
                    task.scatter(
                        np.zeros((2, task.view.nactive, task.view.nactive)),
                        reset=True,
                        download=True,
                    )
                    cp.cuda.ExternalStream(task.view.stream).synchronize()
                    selected = (
                        download(task.view.ao_ids, (task.view.nactive,), np.uintp, task)
                        if task.layout.indexed
                        else np.arange(basis.nao, dtype=np.uintp)
                    )
                np.testing.assert_array_equal(selected, expected)
