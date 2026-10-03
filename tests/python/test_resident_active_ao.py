"""Real-device gates for resident points with nonidentity AO panels.

Selected features are compared with independent retained CPU fixture AO jets;
the test checks map/gather correctness without assigning a screening tolerance.
"""

import ctypes as ct
import os
import typing
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_GRID_CUDA_TEST") != "1",
    reason="opt-in finite Slurm CUDA gate",
)


@pytest.mark.parametrize("selection", ("identity", "subset", "empty"))
def test_resident_selected_features_and_density_jets(
    tmp_path: Path, selection: str
) -> None:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require finite Slurm"
    import cupy as cp
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info
    from generativeqc_compiler.dft import NativeAO
    from generativeqc_compiler.dft.cuda import CudaGrid, compile_cuda
    from generativeqc_compiler.dft.fixtures import basis_arguments, load_fixture

    meta, arrays = load_fixture("water")
    compiler = CudaCompilerAdapter(
        Path(os.environ["GENERATIVEQC_NVCC"]), cuda_target_info("sm_120")
    )
    artifact = compile_cuda(compiler, tmp_path)
    with NativeAO(**basis_arguments(meta)) as basis:
        ids = {
            "identity": None,
            "subset": np.arange(0, basis.nao, 2, dtype=np.uintp),
            "empty": np.empty(0, dtype=np.uintp),
        }[selection]
        active = basis.nao if ids is None else len(ids)
        with CudaGrid(
            basis,
            artifact,
            order=2,
            tile_points=7,
            active_ao_capacity=max(1, active),
            ingredients=("rho", "gradient", "tau"),
        ) as grid:
            grid.set_density(arrays["density"])
            # Several panel lengths exercise arena stride changes and map reuse.
            for count in (7, 3, 1):
                points = cp.asarray(arrays["points"][:count])
                with grid.feature_task_device_points(
                    points.data.ptr, count, ids, ("rho", "gradient", "tau")
                ) as task:
                    view = task.view
                    assert view.nactive == active and view.nao == basis.nao
                    work = task.density_jets(4)

                    # Explicit copies are test diagnostics, outside production.
                    def download(
                        pointer: typing.Any,
                        shape: tuple[int, ...],
                        stream: int = view.stream,
                        owner: typing.Any = task,
                    ) -> np.ndarray:
                        address = ct.cast(pointer, ct.c_void_p).value
                        memory = cp.cuda.UnownedMemory(
                            address, int(np.prod(shape)) * 8, owner
                        )
                        values = cp.ndarray(
                            shape,
                            dtype=np.float64,
                            memptr=cp.cuda.MemoryPointer(memory, 0),
                        )
                        with cp.cuda.ExternalStream(stream):
                            return values.get()

                    features = download(view.features, (13, count))
                    contracted = (
                        download(work, (2, 4, count, active)) if active else None
                    )
                    selected = np.arange(basis.nao) if ids is None else ids
                    ao = arrays["ao_jets"][:4, :count, :][:, :, selected]
                    d = arrays["density"][:, selected[:, None], selected[None, :]]
                    products = np.einsum("jpm,smn->sjpn", ao, d)
                    rho = np.einsum("pm,spm->sp", ao[0], products[:, 0])
                    gradient = 2 * np.einsum("jpm,spm->sjp", ao[1:4], products[:, 0])
                    tau = 0.5 * np.einsum("jpm,sjpm->sp", ao[1:4], products[:, 1:4])
                    np.testing.assert_allclose(
                        features[[0, 5]], rho, atol=1e-11, rtol=1e-10
                    )
                    np.testing.assert_allclose(
                        features[[1, 2, 3, 6, 7, 8]].reshape(2, 3, count),
                        gradient,
                        atol=1e-11,
                        rtol=1e-10,
                    )
                    np.testing.assert_allclose(
                        features[[4, 9]], tau, atol=1e-11, rtol=1e-10
                    )
                    if contracted is not None:
                        np.testing.assert_allclose(
                            contracted, products, atol=1e-11, rtol=1e-10
                        )
                with pytest.raises(RuntimeError, match="expired"):
                    _ = task.view
            for invalid in ([1, 0], [0, 0], [-1], [basis.nao], [0.5]):
                with (
                    pytest.raises(ValueError, match="sorted unique in-range"),
                    grid.feature_task_device_points(
                        points.data.ptr, count, invalid, ("rho",)
                    ),
                ):
                    pass
            if active < basis.nao:
                with (
                    pytest.raises(ValueError, match="identity AO map exceeds"),
                    grid.feature_task_device_points(
                        points.data.ptr, count, None, ("rho",)
                    ),
                ):
                    pass
