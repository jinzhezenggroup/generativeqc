"""Opt-in multi-GPU regression; run inside a scheduler allocation of two GPUs."""

from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path

import numpy as np
import pytest
from generativeqc._cuda_device import device_architecture
from generativeqc._cuda_runtime import preload_cuda_runtime_libraries
from generativeqc.profiles import find_nvcc
from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
from generativeqc_compiler.tensor.cuda_plan import plan_cuda
from generativeqc_compiler.tensor.cuda_resident import (
    PreparedResident,
    compile_resident,
)

from tools.validate_tensor_cuda_compile import _program

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires explicit allocated-GPU opt-in",
)


def test_nonzero_device_initialization_graph_and_resident(tmp_path: Path) -> None:
    """All entry points select their owner and restore the caller's current GPU."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    preload_cuda_runtime_libraries()
    runtime = ct.CDLL("libcudart.so.12")
    for name in ("cudaGetDevice", "cudaGetDeviceCount"):
        getattr(runtime, name).argtypes = [ct.POINTER(ct.c_int)]
        getattr(runtime, name).restype = ct.c_int
    runtime.cudaSetDevice.argtypes = [ct.c_int]
    runtime.cudaSetDevice.restype = ct.c_int
    count, previous = ct.c_int(), ct.c_int()
    assert runtime.cudaGetDeviceCount(ct.byref(count)) == 0
    if count.value < 2:
        pytest.skip("requires two visible GPUs")
    assert runtime.cudaGetDevice(ct.byref(previous)) == 0
    assert runtime.cudaSetDevice(0) == 0

    def restored() -> None:
        current = ct.c_int()
        assert runtime.cudaGetDevice(ct.byref(current)) == 0
        assert current.value == 0
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == visible

    try:
        nvcc = find_nvcc()
        assert nvcc is not None, "allocated-GPU tests require NVCC"
        compiler = CudaCompilerAdapter(nvcc, cuda_target_info(device_architecture(1)))
        cache = Path(os.environ.get("GENERATIVEQC_TENSOR_CACHE", str(tmp_path)))
        plan = plan_cuda(_program(), compiler.target)
        artifact = compile_cuda(plan, compiler, cache)
        values = np.asarray([0.2, 0.7, 1.4, -0.5, 0.3])
        expected = np.zeros(3)
        np.add.at(expected, [0, 0, 1, 2, 2], values * [1.25, -0.5, 2, 0.75, -1.5])
        for mode in ("ordinary", "cuda-graph"):
            with PreparedCuda(
                plan, artifact, device=1, execution_mode=mode
            ) as prepared:
                restored()
                for _ in range(3):
                    result = prepared.execute({"values": values})
                    np.testing.assert_allclose(
                        result.outputs["result"], expected, rtol=2e-13, atol=2e-13
                    )
                    restored()
                if mode == "cuda-graph":
                    assert result.metrics["graph_mode"] == "replay"
                    prepared.invalidate_graph()
                    restored()
            restored()
        extension = r"""
extern "C" int test_copy_input(void* src, void* dst, size_t bytes, char* error, size_t size) {
    using namespace generativeqc_resident;
    return copy(src, resident_inputs, 1, 0, dst, resident_inputs, 1, 0, bytes, error, size);
}
"""
        resident_artifact = compile_resident(plan, compiler, cache, extension=extension)
        with (
            PreparedResident(plan, resident_artifact, device=1) as src,
            PreparedResident(plan, resident_artifact, device=1) as dst,
        ):
            restored()
            src.upload({"values": values})
            restored()
            dst.upload({"values": np.zeros_like(values)})
            restored()
            copy = src._library.test_copy_input
            copy.argtypes = [
                ct.c_void_p,
                ct.c_void_p,
                ct.c_size_t,
                ct.c_char_p,
                ct.c_size_t,
            ]
            copy.restype = ct.c_int
            error = ct.create_string_buffer(1024)
            assert (
                copy(src._pointer, dst._pointer, values.nbytes, error, len(error)) == 0
            ), error.value
            restored()
            for prepared in (src, dst):
                leases, _ = prepared.run()
                restored()
                np.testing.assert_allclose(
                    prepared.download(leases["result"]),
                    expected,
                    rtol=2e-13,
                    atol=2e-13,
                )
                restored()
        restored()
    finally:
        assert runtime.cudaSetDevice(previous.value) == 0
