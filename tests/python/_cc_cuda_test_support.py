"""Explicit compiler admission for CUDA RCCSD(T) response qualification."""

import os

import pytest
from generativeqc.profiles import find_nvcc
from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_target import cuda_target_info


def response_gradient_compiler() -> CudaCompilerAdapter:
    """Require a scheduled, explicitly targeted GPU before compiler lookup."""
    assert os.environ.get("SLURM_JOB_ID"), (
        "CUDA RCCSD(T) response-gradient qualification requires Slurm"
    )
    architecture = os.environ.get("GENERATIVEQC_TENSOR_ARCH", "").strip()
    if not architecture:
        pytest.fail(
            "set GENERATIVEQC_TENSOR_ARCH to the allocated GPU architecture; "
            "CUDA RCCSD(T) qualification must not assume a device target"
        )
    nvcc = find_nvcc()
    assert nvcc is not None
    return CudaCompilerAdapter(nvcc, cuda_target_info(architecture))
