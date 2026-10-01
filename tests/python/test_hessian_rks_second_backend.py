"""Backend-policy checks for semilocal RKS generated second-integral HVPs."""

from __future__ import annotations

import pytest

from generativeqc.rks_hessian_integrals import checked_second_hvp_options


def test_rks_second_backend_requires_explicit_cuda_compiler() -> None:
    with pytest.raises(TypeError, match="explicit CudaCompilerAdapter"):
        checked_second_hvp_options("cuda", None, 0, 64 << 20)


def test_rks_second_backend_rejects_cpu_compiler_argument() -> None:
    with pytest.raises(ValueError, match="only meaningful for CUDA"):
        checked_second_hvp_options("cpu", object(), 0, 64 << 20)


@pytest.mark.parametrize("backend", ("gpu", "auto", ""))
def test_rks_second_backend_rejects_unknown_backend(backend: str) -> None:
    with pytest.raises(ValueError, match="must be cpu or cuda"):
        checked_second_hvp_options(backend, None, 0, 64 << 20)
