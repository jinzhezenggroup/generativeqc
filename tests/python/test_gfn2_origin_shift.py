"""Independent physical moment translation and additive generated VJP contracts."""

import ctypes
import os
import shutil
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.integral.gfn2_sdq_cpu import emit_gfn2_origin_shift


@pytest.fixture(scope="module", params=("host", "cuda"))
def shift_probe(native_cxx: Any, tmp_path_factory: Any, request: Any) -> Any:
    """Compile emitted mathematics; device execution requires a finite srun."""
    directory = tmp_path_factory.mktemp("gfn2-origin-shift")
    compiler = native_cxx
    cuda = request.param == "cuda"
    flags = ("-std=c++20", "-O3", "-ffp-contract=off")
    prefix, qualifier = "", ""
    if cuda:
        if os.environ.get("GENERATIVEQC_RUN_CUDA_TESTS") != "1":
            pytest.skip("scheduled CUDA qualification is opt-in")
        if not os.environ.get("SLURM_JOB_ID"):
            pytest.fail("device qualification requires finite srun")
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            pytest.fail("scheduled CUDA qualification requires nvcc")
        compiler = replace(native_cxx, compiler=nvcc)
        flags = (
            "-forward-unknown-to-host-compiler",
            "-std=c++20",
            "-O3",
            "--fmad=false",
            "-arch=native",
        )
        prefix, qualifier = "#include <cuda_runtime.h>\n", "__device__ "
    body = emit_gfn2_origin_shift().replace("inline void ", qualifier + "inline void ")
    source = directory / ("probe.cu" if cuda else "probe.cpp")
    source.write_text(
        prefix
        + body
        + "\n"
        + qualifier
        + r"""
void translated(const double* input, double* output) {
  gfn2_origin_shift(input, input[3], input + 4, input + 7, output, output + 3);
  for (unsigned index = 0; index < 7; ++index) output[9 + index] = input[22 + index];
  gfn2_origin_shift_pullback(input, input[3], input + 4, input + 13, input + 16,
                            output + 9, output + 10, output + 13);
}
"""
        + (
            r"""
__global__ void probe_kernel(const double* input, double* output) { translated(input, output); }
extern "C" int probe(const double* input, double* output) {
  double* device{};
  if (cudaMalloc(&device, 45 * sizeof(double)) != cudaSuccess) return 1;
  int status = 1;
  if (cudaMemcpy(device, input, 29 * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess) {
    probe_kernel<<<1, 1>>>(device, device + 29);
    if (cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
        cudaMemcpy(output, device + 29, 16 * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess)
      status = 0;
  }
  if (cudaFree(device) != cudaSuccess) status = 1;
  return status;
}
"""
            if cuda
            else r"""
extern "C" int probe(const double* input, double* output) { translated(input, output); return 0; }
"""
        )
    )
    library = directory / "probe.so"
    compiler.build_shared([source], library, compile_args=flags)
    loaded = ctypes.CDLL(str(library))
    function = loaded.probe
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    function.argtypes = (pointer, pointer)
    function.restype = ctypes.c_int

    def evaluate(inputs: np.ndarray) -> np.ndarray:
        output = np.empty(16)
        assert function(np.ascontiguousarray(inputs), output) == 0
        return output

    return evaluate


def _independent_translation(inputs: np.ndarray) -> np.ndarray:
    """Cartesian second-moment matrix translation, not the emitted packed algebra."""
    displacement, overlap, dipole = inputs[:3], inputs[3], inputs[4:7]
    quadrupole = inputs[7:13]
    shift = (
        np.outer(displacement, dipole)
        + np.outer(dipole, displacement)
        + overlap * np.outer(displacement, displacement)
    )
    projection = 1.5 * shift - 0.5 * np.trace(shift) * np.eye(3)
    first, second = (0, 0, 1, 0, 1, 2), (0, 1, 1, 2, 2, 2)
    return np.concatenate(
        (dipole + overlap * displacement, quadrupole + projection[first, second])
    )


@pytest.mark.parametrize("scale", (0.0, 1e-4, 1.0, 1e4))
def test_translated_moments_and_weighted_pullback(
    shift_probe: Any, scale: float
) -> None:
    """Complex-step the independent matrix primal, retaining additive seeds."""
    generator = np.random.default_rng(20261107)
    for _ in range(24):
        inputs = generator.normal(size=29)
        inputs[:3] *= scale
        actual = shift_probe(inputs)
        expected = _independent_translation(inputs)
        np.testing.assert_allclose(actual[:9], expected, rtol=3e-14, atol=2e-14)
        derivatives = []
        for coordinate in (3, 4, 5, 6, 0, 1, 2):
            perturbed = inputs.astype(complex)
            perturbed[coordinate] += 1e-25j
            derivatives.append(
                (_independent_translation(perturbed) @ inputs[13:22]).imag / 1e-25
            )
        np.testing.assert_allclose(
            actual[9:], inputs[22:] + derivatives, rtol=4e-13, atol=3e-13
        )
