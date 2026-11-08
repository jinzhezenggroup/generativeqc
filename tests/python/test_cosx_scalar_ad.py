"""Independent primal-only qualification of COSX scalar AD contractions."""

import ctypes
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.method.cosx_derivative_runtime import (
    build_cosx_esp_derivative_update_program,
    build_cosx_molecular_ao_update_program,
    build_cosx_point_gradient_update_program,
)
from generativeqc_compiler.tensor import execute


@pytest.fixture(scope="module", params=("interpreter", "cuda"))
def cosx_evaluator(request: Any, native_cxx: Any, tmp_path_factory: Any) -> Any:
    """Exercise the actual emitted device helpers, not CUDA copies of formulas."""
    programs = (
        build_cosx_point_gradient_update_program,
        build_cosx_esp_derivative_update_program,
        build_cosx_molecular_ao_update_program,
    )
    if request.param == "interpreter":
        return lambda kind, feeds: execute(programs[kind](), feeds).outputs
    if os.environ.get("GENERATIVEQC_RUN_CUDA_TESTS") != "1":
        pytest.skip("scheduled CUDA qualification is opt-in")
    assert os.environ.get("SLURM_JOB_ID"), "use finite Slurm allocation"
    nvcc = shutil.which("nvcc")
    assert nvcc is not None
    directory = tmp_path_factory.mktemp("cosx-scalar-ad")
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(
                Path(__file__).resolve().parents[2]
                / "tools/generate_cosx_derivative_native.py"
            ),
            "--output",
            str(directory / "generated_cosx_derivative_contractions.cuh"),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    source = directory / "probe.cu"
    source.write_text(r"""
#include <cuda_runtime.h>
#include "generated_cosx_derivative_contractions.cuh"
__global__ void run(int kind, const double* input, double* output) {
  using namespace generativeqc::dft::generated_cosx_derivative;
  bool success = false;
  if (kind == 0)
    success = cosx_point_gradient_update_tensor(input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7], input[8], input[9], output[0]);
  if (kind == 1)
    success = cosx_esp_derivative_update_tensor(input[0], input[1], input[2], input[3], input[4], output[0]);
  if (kind == 2)
    success = cosx_molecular_ao_update_tensor(input[0], input[1], input[2], input[3], input[4], input[5], output[0], output[1]);
  output[2] = success ? 0.0 : 1.0;
}
extern "C" int probe(int kind, const double* input, double* output) {
  double* device{};
  if (cudaMalloc(&device, 13 * sizeof(double)) != cudaSuccess) return 1;
  int status = 1;
  if (cudaMemcpy(device, input, 10 * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess) {
    run<<<1,1>>>(kind, device, device + 10);
    if (cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
        cudaMemcpy(output, device + 10, 3 * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess)
      status = 0;
  }
  if (cudaFree(device) != cudaSuccess) status = 1;
  return status;
}
""")
    library = directory / "probe.so"
    replace(native_cxx, compiler=nvcc).build_shared(
        [source],
        library,
        compile_args=(
            "-forward-unknown-to-host-compiler",
            "-std=c++20",
            "-O3",
            "--fmad=false",
            "-arch=native",
            f"-I{directory}",
            f"-I{Path(__file__).resolve().parents[2] / 'src'}",
            f"-I{Path(__file__).resolve().parents[2] / 'include'}",
        ),
    )
    function = ctypes.CDLL(str(library)).probe
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    function.argtypes = (ctypes.c_int, pointer, pointer)
    function.restype = ctypes.c_int
    orders = (
        (
            "accumulator",
            "density",
            "phi_derivative_row",
            "potential_column",
            "phi_row",
            "potential_derivative_column",
            "phi_derivative_column",
            "potential_row",
            "phi_column",
            "potential_derivative_row",
        ),
        (
            "accumulator",
            "matrix_derivative",
            "projected_value",
            "matrix_value",
            "projected_derivative",
        ),
        (
            "from_left",
            "from_right",
            "density_rc",
            "density_cr",
            "potential",
            "left_potential",
        ),
    )

    def evaluate(kind: int, feeds: dict) -> dict:
        inputs, output = np.zeros(10), np.zeros(3)
        inputs[: len(orders[kind])] = [feeds[name] for name in orders[kind]]
        assert function(kind, inputs, output) == 0
        assert output[2] == 0
        return (
            {"updated": output[0]}
            if kind != 2
            else {"from_left_updated": output[0], "from_right_updated": output[1]}
        )

    return evaluate


def test_cosx_jvp_and_ao_pullbacks_against_high_precision_primals(
    cosx_evaluator: Any,
) -> None:
    mp = pytest.importorskip("mpmath").mp.clone()
    mp.dps = 70
    rng = np.random.default_rng(71341)
    for _ in range(30):
        values = {
            name: mp.mpf(float(value))
            for name, value in zip(
                (
                    "density",
                    "phi_row",
                    "phi_column",
                    "potential_row",
                    "potential_column",
                    "density_rc",
                    "density_cr",
                ),
                rng.normal(size=7),
                strict=True,
            )
        }
        tangents = {
            name: mp.mpf(float(rng.normal()))
            for name in ("phi_row", "phi_column", "potential_row", "potential_column")
        }

        def point_primal(
            step: object, values: dict = values, tangents: dict = tangents
        ) -> object:
            moved = {
                name: value + step * tangents.get(name, 0)
                for name, value in values.items()
            }
            return (
                moved["density"]
                * (
                    moved["phi_row"] * moved["potential_column"]
                    + moved["phi_column"] * moved["potential_row"]
                )
                / 2
            )

        feeds = {
            name: np.asarray(float(values[name])) for name in ("density", *tangents)
        }
        feeds.update(
            {
                name.replace("_row", "_derivative_row").replace(
                    "_column", "_derivative_column"
                ): np.asarray(float(value))
                for name, value in tangents.items()
            }
        )
        feeds["accumulator"] = np.asarray(0.0)
        actual = cosx_evaluator(0, feeds)["updated"]
        np.testing.assert_allclose(
            actual, float(mp.diff(point_primal, 0)), atol=2e-15, rtol=3e-14
        )

        esp_feeds = {
            "accumulator": np.asarray(0.0),
            "matrix_value": feeds["phi_row"],
            "matrix_derivative": feeds["phi_derivative_row"],
            "projected_value": feeds["potential_column"],
            "projected_derivative": feeds["potential_derivative_column"],
        }
        expected = mp.diff(
            lambda step, values=values, tangents=tangents: (
                (values["phi_row"] + step * tangents["phi_row"])
                * (values["potential_column"] + step * tangents["potential_column"])
            ),
            0,
        )
        np.testing.assert_allclose(
            cosx_evaluator(1, esp_feeds)["updated"],
            float(expected),
            atol=2e-15,
            rtol=3e-14,
        )

        def molecular_primal(
            left: object, right: object, values: dict = values
        ) -> object:
            return (values["density_rc"] + values["density_cr"]) / 2 * left * values[
                "potential_row"
            ] + values["density_rc"] * right * values["potential_column"]

        actual = cosx_evaluator(
            2,
            {
                "from_left": np.asarray(0.0),
                "from_right": np.asarray(0.0),
                "density_rc": np.asarray(float(values["density_rc"])),
                "density_cr": np.asarray(float(values["density_cr"])),
                "potential": np.asarray(float(values["potential_row"])),
                "left_potential": np.asarray(float(values["potential_column"])),
            },
        )
        for name, orders in (
            ("from_left_updated", (1, 0)),
            ("from_right_updated", (0, 1)),
        ):
            np.testing.assert_allclose(
                actual[name],
                float(mp.diff(molecular_primal, (0, 0), orders)),
                atol=2e-15,
                rtol=3e-14,
            )
