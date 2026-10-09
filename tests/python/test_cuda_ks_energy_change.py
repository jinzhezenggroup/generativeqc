"""Independent decimal checks of the actual host/device energy-delta helper."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
CASES = (
    (-2569.0, 4e-13, -2569.0, -4e-13),
    (-2569.0, 6e-13, -2569.0, -6e-13),
    (-2569.0, 1e-13, math.nextafter(-2569.0, math.inf), -1e-13),
    (2.0**54, -0.125, 2.0**54, 0.125),
    (2.0**54, 0.0, -1.0, 0.0),
    (-2569.0, 0.0, math.inf, 0.0),
    (-2569.0, 4e-13, -2569.0, 4e-13),
)


def reference_change(values: tuple[float, ...]) -> float:
    """Subtract the exact binary64 inputs with an independent decimal oracle."""
    if math.isinf(values[2]):
        return math.inf
    with localcontext() as context:
        context.prec = 100
        current, correction, previous, previous_correction = map(
            Decimal.from_float, values
        )
        return float(abs(current + correction - previous - previous_correction))


@pytest.fixture(scope="module")
def energy_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("requires c++ and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    directory = tmp_path_factory.mktemp("ks-energy-change")
    source = directory / "probe.cpp"
    source.write_text(
        '#include "dft/energy_change.hpp"\n'
        "#include <cstdlib>\n#include <iomanip>\n#include <iostream>\n"
        "int main(int, char** arguments) {\n"
        "  std::cout << std::setprecision(17)\n"
        "    << generativeqc::dft::detail::electronic_energy_change(\n"
        "      std::strtod(arguments[1], nullptr), std::strtod(arguments[2], nullptr),\n"
        "      std::strtod(arguments[3], nullptr), std::strtod(arguments[4], nullptr));\n"
        "}\n"
    )
    executable = directory / "probe"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            "-I",
            str(ROOT / "src"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
    )
    return executable


@pytest.mark.parametrize("values", CASES)
def test_energy_delta_keeps_sub_ulp_changes(
    energy_probe: Path, values: tuple[float, ...]
) -> None:
    completed = subprocess.run(
        [str(energy_probe), *map(repr, values)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert float(completed.stdout) == reference_change(values)


@pytest.fixture(scope="module")
def cuda_energy_probe(tmp_path_factory: pytest.TempPathFactory) -> Any:
    if os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly Slurm-allocated GPU")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    cupy = pytest.importorskip("cupy")
    compiler, cache = shutil.which("nvcc"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.fail("CUDA energy-delta qualification requires nvcc and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    directory = tmp_path_factory.mktemp("ks-energy-change-device")
    launcher = directory / "nvcc"
    launcher.write_text(f'#!/bin/sh\nexec "{cache}" "{compiler}" "$@"\n')
    launcher.chmod(0o755)
    source = (
        (ROOT / "src/dft/energy_change.hpp").read_text()
        + "\n"
        + r"""
extern "C" __global__ void energy_delta(const double* inputs, double* output) {
  output[0] = generativeqc::dft::detail::electronic_energy_change(
      inputs[0], inputs[1], inputs[2], inputs[3]);
}
"""
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("NVCC", str(launcher))
        patch.setenv("CCACHE_BASEDIR", str(ROOT))
        module = cupy.RawModule(code=source, backend="nvcc", options=("-std=c++17",))
        return module.get_function("energy_delta")


@pytest.mark.parametrize("values", CASES)
def test_cuda_energy_delta_matches_decimal(
    cuda_energy_probe: Any, values: tuple[float, ...]
) -> None:
    import cupy as cp

    inputs, output = cp.asarray(values, dtype=cp.float64), cp.empty(1, dtype=cp.float64)
    cuda_energy_probe((1,), (1,), (inputs, output))
    assert float(cp.asnumpy(output)[0]) == reference_change(values)
