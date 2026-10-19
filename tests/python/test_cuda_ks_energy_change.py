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


def test_stationary_confirmation_preserves_every_acceptance_gate(
    tmp_path: Path, required_native_cxx: Any
) -> None:
    """Only a finite energy-only miss may keep D for another physical rebuild."""
    source = tmp_path / "confirmation.cpp"
    source.write_text(
        r"""
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include "dft/energy_change.hpp"

int main() {
  using generativeqc::dft::detail::stationary_density_needs_energy_confirmation;
  const auto confirm = [](bool history, const std::array<double, 7>& values) {
    return stationary_density_needs_energy_confirmation(
        history, values[0], values[1], values[2], values[3], values[4], values[5], values[6]);
  };
  const std::array<double, 7> stationary{3.6e-12, 7.7e-13, 4.6e-14, 9e-11,
                                         1e-12, 1e-10, 1e-10};
  assert(confirm(true, stationary));
  assert(!confirm(false, stationary));
  auto changed = stationary;
  changed[0] = std::nextafter(stationary[4], 0.0);
  assert(!confirm(true, changed));
  changed[0] = stationary[4];
  assert(confirm(true, changed));
  for (unsigned diagnostic = 0; diagnostic < 4; ++diagnostic) {
    for (double invalid : {-1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
      changed = stationary;
      changed[diagnostic] = invalid;
      assert(!confirm(true, changed));
    }
  }
  for (unsigned diagnostic = 1; diagnostic < 4; ++diagnostic) {
    changed = stationary;
    changed[diagnostic] = stationary[diagnostic == 1 ? 5 : 6];
    assert(!confirm(true, changed));
  }
  for (unsigned tolerance = 4; tolerance < 7; ++tolerance) {
    changed = stationary;
    changed[tolerance] = 0.0;
    assert(!confirm(true, changed));
  }
}
""",
        encoding="utf-8",
    )
    executable = required_native_cxx.build_executable(
        [source],
        tmp_path / "confirmation",
        compile_args=["-std=c++20", f"-I{ROOT / 'src'}"],
    )
    subprocess.run([executable], check=True, capture_output=True, text=True, timeout=10)


@pytest.fixture(scope="module")
def cuda_energy_module(tmp_path_factory: pytest.TempPathFactory) -> Any:
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
extern "C" __global__ void energy_confirmation(const double* inputs, int* output) {
  output[0] = generativeqc::dft::detail::stationary_density_needs_energy_confirmation(
      inputs[0] != 0.0, inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7]);
}
"""
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("NVCC", str(launcher))
        patch.setenv("CCACHE_BASEDIR", str(ROOT))
        module = cupy.RawModule(code=source, backend="nvcc", options=("-std=c++17",))
        return module


@pytest.mark.parametrize("values", CASES)
def test_cuda_energy_delta_matches_decimal(
    cuda_energy_module: Any, values: tuple[float, ...]
) -> None:
    import cupy as cp

    inputs, output = cp.asarray(values, dtype=cp.float64), cp.empty(1, dtype=cp.float64)
    cuda_energy_module.get_function("energy_delta")((1,), (1,), (inputs, output))
    assert float(cp.asnumpy(output)[0]) == reference_change(values)


@pytest.mark.parametrize(
    "changed_index,changed_value,expected",
    [
        (0, 1.0, True),
        (0, 0.0, False),
        (1, 1e-12, True),
        (1, math.nextafter(1e-12, 0.0), False),
        (1, math.inf, False),
        (1, math.nan, False),
        (2, 1e-10, False),
        (3, 1e-10, False),
        (4, 1e-10, False),
        (2, math.nan, False),
        (3, math.nan, False),
        (4, math.nan, False),
        (5, 0.0, False),
        (6, 0.0, False),
        (7, 0.0, False),
    ],
)
def test_cuda_stationary_confirmation_requires_the_physical_gates(
    cuda_energy_module: Any,
    changed_index: int,
    changed_value: float,
    expected: bool,
) -> None:
    """Execute the same confirmation predicate on a Slurm-owned real device."""
    import cupy as cp

    values = [1.0, 3.6e-12, 7.7e-13, 4.6e-14, 9e-11, 1e-12, 1e-10, 1e-10]
    values[changed_index] = changed_value
    inputs = cp.asarray(values, dtype=cp.float64)
    output = cp.empty(1, dtype=cp.int32)
    cuda_energy_module.get_function("energy_confirmation")((1,), (1,), (inputs, output))
    assert bool(cp.asnumpy(output)[0]) is expected
