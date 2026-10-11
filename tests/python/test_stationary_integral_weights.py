"""Portable generated stationary weights match the independent TensorIR executor."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)
from generativeqc_compiler.tensor import execute

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def native_weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate without runtime/providers and compile with a verified cache."""
    directory = tmp_path_factory.mktemp("stationary-weights")
    header = directory / "weights.hpp"
    guard = """
import importlib.abc, runpy, sys
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'generativeqc', 'pyscf', 'torch', 'cupy', 'numpy'}:
            raise AssertionError('generation imported runtime/provider: ' + fullname)
sys.meta_path.insert(0, NoRuntime())
script, output = sys.argv[1:]
sys.argv = [script, '--output', output]
runpy.run_path(script, run_name='__main__')
"""
    command = [
        sys.executable,
        "-c",
        guard,
        str(ROOT / "tools/generate_stationary_integral_weights.py"),
        str(header),
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    original = header.read_bytes()
    subprocess.run(command, check=True, cwd=ROOT)
    assert original == header.read_bytes()
    cache = shutil.which("sccache") or shutil.which("ccache")
    compiler = shutil.which("c++")
    if cache is None or compiler is None:
        pytest.skip("native weight comparison requires compiler and compiler cache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    source = directory / "probe.cpp"
    source.write_text(
        """
#include <array>
#include <iomanip>
#include <iostream>
#include <vector>
#include "weights.hpp"
int main() {
  std::size_t nbf;
  std::cin >> nbf;
  std::vector<double> density(nbf * nbf), weighted_density(nbf * nbf);
  for (auto& value : density) std::cin >> value;
  for (auto& value : weighted_density) std::cin >> value;
  std::array<std::int64_t, 4> indices;
  std::cout << std::setprecision(17);
  while (std::cin >> indices[0] >> indices[1] >> indices[2] >> indices[3]) {
    using namespace generativeqc::scf::generated;
    std::cout << pbe::one_electron(density.data(), weighted_density.data(), nbf, indices.data()) << ' '
              << pbe::overlap_pulay(density.data(), weighted_density.data(), nbf, indices.data()) << ' '
              << pbe::coulomb(density.data(), weighted_density.data(), nbf, indices.data()) << ' '
              << pbe0::exact_exchange(density.data(), weighted_density.data(), nbf, indices.data()) << ' '
              << b3lyp::exact_exchange(density.data(), weighted_density.data(), nbf, indices.data()) << '\\n';
  }
  return std::cin.eof() ? 0 : 1;
}
""",
        encoding="utf-8",
    )
    binary = directory / "probe"
    subprocess.run(
        [cache, compiler, "-std=c++20", str(source), "-o", str(binary)],
        check=True,
        timeout=60,
    )
    return binary


def test_portable_native_weights_match_shared_ad(native_weights: Path) -> None:
    """Asymmetric state and distinct indices expose exchange pairing mistakes."""
    rng = np.random.default_rng(2227)
    nbf = 5
    density, weighted_density = rng.normal(size=(2, nbf, nbf))
    indices = rng.integers(0, nbf, size=(41, 4))
    wire = " ".join(map(str, [nbf, *density.ravel(), *weighted_density.ravel()]))
    wire += "\n" + "\n".join(" ".join(map(str, row)) for row in indices) + "\n"
    response = subprocess.run(
        [str(native_weights)], input=wire, capture_output=True, text=True, check=True
    )
    actual = np.loadtxt(response.stdout.splitlines())
    for column, (method, source) in enumerate(
        (
            ("PBE0", "one_electron"),
            ("PBE0", "overlap_pulay"),
            ("PBE0", "coulomb"),
            ("PBE0", "exact_exchange"),
            ("B3LYP", "exact_exchange"),
        )
    ):
        plan = StationaryGradientPlan(
            resolve_method(method, spin="unpolarized"),
            StationaryMeanField(SCF_POINT_MODEL),
        )
        feeds = {
            "density_left": density[indices[:, 0], indices[:, 1]][None, :],
            "density_right": density[indices[:, 2], indices[:, 3]][None, :],
            "weighted_density": weighted_density[indices[:, 0], indices[:, 1]][None, :],
        }
        if source == "exact_exchange":
            feeds["density_left"] = density[indices[:, 0], indices[:, 2]][None, :]
            feeds["density_right"] = density[indices[:, 1], indices[:, 3]][None, :]
        program = plan.integral_block(source, terms=len(indices)).weights
        required = (
            ("weighted_density",)
            if source == "overlap_pulay"
            else ("density_left",)
            if source == "one_electron"
            else ("density_left", "density_right")
        )
        expected = execute(program, {name: feeds[name] for name in required}).outputs[
            "weights"
        ]
        np.testing.assert_allclose(actual[:, column], expected, atol=2e-15, rtol=2e-15)
