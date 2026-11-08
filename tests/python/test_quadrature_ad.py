"""Independent mathematical gates for compiled radial and Legendre AD."""

import ctypes
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.integral.expr import Graph
from generativeqc_compiler.integral.quadrature import (
    rational_radial_roots,
    treutler_radial_roots,
)
from generativeqc_compiler.integral.scalar_python import scalar_callable
from generativeqc_compiler.xc.atomic_grid_native import emit_atomic_grid_header

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def native_quadrature(tmp_path_factory: Any, native_cxx: Any) -> Any:
    directory = tmp_path_factory.mktemp("quadrature-ad")
    (directory / "generated_atomic_grid.hpp").write_text(emit_atomic_grid_header())
    source = directory / "probe.cpp"
    source.write_text("""
#include "dft/grid.hpp"
#include "generated_atomic_grid.hpp"
extern "C" void legendre(unsigned count, double* nodes, double* weights) {
  generativeqc::core::System system;
  system.atoms = {{1, {0, 0, 0}}};
  generativeqc::dft::GridSpec spec;
  spec.version = 1;
  spec.radial_points = count;
  spec.angular_polar = 1;
  spec.angular_azimuth = 3;
  const generativeqc::dft::MolecularGrid grid(system, spec);
  for (unsigned index = 0; index < count; ++index) {
    const double radius = grid.points()[9 * index];
    nodes[index] = (radius - 1) / (radius + 1);
    const double measure = grid.weights()[3 * index] / (4 * std::acos(-1) / 3);
    weights[index] = measure * std::pow(1 - nodes[index], 2) / (2 * radius * radius);
  }
}
extern "C" void radial(double node, double scale, double weight, double* out) {
  const auto value = generativeqc::generated::atomic_grid::rational_radial(node, scale, weight);
  std::copy(value.begin(), value.end(), out);
}
extern "C" void treutler(double coordinate, double scale, double* out) {
  const auto value = generativeqc::generated::atomic_grid::radial_transform(coordinate, scale);
  std::copy(value.begin(), value.end(), out);
}
extern "C" void partition(unsigned version, unsigned iterations, const double* positions,
                          double* energy, double* gradient) {
  generativeqc::core::System system;
  for (unsigned atom = 0; atom < 3; ++atom)
    system.atoms.push_back({std::array<unsigned,3>{1,8,17}[atom],
                          {positions[3*atom], positions[3*atom+1], positions[3*atom+2]}});
  generativeqc::dft::GridSpec spec;
  spec.version = version;
  spec.radial_points = version == 4 ? 0 : 5;
  spec.angular_polar = version == 4 ? 0 : 3;
  spec.angular_azimuth = version == 4 ? 0 : 6;
  spec.partition_iterations = iterations;
  spec.element_radii[1] = 1.2;
  spec.element_radii[8] = 1.5;
  spec.element_radii[17] = 1.8;
  const generativeqc::dft::MolecularGrid grid(system, spec);
  std::vector<double> seeds(grid.point_count());
  *energy = 0;
  for (std::size_t index = 0; index < seeds.size(); ++index) {
    seeds[index] = std::sin(double(index) / 17) / 1024;
    *energy += grid.weights()[index] * seeds[index];
  }
  const auto result = grid.contract_weight_derivative(seeds);
  std::copy(result.begin(), result.end(), gradient);
}
""")
    library = directory / "quadrature.so"
    native_cxx.build_shared(
        (ROOT / "src/dft/grid.cpp", source),
        library,
        compile_args=(
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            "-I",
            str(directory),
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
        ),
    )
    loaded = ctypes.CDLL(str(library))
    loaded.legendre.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
    loaded.radial.argtypes = [ctypes.c_double] * 3 + [ctypes.c_void_p]
    loaded.treutler.argtypes = [ctypes.c_double] * 2 + [ctypes.c_void_p]
    loaded.partition.argtypes = [ctypes.c_uint] * 2 + [ctypes.c_void_p] * 3
    return loaded


@pytest.mark.parametrize("count", (1, 2, 8, 64, 256, 512))
def test_native_legendre_primal_roots_weights_and_moments(
    native_quadrature: Any, count: int
) -> None:
    mp = pytest.importorskip("mpmath").mp.clone()
    mp.dps = 80
    nodes, weights = np.empty(count), np.empty(count)
    native_quadrature.legendre(count, nodes.ctypes.data, weights.ctypes.data)
    assert np.all(weights > 0) and np.all(np.abs(nodes) < 1)
    np.testing.assert_allclose(nodes, -nodes[::-1], atol=4e-16, rtol=0)
    for degree in range(min(2 * count, 32)):
        expected = 0 if degree % 2 else 2 / (degree + 1)
        assert weights @ nodes**degree == pytest.approx(expected, abs=3e-14)
    for index in sorted({0, count // 2, count - 1}):
        coordinate = mp.mpf(float(nodes[index]))
        polynomial = lambda value: mp.legendre(count, value)
        assert abs(polynomial(coordinate)) < mp.mpf("2e-11")
        slope = mp.diff(polynomial, coordinate)
        expected = 2 / ((1 - coordinate * coordinate) * slope * slope)
        assert weights[index] == pytest.approx(float(expected), rel=3e-11, abs=1e-15)


@pytest.mark.parametrize("count", (1, 8, 64, 512))
def test_radial_native_and_vectorized_ad_against_primal_only_oracle(
    native_quadrature: Any, count: int
) -> None:
    mp = pytest.importorskip("mpmath").mp.clone()
    mp.dps = 80
    nodes, weights = np.polynomial.legendre.leggauss(count)
    scale = 1.37
    graph, roots = rational_radial_roots()
    radii, measures = scalar_callable(graph, roots, ("node", "scale", "weight"), np)(
        nodes, scale, weights
    )
    out = np.empty(2)
    for index in sorted({0, count // 2, count - 1}):
        node, weight = float(nodes[index]), float(weights[index])
        # Qualify the AD at the stored affine-map coordinate. Near the largest
        # node, rounding node+1 is amplified by the fourth-order measure pole.
        unit = mp.mpf(float(0.5 * (node + 1)))
        primal = lambda value: mp.mpf(scale) * value / (1 - value)
        radius = primal(unit)
        expected = [
            float(radius),
            float(mp.mpf(weight) * radius * radius * mp.diff(primal, unit) / 2),
        ]
        native_quadrature.radial(node, scale, weight, out.ctypes.data)
        np.testing.assert_allclose(out, expected, rtol=3e-15, atol=1e-15)
        np.testing.assert_allclose([radii[index], measures[index]], out, rtol=2e-15)


@pytest.mark.parametrize("count", (1, 8, 64, 512))
def test_treutler_joint_ad_native_and_numpy_against_primal_only_oracle(
    native_quadrature: Any, count: int
) -> None:
    mp = pytest.importorskip("mpmath").mp.clone()
    mp.dps = 80
    graph, roots = treutler_radial_roots()
    coordinates = np.cos(np.arange(count, 0, -1) * (math.pi / (count + 1)))
    scale = 1.4
    radii, derivatives = scalar_callable(graph, roots, ("x", "xi"), np)(
        coordinates, scale
    )
    out = np.empty(2)
    for index in sorted({0, count // 2, count - 1}):
        coordinate = float(coordinates[index])
        primal = lambda value: (
            -mp.mpf(scale)
            / mp.log(2)
            * (1 + value) ** mp.mpf("0.6")
            * mp.log((1 - value) / 2)
        )
        expected = [
            float(primal(mp.mpf(coordinate))),
            float(mp.diff(primal, mp.mpf(coordinate))),
        ]
        native_quadrature.treutler(coordinate, scale, out.ctypes.data)
        np.testing.assert_allclose(out, expected, rtol=2e-13, atol=2e-15)
        np.testing.assert_allclose([radii[index], derivatives[index]], out, rtol=3e-15)


def test_python_lowering_rejects_unsupported_branches_and_argument_names() -> None:
    graph = Graph()
    coordinate = graph.variable("coordinate")
    with pytest.raises(ValueError, match="arguments"):
        scalar_callable(graph, (coordinate,), ("not-an-identifier",), math)
    root = graph.select_le(coordinate, 0, coordinate, -coordinate)
    with pytest.raises(ValueError, match="unsupported"):
        scalar_callable(graph, (root,), ("coordinate",), math)


@pytest.mark.parametrize("version", (1, 2, 3, 4))
@pytest.mark.parametrize("iterations", (1, 3, 5))
def test_native_partition_response_against_primal_weight_differences(
    native_quadrature: Any, version: int, iterations: int
) -> None:
    """Differentiate only the independent native grid weights, including v4 radii."""
    positions = np.array([[0.2, -0.3, 0.1], [0.4, 0.5, 2.7], [-2.3, 0.8, -0.6]])

    def evaluate(coordinates: np.ndarray) -> tuple[float, np.ndarray]:
        energy = ctypes.c_double()
        gradient = np.empty_like(coordinates)
        native_quadrature.partition(
            version,
            iterations,
            coordinates.ctypes.data,
            ctypes.byref(energy),
            gradient.ctypes.data,
        )
        return energy.value, gradient

    _, gradient = evaluate(positions)
    np.testing.assert_allclose(gradient.sum(axis=0), 0, atol=2e-12)
    for step in (1e-4, 5e-5, 2.5e-5):
        reference = np.empty_like(positions)
        for coordinate in range(positions.size):
            plus, minus = positions.copy(), positions.copy()
            plus.flat[coordinate] += step
            minus.flat[coordinate] -= step
            reference.flat[coordinate] = (evaluate(plus)[0] - evaluate(minus)[0]) / (
                2 * step
            )
        np.testing.assert_allclose(gradient, reference, rtol=3e-6, atol=3e-8)
