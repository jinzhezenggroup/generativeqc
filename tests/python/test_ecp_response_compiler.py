"""Independent signed layer energy and Taylor-response gates for native ECP."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.array_graph import evaluate_array_graph
from generativeqc_compiler.integral.ecp_fixed_grid import ecp_fixed_quadratures
from generativeqc_compiler.method.ecp_response import (
    ecp_ao_motion,
    ecp_layer_maps,
    ecp_total_density,
)
from generativeqc_compiler.tensor import execute


@pytest.mark.parametrize("points,nao", ((3, 2), (5, 4), (16, 7)))
def test_signed_layer_energy_vjp_jvp(points: int, nao: int) -> None:
    random = np.random.default_rng(821)
    values, motion = random.normal(size=(2, points, nao))
    density, delta_density = random.normal(size=(2, nao, nao))
    density += density.T
    delta_density += delta_density.T
    measure = random.normal(size=points)
    feeds = {
        "density": density,
        "d_density": delta_density,
        "ao0": values,
        "d_ao0": motion,
        "measure": measure,
        "bar_energy": np.array(1.0),
    }
    programs = ecp_layer_maps(points, nao)
    matrix = execute(programs["matrix"], feeds).outputs
    weights = execute(programs["weights"], feeds).outputs

    def oracle(ao: np.ndarray, operator_density: np.ndarray) -> tuple:
        potential = ao.T @ (measure[:, None] * ao)
        cotangent = measure[:, None] * (ao @ operator_density + ao @ operator_density.T)
        return potential, cotangent, float(np.sum(potential * operator_density))

    expected_matrix, expected_weights, _ = oracle(values, density)
    np.testing.assert_allclose(matrix["matrix"], expected_matrix, atol=2e-13)
    np.testing.assert_allclose(weights["weights"], expected_weights, atol=2e-13)
    for step in (3e-4, 1e-4):
        plus = oracle(values + step * motion, density + step * delta_density)
        minus = oracle(values - step * motion, density - step * delta_density)
        fixed_plus = oracle(values + step * motion, density)
        fixed_minus = oracle(values - step * motion, density)
        np.testing.assert_allclose(
            matrix["direction"],
            (fixed_plus[0] - fixed_minus[0]) / (2 * step),
            atol=5e-10,
        )
        np.testing.assert_allclose(
            weights["direction"], (plus[1] - minus[1]) / (2 * step), atol=5e-10
        )
        np.testing.assert_allclose(
            (plus[2] - minus[2]) / (2 * step),
            np.sum(expected_matrix * delta_density) + np.sum(expected_weights * motion),
            atol=1e-4,
            rtol=1e-6,
        )


def test_ecp_taylor_uses_order_two_spatial_jets() -> None:
    graph, roots = ecp_ao_motion()
    random = np.random.default_rng(463)
    for _ in range(10):
        jets = random.normal(size=10)
        left, right = random.normal(size=(2, 3))
        feeds = {
            **{f"x{axis}": 0 for axis in range(3)},
            **{f"jet{index}": value for index, value in enumerate(jets)},
            **{f"left{axis}": value for axis, value in enumerate(left)},
            **{f"right{axis}": value for axis, value in enumerate(right)},
        }
        hessian = np.array(
            [
                [jets[4], jets[5], jets[6]],
                [jets[5], jets[7], jets[8]],
                [jets[6], jets[8], jets[9]],
            ]
        )
        np.testing.assert_allclose(
            evaluate_array_graph(graph, roots, feeds),
            (left @ jets[1:4], right @ jets[1:4], left @ hessian @ right),
            atol=3e-15,
        )


@pytest.mark.parametrize("spins", (1, 2))
def test_physical_spin_sum_retains_signed_motion(spins: int) -> None:
    random = np.random.default_rng(498)
    density, motion = random.normal(size=(2, spins, 5, 5))
    result = execute(
        ecp_total_density(5, spins), {"density": density, "d_density": motion}
    ).outputs
    np.testing.assert_allclose(result["total"], density.sum(axis=0), atol=1e-15)
    np.testing.assert_allclose(result["direction"], motion.sum(axis=0), atol=1e-15)


def test_build_time_quadrature_orthonormality_and_radial_integrals() -> None:
    coarse, fine = ecp_fixed_quadratures()
    for grid, radial, points in ((coarse, 160, 2048), (fine, 224, 3872)):
        assert grid["radial"].shape == (radial, 2)
        assert grid["sphere"].shape == (points, 3)
        np.testing.assert_allclose(
            np.linalg.norm(grid["sphere"], axis=1), 1, atol=3e-15
        )
        np.testing.assert_allclose(grid["angular"].sum(), 4 * np.pi, atol=1e-14)
        harmonics = grid["projector"] / grid["angular"]
        np.testing.assert_allclose(
            grid["projector"] @ harmonics.T, np.eye(16), atol=7e-14
        )
        radius, weights = grid["radial"].T
        np.testing.assert_allclose(weights @ np.exp(-radius), 1, atol=3e-14)


@pytest.mark.parametrize("points,nao", ((0, 2), (3, 0), (True, 2), (3, 1.5)))
def test_ecp_response_extent_admission(points: int, nao: int) -> None:
    with pytest.raises(ValueError):
        ecp_layer_maps(points, nao)


def test_ecp_generation_is_build_only_and_deterministic(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    script = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'generativeqc', 'pyscf', 'torch', 'cupy'}:
            raise AssertionError('forbidden build import: ' + fullname)
sys.meta_path.insert(0, Guard())
from tools.generate_ecp_response import generate_maps, generate_quadrature, generate_density_maps
assert generate_maps() == generate_maps()
assert generate_quadrature() == generate_quadrature()
assert generate_density_maps() == generate_density_maps()
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        timeout=120,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root / "python") + os.pathsep + str(root)},
    )
