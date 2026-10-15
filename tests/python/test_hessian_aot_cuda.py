"""Actual Python-free native leaf execution; all GPU runs require Slurm.

Python supplies independent test oracles and launches a standalone binary. The
binary does not link/load the public runtime or an interpreter. These tests do
not qualify a native molecular Hessian, CPKS, DF factorization or ECP endpoint.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.dft.ao import jet_indices
from generativeqc_compiler.integral.shell_spec import cartesian_components

from tools.generate_hessian_aot import parse_inventory

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = parse_inventory(
    json.loads((ROOT / "manifests/hessian_aot_development.json").read_text())
)
pytestmark = pytest.mark.skipif(
    not os.environ.get("GENERATIVEQC_HESSIAN_AOT_PROBE"),
    reason="requires a built native Hessian AOT probe and an allocated Slurm GPU",
)


@pytest.fixture(scope="module")
def probe() -> Path:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests must run through srun"
    path = Path(os.environ["GENERATIVEQC_HESSIAN_AOT_PROBE"])
    assert path.is_file()
    return path


def _run(
    probe: Path, *arguments: object, values: object = (), env: dict | None = None
) -> object:
    process = subprocess.run(
        [str(probe), *(str(value) for value in arguments)],
        input=" ".join(
            format(float(value), ".17g") for value in np.asarray(values).reshape(-1)
        ),
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
        env=env,
    )
    return json.loads(process.stdout)


@pytest.fixture(scope="module")
def metadata(probe: Path) -> dict:
    return {item["name"]: item for item in _run(probe, "list")}


def _array(probe: Path, metadata: dict, name: str, feeds: dict) -> np.ndarray:
    names = metadata[name]["inputs"]
    assert set(feeds) == set(names)
    panels = np.broadcast_arrays(*(feeds[key] for key in names))
    count = panels[0].size
    values = np.array(_run(probe, "array", name, count, values=panels))
    return values.reshape(metadata[name]["outputs"], count)


def test_native_package_does_not_need_python_environment(
    probe: Path, metadata: dict, tmp_path: Path
) -> None:
    env = {
        **os.environ,
        "PATH": str(tmp_path),
        "PYTHONHOME": str(tmp_path / "absent"),
        "PYTHONPATH": str(tmp_path / "absent"),
    }
    assert _run(probe, "contract", env=env) == {"contract": "passed"}
    packaged_inventory = (
        parse_inventory(
            json.loads(
                Path(os.environ["GENERATIVEQC_HESSIAN_AOT_INVENTORY"]).read_text()
            )
        )
        if os.environ.get("GENERATIVEQC_HESSIAN_AOT_INVENTORY")
        else INVENTORY
    )
    assert tuple(metadata) == packaged_inventory.names
    for request in packaged_inventory.integrals:
        actual = metadata[request.name]
        assert actual["family"] == request.family
        assert actual["angular"] == list(request.angular)
        assert actual["radial"] == request.radial
        assert actual["omega"] == request.payload()["omega"]
    names = metadata["nuclear_pair_hvp"]["inputs"]
    feeds = {
        "charge_product": 2,
        "r0": 0.4,
        "r1": -0.2,
        "r2": 1.3,
        "dr0": 0.03,
        "dr1": -0.01,
        "dr2": 0.02,
    }
    result = np.array(
        _run(
            probe,
            "array",
            "nuclear_pair_hvp",
            1,
            values=[feeds[name] for name in names],
            env=env,
        )
    )
    displacement = np.array([feeds[f"r{index}"] for index in range(3)])
    direction = np.array([feeds[f"dr{index}"] for index in range(3)])
    radius = np.linalg.norm(displacement)
    expected = 2 * (
        3 * displacement * (displacement @ direction) / radius**5
        - direction / radius**3
    )
    np.testing.assert_allclose(result, expected, atol=1e-15, rtol=2e-14)


def test_native_integral_failure_publishes_no_partial_values(probe: Path) -> None:
    records = [-0.7, 1.1, 0.1, -0.2, 0.3, -0.4, 0.2, 1.1, 1]
    values = [*records, 0, 1, 0.03, -0.01, 0.04, -0.02, 0.05, -0.01]
    with pytest.raises(subprocess.CalledProcessError) as failure:
        _run(probe, "integral", "overlap_full_range_0_0_c0", 1, 2, values=values)
    assert failure.value.stdout == ""
    assert "no output published" in failure.value.stderr


def _moments(
    argument: complex, order: int, lower: float = 0, upper: float = 1
) -> complex:
    nodes, weights = np.polynomial.legendre.leggauss(64)
    points = lower + (upper - lower) * (nodes + 1) / 2
    return (
        np.dot(weights, points ** (2 * order) * np.exp(-argument * points**2))
        * (upper - lower)
        / 2
    )


def _overlap(exponents: np.ndarray, centers: np.ndarray, powers: str) -> complex:
    first, second = exponents
    total, reduced = first + second, first * second / (first + second)
    delta = centers[0] - centers[1]
    product_center = (first * centers[0] + second * centers[1]) / total
    nodes, weights = np.polynomial.hermite.hermgauss(8)
    value = np.exp(-reduced * np.sum(delta * delta))
    for axis, label in enumerate("xyz"):
        coordinate = product_center[axis] - centers[0, axis] + nodes / np.sqrt(total)
        value *= np.dot(weights, coordinate ** powers.count(label)) / np.sqrt(total)
    return value


def _primal(
    family: str, exponents: np.ndarray, centers: np.ndarray, component: str, radial: str
) -> complex:
    if family == "overlap":
        return _overlap(exponents, centers, component)
    if family == "kinetic":
        reduced = np.prod(exponents) / exponents.sum()
        delta = centers[0] - centers[1]
        return (
            reduced
            * (3 - 2 * reduced * np.sum(delta * delta))
            * _overlap(exponents, centers, "")
        )
    if family == "nuclear_attraction":
        total, reduced = exponents.sum(), np.prod(exponents) / exponents.sum()
        product_center = np.sum(exponents[:, None] * centers[:2], axis=0) / total
        delta, offset = centers[0] - centers[1], product_center - centers[2]
        return (
            -2
            * np.pi
            / total
            * np.exp(-reduced * np.sum(delta * delta))
            * _moments(total * np.sum(offset * offset), 0)
        )
    if family == "coulomb_metric":
        first, second = exponents
        product_center, other_center = centers
        decay = 1
        decay_first = np.zeros(3)
        derivative_scale = 1
    else:
        first, second = exponents[:2].sum(), exponents[2:].sum()
        product_center = np.sum(exponents[:2, None] * centers[:2], axis=0) / first
        other_center = np.sum(exponents[2:, None] * centers[2:], axis=0) / second
        delta = centers[0] - centers[1]
        decay = np.exp(-np.prod(exponents[:2]) / first * np.sum(delta * delta))
        decay_first = -2 * exponents[1] / first * delta
        derivative_scale = exponents[0] / first
        if family == "eri":
            delta = centers[2] - centers[3]
            decay *= np.exp(-np.prod(exponents[2:]) / second * np.sum(delta * delta))
    reduced = first * second / (first + second)
    offset = product_center - other_center
    argument = reduced * np.sum(offset * offset)
    boundary = 0.3 / math.sqrt(0.3**2 + reduced)
    lower, upper = (
        (0, boundary)
        if radial == "long_range"
        else (boundary, 1)
        if radial == "short_range"
        else (0, 1)
    )
    prefactor = 2 * np.pi**2.5 / (first * second * np.sqrt(first + second)) * decay
    value = _moments(argument, 0, lower, upper)
    if component:
        axis = "xyz".index(component)
        value = 0.5 * decay_first[
            axis
        ] * value - reduced * derivative_scale / exponents[0] * offset[axis] * _moments(
            argument, 1, lower, upper
        )
    return prefactor * value


def _gradient(primal: object, centers: np.ndarray) -> np.ndarray:
    result = np.empty(centers.shape)
    for index in np.ndindex(centers.shape):
        moved = centers.astype(complex)
        moved[index] += 1e-24j
        result[index] = primal(moved).imag / 1e-24
    return result


@pytest.mark.parametrize(
    "specification", INVENTORY.integrals, ids=lambda specification: specification.name
)
def test_native_integral_ad_against_independent_quadrature(
    specification: object, probe: Path
) -> None:
    request = specification
    centers = np.array(
        [
            [0.13, -0.31, 0.24],
            [-0.43, 0.27, 0.51],
            [0.68, -0.14, -0.22],
            [-0.21, 0.48, -0.63],
        ]
    )[: request.centers]
    direction = np.random.default_rng(1657).normal(size=centers.shape) * 0.1
    exponents = np.array([0.6, 0.8, 1.1, 0.9])[: len(request.angular)]
    records = [
        np.r_[exponents * scale, centers.ravel(), weight]
        for scale, weight in ((1, 0.73), (1.2, -0.17))
    ]
    values = np.r_[
        np.array(records).ravel(), np.arange(request.centers), direction.ravel()
    ]
    actual = np.array(
        _run(probe, "integral", request.name, 2, request.centers, values=values)
    ).reshape(len(request.components), -1)
    for packed, index in enumerate(request.components):
        component = cartesian_components(request.angular[0])[index]
        expected_value, expected_gradient, expected_hvp = (
            0,
            np.zeros_like(centers),
            np.zeros_like(centers),
        )
        for scale, weight in ((1, 0.73), (1.2, -0.17)):

            def primal(
                positions: np.ndarray,
                *,
                scale: float = scale,
                weight: float = weight,
                component: str = component,
            ) -> complex:
                return weight * _primal(
                    request.family,
                    exponents * scale,
                    positions,
                    component,
                    request.radial,
                )

            expected_value += primal(centers)
            expected_gradient += _gradient(primal, centers)
            step = 1e-4
            expected_hvp += (
                _gradient(primal, centers + step * direction)
                - _gradient(primal, centers - step * direction)
            ) / (2 * step)
        np.testing.assert_allclose(
            actual[packed, 0], expected_value, atol=2e-12, rtol=2e-12
        )
        np.testing.assert_allclose(
            actual[packed, 1 : 1 + 3 * request.centers].reshape(centers.shape),
            expected_gradient,
            atol=5e-12,
            rtol=5e-12,
        )
        np.testing.assert_allclose(
            actual[packed, 1 + 3 * request.centers :].reshape(centers.shape),
            expected_hvp,
            atol=8e-10,
            rtol=3e-8,
        )


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_native_xc_hessian_against_libxc_energy_gradient_oracle(
    spin: str, probe: Path, metadata: dict
) -> None:
    from generativeqc_compiler.method import resolve_method
    from pyscf.dft import libxc

    functional = resolve_method("WB97M-V", spin=spin).primitives[0].functional
    point = (
        np.array([0.5, 0.031, 0.14])
        if spin == "unpolarized"
        else np.array([0.3, 0.2, 0.015, 0.003, 0.01, 0.08, 0.05])
    )

    def reference(features: np.ndarray) -> tuple:
        if spin == "unpolarized":
            rho = np.array([features[0], np.sqrt(features[1]), 0, 0, 0, features[2]])[
                :, None
            ]
            energy, potential, *_ = libxc.eval_xc(
                "HYB_MGGA_XC_WB97M_V", rho, spin=0, deriv=1
            )
            return energy[0] * features[0], np.array(
                [potential[index][0] for index in (0, 1, 3)]
            )
        ra, rb, aa, ab, bb, ta, tb = features
        rho = np.array(
            [
                [ra, np.sqrt(aa), 0, 0, 0, ta],
                [rb, ab / np.sqrt(aa), np.sqrt(bb - ab * ab / aa), 0, 0, tb],
            ]
        )[:, :, None]
        energy, potential, *_ = libxc.eval_xc(
            "HYB_MGGA_XC_WB97M_V", rho, spin=1, deriv=1
        )
        return energy[0] * (ra + rb), np.r_[
            potential[0][0], potential[1][0], potential[3][0]
        ]

    feeds = dict(zip(functional.features, point, strict=True))
    name = f"xc_{spin}"
    actual = _array(
        probe, metadata, name, {key: feeds[key] for key in metadata[name]["inputs"]}
    )[:, 0]
    energy, gradient = reference(point)
    np.testing.assert_allclose(actual[0], energy, atol=2e-13, rtol=3e-12)
    np.testing.assert_allclose(
        actual[1 : 1 + len(point)], gradient, atol=2e-12, rtol=3e-11
    )
    hessian = np.empty((len(point), len(point)))
    for index in range(len(point)):
        motion = np.zeros_like(point)
        motion[index] = 1e-5
        hessian[:, index] = (
            reference(point + motion)[1] - reference(point - motion)[1]
        ) / (2e-5)
    packed = [
        hessian[first, second]
        for first, second in combinations_with_replacement(range(len(point)), 2)
    ]
    np.testing.assert_allclose(actual[1 + len(point) :], packed, atol=3e-7, rtol=2e-6)


def test_native_ao_jets_include_real_spherical_f_without_runtime_python(
    probe: Path,
) -> None:
    from numpy.polynomial import Polynomial

    centers = np.array([[0.1, -0.2, 0.3], [-0.4, 0.2, 1.1]])
    terms = [
        [(1, (0, 0, 0))],
        [(1, (0, 1, 0))],
        [(-1, (2, 0, 0)), (-1, (0, 2, 0)), (2, (0, 0, 2))],
        [(2, (0, 0, 3)), (-3, (2, 0, 1)), (-3, (0, 2, 1))],
    ]
    primitives = np.array([[0.7, 0.8], [1.1, -0.2]] * len(terms))
    records = np.zeros((len(terms), 16))
    for orbital, polynomial in enumerate(terms):
        records[orbital, :4] = (orbital % 2, 2 * orbital, 2, len(polynomial))
        for term, (coefficient, powers) in enumerate(polynomial):
            records[orbital, 4 + 4 * term : 8 + 4 * term] = (*powers, coefficient)
    points = np.array([[0.2, 0.3, -0.1], [-0.7, 0.4, 1.3]])
    actual = np.array(
        _run(
            probe,
            "ao",
            2,
            len(primitives),
            4,
            len(points),
            20,
            values=np.r_[
                centers.ravel(), primitives.ravel(), records.ravel(), points.ravel()
            ],
        )
    ).reshape(20, len(points), 4)
    expected = np.zeros_like(actual)
    for jet, derivative in enumerate(jet_indices(3)):
        for orbital, polynomial in enumerate(terms):
            displacement = points - centers[orbital % 2]
            for exponent, weight in primitives[2 * orbital : 2 * orbital + 2]:
                for coefficient, powers in polynomial:
                    value = np.full(len(points), coefficient * weight)
                    for axis in range(3):
                        factor = Polynomial([0] * powers[axis] + [1])
                        for _ in range(derivative[axis]):
                            factor = (
                                factor.deriv() - Polynomial([0, 2 * exponent]) * factor
                            )
                        value *= factor(displacement[:, axis])
                    expected[jet, :, orbital] += value * np.exp(
                        -exponent * np.sum(displacement**2, axis=1)
                    )
    np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=4e-13)


@pytest.mark.parametrize("kind", ("norm", "ratio", "log", "becke"))
def test_native_grid_mixed_ad_uses_complete_chain_rule(
    kind: str, probe: Path, metadata: dict
) -> None:
    names = (
        ("x", "y", "z")
        if kind == "norm"
        else ("a", "b")
        if kind == "ratio"
        else ("p",)
        if kind == "log"
        else ("mu",)
    )
    point = np.array([0.7, 0.4, 1.1])[: len(names)]
    left = np.array([0.03, -0.02, 0.01])[: len(names)]
    right = np.array([-0.01, 0.04, 0.02])[: len(names)]
    mixed = np.array([0.02, -0.01, 0.03])[: len(names)]

    def primal(value: np.ndarray) -> complex:
        if kind == "norm":
            return np.sqrt(np.sum(value * value))
        if kind == "ratio":
            return value[0] / value[1]
        if kind == "log":
            return np.log(value[0])
        coordinate = value[0]
        for _ in range(3):
            coordinate = 0.5 * coordinate * (3 - coordinate * coordinate)
        return 0.5 * (1 - coordinate)

    feeds = {
        prefix + key: value
        for prefix, values in (("", point), ("l", left), ("r", right), ("lr", mixed))
        for key, value in zip(names, values, strict=True)
    }
    actual = _array(probe, metadata, f"grid_mixed_{kind}", feeds)[:, 0]
    gradient = _gradient(primal, point)
    step = 1e-4
    moving_gradient = (
        _gradient(primal, point + step * right)
        - _gradient(primal, point - step * right)
    ) / (2 * step)
    expected = [
        primal(point),
        gradient @ left,
        gradient @ right,
        moving_gradient @ left + gradient @ mixed,
    ]
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-9)


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_native_xc_pullback_and_jvp_include_sigma_and_tau(
    spin: str, probe: Path, metadata: dict
) -> None:
    spins = 1 if spin == "unpolarized" else 2
    gradient = np.array([[0.2, -0.1, 0.3], [-0.2, 0.4, 0.1]])[:spins]
    motion = np.array([[-0.03, 0.02, 0.01], [0.04, 0.01, -0.02]])[:spins]
    potential = np.linspace(-0.3, 0.5, 3 if spins == 1 else 7)
    moving_potential = np.linspace(0.04, -0.03, len(potential))

    def reference(current: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        if spins == 1:
            return np.r_[
                coefficients[0], 2 * coefficients[1] * current[0], 0.5 * coefficients[2]
            ]
        return np.r_[
            coefficients[0],
            2 * coefficients[2] * current[0] + coefficients[3] * current[1],
            0.5 * coefficients[5],
            coefficients[1],
            2 * coefficients[4] * current[1] + coefficients[3] * current[0],
            0.5 * coefficients[6],
        ]

    feeds = {
        **{
            f"g{channel}_{axis}": gradient[channel, axis]
            for channel in range(spins)
            for axis in range(3)
        },
        **{
            f"dg{channel}_{axis}": motion[channel, axis]
            for channel in range(spins)
            for axis in range(3)
        },
        **{f"v{index}": value for index, value in enumerate(potential)},
        **{f"dv{index}": value for index, value in enumerate(moving_potential)},
    }
    for prefix in ("coefficients", "coefficient_response"):
        name = f"{prefix}_{spin}"
        actual = _array(
            probe, metadata, name, {key: feeds[key] for key in metadata[name]["inputs"]}
        )[:, 0]
        expected = reference(gradient, potential)
        if prefix == "coefficient_response":
            step = 1e-3
            expected = (
                reference(gradient + step * motion, potential + step * moving_potential)
                - reference(
                    gradient - step * motion, potential - step * moving_potential
                )
            ) / (2 * step)
        np.testing.assert_allclose(actual, expected, atol=8e-14, rtol=2e-12)


def test_native_vv10_pair_hvp_against_independent_real_space_oracle(
    probe: Path, metadata: dict
) -> None:
    from generativeqc_compiler.dft.nonlocal_reference import (
        nonlocal_energy_reference,
        nonlocal_explicit_geometry_derivatives_reference,
        nonlocal_feature_derivatives_reference,
    )
    from generativeqc_compiler.method import resolve_method

    spec = resolve_method("WB97M-V").primitives[-1].spec
    fields = np.array(
        [
            [0.4, 0.7],
            [0.02, -0.03],
            [0.04, 0.01],
            [-0.01, 0.05],
            [0.3, 0.5],
            [0.2, -0.1],
            [-0.4, 0.7],
            [0.1, -0.3],
            [0.6, 1.1],
        ]
    )
    direction = np.random.default_rng(7811).normal(size=fields.shape) * 0.03

    def reference(values: np.ndarray) -> tuple:
        rho, gradient, weights, points = (
            values[0],
            values[1:4].T,
            values[5],
            values[6:].T,
        )
        energy = nonlocal_energy_reference(points, weights, rho, gradient, spec)
        density_partial, sigma_partial = nonlocal_feature_derivatives_reference(
            points, weights, rho, gradient, spec
        )
        point_partial, weight_partial = (
            nonlocal_explicit_geometry_derivatives_reference(
                points, weights, rho, gradient, spec
            )
        )
        partials = np.vstack(
            (
                weights * density_partial,
                (weights[:, None] * 2 * sigma_partial[:, None] * gradient).T,
                np.zeros(len(rho)),
                weight_partial,
                point_partial.T,
            )
        )
        return energy, partials

    def pair_reference(values: np.ndarray) -> tuple:
        energy, partials = reference(values)
        first_energy, first_partials = reference(values[:, :1])
        second_energy, _ = reference(values[:, 1:])
        return energy - first_energy - second_energy, partials[:, 0] - first_partials[
            :, 0
        ]

    feeds = {
        f"{side}{field}": values[field, column]
        for side, values, column in (
            ("a", fields, 0),
            ("b", fields, 1),
            ("da", direction, 0),
            ("db", direction, 1),
        )
        for field in range(9)
    }
    actual = _array(
        probe,
        metadata,
        "vv10_pair",
        {key: feeds[key] for key in metadata["vv10_pair"]["inputs"]},
    )[:, 0]
    energy, partials = pair_reference(fields)
    step = 1e-4
    moving = (
        pair_reference(fields + step * direction)[1]
        - pair_reference(fields - step * direction)[1]
    ) / (2 * step)
    np.testing.assert_allclose(actual[0], energy, atol=3e-18, rtol=2e-12)
    np.testing.assert_allclose(actual[1:10], partials, atol=5e-18, rtol=3e-12)
    np.testing.assert_allclose(actual[10:], moving, atol=3e-14, rtol=3e-7)


@pytest.mark.parametrize("power", range(5))
def test_native_ecp_radial_measure_and_mapping(
    power: int, probe: Path, metadata: dict
) -> None:
    nodes, weights = np.polynomial.legendre.leggauss(9)
    radius, measure = _array(
        probe, metadata, "ecp_radial_map", {"z": nodes, "weight": weights}
    )
    np.testing.assert_allclose(radius, (1 + nodes) / (1 - nodes), atol=0, rtol=2e-15)
    np.testing.assert_allclose(
        measure, 2 * weights / (1 - nodes) ** 2, atol=0, rtol=2e-15
    )
    actual = _array(
        probe,
        metadata,
        f"ecp_radial_{power}",
        {
            "r": radius,
            "weight": measure,
            "alpha": np.full(9, 0.7),
            "coefficient": np.full(9, -1.3),
        },
    )[0]
    np.testing.assert_allclose(
        actual,
        -1.3 * measure * radius**power * np.exp(-0.7 * radius**2),
        atol=1e-16,
        rtol=5e-14,
    )


def test_native_ecp_real_harmonics_are_orthonormal_through_f(
    probe: Path, metadata: dict
) -> None:
    nodes, weights = np.polynomial.legendre.leggauss(8)
    azimuth = np.arange(16) * (2 * np.pi / 16)
    heights, azimuth = np.broadcast_arrays(nodes[:, None], azimuth[None])
    sphere = np.stack(
        (
            np.sqrt(1 - heights**2) * np.cos(azimuth),
            np.sqrt(1 - heights**2) * np.sin(azimuth),
            heights,
        ),
        axis=-1,
    ).reshape(-1, 3)
    angular_weights = np.repeat(weights * 2 * np.pi / 16, 16)
    harmonics = _array(
        probe, metadata, "ecp_harmonics", dict(zip("xyz", sphere.T, strict=True))
    )
    np.testing.assert_allclose(
        (harmonics * angular_weights) @ harmonics.T, np.eye(16), atol=8e-15, rtol=2e-14
    )
    np.testing.assert_allclose(
        harmonics[1:4], np.sqrt(3 / (4 * np.pi)) * sphere.T, atol=2e-16, rtol=2e-15
    )
