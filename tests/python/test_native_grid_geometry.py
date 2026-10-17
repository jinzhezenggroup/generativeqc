"""Build-only geometry AD and opt-in Slurm native moving-grid qualification.

The oracle differentiates polynomial Gaussians independently and evaluates
displaced long-double Becke *values*. No production derivative or runtime is
used to form expected nuclear derivatives.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.array_graph import evaluate_array_graph
from generativeqc_compiler.xc.grid_geometry_response import ao_motion_program
from numpy.polynomial import Polynomial


def _indices(order: int) -> tuple[tuple[int, int, int], ...]:
    """Ordinary libcint/CCA derivative layout, independently enumerated."""
    return (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (2, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (0, 2, 0),
        (0, 1, 1),
        (0, 0, 2),
        (3, 0, 0),
        (2, 1, 0),
        (2, 0, 1),
        (1, 2, 0),
        (1, 1, 1),
        (1, 0, 2),
        (0, 3, 0),
        (0, 2, 1),
        (0, 1, 2),
        (0, 0, 3),
    )[: (1, 4, 10, 20)[order]]


def _fixture(atoms: int = 3, iterations: int = 3, points_per_atom: int = 3) -> dict:
    """Normalized-ABI s/p/real-spherical d/f polynomial records, two primitives."""
    random = np.random.default_rng(1949 + atoms)
    centers = random.normal(size=(atoms, 3)) * 1.7
    terms = [
        [(1.0, (0, 0, 0))],
        [(0.8, (0, 1, 0))],
        [(-0.3, (2, 0, 0)), (-0.3, (0, 2, 0)), (0.6, (0, 0, 2))],
        [(0.4, (0, 0, 3)), (-0.6, (2, 0, 1)), (-0.6, (0, 2, 1))],
    ]
    primitives = np.array([[0.7, 0.8], [1.1, -0.2]] * len(terms))
    records = np.zeros((len(terms), 16))
    for orbital, polynomial in enumerate(terms):
        records[orbital, :4] = (orbital % atoms, 2 * orbital, 2, len(polynomial))
        for term, (coefficient, powers) in enumerate(polynomial):
            records[orbital, 4 + 4 * term : 8 + 4 * term] = (*powers, coefficient)
    owner = np.repeat(np.arange(atoms), points_per_atom)
    points = centers[owner] + random.normal(scale=0.6, size=(len(owner), 3))
    fixture = {
        "centers": centers,
        "primitives": primitives,
        "records": records,
        "points": points,
        "owner": owner,
        "iterations": iterations,
        "atomic": random.uniform(0.2, 0.7, len(owner)),
        "left": random.normal(scale=0.2, size=(atoms, 3)),
        "right": random.normal(scale=0.2, size=(atoms, 3)),
    }
    fixture["weights"] = _partition(fixture)
    return fixture


def _partition(fixture: dict, left: float = 0, right: float = 0) -> np.ndarray:
    """Independent unscreened long-double direct products at displaced geometry.

    Extended precision keeps the oracle distinct from the native shared-scaled
    four-jet product. Only values are computed here; derivatives use differences.
    """
    dtype = np.longdouble
    centers = (
        fixture["centers"].astype(dtype)
        + dtype(left) * fixture["left"].astype(dtype)
        + dtype(right) * fixture["right"].astype(dtype)
    )
    owner = fixture["owner"]
    points = (
        fixture["points"].astype(dtype)
        + dtype(left) * fixture["left"][owner].astype(dtype)
        + dtype(right) * fixture["right"][owner].astype(dtype)
    )
    distances = np.sqrt(np.sum((points[:, None] - centers) ** 2, axis=2))
    products = np.ones_like(distances)
    for first in range(len(centers)):
        for second in range(first):
            separation = np.sqrt(np.sum((centers[first] - centers[second]) ** 2))
            coordinate = np.clip(
                (distances[:, first] - distances[:, second]) / separation, -1, 1
            )
            for _ in range(fixture["iterations"]):
                coordinate = coordinate * (3 - coordinate * coordinate) / 2
            pair = np.clip((1 - coordinate) / 2, 0, 1)
            products[:, first] *= pair
            products[:, second] *= 1 - pair
    return (
        fixture["atomic"].astype(dtype)
        * products[np.arange(len(owner)), owner]
        / products.sum(axis=1)
    )


def _gaussians(fixture: dict, indices: tuple) -> np.ndarray:
    """Differentiate explicit polynomial times exp(-a*r^2), not compiler jets."""
    points, records = fixture["points"], fixture["records"]
    jets = np.zeros((len(indices), len(points), len(records)))
    for orbital, record in enumerate(records):
        atom, begin, count, terms = record[:4].astype(int)
        displacement = points - fixture["centers"][atom]
        for jet, derivative in enumerate(indices):
            for exponent, weight in fixture["primitives"][begin : begin + count]:
                for term in range(terms):
                    powers = record[4 + 4 * term : 7 + 4 * term].astype(int)
                    value = np.full(len(points), record[7 + 4 * term] * weight)
                    for axis in range(3):
                        factor = Polynomial([0] * powers[axis] + [1])
                        for _ in range(derivative[axis]):
                            factor = (
                                factor.deriv() - Polynomial([0, 2 * exponent]) * factor
                            )
                        value *= factor(displacement[:, axis])
                    jets[jet, :, orbital] += value * np.exp(
                        -exponent * np.sum(displacement**2, axis=1)
                    )
    return jets


def _ao_motion(fixture: dict, jets: np.ndarray) -> np.ndarray:
    """Independent first/mixed spatial tensor contraction, including xy twice."""
    lookup: dict[tuple[int, ...], int] = {
        powers: offset for offset, powers in enumerate(_indices(3))
    }
    ao_atoms = fixture["records"][:, 0].astype(int)
    motion = [
        fixture[name][fixture["owner"], None] - fixture[name][ao_atoms][None]
        for name in ("left", "right")
    ]
    result = np.zeros((3, 4, len(fixture["points"]), len(ao_atoms)))
    for base_index, base in enumerate(_indices(1)):
        for axis in range(3):
            shifted = tuple(
                degree + int(index == axis) for index, degree in enumerate(base)
            )
            for direction in range(2):
                result[direction, base_index] += (
                    motion[direction][:, :, axis] * jets[lookup[shifted]]
                )
            for other in range(3):
                second = tuple(
                    degree + int(index == other) for index, degree in enumerate(shifted)
                )
                result[2, base_index] += (
                    motion[0][:, :, axis]
                    * motion[1][:, :, other]
                    * jets[lookup[second]]
                )
    return result


def test_build_time_ao_motion_has_independent_polynomial_oracle() -> None:
    fixture = _fixture()
    jets = _gaussians(fixture, _indices(3))
    graph, roots = ao_motion_program()
    ao_atoms = fixture["records"][:, 0].astype(int)
    feeds = {f"x{axis}": 0.0 for axis in range(3)}
    feeds.update({f"jet{index}": jet for index, jet in enumerate(jets)})
    for name in ("left", "right"):
        motion = fixture[name][fixture["owner"], None] - fixture[name][ao_atoms][None]
        feeds.update({f"{name}{axis}": motion[:, :, axis] for axis in range(3)})
    actual = (
        np.array(evaluate_array_graph(graph, roots, feeds))
        .reshape(4, 3, len(fixture["points"]), len(ao_atoms))
        .transpose(1, 0, 2, 3)
    )
    np.testing.assert_allclose(
        actual, _ao_motion(fixture, jets), atol=3e-15, rtol=3e-14
    )


def test_geometry_aot_generator_is_deterministic_and_build_only(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    script = """
import runpy, sys
class Deny:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'generativeqc', 'cupy', 'torch', 'pyscf'}:
            raise RuntimeError('forbidden build-time runtime import: ' + fullname)
sys.meta_path.insert(0, Deny())
sys.argv = ['generate_grid_geometry_response.py', '--output', sys.argv[1]]
runpy.run_path('tools/generate_grid_geometry_response.py', run_name='__main__')
"""
    outputs = [tmp_path / name for name in ("first.cuh", "second.cuh")]
    for output in outputs:
        subprocess.run(
            [sys.executable, "-c", script, str(output)],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=30,
        )
    assert outputs[0].read_bytes() == outputs[1].read_bytes()


def test_native_host_geometry_admission_without_cuda_initialization() -> None:
    """Pure checked planning is usable with no device or runtime compiler."""
    executable = os.environ.get("GENERATIVEQC_CUDA_GRID_GEOMETRY_PROBE")
    if not executable:
        pytest.skip("set GENERATIVEQC_CUDA_GRID_GEOMETRY_PROBE for host admission")
    process = subprocess.run(
        [str(Path(executable).resolve()), "--host-check"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
        env=dict(
            os.environ,
            PATH="/nonexistent",
            PYTHONHOME="/nonexistent",
            PYTHONPATH="/nonexistent",
        ),
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["host_admission_without_cuda_initialization"]


def _native(
    fixture: dict,
    tmp_path: Path,
    *,
    reject: bool = False,
    point_tile: int | None = None,
    topology: str | None = None,
) -> tuple:
    executable = os.environ.get("GENERATIVEQC_CUDA_GRID_GEOMETRY_PROBE")
    if not executable:
        pytest.skip("set GENERATIVEQC_CUDA_GRID_GEOMETRY_PROBE inside finite Slurm job")
    path = Path(executable).resolve()
    assert path.is_file()
    source, output = tmp_path / "input.bin", tmp_path / "output.bin"
    atoms, primitives, nao = (
        len(fixture[name]) for name in ("centers", "primitives", "records")
    )
    with source.open("wb") as stream:
        np.array(
            [
                atoms,
                primitives,
                nao,
                len(fixture["points"]) // atoms,
                fixture["iterations"],
            ],
            dtype=np.uint64,
        ).tofile(stream)
        np.array([1e-12], dtype=np.float64).tofile(stream)
        for name in (
            "centers",
            "primitives",
            "records",
            "points",
            "weights",
            "atomic",
            "left",
            "right",
        ):
            np.asarray(fixture[name], dtype=np.float64).tofile(stream)
    process = subprocess.run(
        [
            str(path),
            str(source),
            str(output),
            *(
                [topology]
                if topology is not None
                else ["reject"]
                if reject
                else [str(point_tile)]
                if point_tile is not None
                else []
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=dict(
            os.environ,
            PATH="/nonexistent",
            PYTHONHOME="/nonexistent",
            PYTHONPATH="/nonexistent",
        ),
    )
    assert process.returncode == 0, process.stderr
    receipt = json.loads(process.stdout)
    if reject or topology == "understated":
        assert receipt["prepare_rejected"]
        return (), receipt
    values = np.fromfile(output, dtype=np.float64)
    panel = len(fixture["points"]) * nao
    points = len(fixture["points"])
    assert len(values) == 32 * panel + 9 * points
    owners = fixture["owner"]
    np.testing.assert_array_equal(
        values[32 * panel + 3 * points :].reshape(2, points, 3),
        np.stack((fixture["left"][owners], fixture["right"][owners])),
    )
    return (
        values[: 20 * panel].reshape(20, -1, nao),
        values[20 * panel : 32 * panel].reshape(3, 4, -1, nao),
        values[32 * panel : 32 * panel + 3 * points].reshape(3, -1),
    ), receipt


def _weight_differences(fixture: dict, step: float) -> np.ndarray:
    return np.array(
        [
            (_partition(fixture, step) - _partition(fixture, -step)) / (2 * step),
            (_partition(fixture, 0, step) - _partition(fixture, 0, -step)) / (2 * step),
            (
                _partition(fixture, step, step)
                - _partition(fixture, step, -step)
                - _partition(fixture, -step, step)
                + _partition(fixture, -step, -step)
            )
            / (4 * step**2),
        ],
        dtype=np.float64,
    )


def _wide_weight_differences(fixture: dict, step: float) -> np.ndarray:
    """60-digit displaced values avoid far-radial mixed-difference cancellation.

    In molecular radial tails a switching rounding unit can become 1e-10 in
    weighted units and then be amplified by 1/h^2. Form the entire difference
    before converting back to FP64; the derivative under test remains native.
    """
    with localcontext() as context:
        context.prec = 60
        centers, points, left, right = (
            [
                [Decimal.from_float(float(value)) for value in row]
                for row in fixture[name]
            ]
            for name in ("centers", "points", "left", "right")
        )
        atomic = [Decimal.from_float(float(value)) for value in fixture["atomic"]]
        step_wide = Decimal.from_float(step)
        zero = Decimal(0)

        def distance(first: list, second: list) -> Decimal:
            return sum(
                (first_component - second_component) ** 2
                for first_component, second_component in zip(first, second, strict=True)
            ).sqrt()

        def value(left_step: Decimal, right_step: Decimal) -> list:
            moved_centers = [
                [
                    value + left_step * dl + right_step * dr
                    for value, dl, dr in zip(center, dl_row, dr_row, strict=True)
                ]
                for center, dl_row, dr_row in zip(centers, left, right, strict=True)
            ]
            separations = {
                (first, second): distance(moved_centers[first], moved_centers[second])
                for first in range(len(centers))
                for second in range(first)
            }
            result = []
            for index, point in enumerate(points):
                owner = fixture["owner"][index]
                moved_point = [
                    coordinate + left_step * dl + right_step * dr
                    for coordinate, dl, dr in zip(
                        point, left[owner], right[owner], strict=True
                    )
                ]
                distances = [distance(moved_point, center) for center in moved_centers]
                products = [Decimal(1) for _ in centers]
                for first in range(len(centers)):
                    for second in range(first):
                        coordinate = min(
                            Decimal(1),
                            max(
                                Decimal(-1),
                                (distances[first] - distances[second])
                                / separations[first, second],
                            ),
                        )
                        for _ in range(fixture["iterations"]):
                            coordinate = coordinate * (3 - coordinate * coordinate) / 2
                        pair = min(Decimal(1), max(zero, (1 - coordinate) / 2))
                        products[first] *= pair
                        products[second] *= 1 - pair
                result.append(atomic[index] * products[owner] / sum(products))
            return result

        displaced = {
            (dl, dr): value(dl * step_wide, dr * step_wide)
            for dl, dr in (
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            )
        }
        return np.array(
            [
                [
                    (positive - negative) / (2 * step_wide)
                    for positive, negative in zip(
                        displaced[1, 0], displaced[-1, 0], strict=True
                    )
                ],
                [
                    (positive - negative) / (2 * step_wide)
                    for positive, negative in zip(
                        displaced[0, 1], displaced[0, -1], strict=True
                    )
                ],
                [
                    (pp - pm - mp + mm) / (4 * step_wide**2)
                    for pp, pm, mp, mm in zip(
                        displaced[1, 1],
                        displaced[1, -1],
                        displaced[-1, 1],
                        displaced[-1, -1],
                        strict=True,
                    )
                ],
            ],
            dtype=np.float64,
        )


def test_independent_displaced_value_oracles_agree() -> None:
    fixture = _fixture()
    for step in (0.004, 0.002):
        np.testing.assert_allclose(
            _wide_weight_differences(fixture, step),
            _weight_differences(fixture, step),
            atol=3e-12,
            rtol=1e-9,
        )


def test_native_topology_work_bound_is_checked_on_device(tmp_path: Path) -> None:
    """Tighter admission must count contracted AO incidences, not all primitives."""
    fixture = _fixture()
    baseline, original = _native(fixture, tmp_path)
    bounded, receipt = _native(fixture, tmp_path, topology="topology")
    for actual, expected in zip(bounded, baseline, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert receipt["ao_primitive_term_bound"] < original["ao_primitive_term_bound"]
    np.testing.assert_allclose(
        bounded[0], _gaussians(fixture, _indices(3)), atol=2e-13, rtol=2e-12
    )
    _native(fixture, tmp_path, topology="understated")


def test_native_point_parallel_tiles_preserve_geometry(tmp_path: Path) -> None:
    """A non-divisible, multi-tile grid preserves jets and independent values."""
    fixture = _fixture(points_per_atom=345)
    baseline, _ = _native(fixture, tmp_path, point_tile=64)
    parallel, _ = _native(fixture, tmp_path, point_tile=1024)
    for actual, expected in zip(parallel, baseline, strict=True):
        np.testing.assert_array_equal(actual, expected)
    expected = _gaussians(fixture, _indices(3))
    np.testing.assert_allclose(parallel[0], expected, atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(
        parallel[1], _ao_motion(fixture, expected), atol=2e-13, rtol=2e-12
    )
    coarse, fine = (_weight_differences(fixture, step) for step in (0.004, 0.002))
    np.testing.assert_allclose(
        parallel[2], (4 * fine - coarse) / 3, atol=2e-8, rtol=2e-6
    )


@pytest.mark.parametrize("iterations", [1, 2, 3, 4, 5])
def test_native_mixed_geometry_and_source_lifetime(
    tmp_path: Path, iterations: int
) -> None:
    fixture = _fixture(iterations=iterations)
    (spatial, motion, weights), receipt = _native(fixture, tmp_path)
    expected = _gaussians(fixture, _indices(3))
    np.testing.assert_allclose(spatial, expected, atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(
        motion, _ao_motion(fixture, expected), atol=2e-13, rtol=2e-12
    )
    differences = [_weight_differences(fixture, step) for step in (0.004, 0.002)]
    errors = [np.max(abs(weights - value)) for value in differences]
    assert errors[1] < 0.28 * errors[0]
    np.testing.assert_allclose(
        weights, (4 * differences[1] - differences[0]) / 3, atol=2e-9, rtol=2e-7
    )
    assert receipt["capacity_alias_finite_capture_revocation_gates"]
    assert receipt["prepared_ao_values"] == spatial.size
    assert (
        receipt["ao_primitive_term_bound"]
        == spatial.size * len(fixture["primitives"]) * 3
    )
    assert receipt["action_ao_values"] == motion.size
    pairs = len(fixture["points"]) * 3
    assert (
        receipt["prepared_partition_pairs"]
        == receipt["action_partition_pairs"]
        == pairs
    )


@pytest.mark.parametrize(
    "case",
    [
        "translation",
        "one_atom",
        "same_direction",
        "swap",
        "many_atoms",
        "rounded_zero",
        "scaled_measure",
        "zero_measure",
    ],
)
def test_native_geometry_symmetries_and_extended_domain(
    tmp_path: Path, case: str
) -> None:
    fixture = _fixture(
        atoms=1 if case == "one_atom" else 40 if case == "many_atoms" else 3
    )
    if case == "translation":
        fixture["left"][:] = (0.2, -0.1, 0.3)
        fixture["right"][:] = (-0.1, 0.3, 0.2)
    elif case == "one_atom":
        fixture["points"][0] = fixture["centers"][0]
    elif case == "same_direction":
        fixture["right"] = fixture["left"].copy()
    elif case == "rounded_zero":
        fixture["points"][0] = fixture["centers"][0] + np.array([1e-8, -1e-8, 1e-8])
        fixture["weights"] = _partition(fixture)
    elif case == "scaled_measure":
        fixture["atomic"] *= np.geomspace(1e-8, 1e6, len(fixture["atomic"]))
        fixture["weights"] = _partition(fixture)
    elif case == "zero_measure":
        fixture["atomic"][0] = 0
        fixture["weights"] = _partition(fixture)
    (spatial, motion, weights), receipt = _native(fixture, tmp_path)
    if case in ("translation", "one_atom"):
        np.testing.assert_array_equal(motion, 0)
        np.testing.assert_array_equal(weights, 0)
    else:
        np.testing.assert_allclose(
            motion,
            _ao_motion(fixture, _gaussians(fixture, _indices(3))),
            atol=3e-13,
            rtol=3e-12,
        )
        steps = [_weight_differences(fixture, step) for step in (0.002, 0.001)]
        np.testing.assert_allclose(
            weights, (4 * steps[1] - steps[0]) / 3, atol=2e-8, rtol=2e-6
        )
    if case in ("swap", "same_direction"):
        fixture["left"], fixture["right"] = fixture["right"], fixture["left"]
        (swapped_spatial, swapped_motion, swapped_weights), _ = _native(
            fixture, tmp_path
        )
        np.testing.assert_array_equal(swapped_spatial, spatial)
        np.testing.assert_allclose(swapped_motion, motion[[1, 0, 2]], atol=2e-15)
        np.testing.assert_allclose(swapped_weights, weights[[1, 0, 2]], atol=2e-15)
    assert receipt["capacity_alias_finite_capture_revocation_gates"]


@pytest.mark.parametrize(
    "case",
    [
        "coincident",
        "point_center",
        "wrong_measure",
        "wrong_weights",
        "negative_measure",
        "zero_measure_nonzero_weight",
        "negative_weight",
        "overflowing_spatial_jets",
        "invalid_ao_record",
        "fractional_ao_record",
        "invalid_primitive_range",
        "invalid_polynomial_count",
        "invalid_power",
        "invalid_exponent",
        "nonfinite",
    ],
)
def test_native_geometry_rejects_invalid_primal_without_publication(
    tmp_path: Path, case: str
) -> None:
    fixture = _fixture()
    if case == "coincident":
        fixture["centers"][1] = fixture["centers"][0]
    elif case == "point_center":
        fixture["points"][0] = fixture["centers"][0]
    elif case == "wrong_measure":
        fixture["atomic"][0] *= 2
    elif case == "wrong_weights":
        fixture["weights"][0] += 0.1
    elif case == "negative_measure":
        fixture["atomic"][0] = -0.1
    elif case == "zero_measure_nonzero_weight":
        fixture["atomic"][0] = 0
    elif case == "negative_weight":
        fixture["weights"][0] = -0.1
    elif case == "overflowing_spatial_jets":
        fixture["primitives"][:, 1] = 1e308
    elif case == "invalid_ao_record":
        fixture["records"][0, 0] = len(fixture["centers"])
    elif case == "fractional_ao_record":
        fixture["records"][0, 0] = 0.5
    elif case == "invalid_primitive_range":
        fixture["records"][0, 2] = len(fixture["primitives"]) + 1
    elif case == "invalid_polynomial_count":
        fixture["records"][0, 3] = 4
    elif case == "invalid_power":
        fixture["records"][0, 4] = 4
    elif case == "invalid_exponent":
        fixture["primitives"][0, 0] = 0
    elif case == "nonfinite":
        fixture["points"][0, 0] = np.nan
    _native(fixture, tmp_path, reject=True)


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("fitted", [False, True])
def test_actual_wb97mv_native_quadrature_geometry(spins: int, fitted: bool) -> None:
    """Borrow real converged native RKS/UKS Direct/DF owners, not copied labels."""
    executable = os.environ.get("GENERATIVEQC_KS_WB97MV_GRID_PROBE")
    if not executable:
        pytest.skip("set GENERATIVEQC_KS_WB97MV_GRID_PROBE inside finite Slurm job")
    process = subprocess.run(
        [
            str(Path(executable).resolve()),
            "rks" if spins == 1 else "uks",
            "df-geometry" if fitted else "geometry",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        env=dict(
            os.environ,
            PATH="/nonexistent",
            PYTHONHOME="/nonexistent",
            PYTHONPATH="/nonexistent",
        ),
    )
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    atoms, primitives, nao, points = (
        result[name] for name in ("atoms", "primitives", "nao", "point_count")
    )
    basis = np.array(result["basis"])
    fixture = {
        "centers": basis[: 3 * atoms].reshape(atoms, 3),
        "primitives": basis[3 * atoms : 3 * atoms + 2 * primitives].reshape(
            primitives, 2
        ),
        "records": basis[3 * atoms + 2 * primitives :].reshape(nao, 16),
        "points": np.array(result["points"]).reshape(points, 3),
        "owner": np.repeat(np.arange(atoms), result["points_per_atom"]),
        "iterations": result["partition_iterations"],
        "atomic": np.array(result["atomic"]),
        "left": np.array(result["left"]).reshape(atoms, 3),
        "right": np.array(result["right"]).reshape(atoms, 3),
    }
    np.testing.assert_allclose(
        np.array(result["weights"]) / fixture["atomic"],
        _partition(fixture) / fixture["atomic"],
        atol=3e-15,
        rtol=3e-11,
    )
    expected = _gaussians(fixture, _indices(3))
    np.testing.assert_array_equal(
        np.array(result["point_directions"]).reshape(2, points, 3),
        np.stack(
            (fixture["left"][fixture["owner"]], fixture["right"][fixture["owner"]])
        ),
    )
    np.testing.assert_allclose(
        np.array(result["spatial"]).reshape(20, points, nao),
        expected,
        atol=3e-13,
        rtol=3e-12,
    )
    np.testing.assert_allclose(
        np.array(result["geometry_ao"]).reshape(3, 4, points, nao),
        _ao_motion(fixture, expected),
        atol=3e-13,
        rtol=3e-12,
    )
    steps = [_wide_weight_differences(fixture, step) for step in (0.002, 0.001)]
    np.testing.assert_allclose(
        np.array(result["geometry_weights"]).reshape(3, points),
        (4 * steps[1] - steps[0]) / 3,
        atol=2e-8,
        rtol=2e-6,
    )
    assert result["revoked_without_publication"]
    assert result["geometry_action_pairs"] == points * atoms * (atoms - 1) // 2
