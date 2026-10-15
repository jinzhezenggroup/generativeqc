"""Native fixed-grid WB97M-V response: AD composition and independent oracles.

GPU cases are opt-in Slurm work. The standalone executable has no Python
dependency; Python/PySCF below supplies test data and an independent oracle only.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.array_graph import evaluate_array_graph
from generativeqc_compiler.dft.nonlocal_reference import (
    nonlocal_feature_derivatives_reference,
)
from generativeqc_compiler.method.spec import resolve_method
from generativeqc_compiler.method.stationary_grid_second import (
    StationaryGridSecondOrder,
)
from generativeqc_compiler.method.wb97mv_grid_response import grid_maps, point_program
from generativeqc_compiler.tensor import execute


def _fixture(spins: int = 2, points: int = 9, *, vacuum: bool = False) -> dict:
    """Independent Gaussian polynomial jets, including real-spherical d/f.

    Coefficients are already normalized basis records, as required by the
    production AoBasis ABI. No runtime/native AO implementation makes the oracle.
    """
    from numpy.polynomial import Polynomial

    random = np.random.default_rng(1945 + points)
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
    coordinates = random.normal(size=(points, 3)) * 0.7
    coordinates[-1] = 40
    if vacuum:
        coordinates[:] = 40
    jets = np.zeros((4, points, len(terms)))
    for jet, derivative in enumerate(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))):
        for orbital, polynomial in enumerate(terms):
            displacement = coordinates - centers[orbital % 2]
            for exponent, weight in primitives[2 * orbital : 2 * orbital + 2]:
                for coefficient, powers in polynomial:
                    value = np.full(points, coefficient * weight)
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
    nao = len(terms)
    factors = random.normal(scale=0.3, size=(spins, nao, nao))
    density = factors @ factors.transpose(0, 2, 1) + np.eye(nao)[None]
    direction = random.normal(scale=0.1, size=density.shape)
    direction = (direction + direction.transpose(0, 2, 1)) / 2
    if spins == 2:
        direction[1] = 0
    return {
        "spins": spins,
        "nao": nao,
        "centers": centers,
        "primitives": primitives,
        "basis": np.r_[centers.ravel(), primitives.ravel(), records.ravel()],
        "points": coordinates,
        "weights": random.uniform(-0.1, 0.3, points),
        "jets": jets,
        "density": density,
        "direction": direction,
    }


def _fields(jets: np.ndarray, density: np.ndarray) -> np.ndarray:
    """Independent direct Cartesian contractions (no compiler feature AD)."""
    return np.array(
        [
            [
                np.einsum("pi,ij,pj->p", jets[0], block, jets[0]),
                *(
                    2 * np.einsum("pi,ij,pj->p", jet, block, jets[0])
                    for jet in jets[1:]
                ),
                sum(np.einsum("pi,ij,pj->p", jet, block, jet) for jet in jets[1:]) / 2,
            ]
            for block in density
        ]
    )


def _assemble(jets: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Independent physical feature cotangent contraction, with tau one-half."""
    result = []
    for block in coefficients:
        matrix = np.einsum("pi,p,pj->ij", jets[0], block[0], jets[0])
        for axis in range(3):
            cross = np.einsum("pi,p,pj->ij", jets[axis + 1], block[axis + 1], jets[0])
            matrix += cross + cross.T
            matrix += np.einsum(
                "pi,p,pj->ij", jets[axis + 1], block[4] / 2, jets[axis + 1]
            )
        result.append(matrix)
    return np.array(result)


@pytest.mark.parametrize("spins", (1, 2))
def test_scalar_composition_preserves_work_policy_and_coefficients(spins: int) -> None:
    fixture = _fixture(spins)
    fields = _fields(fixture["jets"], fixture["density"]).reshape(5 * spins, -1)
    direction = _fields(fixture["jets"], fixture["direction"]).reshape(5 * spins, -1)
    fields = np.column_stack((fields, np.zeros(5 * spins)))
    direction = np.column_stack((direction, np.ones(5 * spins)))
    weights = np.r_[fixture["weights"], 0.2]
    method = resolve_method(
        "WB97M-V", spin="polarized" if spins == 2 else "unpolarized"
    )
    reference = StationaryGridSecondOrder(
        method.primitives[0].functional, method.primitives[-1]
    )
    values = np.vstack(
        (fields, weights, fixture["points"].T[:, np.r_[np.arange(len(weights) - 1), 0]])
    )
    moving = np.vstack((direction, np.zeros((4, len(weights)))))
    expected = reference.evaluate(values, moving).semilocal
    graph, roots = point_program(spins)
    actual = np.array(
        evaluate_array_graph(
            graph,
            roots,
            {
                **{f"f{index}": value for index, value in enumerate(fields)},
                **{f"d{index}": value for index, value in enumerate(direction)},
                "weight": weights,
            },
        )
    ).reshape(2, 5 * spins, -1)
    np.testing.assert_allclose(
        actual, np.array(expected)[:, : 5 * spins], atol=2e-12, rtol=2e-10
    )
    assert np.isfinite(actual).all()


def test_feature_vjp_matches_independent_contractions() -> None:
    fixture = _fixture()
    jets = fixture["jets"]
    maps = grid_maps(jets.shape[1], jets.shape[2])
    random = np.random.default_rng(1923)
    coefficients = random.normal(size=(2, 5, jets.shape[1]))
    for spin in range(2):
        feeds = {f"ao{index}": value for index, value in enumerate(jets)}
        result = execute(
            maps["features"], {**feeds, "density": fixture["direction"][spin]}
        ).outputs
        np.testing.assert_allclose(
            np.array([result[name] for name in ("rho", "g0", "g1", "g2", "tau")]),
            _fields(jets, fixture["direction"])[spin],
            atol=2e-15,
        )
        feeds.update(
            dict(
                zip(
                    ("bar_rho", "bar_g0", "bar_g1", "bar_g2", "bar_tau"),
                    coefficients[spin],
                    strict=True,
                )
            )
        )
        potential = execute(maps["potential"], feeds).outputs["potential"]
        np.testing.assert_allclose(
            potential, _assemble(jets, coefficients)[spin], atol=2e-14
        )


def test_generation_is_deterministic_and_runtime_free(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    script = """
import runpy, sys
class Deny:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'generativeqc', 'cupy', 'torch', 'pyscf'}:
            raise RuntimeError('forbidden build-time runtime import: ' + fullname)
sys.meta_path.insert(0, Deny())
sys.argv = ['generate_wb97mv_grid_response.py', '--output', sys.argv[1]]
runpy.run_path('tools/generate_wb97mv_grid_response.py', run_name='__main__')
"""
    outputs = []
    for index in range(2):
        path = tmp_path / f"grid{index}.cuh"
        subprocess.run(
            [os.sys.executable, "-c", script, str(path)],
            cwd=root,
            check=True,
            timeout=60,
        )
        outputs.append(path.read_bytes())
    assert outputs[0] == outputs[1]
    assert b"point_2" in outputs[0] and b"pairs(" in outputs[0]


@pytest.fixture
def probe() -> Path:
    value = os.environ.get("GENERATIVEQC_CUDA_WB97MV_GRID_PROBE")
    if not value:
        pytest.skip("requires optional native CUDA WB97M-V grid probe")
    path = Path(value)
    assert path.is_file()
    return path


def _native(
    probe: Path,
    fixture: dict,
    *,
    mode: str = "normal",
    tile: int = 3,
    pairs: int = 16_777_216,
    environment: dict | None = None,
) -> dict:
    assert os.environ.get("SLURM_JOB_ID"), "real-GPU tests must run through finite srun"
    values = np.concatenate(
        [
            fixture[name].ravel()
            for name in ("basis", "points", "weights", "density", "direction")
        ]
    )
    header = f"{len(fixture['centers'])} {len(fixture['primitives'])} {fixture['nao']} {len(fixture['points'])} {fixture['spins']} {tile} {pairs}\n"
    result = subprocess.run(
        [str(probe), mode],
        input=header + " ".join(map(repr, values.tolist())),
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
        env=environment,
    )
    return json.loads(result.stdout)


def _independent_potential(
    fixture: dict, density: np.ndarray, *, wide: Path | None = None
) -> np.ndarray:
    """Pinned upstream potentials plus independent real-space VV10 derivatives.

    Exact empty-spin points require the existing independent libquadmath
    evaluator of upstream Maple C: binary64 Libxc suffers the documented Stoll
    cancellation there. No GenerativeQC expression/AD code makes that oracle.
    """
    fields = _fields(fixture["jets"], density)
    if wide is not None:
        assert fixture["spins"] == 2 and wide.is_file()
        with tempfile.TemporaryDirectory(prefix="wb97mv-grid-oracle-") as directory:
            capture = Path(directory) / "features.bin"
            output = Path(directory) / "potential.bin"
            capture.write_bytes(
                np.array(
                    [len(fixture["points"]), fixture["nao"]], dtype=np.uint64
                ).tobytes()
                + fixture["jets"].tobytes()
                + fields.tobytes()
            )
            subprocess.run(
                [str(wide), str(capture), str(output)], check=True, timeout=30
            )
            rows = np.fromfile(output).reshape(-1, 8)
        vrho, vsigma, vtau = rows[:, 1:3], rows[:, 3:6], rows[:, 6:8]
    else:
        libxc = pytest.importorskip("pyscf.dft.libxc")
        assert libxc.__version__ == "7.0.0", (
            "independent oracle requires pinned Libxc 7.0.0"
        )
        rho = fields if fixture["spins"] == 2 else fields[0]
        _, values, _, _ = libxc.eval_xc(
            "wb97m-v", rho, spin=fixture["spins"] - 1, deriv=1
        )
        vrho, vsigma, _, vtau = values
    coefficients = np.zeros_like(fields)
    if fixture["spins"] == 1:
        coefficients[0, 0], coefficients[0, 4] = vrho, vtau
        coefficients[0, 1:4] = 2 * vsigma * fields[0, 1:4]
    else:
        coefficients[:, 0], coefficients[:, 4] = vrho.T, vtau.T
        coefficients[0, 1:4] = (
            2 * vsigma[:, 0] * fields[0, 1:4] + vsigma[:, 1] * fields[1, 1:4]
        )
        coefficients[1, 1:4] = (
            2 * vsigma[:, 2] * fields[1, 1:4] + vsigma[:, 1] * fields[0, 1:4]
        )
    total = fields.sum(axis=0)
    active = total[0] >= 1e-8
    if active.any():
        vrho, vsigma = nonlocal_feature_derivatives_reference(
            fixture["points"][active],
            fixture["weights"][active],
            total[0, active],
            total[1:4, active].T,
            resolve_method("WB97M-V").primitives[-1].spec,
        )
        for spin in range(fixture["spins"]):
            coefficients[spin, 0, active] += vrho
            coefficients[spin, 1:4, active] += (2 * vsigma * total[1:4, active]).T
    coefficients *= fixture["weights"]
    return _assemble(fixture["jets"], coefficients)


@pytest.mark.parametrize("spins,points,tile", ((1, 9, 3), (2, 9, 5), (2, 133, 32)))
def test_native_against_independent_libxc_and_vv10(
    probe: Path, spins: int, points: int, tile: int
) -> None:
    fixture = _fixture(spins, points)
    result = _native(probe, fixture, tile=tile)
    assert result["prepared"] == result["applied"] == 0
    expected = _independent_potential(fixture, fixture["density"])
    np.testing.assert_allclose(
        np.array(result["potential"]).reshape(expected.shape),
        expected,
        atol=3e-10,
        rtol=3e-9,
    )
    actual = np.array(result["response"]).reshape(expected.shape)
    for step in (1e-3, 3e-4):
        plus, minus = (
            _independent_potential(
                fixture, fixture["density"] + sign * step * fixture["direction"]
            )
            for sign in (1, -1)
        )
        np.testing.assert_allclose(
            actual, (plus - minus) / (2 * step), atol=3e-9, rtol=2e-6
        )
    assert result["active"] == points - 1
    assert result["reference_pairs"] == result["response_pairs"] == (points - 1) ** 2
    assert result["actions"] == 1
    if spins == 2:
        assert np.linalg.norm(actual[1]) > 1e-7


@pytest.mark.parametrize("spins", (1, 2))
def test_native_vacuum_has_no_pairs_or_response(probe: Path, spins: int) -> None:
    result = _native(probe, _fixture(spins, vacuum=True))
    assert result["prepared"] == result["applied"] == result["active"] == 0
    assert result["reference_pairs"] == result["response_pairs"] == 0
    assert not np.any(result["potential"]) and not np.any(result["response"])


@pytest.mark.parametrize("swap_spins", (False, True))
def test_native_empty_minority_spin_and_bilateral_spin_coupling(
    probe: Path, swap_spins: bool
) -> None:
    value = os.environ.get("GENERATIVEQC_WB97MV_WIDE_ORACLE")
    if not value:
        pytest.skip("empty-spin acceptance requires pinned upstream libquadmath oracle")
    wide = Path(value)
    fixture = _fixture()
    fixture["density"][1] = 0
    if swap_spins:
        fixture["density"] = fixture["density"][::-1].copy()
        fixture["direction"] = fixture["direction"][::-1].copy()
    result = _native(probe, fixture)
    assert result["prepared"] == result["applied"] == 0
    expected = _independent_potential(fixture, fixture["density"], wide=wide)
    np.testing.assert_allclose(
        np.array(result["potential"]).reshape(expected.shape),
        expected,
        atol=3e-10,
        rtol=3e-9,
    )
    actual = np.array(result["response"]).reshape(expected.shape)
    differences = []
    for step in (1e-3, 3e-4, 1e-4):
        plus, minus = (
            _independent_potential(
                fixture,
                fixture["density"] + sign * step * fixture["direction"],
                wide=wide,
            )
            for sign in (1, -1)
        )
        differences.append((plus - minus) / (2 * step))
    # The huge empty-spin tau potential needs a converged displacement study,
    # not a weakened tolerance: the 1e-3 truncation is still visible in quad.
    errors = [np.max(np.abs(actual - value)) for value in differences]
    assert errors[1] < errors[0] / 5 and errors[2] < errors[1] / 5
    for difference in differences[1:]:
        np.testing.assert_allclose(actual, difference, atol=5e-9, rtol=3e-6)


def test_native_kernel_is_self_adjoint_and_tile_independent(probe: Path) -> None:
    fixture = _fixture()
    right = fixture["direction"].copy()
    first = _native(probe, fixture, tile=2)
    second = _native(probe, fixture, tile=5)
    np.testing.assert_allclose(
        first["response"], second["response"], atol=2e-14, rtol=2e-13
    )
    fixture["direction"] = np.flip(right, axis=(1, 2)).copy()
    fixture["direction"][1] = np.eye(fixture["nao"]) * 0.12
    third = _native(probe, fixture, tile=4)
    assert first["applied"] == third["applied"] == 0
    left_product = np.vdot(
        fixture["direction"], np.array(first["response"]).reshape(right.shape)
    )
    right_product = np.vdot(right, np.array(third["response"]).reshape(right.shape))
    np.testing.assert_allclose(left_product, right_product, atol=3e-13, rtol=3e-11)


@pytest.mark.parametrize(
    "mode",
    (
        "bad-reference",
        "nan-direction",
        "overflow-direction",
        "asymmetric-direction",
        "stale-entry",
        "stale-action",
        "stale-publish",
        "wrong-device",
        "alias-prepare",
        "alias-output",
        "short-arena",
        "wrong-stream",
        "capture-prepare",
        "capture-action",
    ),
)
def test_native_failure_does_not_publish(probe: Path, mode: str) -> None:
    result = _native(probe, _fixture(), mode=mode)
    assert result["prepared"] != 0 or result["applied"] != 0
    assert np.all(np.array(result["response"]) == 91.25)


def test_native_work_budget_not_just_storage_budget(probe: Path) -> None:
    result = _native(probe, _fixture(), pairs=63)
    assert result["prepared"] != 0
    assert result["reference_pairs"] == result["response_pairs"] == 0
    assert np.all(np.array(result["response"]) == 91.25)


def test_native_without_python_compiler_or_cupy(probe: Path) -> None:
    environment = dict(
        os.environ,
        PATH="/nonexistent",
        PYTHONHOME="/nonexistent",
        PYTHONPATH="/nonexistent",
    )
    result = _native(probe, _fixture(), environment=environment)
    assert result["prepared"] == result["applied"] == 0


def test_native_host_only_contract(probe: Path) -> None:
    result = subprocess.run(
        [str(probe), "contract"], text=True, capture_output=True, check=True, timeout=20
    )
    assert json.loads(result.stdout) == {"contract": True}
