"""Moving WB97M-V grid AD: independent contractions and value differences.

Reference values use pinned Libxc and a separately implemented real-space VV10
oracle. The incumbent stationary partials are a separate policy-reuse check,
never the only scientific oracle for the new nuclear endpoint.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.array_graph import evaluate_array_graph
from generativeqc_compiler.dft.nonlocal_reference import nonlocal_energy_reference
from generativeqc_compiler.method.spec import resolve_method
from generativeqc_compiler.method.stationary_grid_second import (
    StationaryGridSecondOrder,
)
from generativeqc_compiler.method.wb97mv_grid_response import point_field_partials
from generativeqc_compiler.method.wb97mv_nuclear_grid import (
    nonlocal_field_program,
    nuclear_grid_maps,
)
from generativeqc_compiler.tensor import execute
from test_wb97mv_native_grid import (
    _assemble,
    _fields,
    _fixture,
    _independent_potential,
    _signed_density_libxc,
)


def _data(spins: int) -> dict:
    fixture = _fixture(spins)
    fixture["points"] = fixture["points"][:-1].copy()
    fixture["weights"] = fixture["weights"][:-1].copy()
    fixture["jets"] = fixture["jets"][:, :-1].copy()
    fixture["weights"][0] = 0
    random = np.random.default_rng(2853 + spins)
    fixture["left_ao"], fixture["right_ao"], fixture["mixed_ao"] = random.normal(
        scale=0.02, size=(3, *fixture["jets"].shape)
    )
    fixture["weight_left"], fixture["weight_right"], fixture["weight_mixed"] = (
        random.normal(scale=0.03, size=(3, len(fixture["points"])))
    )
    fixture["point_left"], fixture["point_right"] = random.normal(
        scale=0.04, size=(2, *fixture["points"].shape)
    )
    return fixture


def _feeds(fixture: dict, spin: int) -> dict:
    return {
        "density": fixture["density"][spin],
        "d_density": fixture["direction"][spin],
        **{f"ao{axis}": value for axis, value in enumerate(fixture["jets"])},
        **{f"d_ao{axis}": value for axis, value in enumerate(fixture["right_ao"])},
    }


def _ordered(result: dict) -> np.ndarray:
    return np.array([result[name] for name in ("rho", "g0", "g1", "g2", "tau")])


@pytest.mark.parametrize("spins", [1, 2])
def test_feature_first_and_mixed_maps_have_independent_bilinear_oracle(
    spins: int,
) -> None:
    fixture = _data(spins)
    maps = nuclear_grid_maps(len(fixture["points"]), fixture["nao"])
    for spin in range(spins):
        feeds = _feeds(fixture, spin)
        first = _ordered(execute(maps["feature_direction"], feeds).outputs)
        mixed_feeds = {
            **feeds,
            "left_density": np.zeros_like(feeds["density"]),
            "d_left_density": np.zeros_like(feeds["density"]),
            **{
                f"left_ao{axis}": value for axis, value in enumerate(fixture["left_ao"])
            },
            **{
                f"d_left_ao{axis}": value
                for axis, value in enumerate(fixture["mixed_ao"])
            },
        }
        mixed = _ordered(execute(maps["feature_mixed"], mixed_feeds).outputs)

        def fields(left: float, right: float, channel: int = spin) -> np.ndarray:
            jets = (
                fixture["jets"]
                + left * fixture["left_ao"]
                + right * fixture["right_ao"]
                + left * right * fixture["mixed_ao"]
            )
            density = fixture["density"] + right * fixture["direction"]
            return _fields(jets, density)[channel]

        errors = []
        for step in (0.01, 0.005):
            finite_first = (fields(0, step) - fields(0, -step)) / (2 * step)
            finite_mixed = (
                fields(step, step)
                - fields(step, -step)
                - fields(-step, step)
                + fields(-step, -step)
            ) / (4 * step**2)
            errors.append(
                max(
                    np.max(abs(first - finite_first)), np.max(abs(mixed - finite_mixed))
                )
            )
        # A zero spin-density direction makes these polynomial differences
        # exact; below the cancellation floor a convergence ratio is undefined.
        roundoff = (
            32 * np.finfo(float).eps * max(1.0, np.max(abs(fields(0, 0)))) / 0.005**2
        )
        assert errors[1] < max(errors[0] * 0.28, roundoff)
        np.testing.assert_allclose(first, finite_first, atol=2e-8, rtol=3e-6)
        np.testing.assert_allclose(mixed, finite_mixed, atol=2e-8, rtol=3e-6)


def test_potential_jvp_differentiates_ao_and_full_physical_cotangents() -> None:
    fixture = _data(2)
    random = np.random.default_rng(3185)
    coefficients, moving = random.normal(
        scale=0.2, size=(2, 2, 5, len(fixture["points"]))
    )
    program = nuclear_grid_maps(len(fixture["points"]), fixture["nao"])[
        "potential_direction"
    ]
    for spin in range(2):
        feeds = _feeds(fixture, spin)
        feeds.pop("density")
        feeds.pop("d_density")
        for prefix, values in (("bar_", coefficients[spin]), ("d_bar_", moving[spin])):
            feeds.update(
                dict(
                    zip(
                        (prefix + name for name in ("rho", "g0", "g1", "g2", "tau")),
                        values,
                        strict=True,
                    )
                )
            )
        actual = execute(program, feeds).outputs["potential"]
        steps = []
        for step in (0.002, 0.001):
            plus = _assemble(
                fixture["jets"] + step * fixture["right_ao"],
                coefficients + step * moving,
            )[spin]
            minus = _assemble(
                fixture["jets"] - step * fixture["right_ao"],
                coefficients - step * moving,
            )[spin]
            steps.append((plus - minus) / (2 * step))
        np.testing.assert_allclose(
            actual, (4 * steps[1] - steps[0]) / 3, atol=2e-13, rtol=3e-12
        )


@pytest.mark.parametrize("spins", [1, 2])
def test_full_semilocal_partials_reuse_the_fixed_branch_work_policy(spins: int) -> None:
    fixture = _data(spins)
    fields = _fields(fixture["jets"], fixture["density"]).reshape(5 * spins, -1)
    maps = nuclear_grid_maps(len(fixture["points"]), fixture["nao"])
    direction = np.concatenate(
        [
            _ordered(execute(maps["feature_direction"], _feeds(fixture, spin)).outputs)
            for spin in range(spins)
        ]
    )
    full_fields = np.vstack((fields, fixture["weights"], fixture["points"].T))
    full_direction = np.vstack(
        (direction, fixture["weight_right"], fixture["point_right"].T)
    )
    method = resolve_method(
        "WB97M-V", spin="unpolarized" if spins == 1 else "polarized"
    )
    model = StationaryGridSecondOrder(
        method.primitives[0].functional, method.primitives[-1]
    )
    reference = model.evaluate(full_fields, full_direction)
    graph, roots = point_field_partials(spins)
    actual = np.array(
        evaluate_array_graph(
            graph,
            roots,
            {
                **{f"f{index}": value for index, value in enumerate(fields)},
                **{f"d{index}": value for index, value in enumerate(direction)},
                "weight": fixture["weights"],
                "dweight": fixture["weight_right"],
            },
        )
    )
    count = 5 * spins + 1
    np.testing.assert_allclose(
        actual[1 : 1 + count], reference.semilocal[0][:count], atol=3e-12, rtol=3e-10
    )
    np.testing.assert_allclose(
        actual[1 + count :], reference.semilocal[1][:count], atol=3e-12, rtol=3e-10
    )
    total = fields.reshape(spins, 5, -1).sum(axis=0)
    nonlocal_energy = nonlocal_energy_reference(
        fixture["points"],
        fixture["weights"],
        total[0],
        total[1:4].T,
        method.primitives[-1].spec,
    ) * float(method.primitives[-1].coefficient)
    np.testing.assert_allclose(
        actual[0].sum(), reference.energy - nonlocal_energy, atol=3e-12, rtol=3e-10
    )


def _partials(fixture: dict, fields: np.ndarray, direction: np.ndarray) -> tuple:
    """Execute generated scalar mathematics; independent values gate it below."""
    spins = fixture["spins"]
    count = 5 * spins + 1
    graph, roots = point_field_partials(spins)
    values = np.array(
        evaluate_array_graph(
            graph,
            roots,
            {
                **{
                    f"f{index}": value
                    for index, value in enumerate(fields[: 5 * spins])
                },
                **{
                    f"d{index}": value
                    for index, value in enumerate(direction[: 5 * spins])
                },
                "weight": fields[5 * spins],
                "dweight": direction[5 * spins],
            },
        )
    )
    energy = values[0].sum()
    base, moving = np.zeros_like(fields), np.zeros_like(fields)
    base[:count], moving[:count] = values[1 : 1 + count], values[1 + count :]
    total = (
        fields if spins == 1 else np.vstack((fields[:5] + fields[5:10], fields[10:]))
    )
    delta = (
        direction
        if spins == 1
        else np.vstack((direction[:5] + direction[5:10], direction[10:]))
    )
    active = np.flatnonzero(total[0] >= 1e-8)
    nonlocal_base, nonlocal_moving = np.zeros_like(total), np.zeros_like(total)
    for kind in ("pair", "local"):
        graph, roots = nonlocal_field_program(kind)
        feeds = {}
        for prefix, matrix in (("", total), ("d", delta)):
            for index in range(9):
                values = matrix[index, active]
                feeds[f"{prefix}a{index}"] = (
                    values[:, None] if kind == "pair" else values
                )
                if kind == "pair":
                    feeds[f"{prefix}b{index}"] = values[None, :]
        shape = (len(active),) * (2 if kind == "pair" else 1)
        evaluated = np.array(
            [
                np.broadcast_to(value, shape)
                for value in evaluate_array_graph(graph, roots, feeds)
            ]
        )
        factor = float(resolve_method("WB97M-V").primitives[-1].coefficient)
        energy += factor * (0.5 if kind == "pair" else 1) * evaluated[0].sum()
        if kind == "pair":
            evaluated = evaluated.sum(axis=2)
        nonlocal_base[:, active] += factor * evaluated[1:10]
        nonlocal_moving[:, active] += factor * evaluated[10:]
    for target, source in ((base, nonlocal_base), (moving, nonlocal_moving)):
        for spin in range(spins):
            target[5 * spin : 5 * spin + 5] += source[:5]
        target[5 * spins :] += source[5:]
    return energy, base, moving


def _displaced(fixture: dict, left: float, right: float) -> tuple:
    shifted = dict(fixture)
    shifted["jets"] = (
        fixture["jets"]
        + left * fixture["left_ao"]
        + right * fixture["right_ao"]
        + left * right * fixture["mixed_ao"]
    )
    shifted["weights"] = (
        fixture["weights"]
        + left * fixture["weight_left"]
        + right * fixture["weight_right"]
        + left * right * fixture["weight_mixed"]
    )
    shifted["points"] = (
        fixture["points"]
        + left * fixture["point_left"]
        + right * fixture["point_right"]
    )
    density = fixture["density"] + right * fixture["direction"]
    return shifted, density


def _energy(fixture: dict, density: np.ndarray) -> float:
    """Independent pinned upstream semilocal plus direct-space VV10 values."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    assert libxc.__version__ == "7.0.0"
    fields = _fields(fixture["jets"], density)
    exc = libxc.eval_xc(
        fixture.get("libxc_code", "wb97m-v"),
        fields if fixture["spins"] == 2 else fields[0],
        spin=fixture["spins"] - 1,
        deriv=0,
    )[0]
    total = fields.sum(axis=0)
    active = fixture.get("nonlocal_active", total[0] >= 1e-8)
    nonlocal_energy = (
        nonlocal_energy_reference(
            fixture["points"][active],
            fixture["weights"][active],
            total[0, active],
            total[1:4, active].T,
            resolve_method("WB97M-V").primitives[-1].spec,
        )
        if active.any()
        else 0.0
    )
    return float(np.dot(fixture["weights"] * total[0], exc)) + nonlocal_energy


def _compiler_action(fixture: dict) -> tuple:
    spins = fixture["spins"]
    maps = nuclear_grid_maps(len(fixture["points"]), fixture["nao"])
    fields = np.vstack(
        (
            _fields(fixture["jets"], fixture["density"]).reshape(5 * spins, -1),
            fixture["weights"],
            fixture["points"].T,
        )
    )
    right = np.vstack(
        (
            np.concatenate(
                [
                    _ordered(
                        execute(
                            maps["feature_direction"], _feeds(fixture, spin)
                        ).outputs
                    )
                    for spin in range(spins)
                ]
            ),
            fixture["weight_right"],
            fixture["point_right"].T,
        )
    )
    energy, base, moving = _partials(fixture, fields, right)
    first_fields, mixed_fields, fock = [], [], []
    for spin in range(spins):
        feeds = _feeds(fixture, spin)
        left_feeds = {
            **feeds,
            "d_density": np.zeros_like(feeds["density"]),
            **{f"d_ao{axis}": value for axis, value in enumerate(fixture["left_ao"])},
        }
        first_fields.append(
            _ordered(execute(maps["feature_direction"], left_feeds).outputs)
        )
        mixed_fields.append(
            _ordered(
                execute(
                    maps["feature_mixed"],
                    {
                        **feeds,
                        "left_density": np.zeros_like(feeds["density"]),
                        "d_left_density": np.zeros_like(feeds["density"]),
                        **{
                            f"left_ao{axis}": value
                            for axis, value in enumerate(fixture["left_ao"])
                        },
                        **{
                            f"d_left_ao{axis}": value
                            for axis, value in enumerate(fixture["mixed_ao"])
                        },
                    },
                ).outputs
            )
        )
        potential_feeds = {
            name: value
            for name, value in feeds.items()
            if name not in ("density", "d_density")
        }
        for prefix, rows in (
            ("bar_", base[5 * spin : 5 * spin + 5]),
            ("d_bar_", moving[5 * spin : 5 * spin + 5]),
        ):
            potential_feeds.update(
                dict(
                    zip(
                        (prefix + name for name in ("rho", "g0", "g1", "g2", "tau")),
                        rows,
                        strict=True,
                    )
                )
            )
        fock.append(
            execute(maps["potential_direction"], potential_feeds).outputs["potential"]
        )
    first_fields = np.vstack(
        (*first_fields, fixture["weight_left"], fixture["point_left"].T)
    )
    mixed_fields = np.vstack(
        (*mixed_fields, fixture["weight_mixed"], np.zeros_like(fixture["point_left"].T))
    )
    contraction = np.sum(first_fields * moving + mixed_fields * base)
    potential = _assemble(fixture["jets"], base[: 5 * spins].reshape(spins, 5, -1))
    return energy, potential, np.array(fock), contraction


def _native_action(fixture: dict, tile: int = 3) -> dict:
    path = os.environ.get("GENERATIVEQC_CUDA_WB97MV_GRID_PROBE")
    if not path:
        pytest.skip("requires optional native CUDA WB97M-V grid probe")
    assert os.environ.get("SLURM_JOB_ID"), "real-GPU tests require finite srun"
    values = np.concatenate(
        [
            fixture[name].ravel()
            for name in (
                "basis",
                "points",
                "weights",
                "density",
                "direction",
                "right_ao",
                "left_ao",
                "mixed_ao",
                "weight_right",
                "weight_left",
                "weight_mixed",
                "point_right",
                "point_left",
            )
        ]
    )
    header = f"{len(fixture['centers'])} {len(fixture['primitives'])} {fixture['nao']} {len(fixture['points'])} {fixture['spins']} {tile} 16777216\n"
    process = subprocess.run(
        [path, "nuclear"],
        input=header + " ".join(map(repr, values.tolist())),
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
        env=dict(
            os.environ,
            PATH="/nonexistent",
            PYTHONHOME="/nonexistent",
            PYTHONPATH="/nonexistent",
        ),
    )
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["nuclear_lifecycle"]
    np.testing.assert_array_equal(result["retained_mixed"], result["mixed"])
    assert result["nuclear_contractions"] == 2
    return result


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("implementation", ["compiler", "native"])
def test_moving_fock_and_stationary_mixed_contraction_against_independent_values(
    spins: int,
    implementation: str,
) -> None:
    fixture = _data(spins)
    if implementation == "native":
        actual = _native_action(fixture)
        energy = actual["energy"][0]
        fock = np.array(actual["response"]).reshape(
            spins, fixture["nao"], fixture["nao"]
        )
        contraction = actual["mixed"][0]
    else:
        energy, _, fock, contraction = _compiler_action(fixture)
    np.testing.assert_allclose(
        energy, _energy(fixture, fixture["density"]), atol=3e-12, rtol=3e-10
    )
    fock_differences, scalar_differences = [], []
    for step in (0.002, 0.001):
        positive, positive_density = _displaced(fixture, 0, step)
        negative, negative_density = _displaced(fixture, 0, -step)
        fock_differences.append(
            (
                _independent_potential(positive, positive_density)
                - _independent_potential(negative, negative_density)
            )
            / (2 * step)
        )
        scalar_differences.append(
            sum(
                sign_left
                * sign_right
                * _energy(*_displaced(fixture, sign_left * step, sign_right * step))
                for sign_left in (1, -1)
                for sign_right in (1, -1)
            )
            / (4 * step**2)
        )
    np.testing.assert_allclose(
        fock, (4 * fock_differences[1] - fock_differences[0]) / 3, atol=3e-9, rtol=2e-6
    )
    np.testing.assert_allclose(
        contraction,
        (4 * scalar_differences[1] - scalar_differences[0]) / 3,
        atol=1e-7,
        rtol=2e-5,
    )


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("case", ["full", "frozen", "zero", "empty-screen"])
@pytest.mark.parametrize("tile", [1, 3, 64, 256])
def test_native_moving_grid_matches_ad_and_retains_bounded_right_cache(
    spins: int,
    case: str,
    tile: int,
) -> None:
    fixture = _data(spins)
    if case == "frozen":
        fixture["direction"][:] = 0
    elif case == "zero":
        for name in (
            "direction",
            "right_ao",
            "weight_right",
            "point_right",
            "mixed_ao",
            "weight_mixed",
        ):
            fixture[name][:] = 0
    elif case == "empty-screen":
        fixture["density"] *= 1e-12
    actual = _native_action(fixture, tile)
    energy, potential, fock, mixed = _compiler_action(fixture)
    np.testing.assert_allclose(actual["energy"], [energy], atol=2e-12, rtol=2e-10)
    np.testing.assert_allclose(
        np.array(actual["potential"]).reshape(potential.shape),
        potential,
        atol=2e-12,
        rtol=2e-10,
    )
    np.testing.assert_allclose(
        np.array(actual["response"]).reshape(fock.shape), fock, atol=2e-11, rtol=2e-9
    )
    np.testing.assert_allclose(actual["mixed"], [mixed], atol=2e-11, rtol=2e-9)
    active = np.count_nonzero(
        _fields(fixture["jets"], fixture["density"]).sum(axis=0)[0] >= 1e-8
    )
    assert actual["nuclear_reference_pairs"] == active**2
    # Three valid and one final revoked direction traverse pairs, independent of
    # the number of successful or rejected left-coordinate contractions.
    assert actual["nuclear_response_pairs"] == 4 * active**2
    assert 0 < actual["nuclear_affine_terms"] < 1 << 34


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("fitted", [False, True])
def test_actual_molecular_moving_grid_against_independent_values(
    spins: int,
    fitted: bool,
) -> None:
    path = os.environ.get("GENERATIVEQC_KS_WB97MV_GRID_PROBE")
    if not path:
        pytest.skip("requires optional native molecular WB97M-V grid probe")
    assert os.environ.get("SLURM_JOB_ID"), "real-GPU tests require finite srun"
    process = subprocess.run(
        [path, "rks" if spins == 1 else "uks", "df-nuclear" if fitted else "nuclear"],
        text=True,
        capture_output=True,
        check=False,
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
    points, nao = result["point_count"], result["nao"]
    fixture = {
        "spins": spins,
        "nao": nao,
        "jets": np.array(result["spatial"]).reshape(20, points, nao)[:4],
        "points": np.array(result["points"]).reshape(points, 3),
        "weights": np.array(result["weights"]),
        "density": np.array(result["density"]).reshape(spins, nao, nao),
        "direction": np.array(result["direction"]).reshape(spins, nao, nao),
    }
    for name, values in zip(
        ("left_ao", "right_ao", "mixed_ao"),
        np.array(result["geometry_ao"]).reshape(3, 4, points, nao),
        strict=True,
    ):
        fixture[name] = values
    for name, values in zip(
        ("weight_left", "weight_right", "weight_mixed"),
        np.array(result["geometry_weights"]).reshape(3, points),
        strict=True,
    ):
        fixture[name] = values
    for name, values in zip(
        ("point_left", "point_right"),
        np.array(result["point_directions"]).reshape(2, points, 3),
        strict=True,
    ):
        fixture[name] = values
    fixture["nonlocal_active"] = (
        _fields(fixture["jets"], fixture["density"]).sum(axis=0)[0] >= 1e-8
    )
    # The molecular probe uses arbitrary signed D motion, not a PSD path.
    # As in the fixed-grid qualification, disable only Libxc's optional FHC
    # projection on separate handles; default-oracle interior tests stay intact.
    with _signed_density_libxc() as code:
        fixture["libxc_code"] = code
        _qualify_molecular_values(result, fixture)


def _qualify_molecular_values(result: dict, fixture: dict) -> None:
    spins, nao = fixture["spins"], fixture["nao"]
    np.testing.assert_allclose(
        result["nuclear_energy"],
        [_energy(fixture, fixture["density"])],
        atol=2e-10,
        rtol=2e-9,
    )
    fock_differences, scalar_differences = [], []
    for step in (0.004, 0.002):
        positive, positive_density = _displaced(fixture, 0, step)
        negative, negative_density = _displaced(fixture, 0, -step)
        fock_differences.append(
            (
                _independent_potential(
                    positive,
                    positive_density,
                    nonlocal_active=fixture["nonlocal_active"],
                    libxc_code=fixture["libxc_code"],
                )
                - _independent_potential(
                    negative,
                    negative_density,
                    nonlocal_active=fixture["nonlocal_active"],
                    libxc_code=fixture["libxc_code"],
                )
            )
            / (2 * step)
        )
        scalar_differences.append(
            sum(
                sign_left
                * sign_right
                * _energy(*_displaced(fixture, sign_left * step, sign_right * step))
                for sign_left in (1, -1)
                for sign_right in (1, -1)
            )
            / (4 * step**2)
        )
    np.testing.assert_allclose(
        np.array(result["nuclear_fock"]).reshape(spins, nao, nao),
        (4 * fock_differences[1] - fock_differences[0]) / 3,
        atol=1e-7,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        result["nuclear_mixed"],
        [(4 * scalar_differences[1] - scalar_differences[0]) / 3],
        atol=2e-7,
        rtol=3e-5,
    )
    assert result["nuclear_reference_pairs"] == result["active"] ** 2
    assert result["nuclear_response_pairs"] == result["active"] ** 2
    assert result["revoked_without_publication"]


def test_scalar_mixed_contraction_and_energy_reduction() -> None:
    random = np.random.default_rng(518)
    vectors = random.normal(size=(4, 27))
    maps = nuclear_grid_maps()
    actual = execute(
        maps["mixed_contraction"],
        dict(
            zip(
                ("left", "mixed", "base_partial", "right_partial"), vectors, strict=True
            )
        ),
    ).outputs["value"]
    np.testing.assert_allclose(
        actual, vectors[0] @ vectors[3] + vectors[1] @ vectors[2], atol=2e-15
    )
    energy = random.normal(size=3)
    np.testing.assert_allclose(
        execute(maps["energy_sum"], {"energy": energy}).outputs["energy"],
        energy.sum(),
        atol=1e-15,
    )


def test_nuclear_grid_generation_is_build_only_and_deterministic(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    script = """
import runpy, sys
class Deny:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'generativeqc', 'cupy', 'torch', 'pyscf'}:
            raise RuntimeError('forbidden build-time runtime import: ' + fullname)
sys.meta_path.insert(0, Deny())
sys.argv = ['generate_wb97mv_nuclear_grid.py', '--output', sys.argv[1]]
runpy.run_path('tools/generate_wb97mv_nuclear_grid.py', run_name='__main__')
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
