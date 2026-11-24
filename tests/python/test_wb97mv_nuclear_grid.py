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
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.array_graph import evaluate_array_graph
from generativeqc_compiler.common.nonlocal_correlation import NonlocalCorrelationSpec
from generativeqc_compiler.dft.nonlocal_reference import (
    assemble_nonlocal_potential_reference,
    nonlocal_energy_reference,
    nonlocal_feature_derivatives_reference,
)
from generativeqc_compiler.method.spec import resolve_method
from generativeqc_compiler.method.stationary_grid_second import (
    StationaryGridSecondOrder,
)
from generativeqc_compiler.method.wb97mv_grid_response import point_field_partials
from generativeqc_compiler.method.wb97mv_nuclear_grid import (
    nonlocal_field_program,
    nuclear_grid_ao_adjoints,
    nuclear_grid_maps,
)
from generativeqc_compiler.tensor import execute
from test_wb97mv_native_grid import (
    _assemble,
    _fields,
    _fixture,
    _independent_potential,
    _molecular_vv10_active,
    _native,
    _signed_density_libxc,
)


def _data(spins: int, *, points: int = 8, orbital_replicas: int = 1) -> dict:
    fixture = _fixture(spins, points + 1, orbital_replicas=orbital_replicas)
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


def test_nuclear_ao_adjoints_match_signed_forward_contractions() -> None:
    fixture = _data(1)
    shape = len(fixture["points"]), fixture["nao"]
    maps = nuclear_grid_maps(*shape)
    adjoints = nuclear_grid_ao_adjoints(*shape)
    feeds = _feeds(fixture, 0)
    random = np.random.default_rng(1730)
    for name, legs in (
        ("feature_direction", {"d_ao": fixture["left_ao"]}),
        (
            "feature_mixed",
            {"left_ao": fixture["left_ao"], "d_left_ao": fixture["mixed_ao"]},
        ),
    ):
        cotangents = {key: random.normal(size=shape[0]) for key in maps[name].outputs}
        base = {
            **feeds,
            **(
                {"d_density": np.zeros_like(feeds["density"])}
                if name == "feature_direction"
                else {}
            ),
        }
        motions = {
            f"{leg}{axis}": value
            for leg, panel in legs.items()
            for axis, value in enumerate(panel)
        }
        reference = execute(maps[name], {**base, **motions}).outputs
        coefficients = execute(
            adjoints[name],
            {**base, **{f"bar_{key}": value for key, value in cotangents.items()}},
        ).outputs
        expected = sum(
            np.vdot(cotangents[key], value) for key, value in reference.items()
        )
        actual = sum(
            np.vdot(coefficients[f"bar_{key}"], value) for key, value in motions.items()
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


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
    base, moving = np.zeros_like(fields), np.zeros_like(fields)
    revised = fixture.get("revised_nonlocal")
    energy = 0.0
    if revised is None:
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
        base[:count], moving[:count] = values[1 : 1 + count], values[1 + count :]
    total = (
        fields if spins == 1 else np.vstack((fields[:5] + fields[5:10], fields[10:]))
    )
    delta = (
        direction
        if spins == 1
        else np.vstack((direction[:5] + direction[5:10], direction[10:]))
    )
    active = np.flatnonzero((total[0] >= 1e-10) & (np.abs(total[5]) > 1e-14))
    nonlocal_base, nonlocal_moving = np.zeros_like(total), np.zeros_like(total)
    for kind in ("pair", "local"):
        graph, roots = nonlocal_field_program(kind, runtime_rvv10=revised is not None)
        feeds = {}
        if revised is not None:
            damping, gradient_scale, _ = revised
            feeds = (
                {"damping": damping, "gradient_scale": gradient_scale}
                if kind == "pair"
                else {"beta": (3 / damping**2) ** 0.75 / 32}
            )
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
        factor = (
            revised[2]
            if revised is not None
            else float(resolve_method("WB97M-V").primitives[-1].coefficient)
        )
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
    shifted["nonlocal_active"] = fixture.get(
        "nonlocal_active",
        (_fields(fixture["jets"], fixture["density"]).sum(axis=0)[0] >= 1e-10)
        & (np.abs(fixture["weights"]) > 1e-14),
    )
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
    active = fixture.get(
        "nonlocal_active", (total[0] >= 1e-10) & (np.abs(fixture["weights"]) > 1e-14)
    )
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


def _native_action(
    fixture: dict,
    tile: int = 3,
    *,
    mode: str = "nuclear",
    scales: tuple[float, float] | None = None,
) -> dict:
    path = os.environ.get("GENERATIVEQC_CUDA_WB97MV_GRID_PROBE")
    if not path:
        pytest.skip("requires optional native CUDA WB97M-V grid probe")
    assert os.environ.get("SLURM_JOB_ID"), "real-GPU tests require finite srun"
    if mode in ("fixed-m06-2x", "nuclear-m06-2x"):
        capability = subprocess.check_output([path, "has-m06-2x"], text=True).strip()
        assert capability in ("yes", "no")
        if capability == "no":
            pytest.skip("probe was built without the development M06-2X point")
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
    pairs = 16777216 if scales is None and mode != "nuclear-split-semilocal" else 0
    header = f"{len(fixture['centers'])} {len(fixture['primitives'])} {fixture['nao']} {len(fixture['points'])} {fixture['spins']} {tile} {pairs}\n"
    if scales is not None:
        header += f"{scales[0]} {scales[1]}\n"
    if mode == "nuclear-rvv10":
        header += " ".join(map(str, fixture["revised_nonlocal"])) + "\n"
    process = subprocess.run(
        [path, mode],
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
    assert result["private_mixed_replay"]
    assert result["private_direction_replay"]
    np.testing.assert_array_equal(result["retained_mixed"], result["mixed"])
    assert result["nuclear_contractions"] == 2
    return result


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("screened", (False, True))
def test_wb97mv_nuclear_grid_components(spins: int, screened: bool) -> None:
    """Moving AO, weight and coordinate derivatives compose, even with no pairs."""
    fixture = _data(spins)
    if screened:
        fixture["density"] *= 1e-12
    together = _native_action(fixture)
    semilocal = _native_action(fixture, tile=0, mode="nuclear-split-semilocal")
    vv10 = _native_action(fixture, mode="nuclear-split-vv10")
    assert (
        semilocal["nuclear_reference_pairs"] == semilocal["nuclear_response_pairs"] == 0
    )
    assert vv10["nuclear_response_pairs"] == together["nuclear_response_pairs"]
    for field in ("energy", "potential", "response", "mixed"):
        if screened:
            np.testing.assert_array_equal(vv10[field], 0)
        np.testing.assert_allclose(
            np.asarray(semilocal[field]) + vv10[field], together[field], atol=2e-12
        )


def _scaled_libxc_grid(
    grid: dict,
    density: np.ndarray,
    family: str,
    scales: tuple[float, float] = (1.0, 1.0),
) -> tuple[float, np.ndarray]:
    """Independent semilocal energy and AO potential for a supplied quadrature."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    spins = len(density)
    exchange, correlation = (
        ("LDA_X", "LDA_C_PW") if family == "lda" else ("GGA_X_PBE", "GGA_C_PBE")
    )
    code = f"{scales[0]}*{exchange}+{scales[1]}*{correlation}"
    if family == "b3lyp":
        code = "0.08*LDA_X+0.72*GGA_X_B88,0.19*LDA_C_VWN_RPA+0.81*GGA_C_LYP"
    fields = _fields(grid["jets"], density)
    mgga = family in {"r2scan", "m06-2x"}
    if mgga:
        code = family.upper()
        # This oracle is the common interior of both work policies; the
        # dedicated r2SCAN point tests qualify the different continuation.
        assert np.min(fields[:, 0]) > 4e-11 and np.min(fields[:, 4]) > 3e-20
    inputs = fields if mgga else fields[:, 0] if family == "lda" else fields[:, :4]
    exc, potential, *_ = libxc.eval_xc(
        code, inputs[0] if spins == 1 else inputs, spin=spins - 1, deriv=1
    )
    coefficients = np.zeros_like(fields)
    coefficients[:, 0] = np.atleast_2d(potential[0]) if spins == 1 else potential[0].T
    if mgga:
        coefficients[:, 4] = (
            np.atleast_2d(potential[3]) if spins == 1 else potential[3].T
        )
    if family != "lda":
        if spins == 1:
            coefficients[0, 1:4] = 2 * potential[1] * fields[0, 1:4]
        else:
            same_alpha, cross, same_beta = potential[1].T
            coefficients[0, 1:4] = (
                2 * same_alpha * fields[0, 1:4] + cross * fields[1, 1:4]
            )
            coefficients[1, 1:4] = (
                2 * same_beta * fields[1, 1:4] + cross * fields[0, 1:4]
            )
    return (
        float(np.dot(grid["weights"], exc * fields[:, 0].sum(axis=0))),
        _assemble(grid["jets"], coefficients * grid["weights"]),
    )


def test_bounded_private_ao_adjoint_matches_expanded_contraction() -> None:
    """Fit only one resident AO panel, including a short final tile."""
    points = 4109
    fixture = _data(1, points=points, orbital_replicas=9)
    for name in ("weights", "weight_left", "weight_right", "weight_mixed"):
        fixture[name] /= points
    full = _native_action(fixture, tile=0, mode="nuclear-pbe", scales=(1.0, 1.0))
    tight = _native_action(
        fixture, tile=0, mode="nuclear-pbe-adjoint-tight", scales=(1.0, 1.0)
    )
    assert tight["workspace_bytes"] < full["workspace_bytes"]
    np.testing.assert_array_equal(tight["mixed"], full["mixed"])


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize(
    ("family", "scales"),
    (
        ("lda", (1.0, 1.0)),
        ("pbe", (1.0, 1.0)),
        ("pbe", (0.75, 0.81)),
        ("b3lyp", (1.0, 1.0)),
        ("r2scan", (1.0, 1.0)),
        ("m06-2x", (1.0, 1.0)),
    ),
)
def test_scaled_native_semilocal_grid_against_libxc(
    spins: int, family: str, scales: tuple[float, float]
) -> None:
    """Audit both density and geometry actions without the native point model."""
    fixture = _data(spins)
    actual = _native_action(fixture, tile=0, mode=f"nuclear-{family}", scales=scales)

    def oracle(grid: dict, density: np.ndarray) -> tuple[float, np.ndarray]:
        return _scaled_libxc_grid(grid, density, family, scales)

    for count in (
        "active_nonlocal_points",
        "fixed_response_pairs",
        "nuclear_reference_pairs",
        "nuclear_response_pairs",
    ):
        assert actual[count] == 0
    energy, potential = oracle(fixture, fixture["density"])
    np.testing.assert_allclose(actual["energy"], energy, atol=3e-12, rtol=3e-10)
    np.testing.assert_allclose(
        actual["potential"], potential.ravel(), atol=3e-12, rtol=3e-10
    )
    for step in (0.002, 0.001):
        fixed = (
            oracle(fixture, fixture["density"] + step * fixture["direction"])[1]
            - oracle(fixture, fixture["density"] - step * fixture["direction"])[1]
        ) / (2 * step)
        moving = (
            oracle(*_displaced(fixture, 0, step))[1]
            - oracle(*_displaced(fixture, 0, -step))[1]
        ) / (2 * step)
        mixed = sum(
            left * right * oracle(*_displaced(fixture, left * step, right * step))[0]
            for left in (1, -1)
            for right in (1, -1)
        ) / (4 * step**2)
        np.testing.assert_allclose(
            actual["fixed_response"], fixed.ravel(), atol=2e-8, rtol=2e-6
        )
        np.testing.assert_allclose(
            actual["response"], moving.ravel(), atol=2e-8, rtol=2e-6
        )
    np.testing.assert_allclose(actual["mixed"], mixed, atol=1e-7, rtol=2e-5)


@pytest.mark.parametrize(("spins", "family"), ((1, "pbe"), (2, "r2scan")))
@pytest.mark.parametrize("orbital_replicas", (2, 8, 9, 11, 17, 24))
def test_large_native_grid_potentials_against_libxc(
    spins: int, family: str, orbital_replicas: int
) -> None:
    """Check fixed and moving potentials across the one-strip crossover and a short tile."""
    points = 4109 if orbital_replicas > 8 else 4096
    fixture = _data(spins, points=points, orbital_replicas=orbital_replicas)
    for name in ("weights", "weight_left", "weight_right", "weight_mixed"):
        fixture[name] /= points
    scales = (1.0, 1.0)
    actual = _native_action(fixture, tile=0, mode=f"nuclear-{family}", scales=scales)

    def potential(grid: dict, density: np.ndarray) -> np.ndarray:
        return _scaled_libxc_grid(grid, density, family, scales)[1].ravel()

    step = 0.001
    fixed = (
        potential(fixture, fixture["density"] + step * fixture["direction"])
        - potential(fixture, fixture["density"] - step * fixture["direction"])
    ) / (2 * step)
    moving = (
        potential(*_displaced(fixture, 0, step))
        - potential(*_displaced(fixture, 0, -step))
    ) / (2 * step)
    np.testing.assert_allclose(actual["fixed_response"], fixed, atol=2e-8, rtol=2e-6)
    np.testing.assert_allclose(actual["response"], moving, atol=2e-8, rtol=2e-6)
    if spins == 1 and orbital_replicas > 16:
        factored = _native_action(
            fixture, tile=0, mode="nuclear-pbe-factored", scales=scales
        )
        for field in ("energy", "potential", "fixed_response", "response", "mixed"):
            np.testing.assert_allclose(
                factored[field], actual[field], atol=2e-12, rtol=2e-10
            )


@pytest.mark.parametrize(
    ("family", "scales"),
    (("pbe", (0.75, 1.0)), ("b3lyp", (1.0, 1.0)), ("r2scan", (1.0, 1.0))),
)
def test_optional_point_linearization_matches_checked_actions(
    family: str, scales: tuple[float, float]
) -> None:
    """Check signed fixed and moving responses, with vacuum and measure motion."""
    fixture = _data(1, points=4096)
    fixture["points"][0] += 1e4
    for field in ("jets", "left_ao", "right_ao", "mixed_ao"):
        fixture[field][:, 0, :] = 0
    for field in ("weights", "weight_left", "weight_right", "weight_mixed"):
        fixture[field] /= 4096
    mode = f"nuclear-{family}"
    reference = _native_action(fixture, tile=0, mode=mode, scales=scales)
    saved = _native_action(fixture, tile=0, mode=f"{mode}-points", scales=scales)
    tight = _native_action(fixture, tile=0, mode=f"{mode}-points-tight", scales=scales)
    assert saved["workspace_bytes"] > reference["workspace_bytes"]
    assert tight["workspace_bytes"] == reference["workspace_bytes"]
    for field in ("energy", "potential", "fixed_response", "response", "mixed"):
        np.testing.assert_allclose(
            saved[field], reference[field], atol=2e-12, rtol=2e-10
        )
        np.testing.assert_array_equal(tight[field], reference[field])


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("family", ("b3lyp", "r2scan", "m06-2x"))
def test_native_weight_action_uses_physical_feature_direction(
    spins: int, family: str
) -> None:
    """Isolate d(E_xc density) contracted with the left quadrature-weight motion."""
    fixture = _data(spins)
    for name in ("right_ao", "left_ao", "mixed_ao", "weight_right", "weight_mixed"):
        fixture[name].fill(0)
    scales = (1.0, 1.0)
    actual = _native_action(fixture, tile=0, mode=f"nuclear-{family}", scales=scales)
    _, weighted_fock = _scaled_libxc_grid(
        {**fixture, "weights": fixture["weight_left"]},
        fixture["density"],
        family,
        scales,
    )
    expected = np.sum(weighted_fock * fixture["direction"])
    assert (
        abs(expected) > 1e-7
    )  # A vanishing weight contraction cannot detect an extra half.
    np.testing.assert_allclose(actual["mixed"], expected, atol=3e-12, rtol=3e-10)


@pytest.mark.parametrize("spins", [1, 2])
def test_nuclear_graph_and_capacity_fallback_match(spins: int) -> None:
    """Replay, explicit launches and tight-budget fallback publish the same data."""
    fixture = _data(spins)
    replay = _native_action(fixture)
    assert replay["nuclear_graph_replays"] == 2
    for mode in ("nuclear-serial", "nuclear-tight"):
        serial = _native_action(fixture, mode=mode)
        assert serial["nuclear_graph_replays"] == 0
        for field in ("energy", "potential", "response", "mixed"):
            np.testing.assert_array_equal(replay[field], serial[field])


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


def _revised_reference(fixture: dict, density: np.ndarray) -> tuple[float, np.ndarray]:
    """Values and first potentials from the independent real-space rVV10 oracle."""
    damping, gradient_scale, coefficient = fixture["revised_nonlocal"]
    spec = NonlocalCorrelationSpec(
        "rvv10", Fraction(str(damping)), Fraction(str(gradient_scale))
    )
    total = _fields(fixture["jets"], density).sum(axis=0)
    active = fixture["nonlocal_active"]
    weights = fixture["weights"][active]
    gradients = total[1:4, active].T
    inputs = fixture["points"][active], weights, total[0, active], gradients, spec
    energy = coefficient * nonlocal_energy_reference(*inputs)
    vrho, vsigma = nonlocal_feature_derivatives_reference(*inputs)
    potential = coefficient * assemble_nonlocal_potential_reference(
        fixture["jets"][:, active], weights, gradients, vrho, vsigma
    )
    return energy, np.repeat(potential[None], fixture["spins"], axis=0)


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("implementation", ("compiler", "native"))
def test_rvv10_bilateral_moving_grid(spins: int, implementation: str) -> None:
    fixture = _data(spins)
    # Use noncanonical parameters so a compiled-in b, c, or coefficient fails.
    fixture["revised_nonlocal"] = (
        (11.95, 0.0071, 0.7) if spins == 1 else (6.3, 0.0093, 1.2)
    )
    fixture["nonlocal_active"] = _molecular_vv10_active(
        _fields(fixture["jets"], fixture["density"]).sum(axis=0)[0], fixture["weights"]
    )
    shape = fixture["density"].shape
    if implementation == "native":
        result = _native_action(fixture, mode="nuclear-rvv10")
        energy, potential, fock, contraction = (
            result["energy"][0],
            np.asarray(result["potential"]).reshape(shape),
            np.asarray(result["response"]).reshape(shape),
            result["mixed"][0],
        )
        assert (
            result["nuclear_reference_pairs"]
            == np.count_nonzero(fixture["nonlocal_active"]) ** 2
        )
    else:
        energy, potential, fock, contraction = _compiler_action(fixture)
    expected_energy, expected_potential = _revised_reference(
        fixture, fixture["density"]
    )
    np.testing.assert_allclose(energy, expected_energy, atol=3e-13, rtol=1e-11)
    np.testing.assert_allclose(potential, expected_potential, atol=3e-13, rtol=1e-11)

    fock_estimates, mixed_estimates, fixed_estimates = [], [], []
    for step in (0.002, 0.001):
        fock_estimates.append(
            sum(
                sign * _revised_reference(*_displaced(fixture, 0, sign * step))[1]
                for sign in (1, -1)
            )
            / (2 * step)
        )
        mixed_estimates.append(
            sum(
                first
                * second
                * _revised_reference(*_displaced(fixture, first * step, second * step))[
                    0
                ]
                for first in (1, -1)
                for second in (1, -1)
            )
            / (4 * step**2)
        )
        fixed_estimates.append(
            sum(
                sign
                * _revised_reference(
                    fixture, fixture["density"] + sign * step * fixture["direction"]
                )[1]
                for sign in (1, -1)
            )
            / (2 * step)
        )
    np.testing.assert_allclose(
        fock, (4 * fock_estimates[1] - fock_estimates[0]) / 3, atol=2e-10
    )
    np.testing.assert_allclose(
        contraction, (4 * mixed_estimates[1] - mixed_estimates[0]) / 3, atol=2e-9
    )
    if implementation == "native":
        path = Path(os.environ["GENERATIVEQC_CUDA_WB97MV_GRID_PROBE"])
        fixed = _native(path, fixture, mode="fixed-rvv10")
        assert fixed["prepared"] == fixed["applied"] == 0
        np.testing.assert_allclose(
            fixed["potential"], expected_potential.ravel(), atol=3e-13
        )
        np.testing.assert_allclose(
            fixed["response"],
            ((4 * fixed_estimates[1] - fixed_estimates[0]) / 3).ravel(),
            atol=2e-10,
        )
        refused = _native(
            path, fixture, mode="fixed-rvv10", pairs=fixed["reference_pairs"] - 1
        )
        assert refused["prepared"] != 0 and refused["reference_pairs"] == 0
        np.testing.assert_array_equal(
            refused["response"], np.full(fixture["density"].size, 91.25)
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
        (_fields(fixture["jets"], fixture["density"]).sum(axis=0)[0] >= 1e-10)
        & (np.abs(fixture["weights"]) > 1e-14)
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
        _fields(fixture["jets"], fixture["density"]).sum(axis=0)[0] >= 1e-10
    ) & (np.abs(fixture["weights"]) > 1e-14)
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


@pytest.mark.parametrize("spins", (1, 2))
def test_native_screened_weight_pullbacks_match_fixed_domain(spins: int) -> None:
    """Nonzero moving weights must not revive screened VV10 points."""
    fixture = _data(spins)
    threshold = 1e-14
    fixture["weights"][:7] = (
        0,
        threshold / 2,
        -threshold / 2,
        threshold,
        -threshold,
        np.nextafter(threshold, np.inf),
        -np.nextafter(threshold, np.inf),
    )
    # Keep nonzero weight tangents on the screened rows. Unlike the fixed-grid
    # potential, their erroneous nuclear VV10 pullbacks are not tiny.
    actual = _native_action(fixture)
    energy, potential, response, mixed = _compiler_action(fixture)
    count = np.count_nonzero(np.abs(fixture["weights"]) > threshold)
    assert actual["nuclear_reference_pairs"] == count**2
    assert actual["nuclear_response_pairs"] == 4 * count**2
    for name, expected in (
        ("energy", [energy]),
        ("potential", potential.ravel()),
        ("response", response.ravel()),
        ("mixed", [mixed]),
    ):
        np.testing.assert_allclose(actual[name], expected, atol=2e-11, rtol=2e-9)
