"""Native auxiliary-g nuclear contractions against full libcint atom derivatives."""

from __future__ import annotations

import copy
import os
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest
from generativeqc import Calculator, Primitive, Shell
from test_df_derivatives_cuda import fixture_inputs

from tools.generativeqc_validation.df_gradient import (
    execute_df_gradient,
    reference_df_matrices,
)
from tools.validate_df_source import write_input

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DF_DERIVATIVE_CUDA_TEST") != "1",
    reason="requires finite Slurm CUDA allocation",
)


def calculator(inputs: dict) -> Calculator:
    """Construct normalized metadata without invoking a public CUDA method.

    The validation bridge creates its own CUDA context and calls the standalone
    native weighted API. A CPU descriptor admits auxiliary g during metadata
    construction; no CPU value, derivative, or SCF calculation is performed.
    Public CUDA method capability gates deliberately remain unchanged.
    """
    return Calculator(
        device="cpu",
        basis_representation=inputs["basis_representation"],
        basis=[
            Shell(
                s["atom_index"],
                s["angular_momentum"],
                tuple(Primitive(*p) for p in s["primitives"]),
            )
            for s in inputs["shells"]
        ],
    )


def g_fixture(orbital_rep: str, auxiliary_rep: str, shared: bool) -> tuple[dict, dict]:
    """Signed contracted f orbital/g auxiliary functions, in permuted shell order."""
    orbital, auxiliary = fixture_inputs(orbital_rep, shared)
    orbital["shells"][-1]["angular_momentum"] = 3
    auxiliary["basis_representation"] = auxiliary_rep
    auxiliary["shells"][-1]["angular_momentum"] = 4
    auxiliary["shells"][2]["angular_momentum"] = 4
    auxiliary["shells"] = [auxiliary["shells"][i] for i in (3, 0, 2, 1)]
    return orbital, auxiliary


@pytest.mark.parametrize(
    "orbital_rep,auxiliary_rep",
    [
        ("cartesian", "cartesian"),
        ("spherical", "spherical"),
        ("cartesian", "spherical"),
        ("spherical", "cartesian"),
    ],
)
@pytest.mark.parametrize("shared", (False, True))
def test_weighted_all_centers_with_g_expansion_and_tail_tiles(
    orbital_rep: str, auxiliary_rep: str, shared: bool
) -> None:
    assert os.environ.get("SLURM_JOB_ID")
    orbital, auxiliary = g_fixture(orbital_rep, auxiliary_rep, shared)
    a, m, da, dm = reference_df_matrices(orbital, auxiliary)
    rng = np.random.default_rng(1804)
    wa, wm = rng.normal(scale=0.03, size=a.shape), rng.normal(scale=0.03, size=m.shape)
    wa.flat[::7] = 0
    wm.flat[::5] = 0
    expected = np.einsum("axijp,ijp->ax", da, wa) + np.einsum("axpq,pq->ax", dm, wm)
    atoms = [
        (z, tuple(r))
        for z, r in zip(orbital["atomic_numbers"], orbital["coordinates"], strict=True)
    ]
    if not shared:
        assert np.max(np.abs(expected[-1])) > 1e-7
    for schedule, tile, budget in ((0, 257, 65536), (1, 4093, 131072)):
        actual, resources = execute_df_gradient(
            calculator(orbital),
            calculator(auxiliary),
            atoms,
            wa,
            wm,
            schedule=schedule,
            maximum_tile_elements=tile,
            maximum_bytes=budget,
            host_metadata=True,
        )
        np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-11)
        np.testing.assert_allclose(actual.sum(axis=0), 0, atol=2e-10, rtol=0)
        assert resources["host_bytes"] <= budget and resources["device_bytes"] <= budget
        assert resources["weight_tile_elements"] == min(tile, max(wa.size, wm.size))
        assert (
            resources["tiles"]
            == (wa.size + tile - 1) // tile + (wm.size + tile - 1) // tile
        )
        assert resources["device_to_host_bytes"] == expected.nbytes
        assert resources["stream_synchronizations"] == 1


def test_g_nuclear_direction_against_recomputed_independent_values() -> None:
    """One fixed-weight Lagrangian direction moves orbital and auxiliary atoms."""
    assert os.environ.get("SLURM_JOB_ID")
    orbital, auxiliary = g_fixture("spherical", "spherical", False)
    a, m, _, _ = reference_df_matrices(orbital, auxiliary)
    rng = np.random.default_rng(1765)
    wa, wm = rng.normal(scale=0.02, size=a.shape), rng.normal(scale=0.02, size=m.shape)
    direction = rng.normal(scale=0.2, size=(3, 3))
    atoms = [
        (z, tuple(r))
        for z, r in zip(orbital["atomic_numbers"], orbital["coordinates"], strict=True)
    ]
    actual, _ = execute_df_gradient(
        calculator(orbital),
        calculator(auxiliary),
        atoms,
        wa,
        wm,
        maximum_tile_elements=139,
        host_metadata=True,
    )
    analytic = float(np.sum(actual * direction))
    for step in (1e-4, 3e-5):
        energies = []
        for sign in (-1, 1):
            oi, xi = copy.deepcopy(orbital), copy.deepcopy(auxiliary)
            for inputs in (oi, xi):
                inputs["coordinates"] = (
                    np.asarray(inputs["coordinates"]) + sign * step * direction
                ).tolist()
            av, mv, _, _ = reference_df_matrices(oi, xi)
            energies.append(float(np.sum(av * wa) + np.sum(mv * wm)))
        np.testing.assert_allclose(
            (energies[1] - energies[0]) / (2 * step), analytic, atol=3e-8, rtol=3e-7
        )


def test_g_weighted_tight_budget_and_nonfinite_rejection() -> None:
    assert os.environ.get("SLURM_JOB_ID")
    orbital, auxiliary = g_fixture("spherical", "spherical", False)
    a, m, _, _ = reference_df_matrices(orbital, auxiliary)
    atoms = [
        (z, tuple(r))
        for z, r in zip(orbital["atomic_numbers"], orbital["coordinates"], strict=True)
    ]
    o, x = calculator(orbital), calculator(auxiliary)
    with pytest.raises(RuntimeError, match="budget|maximum_bytes"):
        execute_df_gradient(o, x, atoms, a, m, maximum_bytes=32, host_metadata=True)
    a.flat[0] = np.nan
    with pytest.raises(RuntimeError, match="finite"):
        execute_df_gradient(o, x, atoms, a, m, host_metadata=True)


@pytest.mark.parametrize(
    "orbital_rep,auxiliary_rep",
    [
        ("cartesian", "cartesian"),
        ("spherical", "spherical"),
        ("cartesian", "spherical"),
        ("spherical", "cartesian"),
    ],
)
def test_native_g_raw_metric_and_fixed_transform_tiles(
    tmp_path: Path,
    orbital_rep: str,
    auxiliary_rep: str,
) -> None:
    """The second batch item changes shell/atom offsets without changing sizes.

    The external probe must link the same production library as the weighted
    tests. All coordinates are reconstructed only in this independent oracle
    tier; the production weighted traversal visits all centers together.
    """
    assert os.environ.get("SLURM_JOB_ID")
    probe = os.environ.get("GENERATIVEQC_DF_SOURCE_PROBE")
    assert probe, (
        "build tests/native/df_value_probe.cpp and set GENERATIVEQC_DF_SOURCE_PROBE"
    )
    first = g_fixture(orbital_rep, auxiliary_rep, False)
    second = copy.deepcopy(first)
    for basis in second:
        basis["coordinates"][1][0] += 0.13
        basis["shells"] = list(reversed(basis["shells"]))
    systems = [first, second]
    inputs, prefix = tmp_path / "systems.txt", tmp_path / "native"
    write_input(inputs, systems)
    subprocess.run(
        [probe, str(inputs), str(prefix), "137", "11", "--source-derivatives"],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    expected: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "raw",
            "metric",
            "raw_derivative",
            "metric_derivative",
            "transformed_derivative",
        )
    }
    for orbital, auxiliary in systems:
        a, m, da, dm = reference_df_matrices(orbital, auxiliary)
        q = m.shape[0]
        transform = np.fromfunction(
            lambda p, r: np.where(p == r, 0.7, 0.02 / (1 + p + r)), (q, q)
        )
        for key, value in zip(expected, (a, m, da, dm, da @ transform), strict=True):
            expected[key].append(value)
    for key, values in expected.items():
        reference = np.asarray(values)
        actual = np.fromfile(str(prefix) + f"-{key}.bin", dtype=np.float64).reshape(
            reference.shape
        )
        np.testing.assert_allclose(actual, reference, atol=2e-9, rtol=2e-11)
