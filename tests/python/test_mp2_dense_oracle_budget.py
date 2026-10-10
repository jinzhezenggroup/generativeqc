"""Real native dense-oracle admission, live storage and independent NumPy parity."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from tools.generativeqc_mp2.gradient import (
    canonical_energy_adjoint,
    canonical_orbital_rhs,
)
from tools.generativeqc_posthf.fixtures import load_fixture

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUDGET = 1 << 20


@pytest.fixture(scope="module")
def dense_probe(
    required_native_cxx: NativeCxx, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Compile the entire real translation units; discard only unused link sections."""
    directory = tmp_path_factory.mktemp("mp2-dense-oracle")
    arguments = (
        "-std=c++20",
        "-O2",
        "-ffunction-sections",
        "-fdata-sections",
        "-DGENERATIVEQC_HAS_OPENBLAS=0",
        "-I" + str(ROOT / "src"),
        "-I" + str(ROOT / "include"),
    )
    objects = []
    for relative in (
        "src/posthf/mp2_gradient.cpp",
        "src/response/native_gmres.cpp",
        "tests/native/mp2_dense_oracle_probe.cpp",
    ):
        source = ROOT / relative
        obj = directory / (source.stem + ".o")
        required_native_cxx.compile_object(source, obj, args=arguments)
        objects.append(obj)
    executable = directory / "probe"
    required_native_cxx.link(objects, executable, args=("-Wl,--gc-sections",))
    return executable


def planned(orbitals: int, occupied: int, relaxed: bool) -> dict[str, int]:
    """Independent simultaneous-live-vector census, not the native planner."""
    pairs = occupied * (orbitals - occupied)
    return {
        "dense_weight_bytes": 8 * orbitals**4,
        "retained_bytes": 8
        * (orbitals**4 + (2 * orbitals**2 if relaxed else orbitals**2 + 2 * pairs)),
        "peak_bytes": 8
        * (orbitals**4 + 4 * orbitals**2 + (2 * pairs if relaxed else pairs)),
    }


def invoke(
    executable: Path,
    mode: str,
    orbitals: int,
    occupied: int,
    budget: int,
    arrays: tuple[np.ndarray, ...] = (),
) -> dict[str, Any]:
    result = subprocess.run(
        [str(executable), mode, str(orbitals), str(occupied), str(budget)],
        input=" ".join(
            format(float(value), ".17g") for array in arrays for value in array.ravel()
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    observed = json.loads(result.stdout)
    assert observed["dropped_requests"] == 0
    assert observed["allocation_calls"] == len(observed["allocation_requests"])
    return observed


def synthetic(orbitals: int, occupied: int) -> tuple[np.ndarray, ...]:
    """A small canonical Hamiltonian with independently symmetrized chemists' ERIs."""
    generator = np.random.default_rng(orbitals * 100 + occupied)
    raw = generator.normal(size=(orbitals,) * 4) * 0.002
    eri = (
        sum(
            raw.transpose(permutation)
            for permutation in (
                (0, 1, 2, 3),
                (1, 0, 2, 3),
                (0, 1, 3, 2),
                (1, 0, 3, 2),
                (2, 3, 0, 1),
                (3, 2, 0, 1),
                (2, 3, 1, 0),
                (3, 2, 1, 0),
            )
        )
        / 8
    )
    energies = np.linspace(-2.0, 1.0, orbitals)
    hcore = np.diag(energies)
    for index in range(occupied):
        hcore -= 2 * eri[:, :, index, index] - eri[:, index, index, :]
    response = generator.normal(size=occupied * (orbitals - occupied)) * 0.01
    return hcore, eri, energies, response


def reference(
    arrays: tuple[np.ndarray, ...], occupied: int, relaxed: bool
) -> dict[str, np.ndarray | float]:
    hcore, eri, energies, response = arrays
    adjoint = canonical_energy_adjoint(
        eri[:occupied, occupied:, :occupied, occupied:].transpose(0, 2, 1, 3),
        energies,
        occupied,
        reference_identity="independent-fixture",
        hamiltonian_id="same-Hamiltonian",
    )
    orbital = canonical_orbital_rhs(hcore, eri, adjoint, occupied)
    if not relaxed:
        return {
            "one": orbital.one_electron.ravel(),
            "two": orbital.two_electron.ravel(),
            "energy_gradient": orbital.energy_gradient.ravel(),
            "response_rhs": orbital.response_rhs.ravel(),
        }
    one = np.array(orbital.one_electron, copy=True)
    two = np.array(orbital.two_electron, copy=True)
    for first in range(occupied):
        one[first, first] += 2
        for second in range(occupied):
            two[first, first, second, second] += 2
            two[first, second, second, first] -= 1
        for virtual in range(energies.size - occupied):
            row = occupied + virtual
            value = response[first * (energies.size - occupied) + virtual]
            one[row, first] -= value
            for second in range(occupied):
                two[row, first, second, second] -= 2 * value
                two[row, second, second, first] += value
    gradient = np.einsum("pq,tq->tp", one, hcore) + np.einsum("pq,pt->tq", one, hcore)
    gradient += np.einsum("pqrs,tqrs->tp", two, eri) + np.einsum(
        "pqrs,ptrs->tq", two, eri
    )
    gradient += np.einsum("pqrs,pqts->tr", two, eri) + np.einsum(
        "pqrs,pqrt->ts", two, eri
    )
    return {
        "one": one.ravel(),
        "two": two.ravel(),
        "overlap": (-0.25 * (gradient + gradient.T)).ravel(),
        "stationarity_residual": float(np.linalg.norm(gradient - gradient.T)),
    }


@pytest.mark.parametrize("orbitals,occupied", [(2, 1), (5, 2), (9, 1), (12, 6)])
@pytest.mark.parametrize("relaxed", [False, True])
def test_exact_budget_matches_actual_live_storage_and_independent_values(
    dense_probe: Path, orbitals: int, occupied: int, relaxed: bool
) -> None:
    arrays = synthetic(orbitals, occupied)
    plan = planned(orbitals, occupied, relaxed)
    mode = "weights" if relaxed else "rhs"
    observed_plan = invoke(
        dense_probe, "plan-" + mode, orbitals, occupied, plan["peak_bytes"]
    )
    assert observed_plan["error"] is None
    assert observed_plan["allocation_calls"] == 0
    assert observed_plan["plan"] == {**plan, "budget_bytes": plan["peak_bytes"]}
    result = invoke(dense_probe, mode, orbitals, occupied, plan["peak_bytes"], arrays)
    assert result["error"] is None
    assert result["measured_peak_bytes"] == plan["peak_bytes"]
    assert result["measured_retained_bytes"] == plan["retained_bytes"]
    assert result["final_live_bytes"] == 0
    legacy = invoke(
        dense_probe, "legacy-" + mode, orbitals, occupied, DEFAULT_BUDGET, arrays
    )
    for key, expected in reference(arrays, occupied, relaxed).items():
        np.testing.assert_allclose(result[key], expected, atol=3e-11, rtol=3e-11)
        np.testing.assert_array_equal(result[key], legacy[key])


@pytest.mark.parametrize("name", ["h2", "water", "lih", "f_heh"])
@pytest.mark.parametrize("relaxed", [False, True])
def test_committed_pyscf_hamiltonians_preserve_dense_oracle_values(
    dense_probe: Path, name: str, relaxed: bool
) -> None:
    meta, arrays = load_fixture(name)
    occupied = meta["records"]["conventional"]["electron_count"] // 2
    coefficients = arrays["conventional_C"]
    hcore = coefficients.T @ arrays["conventional_h"] @ coefficients
    energies = arrays["conventional_eps"]
    response = np.linspace(0.001, 0.01, occupied * (energies.size - occupied))
    inputs = hcore, arrays["conventional_mo"], energies, response
    mode = "weights" if relaxed else "rhs"
    result = invoke(dense_probe, mode, energies.size, occupied, DEFAULT_BUDGET, inputs)
    assert result["error"] is None
    for key, expected in reference(inputs, occupied, relaxed).items():
        np.testing.assert_allclose(result[key], expected, atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize("relaxed", [False, True])
@pytest.mark.parametrize("budget_kind", ["zero", "one-byte-short"])
def test_rejected_budget_precedes_numeric_allocation(
    dense_probe: Path, relaxed: bool, budget_kind: str
) -> None:
    orbitals, occupied = 12, 6
    plan = planned(orbitals, occupied, relaxed)
    budget = 0 if budget_kind == "zero" else plan["peak_bytes"] - 1
    result = invoke(
        dense_probe,
        "weights" if relaxed else "rhs",
        orbitals,
        occupied,
        budget,
        synthetic(orbitals, occupied),
    )
    assert result["error"] == "length_error"
    assert result["one"] == result["two"] == result["overlap"] == []
    assert result["largest_request"] < 8 * orbitals**2
    assert result["final_live_bytes"] == 0


@pytest.mark.parametrize(
    "mode",
    [
        "plan-rhs",
        "plan-weights",
        "oversized-rhs",
        "oversized-weights",
        "oversized-budget-rhs",
        "oversized-budget-weights",
    ],
)
@pytest.mark.parametrize("orbitals", [13, 1 << 32, (1 << 64) - 1])
def test_legacy_and_explicit_size_gates_precede_input_walk_or_extent_overflow(
    dense_probe: Path, mode: str, orbitals: int
) -> None:
    result = invoke(dense_probe, mode, orbitals, 1, DEFAULT_BUDGET)
    assert result["error"] == "length_error"
    assert result["final_live_bytes"] == 0
    assert result["largest_request"] < 8 * 12**2


@pytest.mark.parametrize("orbitals,occupied", [(0, 0), (2, 0), (2, 2), (2, 3)])
@pytest.mark.parametrize("mode", ["plan-rhs", "plan-weights"])
def test_invalid_dimensions_are_not_resource_acceptance(
    dense_probe: Path, orbitals: int, occupied: int, mode: str
) -> None:
    result = invoke(dense_probe, mode, orbitals, occupied, DEFAULT_BUDGET)
    assert result["error"] == "invalid_argument"


@pytest.mark.parametrize("mode", ["invalid-response", "nonfinite-response"])
def test_invalid_response_is_rejected_before_dense_rhs_construction(
    dense_probe: Path, mode: str
) -> None:
    result = invoke(dense_probe, mode, 12, 6, DEFAULT_BUDGET, synthetic(12, 6))
    assert result["error"] == "invalid_argument"
    assert result["largest_request"] < 8 * 12**2
    assert result["one"] == result["two"] == []
    assert result["final_live_bytes"] == 0
