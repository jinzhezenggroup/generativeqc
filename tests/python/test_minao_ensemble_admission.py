"""Independent metric-spectrum oracle, strict gate and numeric inventory checks."""

from __future__ import annotations

import itertools
import json
import subprocess
import typing
from pathlib import Path

import numpy as np
import pytest
from generativeqc.initial_guess import _minao_numeric_capacity

ROOT = Path(__file__).resolve().parents[2]
CASES = json.loads((ROOT / "tests/data/minao_ensemble_admission.json").read_text())[
    "cases"
]


@pytest.fixture(scope="module")
def probe(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: typing.Any
) -> Path:
    folder = tmp_path_factory.mktemp("minao-admission")
    return required_native_cxx.build_executable(
        [
            ROOT / path
            for path in (
                "tests/native/minao_admission_probe.cpp",
                "src/scf/initial_guess/minao.cpp",
                "src/scf/initial_guess/density.cpp",
                "src/scf/reference/linalg.cpp",
                "src/scf/solver/proposal_control.cpp",
                "src/scf/preliminary_guess.cpp",
                "src/molecule/basis.cpp",
                "src/integrals/minao_basis.cpp",
            )
        ],
        folder / "probe",
        compile_args=(
            "-std=c++20",
            "-O2",
            "-ffunction-sections",
            "-fdata-sections",
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
        ),
        link_args=("-Wl,--gc-sections",),
    )


def active_set_projection(values: typing.Any, electrons: int) -> typing.Any:
    """Independent exact breakpoint solution, deliberately not native bisection."""
    values = np.asarray(values)
    if electrons == 2 * len(values):
        return np.full_like(values, 2.0)
    breaks = np.unique(np.r_[-values, 2.0 - values])
    for low, high in itertools.pairwise(breaks):
        midpoint = (low + high) / 2
        free = (values + midpoint > 0) & (values + midpoint < 2)
        full = values + midpoint >= 2
        if not free.any():
            continue
        shift = (electrons - 2 * full.sum() - values[free].sum()) / free.sum()
        if low - 1e-13 <= shift <= high + 1e-13:
            return np.clip(values + shift, 0, 2)
    raise AssertionError("no feasible active set")


def run(
    probe: Path,
    overlap: typing.Any,
    raw: typing.Any,
    electrons: int,
    numbers: typing.Sequence[int] = (2,),
) -> subprocess.CompletedProcess[str]:
    n = len(overlap)
    data = [
        n,
        electrons,
        len(numbers),
        *numbers,
        *np.asarray(overlap).ravel(),
        *np.asarray(raw).ravel(),
    ]
    return subprocess.run(
        [str(probe)],
        input=" ".join(map(str, data)),
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_admission_matches_independent_ensemble_oracle(probe: Path, case: dict) -> None:
    overlap = np.asarray(case["overlap"])
    raw = np.asarray(case["raw_density"])
    n, electrons = len(raw), case["electron_count"]
    values, vectors = np.linalg.eigh(overlap)
    root = (vectors * np.sqrt(values)) @ vectors.T
    x = (vectors / np.sqrt(values)) @ vectors.T
    normalized = raw * (electrons / np.einsum("ij,ji->", raw, overlap))
    spectrum, orbitals = np.linalg.eigh(root @ normalized @ root)
    occupations = active_set_projection(spectrum, electrons)
    expected = x @ (orbitals * occupations) @ orbitals.T @ x
    np.testing.assert_allclose(expected, case["expected_density"], atol=1e-11)
    numbers = [atom[0] for atom in case["atoms"]]
    result = run(probe, overlap, raw, electrons, numbers)
    assert result.returncode == 0, result.stderr
    peak, capacity = map(int, result.stdout.splitlines()[0].split())
    actual = np.fromstring(result.stdout.splitlines()[1], sep=" ").reshape(n, n)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual, actual.T, atol=1e-13)
    assert np.einsum("ij,ji->", actual, overlap) == pytest.approx(electrons, abs=1e-9)
    admitted_occupations = np.linalg.eigvalsh(root @ actual @ root)
    assert admitted_occupations.min() >= -1e-10
    assert admitted_occupations.max() <= 2 + 1e-10
    assert capacity == _minao_numeric_capacity(n, numbers)
    assert (
        peak <= capacity
    )  # All heap payloads through X/construction/strict validator.
    # Already-admissible MINAO projections stay unchanged to numerical accuracy.
    if case["name"] in ("h2", "na_cation"):
        np.testing.assert_allclose(actual, normalized, atol=1e-10)
    again = run(probe, overlap, raw, electrons, numbers)
    assert result.stdout == again.stdout


@pytest.mark.parametrize("n", [1, 2, 7, 64])
def test_full_rank_and_validator_memory_bound(probe: Path, n: int) -> None:
    result = run(probe, np.eye(n), np.eye(n), 2 * n)
    assert result.returncode == 0, result.stderr
    peak, cap = map(int, result.stdout.splitlines()[0].split())
    assert peak <= cap == _minao_numeric_capacity(n, [2])
    density = np.fromstring(result.stdout.splitlines()[1], sep=" ").reshape(n, n)
    np.testing.assert_allclose(density, 2 * np.eye(n), atol=1e-12)


@pytest.mark.parametrize(
    "overlap,raw,electrons",
    [
        (np.eye(2), np.eye(2), 6),
        (np.diag([1.0, 1e-12]), np.eye(2), 2),
        (np.eye(2), np.zeros((2, 2)), 2),
    ],
)
def test_infeasible_or_singular_admission_fails_closed(
    probe: Path, overlap: typing.Any, raw: typing.Any, electrons: int
) -> None:
    assert run(probe, overlap, raw, electrons).returncode == 2


@pytest.mark.parametrize("z", range(1, 19))
def test_public_and_native_inventory_match_all_source_elements(
    probe: Path, z: int
) -> None:
    result = subprocess.run(
        [str(probe), "capacity"],
        input=f"7 2 1 {z}",
        capture_output=True,
        text=True,
        check=True,
    )
    assert int(result.stdout) == _minao_numeric_capacity(7, [z])


@pytest.mark.parametrize("n", [True, 0, -1, 2**32, 2**63])
def test_public_inventory_fails_closed_before_huge_allocation(n: int) -> None:
    with pytest.raises(ValueError):
        _minao_numeric_capacity(n, [2])


@pytest.mark.parametrize("seed", range(12))
def test_rotated_metric_spectrum_against_independent_active_set(
    probe: Path, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    n = (2, 3, 7, 15)[seed % 4]
    transform = rng.normal(size=(n, n))
    overlap = transform @ transform.T + np.eye(n)
    transform = rng.normal(size=(n, n))
    raw = transform @ transform.T
    electrons = int(rng.integers(1, 2 * n + 1))
    values, vectors = np.linalg.eigh(overlap)
    root = (vectors * np.sqrt(values)) @ vectors.T
    x = (vectors / np.sqrt(values)) @ vectors.T
    normalized = raw * (electrons / np.einsum("ij,ji->", raw, overlap))
    values, vectors = np.linalg.eigh(root @ normalized @ root)
    expected = x @ (vectors * active_set_projection(values, electrons)) @ vectors.T @ x
    result = run(probe, overlap, raw, electrons)
    assert result.returncode == 0, result.stderr
    actual = np.fromstring(result.stdout.splitlines()[1], sep=" ").reshape(n, n)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
