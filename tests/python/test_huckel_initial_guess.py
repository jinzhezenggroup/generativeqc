"""Native Modified Huckel seeds against retained, independent PySCF atomic HF."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from generativeqc import (
    Calculator,
    GridSpec,
    InitialGuessSpec,
    KsOptions,
    Primitive,
    ResourceBudget,
    Shell,
    _native,
)
from generativeqc.initial_guess import _huckel_numeric_capacity

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def probe(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: object
) -> Path:
    library = Path(_native.load_library(device="cpu")._name).resolve()
    directory = tmp_path_factory.mktemp("huckel")
    obj = directory / "probe.o"
    required_native_cxx.compile_object(
        ROOT / "tests/native/huckel_probe.cpp",
        obj,
        args=(
            "-std=c++20",
            "-O2",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
        ),
    )
    binary = directory / "probe"
    required_native_cxx.link(
        [obj], binary, args=(str(library), f"-Wl,-rpath,{library.parent}")
    )
    return binary


def _fixture_input(case: dict) -> str:
    lines = [
        f"{len(case['atoms'])} {len(case['shells'])} {case['charge']} {int(case['spherical'])}"
    ]
    lines += [" ".join(map(str, atom)) for atom in case["atoms"]]
    for atom, angular, primitives in case["shells"]:
        lines.append(f"{atom} {angular} {len(primitives)}")
        lines += [
            f"{exponent:.17g} {coefficient:.17g}"
            for exponent, coefficient in primitives
        ]
    return "\n".join(lines) + "\n"


def _cases() -> list[dict]:
    return json.loads((ROOT / "tests/data/huckel_reference.json").read_text())["cases"]


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["name"])
def test_native_seed_matches_independent_atomic_oracle(probe: Path, case: dict) -> None:
    result = subprocess.run(
        [str(probe)],
        input=_fixture_input(case),
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    inventory, values = result.stdout.strip().splitlines()
    dimension, solves, minimal, iterations, builds, capacity = map(
        int, inventory.split()
    )
    density = np.fromstring(values, sep=" ").reshape(dimension, dimension)
    assert solves == case["atomic_solves"]
    assert minimal == case["minimal_orbitals"]
    assert builds == iterations
    assert (builds == 0) == all(atom[0] == 1 for atom in case["atoms"])
    assert np.max(np.abs(density - case["density"])) < 2e-7
    overlap = np.asarray(case["overlap"])
    assert np.einsum("ij,ji->", density, overlap) == pytest.approx(
        case["electrons"], abs=1e-9
    )
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    occupations = np.linalg.eigvalsh(root @ density @ root)
    assert min(occupations) > -1e-9 and max(occupations) < 2 + 1e-9
    atomic_aos = [0] * len(case["atoms"])
    for atom, angular, _ in case["shells"]:
        if angular == 0 or (angular == 1 and case["atoms"][atom][0] >= 5):
            atomic_aos[atom] += 2 * angular + 1
    assert capacity == _huckel_numeric_capacity(
        dimension,
        max(atomic_aos),
        len(case["atoms"]),
        sum(len(shell[2]) for shell in case["shells"]),
        8,
    )


def test_huckel_descriptor_is_additive() -> None:
    policy = InitialGuessSpec("huckel")
    assert policy.native().kind == 5
    assert policy.native(automatic_cpu=True).kind == 6
    assert policy.native().radial_points == 0
    assert policy.to_payload()["kind"] == "huckel"


@pytest.mark.parametrize("second_exponent,solves", [(0.2, 1), (0.3, 2)])
def test_atomic_cache_uses_the_actual_basis(
    probe: Path, second_exponent: float, solves: int
) -> None:
    case = {
        "atoms": [[1, 0, 0, 0], [1, 0, 0, 1.4]],
        "charge": 0,
        "spherical": False,
        "shells": [[0, 0, [[0.2, 1.0]]], [1, 0, [[second_exponent, 1.0]]]],
    }
    result = subprocess.run(
        [str(probe)],
        input=_fixture_input(case),
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    inventory, values = result.stdout.strip().splitlines()
    assert int(inventory.split()[1]) == solves
    exponents = np.array([0.2, second_exponent])
    energies = 1.5 * exponents - np.sqrt(8 * exponents / np.pi)
    cross = (2 * np.sqrt(exponents.prod()) / exponents.sum()) ** 1.5
    cross *= np.exp(-exponents.prod() / exponents.sum() * 1.4**2)
    overlap = np.array([[1.0, cross], [cross, 1.0]])
    delta = (energies[0] - energies[1]) / energies.sum()
    coupling = 0.5 * (1.75 + delta**2 - 0.75 * delta**4) * cross * energies.sum()
    hamiltonian = np.array([[energies[0], coupling], [coupling, energies[1]]])
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    orthogonalizer = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    _, orbitals = np.linalg.eigh(orthogonalizer @ hamiltonian @ orthogonalizer)
    coefficients = orthogonalizer @ orbitals[:, 0]
    expected = 2 * np.outer(coefficients, coefficients)
    actual = np.fromstring(values, sep=" ").reshape(2, 2)
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=0)


@pytest.mark.parametrize(
    "method", ["rhf", "lda-rks", "pbe-rks", "r2scan-rks", "pbe0-rks"]
)
def test_native_huckel_preserves_distinct_target_methods(method: str) -> None:
    atoms = [
        ("O", (0, 0, 0)),
        ("H", (0, -1.43233673, 1.10715266)),
        ("H", (0, 1.43233673, 1.10715266)),
    ]
    options = (
        {"ks_options": KsOptions(grid=GridSpec(16, 8, 16))} if method != "rhf" else {}
    )
    core = Calculator(method, "sto-3g", device="cpu", initial_guess=None, **options)
    seeded = Calculator(
        method,
        "sto-3g",
        device="cpu",
        initial_guess=InitialGuessSpec("huckel"),
        **options,
    )
    baseline = core.singlepoint(atoms, properties=("energy",))
    actual = seeded.singlepoint(atoms, properties=("energy",))
    assert actual.converged and baseline.converged
    assert actual.energy == pytest.approx(baseline.energy, abs=1e-8)
    assert actual.initial_guess["outcome"] == "used"
    assert actual.initial_guess["target_attempts"] == 1
    assert actual.initial_guess["preliminary_fock_builds"] > 0


def test_huckel_numeric_cap_preserves_bounded_core_fallback() -> None:
    atoms = [("H", (0, 0, 0)), ("H", (0, 0, 1.4))]
    actual = Calculator(
        "rhf",
        "sto-3g",
        device="cpu",
        initial_guess=InitialGuessSpec("huckel", maximum_numeric_bytes=1),
    ).singlepoint(atoms, properties=("energy",))
    baseline = Calculator(
        "rhf", "sto-3g", device="cpu", initial_guess=None
    ).singlepoint(atoms, properties=("energy",))
    assert actual.energy == pytest.approx(baseline.energy, abs=1e-12)
    assert actual.initial_guess["outcome"] == "budget_skipped"
    assert actual.initial_guess["preliminary_fock_builds"] == 0
    assert actual.initial_guess["target_attempts"] == 1


@pytest.mark.parametrize("guess", ["auto", InitialGuessSpec("huckel")])
def test_empty_atomic_subspace_preserves_budgeted_core_fallback(guess: object) -> None:
    """A valid target without atomic occupied channels still has a bounded fallback."""
    atoms = [("He", (0, 0, 0))]
    basis = (Shell(0, 1, (Primitive(1.0, 1.0),)),)
    baseline = Calculator("rhf", basis, device="cpu", initial_guess=None).singlepoint(
        atoms
    )
    calc = Calculator("rhf", basis, device="cpu", initial_guess=guess)
    peak = calc.estimate_resources([atoms]).peak_bytes["host"]
    bounded = Calculator(
        "rhf",
        basis,
        device="cpu",
        initial_guess=guess,
        resource_budget=ResourceBudget(host_bytes=peak),
    )
    actual = bounded.singlepoint(atoms)
    assert actual.converged and baseline.converged
    assert actual.energy == pytest.approx(baseline.energy, abs=1e-12)
    assert actual.initial_guess["outcome"] == "preparation_failed"
    assert actual.initial_guess["target_attempts"] == 1
    assert actual.initial_guess[
        "preparation_numeric_capacity"
    ] == _huckel_numeric_capacity(3, 0, 1, 1, 8)
