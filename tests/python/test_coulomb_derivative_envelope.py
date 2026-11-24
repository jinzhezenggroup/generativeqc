"""Host resource and independent libcint gates for frozen-derivative pruning."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from test_factorized_second import EXPONENTS, POSITIONS, reference

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx) -> Path:
    return native_cxx.build_executable(
        [ROOT / "tests/native/coulomb_derivative_envelope_probe.cpp"],
        tmp_path_factory.mktemp("derivative-envelope") / "probe",
        compile_args=("-std=c++20", "-O2", f"-I{ROOT / 'src'}"),
    )


def _run(
    probe: Path,
    angular: tuple[int, ...],
    positions: np.ndarray,
    exponents: np.ndarray,
    *,
    budget: float = 1e-14,
    limit: int,
    topology: tuple[int, ...] = (),
) -> tuple[str, str]:
    rows = [f"{len(angular)} {budget} {limit}"]
    for order, center, exponent in zip(angular, positions, exponents, strict=True):
        rows.append(" ".join(map(str, [order, *center, 1, exponent, 1])))
    if topology:
        rows.append(" ".join(map(str, [max(topology) + 1, *topology])))
    completed = subprocess.run(
        [str(probe)],
        input="\n".join(rows),
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    return completed.stdout, completed.stderr


def _columns(
    probe: Path,
    angular: tuple[int, ...],
    positions: np.ndarray,
    exponents: np.ndarray,
    topology: tuple[int, ...] = (),
) -> np.ndarray:
    stdout, _ = _run(probe, angular, positions, exponents, limit=0, topology=topology)
    return np.array(stdout.splitlines(), dtype=float)


@pytest.mark.parametrize(
    "angular", ((0, 0, 0, 0), (0, 1, 0, 1), (1, 0, 1, 1), (1, 1, 1, 1))
)
@pytest.mark.parametrize("distance", (0.0, 1.0, 24.0))
def test_quartet_bound_against_libcint(
    probe: Path, angular: tuple[int, ...], distance: float
) -> None:
    pytest.importorskip("pyscf")
    positions = POSITIONS * distance
    bounds = _columns(probe, angular, positions, EXPONENTS)
    # Divide by the symmetry-orbit factor to check one ordered representative,
    # summing all component magnitudes for each independently displaced leg.
    _, gradient = reference(0, angular, positions)
    for column, right in enumerate(np.eye(12).reshape(12, 4, 3)):
        atom, axis = divmod(column, 3)
        assert np.sum(abs(gradient[3 * atom + axis])) <= bounds[atom] / 8
        step = 0.00025
        plus = reference(0, angular, positions + step * right)[1]
        minus = reference(0, angular, positions - step * right)[1]
        mixed = (plus - minus) / (2 * step)
        assert np.max(np.sum(abs(mixed), axis=-1)) <= bounds[atom] / 8


def test_shared_right_atom_counts_both_pair_tangents(probe: Path) -> None:
    pytest.importorskip("pyscf")
    angular = (1, 0, 1, 1)
    topology = (0, 1, 1, 2)
    positions = POSITIONS[list(topology)]
    owners = np.array(topology)
    bounds = _columns(probe, angular, positions, EXPONENTS, topology)
    _, gradient = reference(0, angular, positions)
    gradient = gradient.reshape(4, 3, -1)
    for atom, bound in enumerate(bounds):
        legs = owners == atom
        assert np.max(np.sum(abs(gradient[legs].sum(axis=0)), axis=-1)) <= bound / 8
        for axis in np.eye(3):
            step = 0.00025
            right = legs[:, None] * axis
            plus = reference(0, angular, positions + step * right)[1]
            minus = reference(0, angular, positions - step * right)[1]
            mixed = ((plus - minus) / (2 * step)).reshape(4, 3, -1)
            for left_atom in range(len(bounds)):
                left = mixed[owners == left_atom].sum(axis=0)
                assert np.max(np.sum(abs(left), axis=-1)) <= bound / 8


def test_omissions_obey_aggregate_budget_and_census_bound(probe: Path) -> None:
    positions = np.arange(12)[:, None] * np.array([2.2, 0.1, 0.2])
    angular = (0,) * len(positions)
    exponents = np.full(len(positions), 0.66)
    pairs = len(positions) * (len(positions) + 1) // 2
    census = pairs * (pairs + 1) // 2
    budget = 1e-14
    # The certificate cannot use a smaller rank budget than its actual work.
    stdout, stderr = _run(probe, angular, positions, exponents, limit=pairs)
    assert stdout.splitlines() == ["0"]
    assert stderr.strip() == "candidate_checks=0"
    limit = census - 1
    stdout, stderr = _run(
        probe, angular, positions, exponents, budget=budget, limit=limit
    )
    bound_text, *sources = stdout.splitlines()
    bound = float(bound_text)
    omitted = [tuple(map(int, source.split())) for source in sources]
    checks = int(stderr.split("=")[-1])
    assert 0 < bound <= budget
    assert 0 < checks <= limit
    assert 0 < len(omitted) < census
    assert omitted == sorted(set(omitted))
    # Independently collect the primitive envelopes of all omitted sources.
    # Each only affects its own atoms.
    charged = np.zeros(len(positions))
    for source in omitted:
        atoms = sorted(set(source))
        topology = tuple(atoms.index(atom) for atom in source)
        charged[atoms] += _columns(
            probe,
            (0, 0, 0, 0),
            positions[list(source)],
            exponents[list(source)],
            topology,
        )
    assert 0 < charged.max() <= bound
