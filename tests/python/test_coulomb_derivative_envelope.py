"""Host resource and independent libcint gates for frozen-derivative pruning."""

from __future__ import annotations

import subprocess
from itertools import combinations_with_replacement
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
    value_pair_bounds: bool = False,
    rank_bitmap: bool = False,
) -> tuple[str, str]:
    rows = [f"{len(angular)} {budget} {limit}"]
    for order, center, exponent in zip(angular, positions, exponents, strict=True):
        rows.append(" ".join(map(str, [order, *center, 1, exponent, 1])))
    if topology:
        rows.append(" ".join(map(str, [max(topology) + 1, *topology])))
    completed = subprocess.run(
        [str(probe)]
        + (["--value-pair-bounds"] if value_pair_bounds else [])
        + (["--rank-bitmap"] if rank_bitmap else []),
        input="\n".join(rows),
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    return completed.stdout, completed.stderr


def test_binary64_outward_rounding_matches_nextafter(probe: Path) -> None:
    """Exercise the finite, subnormal, signed-zero and nonfinite boundaries."""
    subprocess.run([str(probe), "--rounding"], check=True, timeout=10)


def test_component_value_bounds_and_signed_coulomb_omission(probe: Path) -> None:
    """Check Cartesian s/p component bounds against libcint without a cached J."""
    gto = pytest.importorskip("pyscf.gto")
    angular = (0, 1, 1, 0)
    positions = np.array([[0.0, 0, 0], [0.9, 0.1, 0], [16.0, 0, 0], [16.8, 0, 0]])
    names = ("H", "He", "Li", "Be")
    molecule = gto.M(
        atom=list(zip((f"ghost-{name}" for name in names), positions, strict=True)),
        basis={
            name: [[order, [exponent, 1.0]]]
            for name, order, exponent in zip(names, angular, EXPONENTS, strict=True)
        },
        cart=True,
        unit="Bohr",
        verbose=0,
    )
    # Pair bounds consume input coefficients before Cartesian projection.
    # Convert libcint's angular/radial normalization to the same monomials.
    from test_factorized_second import _gaussian_squared_norm, cartesian_components

    squared_norms = [
        _gaussian_squared_norm(exponent, component)
        for exponent, order in zip(EXPONENTS, angular, strict=True)
        for component in cartesian_components(order)
    ]
    scales = np.sqrt(molecule.intor("int1e_ovlp_cart").diagonal() / squared_norms)
    with molecule.with_integral_screen(1e-80):
        eri = molecule.intor("int2e_cart") / np.einsum("i,j,k,l->ijkl", *[scales] * 4)
    stdout, _ = _run(
        probe, angular, positions, EXPONENTS, limit=0, value_pair_bounds=True
    )
    shell_norms = np.fromstring(stdout, sep=" ").reshape(4, 4)
    owners = np.repeat(np.arange(4), [2 * order + 1 for order in angular])
    norms = shell_norms[owners[:, None], owners]
    bound = np.einsum("ij,kl->ijkl", norms, norms)
    assert np.all(abs(eri) <= bound)
    threshold = float.fromhex("0x1p-70")
    omitted = bound < threshold
    assert omitted.any() and (~omitted).any()
    rng = np.random.default_rng(41)
    density = rng.uniform(-16, 16, size=norms.shape)
    density = (density + density.T) / 2
    # Sum discarded terms directly: subtracting two order-one J values
    # would measure cancellation in the test instead of the omission.
    delta = np.einsum("ijkl,kl->ij", np.where(omitted, eri, 0), density)
    assert 0 < np.max(abs(delta)) <= threshold * np.sum(abs(density))


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
    # A more generous work lease must not change the ledger or its canonical
    # publication order, even when it admits different rank storage.
    assert _run(
        probe, angular, positions, exponents, budget=budget, limit=4 * census
    ) == (stdout, stderr)
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


def test_rank_bitmap_encodes_same_certified_sources(probe: Path) -> None:
    positions = np.arange(12)[:, None] * np.array([2.2, 0.1, 0.2])
    angular = (0, 1) * 6
    exponents = np.full(len(positions), 0.66)
    source_text, source_checks = _run(probe, angular, positions, exponents, limit=12000)
    bitmap_text, bitmap_checks = _run(
        probe, angular, positions, exponents, limit=12000, rank_bitmap=True
    )
    source_bound, *sources = source_text.splitlines()
    bitmap_bound, metadata, *words = bitmap_text.splitlines()
    ranks, omitted = map(int, metadata.split())
    pairs = list(combinations_with_replacement(range(len(positions)), 2))
    quartets = [
        first + second for first, second in combinations_with_replacement(pairs, 2)
    ]
    bitmap = sum(int(word) << (64 * index) for index, word in enumerate(words))
    assert bitmap_checks == source_checks and bitmap_bound == source_bound
    assert ranks == len(quartets) and not bitmap >> ranks
    assert omitted == bitmap.bit_count() == len(sources) > 0
    assert [
        quartet for rank, quartet in enumerate(quartets) if (bitmap >> rank) & 1
    ] == [tuple(map(int, source.split())) for source in sources]


def test_shared_shell_certificate_matches_exhaustive_ledger(probe: Path) -> None:
    """Aliased centers charge one ledger, in exact bound and source order."""
    topology = (0, 0, 1, 1, 2, 2)
    angular = (0, 1) * 3
    positions = np.asarray(topology)[:, None] * np.array([8.5, 0.45, -0.6])
    exponents = np.tile([0.37, 0.82], 3)
    budget = 1e-14
    pairs = tuple(combinations_with_replacement(range(len(topology)), 2))
    candidates = []
    for left, right in combinations_with_replacement(pairs, 2):
        quartet = left + right
        atoms = tuple(sorted({topology[index] for index in quartet}))
        columns = _columns(
            probe,
            tuple(angular[index] for index in quartet),
            positions[list(quartet)],
            exponents[list(quartet)],
            tuple(atoms.index(topology[index]) for index in quartet),
        )
        if columns.max() <= budget:
            candidates.append((float(columns.max()), quartet, list(atoms), columns))

    ledgers = np.zeros(3)
    omitted = []
    for _, quartet, atoms, columns in sorted(candidates, key=lambda entry: entry[:2]):
        charged = np.nextafter(ledgers[atoms] + columns, np.inf)
        if np.all(charged <= budget):
            ledgers[atoms] = charged
            omitted.append(quartet)
    assert 0 < len(omitted) < len(pairs) * (len(pairs) + 1) // 2

    stdout, _ = _run(
        probe,
        angular,
        positions,
        exponents,
        budget=budget,
        limit=2048,
        topology=topology,
    )
    bound, *sources = stdout.splitlines()
    assert float(bound) == ledgers.max()
    assert [tuple(map(int, source.split())) for source in sources] == sorted(omitted)
