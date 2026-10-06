"""Independent host gates for dense pair coefficients and bounded fallbacks."""

from __future__ import annotations

import ctypes as ct
import shutil
import subprocess

import numpy as np
import pytest
import test_becke_cooperative as retained
import test_becke_phased as phased
from generativeqc_compiler.xc.becke_coefficients import (
    emit_becke_pair_coefficients,
    plan_becke_pair_coefficients,
)
from generativeqc_compiler.xc.grid_native import emit_grid_adjoint, emit_grid_partials
from test_becke_partition_primitive import candidate_routes, operation


def coefficient_harness() -> str:
    """Reuse the ordinary transactional owner with the dense storage schedule.

    Generic mode 0 and ordinary phased mode 1 stay independent comparators.
    Mode 2 selects coefficients only for prepared geometry; direct geometry
    falls back. Mode 3 poisons the two unused reverse words to prove that gather
    cannot accidentally consume the previous primal factors or dense xyz state.
    """
    source = phased.HARNESS

    def change(before: str, after: str) -> None:
        nonlocal source
        assert source.count(before) == 1, "shared phase harness changed"
        source = source.replace(before, after)

    change(
        "static size_t pairs_evaluated, norms_evaluated, logs_evaluated;",
        """static size_t pairs_evaluated, norms_evaluated, logs_evaluated;
static bool coefficient_enabled, poison_unused_pair_words;""",
    )
    change(
        "template <class Geometry>\nbool phases",
        """template <class Geometry>
bool selected_reverse(generativeqc_grid_phased::Workspace work, size_t point,
    size_t first, size_t second, Geometry geometry) {
  if constexpr (requires { geometry.pairs; })
    if (coefficient_enabled)
      return generativeqc_grid_coefficients::pair_coefficient_reverse_phase(
          work, point, first, second, geometry, counted_log);
  return generativeqc_grid_phased::pair_reverse_phase(
      work, point, first, second, geometry, counted_log);
}
template <class Geometry>
void selected_gather(generativeqc_grid_phased::Workspace work, size_t point,
    size_t atom, Geometry geometry) {
  if constexpr (requires { geometry.pairs; })
    if (coefficient_enabled) {
      generativeqc_grid_coefficients::atom_gather_coefficient_phase(
          work, point, atom, geometry.pairs);
      return;
    }
  generativeqc_grid_phased::atom_gather_phase(work, point, atom);
}
template <class Geometry>
bool phases""",
    )
    change(
        "pair_reverse_phase(work, point, first, second, geometry, counted_log)",
        "selected_reverse(work, point, first, second, geometry)",
    )
    change(
        "  for (size_t point = 0; point < work.points; ++point)\n    for (size_t atom = 0; atom < work.atoms; ++atom) atom_gather_phase(work, point, atom);",
        """  if constexpr (requires { geometry.pairs; })
    if (poison_unused_pair_words)
      for (size_t point = 0; point < work.points; ++point)
        for (size_t index = 0; index < work.atoms * (work.atoms - 1) / 2; ++index)
          for (size_t word = 2; word < 4; ++word)
            work.pair(word, index, point) = std::numeric_limits<double>::quiet_NaN();
  for (size_t point = 0; point < work.points; ++point)
    for (size_t atom = 0; atom < work.atoms; ++atom)
      selected_gather(work, point, atom, geometry);""",
    )
    change(
        "  pairs_evaluated = norms_evaluated = logs_evaluated = 0;",
        """  pairs_evaluated = norms_evaluated = logs_evaluated = 0;
  coefficient_enabled = phased >= 2;
  poison_unused_pair_words = phased == 3;""",
    )
    return source


@pytest.fixture(scope="module", params=[1, 3, 5])
def helper(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> ct.CDLL:
    """Compile the canonical AD once per switch iteration using verified ccache."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("C++ compiler and verified ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    directory = tmp_path_factory.mktemp("becke-coefficients")
    source, library = directory / "probe.cpp", directory / "probe.so"
    source.write_text(
        emit_grid_adjoint()
        + emit_grid_partials(request.param)
        + emit_becke_pair_coefficients(operation(request.param))
        + coefficient_harness()
    )
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = ct.CDLL(str(library))
    pointer = ct.POINTER(ct.c_double)
    result.run.argtypes = [
        pointer,
        ct.c_size_t,
        pointer,
        ct.c_size_t,
        ct.POINTER(ct.c_int64),
        pointer,
        ct.c_bool,
        ct.c_size_t,
        pointer,
        ct.POINTER(ct.c_size_t),
    ]
    result.run.restype = ct.c_int
    result.iterations = request.param
    return result


@pytest.mark.parametrize("atoms", [1, 2, 3, 12, 33, 48, 96, 128])
@pytest.mark.parametrize("cached", [False, True])
def test_coefficients_match_generic_phased_and_rebound_geometry(
    helper: ct.CDLL, atoms: int, cached: bool
) -> None:
    """Dense pair work and neighbor order survive changed-geometry rebinding."""
    rng = np.random.default_rng(1894200 + atoms)
    centers = rng.normal(size=(atoms, 3)) * 2
    points = rng.normal(size=(17, 3)) * 3
    owners = rng.integers(atoms, size=len(points), dtype=np.int64)
    seeds = rng.normal(size=len(points))
    for geometry in (centers, centers + rng.normal(size=centers.shape) * 0.05, centers):
        generic = retained.run(helper, points, geometry, owners, seeds, cached, 0)
        baseline = retained.run(helper, points, geometry, owners, seeds, cached, 1)
        actual = retained.run(helper, points, geometry, owners, seeds, cached, 2)
        assert generic[0] == baseline[0] == actual[0] == 0
        np.testing.assert_array_equal(actual[1], baseline[1])
        np.testing.assert_allclose(actual[1], generic[1], rtol=5e-12, atol=2e-11)
        assert actual[2][0] == baseline[2][0] == len(points) * atoms * (atoms - 1) // 2
        np.testing.assert_array_equal(actual[2], baseline[2])


@pytest.mark.parametrize(
    "case",
    [
        "saturated",
        "rounded_zero",
        "coincident",
        "collision",
        "invalid_owner",
        "nonfinite",
        "nan_seed",
        "huge",
        "empty",
    ],
)
def test_coefficient_edge_and_failure_semantics(helper: ct.CDLL, case: str) -> None:
    retained.test_cooperative_edge_semantics(candidate_routes(helper), case)


def test_coefficient_decimal_fd_translation_and_permutation(helper: ct.CDLL) -> None:
    retained.test_cooperative_independent_decimal_fd_translation_and_permutation(
        candidate_routes(helper)
    )


def test_unused_reverse_words_can_be_poisoned(helper: ct.CDLL) -> None:
    rng = np.random.default_rng(1894201)
    centers = rng.normal(size=(48, 3)) * 2
    points = rng.normal(size=(17, 3)) * 3
    owners = rng.integers(48, size=len(points), dtype=np.int64)
    seeds = rng.normal(size=len(points))
    baseline = retained.run(helper, points, centers, owners, seeds, True, 1)
    actual = retained.run(helper, points, centers, owners, seeds, True, 3)
    assert baseline[0] == actual[0] == 0
    np.testing.assert_array_equal(actual[1], baseline[1])


def test_coefficient_plan_charges_concurrent_storage_and_requires_geometry() -> None:
    plan = plan_becke_pair_coefficients(
        operation=operation(),
        atoms=96,
        points=256,
        budget_bytes=1 << 30,
        cached_geometry=True,
    )
    assert plan is not None
    assert plan.scratch_bytes == plan.phased.scratch_bytes == 39716864
    assert plan.reverse_write_bytes == 2 * 8 * 4560 * 256
    assert plan.gather_pair_read_bytes == 2 * plan.reverse_write_bytes
    assert plan.gather_geometry_read_bytes == 3 * 2 * 8 * 4560 * 256
    for cached, occupied, extra, admitted in [
        (True, 0, 0, True),
        (True, 1, 0, False),
        (True, 1, 1, True),
        (False, 0, 0, False),
    ]:
        result = plan_becke_pair_coefficients(
            operation=operation(),
            atoms=96,
            points=256,
            budget_bytes=plan.scratch_bytes + extra,
            occupied_bytes=occupied,
            cached_geometry=cached,
        )
        assert (result is not None) == admitted
    with pytest.raises(ValueError, match="boolean"):
        plan_becke_pair_coefficients(
            operation=operation(),
            atoms=96,
            points=256,
            budget_bytes=1 << 30,
            cached_geometry=1,
        )
