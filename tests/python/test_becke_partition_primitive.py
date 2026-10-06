"""Host gates for an exact dependency-domain candidate, not GPU speed evidence."""

from __future__ import annotations

import ctypes as ct
import shutil
import subprocess
from dataclasses import replace
from typing import cast

import numpy as np
import pytest
import test_becke_cooperative as retained
import test_becke_phased as phased
from generativeqc_compiler.integral.expr import Graph
from generativeqc_compiler.xc.becke_partition import (
    BeckePartitionDerivativeOp,
    emit_becke_partition_derivative,
    plan_becke_partition_derivative,
    recognize_becke_partition_derivative,
)
from generativeqc_compiler.xc.grid_native import emit_grid_adjoint, emit_grid_partials
from generativeqc_compiler.xc.grid_response_ir import grid_response_program


def operation(iterations: int = 3) -> BeckePartitionDerivativeOp:
    """Match actual canonical scalar AD roots for the requested switch order."""
    result = recognize_becke_partition_derivative(
        ratio=grid_response_program("ratio", iterations),
        logarithm=grid_response_program("log", iterations),
        switch=grid_response_program("becke", iterations),
        iterations=iterations,
    )
    assert result is not None
    return result


def candidate_harness() -> str:
    """Reuse the established transactional harness with different domains only."""
    source = phased.HARNESS

    def change(before: str, after: str) -> None:
        nonlocal source
        assert source.count(before) == 1, "shared phase harness changed"
        source = source.replace(before, after)

    change(
        "static size_t pairs_evaluated, norms_evaluated, logs_evaluated;",
        """static size_t pairs_evaluated, norms_evaluated, logs_evaluated;
static bool partition_enabled;
static size_t reverse_visits, gather_visits, live_atoms;
static std::vector<size_t> live_indices, live_counts;
static std::vector<double> common_bars;""",
    )
    change(
        "  using namespace generativeqc_grid_phased;\n  for (size_t point = 0; point < work.points; ++point)",
        """  using namespace generativeqc_grid_phased;
  using namespace generativeqc_grid_partition;
  DerivativeWorkspace input{work, live_indices.data(), live_counts.data(), common_bars.data()};
  for (size_t point = 0; point < work.points; ++point)""",
    )
    change(
        "    if (!point_normalize_phase(work, point, owners[point], seeds[point], local_ratio)) return false;",
        """    if (partition_enabled) {
      if (!partition_normalize_phase(input, point, owners[point], seeds[point], local_ratio)) return false;
      live_atoms += input.counts[point];
    } else if (!point_normalize_phase(work, point, owners[point], seeds[point], local_ratio)) return false;""",
    )
    reverse = """  for (size_t point = 0; point < work.points; ++point)
    for (size_t first = 0; first < work.atoms; ++first)
      for (size_t second = 0; second < first; ++second)
        if (!pair_reverse_phase(work, point, first, second, geometry, counted_log)) return false;"""
    change(
        reverse,
        """  if (partition_enabled) {
    for (size_t point = 0; point < work.points; ++point)
      for (size_t atom = 0; atom < work.atoms; ++atom) {
        size_t visits = 0;
        if (!partition_reverse_atom_phase(input, point, atom, geometry, counted_log, visits)) return false;
        reverse_visits += visits;
      }
  } else {
"""
        + reverse
        + "\n  }",
    )
    change(
        "    for (size_t atom = 0; atom < work.atoms; ++atom) atom_gather_phase(work, point, atom);",
        """    for (size_t atom = 0; atom < work.atoms; ++atom) {
      if (partition_enabled) {
        partition_gather_phase(input, point, atom);
        gather_visits += work.zero_counts(point)[atom] <= 1
            ? work.atoms - 1 : input.counts[point];
      } else atom_gather_phase(work, point, atom);
    }""",
    )
    change(
        "  pairs_evaluated = norms_evaluated = logs_evaluated = 0;",
        """  pairs_evaluated = norms_evaluated = logs_evaluated = 0;
  reverse_visits = gather_visits = live_atoms = 0;
  partition_enabled = phased == 2;
  live_indices.assign(atom_count * point_count, 0);
  live_counts.assign(point_count, 0);
  common_bars.assign(point_count, 0);""",
    )
    return (
        source
        + """
extern "C" void partition_metrics(size_t* counts) {
  counts[0] = reverse_visits;
  counts[1] = gather_visits;
  counts[2] = live_atoms;
}
"""
    )


@pytest.fixture(scope="module", params=[1, 3, 5])
def helper(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> ct.CDLL:
    """Compile the same AD owner and both dependency routes with verified ccache."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("C++ compiler and verified ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    directory = tmp_path_factory.mktemp("becke-partition-primitive")
    source, library = directory / "probe.cpp", directory / "probe.so"
    source.write_text(
        emit_grid_adjoint()
        + emit_grid_partials(request.param)
        + emit_becke_partition_derivative(operation(request.param))
        + candidate_harness()
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
    result.partition_metrics.argtypes = [ct.POINTER(ct.c_size_t)]
    result.partition_metrics.restype = None
    result.iterations = request.param
    return result


@pytest.mark.parametrize("atoms", [1, 2, 3, 12, 33, 48, 96])
@pytest.mark.parametrize("cached", [False, True])
def test_sparse_matches_phased_generic_and_changed_geometry(
    helper: ct.CDLL, atoms: int, cached: bool
) -> None:
    """Every live pair has one writer and two ordered incident consumers."""
    rng = np.random.default_rng(189400 + atoms)
    centers = rng.normal(size=(atoms, 3)) * 2
    points = rng.normal(size=(17, 3)) * 3
    owners = rng.integers(atoms, size=len(points), dtype=np.int64)
    seeds = rng.normal(size=len(points))
    for geometry in (centers, centers + rng.normal(size=centers.shape) * 0.05, centers):
        baseline = retained.run(helper, points, geometry, owners, seeds, cached, 1)
        actual = retained.run(helper, points, geometry, owners, seeds, cached, 2)
        assert baseline[0] == actual[0] == 0
        np.testing.assert_allclose(actual[1], baseline[1], rtol=5e-13, atol=2e-13)
        assert actual[2][0] == baseline[2][0] == len(points) * atoms * (atoms - 1) // 2
        metrics = np.zeros(3, dtype=np.uintp)
        helper.partition_metrics(metrics.ctypes.data_as(ct.POINTER(ct.c_size_t)))
        assert metrics[0] <= actual[2][0]
        assert metrics[1] == 2 * metrics[0]
        assert metrics[2] <= len(points) * atoms


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
def test_sparse_edge_and_failure_semantics(helper: ct.CDLL, case: str) -> None:
    """The existing independent edge suite exercises this candidate route."""
    retained.test_cooperative_edge_semantics(candidate_routes(helper), case)


def test_sparse_independent_decimal_fd_and_invariances(helper: ct.CDLL) -> None:
    retained.test_cooperative_independent_decimal_fd_translation_and_permutation(
        candidate_routes(helper)
    )


def test_exact_saturated_domain_reduces_reverse_and_gather_work(
    helper: ct.CDLL,
) -> None:
    """A measured liveness domain is required; random points need not have zeros."""
    atoms = 96
    centers = np.zeros((atoms, 3))
    centers[:, 0] = np.arange(atoms, dtype=float)
    points = np.array([[-0.75, 0.0, 0.0]])
    owners = np.array([0], dtype=np.int64)
    seeds = np.array([0.3])
    baseline = retained.run(helper, points, centers, owners, seeds, True, 1)
    actual = retained.run(helper, points, centers, owners, seeds, True, 2)
    assert baseline[0] == actual[0] == 0
    np.testing.assert_array_equal(actual[1], baseline[1])
    metrics = np.zeros(3, dtype=np.uintp)
    helper.partition_metrics(metrics.ctypes.data_as(ct.POINTER(ct.c_size_t)))
    assert metrics[0] < actual[2][0]
    assert metrics[1] == 2 * metrics[0]
    assert metrics[2] < atoms


def test_recognizer_rejects_changed_ad_with_stale_identity() -> None:
    ratio = grid_response_program("ratio")
    changed = replace(ratio, roots=(ratio.roots[0] + 1, ratio.roots[1]))
    arguments = {
        "ratio": changed,
        "logarithm": grid_response_program("log"),
        "switch": grid_response_program("becke"),
    }
    assert recognize_becke_partition_derivative(**arguments) is None
    arguments["ratio"] = ratio
    assert (
        recognize_becke_partition_derivative(
            **arguments, partition="unsupported-adjusted-partition"
        )
        is None
    )
    assert operation().identity == operation().identity
    assert operation(1).identity != operation(5).identity


def test_resource_admission_charges_indices_and_concurrent_owners() -> None:
    plan = plan_becke_partition_derivative(
        operation=operation(), atoms=96, points=256, budget_bytes=1 << 30
    )
    assert plan is not None
    assert plan.index_bytes == 98 * 256 * 8
    assert plan.scratch_bytes == 39_917_568
    for spare, admitted in ((-1, False), (0, True), (1, True)):
        actual = plan_becke_partition_derivative(
            operation=operation(),
            atoms=96,
            points=256,
            budget_bytes=1000 + plan.scratch_bytes + spare,
            occupied_bytes=1000,
        )
        assert (actual is not None) == admitted


def test_two_zero_liveness_is_a_canonical_ad_property_not_a_magnitude_cutoff() -> None:
    graph = Graph()
    factors = [graph.variable(f"factor_{index}") for index in range(3)]
    product = graph.multiply_many(factors)
    derivatives = [graph.differentiate(product, factor) for factor in factors]
    for zero_count in range(4):
        values = {
            f"factor_{index}": 0.0 if index < zero_count else 0.5 for index in range(3)
        }
        actual = [graph.evaluate(root, values) for root in derivatives]
        assert (all(value == 0 for value in actual)) == (zero_count >= 2)


def candidate_routes(helper: ct.CDLL) -> ct.CDLL:
    """Select actual compiled candidate execution for the reused non-generic gates."""

    class CandidateRoutes:
        iterations = helper.iterations

        def run(self, *arguments: object) -> int:
            values = list(arguments)
            values[7] = 2 if values[7] else 0
            return helper.run(*values)

    return cast("ct.CDLL", CandidateRoutes())
