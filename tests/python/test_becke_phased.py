"""Host execution of the phase plan: correctness/work counts, not GPU timings."""

from __future__ import annotations

import ctypes as ct
import shutil
import subprocess

import numpy as np
import pytest
import test_becke_cooperative as retained
import test_becke_tiled_cooperative as tiled
from generativeqc_compiler.xc.grid_native import emit_grid_adjoint, emit_grid_partials
from generativeqc_compiler.xc.grid_phased import emit_phased_becke, plan_phased_becke

HARNESS = r"""
#include <cstdint>
#include <vector>
static size_t pairs_evaluated, norms_evaluated, logs_evaluated;
static auto counted_log(double value) { ++logs_evaluated; return local_log(value); }
static auto counted_pair(double value) { ++pairs_evaluated; return local_becke(value); }
static auto counted_norm(double first, double second, double third) {
  ++norms_evaluated; return local_norm(first, second, third);
}
template <class Geometry>
bool phases(generativeqc_grid_phased::Workspace work, const double* points,
    const double* centers, const int64_t* owners, const double* seeds, Geometry geometry) {
  using namespace generativeqc_grid_phased;
  for (size_t point = 0; point < work.points; ++point)
    for (size_t atom = 0; atom < work.atoms; ++atom)
      if (!distance_phase(work, point, atom, points, centers, counted_norm)) return false;
  for (size_t point = 0; point < work.points; ++point)
    for (size_t first = 0; first < work.atoms; ++first)
      for (size_t second = 0; second < first; ++second)
        if (!pair_primal_phase(work, point, first, second, geometry, counted_log, counted_pair))
          return false;
  for (size_t point = 0; point < work.points; ++point)
    for (size_t atom = 0; atom < work.atoms; ++atom)
      atom_logs_phase(work, point, atom);
  for (size_t point = 0; point < work.points; ++point)
    if (!point_normalize_phase(work, point, owners[point], seeds[point], local_ratio)) return false;
  for (size_t point = 0; point < work.points; ++point)
    for (size_t first = 0; first < work.atoms; ++first)
      for (size_t second = 0; second < first; ++second)
        if (!pair_reverse_phase(work, point, first, second, geometry, counted_log)) return false;
  for (size_t point = 0; point < work.points; ++point)
    for (size_t atom = 0; atom < work.atoms; ++atom) atom_gather_phase(work, point, atom);
  for (size_t point = 0; point < work.points; ++point)
    for (size_t atom = 0; atom < work.atoms; ++atom)
      if (!point_motion_phase(work, point, atom, owners[point])) return false;
  return true;
}
extern "C" int run(const double* points, size_t point_count, const double* centers,
    size_t atom_count, const int64_t* owners, const double* seeds, bool cached,
    size_t phased, double* output, size_t* counts) {
  using namespace generativeqc_grid_adjoint;
  using generativeqc_grid_phased::Workspace;
  pairs_evaluated = norms_evaluated = logs_evaluated = 0;
  std::vector<CenterPair> geometry(atom_count * (atom_count - 1) / 2);
  auto* cached_pairs = cached && !geometry.empty() ? geometry.data() : nullptr;
  if (!prepare_center_geometry(centers, atom_count, 1e-12, cached_pairs,
                               counted_norm, local_ratio_geometry)) return -2;
  std::vector<double> gradient(3 * atom_count, 0);
  if (phased) {
    std::vector<double> pairs(4 * geometry.size() * point_count + 2, 987654);
    std::vector<double> fields(11 * atom_count * point_count + 2, 987654);
    std::vector<double> maximum(point_count + 2, 987654);
    std::vector<size_t> zeros(atom_count * point_count + 2, 987654);
    Workspace work{atom_count, point_count, pairs.data() + 1, fields.data() + 1,
                   zeros.data() + 1, maximum.data() + 1};
    const bool valid = cached_pairs
        ? phases(work, points, centers, owners, seeds,
            PreparedCenterGeometry<decltype(&local_ratio_prepared)>{cached_pairs, local_ratio_prepared})
        : phases(work, points, centers, owners, seeds,
            DirectCenterGeometry<decltype(&counted_norm), decltype(&local_ratio)>{
                centers, counted_norm, local_ratio});
    for (const auto* panel : {&pairs, &fields, &maximum})
      if (panel->front() != 987654 || panel->back() != 987654) return -3;
    if (zeros.front() != 987654 || zeros.back() != 987654) return -3;
    if (!valid) return -2;
    for (size_t point = 0; point < point_count; ++point)
      for (size_t atom = 0; atom < atom_count; ++atom)
        for (size_t axis = 0; axis < 3; ++axis)
          gradient[3 * atom + axis] += work.field(8 + axis, point)[atom];
  } else {
    std::vector<double> logs(atom_count), products(atom_count), bars(atom_count), distances_bar(atom_count);
    std::vector<size_t> zeros(atom_count);
    std::vector<std::array<double, 4>> distances(atom_count);
    for (size_t point = 0; point < point_count; ++point)
      if (!contract_point_prepared(points + 3 * point, centers, atom_count, owners[point],
          seeds[point], gradient.data(), logs.data(), products.data(), bars.data(),
          distances_bar.data(), zeros.data(), distances.data(), counted_norm, local_ratio,
          counted_log, counted_pair, cached_pairs, local_ratio_prepared)) return -2;
  }
  for (double value : gradient) if (!std::isfinite(value)) return -2;
  std::copy(gradient.begin(), gradient.end(), output);
  counts[0] = pairs_evaluated; counts[1] = norms_evaluated; counts[2] = logs_evaluated;
  return 0;
}
"""


@pytest.fixture(scope="module", params=[1, 3, 5])
def helper(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> ct.CDLL:
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("C++ compiler and verified ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    directory = tmp_path_factory.mktemp("becke-phased")
    source, library = directory / "probe.cpp", directory / "probe.so"
    source.write_text(
        emit_grid_adjoint()
        + emit_grid_partials(request.param)
        + emit_phased_becke()
        + HARNESS
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


@pytest.mark.parametrize("atoms", [1, 2, 5, 32, 33, 96, 128])
@pytest.mark.parametrize("cached", [False, True])
def test_phases_halve_pair_production_without_changing_result(
    helper: ct.CDLL, atoms: int, cached: bool
) -> None:
    rng = np.random.default_rng(10300 + atoms)
    centers = rng.normal(size=(atoms, 3)) * 2
    points = rng.normal(size=(13, 3)) * 3
    owners = rng.integers(atoms, size=len(points), dtype=np.int64)
    seeds = rng.normal(size=len(points))
    moved = centers + rng.normal(size=centers.shape) * 0.1
    for geometry in (centers, moved, centers):
        expected = retained.run(helper, points, geometry, owners, seeds, cached, 0)
        actual = retained.run(helper, points, geometry, owners, seeds, cached, 1)
        assert actual[0] == expected[0] == 0
        np.testing.assert_allclose(actual[1], expected[1], rtol=5e-13, atol=2e-13)
        pairs = atoms * (atoms - 1) // 2
        assert actual[2][0] == len(points) * pairs
        assert expected[2][0] == 2 * actual[2][0]
        if cached:
            assert actual[2][1] == pairs + len(points) * atoms


@pytest.mark.parametrize(
    "case",
    [
        "saturated",
        "rounded_zero",
        "positive_zero",
        "negative_zero",
        "coincident",
        "collision",
        "nonfinite",
        "tiny",
        "huge",
        "empty",
    ],
)
def test_phase_edge_semantics(helper: ct.CDLL, case: str) -> None:
    retained.test_cooperative_edge_semantics(helper, case)


def test_phase_independent_oracle_and_invariances(helper: ct.CDLL) -> None:
    retained.test_cooperative_independent_decimal_fd_translation_and_permutation(helper)
    retained.test_single_atom_point_reuse_then_collision_is_collective(helper)
    tiled.test_tiled_multi_strip_independent_decimal_directional_derivative(helper)


def test_phase_point_tail(helper: ct.CDLL) -> None:
    rng = np.random.default_rng(10403)
    centers = rng.normal(size=(5, 3))
    points = rng.normal(size=(257, 3))
    owners = rng.integers(5, size=257, dtype=np.int64)
    seeds = rng.normal(size=257)
    expected = retained.run(helper, points, centers, owners, seeds, True, 0)
    actual = retained.run(helper, points, centers, owners, seeds, True, 1)
    assert actual[0] == expected[0] == 0
    np.testing.assert_allclose(actual[1], expected[1], rtol=5e-13, atol=2e-13)


def test_phase_admission_charges_all_live_panels_and_falls_back() -> None:
    plan = plan_phased_becke(atoms=96, points=256, budget_bytes=1 << 30)
    assert plan is not None
    assert plan.pair_bytes == 37_355_520
    assert plan.pair_evaluations == 256 * 4560
    assert plan.scratch_bytes == plan.pair_bytes + (12 * 96 + 1) * 256 * 8
    assert len(plan.phases) == 7
    for spare, admitted in ((0, True), (-1, False), (1, True)):
        actual = plan_phased_becke(
            atoms=96,
            points=256,
            budget_bytes=1000 + plan.scratch_bytes + spare,
            occupied_bytes=1000,
        )
        assert (actual is not None) == admitted
    assert plan_phased_becke(atoms=1, points=0, budget_bytes=0) is not None
    assert (
        plan_phased_becke(atoms=1, points=0, budget_bytes=0, occupied_bytes=1) is None
    )
    assert (
        plan_phased_becke(atoms=1 << 40, points=1 << 40, budget_bytes=1 << 62) is None
    )
    for field, value in (("atoms", 0), ("points", -1), ("points", True)):
        arguments = {"atoms": 96, "points": 256, "budget_bytes": 1 << 30}
        arguments[field] = value
        with pytest.raises((TypeError, ValueError)):
            plan_phased_becke(**arguments)
