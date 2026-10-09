"""Independent AD/Decimal/finite-difference gates for atom-adjoint graph cuts."""

from __future__ import annotations

import ctypes as ct
from typing import TYPE_CHECKING

import numpy as np
import pytest
import test_becke_cooperative as retained
import test_becke_phased as phased
from generativeqc_compiler.xc.becke_normalized_adjoint import (
    _emit_atom_weight,
    _emit_pair_term,
    emit_becke_normalized_adjoint,
)
from generativeqc_compiler.xc.grid_native import emit_grid_adjoint, emit_grid_partials
from generativeqc_compiler.xc.grid_phased import emit_phased_becke
from test_becke_partition_primitive import candidate_routes, operation

if TYPE_CHECKING:
    from conftest import NativeCxx


@pytest.fixture(scope="module", params=[1, 3, 5])
def helper(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    native_cxx: NativeCxx,
) -> ct.CDLL:
    """Execute the emitted candidate alongside unchanged generic/phased routes."""
    harness = phased.HARNESS.replace(
        "  for (size_t point = 0; point < work.points; ++point)\n"
        "    for (size_t first = 0; first < work.atoms; ++first)\n"
        "      for (size_t second = 0; second < first; ++second)\n"
        "        if (!pair_reverse_phase(work, point, first, second, geometry, counted_log)) return false;",
        "  if (normalized_enabled)\n"
        "    for (size_t point = 0; point < work.points; ++point)\n"
        "      for (size_t atom = 0; atom < work.atoms; ++atom)\n"
        "        generativeqc_grid_normalized::prepare_atom_weight(work, point, atom);\n"
        "  for (size_t point = 0; point < work.points; ++point)\n"
        "    for (size_t first = 0; first < work.atoms; ++first)\n"
        "      for (size_t second = 0; second < first; ++second)\n"
        "        if (!(normalized_enabled\n"
        "          ? generativeqc_grid_normalized::pair_reverse_phase(work, point, first, second, geometry, counted_log)\n"
        "          : pair_reverse_phase(work, point, first, second, geometry, counted_log))) return false;",
    )
    assert harness != phased.HARNESS
    harness = harness.replace(
        "static size_t pairs_evaluated, norms_evaluated, logs_evaluated;",
        "static size_t pairs_evaluated, norms_evaluated, logs_evaluated;\n"
        "static bool normalized_enabled;",
    ).replace(
        "  pairs_evaluated = norms_evaluated = logs_evaluated = 0;",
        "  pairs_evaluated = norms_evaluated = logs_evaluated = 0;\n"
        "  normalized_enabled = phased == 2;",
    )
    directory = tmp_path_factory.mktemp("becke-normalized-adjoint")
    source = directory / "probe.cpp"
    library = directory / "probe.so"
    source.write_text(
        emit_grid_adjoint()
        + emit_grid_partials(request.param)
        + emit_phased_becke()
        + emit_becke_normalized_adjoint(operation(request.param))
        + harness
        + r"""
extern "C" double cut_weight(double product, double seed) {
  std::array<double, 22> fields{};
  std::array<size_t, 2> zeros{};
  double maximum = 0;
  generativeqc_grid_phased::Workspace work{2, 1, nullptr, fields.data(), zeros.data(), &maximum};
  work.field(5, 0)[0] = product;
  work.field(6, 0)[0] = seed;
  generativeqc_grid_normalized::prepare_atom_weight(work, 0, 0);
  return work.field(7, 0)[0];
}
"""
    )
    native_cxx.build_shared(
        [source], library, compile_args=["-std=c++20", "-O2", "-ffp-contract=off"]
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
    result.cut_weight.argtypes = [ct.c_double, ct.c_double]
    result.cut_weight.restype = ct.c_double
    result.iterations = request.param
    return result


def test_generated_pullbacks_exclude_primal_transcendentals() -> None:
    """Graph cuts must remove work structurally, not rely on compiler inlining."""
    weight = _emit_atom_weight()
    pair = _emit_pair_term()
    assert "exp(" not in weight and "log(" not in pair
    assert "unused_product_ancestor" not in weight
    assert "maximum" not in weight and "logarithm" not in weight


@pytest.mark.parametrize("seed", [1e-100, -1e-100])
def test_premature_underflow_requests_original_pair_reverse(
    helper: ct.CDLL, seed: float
) -> None:
    """Do not zero an adjoint that the following reciprocal can make observable."""
    assert np.isnan(helper.cut_weight(1e-300, seed))
    assert helper.cut_weight(0, seed) == 0
    assert helper.cut_weight(1e-200, seed) == 1e-200 * seed


def test_overflow_requests_original_pair_reverse(helper: ct.CDLL) -> None:
    assert np.isnan(helper.cut_weight(float("inf"), 1))


@pytest.mark.parametrize("atoms", [1, 2, 12, 48, 96, 128])
@pytest.mark.parametrize("cached", [False, True])
def test_normalized_adjoint_matches_generic_and_phased(
    helper: ct.CDLL, atoms: int, cached: bool
) -> None:
    """Rebind moved geometry without retaining point-dependent adjoints."""
    generator = np.random.default_rng(1894900 + atoms)
    centers = generator.normal(size=(atoms, 3)) * 2
    points = generator.normal(size=(17, 3)) * 3
    owners = generator.integers(atoms, size=len(points), dtype=np.int64)
    seeds = generator.normal(size=len(points))
    for geometry in (centers, centers + 0.03, centers):
        generic = retained.run(helper, points, geometry, owners, seeds, cached, 0)
        baseline = retained.run(helper, points, geometry, owners, seeds, cached, 1)
        actual = retained.run(helper, points, geometry, owners, seeds, cached, 2)
        assert actual[0] == baseline[0] == generic[0] == 0
        np.testing.assert_allclose(actual[1], baseline[1], rtol=5e-12, atol=2e-11)
        np.testing.assert_allclose(actual[1], generic[1], rtol=5e-12, atol=2e-11)
        assert actual[2][0] == baseline[2][0]


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
def test_normalized_adjoint_edge_semantics(helper: ct.CDLL, case: str) -> None:
    retained.test_cooperative_edge_semantics(candidate_routes(helper), case)


def test_normalized_adjoint_decimal_and_finite_difference(helper: ct.CDLL) -> None:
    retained.test_cooperative_independent_decimal_fd_translation_and_permutation(
        candidate_routes(helper)
    )
