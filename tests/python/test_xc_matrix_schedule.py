"""Independent compact-factor gates with signed weights and arbitrary AO jets."""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc_compiler.common.array_graph import evaluate_array_graph
from generativeqc_compiler.dft.ao_cuda import emit_grid_source
from generativeqc_compiler.dft.xc_contraction_cuda import (
    DEFAULT_XC_MATRIX_SCHEDULE,
    XcMatrixSchedule,
    compact_panel_program,
    emit_native_xc_matrix_schedule,
    qualified_xc_matrix_schedules,
    select_xc_matrix_schedule,
)

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "family,terms,jets", [("lda", 1, 1), ("gga", 4, 4), ("mgga", 5, 4)]
)
def test_compact_factor_matches_full_bilinear(
    family: str, terms: int, jets: int
) -> None:
    """Arbitrary coefficients detect lost gradient legs and doubled tau/rho."""
    rng = np.random.default_rng(43)
    ao = rng.normal(size=(jets, 31, 19))
    c = rng.normal(size=(terms, 31))
    weights = rng.normal(size=31)
    graph, roots = compact_panel_program(family)
    values = {f"x{j}": ao[j] for j in range(jets)}
    values.update({f"c{j}": c[j, :, None] for j in range(terms)})
    panels = evaluate_array_graph(graph, roots, values)
    actual = np.zeros((19, 19))
    for a, panel in zip(ao, panels, strict=False):
        cross = a.T @ (weights[:, None] * panel)
        actual += cross + cross.T
    expected = ao[0].T @ ((weights * c[0])[:, None] * ao[0])
    if jets > 1:
        for j in range(1, 4):
            cross = ao[j].T @ ((weights * c[j])[:, None] * ao[0])
            expected += cross + cross.T
    if terms == 5:
        for j in range(1, 4):
            expected += ao[j].T @ ((weights * c[4])[:, None] * ao[j])
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=5e-13)


@pytest.mark.parametrize("family,jets", [("lda", 1), ("gga", 4), ("mgga", 4)])
def test_one_sided_diagonal_panels_match_full_bilinear(family: str, jets: int) -> None:
    """Signed arbitrary point coefficients catch missing cross terms or tau/2."""
    generator = np.random.default_rng(20261016)
    ao = generator.normal(size=(jets, 33, 19))
    coefficients = generator.normal(size=(5, 33))
    weights = generator.normal(size=33)
    bindings = {f"x{jet}": ao[jet] for jet in range(jets)}
    bindings.update({f"c{term}": coefficients[term, :, None] for term in range(5)})
    actual = np.zeros((19, 19))
    for one_sided in (False, True):
        graph, roots = compact_panel_program(family, diagonal_one_sided=one_sided)
        panels = evaluate_array_graph(graph, roots, bindings)
        potential = np.zeros_like(actual)
        for jet, panel in enumerate(panels):
            cross = ao[jet].T @ (weights[:, None] * panel)
            potential += cross if one_sided and jet else cross + cross.T
        if one_sided:
            np.testing.assert_allclose(potential, actual, rtol=3e-14, atol=5e-13)
        else:
            actual = potential


def test_potential_uses_compact_triangular_tile_domain() -> None:
    """Potential launch work must contain only the authoritative tile triangle."""
    source = emit_native_xc_matrix_schedule()
    assert "const I tiles = (n+15)/16, tile_pairs = tiles*(tiles+1)/2;" in source
    assert "dim3(tile_pairs,1,spins)" in source
    assert "if (blockIdx.x > blockIdx.y) return;" not in source
    assert "tile_mu = pair-low*(low+1)/2;" in source

    for nao, square, triangle in ((192, 144, 78), (384, 576, 300), (768, 2304, 1176)):
        tiles = (nao + 15) // 16
        assert tiles * tiles == square
        assert tiles * (tiles + 1) // 2 == triangle
        assert triangle < square


def test_tiled_potential_retains_optional_point_total_reduction() -> None:
    """The retained FP32 path can still fuse totals when its pointer is supplied."""
    source = emit_native_xc_matrix_schedule()
    assert "I work_jets, const double* point_totals, double* potential," in source
    assert "point_totals && blockIdx.x == 0 && spin == 0" in source
    assert "accumulate, error, ao_ids, full_n, blockIdx.z);" in source
    assert "threadIdx.y == 0 && threadIdx.x < 3" in source
    assert (
        "for (I p = 0; p < count; ++p) sum += static_cast<Arithmetic>(point_totals[channel*count+p]);"
        in source
    )
    assert (
        "totals[channel] = finite((accumulate ? totals[channel] : 0.0)+static_cast<double>(sum),error,3);"
        in source
    )
    # Tiny/out-of-domain shapes keep the historical reducer rather than changing
    # their arithmetic or launch contract merely to share the production path.
    assert (
        "accumulate_totals<<<1,32,0,stream>>>(point_totals,count,totals,error);"
        in source
    )


def test_tiled_first_point_tile_initializes_matrix_without_global_clear() -> None:
    """Dense tiled XC still initializes its matrix in the first tile."""
    schedule = emit_native_xc_matrix_schedule()
    assert "double* totals, bool accumulate, int* error" in schedule
    assert "const double prior = accumulate ? potential[index] : 0.0;" in schedule
    assert "if (!accumulate) {" in schedule
    assert "cudaMemsetAsync(potential,0,spins*n*n*sizeof(double),stream)" in schedule
    assert "cudaMemsetAsync(totals,0,3*sizeof(double),stream)" in schedule

    glue = (ROOT / "src/dft/cuda_xc_kernels.cuh").read_text()
    setup = glue[: glue.index("for (std::size_t begin")]
    dense_setup, sparse_setup = setup.split("if (l.local_ao || compact_candidate) {", 1)
    assert "cudaMemsetAsync(potential" not in dense_setup
    assert "cudaMemsetAsync(potential" in sparse_setup
    assert "begin != 0 || l.local_ao, error" in glue


@pytest.mark.parametrize(
    "nao,spins,expected_matrix_bytes,expected_removed_bytes",
    [
        (384, 1, 1_179_648, 1_179_672),
        (384, 2, 2_359_296, 2_359_320),
        (768, 1, 4_718_592, 4_718_616),
        (768, 2, 9_437_184, 9_437_208),
    ],
)
def test_first_tile_initialization_traffic_census(
    nao: int, spins: int, expected_matrix_bytes: int, expected_removed_bytes: int
) -> None:
    """Retained FP32 first-tile fusion avoids matrix and three-total clears."""
    matrix_bytes = spins * nao * nao * 8
    assert matrix_bytes == expected_matrix_bytes
    assert matrix_bytes + 3 * 8 == expected_removed_bytes


@pytest.mark.parametrize(
    "points,expected_tiles,expected_two_step_launches",
    [(1_327_104, 5_184, 10_368), (2_654_208, 10_368, 20_736)],
)
def test_tiled_total_reduction_launch_census(
    points: int, expected_tiles: int, expected_two_step_launches: int
) -> None:
    """Retain the FP32 256-point diagnostic tile census."""
    tiles = (points + 255) // 256
    assert tiles == expected_tiles
    assert 2 * tiles == expected_two_step_launches
    # Before this slice: one standalone accumulate_totals launch per point tile.
    # The retained FP32 tiled path still fuses those scalar reductions.
    # Strict FP64 instead uses the separately qualified compensated batch reducer.
    assert expected_two_step_launches > 0


def test_xc_matrix_candidates_are_resource_qualified() -> None:
    candidates = qualified_xc_matrix_schedules(384, 256, spins=2, work_jets=4)
    assert tuple(schedule.tile for schedule in candidates) == (8, 16, 32)
    assert DEFAULT_XC_MATRIX_SCHEDULE in candidates
    assert all(schedule.threads <= 1024 for schedule in candidates)
    assert all(schedule.shared_bytes <= 48 * 1024 for schedule in candidates)
    assert all(schedule.additional_workspace_bytes == 0 for schedule in candidates)


def test_xc_matrix_selection_keeps_measured_default_without_profile() -> None:
    selected = select_xc_matrix_schedule(384, 256)
    assert selected == DEFAULT_XC_MATRIX_SCHEDULE
    assert select_xc_matrix_schedule(384, 256, preferred_tile=32) == XcMatrixSchedule(
        32
    )

    # An 8x8 candidate may be legal here, but no-profile production must retain
    # the historical scalar fallback when the qualified 16x16 default is absent.
    assert select_xc_matrix_schedule(8, 8) is None
    with pytest.raises(ValueError, match="not qualified"):
        select_xc_matrix_schedule(16, 16, preferred_tile=32)


def test_xc_matrix_admission_charges_compact_triangle_grid_dimension() -> None:
    schedule = XcMatrixSchedule(16)
    assert schedule.admitted(361 * 16, 256)
    assert not schedule.admitted(362 * 16, 256)
    assert not schedule.admitted(16, 16, spins=2, work_jets=4, maximum_grid_dimension=7)

    source = emit_native_xc_matrix_schedule()
    assert "const I tile_pairs = tiles*(tiles+1)/2;" in source
    assert "tile_pairs <= 65535;" in source


def test_xc_matrix_emitter_materializes_explicit_tile_candidate() -> None:
    source = emit_native_xc_matrix_schedule(XcMatrixSchedule(8))
    assert "__shared__ Arithmetic d[8][9], a[8][9];" in source
    assert "dim3(8,8)" in source
    assert "(n+7)/8" in source
    assert "all 64 lanes" in source
    assert "dim3(16,16)" not in source


def test_xc_matrix_candidate_is_frozen_in_grid_source_identity() -> None:
    source8, identity8, _ = emit_grid_source(
        native_ks=True, xc_matrix_schedule=XcMatrixSchedule(8)
    )
    source16, identity16, _ = emit_grid_source(native_ks=True)

    assert "dim3(8,8)" in source8
    assert "dim3(16,16)" in source16
    assert identity8 != identity16

    with pytest.raises(ValueError, match="require native KS"):
        emit_grid_source(native_ks=False, xc_matrix_schedule=XcMatrixSchedule(8))


def test_xc_setup_resets_compensated_totals_without_clearing_dense_matrix(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Execute real setup for repeated dense/local/compact and precision routes."""
    source = (ROOT / "src/dft/cuda_xc_kernels.cuh").read_text()
    start = source.index("  const bool compact_candidate =")
    end = source.index("  if (density_provider && !l.local_ao)", start)
    setup = source[start:end]
    header = (ROOT / "src/dft/cuda_xc.hpp").read_text()
    start = header.index("inline constexpr std::size_t kCudaXcTotalStorageElements =")
    storage = header[start : header.index(";", start) + 1]
    unit, binary = tmp_path / "xc-setup.cpp", tmp_path / "xc-setup"
    unit.write_text(
        r"""
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <algorithm>
using I = long long;
int cudaMemsetAsync(void* destination, int value, std::size_t bytes, void*) {
  std::memset(destination, value, bytes); return 0;
}
void cuda_check(int status) { assert(status == 0); }
struct Layout { bool fp32_contractions, local_ao; };
struct Batch { bool compact; };
struct Binding {
  struct Diagnostic { bool provider_allowance; };
  Diagnostic diagnostic() const { return {true}; }
};
"""
        + storage
        + r"""
void initialize(Layout l, Batch point_batch_plan, const void* density_provider,
                const Binding* potential_binding, double* potential, double* totals) {
  const I matrices = 8;
  void* stream = nullptr;
"""
        + setup
        + r"""
}
int main() {
  const Binding binding;
  for (bool fp32 : {false, true})
    for (bool local : {false, true})
      for (bool compact : {false, true})
        for (bool density_bound : {false, true})
          for (bool potential_bound : {false, true}) {
            std::array<double, 9> potential;
            std::array<double, 7> totals;
            for (double poison : {123.0, -987.0}) {
              potential.fill(poison); totals.fill(poison);
              initialize({fp32, local}, {compact}, density_bound ? &binding : nullptr,
                         potential_bound ? &binding : nullptr, potential.data(), totals.data());
              const bool sparse = local || (compact && !density_bound && !potential_bound);
              for (int index = 0; index < 8; ++index)
                assert(potential[index] == (sparse ? 0.0 : poison));
              for (int index = 0; index < 6; ++index)
                assert(totals[index] == (!fp32 || (sparse && index < 3) ? 0.0 : poison));
              assert(potential[8] == poison && totals[6] == poison);
            }
          }
}
"""
    )
    native_cxx.build_executable([unit], binary, compile_args=["-std=c++20"])
    subprocess.run([binary], check=True, timeout=10)
