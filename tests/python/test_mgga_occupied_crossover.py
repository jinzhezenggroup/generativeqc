"""Host-only MGGA tau projection/crossover checks, no GPU required."""

from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.mgga_occupied_crossover import (
    active_ids,
    benchmark,
    hybrid_features,
    projection_work,
    water_atoms,
)


@pytest.mark.parametrize("n,rank", [(0, 0), (7, 0), (8, 3), (12, 5)])
@pytest.mark.parametrize("indexed", (False, True))
def test_hybrid_matches_independent_four_density_jet_bilinears(
    n: int, rank: int, indexed: bool
) -> None:
    rng = np.random.default_rng(20261015 + n + rank)
    factor = rng.normal(size=(n, rank))
    density = factor @ factor.T
    jets = rng.normal(size=(4, 5, n))
    ids = np.arange(0, n, 2) if indexed else None
    use = jets if ids is None else jets[:, :, ids]
    d = density if ids is None else density[np.ix_(ids, ids)]
    observed = hybrid_features(use, density, factor, ids)
    expected = {
        "rho": np.einsum("pi,ij,pj->p", use[0], d, use[0]),
        "gradient": 2 * np.einsum("kpi,ij,pj->pk", use[1:4], d, use[0]),
        "tau": 0.5 * np.einsum("kpi,ij,kpj->p", use[1:4], d, use[1:4]),
    }
    for key, value in expected.items():
        np.testing.assert_allclose(observed[key], value, atol=1e-12, rtol=1e-12)


def test_stale_factor_and_invalid_map_rejected() -> None:
    rng = np.random.default_rng(4)
    factor = rng.normal(size=(6, 3))
    d = factor @ factor.T
    ao = rng.normal(size=(4, 4, 6))
    with pytest.raises(ValueError, match="witness"):
        hybrid_features(ao, d * 0.99, factor)
    with pytest.raises(ValueError, match="sorted"):
        hybrid_features(ao[:, :, [2, 2]], d, factor, np.array([2, 2]))
    with pytest.raises(ValueError, match="disagree"):
        hybrid_features(ao, d, factor, np.array([0, 1]))
    with pytest.raises(ValueError, match="nonfinite"):
        hybrid_features(np.full_like(ao, np.nan), d, factor)


def test_crossovers_use_local_ao_count_not_global_ao_count() -> None:
    dense = projection_work(768, 160, 64)
    local = projection_work(48, 160, 64)
    assert dense["one_density_three_occupied"] < dense["four_density_jets"]
    assert local["one_density_three_occupied"] > local["four_density_jets"]
    assert (
        projection_work(160, 160, 10)["one_density_three_occupied"]
        == projection_work(160, 160, 10)["four_density_jets"]
    )
    for wrong in ((-1, 1, 2), (3, True, 5), (1, 1, -2)):
        with pytest.raises(ValueError, match="counts"):
            projection_work(*wrong)


def test_exact_sampled_jet_cutoff_and_replicated_atom_shapes() -> None:
    ao = np.zeros((4, 5, 7))
    ao[1, 2, 3] = -2e-16
    ao[3, 3, 5] = 2e-16
    ao[0, 4, 6] = 1e-18
    np.testing.assert_array_equal(active_ids(ao), [3, 5])
    assert len(water_atoms(32)) == 96
    with pytest.raises(ValueError, match="replicas"):
        water_atoms(0)


@pytest.mark.parametrize(
    "active,rank,points",
    [(0, 0, 1), (8, 3, 5), (48, 160, 64), (160, 160, 10), (768, 160, 64)],
)
def test_hybrid_projection_is_never_cheapest_by_product_count(
    active: int, rank: int, points: int
) -> None:
    work = projection_work(active, rank, points)
    dense, hybrid, orbital = (
        work["four_density_jets"],
        work["one_density_three_occupied"],
        work["four_occupied_jets"],
    )
    assert 4 * hybrid == dense + 3 * orbital
    assert min(dense, orbital) <= hybrid <= max(dense, orbital)


def test_benchmark_uses_current_density_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise both benchmark routes with real source validation and host features."""
    from generativeqc_compiler import dft
    from generativeqc_compiler.common import provenance
    from generativeqc_compiler.dft import DensitySource
    from generativeqc_compiler.dft import cuda as cuda_module

    def evaluate_ao(points: np.ndarray, *, order: int) -> np.ndarray:
        assert order == 1
        jets = np.arange(24, dtype=np.float64).reshape(4, 1, 6)
        jets = jets + points[None, :, :1]
        jets[:, :, -1] = 0.0
        return jets

    basis = SimpleNamespace(nao=6, identity="0" * 64, evaluate=evaluate_ao)
    artifact = SimpleNamespace(metadata={"binary_sha256": "a" * 64})
    monkeypatch.setattr(provenance, "find_nvcc", lambda: Path("/fake/nvcc"))
    monkeypatch.setattr(dft, "NativeAO", lambda *args, **kwargs: nullcontext(basis))
    monkeypatch.setattr(cuda_module, "compile_cuda", lambda *args, **kwargs: artifact)

    def grid_factory(*args: object, **kwargs: object) -> nullcontext:
        grid = SimpleNamespace(source_kind="density_matrix", source_statistics={})
        sources: list[DensitySource] = []

        def set_source(source: DensitySource, *, stamp: object, route: str) -> None:
            assert stamp == source.stamp
            assert source.source_kind == "orbitals"
            sources[:] = [source]
            grid.source_kind = route
            grid.source_statistics = {"source_identity": source.stamp.identity}

        def evaluate(
            points: np.ndarray, *, ao_ids: np.ndarray | None, stamp: object
        ) -> dict[str, np.ndarray]:
            jets = evaluate_ao(points, order=1)
            if ao_ids is not None:
                jets = jets[:, :, ao_ids]
            return sources[0].features(
                jets,
                stamp=stamp,
                route=grid.source_kind,
                ao_ids=ao_ids,
                ingredients=("rho", "gradient", "tau"),
            )

        grid.set_source = set_source
        grid.evaluate = evaluate
        grid.metrics = lambda: dict.fromkeys(
            ("runtime_version", "driver_version", "cublas_version"), 0
        )
        return nullcontext(grid)

    monkeypatch.setattr(cuda_module, "CudaGrid", grid_factory)
    result = benchmark(
        Namespace(
            arch="sm_120",
            cache="unused-mocked-cache",
            replicas=[1],
            points_per_atom=2,
            tile_points=4,
            orbital_tile=3,
            warmups=1,
            repeats=2,
        )
    )
    assert result["artifact_sha256"] == "a" * 64
    dense, indexed = result["cases"]
    assert dense["active_ao_per_tile"] == [6, 6]
    assert indexed["active_ao_per_tile"] == [5, 5]
    for case in (dense, indexed):
        assert case["occupied_per_spin"] == 5
        assert set(case["source_uploads_excluded_from_timing"]) == {
            "density_matrix",
            "orbitals",
        }
        assert all(len(samples) == 2 for samples in case["timings_seconds"].values())
        assert max(case["max_abs_hybrid_reference_error"].values()) < 1e-10
