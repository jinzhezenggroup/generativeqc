"""Host-only MGGA tau projection/crossover checks, no GPU required."""

import numpy as np
import pytest

from benchmarks.mgga_occupied_crossover import (
    active_ids,
    hybrid_features,
    projection_work,
    water_atoms,
)


@pytest.mark.parametrize("n,rank", [(0, 0), (7, 0), (8, 3), (12, 5)])
@pytest.mark.parametrize("indexed", (False, True))
def test_hybrid_matches_independent_four_density_jet_bilinears(n, rank, indexed):
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
    for key in expected:
        np.testing.assert_allclose(observed[key], expected[key], atol=1e-12, rtol=1e-12)


def test_stale_factor_and_invalid_map_rejected():
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


def test_crossovers_use_local_ao_count_not_global_ao_count():
    dense = projection_work(768, 160, 64)
    local = projection_work(48, 160, 64)
    assert dense["one_density_three_occupied"] < dense["four_density_jets"]
    assert local["one_density_three_occupied"] > local["four_density_jets"]
    assert projection_work(160, 160, 10)["one_density_three_occupied"] == projection_work(160, 160, 10)["four_density_jets"]
    for wrong in ((-1, 1, 2), (3, True, 5), (1, 1, -2)):
        with pytest.raises(ValueError, match="counts"):
            projection_work(*wrong)


def test_exact_sampled_jet_cutoff_and_replicated_atom_shapes():
    ao = np.zeros((4, 5, 7))
    ao[1, 2, 3] = -2e-16
    ao[3, 3, 5] = 2e-16
    ao[0, 4, 6] = 1e-18
    np.testing.assert_array_equal(active_ids(ao), [3, 5])
    assert len(water_atoms(32)) == 96
    with pytest.raises(ValueError, match="replicas"):
        water_atoms(0)
