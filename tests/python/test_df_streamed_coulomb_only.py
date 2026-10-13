"""J-only DF response is linear in streamed panels and independent of exact K.

The analytic identity is checked with nonsymmetric densities and a full-rank
metric. The source-level check prevents accidental reintroduction of the
quadratic P/Q traversal into the semilocal CUDA fallback; a separate native
test exercises the real GPU producer when a CUDA device is available.
"""

from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("tile", [1, 2, 3, 5])
def test_full_rank_j_only_panel_adjoint(tile: int) -> None:
    rng = np.random.default_rng(2411)
    n, a = 3, 5
    pairs = n * n
    raw = rng.normal(size=(pairs, a))
    densities = rng.normal(size=(2, pairs))
    coefficients = np.array([1.0, -0.31])
    u = rng.normal(size=(a, a))
    metric = u @ u.T + 2 * np.eye(a)
    inverse = np.linalg.inv(metric)

    # The native streamed fitted reader emits A M^{-1}, not A M^{-1/2}.
    fitted = raw @ inverse
    potentials = np.empty((len(coefficients), a))
    weights = np.empty((len(coefficients), pairs, a))
    for begin in range(0, a, tile):
        count = min(tile, a - begin)
        potentials[:, begin : begin + count] = (
            densities @ fitted[:, begin : begin + count]
        )
        weights[:, :, begin : begin + count] = (
            coefficients[:, None, None]
            * densities[:, :, None]
            * potentials[:, None, begin : begin + count]
        )
    bar_raw = weights.sum(axis=0)
    bar_metric = -0.5 * sum(
        coefficient * np.outer(potential, potential)
        for coefficient, potential in zip(coefficients, potentials)
    )

    def energy(values: np.ndarray, m: np.ndarray) -> float:
        charge = densities @ values
        potential = np.linalg.solve(m, charge.T).T
        return 0.5 * float(np.sum(coefficients * np.sum(charge * potential, axis=1)))

    eps = 1e-6
    for i in range(pairs):
        for p in range(a):
            delta = np.zeros_like(raw)
            delta[i, p] = eps
            numeric = (
                energy(raw + delta, metric) - energy(raw - delta, metric)
            ) / (2 * eps)
            assert numeric == pytest.approx(bar_raw[i, p], abs=2e-8)
    for p in range(a):
        for q in range(p + 1):
            delta = np.zeros_like(metric)
            delta[p, q] = eps
            if p != q:
                delta[q, p] = eps
            numeric = (
                energy(raw, metric + delta) - energy(raw, metric - delta)
            ) / (2 * eps)
            expected = bar_metric[p, q] * (2 if p != q else 1)
            assert numeric == pytest.approx(expected, abs=2e-8)


def test_semilocal_ks_selects_packed_before_streaming() -> None:
    source = (ROOT / "src/scf/fock_prepared.cpp").read_text()
    begin = source.index("const bool coulomb_only_packing =")
    end = source.index("const bool needs_occupied_capacity", begin)
    admission = source[begin:end]
    assert "FockSpin::Restricted" in admission
    assert "!strategy.spec.exchange.present" in admission
    assert "reserved_rank != 0 || coulomb_only_packing" in admission
    assert "plan_requested_density_fitting_tiles(" in admission


def test_streamed_j_only_does_not_revisit_auxiliary_panel_pairs() -> None:
    source = (ROOT / "src/scf/cuda/df_response_weights.cu").read_text()
    begin = source.index("static cudaError_t contract_streamed_coulomb_response(")
    end = source.index("static cudaError_t contract_full_rank_panels(", begin)
    specialized = source[begin:end]
    assert specialized.count("read_fitted(begin, count, fitted_panel);") == 1
    assert "response_streamed_coulomb_fitted_panels" in specialized
    assert "coulomb_metric_kernel" in specialized
    assert "exchange_weights_kernel" not in specialized
    dispatch = source.index("return contract_streamed_coulomb_response(")
    assert dispatch < source.index(
        "return contract_full_rank_panels(", dispatch
    )
