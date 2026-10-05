"""Independent dense oracles for the bounded DF orbital preconditioner."""

import numpy as np
import pytest
from generativeqc_compiler.method.rhf_orbital_preconditioner import (
    build_rhf_df_preconditioner,
)
from generativeqc_compiler.method.rhf_orbital_response import build_rhf_frame_response
from generativeqc_compiler.tensor import execute


@pytest.mark.parametrize("o,v,q", [(1, 2, 3), (2, 3, 5), (3, 2, 1)])
def test_low_rank_matrix_matches_physical_diagonal_and_coulomb(
    o: int, v: int, q: int
) -> None:
    rng = np.random.default_rng(1901 + o + v)
    n = o + v
    c = np.linalg.qr(rng.normal(size=(n, n)))[0]
    factors = rng.normal(scale=0.03, size=(q, n, n))
    factors = (factors + factors.transpose(0, 2, 1)) / 2
    eps = np.r_[np.linspace(-1.3, -0.7, o), np.linspace(0.3, 1.1, v)]
    ao_factors = np.einsum("up,Qpq,vq->Quv", c, factors, c)
    g = np.einsum("Quv,Qwx->uvwx", ao_factors, ao_factors)
    gmo = np.einsum("Qpq,Qrs->pqrs", factors, factors)
    fock = c @ np.diag(eps) @ c.T
    maps = build_rhf_frame_response(o, v)
    hessian = np.empty((o * v, o * v))
    for column in range(o * v):
        x = np.zeros((o, v))
        x.flat[column] = 1
        rotation = np.zeros((n, n))
        rotation[:o, o:] = x
        rotation[o:, :o] = -x.T
        feeds = {
            "coefficients": c,
            "rotation": np.eye(n),
            "d_rotation": rotation,
            "fock_ao": fock,
        }
        density = execute(maps.density_direction, feeds).outputs["density_direction"]
        dg = np.einsum("uvwx,wx->uv", g, density) - 0.5 * np.einsum(
            "uwvx,wx->uv", g, density
        )
        hessian[:, column] = (
            execute(maps.orbital_action, {**feeds, "d_fock_ao": dg})
            .outputs["orbital_action"]
            .ravel()
        )
    expected = np.empty_like(hessian)
    for i in range(o):
        for a in range(v):
            for j in range(o):
                for b in range(v):
                    expected[i * v + a, j * v + b] = (
                        ((eps[o + a] - eps[i]) if (i, a) == (j, b) else 0.0)
                        + 4 * gmo[i, o + a, j, o + b]
                        - gmo[i, j, o + a, o + b]
                        - gmo[i, o + b, j, o + a]
                    )
    np.testing.assert_allclose(hessian, expected, atol=2e-14, rtol=2e-14)
    result = execute(
        build_rhf_df_preconditioner(o, v, q),
        {
            "eps_o": eps[:o],
            "eps_v": eps[o:],
            "boo": factors[:, :o, :o],
            "bov": factors[:, :o, o:],
            "bvv": factors[:, o:, o:],
        },
    ).outputs
    u = result["low_rank"].reshape(q, o * v)
    p = np.diag(result["diagonal"].ravel()) + u.T @ u
    np.testing.assert_allclose(np.diag(p), np.diag(hessian), atol=2e-14, rtol=2e-14)
    coulomb = 4 * gmo[:o, o:, :o, o:].reshape(o * v, o * v)
    off_diagonal = ~np.eye(o * v, dtype=bool)
    np.testing.assert_allclose(
        p[off_diagonal], coulomb[off_diagonal], atol=2e-15, rtol=2e-15
    )
    # Positivity is not manufactured in the equation. Runtime admission must
    # reject a bad diagonal instead of silently replacing it by the gap.
    huge = np.full_like(factors, 100.0)
    bad = execute(
        build_rhf_df_preconditioner(o, v, q),
        {
            "eps_o": eps[:o],
            "eps_v": eps[o:],
            "boo": huge[:, :o, :o],
            "bov": huge[:, :o, o:],
            "bvv": huge[:, o:, o:],
        },
    ).outputs["diagonal"]
    assert np.any(bad < 0)
