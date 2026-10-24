"""Independent PySCF analytic force with a residual-certified small CPHF oracle."""

from __future__ import annotations

import typing
from unittest.mock import patch

import numpy as np


def analytic_gradient(oracle: typing.Any) -> np.ndarray:
    """Keep PySCF gradients/J-K, solving its small physical response independently.

    PySCF 2.14's default Krylov termination can leave a large true residual in
    a degenerate OH manifold even when its requested tol is tightened. Use a
    bounded NumPy SVD solve of PySCF's own physical J/K action; no native result,
    generated TensorIR or production response buffer enters this oracle.
    """
    from pyscf.scf import ucphf

    def solve(
        fvind: typing.Any,
        energies: typing.Any,
        occupations: typing.Any,
        h1: typing.Any,
        s1: typing.Any = None,
        **kwargs: typing.Any,
    ) -> typing.Any:
        assert s1 is None and kwargs.get("level_shift", 0) == 0
        gaps = np.concatenate(
            [
                (e[occ == 0, None] - e[None, occ > 0]).ravel()
                for e, occ in zip(energies, occupations, strict=True)
            ]
        )
        rhs = -np.concatenate([np.asarray(x).ravel() for x in h1])
        assert 0 < rhs.size <= 128

        def action(vector: np.ndarray) -> np.ndarray:
            return np.asarray(fvind(vector.reshape(1, -1))).ravel()

        matrix = np.diag(gaps) + np.column_stack(
            [action(c) for c in np.eye(rhs.size).T]
        )
        assert np.isfinite(matrix).all() and np.isfinite(rhs).all()
        np.testing.assert_allclose(matrix, matrix.T, atol=2e-11, rtol=0)
        # Equivalent rotations can be true zero modes. Accept only a compatible
        # RHS and a fresh physical residual, rather than inverting that gauge.
        solution = np.linalg.lstsq(matrix, rhs, rcond=1e-12)[0]
        residual = gaps * solution + action(solution) - rhs
        assert np.max(np.abs(residual)) < 1e-11
        split = h1[0].size
        return (
            solution[:split].reshape(h1[0].shape),
            solution[split:].reshape(h1[1].shape),
        ), None

    with patch.object(ucphf, "solve", solve):
        result = np.asarray(oracle.nuc_grad_method().kernel())
    assert np.isfinite(result).all()
    return result
