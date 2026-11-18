"""Opt-in scale-invariant DIIS for independent cold SCF references only.

PySCF's augmented DIIS solve uses an absolute eigenvalue cutoff. At very small
physical errors, its Gram block can be discarded relative to the affine
constraint. A common positive scaling of the entire error Gram block leaves
the exact extrapolation coefficients unchanged, while resolving that loss of
scale. This is not a production solver or an electronic-convergence criterion;
the independent SCF must still certify its actual physical orbital gradient.
"""

from __future__ import annotations

import numpy as np


def normalized_error_coefficients(error_gram: object) -> np.ndarray:
    """Solve the real, bounded-history affine DIIS problem at relative scale.

    All history errors share one scale; individual vectors are never rescaled.
    Rank-deficient histories use a relative pseudoinverse, with explicit finite,
    affine-residual and coefficient bounds. Exactly zero errors return a bounded
    affine average, which does not itself establish electronic convergence.
    """
    if np.iscomplexobj(error_gram):
        raise ValueError("independent DIIS requires a real error Gram matrix")
    gram = np.asarray(error_gram, dtype=float)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not 1 <= len(gram) <= 64:
        raise ValueError("independent DIIS requires 1--64 square history entries")
    if not np.all(np.isfinite(gram)):
        raise ValueError("independent DIIS Gram entries must be finite")
    scale = float(np.max(abs(gram)))
    if scale == 0:
        return np.full(len(gram), 1.0 / len(gram))
    normalized = gram / scale
    if np.max(abs(normalized - normalized.T)) > 1e-12:
        raise ValueError("independent DIIS error Gram must be symmetric")
    if np.linalg.eigvalsh(normalized).min() < -1e-12 * len(gram):
        raise ValueError("independent DIIS error Gram must be positive semidefinite")
    augmented = np.ones((len(gram) + 1, len(gram) + 1))
    augmented[0, 0] = 0
    augmented[1:, 1:] = normalized
    target = np.zeros(len(gram) + 1)
    target[0] = 1
    eigenvalues, eigenvectors = np.linalg.eigh(augmented)
    cutoff = 64 * np.finfo(float).eps * max(1.0, float(np.max(abs(eigenvalues))))
    retained = abs(eigenvalues) > cutoff
    solution = eigenvectors[:, retained] @ (
        (eigenvectors[:, retained].T @ target) / eigenvalues[retained]
    )
    coefficients = solution[1:]
    if not np.all(np.isfinite(coefficients)) or np.max(abs(coefficients)) > 1e6:
        raise ArithmeticError("independent DIIS extrapolation is unbounded")
    if np.max(abs(augmented @ solution - target)) > 1e-8:
        raise ArithmeticError("independent DIIS affine system did not close")
    if abs(float(coefficients.sum()) - 1) > 1e-8:
        raise ArithmeticError("independent DIIS lost its affine constraint")
    return coefficients


def normalized_diis_factory(solver: object) -> object:
    """Attach a PySCF-compatible extrapolator without modifying upstream code.

    Only the small DIIS coefficient solve is replaced. Physical Focks, error
    vectors, occupation selection, energy evaluation and all actual residual
    certification stay in the independent solver. Default references do not
    select this factory implicitly.
    """
    from pyscf.scf.diis import CDIIS

    class NormalizedErrorDIIS(CDIIS):
        """Use homogeneous Gram normalization, retaining bounded diagnostics."""

        def __init__(self, independent_solver: object) -> None:
            super().__init__(independent_solver)
            self.normalization_records = []

        def extrapolate(self, nd: int | None = None) -> np.ndarray:
            count = self.get_num_vec() if nd is None else nd
            if count == 0:
                raise RuntimeError("independent DIIS has no history")
            gram = self._H[1 : count + 1, 1 : count + 1]
            coefficients = normalized_error_coefficients(gram)
            output = None
            for index, coefficient in enumerate(coefficients):
                vector = self.get_vec(index)
                if output is None:
                    output = np.zeros_like(vector)
                output += coefficient * vector
            self.normalization_records.append(
                {
                    "history": count,
                    "unscaled_gram_maximum": float(np.max(abs(gram))),
                    "maximum_coefficient": float(np.max(abs(coefficients))),
                    "affine_residual": abs(float(coefficients.sum()) - 1),
                }
            )
            return output

    return NormalizedErrorDIIS(solver)
