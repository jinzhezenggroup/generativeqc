"""Experimental host multisecant inverse for CCSD Lambda qualification.

This module is tooling, never a default production preconditioner. All vectors
must already use the same sqrt-weighted pair ordering as the Lambda operator.
It neither evaluates CC equations nor constructs a full Jacobian. Its inverse
transpose is a right preconditioner; acceptance still needs the true residual.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise

import numpy as np


def _fp64(value: np.ndarray, ndim: int) -> np.ndarray:
    """Reject silent upcasts of FP32 histories and nonfinite input."""
    array = np.asarray(value)
    if array.dtype != np.float64 or array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(
            "multisecant inputs must be finite FP64 arrays of the required rank"
        )
    return array


def recover_jacobi_input(
    image: np.ndarray, residual: np.ndarray, denominators: np.ndarray, damping: float
) -> np.ndarray:
    """Recover t from G(t)=t+(1-d) R(t)/D without evaluating R again.

    D is the CC iteration denominator, NOT the physical Jacobian diagonal -D.
    Diagonal scaling commutes with sqrt weights, but all three arrays must use
    identical ordering. This is invalid after clipping, projection beyond
    rounding, changing denominators, or a mixed-precision residual evaluation.
    """
    image, residual, denominators = (
        _fp64(value, 1) for value in (image, residual, denominators)
    )
    if image.shape != residual.shape or image.shape != denominators.shape:
        raise ValueError("Jacobi history coordinate shape mismatch")
    if not np.isfinite(damping) or not 0 <= damping < 1:
        raise ValueError("Jacobi damping must satisfy 0 <= damping < 1")
    if np.any(np.abs(denominators) <= 1e-14):
        raise ValueError("unsafe Jacobi denominator")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        return image - (1 - damping) * residual / denominators


class SecantHistory:
    """Bounded, detached same-state samples from one strict precision epoch.

    The caller provides an epoch identity covering reference, coordinate ordering
    and numerical transformations. Denominator/damping changes are also checked.
    An epoch change or any FP32 observation discards old samples. ``append_jacobi`` owns copies
    before the producer reuses its ring buffers. No device pointers are retained.
    At most ``max_rank + 1`` states are live; rank six needs seven states.
    """

    def __init__(self, dimension: int, max_rank: int = 6, *, max_bytes: int) -> None:
        if type(dimension) is not int or dimension < 1:
            raise ValueError("history dimension must be positive")
        if type(max_rank) is not int or not 0 <= max_rank <= 6:
            raise ValueError("experimental rank must be between zero and six")
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("history budget must be nonnegative")
        self.dimension, self.max_rank = dimension, max_rank
        self.capacity_bytes = 2 * (max_rank + 1) * dimension * 8 if max_rank else 0
        self.admitted = bool(max_rank and self.capacity_bytes <= max_bytes)
        self.epoch: str | None = None
        self._jacobi_signature: tuple[str, float, bytes] | None = None
        self._samples: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=max_rank + 1)

    @property
    def samples(self) -> int:
        return len(self._samples)

    @property
    def retained_bytes(self) -> int:
        return sum(t.nbytes + r.nbytes for t, r in self._samples)

    def append_pair(
        self, amplitude: np.ndarray, residual: np.ndarray, *, epoch: str, precision: str
    ) -> None:
        """Append a physical (t,R(t)) sample; skip and invalidate FP32 epochs."""
        if not isinstance(epoch, str) or not epoch or precision not in {"fp32", "fp64"}:
            raise ValueError("an explicit epoch identity and precision are required")
        if precision != "fp64" or epoch != self.epoch:
            self._samples.clear()
            self.epoch = epoch
            self._jacobi_signature = None
        if precision != "fp64" or not self.admitted:
            return
        amplitude, residual = _fp64(amplitude, 1), _fp64(residual, 1)
        if amplitude.shape != (self.dimension,) or residual.shape != amplitude.shape:
            raise ValueError("history coordinate shape mismatch")
        # Retire before copying, so insertion never briefly owns an extra row.
        if len(self._samples) == self._samples.maxlen:
            self._samples.popleft()
        self._samples.append((amplitude.copy(), residual.copy()))

    def append_jacobi(
        self,
        image: np.ndarray,
        residual: np.ndarray,
        denominators: np.ndarray,
        damping: float,
        *,
        epoch: str,
        precision: str,
    ) -> None:
        """Detach an Anderson row, restoring its incoming amplitude first."""
        if precision != "fp64" or not self.admitted:
            self.append_pair(image, residual, epoch=epoch, precision=precision)
            return
        incoming = recover_jacobi_input(image, residual, denominators, damping)
        signature = (
            epoch,
            damping,
            sha256(np.ascontiguousarray(denominators)).digest(),
        )
        if self._jacobi_signature is not None and signature != self._jacobi_signature:
            self._samples.clear()
        self.append_pair(incoming, residual, epoch=epoch, precision=precision)
        self._jacobi_signature = signature

    def differences(self, *, epoch: str, rank: int) -> tuple[np.ndarray, np.ndarray]:
        """Return chronological adjacent S,Y columns from the newest states.

        These detached arrays have separate storage from the history owner.
        A consumer must charge their bytes and construction workspace as well.
        """
        if epoch != self.epoch:
            raise ValueError("stale multisecant history epoch")
        if type(rank) is not int or not 0 <= rank <= self.max_rank:
            raise ValueError("requested rank exceeds history capacity")
        states = list(self._samples)[-(rank + 1) :]
        pairs = list(pairwise(states)) if rank else []
        if not pairs:
            return np.empty((self.dimension, 0)), np.empty((self.dimension, 0))
        steps = np.column_stack([after[0] - before[0] for before, after in pairs])
        changes = np.column_stack([after[1] - before[1] for before, after in pairs])
        return steps, changes


@dataclass(frozen=True)
class MultisecantInverse:
    """H = diag(h0) + correction @ basis.T, with orthonormal basis columns.

    ``apply`` exposes H.T for the Lambda solver's right-preconditioner protocol.
    ``forward`` is only needed for secant diagnostics. Retained storage is
    O(nr); apply uses two thin matrix/vector products and diagonal scaling.
    """

    inverse_diagonal: np.ndarray
    basis: np.ndarray
    correction: np.ndarray
    singular_values: tuple[float, ...]
    build_seconds: float
    workspace_bound_bytes: int
    reason: str

    @property
    def rank(self) -> int:
        return self.basis.shape[1]

    @property
    def retained_bytes(self) -> int:
        return self.inverse_diagonal.nbytes + self.basis.nbytes + self.correction.nbytes

    def apply(self, vector: np.ndarray) -> np.ndarray:
        """Apply H.T, never H, to a Lambda Krylov vector."""
        vector = _fp64(vector, 1)
        if vector.shape != self.inverse_diagonal.shape:
            raise ValueError("multisecant action coordinate shape mismatch")
        return self.inverse_diagonal * vector + self.basis @ (
            self.correction.T @ vector
        )

    def forward(self, vector: np.ndarray) -> np.ndarray:
        """Apply H to check HY=S without materializing H."""
        vector = _fp64(vector, 1)
        if vector.shape != self.inverse_diagonal.shape:
            raise ValueError("multisecant action coordinate shape mismatch")
        return self.inverse_diagonal * vector + self.correction @ (
            self.basis.T @ vector
        )


def fit_multisecant(
    inverse_diagonal: np.ndarray,
    steps: np.ndarray,
    residual_changes: np.ndarray,
    *,
    max_rank: int = 6,
    relative_cutoff: float = 1e-8,
    absolute_cutoff: float = 1e-14,
    max_bytes: int,
) -> MultisecantInverse:
    """Build a rank-filtered inverse using a thin SVD, never normal equations.

    Scale each (s,y) column together before SVD, so late small residuals do not
    lose rank solely to earlier columns' size. In full column rank this is
    exactly H0+(S-H0Y) pinv(Y). Truncation fits the retained singular subspace
    of the scaled history. The budget bounds construction's estimated numeric
    workspace (including copies and LAPACK work), not an allocator RSS peak.
    Caller-owned history/S/Y must be charged separately. Budget refusal retains
    only H0; malformed/nonfinite input raises instead
    of silently changing the scientific state. No GPU allocation is performed.
    """
    started = time.perf_counter()
    h0, s, y = _fp64(inverse_diagonal, 1), _fp64(steps, 2), _fp64(residual_changes, 2)
    if s.shape != y.shape or s.shape[0] != h0.size or h0.size == 0:
        raise ValueError("multisecant fit coordinate shape mismatch")
    if np.any(h0 == 0) or type(max_rank) is not int or not 0 <= max_rank <= 6:
        raise ValueError(
            "nonzero inverse diagonal and rank between zero and six required"
        )
    if not np.isfinite(relative_cutoff) or not 0 < relative_cutoff < 1:
        raise ValueError("relative rank cutoff must be between zero and one")
    if not np.isfinite(absolute_cutoff) or absolute_cutoff < 0:
        raise ValueError("absolute rank cutoff must be nonnegative")
    if type(max_bytes) is not int or max_bytes < h0.nbytes:
        raise ValueError("budget cannot retain even the diagonal inverse")
    n, m = y.shape
    required = 8 * (12 * n * m + 8 * n + 16 * m * m + 16 * m)
    basis = correction = np.empty((n, 0))
    singular = np.empty(0)
    reason = "empty_history"
    if max_rank and m and required <= max_bytes:
        # Scaling by a maximum before taking norms avoids squaring subnormals
        # or overflowing a finite residual. Tiny/zero columns carry no evidence.
        scales = np.max(np.abs(y), axis=0)
        keep = scales > absolute_cutoff
        if np.any(keep):
            ys = y[:, keep] / scales[keep]
            norms = np.linalg.norm(ys, axis=0)
            ys /= norms
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                ss = (s[:, keep] / scales[keep]) / norms
                left, singular, right = np.linalg.svd(ys, full_matrices=False)
                rank = min(
                    max_rank,
                    int(np.count_nonzero(singular > relative_cutoff * singular[0])),
                )
                basis = left[:, :rank].copy()
                correction = ((ss - h0[:, None] * ys) @ right[:rank].T) / singular[
                    :rank
                ]
                if not np.isfinite(correction).all():
                    raise FloatingPointError("nonfinite multisecant inverse correction")
            reason = "accepted" if rank else "rank_deficient"
        else:
            reason = "negligible_history"
    elif not max_rank:
        reason = "disabled"
    elif m:
        reason = "workspace_limit"
    h0 = h0.copy()
    for array in (h0, basis, correction):
        array.setflags(write=False)
    return MultisecantInverse(
        h0,
        basis,
        correction,
        tuple(map(float, singular)),
        time.perf_counter() - started,
        required if max_rank and m and required <= max_bytes else h0.nbytes,
        reason,
    )
