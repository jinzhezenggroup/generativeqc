"""Issue #1791 diagnostics, restricted to the explicit six-AO OH fixture.

This is a validation tool, never a production density transformation or a
general determinant-equivalence policy. Raw gates remain independent.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

AO_LABELS = ("O:1s", "O:2s", "O:2px", "O:2py", "O:2pz", "H:1s")
RAW_TOL = 1e-7
GATES = {
    "geometry": 1e-12,
    "overlap_invariance": 1e-12,
    "hcore_invariance": 1e-12,
    "eri_invariance": 1e-12,
    "fock_covariance": 1e-11,
    "density_symmetry": 1e-11,
    "electron_count": 1e-9,
    "idempotency": 1e-9,
    "physical_residual": 1e-8,
    "energy": 1e-9,
    "density": RAW_TOL,
    "metric_projector": RAW_TOL,
}


def _array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError("expected finite real array")
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"expected finite real array of shape {shape}")
    return a


def _max(a: np.ndarray) -> float:
    return float(np.max(np.abs(a)))


def raw_comparison(density: Any, reference: Any, overlap: Any) -> dict:
    """Strict spin-resolved raw D and S^(1/2) D S^(1/2) comparisons."""
    if np.iscomplexobj(overlap):
        raise ValueError("expected finite real overlap")
    s = np.asarray(overlap, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError("overlap must be square")
    n = len(s)
    s = _array(s, (n, n))
    if _max(s - s.T) > 1e-12:
        raise ValueError("overlap must be symmetric")
    w, v = np.linalg.eigh(s)
    if np.min(w) <= 0:
        raise ValueError("overlap must be positive definite")
    sqrt_s = (v * np.sqrt(w)) @ v.T
    delta = _array(density, (2, n, n)) - _array(reference, (2, n, n))
    errors = {"density": _max(delta), "metric_projector": _max(sqrt_s @ delta @ sqrt_s)}
    return {
        "errors": errors,
        "tolerance": RAW_TOL,
        "status": "PASS" if max(errors.values()) <= RAW_TOL else "FAIL",
    }


def axial_rotation(angle: float) -> tuple[np.ndarray, np.ndarray]:
    """The *same spatial* SO(2) rotation acts on both spin blocks, p=(x,y,z)."""
    if not math.isfinite(angle):
        raise ValueError("angle must be finite")
    c, s = math.cos(angle), math.sin(angle)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    u = np.eye(6)
    u[2:5, 2:5] = r
    return r, u


def physical_state(
    density: np.ndarray, hcore: np.ndarray, eri: np.ndarray, nuclear_energy: float
) -> tuple[np.ndarray, float]:
    """Independent full-tensor UHF contraction; no SCF solve or DIIS Fock."""
    total = density.sum(axis=0)
    j = np.einsum("ijkl,kl->ij", eri, total)
    k = np.einsum("ikjl,skl->sij", eri, density)
    fock = hcore[None] + j[None] - k
    energy = nuclear_energy + np.einsum("sij,ij->", density, hcore)
    energy += 0.5 * np.einsum("sij,sij->", density, fock - hcore)
    return fock, float(energy)


def fit_axial_angle(density: Any, reference: Any) -> float:
    """Fit only the pi-plane orientation, never arbitrary AO/occupied rotations.

    Use the most anisotropic spin's 2x2 transverse density. Two signs of its
    principal axis give candidates separated by pi; assess both on full D.
    Certification, rather than fitting, decides physical equivalence.
    """
    d, ref = _array(density, (2, 6, 6)), _array(reference, (2, 6, 6))
    gaps = [np.ptp(np.linalg.eigvalsh(x[2:4, 2:4])) for x in d]
    spin = int(np.argmax(gaps))
    if gaps[spin] <= RAW_TOL:
        return 0.0
    a = np.linalg.eigh(d[spin, 2:4, 2:4])[1][:, -1]
    b = np.linalg.eigh(ref[spin, 2:4, 2:4])[1][:, -1]
    theta = math.atan2(b[1], b[0]) - math.atan2(a[1], a[0])
    candidates = (theta, theta + math.pi)
    return min(
        candidates,
        key=lambda t: _max(axial_rotation(t)[1] @ d @ axial_rotation(t)[1].T - ref),
    )


def certify_oh(
    *,
    coordinates: Any,
    ao_labels: Any,
    overlap: Any,
    hcore: Any,
    eri: Any,
    density: Any,
    reference: Any,
    endpoint_energy: float,
    reference_energy: float,
    angle: float,
) -> dict:
    """Check an explicit axial transformation; retain raw FAIL even if certified.

    Caller must bind the supplied independent tensors to the exact bundled
    STO-3G primitives and primary endpoint (the companion runner does so).
    Unsupported geometries/AO orders fail closed. No matrix is changed in place.
    """
    xyz = _array(coordinates, (2, 3))
    if tuple(ao_labels) != AO_LABELS or not (
        np.array_equal(xyz[0], [0, 0, 0])
        and np.array_equal(xyz[1, :2], [0, 0])
        and xyz[1, 2] in (1.834, 1.85234)
    ):
        raise ValueError(
            "certificate only supports frozen OH/STO-3G original/moved AO frame"
        )
    s, h = _array(overlap, (6, 6)), _array(hcore, (6, 6))
    g = _array(eri, (6, 6, 6, 6))
    if (
        max(
            _max(h - h.T),
            _max(g - g.transpose(1, 0, 2, 3)),
            _max(g - g.transpose(0, 1, 3, 2)),
            _max(g - g.transpose(2, 3, 0, 1)),
        )
        > 1e-12
    ):
        raise ValueError("Hamiltonian must have real chemist integral symmetry")
    d, ref = _array(density, (2, 6, 6)), _array(reference, (2, 6, 6))
    if not all(math.isfinite(x) for x in (endpoint_energy, reference_energy)):
        raise ValueError("energies must be finite")
    raw = raw_comparison(d, ref, s)
    r, u = axial_rotation(angle)
    rotated = u @ d @ u.T
    ne = 8.0 / xyz[1, 2]
    f, e = physical_state(d, h, g, ne)
    fr, er = physical_state(rotated, h, g, ne)
    fref, eref = physical_state(ref, h, g, ne)
    rotated_raw = raw_comparison(rotated, ref, s)
    errors = {
        "geometry": _max(xyz @ r.T - xyz),
        "overlap_invariance": _max(u.T @ s @ u - s),
        "hcore_invariance": _max(u.T @ h @ u - h),
        "eri_invariance": _max(
            np.einsum("ai,bj,ck,dl,abcd->ijkl", u, u, u, u, g, optimize=True) - g
        ),
        "fock_covariance": _max(fr - u @ f @ u.T),
        "density_symmetry": max(
            _max(x - x.swapaxes(-1, -2)) for x in (d, ref, rotated)
        ),
        "electron_count": max(
            _max(np.einsum("sij,ji->s", x, s) - [5, 4]) for x in (d, ref, rotated)
        ),
        "idempotency": max(_max(x @ s @ x - x) for x in (d, ref, rotated)),
        "physical_residual": max(
            _max(ff @ x @ s - s @ x @ ff)
            for ff, x in ((f, d), (fr, rotated), (fref, ref))
        ),
        "energy": max(
            abs(e - endpoint_energy),
            abs(er - endpoint_energy),
            abs(eref - reference_energy),
            abs(e - reference_energy),
        ),
        **rotated_raw["errors"],
    }
    failed = [k for k, v in errors.items() if not math.isfinite(v) or v > GATES[k]]
    return {
        "raw": raw,
        "symmetry": {
            "status": "FAIL" if failed else "PASS",
            "angle_radians": float(angle),
            "spatial_rotation": r.tolist(),
            "ao_rotation": u.tolist(),
            "errors": errors,
            "gates": dict(GATES),
            "failed_gates": failed,
        },
        "production_density_modified": False,
        "canonical_determinant_policy": "unresolved; issue #1791 remains open",
    }
