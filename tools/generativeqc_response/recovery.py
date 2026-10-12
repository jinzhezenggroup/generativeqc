"""Bounded CPU UHF orbital-stability recovery using the shared response action.

The SCF callback is an existing, independently qualified UHF solver. This module
never implements another J/K equation, SCF solver or PySCF production fallback.
Only real, internal, fixed-geometry UHF rotations are currently supported.
"""

from __future__ import annotations

import math
import typing
from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import pairwise
from pathlib import Path

import numpy as np
from generativeqc.profiles import canonical_hash
from scipy.linalg import expm

from tools.generativeqc_posthf.reference import immutable

from .stability import (
    StabilityOptions,
    StabilityUnqualified,
    diagnose_uhf_stability,
)
from .uhf import UHFResponseOperator

if typing.TYPE_CHECKING:
    from .stability import UHFStabilityCertificate


class RecoveryUnqualified(StabilityUnqualified):
    """A trustworthy recovered energy/force/root is not available."""


def _assert_source_current(operator: UHFResponseOperator) -> None:
    """Check the live source owner as well as immutable scientific metadata."""
    try:
        operator.validate_current()
        source = getattr(operator.backend, "source", None)
        check_open = getattr(source, "_check_open", None)
        if check_open is not None:
            check_open()
    except Exception as error:
        raise RecoveryUnqualified("stale or closed recovery source") from error


def _assert_current(
    certificate: UHFStabilityCertificate, operator: UHFResponseOperator
) -> None:
    try:
        certificate.assert_current(operator)
    except Exception as error:
        raise RecoveryUnqualified("stale recovery certificate or source") from error
    _assert_source_current(operator)


def _provider_bytes(operator: UHFResponseOperator) -> int:
    value = getattr(operator.backend, "host_workspace_bytes", None)
    if type(value) is not int or value < 0:
        raise RecoveryUnqualified("provider must declare host workspace bytes")
    return value


@dataclass(frozen=True)
class RecoveryOptions:
    """Explicit, finite CPU restart work and root-equivalence tolerances."""

    angles: tuple[float, ...] = (0.15, 0.35, 0.65)
    max_nbf: int = 32
    budget_bytes: int = 32 << 20
    downhill_tolerance: float = 1e-9
    energy_tolerance: float = 2e-7
    root_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        angles = tuple(self.angles)
        if (
            not angles
            or len(angles) > 6
            or any(type(x) not in (float, int) or not math.isfinite(x) for x in angles)
            or any(not 0 < a < math.pi / 2 for a in angles)
            or any(a >= b for a, b in pairwise(angles))
        ):
            raise ValueError(
                "angles require 1-6 strictly increasing finite rotations (0, pi/2)"
            )
        object.__setattr__(self, "angles", angles)
        if type(self.max_nbf) is not int or self.max_nbf < 1:
            raise ValueError("max_nbf must be a positive integer")
        if type(self.budget_bytes) is not int or self.budget_bytes < 1:
            raise ValueError("budget_bytes must be a positive integer")
        for field in ("downhill_tolerance", "energy_tolerance", "root_tolerance"):
            value = getattr(self, field)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{field} must be finite and positive")


@dataclass(frozen=True, eq=False)
class UHFSpinDiagnostics:
    """Overlap-covariant spin expectation and per-AO Mulliken populations."""

    spin_squared: float
    spin_projection: float
    alpha_ao_population: np.ndarray
    beta_ao_population: np.ndarray
    spin_density_norm: float
    density_identity: str


def _densities(reference: object) -> np.ndarray:
    blocks = []
    for spin in ("alpha", "beta"):
        c = getattr(reference, f"coefficients_{spin}")
        occupied = c[:, : reference.nocc(spin)]
        blocks.append(occupied @ occupied.T)
    return immutable(np.stack(blocks))


def _metrics(reference: object, densities: np.ndarray) -> UHFSpinDiagnostics:
    overlap = reference.overlap
    na, nb = reference.nocc("alpha"), reference.nocc("beta")
    sz = (na - nb) / 2.0
    # Collinear determinant: <S^2> = Sz(Sz+1) + Nb - Tr(Da S Db S).
    squared = float(
        sz * (sz + 1) + nb - np.trace(densities[0] @ overlap @ densities[1] @ overlap)
    )
    squared = max(0.0, squared) if squared >= -1e-10 else squared
    populations = [np.diag(d @ overlap).copy() for d in densities]
    if (
        not math.isfinite(squared)
        or squared < 0
        or any(not np.isfinite(p).all() for p in populations)
    ):
        raise RecoveryUnqualified("nonfinite or invalid spin diagnostics")
    return UHFSpinDiagnostics(
        squared,
        sz,
        immutable(populations[0]),
        immutable(populations[1]),
        float(np.linalg.norm((densities[0] - densities[1]) @ overlap)),
        canonical_hash(
            {
                "reference": reference.identity,
                "densities": sha256(densities.tobytes()).hexdigest(),
            }
        ),
    )


def _electronic_energy(operator: UHFResponseOperator, densities: np.ndarray) -> float:
    """Reuse the exact source-owned Fock seam; do not build a second J/K algebra."""
    reference = operator.problem.reference
    a, b = operator._fock_response(densities[0], densities[1])
    a, b = np.asarray(a), np.asarray(b)
    if any(
        m.shape != reference.hcore.shape
        or np.iscomplexobj(m)
        or not np.isfinite(m).all()
        for m in (a, b)
    ):
        raise RecoveryUnqualified("invalid source Fock during trial energy")
    result = float(
        np.sum((densities[0] + densities[1]) * reference.hcore)
        + 0.5 * (np.sum(densities[0] * a) + np.sum(densities[1] * b))
    )
    if not math.isfinite(result):
        raise RecoveryUnqualified("nonfinite trial electronic energy")
    return result


def _rotate_density(
    operator: UHFResponseOperator, direction: np.ndarray, angle: float
) -> np.ndarray:
    reference = operator.problem.reference
    layout = operator.problem.layout
    blocks = []
    for spin in ("alpha", "beta"):
        generator = layout.generator_matrix(spin, direction)
        rotated = getattr(reference, f"coefficients_{spin}") @ expm(-angle * generator)
        occupied = rotated[:, : reference.nocc(spin)]
        density = occupied @ occupied.T
        blocks.append(0.5 * (density + density.T))
    value = np.stack(blocks)
    if not np.isfinite(value).all():
        raise RecoveryUnqualified("nonfinite rotated seed")
    return immutable(value)


def _same_root(
    left: np.ndarray,
    right: np.ndarray,
    overlap: np.ndarray,
    tolerance: float,
    *,
    spin_flip: bool,
) -> bool:
    def defect(pair: np.ndarray) -> float:
        return float(max(np.linalg.norm((left[i] - pair[i]) @ overlap) for i in (0, 1)))

    if defect(right) < tolerance:
        return True
    return bool(spin_flip and defect(right[::-1]) < tolerance)


@dataclass(frozen=True, eq=False)
class UHFRecoveryResult:
    """Evidence for one qualified unchanged or symmetry-broken CPU UHF root."""

    status: str
    initial: UHFStabilityCertificate
    recovered: UHFStabilityCertificate
    operator: UHFResponseOperator
    initial_operator: UHFResponseOperator
    energy_drop: float
    spin: UHFSpinDiagnostics
    attempted_seeds: int
    converged_roots: int
    equivalent_restarts: int
    distinct_roots: int
    identity: str

    def assert_current(self) -> None:
        _assert_current(self.initial, self.initial_operator)
        _assert_current(self.recovered, self.operator)
        if (
            self.status not in ("RECOVERED", "UNCHANGED")
            or self.recovered.status != "STABLE"
        ):
            raise RecoveryUnqualified("unqualified root publication")


def recover_uhf_stability(
    operator: UHFResponseOperator,
    solve: typing.Callable[[np.ndarray], UHFResponseOperator],
    *,
    options: RecoveryOptions | None = None,
    stability_options: StabilityOptions | None = None,
) -> UHFRecoveryResult:
    """Rotate a certified saddle and reconverge through a supplied native UHF SCF.

    The solve callback consumes an immutable [spin, AO, AO] density and must
    return a fresh, canonical, converged UHFResponseOperator for the same
    Hamiltonian. A caller can adapt an existing FockPlan.solve(initial_density=...)
    and its reference exporter. This API cannot publish a force: the caller must
    assert the returned certificates current before a downstream force operation.
    Every attempted seed has bounded work; failures publish no partial root.
    """
    options = options or RecoveryOptions()
    if not isinstance(options, RecoveryOptions) or not callable(solve):
        raise TypeError("expected RecoveryOptions and a native UHF SCF callback")
    if type(operator) is not UHFResponseOperator:
        raise RecoveryUnqualified("only real internal UHF recovery is qualified")
    r0 = operator.problem.reference
    n = r0.nbf
    stability_options = stability_options or StabilityOptions()
    if not isinstance(stability_options, StabilityOptions):
        raise TypeError("expected StabilityOptions")
    provider_bytes = _provider_bytes(operator)
    max_roots = 2 * len(options.angles)
    # Reserve all simultaneously retained reference/provider declarations,
    # spin densities, populations and certificate eigenvalues. The remaining
    # allowance belongs to the shared diagnostic, which owns admission for
    # its full Hessian/eigensolver/AO workspace. Callback-private SCF scratch
    # remains the callback's responsibility.
    retained_bytes = (max_roots + 1) * (r0.numeric_bytes + provider_bytes) + 8 * (
        36 * n * n
        + 8 * operator.dimension
        + max_roots * (2 * n * n + 2 * n + 2 * operator.dimension)
    )
    if n > options.max_nbf or retained_bytes >= options.budget_bytes:
        raise RecoveryUnqualified("recovery numeric budget rejected before execution")
    diagnostic_options = replace(
        stability_options,
        budget_bytes=min(
            stability_options.budget_bytes, options.budget_bytes - retained_bytes
        ),
    )

    def diagnose(candidate: UHFResponseOperator) -> UHFStabilityCertificate:
        _assert_source_current(candidate)
        if _provider_bytes(candidate) > provider_bytes:
            raise RecoveryUnqualified("SCF result exceeds reserved provider budget")
        try:
            certificate = diagnose_uhf_stability(candidate, options=diagnostic_options)
        except StabilityUnqualified as error:
            raise RecoveryUnqualified(str(error)) from error
        _assert_current(certificate, candidate)
        if _provider_bytes(candidate) > provider_bytes:
            raise RecoveryUnqualified("SCF result exceeds reserved provider budget")
        return certificate

    initial = diagnose(operator)
    if initial.status in ("EMPTY", "NEAR_SINGULAR"):
        raise RecoveryUnqualified(
            f"cannot publish unresolved {initial.status} reference"
        )
    original = _densities(r0)
    if initial.status == "STABLE":
        spin = _metrics(r0, original)
        return UHFRecoveryResult(
            "UNCHANGED",
            initial,
            initial,
            operator,
            operator,
            0.0,
            spin,
            0,
            1,
            0,
            1,
            canonical_hash({"certificate": initial.identity, "status": "UNCHANGED"}),
        )
    if initial.direction is None:
        raise RecoveryUnqualified("negative direction absent from unstable certificate")
    _assert_current(initial, operator)
    base_energy = _electronic_energy(operator, original)
    roots: list[
        tuple[
            UHFResponseOperator, UHFStabilityCertificate, np.ndarray, UHFSpinDiagnostics
        ]
    ] = []

    def check_current() -> None:
        _assert_current(initial, operator)
        # A failed callback can mutate a previously retained provider too.
        # Recheck every declaration before further source work/publication.
        for retained in (operator, *(item[0] for item in roots)):
            if _provider_bytes(retained) > provider_bytes:
                raise RecoveryUnqualified("retained source exceeds provider budget")

    attempted = 0
    equivalent = 0
    for angle in options.angles:
        for sign in (1.0, -1.0):
            check_current()
            proposal = _rotate_density(operator, initial.direction, sign * angle)
            trial_energy = _electronic_energy(operator, proposal)
            check_current()
            if trial_energy >= base_energy - options.downhill_tolerance:
                continue
            attempted += 1
            try:
                candidate = solve(proposal)
            except Exception:  # noqa: BLE001
                # The callback can raise backend-specific SCF errors.
                # Non-convergence of one seed does not disqualify another. A
                # mutated provider, however, invalidates every pending seed.
                check_current()
                continue
            check_current()
            if type(candidate) is not UHFResponseOperator:
                raise RecoveryUnqualified("SCF callback did not return a UHF operator")
            r1 = candidate.problem.reference
            if (
                any(
                    getattr(r1, attr) != getattr(r0, attr)
                    for attr in (
                        "algorithm",
                        "geometry_hash",
                        "basis_hash",
                        "hamiltonian_id",
                        "representation",
                        "precision",
                    )
                )
                or r1.nbf != n
                or any(r1.nocc(s) != r0.nocc(s) for s in ("alpha", "beta"))
                or candidate.backend.identity != operator.backend.identity
                or not np.array_equal(r1.overlap, r0.overlap)
                or not np.array_equal(r1.hcore, r0.hcore)
            ):
                raise RecoveryUnqualified(
                    "SCF result changes geometry, Hamiltonian or spin occupation"
                )
            if (
                r1.numeric_bytes > r0.numeric_bytes
                or _provider_bytes(candidate) > provider_bytes
            ):
                raise RecoveryUnqualified(
                    "SCF result exceeds reserved reference/provider budget"
                )
            # Matching AO/spin dimensions and capped provider storage keep
            # candidate work within the initial diagnostic's admitted bound.
            _assert_source_current(candidate)
            new_density = _densities(r1)
            if _same_root(
                new_density,
                original,
                r0.overlap,
                options.root_tolerance,
                spin_flip=False,
            ):
                continue
            post_energy = _electronic_energy(candidate, new_density)
            if (
                abs(
                    (r1.reference_energy - r0.reference_energy)
                    - (post_energy - base_energy)
                )
                > options.energy_tolerance
            ):
                raise RecoveryUnqualified(
                    "SCF energy does not match its converged source-owned Fock"
                )
            if r1.reference_energy >= r0.reference_energy - options.downhill_tolerance:
                continue
            certificate = diagnose(candidate)
            if certificate.status != "STABLE":
                continue
            spin = _metrics(r1, new_density)
            duplicate = next(
                (
                    index
                    for index, old in enumerate(roots)
                    if _same_root(
                        new_density,
                        old[2],
                        r0.overlap,
                        options.root_tolerance,
                        spin_flip=r0.nocc("alpha") == r0.nocc("beta"),
                    )
                ),
                None,
            )
            if duplicate is None:
                roots.append((candidate, certificate, new_density, spin))
            else:
                equivalent += 1
                # Preserve the energetically best member of each physical
                # equivalence class, not whichever SCF seed was tried first.
                if (
                    r1.reference_energy
                    < roots[duplicate][0].problem.reference.reference_energy
                ):
                    roots[duplicate] = (candidate, certificate, new_density, spin)
    check_current()
    if not roots:
        raise RecoveryUnqualified(
            "no lower-energy, fully stable, converged UHF root after bounded restarts"
        )
    winner = min(roots, key=lambda item: item[0].problem.reference.reference_energy)
    recovered, evidence, _, spin = winner
    _assert_current(evidence, recovered)
    identity = canonical_hash(
        {
            "kind": "bounded-real-uhf-recovery-v1",
            "original": initial.identity,
            "recovered": evidence.identity,
            "spin": spin.density_identity,
            "energy_change": recovered.problem.reference.reference_energy
            - r0.reference_energy,
            "source": sha256(Path(__file__).read_bytes()).hexdigest(),
            "options": vars(options),
        }
    )
    return UHFRecoveryResult(
        "RECOVERED",
        initial,
        evidence,
        recovered,
        operator,
        r0.reference_energy - recovered.problem.reference.reference_energy,
        spin,
        attempted,
        len(roots) + equivalent,
        equivalent,
        len(roots),
        identity,
    )


def native_uhf_operator_from_scf(
    previous: UHFResponseOperator,
    scf_result: object,
    *,
    residual_tolerance: float = 1e-8,
) -> UHFResponseOperator:
    """Export a native FockPlan UHF result into a fresh canonical response state.

    Independently recomputes the source-owned Focks and generalized MO frames;
    it does not accept a raw density as a converged or canonical snapshot.
    """
    from generativeqc.fock import FockScfResult
    from scipy.linalg import eigh

    if type(previous) is not UHFResponseOperator or not isinstance(
        scf_result, FockScfResult
    ):
        raise RecoveryUnqualified("expected native FockScfResult and UHF operator")
    if not math.isfinite(residual_tolerance) or not 0 < residual_tolerance <= 1e-7:
        raise ValueError("invalid native UHF export residual tolerance")
    r0 = previous.problem.reference
    n = r0.nbf
    d = np.asarray(scf_result.density)
    if d.shape != (2, n, n) or np.iscomplexobj(d) or not np.isfinite(d).all():
        raise RecoveryUnqualified("invalid native spin-resolved density")
    if not scf_result.initial_density_used or not math.isfinite(scf_result.energy):
        raise RecoveryUnqualified("native SCF did not consume the supplied spin seed")
    if (
        not math.isfinite(scf_result.density_rms)
        or scf_result.density_rms > residual_tolerance
    ):
        raise RecoveryUnqualified("native SCF reported inadequate convergence")
    _assert_source_current(previous)
    # Check native density's physical metric and spin electron counts.
    overlap = r0.overlap
    for i, spin in enumerate(("alpha", "beta")):
        if (
            np.max(np.abs(d[i] - d[i].T)) > residual_tolerance
            or abs(np.trace(d[i] @ overlap) - r0.nocc(spin)) > residual_tolerance
            or np.max(np.abs(d[i] @ overlap @ d[i] - d[i])) > residual_tolerance
        ):
            raise RecoveryUnqualified(
                f"native {spin} density is not an occupied projector"
            )
    a, b = previous._fock_response(d[0], d[1])
    focks = []
    frames = []
    energies = []
    mismatch = 0.0
    for spin, correction, density in zip(("alpha", "beta"), (a, b), d, strict=True):
        fock = r0.hcore + np.asarray(correction)
        if fock.shape != (n, n) or np.iscomplexobj(fock) or not np.isfinite(fock).all():
            raise RecoveryUnqualified("invalid native source Fock")
        eigvals, coefficients = eigh(fock, overlap, check_finite=True)
        occupied = coefficients[:, : r0.nocc(spin)]
        projector = occupied @ occupied.T
        mismatch = max(mismatch, float(np.max(np.abs(projector - density))))
        focks.append(fock)
        frames.append(coefficients)
        energies.append(eigvals)
    if mismatch > residual_tolerance:
        raise RecoveryUnqualified(
            "native final density is not canonical self-consistent"
        )
    generation = canonical_hash(
        {
            "kind": "native-uhf-scf-export-v1",
            "prior": r0.identity,
            "native": scf_result.identity,
            "density": sha256(np.ascontiguousarray(d).tobytes()).hexdigest(),
        }
    )
    result_reference = replace(
        r0,
        fock_alpha=focks[0],
        fock_beta=focks[1],
        coefficients_alpha=frames[0],
        coefficients_beta=frames[1],
        orbital_energies_alpha=energies[0],
        orbital_energies_beta=energies[1],
        reference_energy=float(scf_result.energy),
        scf_residual=max(float(scf_result.density_rms), mismatch),
        generation_id=generation,
    )
    _assert_source_current(previous)
    return UHFResponseOperator(
        UHFResponseOperator.build_problem(result_reference, previous.backend),
        previous.backend,
    )


def recover_uhf_with_native_fock_plan(
    operator: UHFResponseOperator,
    plan: object,
    *,
    options: RecoveryOptions | None = None,
    stability_options: StabilityOptions | None = None,
    max_iterations: int = 200,
) -> UHFRecoveryResult:
    """Run bounded recovery through an already prepared native CPU UHF plan.

    Reject a mismatched basis/source and non-HF/fitted/device plan rather than
    silently adapting to a different Coulomb/exchange Hamiltonian.
    """
    from generativeqc.fock import FockBuildSpec, FockPlan

    from .backends import NativeJKBackend

    if type(operator) is not UHFResponseOperator or not isinstance(plan, FockPlan):
        raise RecoveryUnqualified("native recovery requires UHF operator and FockPlan")
    if type(operator.backend) is not NativeJKBackend:
        raise RecoveryUnqualified("native CPU J/K provider is required")
    if (
        plan._device_id is not None
        or plan.spec
        not in (
            FockBuildSpec.hf("unrestricted", derivative_order=0),
            FockBuildSpec.hf("unrestricted", derivative_order=1),
        )
        or plan.diagnostics["screening_tolerance"] != 0.0
    ):
        raise RecoveryUnqualified(
            "only unscreened exact unrestricted CPU HF is admitted"
        )
    source = operator.backend.source
    if (
        plan.basis.nao != operator.problem.reference.nbf
        or plan.basis.representation != source.representation
        or plan.basis.atoms != source.atoms
        or plan.basis.shells != source.shells
        or plan.basis.charge != source.charge
        or plan.basis.multiplicity != source.multiplicity
    ):
        raise RecoveryUnqualified("native SCF plan and response source do not match")
    if type(max_iterations) is not int or not 0 < max_iterations <= 500:
        raise ValueError("max_iterations must be in [1, 500]")

    def solve(seed: np.ndarray) -> UHFResponseOperator:
        result = plan.solve(
            initial_density=seed,
            compute_forces=False,
            max_iterations=max_iterations,
            energy_tolerance=1e-11,
            density_tolerance=1e-10,
        )
        return native_uhf_operator_from_scf(operator, result)

    return recover_uhf_stability(
        operator, solve, options=options, stability_options=stability_options
    )
