"""Explicit native CPU stability-qualified singlepoint and geometry continuation."""

from __future__ import annotations

import time
import typing
from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.dft import NativeAO

from . import _native
from ._api_types import Atom
from ._spin_root_control import (
    _evaluate,
    _recover_spin_root,
    _root_distance,
    _spin_diagnostics,
    _Work,
)
from .elements import checked_integer
from .overlap import cross_overlap
from .profiles import canonical_hash
from .spin_roots import (
    SpinRootPolicy,
    SpinRootResult,
    SpinRootUnqualified,
    _has_uhf_reference,
)


def _source_identity() -> str:
    """Execution provenance, separate from native method/scientific identities."""
    root = Path(__file__).parent
    return canonical_hash(
        {
            name: sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "_spin_root_workflow.py",
                "_spin_root_control.py",
                "_spin_root_native.py",
                "_response_spectrum.py",
                "spin_stationary.py",
                "rks_response.py",
                "spin_roots.py",
            )
        }
    )


def _preflight(
    calculator: typing.Any,
    atoms: tuple[Atom, ...],
    charge: int,
    multiplicity: int,
    policy: SpinRootPolicy,
) -> tuple:
    if (
        calculator._context_descriptor().backend != _native.BACKEND_CPU_REFERENCE
        or calculator._precision_mode != _native.PRECISION_FP64
        or calculator._density_fitting_mode != _native.DENSITY_FITTING_NONE
        or not 0.0 <= calculator._screening_tolerance <= 1e-12
        or calculator._dispersion_method_ir is not None
        or getattr(calculator, "_application_precision_budget", None) is not None
    ):
        raise SpinRootUnqualified(
            "spin-root qualification requires all-electron CPU FP64 and SCF screening at most 1e-12"
        )
    hf = _has_uhf_reference(calculator._method)
    if not hf:
        if (
            calculator._capabilities.family != "density_functional"
            or calculator.ks_options.method_ir.spin != "polarized"
        ):
            raise SpinRootUnqualified(
                "spin-root qualification requires UHF or qualified polarized semilocal KS"
            )
        spec = calculator.ks_options.functional
        if (
            spec.exact_exchange
            or spec.range_omega
            or spec.long_range_exchange
            or spec.ingredients not in (("rho",), ("rho", "sigma"))
        ):
            raise SpinRootUnqualified(
                "UKS root point response is qualified for semilocal LDA/GGA only"
            )
    metadata = calculator.basis_metadata(
        atoms, charge=charge, multiplicity=multiplicity
    )
    electrons = metadata["orbital"]["electrons"]
    if any(electrons["ecp_core_electrons"]):
        raise SpinRootUnqualified("spin-root qualification does not admit ECPs")
    shells = calculator._shells_for_atoms(atoms)
    n = sum(
        2 * s.angular_momentum + 1
        if calculator._representation_name == "real_spherical"
        else (s.angular_momentum + 1) * (s.angular_momentum + 2) // 2
        for s in shells
    )
    occupied = (electrons["nalpha"], electrons["nbeta"])
    dimension = sum(count * (n - count) for count in occupied)
    if (
        n > policy.maximum_ao
        or dimension > policy.maximum_dimension
        or 2 * dimension + 1 > policy.maximum_actions
    ):
        raise SpinRootUnqualified(
            "complete real spin domain rejected before native SCF"
        )
    minimum_numeric = 8 * (128 * n * n + 12 * dimension * dimension + 20 * dimension)
    if minimum_numeric > policy.memory_budget_bytes:
        raise SpinRootUnqualified("spin-root numeric budget rejected before native SCF")
    identity = canonical_hash(
        {
            "schema": "generativeqc.spin-root-model/v1",
            "calculator": calculator._model_signature(),
            "charge": charge,
            "multiplicity": multiplicity,
            "nuclei": [a.atomic_number for a in atoms],
        }
    )
    return shells, occupied, identity


def stable_singlepoint(
    calculator: typing.Any,
    atoms: typing.Any,
    *,
    charge: int,
    multiplicity: int,
    properties: typing.Any,
    policy: SpinRootPolicy | None,
    previous: SpinRootResult | None,
) -> SpinRootResult:
    """Publish only a stable, stationary root of the requested native model."""
    started = time.perf_counter()
    policy = SpinRootPolicy() if policy is None else policy
    if not isinstance(policy, SpinRootPolicy):
        raise TypeError("policy must be SpinRootPolicy")
    charge = checked_integer(charge, "charge", low=-(2**31), high=2**31 - 1)
    multiplicity = checked_integer(multiplicity, "multiplicity", low=1, high=2**31 - 1)
    atoms = tuple(Atom.from_value(atom) for atom in atoms)
    if not atoms:
        raise ValueError("at least one atom is required")
    shells, occupied, model_identity = _preflight(
        calculator, atoms, charge, multiplicity, policy
    )
    if previous is not None and (
        not isinstance(previous, SpinRootResult)
        or previous.reference.model_identity != model_identity
    ):
        raise SpinRootUnqualified(
            "previous root has a different model, ordered nuclei, charge or spin"
        )
    requested = tuple(
        sorted(
            set(calculator._default_properties() if properties is None else properties)
            | {"energy"}
        )
    )
    if set(requested) - {"energy", "forces"}:
        raise ValueError("spin-root properties are energy and optional forces")
    try:
        from ._spin_root_native import _NativeSpinModel
    except ModuleNotFoundError as error:
        if error.name == "scipy" or error.name.startswith("scipy."):
            raise ImportError(
                "CPU root qualification requires generativeqc[stability]"
            ) from error
        raise
    algorithm = _source_identity()
    signature = calculator._model_signature()
    calls = []
    with ExitStack() as owners:
        basis = owners.enter_context(
            NativeAO(
                atoms,
                basis=shells,
                representation=calculator._representation_name,
                charge=charge,
                multiplicity=multiplicity,
            )
        )

        def solve(seed: np.ndarray | None, energy: float | None) -> typing.Any:
            if (
                calculator._model_signature() != signature
                or _source_identity() != algorithm
            ):
                raise SpinRootUnqualified("root model/source changed during execution")
            try:
                model = _NativeSpinModel(
                    calculator,
                    basis,
                    model_identity,
                    charge=charge,
                    multiplicity=multiplicity,
                    properties=requested,
                    policy=policy,
                    seed=seed,
                    seed_energy=energy,
                )
            except Exception as error:
                calls.append(
                    {
                        "status": "failed",
                        "error": type(error).__name__,
                        "seeded": seed is not None,
                    }
                )
                raise
            owners.callback(model.close)
            calls.append(
                {
                    "status": "converged",
                    "energy": model.item.energy,
                    "reference_energy": model.reference.energy,
                    "iterations": model.item.iterations,
                    "fock_builds": model.item.fock_builds,
                    "seeded": seed is not None,
                    "seed_consumed": model.item.warm_start_used,
                    "density_rms": model.item.density_rms,
                }
            )
            return model

        initial = solve(None, None)
        distance = None
        projection_receipts = []
        work = _Work(policy.maximum_actions)
        if previous is not None:
            if (
                2 * (initial.reference.numeric_bytes + initial.workspace_bytes)
                > policy.memory_budget_bytes
            ):
                raise SpinRootUnqualified(
                    "root continuation staging exceeds numeric budget"
                )
            from . import _projection_native
            from .projection import ProjectionPolicy

            cross = cross_overlap(
                calculator,
                calculator,
                atoms,
                source_atoms=previous.reference.atoms,
                charge=charge,
                multiplicity=multiplicity,
                maximum_bytes=policy.memory_budget_bytes,
            )
            seed = []
            projection_policy = ProjectionPolicy(maximum_ao=policy.maximum_ao)
            for spin, count in enumerate(occupied):
                if count == 0:
                    seed.append(np.zeros_like(initial.reference.overlap))
                    projection_receipts.append({"execution_owner": "empty-spin"})
                    continue
                projection = _projection_native.project(
                    previous.reference.overlap,
                    initial.reference.overlap,
                    cross,
                    previous.reference.coefficients[spin, :, :count],
                    occupied=None,
                    occupation=1,
                    policy=projection_policy,
                )
                if projection is None or projection.execution_owner != "native_cpu":
                    raise SpinRootUnqualified(
                        "root continuation requires qualified native metric projection"
                    )
                seed.append(projection.density)
                projection_receipts.append(
                    {
                        "execution_owner": projection.execution_owner,
                        **asdict(projection.diagnostics),
                    }
                )
            seed = immutable(np.asarray(seed))
            proposal = _evaluate(initial, seed, work)
            continued = solve(seed, proposal.energy)
            initial.close()
            initial = continued

            def distance(reference: typing.Any) -> tuple[float, str]:
                direct = _root_distance(previous.reference, reference, cross.T)
                if occupied[0] == occupied[1]:
                    flipped = _root_distance(
                        previous.reference, reference, cross.T, flip=True
                    )
                    if flipped + policy.root_tolerance < direct:
                        return flipped, "spin_equivalent"
                return direct, "same"

        winner, first, spectrum, work, distinct, equivalents = _recover_spin_root(
            initial, solve, policy, preferred_distance=distance, work=work
        )
        relation, root_distance = "initial", None
        if distance is not None:
            root_distance, relation = distance(winner.reference)
            if root_distance > policy.tracking_tolerance:
                relation = "switched"
                if not policy.allow_root_switch:
                    raise SpinRootUnqualified(
                        "qualified state differs from the intended geometry-sequence root"
                    )
        if _source_identity() != algorithm:
            raise SpinRootUnqualified("root source changed before publication")
        winner.validate_current()
        result = winner.result()
        reference = winner.reference
        spin_squared, alpha, beta = _spin_diagnostics(reference)
        atom_populations = np.zeros((2, len(atoms)))
        offset = 0
        for shell in basis.shells:
            count = (
                2 * shell.angular_momentum + 1
                if basis.representation == "real_spherical"
                else (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
            )
            atom_populations[0, shell.atom_index] += alpha[
                offset : offset + count
            ].sum()
            atom_populations[1, shell.atom_index] += beta[offset : offset + count].sum()
            offset += count
        diagnostics = {
            "schema": "generativeqc.native-spin-root/v1",
            "domain": "complete real internal CPU OV",
            "policy_identity": policy.identity,
            "algorithm_identity": algorithm,
            "initial_reference": initial.reference.identity,
            "reference": reference.identity,
            "previous": None if previous is None else previous.identity,
            "initial_status": first.status,
            "status": spectrum.status,
            "lowest_curvature": float(spectrum.eigenvalues[0])
            if len(spectrum.eigenvalues)
            else None,
            "uncertainty": spectrum.uncertainty,
            "symmetry_defect": spectrum.symmetry_defect,
            "eigen_residual_bound": spectrum.residual_bound,
            "scf_residual": reference.residual,
            "spin_squared": spin_squared,
            "occupied": occupied,
            "alpha_atom_populations": atom_populations[0].tolist(),
            "beta_atom_populations": atom_populations[1].tolist(),
            "root_relation": relation,
            "root_distance": root_distance,
            "distinct_roots": distinct,
            "equivalent_restarts": equivalents,
            "native_scf_calls": calls,
            "response_actions": work.response_actions,
            "fixed_density_evaluations": work.fixed_density_evaluations,
            "scf_restarts": work.scf_restarts,
            "projection": projection_receipts,
            "numeric_memory_scope": "controller/reference/response buffers; native SCF has the calculator resource contract",
        }
        identity = canonical_hash(diagnostics)
        eigenvalues = spectrum.eigenvalues
    # Complete synchronous endpoint includes cleanup of every native attempt.
    diagnostics["endpoint_seconds"] = time.perf_counter() - started
    return SpinRootResult(
        result,
        reference,
        first.status,
        first.status not in ("STABLE", "EMPTY"),
        eigenvalues,
        spin_squared,
        alpha,
        beta,
        relation,
        identity,
        diagnostics,
    )
