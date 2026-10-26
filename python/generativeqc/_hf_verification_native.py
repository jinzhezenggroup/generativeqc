"""Typed metadata transport for native progressive-HF final verification.

The running controller supplies its selected library. Standalone verification
uses the first configured library; only missing protocols or Python integer
domains outside the ABI use explicit compatibility. Native errors never retry.
"""

from __future__ import annotations

import ctypes
import typing
from functools import lru_cache

import numpy as np

from . import _native


class Descriptor(ctypes.Structure):
    """Borrowed observations and exact work totals; no serialized final status."""

    _fields_ = (
        [
            (name, ctypes.c_uint32)
            for name in ("struct_size", "abi_version", "flags", "accuracy")
        ]
        + [
            (name, ctypes.c_uint64)
            for name in ("iterations", "iteration_limit", "fock_builds", "fock_limit")
        ]
        + [
            (name, ctypes.c_double)
            for name in (
                "energy",
                "energy_change",
                "density_rms",
                "residual_rms",
                "audit_rms",
                "audit_max",
                "audit_energy_difference",
                "energy_tolerance",
                "density_tolerance",
                "residual_tolerance",
            )
        ]
    )


class Decision(ctypes.Structure):
    """Detached status and ordered failure masks, without owning scientific state."""

    _fields_ = [
        (name, ctypes.c_uint32)
        for name in (
            "struct_size",
            "abi_version",
            "status",
            "target_reasons",
            "budget_reasons",
            "cleanup",
            "accuracy",
        )
    ]


@lru_cache(maxsize=8)
def _load(path: str) -> typing.Any:
    """Keep discovery local to the first selected immutable library path."""
    library = ctypes.CDLL(path)
    if getattr(library, "generativeqc_hf_verify_final_v1", None) is None:
        return None
    library.generativeqc_get_abi_version.argtypes = []
    library.generativeqc_get_abi_version.restype = ctypes.c_uint32
    if library.generativeqc_get_abi_version() != _native.ABI_VERSION:
        raise RuntimeError("GENERATIVEQC Python/native ABI version mismatch")
    library.generativeqc_status_message.argtypes = [ctypes.c_int]
    library.generativeqc_status_message.restype = ctypes.c_char_p
    return library


def optional_library() -> typing.Any:
    """A missing first protocol means compatibility, never another library search."""
    for candidate in _native._candidate_paths():
        if candidate.exists():
            return _load(str(candidate.resolve()))
    return None


def finalize(
    problem: typing.Any,
    target_stage: typing.Any,
    result: typing.Any,
    actual_model: typing.Any,
    actual_provider_hashes: typing.Any,
    budget: typing.Any,
    executions: typing.Any,
    *,
    accuracy: typing.Any = None,
    physical_audit: typing.Any = None,
    physical_audit_error: str | None = None,
    physical_audit_budget_exhausted: bool = False,
    library: typing.Any = None,
) -> typing.Any:
    """Normalize typed facts, execute one decision, and present public diagnostics."""
    from .accuracy import AccuracyAssessment
    from .progressive_controller import (
        FinalVerification,
        HFPhysicalResidualAudit,
        StageRole,
        TargetProblem,
    )

    if (
        not isinstance(problem, TargetProblem)
        or target_stage.role is not StageRole.TARGET
    ):
        raise TypeError("final verification requires the typed target stage")
    if physical_audit is not None and not isinstance(
        physical_audit, HFPhysicalResidualAudit
    ):
        raise TypeError("physical audit must be an HFPhysicalResidualAudit")
    selected_accuracy = result.accuracy if accuracy is None else accuracy
    if selected_accuracy is not None and not isinstance(
        selected_accuracy, AccuracyAssessment
    ):
        raise TypeError("final accuracy must be an AccuracyAssessment")

    total_iterations = sum(item.iterations for item in executions)
    fock_values = tuple(item.fock_builds for item in executions)
    total_fock_builds = (
        sum(fock_values) if all(value is not None for value in fock_values) else None
    )
    counts = (
        total_iterations,
        budget.maximum_total_iterations,
        total_fock_builds,
        budget.maximum_total_fock_builds,
    )
    if any(
        value is not None and (type(value) is not int or not 0 <= value <= 2**64 - 1)
        for value in counts
    ):
        return None
    selected_library = optional_library() if library is None else library
    operation = getattr(selected_library, "generativeqc_hf_verify_final_v1", None)
    if operation is None:
        return None

    actual_provider_identity = result.basis_metadata.get("model_identity", "")
    actual_hashes = tuple(sorted(tuple(pair) for pair in actual_provider_hashes))
    capabilities = {"energy"}
    if result.forces is not None:
        forces = np.asarray(result.forces)
        real_numeric = np.issubdtype(forces.dtype, np.number) and not np.issubdtype(
            forces.dtype, np.complexfloating
        )
        if (
            forces.shape == (problem.atom_count, 3)
            and real_numeric
            and np.isfinite(forces).all()
        ):
            capabilities.add("forces")
    missing = {item.observable for item in problem.accuracy.observables} - capabilities
    precision = result.precision
    precision_mapping = precision if isinstance(precision, dict) else {}
    accuracy_status = (
        "unverified" if selected_accuracy is None else selected_accuracy.status
    )
    accuracy_codes = {
        "unverified": 0,
        "unconverged": 1,
        "observed_unmet": 2,
        "estimated_above_target": 3,
        "observed_met": 4,
        "estimated_below_target": 5,
    }
    facts = (
        actual_model.identity == problem.model.identity,
        actual_provider_identity == problem.provider_identity,
        actual_hashes == problem.provider_hashes,
        bool(getattr(result, "succeeded", False)),
        bool(result.converged),
        result.physical_residual_rms is not None,
        physical_audit is not None,
        physical_audit is not None
        and physical_audit.model_identity == problem.model.identity
        and physical_audit.provider_identity == problem.provider_identity,
        physical_audit_error is not None,
        bool(physical_audit_budget_exhausted),
        not missing,
        isinstance(precision, dict)
        and precision.get("requested_mode")
        == target_stage.arithmetic.requested_precision,
        precision_mapping.get("effective_bits") == 64,
        target_stage.arithmetic.requested_precision == "auto",
        bool(precision_mapping.get("strict_refinement_applied")),
        selected_accuracy is not None,
        selected_accuracy is not None
        and selected_accuracy.model == problem.model
        and selected_accuracy.target == problem.accuracy,
        budget.maximum_total_fock_builds is not None,
        total_fock_builds is not None,
    )
    descriptor = Descriptor(
        struct_size=ctypes.sizeof(Descriptor),
        abi_version=_native.ABI_VERSION,
        flags=sum(int(value) << index for index, value in enumerate(facts)),
        accuracy=accuracy_codes.get(accuracy_status, 6),
        iterations=total_iterations,
        iteration_limit=budget.maximum_total_iterations,
        fock_builds=total_fock_builds or 0,
        fock_limit=budget.maximum_total_fock_builds or 0,
        energy=float(result.energy),
        energy_change=float(result.energy_change),
        density_rms=float(result.density_rms),
        residual_rms=0.0
        if result.physical_residual_rms is None
        else float(result.physical_residual_rms),
        audit_rms=0.0 if physical_audit is None else physical_audit.rms_commutator,
        audit_max=0.0 if physical_audit is None else physical_audit.maximum_commutator,
        audit_energy_difference=0.0
        if physical_audit is None
        else physical_audit.energy_difference,
        energy_tolerance=problem.convergence.energy_change,
        density_tolerance=problem.convergence.density_rms,
        residual_tolerance=problem.convergence.physical_residual_rms,
    )
    decision = Decision(
        struct_size=ctypes.sizeof(Decision), abi_version=_native.ABI_VERSION
    )
    operation.argtypes = [ctypes.POINTER(Descriptor), ctypes.POINTER(Decision)]
    operation.restype = ctypes.c_int
    _native.check(
        selected_library, operation(ctypes.byref(descriptor), ctypes.byref(decision))
    )
    if (
        decision.status > 3
        or decision.cleanup > 3
        or decision.accuracy > 6
        or decision.target_reasons >> 17
        or decision.budget_reasons >> 4
    ):
        raise RuntimeError("native HF verification returned an inconsistent receipt")

    target_reasons = (
        "executed scientific model differs from TargetProblem",
        "executed provider identity differs from TargetProblem",
        "executed provider hashes differ from TargetProblem",
        "target HF did not converge successfully",
        "target energy is nonfinite",
        "target energy-change tolerance is unmet",
        "target density tolerance is unmet",
        "physical residual audit changed the requested target",
        "fixed-density audit energy differs from target result",
        f"target physical residual audit failed: {physical_audit_error}",
        "target physical residual is unavailable",
        "target physical residual tolerance is unmet",
        "target result omitted required observables: " + ", ".join(sorted(missing)),
        "executed arithmetic policy differs from StagePlan",
        "strict final arithmetic was not established",
        "auto arithmetic omitted strict target cleanup",
        "accuracy assessment changed the requested target",
    )
    budget_reasons = (
        "physical verification host budget was exhausted",
        "actual iterations exceeded the progressive budget",
        "Fock-build budget could not be verified",
        "actual Fock builds exceeded the progressive budget",
    )
    reasons = [
        text
        for index, text in enumerate(target_reasons)
        if decision.target_reasons & (1 << index)
    ]
    reasons.extend(
        text
        for index, text in enumerate(budget_reasons)
        if decision.budget_reasons & (1 << index)
    )
    if decision.accuracy == 0:
        accuracy_status = "unverified"
    status = ("unverified", "verified", "unmet", "budget_exhausted")[decision.status]
    if status == "unverified":
        reasons.append(f"observable accuracy status is {accuracy_status}")
    residual = (
        result.physical_residual_rms
        if physical_audit is None
        else physical_audit.rms_commutator
    )
    return FinalVerification(
        status=status,
        target_established=not decision.target_reasons,
        problem_identity=problem.identity,
        requested_model_identity=problem.model.identity,
        actual_model_identity=actual_model.identity,
        requested_provider_identity=problem.provider_identity,
        actual_provider_identity=actual_provider_identity,
        requested_provider_hashes=problem.provider_hashes,
        actual_provider_hashes=actual_hashes,
        physical_residual_rms=residual,
        physical_residual_max=None
        if physical_audit is None
        else physical_audit.maximum_commutator,
        physical_residual_tolerance=problem.convergence.physical_residual_rms,
        physical_residual_source=(
            "fixed_density_target_audit"
            if physical_audit is not None
            else "native_result"
            if residual is not None
            else "unavailable"
        ),
        physical_audit_identity=None
        if physical_audit is None
        else physical_audit.operator_identity,
        fixed_density_energy_difference=None
        if physical_audit is None
        else physical_audit.energy_difference,
        accuracy_status=accuracy_status,
        capabilities=tuple(sorted(capabilities)),
        total_iterations=total_iterations,
        total_fock_builds=total_fock_builds,
        strict_cleanup=("unavailable", "missing", "applied", "not_required_fp64")[
            decision.cleanup
        ],
        reasons=tuple(reasons),
    )
