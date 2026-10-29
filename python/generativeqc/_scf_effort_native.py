"""Call-local metadata transport for the shared native SCF-effort decision."""

from __future__ import annotations

import ctypes
from functools import lru_cache
from typing import Any

from . import _native


class Descriptor(ctypes.Structure):
    """Versioned borrowed policy/calibration/state input; no physical owner."""

    _fields_ = (
        [
            (name, ctypes.c_uint32)
            for name in ("struct_size", "abi_version", "flags", "calibration_domain")
        ]
        + [
            (name, ctypes.c_uint64)
            for name in ("level_count", "level_index", "safe_streak", "relax_after")
        ]
        + [
            ("density_tolerances", ctypes.POINTER(ctypes.c_double)),
        ]
        + [
            (name, ctypes.c_double)
            for name in (
                "optimizer_force_tolerance",
                "intermediate_energy_error",
                "intermediate_force_fraction",
                "max_intermediate_force_error",
                "near_stationary_factor",
                "energy_scale",
                "force_max_scale",
                "force_rms_scale",
                "diagnostic_floor",
                "current_force_max",
            )
        ]
    )


class Decision(ctypes.Structure):
    """Detached scalar publication; unused estimate fields are zero."""

    _fields_ = (
        [
            (name, ctypes.c_uint32)
            for name in (
                "struct_size",
                "abi_version",
                "action",
                "reason",
                "has_estimate",
                "reserved",
            )
        ]
        + [(name, ctypes.c_uint64) for name in ("level_index", "safe_streak")]
        + [
            (name, ctypes.c_double)
            for name in (
                "allowed_energy_error",
                "allowed_force_error",
                "energy_abs",
                "force_max_abs",
                "force_rms",
                "diagnostic_value",
            )
        ]
    )


class FinalDescriptor(ctypes.Structure):
    """Measured final gates and caller-compared identities, not an empirical bound."""

    _fields_ = [
        (name, ctypes.c_uint32)
        for name in ("struct_size", "abi_version", "flags", "reserved")
    ] + [
        (name, ctypes.c_double)
        for name in (
            "energy",
            "energy_change",
            "density_rms",
            "force_max_abs",
            "energy_tolerance",
            "density_tolerance",
            "optimizer_force_tolerance",
        )
    ]


class Verification(ctypes.Structure):
    """Detached final status and complete unmet-gate bit mask."""

    _fields_ = [
        (name, ctypes.c_uint32)
        for name in ("struct_size", "abi_version", "status", "unmet_gates")
    ]


@lru_cache(maxsize=8)
def _load(path: str) -> Any:
    """Bind each immutable library path once; failed loads are never cached/retried."""
    library = ctypes.CDLL(path)
    operation = getattr(library, "generativeqc_scf_effort_decide_v1", None)
    verification = getattr(library, "generativeqc_scf_effort_verify_v1", None)
    if operation is None or verification is None:
        raise RuntimeError(
            "SCF effort decisions require the current native decide/verify protocol; "
            "rebuild or reinstall GenerativeQC"
        )
    library.generativeqc_get_abi_version.argtypes = []
    library.generativeqc_get_abi_version.restype = ctypes.c_uint32
    if library.generativeqc_get_abi_version() != _native.ABI_VERSION:
        raise RuntimeError("GENERATIVEQC Python/native ABI version mismatch")
    library.generativeqc_status_message.argtypes = [ctypes.c_int]
    library.generativeqc_status_message.restype = ctypes.c_char_p
    operation.argtypes = [ctypes.POINTER(Descriptor), ctypes.POINTER(Decision)]
    operation.restype = ctypes.c_int
    verification.argtypes = [
        ctypes.POINTER(FinalDescriptor),
        ctypes.POINTER(Verification),
    ]
    verification.restype = ctypes.c_int
    return library


def require_library() -> Any:
    """Bind the first selected image; missing support never enables Python decisions.

    Built-in policies share the installed SDK's current protocol. Keep load and
    protocol failures visible instead of silently selecting a second image or
    changing the owner of a scientific decision.
    """
    for candidate in _native._candidate_paths():
        if candidate.exists():
            return _load(str(candidate.resolve()))
    raise RuntimeError(
        "SCF effort decisions require the current native library; "
        "build or install GenerativeQC"
    )


def decide(
    policy: Any, state: Any, force: float | None, method: str, basis: str
) -> Any:
    """Borrow typed values; unsupported Python integer domains stay explicit compatibility."""
    if any(
        type(value) is not int or not 0 <= value <= 2**64 - 1
        for value in (state.level_index, state.safe_streak, policy.relax_after)
    ):
        return None
    library = require_library()
    calibration = policy.estimator
    domain = (
        1
        if method not in calibration.methods
        else 2
        if (method, basis) not in calibration.method_basis_domains
        else 0
    )
    levels = (ctypes.c_double * len(policy.levels))(
        *(level.density_tolerance for level in policy.levels)
    )
    descriptor = Descriptor(
        struct_size=ctypes.sizeof(Descriptor),
        abi_version=_native.ABI_VERSION,
        flags=int(policy.strict_reproducible) | (2 if force is not None else 0),
        calibration_domain=domain,
        level_count=len(levels),
        level_index=state.level_index,
        safe_streak=state.safe_streak,
        relax_after=policy.relax_after,
        density_tolerances=levels,
        **{
            name: getattr(policy, name)
            for name in (
                "optimizer_force_tolerance",
                "intermediate_energy_error",
                "intermediate_force_fraction",
                "max_intermediate_force_error",
                "near_stationary_factor",
            )
        },
        **{
            name: getattr(calibration, name)
            for name in (
                "energy_scale",
                "force_max_scale",
                "force_rms_scale",
                "diagnostic_floor",
            )
        },
        current_force_max=0.0 if force is None or policy.strict_reproducible else force,
    )
    result = Decision(
        struct_size=ctypes.sizeof(Decision), abi_version=_native.ABI_VERSION
    )
    _native.check(
        library,
        library.generativeqc_scf_effort_decide_v1(
            ctypes.byref(descriptor), ctypes.byref(result)
        ),
    )
    return result, domain


def verify(
    policy: Any,
    model_matches: bool,
    strict_level: bool,
    converged: bool,
    energy: float,
    change: float,
    density: float,
    force: float,
) -> Any:
    """Publish actual final gates; native errors never turn into Python verification."""
    library = require_library()
    descriptor = FinalDescriptor(
        struct_size=ctypes.sizeof(FinalDescriptor),
        abi_version=_native.ABI_VERSION,
        flags=int(model_matches) | (int(strict_level) << 1) | (int(converged) << 2),
        energy=energy,
        energy_change=change,
        density_rms=density,
        force_max_abs=force,
        energy_tolerance=policy.strict_level.energy_tolerance,
        density_tolerance=policy.strict_level.density_tolerance,
        optimizer_force_tolerance=policy.optimizer_force_tolerance,
    )
    result = Verification(
        struct_size=ctypes.sizeof(Verification), abi_version=_native.ABI_VERSION
    )
    _native.check(
        library,
        library.generativeqc_scf_effort_verify_v1(
            ctypes.byref(descriptor), ctypes.byref(result)
        ),
    )
    reasons = tuple(
        message
        for index, message in enumerate(
            (
                "final model identity does not match target",
                "final evaluation did not use strict SCF effort",
                "final SCF did not converge",
                "final energy-change gate is unmet",
                "final density gate is unmet",
                "optimizer force termination gate is unmet",
            )
        )
        if result.unmet_gates & (1 << index)
    )
    return ("observed_met", "observed_unmet", "unverified")[result.status], reasons


def reasons(result: Decision, domain: int) -> tuple[str, str]:
    """Render public decision/history labels without selecting or changing effort."""
    public = {
        0: "strict reproducible mode",
        1: "no trustworthy previous-force signal",
        2: "near-stationary strict cleanup",
        3: (
            "method is outside SCF force-estimator calibration domain"
            if domain == 1
            else "method/basis pair is outside SCF force-estimator calibration domain"
        ),
        4: "energy_abs must be finite and nonnegative",
        5: "force_max_abs must be finite and nonnegative",
        6: "force_rms must be finite and nonnegative",
        7: "no calibrated level meets observable allowance",
        8: "waiting for SCF-effort relaxation hysteresis",
        9: "sustained calibrated force margin",
        10: "observable allowance requires tighter SCF",
        11: "current SCF effort matches calibrated allowance",
    }[result.reason]
    history = (
        "calibration-domain fallback"
        if result.action == 1
        else "calibrated force margin permits looser next step"
        if result.reason == 9
        else public
    )
    return public, history
