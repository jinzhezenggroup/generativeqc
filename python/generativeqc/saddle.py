"""Synchronous Python access to the native, bounded local saddle corrector.

The C++ optimizer owns stepping, mode tracking and final verification. Derivative
callbacks must bind one qualified physical model; this interface does not confer
molecular, electronic-state or method-domain qualification on those callbacks.
"""

from __future__ import annotations

import ctypes as ct
import math
import typing
from dataclasses import dataclass, fields
from time import perf_counter
from types import MappingProxyType

import numpy as np

from . import _native

if typing.TYPE_CHECKING:
    import generativeqc.vibrations

    from .calculator import Atom, Calculator
    from .response_solver import GMRESOptions

__all__ = [
    "MolecularSaddleResult",
    "SaddleOptions",
    "SaddleProgress",
    "SaddleResult",
    "correct_saddle",
    "optimize_saddle",
]

_STATUSES = (
    "first_order_saddle",
    "other_index",
    "ambiguous_curvature",
    "attempt_limit",
    "target_not_negative",
    "mode_tracking_lost",
    "trust_exhausted",
    "evaluation_failed",
    "hessian_failed",
    "cancelled",
)
_COUNTS = (
    "attempts",
    "accepted_steps",
    "energy_gradient_calls",
    "hessian_calls",
    "negative_curvatures",
    "near_zero_curvatures",
    "rigid_modes",
    "active_modes",
    "numeric_capacity_bytes",
)
_METRICS = (
    "energy",
    "energy_change",
    "gradient_rms",
    "gradient_max",
    "displacement_rms_angstrom",
    "displacement_max_angstrom",
    "trust_radius",
    "target_curvature",
    "target_overlap",
)


@dataclass(frozen=True)
class SaddleOptions:
    """Explicit native-corrector controls, with at most 100 attempted steps.

    Energy is Hartree, gradient controls Hartree/Bohr, displacement controls
    Angstrom and trust radii Bohr. Both RMS denominators are 3N; no rigid
    alignment is applied. ``maximum_numeric_bytes`` bounds only native optimizer
    storage, excluding callback memory and Python result copies. Rigid projection
    is opt-in for general callbacks; enable it for isolated molecular potentials.
    Positive finite controls and ordered radii are checked at construction;
    malformed controls raise ValueError or TypeError. The immutable record owns no resources.
    """

    max_attempts: int = 100
    max_coordinates: int = 300
    maximum_numeric_bytes: int = 16 << 20
    project_rigid_motions: bool = False
    energy_change: float = 1e-6
    gradient_rms: float = 3e-4
    gradient_max: float = 4.5e-4
    displacement_rms_angstrom: float = 1.2e-3
    displacement_max_angstrom: float = 1.8e-3
    initial_radius: float = 0.1
    minimum_radius: float = 1e-8
    maximum_radius: float = 0.5
    curvature_tolerance: float = 1e-7
    raw_symmetry_tolerance: float = 1e-10
    minimum_mode_overlap: float = 0.5
    acceptance_ratio: float = 0.1

    def __post_init__(self) -> None:
        """Reject invalid controls before loading a library or invoking callbacks."""
        for name, maximum in (
            ("max_attempts", 100),
            ("max_coordinates", 300),
            ("maximum_numeric_bytes", (1 << 64) - 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [1, {maximum}]")
        if type(self.project_rigid_motions) is not bool:
            raise ValueError("project_rigid_motions must be bool")
        for field in fields(self)[4:]:
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field.name} must be finite and positive")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field.name} must be finite and positive")
        if not self.minimum_radius <= self.initial_radius <= self.maximum_radius:
            raise ValueError(
                "require minimum_radius <= initial_radius <= maximum_radius"
            )
        if self.minimum_mode_overlap > 1 or self.acceptance_ratio >= 1:
            raise ValueError(
                "require minimum_mode_overlap <= 1 and acceptance_ratio < 1"
            )


@dataclass(frozen=True, eq=False)
class SaddleProgress:
    """Detached accepted-state observation, including after a rejected trial.

    Coordinates have shape (N, 3) in Bohr; receipt contains the native status,
    work counts and convergence metrics. Arrays and mapping are read-only and
    remain usable after callback return. ``elapsed_seconds`` includes callback
    work since entry to the complete corrector endpoint. A progress observation
    alone is not a successful final result.
    """

    coordinates: np.ndarray
    receipt: typing.Mapping[str, typing.Any]
    elapsed_seconds: float


@dataclass(frozen=True, eq=False)
class SaddleResult:
    """Detached native outcome; C execution success is not optimization success.

    Only ``converged`` reports a freshly verified mathematical index-one saddle.
    Failure retains the last fully accepted coordinates, not a failed trial.
    Gradient, raw Hessian, target mode and active curvatures are None when initial
    preparation never produced a valid state; inspect ``has_valid_state`` before
    interpreting receipt metrics. Shapes are (N, 3), (3N, 3N), (N, 3) and
    (active_modes,), in atomic units. Curvatures are not harmonic frequencies.
    Arrays and receipt are read-only detached copies. ``failures`` retains
    (stage, exception type, message) for callback errors, including cancellation
    and progress errors, which never unwind through the C ABI. No owner survives
    the synchronous call. Callback resources must not be shared concurrently.
    """

    coordinates: np.ndarray
    gradient: np.ndarray | None
    hessian: np.ndarray | None
    target_mode: np.ndarray | None
    curvatures: np.ndarray | None
    receipt: typing.Mapping[str, typing.Any]
    detail: str
    failures: tuple[tuple[str, str, str], ...]
    elapsed_seconds: float

    @property
    def converged(self) -> bool:
        """Whether fresh final verification establishes an unambiguous index one."""
        return bool(
            not self.failures
            and self.receipt["has_valid_state"]
            and self.receipt["stationary_verified"]
            and self.receipt["status"] == "first_order_saddle"
        )


@dataclass(frozen=True, eq=False)
class MolecularSaddleResult:
    """Fixed-model molecular result with native work and root-continuity receipts.

    ``optimizer`` retains the raw native outcome and last accepted Bohr geometry.
    ``atoms`` follows the input nucleus order at that geometry. ``harmonics`` is
    an unscaled analysis of a freshly verified stationary Hessian, or None if
    verification failed. All modes, including soft/negative modes, are retained.
    ``diagnostics`` records completed/failed derivative stages, SCF and response
    work, state-overlap checks, model identity and complete endpoint seconds.
    Records and arrays are detached and read-only; no electronic owner remains.
    This does not qualify methods outside the admitted Calculator domain.
    """

    optimizer: SaddleResult
    atoms: tuple[Atom, ...]
    harmonics: generativeqc.vibrations.HarmonicAnalysis | None
    diagnostics: typing.Mapping[str, typing.Any]

    @property
    def converged(self) -> bool:
        """Whether native final index-one verification and harmonic analysis passed."""
        return self.optimizer.converged and self.harmonics is not None


def optimize_saddle(
    calculator: Calculator,
    atoms: typing.Iterable[Atom | tuple[str | int, typing.Sequence[float]]],
    *,
    masses: typing.Any,
    charge: int = 0,
    multiplicity: int = 1,
    options: SaddleOptions | None = None,
    initial_mode: typing.Any = None,
    minimum_occupied_overlap: float = 0.9,
    root_overlap_budget_bytes: int = 128 << 20,
    integral_budget_bytes: int = 64 << 20,
    max_eri_primitives: int | None = None,
    output_budget_bytes: int = 64 << 20,
    grid_budget_bytes: int = 128 << 20,
    solver_options: GMRESOptions | None = None,
    progress: typing.Callable[[SaddleProgress], bool | None] | None = None,
    evaluation_progress: typing.Callable[[typing.Mapping[str, typing.Any]], None]
    | None = None,
    cancelled: typing.Callable[[], bool] | None = None,
) -> MolecularSaddleResult:
    """Correct a local saddle with one fixed qualified CPU hybrid RKS/UKS model.

    Supply a Calculator whose public Hessian and forces are admitted, finite
    ordered atoms in Bohr, and positive physical/isotopic masses in dalton.
    This adapter currently requires canonical B3LYP, all-electron, direct,
    CPU RKS/UKS; no ECP, DF, GPU, reference engine or lower-level
    preoptimization is used.
    A qualified uncut two-body D3(BJ) source is retained unchanged. The native
    C++ corrector owns all optimizer steps and the <=100-attempt cap. Default
    controls enable rigid projection and use the five reference thresholds.

    Each geometry uses one prepared SCF for energy/forces and raw Hessian.
    Independent cold SCFs must retain each occupied spin subspace and its
    integer population: all principal overlaps with the last accepted state's
    corresponding orbitals must meet
    ``minimum_occupied_overlap`` in (0, 1]. This is a fail-closed local continuity
    policy, not a stability proof, a global ground-state guarantee or spin-root
    selection. It never follows rejected-trial roots. MethodIR, basis, grid,
    charge and multiplicity remain fixed; changing the calculator in callbacks
    fails explicitly. The retained native batch supplies coupled response and
    exactly-once D3 curvature through the same bridge as Calculator.hessian.
    No Python AO/grid reconstruction, numerical response or runtime compiler is
    used. Native canonical orbitals are exported only for root continuity.

    Byte controls independently bound their named providers, not total RSS.
    ``max_eri_primitives`` caps first-plus-curvature nuclear ERI primitive
    visits over each complete requested Hessian (default 100,000,000), not over
    individual columns or the entire optimization. Increasing it does not
    change the model, qualify a domain, or bound endpoint time/induced-Fock work.
    ``progress`` follows correct_saddle; ``evaluation_progress`` observes
    detached start/complete/failed stage records, including geometry and elapsed
    seconds. Callback/SCF/response failures preserve the last accepted geometry
    without successful final classification. Invalid controls raise ValueError
    or TypeError; unqualified domains raise NotImplementedError before SCF.
    Calls sharing a calculator must be serialized. Returned data outlive all
    providers. Frequencies are unscaled, with no deletion of unwanted modes.
    """
    from ._molecular_saddle import run

    return run(
        calculator,
        atoms,
        masses=masses,
        charge=charge,
        multiplicity=multiplicity,
        options=options,
        initial_mode=initial_mode,
        minimum_occupied_overlap=minimum_occupied_overlap,
        root_overlap_budget_bytes=root_overlap_budget_bytes,
        integral_budget_bytes=integral_budget_bytes,
        max_eri_primitives=max_eri_primitives,
        output_budget_bytes=output_budget_bytes,
        grid_budget_bytes=grid_budget_bytes,
        solver_options=solver_options,
        progress=progress,
        evaluation_progress=evaluation_progress,
        cancelled=cancelled,
    )


class _Options(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_size_t),
        ("abi_version", ct.c_uint32),
        ("project_rigid_motions", ct.c_uint32),
        ("max_attempts", ct.c_uint64),
        ("max_coordinates", ct.c_uint64),
        ("maximum_numeric_bytes", ct.c_uint64),
        *((field.name, ct.c_double) for field in fields(SaddleOptions)[4:]),
    ]


class _Receipt(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_size_t),
        ("abi_version", ct.c_uint32),
        ("status", ct.c_uint32),
        ("has_valid_state", ct.c_uint32),
        ("stationary_verified", ct.c_uint32),
        *((name, ct.c_uint64) for name in _COUNTS),
        *((name, ct.c_double) for name in _METRICS),
    ]


_DoublePointer = ct.POINTER(ct.c_double)
_Evaluate = ct.CFUNCTYPE(
    ct.c_int,
    ct.c_void_p,
    ct.c_uint32,
    _DoublePointer,
    ct.c_size_t,
    _DoublePointer,
    _DoublePointer,
    _DoublePointer,
)
_Progress = ct.CFUNCTYPE(
    ct.c_int,
    ct.c_void_p,
    ct.POINTER(_Receipt),
    _DoublePointer,
    ct.c_size_t,
)
_Cancel = ct.CFUNCTYPE(ct.c_int, ct.c_void_p)


def _readonly(values: typing.Any) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _real_array(values: typing.Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return np.array(array, dtype=np.float64, order="C", copy=True)


def _receipt(value: _Receipt) -> typing.Mapping[str, typing.Any]:
    return MappingProxyType(
        {
            "status": _STATUSES[value.status],
            "has_valid_state": bool(value.has_valid_state),
            "stationary_verified": bool(value.stationary_verified),
            **{name: getattr(value, name) for name in _COUNTS},
            **{
                name: getattr(value, name) if value.has_valid_state else None
                for name in _METRICS
            },
        }
    )


def correct_saddle(
    coordinates: typing.Any,
    *,
    energy_gradient: typing.Callable[[np.ndarray], tuple[float, np.ndarray]],
    hessian: typing.Callable[[np.ndarray], np.ndarray],
    options: SaddleOptions | None = None,
    initial_mode: typing.Any = None,
    progress: typing.Callable[[SaddleProgress], bool | None] | None = None,
    cancelled: typing.Callable[[], bool] | None = None,
) -> SaddleResult:
    """Run the native CPU local corrector over caller-owned derivative providers.

    Coordinates must be finite Bohr values of shape (N, 3), 3N<=300. Providers
    receive detached read-only coordinates and return energy in Hartree,
    gradient (not forces) of shape (N, 3) and a raw (3N, 3N) Hessian in
    Hartree/Bohr^2. They must use one fixed physical model, units and state
    policy; no method selection, oracle, external optimizer or finite-difference
    implementation is supplied here. Initial mode, when given, has shape (N, 3).

    Every accepted/rejected trial consumes one attempt; initial and fresh-final
    derivatives count separately. ``progress`` observes only accepted-state
    geometry and returns true to cancel (None/false continues). ``cancelled`` is
    checked before derivatives. Exceptions from any callback become explicit
    failed/cancelled results with diagnostic strings, never C-stack unwinding.
    Argument errors raise ValueError, native capacity refusal raises MemoryError,
    and library/ABI failures raise RuntimeError. Numerical and convergence
    failures return SaddleResult, not success-like arrays for an invalid state.

    The call is synchronous. Providers/progress run on this thread, cannot retain
    borrowed native pointers, and must serialize shared resources. Results own
    read-only copies. This is callback infrastructure, not an installed molecular
    TS task or a promotion of any method's public derivative capabilities.
    """
    started = perf_counter()
    controls = SaddleOptions() if options is None else options
    if not isinstance(controls, SaddleOptions):
        raise TypeError("options must be SaddleOptions")
    initial = _real_array(coordinates, "coordinates")
    if (
        initial.ndim != 2
        or initial.shape[1] != 3
        or not initial.size
        or initial.size > controls.max_coordinates
        or not np.isfinite(initial).all()
    ):
        raise ValueError(
            "coordinates require finite shape (N, 3) within max_coordinates"
        )
    selected = None
    if initial_mode is not None:
        selected = _real_array(initial_mode, "initial_mode")
        norm_squared = float(np.sum(selected * selected))
        if selected.shape != initial.shape or not 0 < norm_squared < math.inf:
            raise ValueError("initial_mode requires finite nonzero shape (N, 3)")
    for callback in (energy_gradient, hessian, progress, cancelled):
        if callback is not None and not callable(callback):
            raise ValueError(
                "derivative, progress and cancellation providers must be callable"
            )
    if energy_gradient is None or hessian is None:
        raise ValueError("energy_gradient and hessian are required")

    failures = []

    def failed(stage: str, error: BaseException) -> int:
        try:
            message = str(error)
        except BaseException:  # noqa: BLE001 - Exception formatting cannot unwind through C either.
            message = "unprintable callback exception"
        failures.append((stage, type(error).__name__, message))
        return 1

    @_Evaluate
    def evaluate(
        user: typing.Any,
        stage: int,
        positions: typing.Any,
        count: int,
        energy: typing.Any,
        gradient: typing.Any,
        matrix: typing.Any,
    ) -> int:
        try:
            observed = _readonly(
                np.ctypeslib.as_array(positions, shape=(count,))
            ).reshape(initial.shape)
            if stage == 0:
                value, derivative = energy_gradient(observed)
                derivative = _real_array(derivative, "gradient")
                if (
                    not math.isfinite(value)
                    or derivative.shape != initial.shape
                    or not np.isfinite(derivative).all()
                ):
                    raise ValueError(
                        "energy/gradient provider returned invalid values or shape"
                    )
                energy[0] = value
                np.ctypeslib.as_array(gradient, shape=(count,))[:] = derivative.ravel()
            elif stage == 1:
                derivative = _real_array(hessian(observed), "Hessian")
                if (
                    derivative.shape != (count, count)
                    or not np.isfinite(derivative).all()
                ):
                    raise ValueError(
                        "Hessian provider returned invalid values or shape"
                    )
                np.ctypeslib.as_array(matrix, shape=(count * count,))[:] = (
                    derivative.ravel()
                )
            else:
                raise ValueError("unsupported native derivative stage")
            return 0
        except BaseException as error:  # noqa: BLE001 - No Python exception may unwind through a C callback.
            return failed("energy_gradient" if stage == 0 else "hessian", error)

    @_Progress
    def observe(
        user: typing.Any, pointer: typing.Any, positions: typing.Any, count: int
    ) -> int:
        try:
            if progress is None:
                return 0
            observed = _readonly(
                np.ctypeslib.as_array(positions, shape=(count,))
            ).reshape(initial.shape)
            return int(
                bool(
                    progress(
                        SaddleProgress(
                            observed,
                            _receipt(pointer.contents),
                            perf_counter() - started,
                        )
                    )
                )
            )
        except BaseException as error:  # noqa: BLE001 - No Python exception may unwind through a C callback.
            return failed("progress", error)

    @_Cancel
    def cancellation(user: typing.Any) -> int:
        try:
            return int(cancelled is not None and bool(cancelled()))
        except BaseException as error:  # noqa: BLE001 - No Python exception may unwind through a C callback.
            return failed("cancelled", error)

    library = _native.load_library()
    function = library.generativeqc_saddle_correct_v1
    function.restype = ct.c_int
    function.argtypes = [
        ct.POINTER(_Options),
        _DoublePointer,
        ct.c_size_t,
        _DoublePointer,
        _Evaluate,
        _Progress,
        _Cancel,
        ct.c_void_p,
        ct.POINTER(_Receipt),
        _DoublePointer,
        _DoublePointer,
        _DoublePointer,
        _DoublePointer,
        _DoublePointer,
        ct.POINTER(ct.c_char),
        ct.c_size_t,
    ]
    native_options = _Options(
        struct_size=ct.sizeof(_Options),
        abi_version=_native.ABI_VERSION,
        **{field.name: getattr(controls, field.name) for field in fields(controls)},
    )
    receipt = _Receipt(struct_size=ct.sizeof(_Receipt), abi_version=_native.ABI_VERSION)
    output = initial.copy()
    gradient = np.full_like(initial, np.nan)
    matrix = np.full((initial.size, initial.size), np.nan)
    mode = np.full_like(initial, np.nan)
    curvatures = np.full(initial.size, np.nan)
    detail = ct.create_string_buffer(1024)

    def pointer(array: np.ndarray | None) -> typing.Any:
        return None if array is None else array.ctypes.data_as(_DoublePointer)

    status = function(
        ct.byref(native_options),
        pointer(initial),
        initial.size,
        pointer(selected),
        evaluate,
        observe,
        cancellation,
        None,
        ct.byref(receipt),
        pointer(output),
        pointer(gradient),
        pointer(matrix),
        pointer(mode),
        pointer(curvatures),
        detail,
        len(detail),
    )
    if status == _native.STATUS_RESOURCE_LIMIT:
        raise MemoryError("native saddle storage exceeds maximum_numeric_bytes")
    _native.check(library, status)
    valid = bool(receipt.has_valid_state)
    return SaddleResult(
        coordinates=_readonly(output),
        gradient=_readonly(gradient) if valid else None,
        hessian=_readonly(matrix) if valid else None,
        target_mode=_readonly(mode) if valid else None,
        curvatures=_readonly(curvatures[: receipt.active_modes]) if valid else None,
        receipt=_receipt(receipt),
        detail=detail.value.decode("utf-8", errors="replace"),
        failures=tuple(failures),
        elapsed_seconds=perf_counter() - started,
    )
