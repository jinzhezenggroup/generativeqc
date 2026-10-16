"""Thin WB97M-V CUDA second-order frontend; all numerical work is linked C++/CUDA.

This module validates/marshals public inputs and publishes detached results.
It must not construct a KS snapshot, AO provider, Python response operator or
runtime compiler. CPU/development second-order implementations remain separate.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import json
import time
import typing
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from . import _native

if typing.TYPE_CHECKING:
    from .calculator import Atom, Calculator
    from .response_solver import GMRESOptions


class _Options(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_size_t),
        ("abi_version", ct.c_uint32),
        *(
            (name, ct.c_size_t)
            for name in (
                "device_bytes",
                "integral_bytes",
                "grid_bytes",
                "response_bytes",
                "output_bytes",
            )
        ),
        ("max_nonlocal_pairs", ct.c_uint64),
        ("restart", ct.c_size_t),
        ("max_iterations", ct.c_size_t),
        ("reorthogonalizations", ct.c_uint),
        ("relative_tolerance", ct.c_double),
        ("absolute_tolerance", ct.c_double),
    ]


class _Receipt(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_size_t),
        ("abi_version", ct.c_uint32),
        *(
            (name, ct.c_uint64)
            for name in (
                "device_bytes",
                "completed_directions",
                "induced_fock_actions",
                "exact_primitive_components",
                "one_electron_primitive_components",
                "df_value_passes",
                "df_contraction_passes",
                "grid_right_directions",
                "grid_mixed_contractions",
                "vv10_reference_pairs",
                "response_iterations",
                "response_workspace_bytes",
                "ecp_workspace_bytes",
                "df_workspace_bytes",
            )
        ),
        *(
            (name, ct.c_double)
            for name in (
                "response_rhs_norm",
                "response_residual_norm",
                "response_tolerance",
                "execution_seconds",
            )
        ),
    ]


@dataclass(frozen=True, eq=False)
class NativeKsHVPResult:
    """Immutable aggregate HVP; native intermediates/source splits stay on device."""

    direction: np.ndarray
    value: np.ndarray
    identity: str
    _diagnostics: typing.Mapping[str, typing.Any]

    @property
    def diagnostics(self) -> dict[str, typing.Any]:
        """Return detached scalar/work provenance, never live native resources."""
        return deepcopy(dict(self._diagnostics))


@dataclass(frozen=True, eq=False)
class NativeKsHessianResult:
    """Immutable raw row-major Hessian; no post-hoc symmetry repair."""

    matrix: np.ndarray
    identity: str
    _diagnostics: typing.Mapping[str, typing.Any]

    @property
    def diagnostics(self) -> dict[str, typing.Any]:
        """Return detached scalar/work provenance, never live native resources."""
        return deepcopy(dict(self._diagnostics))


def checked_direction(direction: typing.Any, atoms: int) -> np.ndarray:
    """Freeze finite real input before the SCF can release the Python GIL."""
    raw = np.asarray(direction)
    if raw.shape != (atoms, 3) or raw.dtype.kind not in "iuf":
        raise ValueError("direction must be a finite real array with shape (natoms, 3)")
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("direction must be finite and representable in FP64")
    return np.frombuffer(value.tobytes(), dtype=np.float64).reshape(value.shape)


def _capacity(value: typing.Any, name: str) -> int:
    if type(value) is not int or not 0 < value < 2**63:
        raise ValueError(f"{name} must be a positive int64 capacity")
    return value


def _options(
    integral_budget_bytes: int,
    grid_budget_bytes: int,
    output_budget_bytes: int,
    max_nonlocal_pairs: int,
    solver_options: GMRESOptions | None,
) -> _Options:
    options = _Options(
        struct_size=ct.sizeof(_Options),
        abi_version=_native.ABI_VERSION,
        device_bytes=2 << 30,
        integral_bytes=_capacity(integral_budget_bytes, "integral_budget_bytes"),
        grid_bytes=_capacity(grid_budget_bytes, "grid_budget_bytes"),
        output_bytes=_capacity(output_budget_bytes, "output_budget_bytes"),
        max_nonlocal_pairs=_capacity(max_nonlocal_pairs, "max_nonlocal_pairs"),
        response_bytes=64 << 20,
        restart=30,
        max_iterations=200,
        reorthogonalizations=2,
        relative_tolerance=1e-10,
        absolute_tolerance=0.0,
    )
    if solver_options is not None:
        from .response_solver import GMRESOptions

        if not isinstance(solver_options, GMRESOptions):
            raise TypeError("solver_options must be GMRESOptions or None")
        # Native GMRES checks every true residual and has its own breakdown
        # policy, without Python's heuristic stagnation exit or recycled bases.
        for name, default in (
            ("true_residual_every", 1),
            ("breakdown_tolerance", 1e-14),
            ("stagnation_window", 25),
            ("stagnation_tolerance", 1e-14),
        ):
            if getattr(solver_options, name) != default:
                raise NotImplementedError(f"native CUDA Hessian does not expose {name}")
        options.response_bytes = _capacity(
            solver_options.max_workspace_bytes, "max_workspace_bytes"
        )
        options.restart = _capacity(solver_options.restart, "restart")
        options.max_iterations = _capacity(
            solver_options.max_iterations, "max_iterations"
        )
        if not 0 < solver_options.reorthogonalize < 2**32:
            raise ValueError("reorthogonalize must fit a positive uint32")
        options.reorthogonalizations = solver_options.reorthogonalize
        options.relative_tolerance = solver_options.rtol
        options.absolute_tolerance = solver_options.atol
    return options


def run_native_cuda(
    calculator: Calculator,
    atoms: tuple[Atom, ...],
    *,
    charge: int,
    multiplicity: int,
    direction: np.ndarray | None = None,
    integral_budget_bytes: int = 64 << 20,
    grid_budget_bytes: int = 128 << 20,
    output_budget_bytes: int = 64 << 20,
    max_nonlocal_pairs: int = 16_777_216,
    solver_options: GMRESOptions | None = None,
) -> NativeKsHVPResult | NativeKsHessianResult:
    """Execute one accepted native batch under its context lock, then detach it.

    The AOT capability check precedes SCF preparation. Missing libraries or
    selector closure raise explicitly, with no Python/CuPy/JIT fallback. Only
    the final HVP/matrix and scalar receipt cross the numerical boundary.
    """
    started = time.perf_counter()
    if getattr(calculator, "_resource_budget", None) is not None:
        raise NotImplementedError(
            "native CUDA Hessian resources are not part of the global ResourceBudget plan; "
            "use the endpoint's explicit integral/grid/response/output limits"
        )
    options = _options(
        integral_budget_bytes,
        grid_budget_bytes,
        output_budget_bytes,
        max_nonlocal_pairs,
        solver_options,
    )
    coordinates = 3 * len(atoms)
    shape = (coordinates, coordinates) if direction is None else (len(atoms), 3)
    if 2 * int(np.prod(shape)) * 8 > options.output_bytes:
        raise ValueError(
            "native Hessian result publication exceeds output_budget_bytes"
        )
    library = calculator._library
    version = getattr(library, "generativeqc_ks_hessian_cuda_version_v1", None)
    execute = getattr(library, "generativeqc_ks_hessian_cuda_v1", None)
    if version is None or execute is None:
        raise NotImplementedError(
            "native CUDA Hessian bridge is unavailable; no Python/JIT fallback"
        )
    version.argtypes, version.restype = [], ct.c_uint32
    if version() != 1:
        raise NotImplementedError(
            "CUDA Hessian requires a linked build-time AOT package; no Python/JIT fallback"
        )
    pointer = ct.POINTER(ct.c_double)
    execute.argtypes = [
        ct.c_void_p,
        ct.c_size_t,
        ct.POINTER(_Options),
        pointer,
        ct.c_size_t,
        pointer,
        ct.c_size_t,
        ct.POINTER(_Receipt),
    ]
    execute.restype = ct.c_int
    output = np.empty(shape, dtype=np.float64)
    receipt = _Receipt(struct_size=ct.sizeof(_Receipt), abi_version=_native.ABI_VERSION)
    with calculator.prepare_batch(
        [atoms], charges=[charge], multiplicities=[multiplicity], warm_start=False
    ) as batch:
        batch.execute(strict=True, properties=("energy",))
        model_signature = batch._model_signature
        _native.check(
            library,
            execute(
                batch._batch,
                0,
                ct.byref(options),
                None if direction is None else direction.ctypes.data_as(pointer),
                0 if direction is None else direction.size,
                output.ctypes.data_as(pointer),
                output.size,
                ct.byref(receipt),
            ),
            context=batch._context,
        )
    value = np.frombuffer(output.tobytes(), dtype=np.float64).reshape(shape)
    diagnostics = {name: getattr(receipt, name) for name, _ in _Receipt._fields_[2:]}
    diagnostics.update(
        derivative_authority="compiler-ad",
        derivative_method="compiler-automatic-differentiation",
        execution_residency="device-resident",
        numerical_runtime="native-cpp-cuda",
        runtime_compilation=False,
        displaced_scf_evaluations=0,
        numerical_host_fallbacks=0,
        intermediate_panel_downloads=0,
        posthoc_symmetrization=False,
        response_algorithm="native-restarted-gmres",
        response_strategy="scalar",
        first_integral_backend="cuda",
        second_integral_backend="cuda",
        grid_ad_backend="cuda",
        response_backend="cuda",
        component_publication="aggregate-only",
        ecp_ad_backend="cuda" if receipt.ecp_workspace_bytes else None,
        df_integral_ad_backend="cuda" if receipt.df_workspace_bytes else None,
        endpoint_seconds=time.perf_counter() - started,
    )
    identity = hashlib.sha256(b"generativeqc.native-ks-second-order/v1")
    identity.update(
        json.dumps(
            {
                "model": model_signature,
                "atoms": [(atom.atomic_number, atom.position) for atom in atoms],
                "charge": charge,
                "multiplicity": multiplicity,
                "shape": shape,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    identity.update(memoryview(value).cast("B"))
    if direction is not None:
        identity.update(memoryview(direction).cast("B"))
        return NativeKsHVPResult(
            direction, value, identity.hexdigest(), MappingProxyType(diagnostics)
        )
    return NativeKsHessianResult(
        value, identity.hexdigest(), MappingProxyType(diagnostics)
    )
