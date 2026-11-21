"""Thin KS CUDA second-order frontend; all numerical work is linked C++/CUDA.

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
from .ks_diagnostics import (
    CUDA_M062X_STATIONARY_DOMAIN_VERSION,
    CUDA_MN15_STATIONARY_DOMAIN_VERSION,
    CUDA_R2SCAN_STATIONARY_DOMAIN,
    CUDA_R2SCAN_STATIONARY_DOMAIN_VERSION,
    CUDA_SPLIT_HYBRID_STATIONARY_DOMAINS,
)

if typing.TYPE_CHECKING:
    from .calculator import Atom, Calculator
    from .response_solver import GMRESOptions


def has_cuda_r2scan_stationary_domain(library: typing.Any) -> bool:
    """Require the variational point program and Hessian in the loaded package."""
    query = getattr(library, "generativeqc_ks_hessian_cuda_r2scan_domain_v1", None)
    if query is None:
        return False
    query.argtypes, query.restype = [], ct.c_uint32
    return query() == CUDA_R2SCAN_STATIONARY_DOMAIN_VERSION


def linked_cuda_split_hybrid_domain(library: typing.Any, method: str) -> str | None:
    """Require the linked energy/force/response domain, not just XC support."""
    key = {
        "M06-2X": ("m062x", CUDA_M062X_STATIONARY_DOMAIN_VERSION),
        "MN15": ("mn15", CUDA_MN15_STATIONARY_DOMAIN_VERSION),
    }.get(method)
    if key is None:
        return None
    tag, version = key
    query = getattr(library, f"generativeqc_ks_hessian_cuda_{tag}_domain_v1", None)
    if query is None:
        return None
    query.argtypes = []
    query.restype = ct.c_uint32
    return CUDA_SPLIT_HYBRID_STATIONARY_DOMAINS[version] if query() == version else None


def canonical_cuda_split_hybrid_domain(
    library: typing.Any, graph: typing.Any
) -> str | None:
    """Authorize the linked domain only for the exact physical MethodIR."""
    from generativeqc_compiler.method import resolve_method

    domain = linked_cuda_split_hybrid_domain(library, graph.identifier)
    if (
        domain
        and graph.identity == resolve_method(graph.identifier, spin=graph.spin).identity
    ):
        return domain
    return None


def cuda_hessian_bridge(library: typing.Any) -> typing.Any:
    """Resolve the linked ABI without initializing CUDA or preparing a molecule."""
    version = getattr(library, "generativeqc_ks_hessian_cuda_version_v2", None)
    execute = getattr(library, "generativeqc_ks_hessian_cuda_v2", None)
    if version is None or execute is None:
        raise NotImplementedError(
            "native CUDA Hessian bridge is unavailable; no Python/JIT fallback"
        )
    version.argtypes, version.restype = [], ct.c_uint32
    if version() != 2:
        raise NotImplementedError(
            "CUDA Hessian requires a linked build-time AOT package; no Python/JIT fallback"
        )
    return execute


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
        ("response_fp32_proposal", ct.c_uint32),
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
        *(
            (name, ct.c_uint64)
            for name in (
                "cpks_operator_bytes",
                "cpks_physical_checks",
                "cpks_cached_actions",
                "cpks_proposal_iterations",
                "cpks_proposal_seeds",
                "cpks_strict_iterations",
            )
        ),
    ]


@dataclass(frozen=True, eq=False)
class NativeKsHVPResult:
    """Immutable aggregate HVP; intermediates stay with their native owner."""

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
        # Match the native solver's absolute gate for symmetry-zero RHS columns.
        absolute_tolerance=1e-12,
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
    response_cache_precision: str = "fp64",
) -> NativeKsHVPResult | NativeKsHessianResult:
    """Execute one accepted native batch under its context lock, then detach it.

    The AOT capability check precedes SCF preparation. Missing libraries or
    selector closure raise explicitly, with no Python/CuPy/JIT fallback. Only
    the final HVP/matrix and scalar receipt cross the numerical boundary.
    """
    started = time.perf_counter()
    budget = getattr(calculator, "_resource_budget", None)
    options = _options(
        integral_budget_bytes,
        grid_budget_bytes,
        output_budget_bytes,
        max_nonlocal_pairs,
        solver_options,
    )
    if response_cache_precision not in ("fp64", "fp32-proposal"):
        raise ValueError("response_cache_precision must be 'fp64' or 'fp32-proposal'")
    if direction is not None and response_cache_precision != "fp64":
        raise ValueError("fp32-proposal response cache requires a dense Hessian")
    options.response_fp32_proposal = response_cache_precision == "fp32-proposal"
    coordinates = 3 * len(atoms)
    shape = (coordinates, coordinates) if direction is None else (len(atoms), 3)
    if 2 * int(np.prod(shape)) * 8 > options.output_bytes:
        raise ValueError(
            "native Hessian result publication exceeds output_budget_bytes"
        )
    library = calculator._library
    execute = cuda_hessian_bridge(library)
    batch_controls = {}
    if budget is not None:
        from ._hessian_resources import plan_cuda_hessian

        resource_version = getattr(
            library, "generativeqc_ks_hessian_cuda_resource_version_v1", None
        )
        if resource_version is None:
            raise NotImplementedError(
                "native CUDA Hessian lacks shared ResourceBudget admission"
            )
        resource_version.argtypes, resource_version.restype = [], ct.c_uint32
        if resource_version() != 1:
            raise NotImplementedError(
                "native CUDA Hessian lacks shared ResourceBudget admission"
            )
        plan, request, capacity = plan_cuda_hessian(
            calculator,
            atoms,
            charge=charge,
            multiplicity=multiplicity,
            dense=direction is None,
            budget=budget,
            options=options,
        )
        options.device_bytes = capacity
        batch_controls.update(resource_plan=plan, _hessian_request=request)
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
    scf_domain = None
    resource_diagnostics = None
    with calculator.prepare_batch(
        [atoms],
        charges=[charge],
        multiplicities=[multiplicity],
        warm_start=False,
        **batch_controls,
    ) as batch:
        # Compatibility corrections live outside the native SCF handle. The
        # linked Hessian could otherwise return only the electronic response.
        if (
            calculator._dispersion_method_ir is not None
            and batch._native_dispersion is None
        ):
            raise NotImplementedError(
                "CUDA Hessian requires dispersion owned by the native KS batch"
            )
        solved = batch.execute(strict=True, properties=("energy",))
        scf_precision = getattr(solved.items[0], "precision", None)
        required_domain = getattr(calculator, "_cuda_split_hybrid_stationary", None)
        if getattr(calculator, "_cuda_r2scan_stationary", False):
            required_domain = ("r2SCAN", CUDA_R2SCAN_STATIONARY_DOMAIN)
        if required_domain is not None:
            method_name, expected_domain = required_domain
            diagnostic = solved.items[0].ks_diagnostic
            if diagnostic is None or diagnostic.scf_domain != expected_domain:
                raise RuntimeError(
                    f"CUDA {method_name} Hessian requires the variational SCF domain"
                )
            scf_domain = diagnostic.scf_domain
        model_signature = batch._model_signature

        def invoke() -> int:
            return execute(
                batch._batch,
                0,
                ct.byref(options),
                None if direction is None else direction.ctypes.data_as(pointer),
                0 if direction is None else direction.size,
                output.ctypes.data_as(pointer),
                output.size,
                ct.byref(receipt),
            )

        if budget is None:
            _native.check(library, invoke(), context=batch._context)
        else:
            from ._hessian_resources import HESSIAN_RESOURCE_OWNER
            from .resources_native import check_resource_status, observe_method_call

            ledger = batch._resource_ledger
            if ledger is None or ledger.prepared_call is None:
                raise RuntimeError(
                    "CUDA Hessian resource plan has no native device ledger"
                )
            status, resource_diagnostics = observe_method_call(
                library,
                batch.resource_plan,
                ledger,
                invoke,
                owner=HESSIAN_RESOURCE_OWNER,
                phase="hessian_response",
                previous=batch.resource_diagnostics,
                native_managed=True,
            )
            check_resource_status(
                library, status, resource_diagnostics, context=batch._context
            )
    value = np.frombuffer(output.tobytes(), dtype=np.float64).reshape(shape)
    diagnostics = {name: getattr(receipt, name) for name, _ in _Receipt._fields_[2:]}
    if scf_precision is not None:
        diagnostics["scf_requested_precision"] = scf_precision["requested_mode"]
        diagnostics["scf_mixed_fock_builds"] = scf_precision["mixed_stage_fock_builds"]
    if scf_domain is not None:
        diagnostics["scf_domain"] = scf_domain
    if resource_diagnostics is not None:
        diagnostics["resources"] = resource_diagnostics
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
        cpks_cache_precision=response_cache_precision
        if receipt.cpks_operator_bytes
        else None,
        component_publication="aggregate-only",
        ecp_ad_backend="cuda" if receipt.ecp_workspace_bytes else None,
        df_integral_ad_backend="cuda" if receipt.df_workspace_bytes else None,
        endpoint_seconds=time.perf_counter() - started,
    )
    identity = hashlib.sha256(b"generativeqc.native-ks-second-order/v1")
    identity.update(
        json.dumps(
            {
                "model": model_signature
                if scf_domain is None
                else (model_signature, scf_domain),
                "atoms": [(atom.atomic_number, atom.position) for atom in atoms],
                "charge": int(charge),
                "multiplicity": int(multiplicity),
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
