"""Marshal linked CPU stationary derivatives without a Python numerical owner.

Only detached results and scalar receipts cross this bridge. Native code borrows
the accepted batch's actual sources, solves coupled spin response and composes
the native geometry-only correction exactly once. No AO reconstruction, runtime
compiler, displaced production SCF or reference electronic engine is used here.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import json
import math
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from . import _native
from ._native_hessian import NativeKsHessianResult, NativeKsHVPResult, _capacity

if TYPE_CHECKING:
    from ._api_types import Atom
    from .batch import PreparedBatch
    from .calculator import Calculator
    from .response_solver import GMRESOptions


class _Options(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_size_t),
        ("abi_version", ct.c_uint32),
        ("expected_atoms", ct.POINTER(_native.AtomDescriptor)),
        ("expected_atom_count", ct.c_size_t),
        ("expected_charge", ct.c_int32),
        ("expected_multiplicity", ct.c_int32),
        *(
            (name, ct.c_size_t)
            for name in (
                "numeric_bytes",
                "integral_bytes",
                "grid_bytes",
                "response_bytes",
                "ecp_bytes",
                "dispersion_bytes",
                "output_bytes",
                "max_eri_primitives",
                "max_map_terms",
                "max_grid_visits",
                "max_d3_pairs",
                "restart",
                "max_iterations",
                "gmres_bytes",
                "true_residual_every",
                "stagnation_window",
            )
        ),
        ("reorthogonalizations", ct.c_uint),
        *(
            (name, ct.c_double)
            for name in (
                "relative_tolerance",
                "absolute_tolerance",
                "breakdown_tolerance",
                "stagnation_tolerance",
                "symmetry_absolute_tolerance",
                "symmetry_relative_tolerance",
            )
        ),
    ]


class _Receipt(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_size_t),
        ("abi_version", ct.c_uint32),
        *(
            (name, ct.c_uint64)
            for name in (
                "numeric_reservation_bytes",
                "frame_bytes",
                "geometry_bytes",
                "output_bytes",
                "fixed_bound_bytes",
                "gmres_workspace_bytes",
                "completed_directions",
                "response_iterations",
                "induced_fock_actions",
                "first_eri_primitives",
                "curvature_eri_primitives",
                "first_one_electron_primitives",
                "curvature_one_electron_primitives",
                "source_xc_point_visits",
                "curvature_xc_point_visits",
                "response_xc_point_visits",
                "ecp_kernel_blocks",
                "ecp_radial_layers",
                "ecp_ao_primitive_points",
                "ecp_contraction_terms",
                "d3_pair_visits",
                "d3_reference_terms",
            )
        ),
        *(
            (name, ct.c_double)
            for name in (
                "response_residual_norm",
                "response_relative_residual",
                "electronic_energy",
                "total_energy",
                "d3_energy",
                "source_energy_error",
                "raw_symmetry_error",
                "execution_seconds",
            )
        ),
    ]


class _OrbitalsReceipt(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_size_t),
        ("abi_version", ct.c_uint32),
        *(
            (name, ct.c_uint64)
            for name in (
                "basis",
                "spins",
                "alpha_occupied",
                "beta_occupied",
                "output_elements",
                "numeric_reservation_bytes",
            )
        ),
    ]


@dataclass(frozen=True)
class _OrbitalIdentity:
    spin: str


@dataclass(frozen=True)
class _OrbitalFrame:
    """Detached canonical C and integer occupations, not a response snapshot."""

    identity: _OrbitalIdentity
    coefficients: np.ndarray
    occupations: np.ndarray
    numeric_reservation_bytes: int


def _options(
    integral_budget_bytes: int,
    grid_budget_bytes: int,
    output_budget_bytes: int,
    dispersion_budget_bytes: int,
    solver_options: GMRESOptions | None,
    max_eri_primitives: int | None = None,
) -> _Options:
    """Keep fixed response admission distinct from the solver workspace limit.

    The explicit default absolute gate handles symmetry-zero right-hand sides;
    cancellation roundoff cannot satisfy a relative-only tolerance there. An
    explicitly supplied GMRESOptions retains its original absolute tolerance.
    """
    from .response_solver import GMRESOptions

    solver = GMRESOptions(atol=1e-12) if solver_options is None else solver_options
    if not isinstance(solver, GMRESOptions):
        raise TypeError("solver_options must be GMRESOptions or None")
    if not 0 < solver.reorthogonalize < 2**32:
        raise ValueError("reorthogonalize must fit a positive uint32")
    if not (solver.rtol > 0 or solver.atol > 0):
        raise ValueError("native CPU response requires a positive residual tolerance")
    return _Options(
        struct_size=ct.sizeof(_Options),
        abi_version=_native.ABI_VERSION,
        numeric_bytes=512 << 20,
        integral_bytes=_capacity(integral_budget_bytes, "integral_budget_bytes"),
        grid_bytes=_capacity(grid_budget_bytes, "grid_budget_bytes"),
        response_bytes=128 << 20,
        ecp_bytes=128 << 20,
        dispersion_bytes=_capacity(
            dispersion_budget_bytes, "dispersion_memory_budget_bytes"
        ),
        output_bytes=_capacity(output_budget_bytes, "output_budget_bytes"),
        max_eri_primitives=_capacity(
            100_000_000 if max_eri_primitives is None else max_eri_primitives,
            "max_eri_primitives",
        ),
        max_map_terms=1 << 40,
        max_grid_visits=1 << 40,
        max_d3_pairs=16_777_216,
        restart=_capacity(solver.restart, "restart"),
        max_iterations=_capacity(solver.max_iterations, "max_iterations"),
        gmres_bytes=_capacity(solver.max_workspace_bytes, "max_workspace_bytes"),
        true_residual_every=_capacity(
            solver.true_residual_every, "true_residual_every"
        ),
        stagnation_window=_capacity(solver.stagnation_window, "stagnation_window"),
        reorthogonalizations=solver.reorthogonalize,
        relative_tolerance=solver.rtol,
        absolute_tolerance=solver.atol,
        breakdown_tolerance=solver.breakdown_tolerance,
        stagnation_tolerance=solver.stagnation_tolerance,
        symmetry_absolute_tolerance=1e-7,
        symmetry_relative_tolerance=1e-7,
    )


def preflight(calculator: Calculator) -> None:
    """Authenticate linked electronic and complete correction selectors before SCF."""
    if calculator._device_name != "cpu":
        raise NotImplementedError("native CPU Hessian requires a CPU prepared owner")
    if calculator._resource_budget is not None:
        raise NotImplementedError(
            "native CPU Hessian uses separate derivative reservations, not ResourceBudget"
        )
    library = calculator._library
    version = getattr(library, "generativeqc_ks_hessian_cpu_version_v1", None)
    query = getattr(library, "generativeqc_ks_hessian_cpu_method_v1", None)
    if (
        version is None
        or query is None
        or not hasattr(library, "generativeqc_ks_hessian_cpu_v1")
    ):
        raise NotImplementedError(
            "native CPU Hessian bridge is unavailable; no Python/JIT fallback"
        )
    version.argtypes, version.restype = [], ct.c_uint32
    query.argtypes, query.restype = [ct.c_char_p], ct.c_uint32
    if (
        version() != 1
        or query(calculator.ks_options.method_ir.identity.encode("ascii")) != 1
    ):
        raise NotImplementedError(
            "native CPU Hessian has no exact linked electronic MethodIR; no Python/JIT fallback"
        )
    if calculator._dispersion_method_ir is not None:
        from ._named_ks import select_named_program

        if (
            select_named_program(
                library, calculator.method_ir, _native.BACKEND_CPU_REFERENCE
            )
            is None
        ):
            raise NotImplementedError(
                "native CPU Hessian requires the exact total native correction program; "
                "electronic-only publication is forbidden"
            )


def _bind_batch(
    calculator: Calculator,
    batch: PreparedBatch,
    atoms: tuple[Atom, ...],
    charge: int,
    multiplicity: int,
    index: int,
    options: _Options,
) -> None:
    """Bind presentation geometry to the actual live native producer."""
    preflight(calculator)
    batch._ensure_open()
    if (
        batch._calculator is not calculator
        or batch._model_signature != calculator._model_signature()
    ):
        raise ValueError("native CPU Hessian calculator/batch model changed")
    if type(index) is not int or not 0 <= index < len(batch._atom_counts):
        raise ValueError("native CPU Hessian batch index is invalid")
    if len(atoms) != batch._atom_counts[index]:
        raise ValueError("native CPU Hessian geometry shape disagrees with its batch")
    if type(charge) is not int or not -(1 << 31) <= charge < 1 << 31:
        raise ValueError("native CPU Hessian charge must fit int32")
    if type(multiplicity) is not int or not 0 < multiplicity < 1 << 31:
        raise ValueError("native CPU Hessian multiplicity must fit positive int32")
    if (
        calculator._dispersion_method_ir is not None
        and batch._native_dispersion is None
    ):
        raise NotImplementedError(
            "native CPU Hessian refuses a standalone frontend correction"
        )
    options.expected_atoms = (_native.AtomDescriptor * len(atoms))(
        *(_native.AtomDescriptor(atom.atomic_number, *atom.position) for atom in atoms)
    )
    options.expected_atom_count = len(atoms)
    options.expected_charge = charge
    options.expected_multiplicity = multiplicity


def orbital_frame(
    calculator: Calculator,
    batch: PreparedBatch,
    atoms: tuple[Atom, ...],
    *,
    charge: int,
    multiplicity: int,
    maximum_bytes: int,
    index: int = 0,
) -> _OrbitalFrame:
    """Export verified native orbitals for root continuity without AO/grid/JIT.

    Shape and numerical reservation are admitted before a Python output panel or
    a native verified frame is allocated. The second call authenticates the
    producer and expected geometry again; a failed export publishes no C.
    This budget does not include eigensolver scratch or detached Python copies.
    """
    options = _options(64 << 20, 128 << 20, 64 << 20, 64 << 20, None)
    options.numeric_bytes = _capacity(maximum_bytes, "root_overlap_budget_bytes")
    _bind_batch(calculator, batch, atoms, charge, multiplicity, index, options)
    export = getattr(calculator._library, "generativeqc_ks_orbitals_cpu_v1", None)
    if export is None:
        raise NotImplementedError("native CPU occupied-root bridge is unavailable")
    pointer = ct.POINTER(ct.c_double)
    export.argtypes = [
        ct.c_void_p,
        ct.c_size_t,
        ct.POINTER(_Options),
        pointer,
        ct.c_size_t,
        ct.POINTER(_OrbitalsReceipt),
    ]
    export.restype = ct.c_int
    receipt = _OrbitalsReceipt(
        struct_size=ct.sizeof(_OrbitalsReceipt), abi_version=_native.ABI_VERSION
    )

    def read(output: np.ndarray | None) -> None:
        _native.check(
            calculator._library,
            export(
                batch._batch,
                index,
                ct.byref(options),
                None if output is None else output.ctypes.data_as(pointer),
                0 if output is None else output.size,
                ct.byref(receipt),
            ),
            context=batch._context,
        )

    read(None)
    shape = (receipt.spins, receipt.basis, receipt.basis)
    if (
        receipt.spins
        != (1 if calculator.ks_options.method_ir.spin == "unpolarized" else 2)
        or not receipt.basis
        or receipt.output_elements != math.prod(shape)
        or receipt.numeric_reservation_bytes > maximum_bytes
        or receipt.alpha_occupied > receipt.basis
        or receipt.beta_occupied > receipt.basis
    ):
        raise RuntimeError("native CPU occupied-root shape/receipt is invalid")
    planned = bytes(receipt)
    output = np.empty(shape, dtype=np.float64)
    read(output)
    if bytes(receipt) != planned:
        raise RuntimeError("native CPU occupied-root shape changed before publication")
    coefficients = np.frombuffer(output.tobytes(), dtype=np.float64).reshape(shape)
    occupations = np.zeros(shape[:2], dtype=np.float64)
    occupations[0, : receipt.alpha_occupied] = 2.0 if receipt.spins == 1 else 1.0
    if receipt.spins == 2:
        occupations[1, : receipt.beta_occupied] = 1.0
    occupations = np.frombuffer(occupations.tobytes(), dtype=np.float64).reshape(
        shape[:2]
    )
    return _OrbitalFrame(
        _OrbitalIdentity(calculator.ks_options.method_ir.spin),
        coefficients,
        occupations,
        receipt.numeric_reservation_bytes,
    )


def execute_batch(
    calculator: Calculator,
    batch: PreparedBatch,
    atoms: tuple[Atom, ...],
    *,
    charge: int,
    multiplicity: int,
    direction: np.ndarray | None = None,
    integral_budget_bytes: int = 64 << 20,
    grid_budget_bytes: int = 128 << 20,
    output_budget_bytes: int = 64 << 20,
    solver_options: GMRESOptions | None = None,
    max_eri_primitives: int | None = None,
    index: int = 0,
) -> NativeKsHVPResult | NativeKsHessianResult:
    """Consume the current accepted native state without starting another SCF.

    This synchronous helper retains the batch through publication. A geometry
    optimizer may reuse its primal owner, but neither a changed calculator nor a
    standalone frontend correction authorizes a native total-method result.
    """
    options = _options(
        integral_budget_bytes,
        grid_budget_bytes,
        output_budget_bytes,
        calculator._dispersion_memory_budget_bytes,
        solver_options,
        max_eri_primitives,
    )
    _bind_batch(calculator, batch, atoms, charge, multiplicity, index, options)
    coordinates = 3 * len(atoms)
    shape = (coordinates, coordinates) if direction is None else (len(atoms), 3)
    if direction is not None:
        from ._native_hessian import checked_direction

        direction = checked_direction(direction, len(atoms))
    panel_bytes = 8 * (2 * int(np.prod(shape)) + 3 * coordinates + 2)
    if panel_bytes > options.output_bytes:
        raise ValueError("native CPU result/staging exceeds output_budget_bytes")
    pointer = ct.POINTER(ct.c_double)
    library = calculator._library
    execute = library.generativeqc_ks_hessian_cpu_v1
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
    _native.check(
        library,
        execute(
            batch._batch,
            index,
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
    from .profiles import canonical_hash

    sources = ("nuclear_integrals", "moving_grid_xc", "scalar_ecp", "nuclear_repulsion")
    if calculator._d3_second_order_spec is not None:
        sources = (*sources, "d3_bj_two_body")
    diagnostics.update(
        derivative_authority="compiler-ad",
        numerical_runtime="native-cpp-cpu",
        runtime_compilation=False,
        displaced_scf_evaluations=0,
        reference_engine_fallbacks=0,
        posthoc_symmetrization=False,
        response_algorithm="native-restarted-gmres",
        response_strategy="scalar",
        numerical_reservation_scope="derivative-owned; excludes primal/export-scratch/BLAS/RSS",
        component_publication="aggregate-with-scalar-energy-receipt",
        nuclear_response_solves=receipt.completed_directions,
        response_absolute_tolerance=options.absolute_tolerance,
        response_relative_tolerance=options.relative_tolerance,
        source_names=sources,
        source_order=sources,
        complete_source_coverage=True,
        public_calculator_endpoint=False,
        max_eri_primitives=options.max_eri_primitives,
        spin=calculator.ks_options.method_ir.spin,
        geometry_identity=canonical_hash(
            {
                "schema": "generativeqc.stationary-geometry/v1",
                "atoms": [
                    [atom.atomic_number, *map(float, atom.position)] for atom in atoms
                ],
                "units": "Bohr",
            }
        ),
        electronic_method_identity=calculator.ks_options.method_ir.identity,
        total_method_identity=calculator.method_ir.identity,
        total_method_manifest_identity=calculator.method_ir.manifest_identity,
        dispersion_identity=(
            None
            if calculator._d3_second_order_spec is None
            else calculator._d3_second_order_spec.identity
        ),
    )
    if calculator._d3_second_order_spec is not None:
        diagnostics["dispersion"] = {
            "electronic_response_solves": 0,
            "pair_visits": receipt.d3_pair_visits,
            "reference_terms": receipt.d3_reference_terms,
            "energy": receipt.d3_energy,
            "identity": calculator._d3_second_order_spec.identity,
        }
    identity = hashlib.sha256(b"generativeqc.native-cpu-second-order/v1")
    identity.update(
        json.dumps(
            {
                "model": batch._model_signature,
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


def run_native_cpu(
    calculator: Calculator,
    atoms: tuple[Atom, ...],
    *,
    charge: int,
    multiplicity: int,
    direction: np.ndarray | None = None,
    integral_budget_bytes: int = 64 << 20,
    grid_budget_bytes: int = 128 << 20,
    output_budget_bytes: int = 64 << 20,
    solver_options: GMRESOptions | None = None,
    max_eri_primitives: int | None = None,
) -> NativeKsHVPResult | NativeKsHessianResult:
    """Run one native primal, consume it synchronously, and detach the result."""
    from dataclasses import replace

    started = time.perf_counter()
    preflight(calculator)
    options = _options(
        integral_budget_bytes,
        grid_budget_bytes,
        output_budget_bytes,
        calculator._dispersion_memory_budget_bytes,
        solver_options,
        max_eri_primitives,
    )
    coordinates = 3 * len(atoms)
    elements = coordinates * coordinates if direction is None else coordinates
    if 8 * (2 * elements + 3 * coordinates + 2) > options.output_bytes:
        raise ValueError("native CPU result/staging exceeds output_budget_bytes")
    with calculator.prepare_batch(
        [atoms], charges=[charge], multiplicities=[multiplicity], warm_start=False
    ) as batch:
        primal = batch.execute(strict=True, properties=("energy",)).items[0]
        result = execute_batch(
            calculator,
            batch,
            atoms,
            charge=charge,
            multiplicity=multiplicity,
            direction=direction,
            integral_budget_bytes=integral_budget_bytes,
            grid_budget_bytes=grid_budget_bytes,
            output_budget_bytes=output_budget_bytes,
            solver_options=solver_options,
            max_eri_primitives=max_eri_primitives,
        )
    return replace(
        result,
        _diagnostics=MappingProxyType(
            {
                **result.diagnostics,
                "endpoint_seconds": time.perf_counter() - started,
                "public_calculator_endpoint": True,
                "scf_iterations": primal.iterations,
                "scf_fock_builds": primal.fock_builds,
                "scf_executed_backend": primal.executed_backend,
            }
        ),
    )
