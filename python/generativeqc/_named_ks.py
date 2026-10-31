"""Presentation of complete native KS programs and their correction receipts.

This adapter performs no energy/gradient arithmetic. The native prepared KS
owner executes and combines all terms before publishing its result.
"""

from __future__ import annotations

import ctypes as ct
from typing import Any

import numpy as np

from . import _native


class CompositeCorrectionResult(ct.Structure):
    """Borrowed output buffers for the native complete correction receipt."""

    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("abi_version", ct.c_uint32),
        ("energy", ct.c_double),
        ("gcp_energy", ct.c_double),
        ("gradient", ct.POINTER(ct.c_double)),
        ("gradient_count", ct.c_uint64),
        ("gcp_gradient", ct.POINTER(ct.c_double)),
        ("gcp_gradient_count", ct.c_uint64),
    ]


def select_named_program(library: Any, graph: Any, backend: int) -> bytes | None:
    """Discover an exact full graph; missing symbols/catalog entries retain compatibility."""
    query = getattr(library, "generativeqc_ks_catalog_name_v1", None)
    if query is None or not hasattr(library, "generativeqc_batch_prepare_named_ks_v1"):
        return None
    query.argtypes = [ct.c_char_p, ct.c_int32, ct.POINTER(ct.c_char_p)]
    query.restype = ct.c_int
    selected = ct.c_char_p()
    _native.check(
        library, query(graph.identity.encode("ascii"), backend, ct.byref(selected))
    )
    return selected.value


def select_resource_program(calculator: Any, electronic_request: Any) -> bytes | None:
    """Select a complete program only with matching correction capacity ownership.

    Missing additive resource support retains the standalone compatibility
    owner. After positive selection, query/preparation errors are authoritative.
    This does not extend the global planner's electronic or correction domains.
    CUDA additionally needs the existing native stationary-capacity receipt:
    compatibility forces would otherwise allocate a second correction owner.
    """
    from generativeqc_compiler.method import (
        D3Spec,
        D4Spec,
        DispersionCorrectionPrimitive,
        GeometricCounterpoisePrimitive,
    )

    graph = calculator._dispersion_method_ir
    if graph is None or any(
        isinstance(node, GeometricCounterpoisePrimitive) for node in graph.primitives
    ):
        return None
    corrections = [
        node.specification
        for node in graph.primitives
        if isinstance(node, DispersionCorrectionPrimitive)
    ]
    if len(corrections) != 1 or not isinstance(corrections[0], (D3Spec, D4Spec)):
        return None
    family = "d4" if isinstance(corrections[0], D4Spec) else "d3"
    library = calculator._library
    if (
        calculator._dispersion_method_ir is None
        or calculator._precision_mode != _native.PRECISION_FP64
        or not hasattr(library, f"generativeqc_ks_{family}_resource_query_v1")
        or (
            calculator._device_name == "cuda"
            and (
                not hasattr(
                    library,
                    "generativeqc_batch_prepare_named_ks_with_device_resources_v1",
                )
                or not electronic_request.candidates
                or any(
                    "cuda_stationary_cache" not in dict(candidate.decisions)
                    for candidate in electronic_request.candidates
                )
            )
        )
    ):
        return None
    return select_named_program(
        library,
        calculator._dispersion_method_ir,
        _native.BACKEND_CUDA
        if calculator._device_name == "cuda"
        else _native.BACKEND_CPU_REFERENCE,
    )


def create_named_system(
    library: Any,
    context: Any,
    atoms: Any,
    charge: int,
    multiplicity: int,
    name: bytes,
    graph: Any,
) -> Any:
    """Let the native catalog materialize the method's defining orbital template.

    Callers validate scalar widths and the exact offline basis snapshot first.
    A selected native refusal propagates without retrying Python expansion.
    """
    atom_array = (_native.AtomDescriptor * len(atoms))(
        *(_native.AtomDescriptor(atom.atomic_number, *atom.position) for atom in atoms)
    )
    descriptor = _native.SystemDescriptor(
        ct.sizeof(_native.SystemDescriptor),
        _native.ABI_VERSION,
        atom_array,
        len(atoms),
        None,
        0,
        None,
        0,
        charge,
        multiplicity,
        _native.BASIS_SPHERICAL,
    )
    named = _native.NamedKsDescriptor(
        ct.sizeof(_native.NamedKsDescriptor),
        _native.ABI_VERSION,
        name,
        graph.identity.encode("ascii"),
        0,
    )
    create = library.generativeqc_system_create_named_ks_v1
    create.argtypes = [
        ct.c_void_p,
        ct.POINTER(_native.SystemDescriptor),
        ct.POINTER(_native.NamedKsDescriptor),
        ct.POINTER(ct.c_void_p),
    ]
    create.restype = ct.c_int
    system = ct.c_void_p()
    _native.check(
        library,
        create(context, ct.byref(descriptor), ct.byref(named), ct.byref(system)),
        context=context,
    )
    return system


def prepare_named_batch(
    library: Any,
    context: Any,
    systems: Any,
    count: int,
    method: Any,
    flags: int,
    output: Any,
    name: bytes,
    graph: Any,
    maximum_bytes: int,
    resources: Any = None,
) -> int:
    """Pass controls plus a pinned full identity to the selected library."""
    prepare = (
        library.generativeqc_batch_prepare_named_ks_with_device_resources_v1
        if resources is not None
        else library.generativeqc_batch_prepare_named_ks_v1
    )
    prepare.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_void_p),
        ct.c_uint32,
        ct.POINTER(_native.MethodDescriptor),
        ct.POINTER(_native.NamedKsDescriptor),
        ct.c_uint32,
    ]
    if resources is not None:
        prepare.argtypes += [ct.c_void_p]
    prepare.argtypes += [ct.POINTER(ct.c_void_p)]
    prepare.restype = ct.c_int
    named = _native.NamedKsDescriptor(
        ct.sizeof(_native.NamedKsDescriptor),
        _native.ABI_VERSION,
        name,
        graph.identity.encode("ascii"),
        maximum_bytes,
    )
    # The complete catalog graph owns its spin. PBE here is only the existing
    # ABI transport carrier, including when the public alias is legacy PBE-D4.
    method.method = (
        _native.METHOD_PBE_RKS
        if graph.spin == "unpolarized"
        else _native.METHOD_PBE_UKS
    )
    arguments = [context, systems, count, ct.byref(method), ct.byref(named), flags]
    if resources is not None:
        arguments.append(resources)
    return prepare(*arguments, output)


class NativeKsDispersion:
    """Borrow correction receipts from a live KS batch; owns no numerical plan."""

    def __init__(self, batch: Any, graph: Any) -> None:
        from generativeqc_compiler.method import (
            D4Spec,
            DispersionCorrectionPrimitive,
            GeometricCounterpoisePrimitive,
        )

        self.batch, self.graph = batch, graph
        self.spec = next(
            node.specification
            for node in graph.primitives
            if isinstance(node, DispersionCorrectionPrimitive)
        )
        self.d4 = isinstance(self.spec, D4Spec)
        self.gcp = next(
            (
                node
                for node in graph.primitives
                if isinstance(node, GeometricCounterpoisePrimitive)
            ),
            None,
        )
        self.library = batch._library

    def _identity(self, suffix: str) -> str:
        family = "d4" if self.d4 else "d3"
        query = getattr(self.library, f"generativeqc_{family}_{suffix}")
        query.argtypes, query.restype = [], ct.c_char_p
        value = query()
        if not value:
            raise RuntimeError("native dispersion identity is unavailable")
        return value.decode("ascii")

    def compatibility_owner(self) -> Any:
        """Retain the established correction route for unqualified electronic forces.

        Only those force consumers still assemble outside the native endpoint;
        their correction energy must never be added to the already complete energy.
        """
        from .dispersion import (
            D3CorrectionBatch,
            D4CorrectionBatch,
            R2SCAN3CCorrectionBatch,
        )

        systems = []
        for index, atoms in enumerate(self.batch._systems):
            item = (
                self.batch._atomic_numbers[index],
                [atom.position for atom in atoms],
            )
            systems.append((*item, self.batch._charges[index]) if self.d4 else item)
        calculator = self.batch._calculator
        owner = (
            R2SCAN3CCorrectionBatch
            if self.gcp
            else D4CorrectionBatch
            if self.d4
            else D3CorrectionBatch
        )
        return owner(
            self.graph,
            systems,
            device=calculator._device_name,
            device_id=calculator._device_id,
            maximum_bytes=calculator._dispersion_memory_budget_bytes,
        )

    def result(self, index: int, gradients: bool) -> Any:
        """Copy published correction data; failed/rejected items have no receipt."""
        from .dispersion import D3CorrectionResult, D4CorrectionResult

        atoms = self.batch._atom_counts[index]
        gradient = np.empty((atoms, 3), dtype=np.float64) if gradients else None
        charges = np.empty(atoms, dtype=np.float64) if self.d4 else None
        pointer = ct.POINTER(ct.c_double)
        native = _native.D4BatchItemResultDescriptor(
            struct_size=ct.sizeof(_native.D4BatchItemResultDescriptor),
            abi_version=_native.ABI_VERSION,
            gradient=gradient.ctypes.data_as(pointer) if gradient is not None else None,
            gradient_count=3 * atoms if gradients else 0,
            charges=charges.ctypes.data_as(pointer) if charges is not None else None,
            charge_count=atoms if self.d4 else 0,
        )
        query = self.library.generativeqc_ks_batch_dispersion_result_v1
        query.argtypes = [ct.c_void_p, ct.c_uint32, ct.POINTER(type(native))]
        query.restype = ct.c_int
        _native.check(
            self.library,
            query(self.batch._batch, index, ct.byref(native)),
            context=self.batch._context,
        )
        backend = self.batch._calculator._device_name
        if self.d4:
            result = D4CorrectionResult(
                0,
                native.energy,
                native.two_body_energy,
                native.atm_energy,
                gradient,
                charges,
                backend,
                "success",
            )
            return (
                self._composite_result(index, gradients, result) if self.gcp else result
            )
        query = self.library.generativeqc_ks_batch_dispersion_variant_v1
        query.argtypes, query.restype = [ct.c_void_p, ct.c_uint32], ct.c_char_p
        variant = query(self.batch._batch, index)
        if not variant:
            raise RuntimeError("native D3 receipt omitted its variant identity")
        return D3CorrectionResult(
            0,
            native.energy,
            gradient,
            backend,
            "success",
            self.spec.identity,
            variant.decode("ascii"),
            self._identity("provider_identity"),
            self._identity("scheduler_identity"),
        )

    def _gcp_identities(self) -> tuple[str, str]:
        """Present component provenance without constructing a correction owner."""
        from generativeqc_compiler.common.provenance import canonical_hash

        query = self.library.generativeqc_r2scan3c_gcp_provider_identity
        query.argtypes, query.restype = [], ct.c_char_p
        provider = query().decode("ascii")
        if self.batch._calculator._device_name == "cuda":
            provider = "r2scan3c-gcp-cuda-v1"
        return provider, canonical_hash(self.gcp.semantic_payload())

    def _composite_result(self, index: int, gradients: bool, d4: Any) -> Any:
        """Copy the already combined result and gCP term, with no Python arithmetic."""
        from .dispersion import GCPCorrectionResult, R2SCAN3CCorrectionResult

        atoms = self.batch._atom_counts[index]
        total = np.empty((atoms, 3)) if gradients else None
        gcp = np.empty((atoms, 3)) if gradients else None
        pointer = ct.POINTER(ct.c_double)
        output = CompositeCorrectionResult(
            ct.sizeof(CompositeCorrectionResult),
            _native.ABI_VERSION,
            0,
            0,
            total.ctypes.data_as(pointer) if gradients else None,
            3 * atoms if gradients else 0,
            gcp.ctypes.data_as(pointer) if gradients else None,
            3 * atoms if gradients else 0,
        )
        query = self.library.generativeqc_ks_batch_composite_correction_result_v1
        query.argtypes, query.restype = (
            [ct.c_void_p, ct.c_uint32, ct.POINTER(type(output))],
            ct.c_int,
        )
        _native.check(
            self.library,
            query(self.batch._batch, index, ct.byref(output)),
            context=self.batch._context,
        )
        component = GCPCorrectionResult(
            0,
            output.gcp_energy,
            gcp,
            self.batch._calculator._device_name,
            "success",
            *self._gcp_identities(),
        )
        return R2SCAN3CCorrectionResult(
            0,
            output.energy,
            total,
            self.batch._calculator._device_name,
            "success",
            d4,
            component,
        )

    def diagnostic(self) -> Any:
        """Aggregate current per-system owners, including duplicated retained tables.

        Native KS rebuilds an owner on geometry change; these counters describe
        those current owners, not retired correction plans.
        """
        from generativeqc_compiler.common.provenance import canonical_hash

        from .dispersion import D3RuntimeDiagnostic, D4RuntimeDiagnostic

        family = "d4" if self.d4 else "d3"
        descriptor = (
            _native.D4RuntimeDiagnostic if self.d4 else _native.D3RuntimeDiagnostic
        )
        query = getattr(self.library, f"generativeqc_ks_batch_{family}_diagnostic_v1")
        query.argtypes, query.restype = (
            [ct.c_void_p, ct.c_uint32, ct.POINTER(descriptor)],
            ct.c_int,
        )
        records = []
        for index in range(len(self.batch._systems)):
            record = descriptor(ct.sizeof(descriptor), _native.ABI_VERSION)
            _native.check(
                self.library,
                query(self.batch._batch, index, ct.byref(record)),
                context=self.batch._context,
            )
            records.append(record)
        summed = (
            "plan_host_bytes",
            "execution_host_bytes",
            "device_bytes",
            "table_bytes",
            "workspace_bytes",
            "maximum_bytes",
            "total_atoms",
            "system_count",
        )
        values = {name: sum(getattr(item, name) for item in records) for name in summed}
        values.update(
            backend=self.batch._calculator._device_name,
            maximum_atoms=max(item.maximum_atoms for item in records),
            method_ir_identity=self.graph.identity,
            correction_identity=self.spec.identity,
            table_sha256=self._identity("table_sha256"),
            provider_identity=self._identity("provider_identity"),
            scheduler_identity=self._identity("scheduler_identity"),
        )
        if not self.d4:
            variant = (
                "d3.zero-two-body"
                if self.spec.damping == "zero"
                else "d3.bj-atm"
                if self.spec.s9
                else "d3.bj-two-body"
            )
            return D3RuntimeDiagnostic(
                **values,
                radii_sha256=self._identity("radii_sha256"),
                variant_identity=variant,
            )
        counters = (
            "worker_blocks",
            "workspace_slots",
            "execution_count",
            "unchanged_geometry_replays",
            "changed_geometry_replays",
            "coordinate_h2d_bytes",
            "kernel_launches",
        )
        values.update(
            {name: sum(getattr(item, name) for item in records) for name in counters}
        )
        values.update(
            profile=self.spec.profile,
            atm_enabled=bool(self.spec.s9),
            charge_parameter_sha256=self._identity("charge_parameter_sha256"),
            derivative_identity=self._identity("derivative_identity"),
        )
        values["cache_identity"] = canonical_hash(
            {
                "provider": values["provider_identity"],
                "scheduler": values["scheduler_identity"],
                "method_ir": self.graph.identity,
                "correction": self.spec.identity,
                "table_sha256": values["table_sha256"],
                "charge_parameter_sha256": values["charge_parameter_sha256"],
                "derivative": values["derivative_identity"],
            }
        )
        result = D4RuntimeDiagnostic(
            **values,
            peak_host_bytes=values["plan_host_bytes"] + values["execution_host_bytes"],
            peak_device_bytes=values["device_bytes"],
        )
        if self.gcp:
            from .dispersion import R2SCAN3CRuntimeDiagnostic

            return R2SCAN3CRuntimeDiagnostic(
                self.graph.identity,
                result,
                self.batch._calculator._device_name,
                *self._gcp_identities(),
            )
        return result
