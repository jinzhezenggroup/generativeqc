"""Persistent ragged-batch interface backed by the native fleet plan."""

from __future__ import annotations

import ctypes
import os
import typing
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Self

import numpy as np

from . import _native
from ._api_types import Atom, MethodCapabilities
from ._batch_diagnostics import (
    DensityFittingMetricDiagnostic,
    EigensolverDiagnostic,
    InactiveEigensolverProfileEntry,
    PppsQueueProfile,
    ShellClassProfileEntry,
    read_density_fitting_metric_diagnostics,
    read_eigensolver_diagnostics,
    read_inactive_eigensolver_profile,
    read_ppps_queue_profile,
    read_shell_class_profile,
)
from ._result_translation import (
    backend_name,
    copy_force_array,
    status_message,
)
from ._result_translation import (
    read_cc_performance_result as _read_cc_performance_result,
)
from ._result_translation import (
    read_correlation_result as _read_correlation_result,
)
from ._warm_state import WarmStartState
from .ks_diagnostics import (
    KsDiagnostic,
    KsTransportDiagnostic,
    read_ks_diagnostic,
    read_ks_transport_diagnostic,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ._api_types import CcPerformanceResult, CorrelationResult
    from .accuracy import AccuracyAssessment
    from .calculator import Calculator


@dataclass(frozen=True)
class BatchItemResult:
    "Result for one input-indexed ragged-batch member.\n\nEnergy is in Hartree and optional forces are in Hartree/Bohr. Inspect\n``succeeded`` or ``status`` before consuming a member's numerical outputs."

    index: int
    status: int
    status_message: str
    energy: float
    forces: np.ndarray | None
    converged: bool
    iterations: int
    energy_change: float
    density_rms: float
    executed_backend: str
    bucket_id: int
    warm_start_used: bool
    warm_start_fallback: bool
    basis_metadata: dict = field(default_factory=dict)
    accuracy: AccuracyAssessment | None = None
    restart_origin: str = "cold"
    fock_builds: int | None = None
    # None means this item did not complete a solve, or the library predates the query.
    precision: dict | None = None
    incremental_direct_jk: dict | None = None
    # Physical commutator at the returned density; absent for unsupported methods.
    physical_residual_rms: float | None = None
    ks_diagnostic: KsDiagnostic | None = None
    correlation: CorrelationResult | None = None
    cc_performance: CcPerformanceResult | None = None
    dispersion: object | None = None
    initial_guess: dict | None = None

    @property
    def succeeded(self) -> bool:
        """Report whether the native item status indicates success."""
        return self.status == _native.STATUS_SUCCESS


@dataclass(frozen=True)
class BatchResult:
    """Input-ordered results for a ragged batch; forces are never padded.

    ``energies`` is a fresh float64 vector in Hartree with NaN for failed
    items. Check status before using an item's raw scalar diagnostics.
    ``raise_for_failures`` raises RuntimeError with item indices and details.
    See :ref:`python-results-values` and :ref:`python-results-ownership`."""

    items: tuple[BatchItemResult, ...]

    @property
    def energies(self) -> np.ndarray:
        """Return float64 item energies, substituting NaN for failed items."""
        return np.asarray(
            [item.energy if item.succeeded else np.nan for item in self.items],
            dtype=np.float64,
        )

    @property
    def failure_indices(self) -> tuple[int, ...]:
        """Return the indices of items whose native status indicates failure."""
        return tuple(item.index for item in self.items if not item.succeeded)

    @property
    def succeeded(self) -> bool:
        """Report whether every item in the batch succeeded."""
        return not self.failure_indices

    def raise_for_failures(self) -> None:
        """Raise RuntimeError listing each failed item index and status message."""
        failures = [
            f"{item.index}: {item.status_message}"
            for item in self.items
            if not item.succeeded
        ]
        if failures:
            raise RuntimeError("batched item failures: " + "; ".join(failures))


class PreparedBatch:
    """Persistent topology-aware native fleet plan.

    The object is not concurrently re-entrant because successful executions
    may update per-system warm-start densities. Use separate plans for
    concurrent callers. Warm-start updates can be frozen after an initial
    execution when reproducible replays from one fixed dm0 are required.

    Coordinates are in Bohr; energies and forces use Hartree and Hartree/Bohr.
    Use context-managed lifetime or explicit ``close``. Result arrays outlive
    the plan. See :ref:`python-batch-values`, :ref:`python-batch-errors`,
    :ref:`python-batch-ownership` and :ref:`python-batch-backends`."""

    @property
    def capabilities(self) -> MethodCapabilities:
        """Report execution-context capabilities inherited from the calculator."""
        return self._calculator.capabilities

    # Compatibility views preserve the private attributes used by checkpoint
    # and progressive helpers while making WarmStartState their single owner.
    @property
    def _warm_enabled(self) -> bool:
        return self._warm_state.enabled

    @_warm_enabled.setter
    def _warm_enabled(self, value: bool) -> None:
        self._warm_state.enabled = bool(value)

    @property
    def _warm_updates(self) -> bool:
        return self._warm_state.updates

    @_warm_updates.setter
    def _warm_updates(self, value: bool) -> None:
        self._warm_state.updates = bool(value)

    @property
    def _warm_metadata(self) -> list[dict[str, typing.Any] | None]:
        return self._warm_state.metadata

    @_warm_metadata.setter
    def _warm_metadata(self, value: list[dict[str, typing.Any] | None]) -> None:
        if len(value) != self._warm_state.item_count:
            raise ValueError("warm-start metadata must match the batch size")
        self._warm_state.metadata = value

    @property
    def _restart_indices(self) -> set[int]:
        return self._warm_state.restart_indices

    @_restart_indices.setter
    def _restart_indices(self, value: set[int]) -> None:
        self._warm_state.restart_indices = value

    @property
    def _projection_indices(self) -> set[int]:
        return self._warm_state.projection_indices

    @_projection_indices.setter
    def _projection_indices(self, value: set[int]) -> None:
        self._warm_state.projection_indices = value

    @property
    def projection_diagnostics(self) -> dict[str, typing.Any] | None:
        """Return the most recent warm-start projection diagnostics, if any."""
        return self._warm_state.projection_diagnostics

    @projection_diagnostics.setter
    def projection_diagnostics(self, value: dict[str, typing.Any] | None) -> None:
        """Replace the stored warm-start projection diagnostics."""
        self._warm_state.projection_diagnostics = value

    @property
    def checkpoint_diagnostics(self) -> dict[str, typing.Any] | None:
        """Return the most recent warm-start checkpoint diagnostics, if any."""
        return self._warm_state.checkpoint_diagnostics

    @checkpoint_diagnostics.setter
    def checkpoint_diagnostics(self, value: dict[str, typing.Any] | None) -> None:
        """Replace the stored warm-start checkpoint diagnostics."""
        self._warm_state.checkpoint_diagnostics = value

    def __init__(
        self,
        calculator: Calculator,
        systems: Sequence[Iterable[Atom | tuple[str | int, Sequence[float]]]],
        *,
        charges: Sequence[int] | None = None,
        multiplicities: Sequence[int] | None = None,
        warm_start: bool = True,
        shell_class_profiling: bool = False,
        inactive_eigensolver_profiling: bool = False,
        resource_plan: typing.Any = None,
    ) -> None:
        """Prepare native systems and method resources for repeated batch evaluation.

        Charges and multiplicities default to zero and one per system. Explicit
        sequences must match the nonempty batch; an optional resource plan must
        match the requested systems, method, and calculator budget.
        """
        if not systems:
            raise ValueError("a batch requires at least one system")
        self._last_statuses = None
        self._dispersion_batch: typing.Any = None
        self._intrinsic_dispersion_energy = (
            calculator._method == _native.METHOD_PBE_D4_RKS
        )
        # The KS ResourcePlan reserves one serialized generated-force staging cap.
        # Keep one retained execution per PreparedBatch and reprepare on topology drift.
        self._stationary_cuda_execution: typing.Any = None
        self._snapshot_grid_cache: typing.Any = None
        self._calculator = calculator
        self._initial_guess_spec = calculator._initial_guess
        self._library = calculator._library
        self._systems = tuple(
            tuple(Atom.from_value(atom) for atom in system) for system in systems
        )
        if any(not system for system in self._systems):
            raise ValueError("every batch item requires at least one atom")
        count = len(self._systems)
        self._warm_state = WarmStartState(count, enabled=warm_start)
        self._charges = (
            tuple(0 for _ in range(count)) if charges is None else tuple(charges)
        )
        self._multiplicities = (
            tuple(1 for _ in range(count))
            if multiplicities is None
            else tuple(multiplicities)
        )
        if len(self._charges) != count or len(self._multiplicities) != count:
            raise ValueError("charges and multiplicities must match the batch size")
        self._ks_profile_selection = calculator._effective_ks_selection(
            self._systems,
            charges=self._charges,
            multiplicities=self._multiplicities,
        )
        self._effective_ks_options = self._ks_profile_selection.options
        self.resource_plan = resource_plan
        self._planned_properties = None
        from ._cpu_force_resources import qualified_direct_semilocal_context

        self._output_aware_cpu_forces = qualified_direct_semilocal_context(calculator)
        if self._output_aware_cpu_forces and resource_plan is not None:
            self._planned_properties = next(
                (
                    request.identity.observables
                    for request in resource_plan.requests
                    if request.name == "ks"
                ),
                None,
            )
        prepared_properties = (
            calculator._resource_properties(self._planned_properties)
            if self._output_aware_cpu_forces
            else calculator._capabilities.supported_properties
        )
        for atoms in self._systems:
            calculator._preflight_hf_basis(
                atoms, compute_forces="forces" in prepared_properties
            )
        self.resource_diagnostics = None
        self._resource_ledger = None
        dispersion_request = None
        if resource_plan is not None or calculator._resource_budget is not None:
            request = calculator._resource_request(
                self._systems,
                charges=self._charges,
                multiplicities=self._multiplicities,
                ks_options=self._effective_ks_options,
                properties=self._planned_properties,
            )
            dispersion_request = calculator._dispersion_resource_request(self._systems)
            requests = (
                (request,)
                if dispersion_request is None
                else (request, dispersion_request)
            )
            if resource_plan is None:
                from generativeqc_compiler.common.resources import plan_resources

                self.resource_plan = plan_resources(
                    requests, calculator._resource_budget
                )
            else:
                owned = {r.name: r for r in resource_plan.requests}
                for expected in requests:
                    if owned.get(expected.name) != expected:
                        raise ValueError(
                            f"prepared {expected.name.upper()} inputs differ from the global resource plan"
                        )
                if (
                    calculator._resource_budget is not None
                    and resource_plan.budget != calculator._resource_budget
                ):
                    raise ValueError(
                        "global resource plan differs from the calculator budget"
                    )
            self.resource_plan.require_feasible()
            if request.identity.backend == "cuda":
                from .resources_native import NativeDeviceLedger

                self._resource_ledger = NativeDeviceLedger(
                    self._library, self.resource_plan, owner=request.name
                )
            if request.identity.backend == "cuda" and (
                shell_class_profiling or inactive_eigensolver_profiling
            ):
                raise NotImplementedError(
                    "CUDA resource plans exclude optional profiling allocations"
                )
        self._model_signature = calculator._model_signature()
        self._basis_metadata = tuple(
            calculator.basis_metadata(atoms, charge=charge, multiplicity=multiplicity)
            for atoms, charge, multiplicity in zip(
                self._systems, self._charges, self._multiplicities, strict=True
            )
        )
        self._atom_counts = tuple(len(system) for system in self._systems)
        self._atomic_numbers = tuple(
            tuple(atom.atomic_number for atom in system) for system in self._systems
        )
        self._context = ctypes.c_void_p()
        self._batch = ctypes.c_void_p()
        auxiliary_handle = ctypes.c_void_p()
        self._shell_class_profiling = shell_class_profiling
        self._inactive_eigensolver_profiling = inactive_eigensolver_profiling

        _native.check(
            self._library,
            self._library.generativeqc_context_create(
                ctypes.byref(calculator._context_descriptor()),
                ctypes.byref(self._context),
            ),
        )
        system_handles: list[ctypes.c_void_p] = []
        try:
            for atoms, charge, multiplicity in zip(
                self._systems, self._charges, self._multiplicities, strict=True
            ):
                system_handles.append(
                    calculator._create_native_system(
                        self._context, atoms, charge, multiplicity
                    )
                )
            handle_array = (ctypes.c_void_p * count)(
                *(handle.value for handle in system_handles)
            )
            if calculator._auxiliary_basis is not None:
                auxiliary_handle = calculator._create_native_system(
                    self._context,
                    self._systems[0],
                    self._charges[0],
                    self._multiplicities[0],
                    calculator._auxiliary_basis,
                )
            method = calculator._method_descriptor(
                auxiliary_handle if auxiliary_handle.value else None,
                resource_plan=self.resource_plan,
                ks_options=self._effective_ks_options,
                systems=self._systems,
            )
            flags = _native.BATCH_ENABLE_WARM_STARTS if warm_start else 0
            if shell_class_profiling:
                flags |= _native.BATCH_ENABLE_SHELL_CLASS_PROFILING
            if inactive_eigensolver_profiling:
                flags |= _native.BATCH_ENABLE_INACTIVE_EIGENSOLVER_PROFILING

            def prepare() -> typing.Any:
                return self._library.generativeqc_batch_prepare(
                    self._context,
                    handle_array,
                    count,
                    ctypes.byref(method),
                    flags,
                    ctypes.byref(self._batch),
                )

            if self.resource_plan is None:
                _native.check(
                    self._library,
                    prepare(),
                    context=(
                        self._context
                        if calculator._method
                        in (
                            _native.METHOD_MP2,
                            _native.METHOD_UMP2,
                            _native.METHOD_RCCSD,
                            _native.METHOD_RCCSD_T,
                        )
                        else None
                    ),
                )
            else:
                from .resources_native import check_resource_status, observe_method_call

                status, self.resource_diagnostics = observe_method_call(
                    self._library,
                    self.resource_plan,
                    self._resource_ledger,
                    prepare,
                    owner=request.name,
                    phase="preparation",
                )
                check_resource_status(self._library, status, self.resource_diagnostics)
            if calculator._dispersion_method_ir is not None:
                from generativeqc_compiler.method import (
                    D4Spec,
                    DispersionCorrectionPrimitive,
                    GeometricCounterpoisePrimitive,
                )

                from .dispersion import (
                    D3CorrectionBatch,
                    D4CorrectionBatch,
                    R2SCAN3CCorrectionBatch,
                )

                graph = calculator._dispersion_method_ir
                correction_nodes = tuple(
                    node
                    for node in graph.primitives
                    if isinstance(node, DispersionCorrectionPrimitive)
                )
                gcp_nodes = tuple(
                    node
                    for node in graph.primitives
                    if isinstance(node, GeometricCounterpoisePrimitive)
                )
                correction = correction_nodes[0].specification
                if isinstance(correction, D4Spec):
                    d4_systems = [
                        (
                            self._atomic_numbers[index],
                            [atom.position for atom in atoms],
                            self._charges[index],
                        )
                        for index, atoms in enumerate(self._systems)
                    ]
                    if gcp_nodes:
                        if len(gcp_nodes) != 1:
                            raise RuntimeError(
                                "D4 composite execution requires exactly one gCP primitive"
                            )
                        self._dispersion_batch = R2SCAN3CCorrectionBatch(
                            graph,
                            d4_systems,
                            device=calculator._device_name,
                            device_id=calculator._device_id,
                            maximum_bytes=calculator._dispersion_memory_budget_bytes,
                        )
                    else:
                        self._dispersion_batch = D4CorrectionBatch(
                            graph,
                            d4_systems,
                            device=calculator._device_name,
                            device_id=calculator._device_id,
                            maximum_bytes=calculator._dispersion_memory_budget_bytes,
                        )
                else:
                    self._dispersion_batch = D3CorrectionBatch(
                        graph,
                        [
                            (
                                self._atomic_numbers[index],
                                [atom.position for atom in atoms],
                            )
                            for index, atoms in enumerate(self._systems)
                        ],
                        device=calculator._device_name,
                        device_id=calculator._device_id,
                        maximum_bytes=calculator._dispersion_memory_budget_bytes,
                    )
                    if (
                        self.resource_plan is not None
                        and dispersion_request is not None
                    ):
                        selected = dict(self.resource_plan.selections)[
                            dispersion_request.name
                        ]
                        candidate = next(
                            candidate
                            for candidate in dispersion_request.candidates
                            if candidate.name == selected
                        )
                        planned = dict(candidate.decisions)
                        diagnostic = self._dispersion_batch.diagnostic()
                        actual = {
                            "plan_host_bytes": diagnostic.plan_host_bytes,
                            "execution_host_bytes": diagnostic.execution_host_bytes,
                            "device_bytes": diagnostic.device_bytes,
                            "table_bytes": diagnostic.table_bytes,
                            "workspace_bytes": diagnostic.workspace_bytes,
                        }
                        expected = {key: int(planned[key]) for key in actual}
                        if actual != expected:
                            raise RuntimeError(
                                "prepared D3 resource inventory differs from the global ResourcePlan"
                            )
        except Exception:
            # Construction owns native handles before resource-status conversion,
            # which can raise MemoryError as well as ordinary validation errors.
            self.close()
            raise
        finally:
            if auxiliary_handle.value:
                self._library.generativeqc_system_destroy(auxiliary_handle)
            for handle in system_handles:
                self._library.generativeqc_system_destroy(handle)

    @property
    def system_count(self) -> int:
        """Return the number of systems in the open native batch."""
        self._ensure_open()
        return int(self._library.generativeqc_batch_get_system_count(self._batch))

    @property
    def atomic_numbers(self) -> tuple[tuple[int, ...], ...]:
        """Return the prepared atomic numbers grouped in batch order."""
        return self._atomic_numbers

    @property
    def charges(self) -> tuple[int, ...]:
        """Charges retained by this fixed-topology native plan."""

        return self._charges

    @property
    def multiplicities(self) -> tuple[int, ...]:
        """Spin multiplicities retained by this fixed-topology native plan."""

        return self._multiplicities

    @property
    def basis_metadata(self) -> typing.Any:
        """Detached resolved provenance/identities for benchmark and result records."""
        return deepcopy(self._basis_metadata)

    @property
    def dispersion_diagnostic(self) -> typing.Any:
        """Return the retained external-correction diagnostic, if present."""
        self._ensure_open()
        return (
            None
            if self._dispersion_batch is None
            else self._dispersion_batch.diagnostic()
        )

    @property
    def ks_transport_diagnostics(self) -> tuple[KsTransportDiagnostic | None, ...]:
        """Input-ordered cumulative CUDA KS movement snapshots."""
        self._ensure_open()
        return tuple(
            read_ks_transport_diagnostic(self._library, self._batch, index)
            for index in range(self.system_count)
        )

    def _ensure_open(self) -> None:
        if not self._batch.value:
            raise RuntimeError("prepared batch is closed")

    def _stationary_cuda_target(self) -> typing.Any:
        """Resolve the native execution target without discovering a compiler."""
        from generativeqc_compiler.common.cuda_target import cuda_target_info

        from .profiles import probe_device

        device = probe_device(self._library, self._calculator._device_id)["device"]
        self._stationary_cuda_device_name = device["name"]
        return cuda_target_info(f"sm_{device['major']}{device['minor']}")

    def _stationary_cuda_compiler(self) -> typing.Any:
        """Lazily bind the generated-force compiler to this native device."""
        compiler = getattr(self, "_c2_stationary_compiler", None)
        if compiler is not None:
            return compiler
        from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
        from generativeqc_compiler.common.cuda_target import cuda_target_info

        from .profiles import find_nvcc, probe_device

        nvcc = find_nvcc()
        if nvcc is None:
            raise NotImplementedError(
                "public CUDA DFT forces require NVCC; set CUDACXX or CUDA_PATH"
            )
        device = probe_device(self._library, self._calculator._device_id)["device"]
        target = cuda_target_info(f"sm_{device['major']}{device['minor']}")
        compiler = CudaCompilerAdapter(Path(nvcc), target, compile_timeout=600)
        self._c2_stationary_compiler = compiler
        return compiler

    def _public_dft_cuda_force(
        self, index: typing.Any, atoms: typing.Any
    ) -> typing.Any:
        """Execute a compiler-selected stationary CUDA force composition."""
        from generativeqc_compiler.dft import NativeAO
        from generativeqc_compiler.method.stationary_cuda import (
            stationary_aot_profile_for_plan,
        )
        from generativeqc_compiler.method.stationary_gradient import (
            SCF_POINT_MODEL,
            StationaryGradientPlan,
            StationaryMeanField,
        )

        from ._dft_gradient import StationaryKsState
        from ._force_active_ao import (
            QUALIFIED_FORCE_ACTIVE_AO_PROFILES,
            ForceActiveAoWorkload,
            force_active_ao_policy_record,
            resolve_force_active_ao_policy,
        )
        from ._snapshot_grid_cache import SnapshotGridCache
        from ._stationary_composite_cuda import (
            PreparedCompositeStationaryCudaGradient,
            requires_composite_stationary_cuda,
        )
        from ._stationary_cuda import (
            PreparedStationaryCudaExecution,
            PreparedStationaryCudaTopologyMismatch,
            complete_rks_cuda_gradient_diagnostic,
        )

        calculator = self._calculator
        if self._snapshot_grid_cache is None:
            self._snapshot_grid_cache = SnapshotGridCache()
        with NativeAO(
            atoms,
            basis=calculator._basis,
            representation=calculator._representation_name,
            charge=self._charges[index],
            multiplicity=self._multiplicities[index],
        ) as basis:
            state = StationaryKsState.from_native(self, basis, index=index)
            try:
                if state._source.backend != "cuda" or (
                    "forces" not in calculator._capabilities.supported_properties
                ):
                    raise NotImplementedError(
                        "public CUDA DFT forces require a qualified CUDA owner"
                    )
                source = state._source
                composite_force = requires_composite_stationary_cuda(state)
                target = self._stationary_cuda_target()
                if composite_force:
                    max_device_bytes, max_host_bytes = 1 << 30, 2 << 30
                    tile_policy, policy_tile_points = "budget-auto", None
                else:
                    max_device_bytes, max_host_bytes = 512 << 20, 256 << 20
                    tile_policy, policy_tile_points = "budget-auto", None

                profiles_present = bool(QUALIFIED_FORCE_ACTIVE_AO_PROFILES)
                resident_provider = getattr(source, "cuda_resident_grid", None)
                resident_grid_available = callable(resident_provider)
                if (
                    profiles_present
                    and source.hamiltonian == "all-electron"
                    and resident_grid_available
                ):
                    resident_grid_available = resident_provider() is not None
                density_fitted = (
                    bool(getattr(source, "density_fitted", False))
                    if profiles_present
                    else calculator._density_fitting_mode
                    != _native.DENSITY_FITTING_NONE
                )
                workload = ForceActiveAoWorkload(
                    architecture=target.architecture,
                    device_name=getattr(self, "_stationary_cuda_device_name", None),
                    derivative_order=(
                        2
                        if composite_force or "sigma" in state.identity.ingredients
                        else 1
                    ),
                    spin_blocks=1 if state.identity.spin == "unpolarized" else 2,
                    composition="composite" if composite_force else "ordinary",
                    hamiltonian=source.hamiltonian,
                    density_fitted=density_fitted,
                    atoms=basis.natom,
                    aos=basis.nao,
                    grid_points=len(state.grid.points),
                    tile_policy=tile_policy,
                    tile_points=policy_tile_points,
                    max_device_bytes=max_device_bytes,
                    max_host_bytes=max_host_bytes,
                    resident_grid=resident_grid_available,
                )
                decision = resolve_force_active_ao_policy(workload)

                if composite_force:
                    prepared = self._stationary_cuda_execution
                    if not isinstance(
                        prepared, PreparedCompositeStationaryCudaGradient
                    ):
                        if prepared is not None:
                            prepared.close()
                        prepared = PreparedCompositeStationaryCudaGradient()
                        self._stationary_cuda_execution = prepared
                    forces, work = prepared.execute(
                        state,
                        basis,
                        compiler=self._stationary_cuda_compiler(),
                        cache=Path(
                            os.environ.get(
                                "GENERATIVEQC_STATIONARY_CACHE",
                                ".cache/stationary-cuda",
                            )
                        ),
                        library=Path(str(self._library._name)).resolve(),
                        max_device_bytes=max_device_bytes,
                        max_host_bytes=max_host_bytes,
                        active_ao_cutoff=decision.cutoff,
                        active_ao_cache_bytes=decision.cache_bytes,
                        active_ao_producer=decision.producer,
                        active_ao_max_active_fraction=decision.max_active_fraction,
                    )
                    work = dict(work)
                    work["force_active_ao_policy"] = force_active_ao_policy_record(
                        decision, work.get("active_ao_maps")
                    )
                    return forces, work

                prepared = self._stationary_cuda_execution
                if not isinstance(prepared, PreparedStationaryCudaExecution):
                    if prepared is not None:
                        prepared.close()
                    prepared = PreparedStationaryCudaExecution()
                    self._stationary_cuda_execution = prepared
                native_library = Path(str(self._library._name)).resolve()
                all_electron = source.hamiltonian == "all-electron"
                # A point code alone does not identify an exchange composition.
                # Use the actual plan and spin; the loader remains responsible
                # for domain/target/precision/binary admission and fails closed.
                packaged = (
                    all_electron
                    and stationary_aot_profile_for_plan(
                        int(source.metadata[6]),
                        state.identity.spin,
                        StationaryGradientPlan(
                            source.method_ir, StationaryMeanField(SCF_POINT_MODEL)
                        ),
                    )
                    is not None
                )
                kwargs = {
                    "tile_points": policy_tile_points,
                    "compiler": (
                        None if packaged else self._stationary_cuda_compiler()
                    ),
                    "target": target,
                    "cache": Path(
                        os.environ.get(
                            "GENERATIVEQC_STATIONARY_CACHE", ".cache/stationary-cuda"
                        )
                    ),
                    "aot_directory": native_library.parent if packaged else None,
                    "native_grid_library": native_library,
                    "max_device_bytes": max_device_bytes,
                    "max_host_bytes": max_host_bytes,
                    "resident_ao_cutoff": decision.cutoff,
                    "resident_ao_cache_bytes": decision.cache_bytes,
                    "resident_ao_producer": decision.producer,
                    "resident_ao_max_active_fraction": decision.max_active_fraction,
                    "max_grid_points": None,
                    "max_grid_pair_visits": None,
                }
                try:
                    result = complete_rks_cuda_gradient_diagnostic(
                        state, basis, prepared=prepared, **kwargs
                    )
                except PreparedStationaryCudaTopologyMismatch:
                    prepared.close()
                    prepared = PreparedStationaryCudaExecution()
                    self._stationary_cuda_execution = prepared
                    result = complete_rks_cuda_gradient_diagnostic(
                        state, basis, prepared=prepared, **kwargs
                    )
                work = dict(result.work)
                selection = work.get("resident_ao_selection")
                map_work = (
                    selection.get("work") if isinstance(selection, dict) else None
                )
                work["force_active_ao_policy"] = force_active_ao_policy_record(
                    decision, map_work
                )
                return -np.asarray(result.gradient).copy(), work
            finally:
                state._source.close()

    def _public_dft_cpu_force(self, index: typing.Any, atoms: typing.Any) -> typing.Any:
        """Bounded CPU stationary force for qualified ECP or all-electron owners."""
        from generativeqc_compiler.dft import NativeAO

        from ._cpu_force_resources import (
            CPU_FORCE_HOST_CAP,
            cpu_force_inventory,
            qualified_basis,
            qualified_direct_semilocal_context,
        )
        from ._dft_gradient import StationaryKsState
        from ._stationary_cpu import complete_rks_gradient_diagnostic
        from .ecp import resolve_ecp

        calculator = self._calculator
        ecp_force = qualified_basis(calculator._basis)
        direct_semilocal_all_electron = (
            not ecp_force and qualified_direct_semilocal_context(calculator)
        )
        from .ks import cpu_stationary_all_electron_force_eligible

        direct_compiled_all_electron = (
            not ecp_force
            and calculator._automatic_libxc_name is None
            and calculator._ks_options is not None
            and cpu_stationary_all_electron_force_eligible(
                calculator._ks_options.method_ir,
                dispersion_method_ir=calculator._dispersion_method_ir,
            )
        )
        direct_all_electron = (
            direct_semilocal_all_electron or direct_compiled_all_electron
        )
        density_fitted_all_electron = (
            calculator._density_fitting_mode != _native.DENSITY_FITTING_NONE
            and not ecp_force
            and calculator._automatic_libxc_name is None
            and calculator._dispersion_method_ir is None
            and calculator._ks_options is not None
            and calculator._ks_options.execution_plan.nonlocal_correlation is None
            and all(
                term.operator == "full-range"
                for term in calculator._ks_options.execution_plan.exchange
            )
        )
        if calculator._device_name != "cpu" or not (
            ecp_force or direct_all_electron or density_fitted_all_electron
        ):
            raise NotImplementedError(
                "public CPU forces require a qualified ECP or all-electron owner"
            )
        if len(atoms) > 8:
            raise ValueError("CPU public force dense-export domain exceeded")
        with NativeAO(
            atoms,
            basis=calculator._basis,
            representation=calculator._representation_name,
            charge=self._charges[index],
            multiplicity=self._multiplicities[index],
        ) as basis:
            grid = calculator._ks_options.grid
            cores, terms = (), ()
            if ecp_force:
                cores, terms = resolve_ecp(calculator._basis, atoms)
            inventory = cpu_force_inventory(
                basis,
                grid_points=len(atoms)
                * grid.radial_points
                * grid.angular_polar
                * grid.angular_azimuth,
                ecp_terms=len(terms),
                range_exchange_sources=sum(
                    term.operator in ("short-range", "long-range")
                    for term in calculator._ks_options.execution_plan.exchange
                ),
                nonlocal_correlation=(
                    getattr(
                        getattr(calculator._ks_options, "execution_plan", None),
                        "nonlocal_correlation",
                        None,
                    )
                    is not None
                ),
            )
            if sum(inventory.values()) > CPU_FORCE_HOST_CAP:
                raise ValueError("CPU force additional-host byte budget exceeded")
            # Reject before exporting the live SCF/grid snapshot. The consumer
            # repeats admission using the actual exported shape and provider.
            state = StationaryKsState.from_native(self, basis, index=index)
            try:
                if state._source.backend != "cpu":
                    raise NotImplementedError(
                        "public CPU forces require a qualified CPU owner"
                    )
                expected_hamiltonian = (
                    "scalar-semilocal-ecp" if any(cores) else "all-electron"
                )
                if state._source.hamiltonian != expected_hamiltonian:
                    raise ValueError("public CPU force Hamiltonian identity mismatch")
                result = complete_rks_gradient_diagnostic(
                    state,
                    basis,
                    execution="native",
                    max_host_bytes=CPU_FORCE_HOST_CAP,
                    cache=Path(
                        os.environ.get(
                            "GENERATIVEQC_STATIONARY_CACHE", ".cache/stationary-cpu"
                        )
                    ),
                )
                work = dict(result.work)
                if ecp_force:
                    work["ecp_provider"] = "checked-native-cpu-two-grid-v1"
                else:
                    work["hamiltonian_provider"] = "checked-native-cpu-all-electron-v1"
                work["host_inventory"] = inventory
                return -np.asarray(result.gradient).copy(), work
            finally:
                state._source.close()

    def _native_dft_force_eligible(self) -> bool:
        """Use only the native *prepared* item capability for force routing.

        This deliberately does not duplicate PBE/PBE0, CPU/DF, spin, ECP, or
        MethodIR admission predicates in Python. Older libraries and native
        contexts without the full force provider keep the existing Python
        force path; an actual query error is never silently downgraded.
        """
        query = getattr(
            self._library, "generativeqc_ks_batch_supported_properties_v1", None
        )
        if query is None:
            return False
        query.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        query.restype = ctypes.c_int
        for index in range(len(self._systems)):
            properties = ctypes.c_uint32()
            status = query(self._batch, index, ctypes.byref(properties))
            if status == _native.STATUS_NOT_IMPLEMENTED:
                return False
            _native.check(self._library, status, context=self._context)
            if not properties.value & _native.PROPERTY_FORCES:
                return False
        return True

    def _execution_resource_request(self, properties: frozenset[str]) -> typing.Any:
        """Rebudget optional CPU force scratch without changing resident owners."""
        calculator = self._calculator
        options = {
            "charges": self._charges,
            "multiplicities": self._multiplicities,
            "ks_options": self._effective_ks_options,
        }
        expected = calculator._resource_request(
            self._systems, **options, properties=self._planned_properties
        )
        if expected != next(
            request
            for request in self.resource_plan.requests
            if request.name == expected.name
        ):
            raise ValueError(
                f"{expected.name.upper()} resource inputs or execution schedule changed after preparation"
            )
        if not self._output_aware_cpu_forces:
            return expected, self.resource_plan
        candidate = self.resource_plan
        current = calculator._resource_request(
            self._systems, **options, properties=properties
        )
        if current != expected:
            # Preserve all provider selections, phases, other owners and the
            # caller's budget. Only this serialized CPU workspace changes.
            candidate = replace(
                self.resource_plan,
                requests=tuple(
                    current if request.name == current.name else request
                    for request in self.resource_plan.requests
                ),
                status="infeasible",
                diagnostic="requested CPU forces exceed the prepared resource budget",
            )
            if all(
                candidate.peak_bytes.get(space, 0) <= limit
                for space, limit in candidate.budget.limits().items()
            ):
                candidate = replace(candidate, status="feasible", diagnostic=None)
            candidate.require_feasible()
        return current, candidate

    def execute(
        self,
        coordinates: Sequence[Sequence[Sequence[float]] | np.ndarray | None]
        | None = None,
        *,
        strict: bool = False,
        properties: Iterable[str] | None = None,
    ) -> BatchResult:
        """Replay the fleet, optionally omitting analytic forces.

        Each supplied coordinate array must be real and have shape
        ``(natoms, 3)``. Atom-count and nonfinite-coordinate errors retain the
        native per-item failure contract; malformed layouts fail before replay.

        RCCSD retains its energy-only default; request
        ``properties=("energy", "forces")`` explicitly for its qualified force
        domain. Direct all-electron CPU LDA/PBE also defaults to energy only.
        Other methods default to their supported properties. Energy-only
        execution returns ``forces=None``.
        Output selection does not change the prepared model or warm snapshot;
        a later force replay rebuilds response caches when necessary. Resource
        plans revalidate optional CPU semilocal force capacity before replay;
        other owners retain their conservative force capacity allowance.
        Generated force failures retain the original exception type and detail
        in the failed item's ``status_message``, including in strict mode.

        Coordinates use Bohr. This call is synchronous and must not race another
        operation on this plan. ``strict=False`` preserves input-ordered item
        statuses; failed-item forces are None and their scalar energy is not a
        usable result. ``strict=True`` raises after any failed item, without
        rolling back successful neighbors' warm-state updates. Whole-call and
        diagnostic-query errors raise in either mode. Result storage outlives
        the plan. See :ref:`python-batch-errors`."""
        self._ensure_open()
        from .initial_guess import read_initial_guess_diagnostic

        if properties is None:
            properties = self._calculator._default_properties(batch=True)
        if isinstance(properties, (str, bytes)):
            raise TypeError("properties must be an iterable of property names")
        try:
            requested = frozenset(properties)
        except TypeError as error:
            raise TypeError(
                "properties must contain hashable property names"
            ) from error
        if "energy" not in requested:
            raise ValueError("properties must include 'energy'")
        unknown = requested - {"energy", "forces"}
        if unknown:
            names = ", ".join(sorted(repr(name) for name in unknown))
            raise ValueError(f"unsupported properties: {names}")
        unsupported = requested - self._calculator._capabilities.supported_properties
        if unsupported:
            raise ValueError(
                f"method {self._calculator._method_name!r} does not support properties: "
                + ", ".join(sorted(unsupported))
            )
        compute_forces = "forces" in requested
        public_dft_forces = (
            compute_forces
            and self._calculator._capabilities.family == "density_functional"
        )
        if self._calculator._initial_guess != self._initial_guess_spec:
            raise RuntimeError("preliminary SCF policy changed; prepare a new batch")
        if self._calculator._model_signature() != self._model_signature:
            raise RuntimeError(
                "prepared basis/model identity changed; prepare a new batch before reusing densities or Fock/DIIS state"
            )
        native_dft_forces = (
            public_dft_forces and self._native_dft_force_eligible()
        )
        public_dft_forces = public_dft_forces and not native_dft_forces
        native_compute_forces = compute_forces and not public_dft_forces
        if compute_forces and self._output_aware_cpu_forces:
            # Energy-default preparation admits only value operators. Recheck
            # derivative support before a later explicit force replay; never
            # infer derivative admission from the advertised capability alone.
            for atoms in self._systems:
                self._calculator._preflight_hf_basis(atoms, compute_forces=True)
        from .checkpoint import _controls

        controls = _controls(self._calculator)
        count = len(self._systems)
        coordinate_storage: list[np.ndarray] = []
        replay_systems = list(self._systems)
        inputs_pointer = None
        input_count = 0
        if coordinates is not None:
            if len(coordinates) != count:
                raise ValueError("coordinate list must match the prepared batch size")
            input_descriptors: list[_native.BatchInputDescriptor] = []
            for index, item in enumerate(coordinates):
                if item is None:
                    input_descriptors.append(
                        _native.BatchInputDescriptor(
                            ctypes.sizeof(_native.BatchInputDescriptor),
                            _native.ABI_VERSION,
                            None,
                            0,
                        )
                    )
                    continue
                raw = np.asarray(item)
                if np.iscomplexobj(raw):
                    raise ValueError("coordinates must be real")
                # Wrong element counts already have a native per-item failure
                # contract, including flat invalid sentinels. Reject a layout
                # here only when its count could otherwise pass native admission.
                if raw.size == 3 * self._atom_counts[index] and (
                    raw.ndim != 2 or raw.shape[1] != 3
                ):
                    raise ValueError("coordinates must have shape (natoms, 3)")
                array = np.ascontiguousarray(raw, dtype=np.float64).reshape(-1)
                coordinate_storage.append(array)
                if raw.shape == (self._atom_counts[index], 3) and np.all(
                    np.isfinite(raw)
                ):
                    replay_systems[index] = tuple(
                        Atom(
                            atom.atomic_number,
                            tuple(float(value) for value in position),
                        )
                        for atom, position in zip(
                            self._systems[index], raw, strict=True
                        )
                    )
                else:
                    # Preserve native per-item failure semantics for malformed or
                    # nonfinite payloads. Keep the prepared geometry only as the
                    # profile-selection placeholder for this invalid row; every
                    # other executable row must still be requalified.
                    replay_systems[index] = self._systems[index]
                input_descriptors.append(
                    _native.BatchInputDescriptor(
                        ctypes.sizeof(_native.BatchInputDescriptor),
                        _native.ABI_VERSION,
                        array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        array.size,
                    )
                )
            input_array = (_native.BatchInputDescriptor * count)(*input_descriptors)
            inputs_pointer = input_array
            input_count = count
            replay_selection = self._calculator._effective_ks_selection(
                tuple(replay_systems),
                charges=self._charges,
                multiplicities=self._multiplicities,
            )
            if replay_selection.options != self._effective_ks_options:
                raise RuntimeError(
                    "profile-selected KS execution schedule changed for replay coordinates; prepare a new batch"
                )
            if (
                self._ks_profile_selection.exact_profile_match
                and not replay_selection.exact_profile_match
            ):
                raise RuntimeError(
                    "profile-selected KS execution schedule is not qualified for replay coordinates; prepare a new batch"
                )

        force_storage = [
            (ctypes.c_double * (3 * atom_count))() if native_compute_forces else None
            for atom_count in self._atom_counts
        ]
        output_array = (_native.BatchItemResultDescriptor * count)(
            *(
                _native.BatchItemResultDescriptor(
                    ctypes.sizeof(_native.BatchItemResultDescriptor),
                    _native.ABI_VERSION,
                    _native.STATUS_INVALID_ARGUMENT,
                    0.0,
                    force_storage[index],
                    len(force_storage[index]) if native_compute_forces else 0,
                    0,
                    0.0,
                    0.0,
                    0,
                    _native.BACKEND_CPU_REFERENCE,
                    0,
                    0,
                    0,
                )
                for index in range(count)
            )
        )
        execution_plan = self.resource_plan
        if execution_plan is None:
            status = self._library.generativeqc_batch_execute(
                self._batch,
                inputs_pointer,
                input_count,
                output_array,
                count,
            )
        else:
            from .resources_native import observe_method_call

            current, execution_plan = self._execution_resource_request(requested)
            status, self.resource_diagnostics = observe_method_call(
                self._library,
                execution_plan,
                self._resource_ledger,
                lambda: self._library.generativeqc_batch_execute(
                    self._batch, inputs_pointer, input_count, output_array, count
                ),
                owner=current.name,
                previous=self.resource_diagnostics,
            )
        if self.resource_diagnostics is None:
            _native.check(self._library, status)
        else:
            from .resources_native import check_resource_status

            check_resource_status(self._library, status, self.resource_diagnostics)

        correction_results = None
        if (
            self._dispersion_batch is None
            and self._intrinsic_dispersion_energy
            and compute_forces
        ):
            from .dispersion import D4CorrectionBatch

            options = self._calculator._ks_options
            if options is None:
                raise RuntimeError(
                    "PBE-D4 force composition requires resolved KS options"
                )
            self._dispersion_batch = D4CorrectionBatch(
                options.method_ir,
                [
                    (
                        self._atomic_numbers[index],
                        [atom.position for atom in atoms],
                        self._charges[index],
                    )
                    for index, atoms in enumerate(self._systems)
                ],
                device=self._calculator._device_name,
                device_id=self._calculator._device_id,
                maximum_bytes=self._calculator._dispersion_memory_budget_bytes,
            )
        if self._dispersion_batch is not None and (
            compute_forces or not self._intrinsic_dispersion_energy
        ):
            correction_geometries = None
            if coordinates is not None:
                correction_geometries = []
                for index, output in enumerate(output_array):
                    if (
                        output.status == _native.STATUS_SUCCESS
                        and coordinates[index] is not None
                    ):
                        correction_geometries.append(
                            np.asarray(coordinates[index], dtype=np.float64).reshape(
                                self._atom_counts[index], 3
                            )
                        )
                    else:
                        # Preserve the native per-item malformed-input boundary:
                        # D3 does not inspect a coordinate update already rejected by KS.
                        correction_geometries.append(None)
            correction_results = self._dispersion_batch.execute(
                correction_geometries, gradients=compute_forces
            )

        if self.resource_diagnostics is not None:
            # Separate from the native SCF ledger: generated libraries own
            # their own bounded allocations and export/work observations.
            self.resource_diagnostics["generated_force"] = []
        items: list[BatchItemResult] = []
        for index, output in enumerate(output_array):
            builds = ctypes.c_uint64()
            count_status = self._library.generativeqc_batch_get_last_fock_builds(
                self._batch, index, ctypes.byref(builds)
            )
            if count_status not in (
                _native.STATUS_SUCCESS,
                _native.STATUS_NOT_IMPLEMENTED,
            ):
                _native.check(self._library, count_status)
            succeeded = output.status == _native.STATUS_SUCCESS
            dispersion = (
                None if correction_results is None else correction_results[index]
            )
            dispersion_failure_message = None
            if succeeded and dispersion is not None and not dispersion.ok:
                output.status = dispersion.status
                succeeded = False
                dispersion_failure_message = f"external correction failed ({dispersion.status}): {dispersion.message}"
            public_force = None
            force_failure_message = None
            if succeeded and public_dft_forces:
                atoms = self._systems[index]
                if coordinates is not None and coordinates[index] is not None:
                    xyz = np.asarray(coordinates[index], dtype=np.float64).reshape(
                        -1, 3
                    )
                    atoms = tuple(
                        Atom(atom.atomic_number, tuple(position))
                        for atom, position in zip(atoms, xyz, strict=True)
                    )
                try:
                    consumer = (
                        self._public_dft_cuda_force
                        if self._calculator._device_name == "cuda"
                        else self._public_dft_cpu_force
                    )
                    public_force, force_work = consumer(index, atoms)
                    if self.resource_diagnostics is not None:
                        self.resource_diagnostics["generated_force"].append(
                            {"index": index, "work": force_work}
                        )
                except (
                    RuntimeError,
                    OSError,
                    ArithmeticError,
                    MemoryError,
                    TypeError,
                    ValueError,
                ) as error:
                    # A missing library or loader failure is an execution
                    # problem, not evidence that the converged SCF is invalid.
                    # Keep the cause per item so strict mode and failed-neighbor
                    # isolation expose the same actionable diagnostic.
                    if isinstance(error, NotImplementedError):
                        output.status = _native.STATUS_NOT_IMPLEMENTED
                    elif isinstance(error, MemoryError):
                        output.status = _native.STATUS_OUT_OF_MEMORY
                    elif isinstance(error, (TypeError, ValueError)):
                        output.status = _native.STATUS_INVALID_ARGUMENT
                    elif isinstance(error, OSError):
                        output.status = _native.STATUS_INTERNAL_ERROR
                    else:
                        output.status = _native.STATUS_NUMERICAL_FAILURE
                    force_failure_message = (
                        f"generated force failed ({type(error).__name__}): {error}"
                    )
                    succeeded = False
            physical_residual_rms = None
            scf_getter = getattr(
                self._library, "generativeqc_batch_get_scf_diagnostic", None
            )
            if scf_getter is not None:
                diagnostic = _native.ScfDiagnostic(
                    ctypes.sizeof(_native.ScfDiagnostic), _native.ABI_VERSION
                )
                status = scf_getter(self._batch, index, ctypes.byref(diagnostic))
                if status != _native.STATUS_NOT_IMPLEMENTED:
                    _native.check(self._library, status, context=self._context)
                    physical_residual_rms = diagnostic.physical_residual_rms
            forces = (
                public_force
                if succeeded and public_dft_forces
                else copy_force_array(force_storage[index], self._atom_counts[index])
                if succeeded and native_compute_forces
                else None
            )
            if succeeded and dispersion is not None and compute_forces:
                if dispersion.gradient is None:
                    raise RuntimeError(
                        "successful external correction omitted requested dE/dR"
                    )
                if forces is None:
                    raise RuntimeError(
                        "external-correction force composition requires an electronic force"
                    )
                forces = forces - dispersion.gradient
            message = (
                dispersion_failure_message
                if dispersion_failure_message is not None
                else force_failure_message
                if force_failure_message is not None
                else status_message(self._library, output.status)
            )
            total_energy = output.energy
            if (
                succeeded
                and dispersion is not None
                and not self._intrinsic_dispersion_energy
            ):
                total_energy += dispersion.energy
            correlation = None
            if self._calculator._method in (
                _native.METHOD_MP2,
                _native.METHOD_UMP2,
                _native.METHOD_RCCSD,
                _native.METHOD_RCCSD_T,
            ):
                correlation = _read_correlation_result(
                    self._library,
                    self._batch,
                    index=index,
                    context=self._context,
                )
            cc_performance = None
            if self._calculator._method in (
                _native.METHOD_RCCSD,
                _native.METHOD_RCCSD_T,
            ):
                cc_performance = _read_cc_performance_result(
                    self._library,
                    self._batch,
                    index=index,
                    context=self._context,
                )
            accuracy = None
            if succeeded and self._calculator._target_accuracy is not None:
                atoms = self._systems[index]
                if coordinates is not None and coordinates[index] is not None:
                    xyz = np.asarray(coordinates[index], dtype=np.float64).reshape(
                        -1, 3
                    )
                    atoms = tuple(
                        Atom(atom.atomic_number, tuple(position))
                        for atom, position in zip(atoms, xyz, strict=True)
                    )
                accuracy = self._calculator._accuracy_assessment(
                    atoms,
                    self._charges[index],
                    self._multiplicities[index],
                    bool(output.converged),
                )
            items.append(
                BatchItemResult(
                    index=index,
                    status=output.status,
                    fock_builds=builds.value
                    if count_status == _native.STATUS_SUCCESS
                    else None,
                    status_message=message,
                    energy=total_energy,
                    forces=forces,
                    converged=bool(output.converged),
                    iterations=output.iterations,
                    energy_change=output.energy_change,
                    density_rms=output.density_rms,
                    physical_residual_rms=physical_residual_rms,
                    ks_diagnostic=read_ks_diagnostic(
                        self._library,
                        self._batch,
                        index,
                        expected_domain=self._calculator._ks_options.scf_domain,
                    )
                    if self._calculator._ks_options is not None
                    else None,
                    initial_guess=read_initial_guess_diagnostic(
                        self._library, self._batch, index
                    ),
                    correlation=correlation,
                    cc_performance=cc_performance,
                    dispersion=dispersion if succeeded else None,
                    executed_backend=backend_name(output.executed_backend),
                    bucket_id=output.bucket_id,
                    warm_start_used=bool(output.warm_start_used),
                    warm_start_fallback=bool(output.warm_start_fallback),
                    basis_metadata=deepcopy(self._basis_metadata[index]),
                    precision=self._calculator._precision_report(self._batch, index),
                    incremental_direct_jk=self._calculator._incremental_direct_jk_diagnostic(
                        self._batch, index
                    ),
                    accuracy=accuracy,
                    restart_origin=self._warm_state.origin(
                        index,
                        warm_start_used=bool(output.warm_start_used),
                        warm_start_fallback=bool(output.warm_start_fallback),
                    ),
                )
            )
        result = BatchResult(tuple(items))
        self._last_statuses = tuple(item.status for item in result.items)
        for index, item in enumerate(result.items):
            if item.succeeded:
                self._warm_state.record_success(
                    index,
                    controls=deepcopy(controls),
                    backend="cuda" if item.executed_backend == "cuda" else "cpu",
                )
        if self._warm_state.projection_diagnostics:
            self._warm_state.projection_diagnostics["target_verification"] = "executed"
            self._warm_state.projection_diagnostics["target_results"] = [
                {
                    "index": i.index,
                    "converged": i.converged,
                    "status": i.status,
                    "energy": i.energy if i.succeeded else None,
                    "density_rms": i.density_rms if i.succeeded else None,
                    "iterations": i.iterations,
                    "fock_builds": i.fock_builds,
                    "restart_origin": i.restart_origin,
                }
                for i in result.items
            ]
        if (
            self._warm_state.checkpoint_diagnostics
            and "target_verification" in self._warm_state.checkpoint_diagnostics
        ):
            self._warm_state.checkpoint_diagnostics["target_verification"] = "executed"
            self._warm_state.checkpoint_diagnostics["target_results"] = [
                {
                    "index": i.index,
                    "converged": i.converged,
                    "status": i.status,
                    "energy": i.energy if i.succeeded else None,
                    "density_rms": i.density_rms if i.succeeded else None,
                    "restart_origin": i.restart_origin,
                }
                for i in result.items
            ]
        if strict:
            try:
                result.raise_for_failures()
            except RuntimeError as error:
                if self.resource_diagnostics is not None:
                    error.resource_diagnostics = self.resource_diagnostics
                raise
        if execution_plan is not None and all(item.succeeded for item in result.items):
            self.resource_plan = execution_plan
            if self._output_aware_cpu_forces:
                self._planned_properties = requested
        return result

    def save_checkpoint(
        self, path: typing.Any, *, max_bytes: typing.Any = 256 << 20
    ) -> typing.Any:
        """Atomically persist retained HF seeds, identities and source diagnostics.

        Failed/no-state items keep their input slots. A one-item prepared batch
        provides single-system checkpoint/restart with the same contract.
        """
        from .checkpoint import save_checkpoint

        return save_checkpoint(self, path, max_bytes=max_bytes)

    def load_checkpoint(
        self,
        path: typing.Any,
        *,
        allow_warm: typing.Any = False,
        strict: typing.Any = True,
        max_bytes: typing.Any = 256 << 20,
    ) -> typing.Any:
        """Restore compatible seeds as proposals for the next normal execution.

        Exact restart is the default. ``allow_warm`` permits changed geometry or
        numerical controls with the same scientific model. ``strict=False``
        preserves incompatible neighbors; corruption always rejects the file
        before any seed is applied. Runtime resources follow this target plan.
        """
        from .checkpoint import load_checkpoint

        return load_checkpoint(
            self, path, allow_warm=allow_warm, strict=strict, max_bytes=max_bytes
        )

    def clear_warm_starts(self) -> None:
        """Clear both native warm starts and the associated Python warm-state records."""
        self._ensure_open()
        _native.check(
            self._library,
            self._library.generativeqc_batch_clear_warm_starts(self._batch),
        )
        self._warm_state.clear()

    def initialize_from(
        self,
        source: typing.Any,
        *,
        policy: typing.Any = None,
        strict: typing.Any = True,
        maximum_host_bytes: typing.Any = 256 << 20,
    ) -> typing.Any:
        """Project a converged source batch into this fresh target's AO metric.

        The next execute rebuilds and converges the target equations. See
        ``generativeqc.progressive.initialize_from`` for compatibility and fallback.
        """
        from .progressive import initialize_from

        return initialize_from(
            self,
            source,
            policy=policy,
            strict=strict,
            maximum_host_bytes=maximum_host_bytes,
        )

    def set_warm_start_updates(self, enabled: bool) -> None:
        """Control whether successful executions replace retained densities.

        Disabling updates freezes the current per-system snapshots without
        disabling warm starts. It is intended for controlled A/B benchmarks
        where every replay must begin from exactly the same post-cold dm0.
        """

        self._ensure_open()
        _native.check(
            self._library,
            self._library.generativeqc_batch_set_warm_start_updates(
                self._batch, int(bool(enabled))
            ),
        )
        self._warm_state.updates = bool(enabled)

    def last_shell_class_profile(self) -> tuple[ShellClassProfileEntry, ...]:
        """Return work surviving the most recent final-density CUDA screening.

        Profiling is intentionally opt-in because collecting it adds a CUDA
        reduction and device-to-host copy outside the normal hot path.
        """

        self._ensure_open()
        if not self._shell_class_profiling:
            raise RuntimeError(
                "the batch was not prepared with shell_class_profiling=True"
            )
        return read_shell_class_profile(self._library, self._batch)

    def last_ppps_queue_profile(self) -> PppsQueueProfile:
        """Return production PPPS occupancy and primitive-divergence data.

        The batch must opt into ``shell_class_profiling``. Collection copies a
        compact signature for every screened PPPS ket task and is therefore a
        benchmark/debug operation, not part of normal endpoint timing.
        """

        self._ensure_open()
        if not self._shell_class_profiling:
            raise RuntimeError(
                "the batch was not prepared with shell_class_profiling=True"
            )
        return read_ppps_queue_profile(self._library, self._batch)

    def last_eigensolver_diagnostics(
        self,
    ) -> tuple[EigensolverDiagnostic, ...]:
        """Return one cached setup decision for every CUDA workload bucket."""

        self._ensure_open()
        return read_eigensolver_diagnostics(self._library, self._batch)

    def last_density_fitting_metric_diagnostics(
        self,
    ) -> tuple[DensityFittingMetricDiagnostic, ...]:
        """Return CUDA DF metric conditioning/allocation records from the last run."""

        self._ensure_open()
        return read_density_fitting_metric_diagnostics(self._library, self._batch)

    def last_inactive_eigensolver_profile(
        self,
    ) -> tuple[InactiveEigensolverProfileEntry, ...]:
        """Return device-timed iteration records from the last CUDA run."""

        self._ensure_open()
        if not self._inactive_eigensolver_profiling:
            raise RuntimeError(
                "the batch was not prepared with inactive_eigensolver_profiling=True"
            )
        return read_inactive_eigensolver_profile(self._library, self._batch)

    def close(self) -> None:
        """Release cached execution, correction, batch, context, and ledger resources."""
        if self._snapshot_grid_cache is not None:
            self._snapshot_grid_cache.clear()
            self._snapshot_grid_cache = None
        if self._dispersion_batch is not None:
            with suppress(Exception):
                self._dispersion_batch.close()
            self._dispersion_batch = None
        if self._stationary_cuda_execution is not None:
            with suppress(Exception):
                self._stationary_cuda_execution.close()
            self._stationary_cuda_execution = None
        if self._batch.value:
            self._library.generativeqc_batch_destroy(self._batch)
            self._batch = ctypes.c_void_p()
        if self._context.value:
            self._library.generativeqc_context_destroy(self._context)
            self._context = ctypes.c_void_p()
        if self._resource_ledger is not None:
            self._resource_ledger.close()

    def __enter__(self) -> Self:
        """Require an open batch and return it for context-managed use."""
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Release the prepared batch resources when leaving the context."""
        self.close()

    def __del__(self) -> None:
        """Attempt to close the batch, suppressing cleanup errors during finalization."""
        with suppress(Exception):
            self.close()
