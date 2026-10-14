"""Native precision fleets with bounded, sequential observable-audit ownership."""

from __future__ import annotations

import ctypes
from contextlib import ExitStack, suppress
from copy import deepcopy
from dataclasses import replace
from time import perf_counter
from typing import Any, Self

import numpy as np

from . import _native
from ._api_types import Atom
from .batch import BatchResult
from .checkpoint import _descriptor, _pointer, _resource_check
from .precision_policy import PrecisionBudgetError, _model_identity, _solve_record


def _seed_capacity(batch: Any) -> tuple[int, int]:
    """Bound export before the native size query materializes resident densities.

    The old query ABI can itself perform D2H allocation. Use prepared topology
    for a conservative all-slot bound instead of allocating first and checking
    afterward. This does not execute integrals or inspect a reference density.
    """
    calculator = batch._calculator
    options = calculator._ks_options
    spins = (
        2
        if (
            calculator._method == _native.METHOD_UHF
            or (options is not None and options.method_ir.spin == "polarized")
        )
        else 1
    )
    numeric = largest = 0
    for atoms, metadata in zip(batch._systems, batch._basis_metadata, strict=True):
        spherical = metadata["orbital"]["representation"] == "spherical"
        aos = sum(
            2 * shell.angular_momentum + 1
            if spherical
            else (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
            for shell in calculator._shells_for_atoms(atoms)
        )
        density = 8 * spins * aos * aos
        numeric += density + 24 * len(atoms)
        largest = max(largest, density)
    return numeric, largest


class _SeedLease:
    """Detach only this candidate's seeds before releasing its device owner.

    The historical HF-named native ABI also implements KS warm states. This is
    same-model in-process transport, not a portable checkpoint or an audit seed.
    Native restoration validates source overlap/occupations and always reruns
    the target SCF. Numeric staging and its validation scratch are bounded.
    """

    def __init__(self, batch: Any, maximum_bytes: int) -> None:
        self.signature = batch._calculator._model_signature()
        self.systems = batch._systems
        self.metadata = deepcopy(batch._warm_state)
        self.states = None
        self.storage = []
        self.numeric_bytes = 0
        self.reserved_host_bytes = 0
        if not batch._warm_state.enabled:
            return
        if not all(
            hasattr(batch._library, name)
            for name in (
                "generativeqc_batch_get_hf_warm_state",
                "generativeqc_batch_restore_hf_warm_states",
            )
        ):
            raise NotImplementedError(
                "native precision audit seed transport unavailable"
            )
        bound, largest_bound = _seed_capacity(batch)
        if bound > maximum_bytes:
            raise MemoryError("precision audit seed staging exceeds maximum_seed_bytes")
        _resource_check(batch, bound, largest_bound)
        descriptors = [_descriptor() for _ in range(batch.system_count)]
        largest = 0
        for index, state in enumerate(descriptors):
            _native.check(
                batch._library,
                batch._library.generativeqc_batch_get_hf_warm_state(
                    batch._batch, index, ctypes.byref(state)
                ),
                context=batch._context,
            )
            self.numeric_bytes += 8 * (state.density_count + state.coordinate_count)
            largest = max(largest, 8 * state.density_count)
        if self.numeric_bytes > bound or largest > largest_bound:
            raise RuntimeError("native seed dimensions exceed prepared topology")
        self.reserved_host_bytes = _resource_check(batch, self.numeric_bytes, largest)
        self.largest_density_bytes = largest
        self.states = (_native.HfWarmState * batch.system_count)()
        for index, state in enumerate(descriptors):
            if state.present:
                density = np.empty(state.density_count, dtype=np.float64)
                coordinates = np.empty(state.coordinate_count, dtype=np.float64)
                state.density, state.coordinates = (
                    _pointer(density),
                    _pointer(coordinates),
                )
                _native.check(
                    batch._library,
                    batch._library.generativeqc_batch_get_hf_warm_state(
                        batch._batch, index, ctypes.byref(state)
                    ),
                    context=batch._context,
                )
                density.setflags(write=False)
                coordinates.setflags(write=False)
                self.storage.extend((density, coordinates))
            self.states[index] = state

    def restore(self, batch: Any, *, updates: bool) -> None:
        """Rebind the same model, retaining frozen source geometry and metadata."""
        if (
            batch._calculator._model_signature() != self.signature
            or batch._systems != self.systems
        ):
            raise ValueError("precision seed lease model/topology changed")
        if self.states is not None:
            _resource_check(batch, self.numeric_bytes, self.largest_density_bytes)
            _native.check(
                batch._library,
                batch._library.generativeqc_batch_restore_hf_warm_states(
                    batch._batch, self.states, batch.system_count
                ),
                context=batch._context,
            )
        batch._warm_state = deepcopy(self.metadata)
        batch.set_warm_start_updates(updates)


class PreparedPrecisionBatch:
    """Reuse one native candidate fleet and audit only items that need it.

    The prepared candidate supports the facade's full observable set, choosing
    strict SCF when AUTO lacks a force consumer. Singlepoint selection is not
    changed. Qualified calls retain native resources and frozen seeds exactly;
    audited calls release device residency and lazily rebuild on the next call.
    Warm proposals remain candidate-owned, even when an audited value replaces
    a candidate. Auditor densities never enter the candidate's warm history.
    """

    def __init__(
        self, facade: Any, systems: Any, *, maximum_seed_bytes: int, **options: Any
    ) -> None:
        """Prepare one candidate owner; do not preallocate any audit workspace."""
        if type(maximum_seed_bytes) is not int or not 0 <= maximum_seed_bytes < 2**63:
            raise ValueError(
                "maximum_seed_bytes must be a nonnegative int64 byte count"
            )
        self._facade = facade
        self._closed = False
        self._owner = None
        self._seeds = None
        self._maximum_seed_bytes = maximum_seed_bytes
        self._updates = True
        self._calculator, self._precision, self._reasons = facade._candidate_for(
            facade.capabilities.supported_properties
        )
        self._systems = tuple(
            tuple(Atom.from_value(atom) for atom in system) for system in systems
        )
        self._model_cache: list[tuple[tuple[Atom, ...], str] | None] = [
            None for _ in self._systems
        ]
        from .ecp import resolve_ecp

        if any(
            any(resolve_ecp(self._calculator._basis, atoms)[0])
            for atoms in self._systems
        ):
            raise NotImplementedError(
                "application precision ECP model identity is not implemented"
            )
        self._options = dict(options)
        self._signature = self._calculator._model_signature()
        self._owner = self._calculator.prepare_batch(self._systems, **self._options)
        try:
            self._charges = self._owner.charges
            self._multiplicities = self._owner.multiplicities
            self._options.update(
                charges=self._charges, multiplicities=self._multiplicities
            )
            self._basis_metadata = self._owner.basis_metadata
            self.resource_plan = self._owner.resource_plan
            self.resource_diagnostics = self._owner.resource_diagnostics
            self._transport = tuple(None for _ in self._systems)
        except BaseException:
            self.close()
            raise

    @property
    def capabilities(self) -> Any:
        """Return this plan's actual property boundary without response promotion."""
        return replace(
            self._facade.capabilities,
            supported_properties=self._calculator.capabilities.supported_properties,
        )

    @property
    def system_count(self) -> int:
        """Return the fixed input slot count."""
        return len(self._systems)

    @property
    def atomic_numbers(self) -> tuple:
        """Return immutable, input-ordered nuclei."""
        return tuple(
            tuple(atom.atomic_number for atom in atoms) for atoms in self._systems
        )

    @property
    def charges(self) -> tuple:
        """Return snapshotted per-item charges."""
        return self._charges

    @property
    def multiplicities(self) -> tuple:
        """Return snapshotted per-item multiplicities."""
        return self._multiplicities

    @property
    def basis_metadata(self) -> Any:
        """Return detached native basis/model metadata."""
        return deepcopy(self._basis_metadata)

    @property
    def ks_transport_diagnostics(self) -> tuple:
        """Return the last candidate fleet's transport, including after suspension."""
        return self._transport

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("prepared precision batch is closed")
        if self._calculator._model_signature() != self._signature:
            raise ValueError("prepared precision model changed; prepare a new batch")

    def _resume(self) -> None:
        if self._owner is not None:
            return
        owner = self._calculator.prepare_batch(self._systems, **self._options)
        try:
            if self._seeds is not None:
                self._seeds.restore(owner, updates=self._updates)
            else:
                owner.set_warm_start_updates(self._updates)
        except BaseException:
            owner.close()
            raise
        self._owner = owner
        self._seeds = None

    def _candidate_model(self, index: int, atoms: tuple[Atom, ...]) -> str:
        """Reuse only geometry-bound provenance under the checked model signature.

        Keep at most one immutable geometry/identity pair per prepared slot,
        replacing it on movement rather than retaining a trajectory. Charge,
        multiplicity and topology are fixed by this plan; execute() rejects a
        changed calculator signature before lookup. No density, qualification,
        result or auditor identity is cached here.
        """
        cached = self._model_cache[index]
        if cached is not None and cached[0] == atoms:
            return cached[1]
        identity = _model_identity(
            self._calculator, atoms, self._charges[index], self._multiplicities[index]
        )
        self._model_cache[index] = (atoms, identity)
        return identity

    def _validate_item(
        self,
        original: Any,
        coordinates: Any,
        requested: frozenset[str],
        started: float,
        reference_factory: Any,
        audit_calls: list,
    ) -> Any:
        """Isolate item audits, recording setup failures as well as solved work."""
        index = original.index
        solves = []
        attempt = None
        audit_started = None

        def reference(*, tighter: bool = False) -> Any:
            nonlocal attempt, audit_started
            attempt = {
                "index": index,
                "role": "fp64_fallback_verification" if tighter else "fp64_audit",
                "completed": False,
                "seconds": None,
            }
            audit_calls.append(attempt)
            audit_started = perf_counter()
            try:
                return reference_factory(tighter=tighter)
            except BaseException as error:
                attempt["error"] = f"{type(error).__name__}: {error}"
                attempt["seconds"] = perf_counter() - audit_started
                raise

        def solve(calculator: Any, role: str, precision: str) -> Any:
            if role == "candidate":
                observed = original
                seconds = None
                record = _solve_record(
                    calculator,
                    observed,
                    role=role,
                    precision=precision,
                    seconds=seconds,
                    requested=requested,
                    atom_count=len(atoms),
                    backend=backend,
                )
            else:
                try:
                    with ExitStack() as cleanup:
                        cleanup.callback(calculator.clear_cache)
                        if (
                            _model_identity(
                                calculator,
                                atoms,
                                self._charges[index],
                                self._multiplicities[index],
                            )
                            != model
                        ):
                            raise ValueError(
                                "precision audit cannot change the scientific model"
                            )
                        with calculator.prepare_batch(
                            [atoms],
                            charges=[self._charges[index]],
                            multiplicities=[self._multiplicities[index]],
                            warm_start=False,
                        ) as owner:
                            observed = owner.execute(
                                strict=False, properties=requested
                            ).items[0]
                            attempt.update(
                                status=observed.status,
                                fock_builds=observed.fock_builds,
                                iterations=observed.iterations,
                            )
                            if not observed.succeeded:
                                raise RuntimeError(observed.status_message)
                        observed = replace(observed, index=index)
                        record = _solve_record(
                            calculator,
                            observed,
                            role=role,
                            precision=precision,
                            seconds=None,
                            requested=requested,
                            atom_count=len(atoms),
                            backend=backend,
                        )
                    attempt["completed"] = True
                except BaseException as error:
                    attempt["error"] = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    attempt["seconds"] = perf_counter() - audit_started
                record = replace(record, seconds=attempt["seconds"])
            solves.append(record)
            return observed

        try:
            atoms = self._systems[index]
            if coordinates is not None and coordinates[index] is not None:
                atoms = tuple(
                    Atom(atom.atomic_number, tuple(position))
                    for atom, position in zip(
                        atoms,
                        coordinates[index].reshape(-1, 3),
                        strict=True,
                    )
                )
            model = self._candidate_model(index, atoms)
            backend = (
                "cpu_reference" if self._facade._options["device"] == "cpu" else "cuda"
            )
            result = self._facade._validate_endpoint(
                candidate_calculator=self._calculator,
                candidate_precision=self._precision,
                requested=requested,
                route_reasons=list(self._reasons),
                model=model,
                solves=solves,
                solve=solve,
                started=started,
                reference_factory=reference,
            )
            return replace(
                result,
                precision_validation=replace(
                    result.precision_validation,
                    schema_version=2,
                    endpoint_seconds=None,
                    timing_scope="shared_batch_endpoint",
                ),
                fock_builds=sum(stage.fock_builds for stage in solves)
                if all(stage.fock_builds is not None for stage in solves)
                else None,
            )
        except (
            RuntimeError,
            ValueError,
            TypeError,
            ArithmeticError,
            MemoryError,
            OSError,
        ) as error:
            validation = (
                error.validation if isinstance(error, PrecisionBudgetError) else None
            )
            if validation is not None:
                validation = replace(
                    validation,
                    schema_version=2,
                    endpoint_seconds=None,
                    timing_scope="shared_batch_endpoint",
                )
            return replace(
                original,
                status=_native.STATUS_OUT_OF_MEMORY
                if isinstance(error, MemoryError)
                else _native.STATUS_NUMERICAL_FAILURE,
                status_message=f"application precision failed ({type(error).__name__}): {error}",
                energy=float("nan"),
                forces=None,
                converged=False,
                precision=None,
                precision_validation=validation,
                fock_builds=None,
            )

    def execute(
        self, coordinates: Any = None, *, strict: bool = False, properties: Any = None
    ) -> BatchResult:
        """Execute one fleet, preserving per-item failures and complete audit cost.

        Coordinate semantics and units match PreparedBatch. Per-item validation
        uses schema2 timing: shared candidate seconds are None, while the batch
        report owns the measured fleet/end-to-end duration. Failed audits publish
        no numerical result. strict=True raises only after all peers are handled.
        """
        with self._facade._lock:
            self._ensure_open()
            started = perf_counter()
            requested = self._facade._properties(properties)
            if requested - self.capabilities.supported_properties:
                raise ValueError(
                    "properties are unavailable for this prepared precision plan"
                )
            if coordinates is not None:
                if len(coordinates) != self.system_count:
                    raise ValueError(
                        "coordinate list must match the prepared batch size"
                    )
                # Preserve shape/dtype for native malformed-input semantics while
                # preventing caller mutations between candidate and audit solves.
                coordinates = tuple(
                    None if value is None else np.array(value, copy=True)
                    for value in coordinates
                )
                for value in coordinates:
                    if value is not None:
                        value.setflags(write=False)
            self._resume()
            resume_seconds = perf_counter() - started
            candidate_started = perf_counter()
            candidate = self._owner.execute(
                coordinates, strict=False, properties=requested
            )
            candidate_seconds = perf_counter() - candidate_started
            self.resource_plan = self._owner.resource_plan
            self.resource_diagnostics = deepcopy(self._owner.resource_diagnostics)
            self._transport = self._owner.ks_transport_diagnostics
            budget = (
                self.resource_plan.budget
                if self.resource_plan is not None
                else self._calculator._resource_budget
            )
            suspension_seconds = 0.0
            suspension_error = None
            audit_calls = []

            def reference_factory(*, tighter: bool = False) -> Any:
                nonlocal suspension_seconds, suspension_error
                if suspension_error is not None:
                    raise suspension_error
                if self._owner is not None:
                    began = perf_counter()
                    try:
                        seeds = _SeedLease(self._owner, self._maximum_seed_bytes)
                        self._owner.close()
                        self._owner = None
                        self._seeds = seeds
                    except BaseException as error:
                        suspension_error = error
                        raise
                    finally:
                        suspension_seconds += perf_counter() - began
                return self._facade._reference(
                    tighter=tighter,
                    _host_reservation=(budget, self._seeds.reserved_host_bytes),
                )

            items = []
            for original in candidate.items:
                if not original.succeeded:
                    items.append(original)
                    continue
                items.append(
                    self._validate_item(
                        original,
                        coordinates,
                        requested,
                        started,
                        reference_factory,
                        audit_calls,
                    )
                )
            result = BatchResult(
                tuple(items),
                precision_validation={
                    "schema_version": 1,
                    "profile": self._facade.policy.profile,
                    "candidate_calls": 1,
                    "candidate_seconds": candidate_seconds,
                    "candidate_fock_builds": [
                        item.fock_builds for item in candidate.items
                    ],
                    "candidate_precision": [item.precision for item in candidate.items],
                    "resume_seconds": resume_seconds,
                    "suspension_seconds": suspension_seconds,
                    "audit_calls": audit_calls,
                    "endpoint_seconds": perf_counter() - started,
                    "seed_bytes_retained": 0
                    if self._seeds is None
                    else self._seeds.numeric_bytes,
                    "scope": "complete-fleet-and-sequential-independent-audits",
                },
            )
            if strict:
                result.raise_for_failures()
            return result

    def set_warm_start_updates(self, enabled: bool) -> None:
        """Freeze or update candidate-owned proposals without importing audit seeds."""
        with self._facade._lock:
            self._ensure_open()
            if self._owner is not None:
                self._owner.set_warm_start_updates(enabled)
            self._updates = bool(enabled)

    def clear_warm_starts(self) -> None:
        """Discard native or suspended candidate seeds without closing the plan."""
        with self._facade._lock:
            self._ensure_open()
            if self._owner is not None:
                self._owner.clear_warm_starts()
            self._seeds = None

    def close(self) -> None:
        """Drain and release native residency and detached seed ownership."""
        with self._facade._lock:
            if self._owner is not None:
                self._owner.close()
                self._owner = None
            self._seeds = None
            self._model_cache.clear()
            self._closed = True

    def __enter__(self) -> Self:
        """Require an open owner for context-managed execution."""
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        """Close the plan even after an execution exception."""
        self.close()

    def __del__(self) -> None:
        """Best-effort cleanup for abandoned or partially constructed owners."""
        with suppress(Exception):
            self.close()
