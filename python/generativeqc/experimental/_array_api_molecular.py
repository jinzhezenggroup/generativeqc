"""Explicit native molecular source cuts for first-order Array API capture.

The native method owns SCF, integrals, and analytic nuclear derivatives. TensorIR
owns only the surrounding expression and its chain rule. A source's first-order
jet is bound at each actual geometry, never a differentiated iteration history
or a numerical surrogate for the method's energy away from that geometry.
"""

from __future__ import annotations

import math
import os
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from generativeqc_compiler.array_api import VibeArray, input_array
from generativeqc_compiler.array_api import namespace as symbolic
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.tensor import Program, TensorSpec
from typing_extensions import Self

from . import _array_api_eager as eager

if TYPE_CHECKING:
    from generativeqc.ks import KsOptions
    from generativeqc_compiler.method import MethodIR


_PREFIX = "__gqc_molecular_"
_CAPTURE: ContextVar[_MolecularCapture | None] = ContextVar(
    "generativeqc_molecular_capture", default=None
)


@dataclass(frozen=True, init=False)
class MolecularModel:
    """Immutable, method-neutral native molecular energy source.

    ``method`` accepts a native selector or an immutable, audited MethodIR.
    ``ks_options`` may specify the grid and XC scheduling but cannot override
    the method-owned XC composition. Unsupported science fails via Calculator.

    Native Calculator owns method resolution, electronic structure and its
    available analytic forces; TensorIR compiles only the surrounding algebra
    and first-order chain rule. No new electronic method, numerical-gradient
    fallback, differentiated SCF history or implicit higher derivative is
    invented. Coordinates are FP64 (natoms, 3) in Bohr; energy is in Hartree
    and complete native analytic gradients are in Hartree/Bohr.

    CPU/CUDA selection is explicit and consistent with the compiled algebra.
    CUDA currently stages coordinates/results via the host. Native method and
    TensorIR compilation resource limits are independently enforced.
    """

    _numbers: tuple[int, ...]
    _method: str | MethodIR
    _basis: str
    _charge: int
    _multiplicity: int
    _density_fitting: str
    _auxiliary_basis: str | None
    _backend: str
    _device: int
    _controls: tuple[tuple[str, Any], ...]
    _resource_budget: Any
    _ks_options: KsOptions | None

    def __init__(
        self,
        elements: tuple[str | int, ...],
        *,
        method: str | MethodIR,
        basis: str = "sto-3g",
        charge: int = 0,
        multiplicity: int = 1,
        backend: str = "cpu",
        device: int = 0,
        density_fitting: str = "none",
        auxiliary_basis: str | None = None,
        ks_options: KsOptions | None = None,
        max_iterations: int = 100,
        diis_history: int = 8,
        energy_tolerance: float = 1e-12,
        density_tolerance: float = 1e-10,
        screening_tolerance: float = 1e-14,
        resource_budget: Any = None,
    ) -> None:
        """Bind a method selector and immutable first-order source contract.

        Admission validates static nuclei, spin, resource and solver settings
        without loading native code. Unsupported method/basis/device or missing
        analytic-gradient capabilities fail through the existing Calculator
        at execution, with no reference/CPU or finite-difference fallback.
        Compiled owners must be closed to release native workspaces.
        """
        from generativeqc_compiler.common.resources import ResourceBudget
        from generativeqc_compiler.method import MethodIR

        from generativeqc.elements import atomic_number

        if not isinstance(elements, tuple) or not 1 <= len(elements) <= 256:
            raise ValueError("molecular elements must be a tuple of 1 to 256 nuclei")
        numbers = tuple(atomic_number(element) for element in elements)
        if type(charge) is not int or not -(2**31) <= charge < 2**31:
            raise ValueError("molecular charge must be an int32")
        if isinstance(method, str):
            if not method.strip():
                raise ValueError("molecular method must be a nonempty native selector")
            method = method.strip().lower()
        elif not isinstance(method, MethodIR):
            raise TypeError("molecular method must be a native selector or MethodIR")
        electrons = sum(numbers) - charge
        if method == "rhf" and (electrons <= 0 or electrons % 2):
            raise ValueError("RHF requires a positive even electron count")
        if (
            type(multiplicity) is not int
            or multiplicity < 1
            or electrons <= 0
            or multiplicity > electrons + 1
            or (electrons - multiplicity + 1) % 2
        ):
            raise ValueError(
                "molecular spin multiplicity is incompatible with electrons"
            )
        if method == "rhf" and multiplicity != 1:
            raise ValueError("RHF requires singlet multiplicity")
        if not isinstance(basis, str) or not basis.strip():
            raise ValueError("molecular basis must be a nonempty Gaussian basis name")
        if backend not in ("cpu", "cuda"):
            raise ValueError("molecular backend must be 'cpu' or 'cuda'")
        if type(device) is not int or not 0 <= device < 2**31:
            raise ValueError("molecular device must be a nonnegative int32")
        if any(
            type(value) is not int or not 0 < value < 2**32 - 1
            for value in (max_iterations, diis_history)
        ):
            raise ValueError(
                "molecular iteration counts must be positive uint32 values"
            )
        for name, value in (
            ("energy_tolerance", energy_tolerance),
            ("density_tolerance", density_tolerance),
            ("screening_tolerance", screening_tolerance),
        ):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"molecular {name} must be finite and positive")
        if resource_budget is not None and not isinstance(
            resource_budget, ResourceBudget
        ):
            raise TypeError("molecular resource_budget must be a ResourceBudget")
        if not isinstance(density_fitting, str) or density_fitting.lower() not in (
            "none",
            "auto",
            "cpu",
            "cpu_reference",
            "cuda",
        ):
            raise ValueError("unsupported molecular density_fitting mode")
        density_fitting = density_fitting.lower()
        if auxiliary_basis is not None and (
            not isinstance(auxiliary_basis, str)
            or not auxiliary_basis.strip()
            or density_fitting == "none"
        ):
            raise ValueError("auxiliary_basis requires enabled density fitting")

        if ks_options is not None:
            from generativeqc.ks import KsOptions

            if not isinstance(ks_options, KsOptions):
                raise TypeError("molecular ks_options must be a KsOptions record")
            if (
                ks_options.functional is not None
                or ks_options.composition is not None
                or ks_options._method_ir is not None
            ):
                raise ValueError(
                    "ks_options controls grid/schedule; XC belongs to method"
                )
        object.__setattr__(self, "_numbers", numbers)
        object.__setattr__(self, "_method", method)
        object.__setattr__(self, "_basis", basis)
        object.__setattr__(self, "_charge", charge)
        object.__setattr__(self, "_multiplicity", multiplicity)
        object.__setattr__(self, "_density_fitting", density_fitting)
        object.__setattr__(self, "_auxiliary_basis", auxiliary_basis)
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_device", device)
        object.__setattr__(
            self,
            "_controls",
            (
                ("max_iterations", max_iterations),
                ("diis_history", diis_history),
                ("energy_tolerance", float(energy_tolerance)),
                ("density_tolerance", float(density_tolerance)),
                ("screening_tolerance", float(screening_tolerance)),
            ),
        )
        object.__setattr__(self, "_resource_budget", resource_budget)
        object.__setattr__(self, "_ks_options", ks_options)

    def _payload(self) -> dict[str, Any]:
        payload = {
            "schema": "generativeqc.array-api.molecular-source/v1",
            "native_method": (
                self._method
                if isinstance(self._method, str)
                else f"methodir:{self._method.identity}"
            ),
            "atomic_numbers": self._numbers,
            "basis": self._basis,
            "charge": self._charge,
            "multiplicity": self._multiplicity,
            "density_fitting": self._density_fitting,
            "auxiliary_basis": self._auxiliary_basis,
            "backend": self._backend,
            "device": self._device,
            "controls": dict(self._controls),
            "precision": "fp64",
            "approximation": "native-calculator-selected",
            "derivative": "native-analytic-nuclear-first-order",
            "resource_budget": (
                None if self._resource_budget is None else asdict(self._resource_budget)
            ),
        }
        if not isinstance(self._method, str):
            # Canonical science, not a user label, determines deduplication/cache IDs.
            payload["method_ir_identity"] = self._method.identity
        if self._ks_options is not None:
            payload["ks_options"] = asdict(self._ks_options)
        return payload

    def _calculator(self) -> Any:
        from generativeqc import Calculator

        # Native Calculator retains the only scientific method registry and
        # exact-geometry basis/ECP/KS capability gate; do not preflight at a
        # fictitious all-zero geometry.
        return Calculator(
            method=self._method,
            basis=self._basis,
            device=self._backend,
            device_id=self._device,
            precision="fp64",
            density_fitting=self._density_fitting,
            auxiliary_basis=self._auxiliary_basis,
            ks_options=self._ks_options,
            initial_guess=None,
            resource_budget=self._resource_budget,
            **dict(self._controls),
        )

    def _atoms(self, coordinates: np.ndarray) -> tuple[Any, ...]:
        return tuple(zip(self._numbers, coordinates, strict=True))

    def _coordinates(self, value: Any) -> np.ndarray:
        # Compiled TensorIR already returns NumPy arrays on this host-staged
        # boundary. Avoid a redundant eager Array API conversion on that path.
        coordinates = (
            value
            if isinstance(value, np.ndarray)
            else np.asarray(eager.call("asarray", value))
        )
        if coordinates.dtype != np.float64:
            raise TypeError("molecular coordinates must be float64")
        if coordinates.shape != (len(self._numbers), 3):
            raise ValueError(
                "molecular coordinates must have shape (number of nuclei, 3)"
            )
        if not np.isfinite(coordinates).all():
            raise ValueError("molecular coordinates must be finite")
        return np.array(coordinates, copy=True, order="C")

    def _result_values(
        self, result: Any, geometry: np.ndarray, derivative: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        expected_backend = "cpu_reference" if self._backend == "cpu" else "cuda"
        if not result.converged or result.executed_backend != expected_backend:
            raise RuntimeError(
                "molecular source did not converge on the selected backend"
            )
        energy = np.asarray(result.energy, dtype=np.float64)
        gradient = None
        if derivative:
            if result.forces is None:
                raise ArithmeticError(
                    "molecular source returned no analytic nuclear gradient"
                )
            gradient = -np.asarray(result.forces, dtype=np.float64)
        if (
            energy.shape
            or not np.isfinite(energy)
            or (
                gradient is not None
                and (
                    gradient.shape != geometry.shape or not np.isfinite(gradient).all()
                )
            )
        ):
            raise ArithmeticError(
                "molecular source returned invalid energy/nuclear gradient"
            )
        return energy, gradient

    def __call__(self, coordinates: Any) -> Any:
        """Evaluate native total energy, or bind a symbolic molecular source.

        Eager calls release their native owner before returning detached CPU
        storage. Compiled calls retain owners until the compiled function is
        closed; lowering alone never loads a native library or runs SCF.
        """
        if isinstance(coordinates, VibeArray):
            capture = _CAPTURE.get()
            if capture is None:
                raise NotImplementedError(
                    "molecular sources require public compile/AD capture"
                )
            return capture.bind(self, coordinates)
        from generativeqc_compiler.array_api.trace import active_capture

        if active_capture():
            raise TypeError("molecular capture requires symbolic coordinates")
        coordinates = self._coordinates(coordinates)
        calculator = self._calculator()
        try:
            result = calculator.singlepoint(
                self._atoms(coordinates),
                charge=self._charge,
                multiplicity=self._multiplicity,
                properties=("energy",),
            )
            return eager.wrap(self._result_values(result, coordinates, False)[0])
        finally:
            calculator.clear_cache()


@dataclass(frozen=True)
class _MolecularSource:
    operator: MolecularModel
    geometry: Program
    energy: str
    gradient: str
    snapshot: str

    @property
    def names(self) -> tuple[str, ...]:
        return self.energy, self.gradient, self.snapshot

    def payload(self) -> dict[str, Any]:
        return {**self.operator._payload(), "geometry": self.geometry.logical_hash}


class _MolecularCapture:
    def __init__(self, backend: str, device: int, input_names: set[str]) -> None:
        if any(name.startswith(_PREFIX) for name in input_names):
            raise ValueError(f"compiled inputs cannot use reserved prefix {_PREFIX}")
        self.backend, self.device = backend, device
        self.sources: list[_MolecularSource] = []

    def __enter__(self) -> Self:
        self.token = _CAPTURE.set(self)
        return self

    def __exit__(self, *unused: object) -> None:
        _CAPTURE.reset(self.token)

    def bind(self, operator: MolecularModel, coordinates: VibeArray) -> VibeArray:
        if operator._backend != self.backend or operator._device != self.device:
            raise ValueError(
                "molecular source and compiled algebra must use the same backend/device"
            )
        if coordinates.dtype != "float64":
            raise TypeError("molecular coordinates must be float64")
        if coordinates.shape != (len(operator._numbers), 3):
            raise ValueError(
                "molecular coordinates must have shape (number of nuclei, 3)"
            )
        if not symbolic._is_generic_array(coordinates):
            indices = coordinates.node.spec.indices
            if tuple(index.space.kind for index in indices) != (
                "atom",
                "cartesian",
            ) or any(
                index.start != 0
                or index.stop != index.space.size
                or index.selection is not None
                or index.space.spin is not None
                for index in indices
            ):
                raise ValueError(
                    "molecular scientific coordinates require complete atom/cartesian domains"
                )
        geometry = Program({"coordinates": coordinates.node})
        if any(
            node.op == "input" and node.attrs["name"].startswith(_PREFIX)
            for node in geometry.live_nodes
        ):
            raise NotImplementedError(
                "nested molecular geometry dependencies are unsupported"
            )
        payload = {**operator._payload(), "geometry": geometry.logical_hash}
        identity = canonical_hash(payload)
        source = next(
            (
                source
                for source in self.sources
                if canonical_hash(source.payload()) == identity
            ),
            None,
        )
        if source is None:
            if len(self.sources) >= 16:
                raise ValueError(
                    "compiled molecular source count exceeds the 16-source budget"
                )
            prefix = f"{_PREFIX}{identity}_"
            source = _MolecularSource(
                operator,
                geometry,
                prefix + "energy",
                prefix + "gradient",
                prefix + "snapshot",
            )
            self.sources.append(source)
        energy = input_array(
            source.energy, TensorSpec((), role="input", differentiable=False)
        )
        if not any(node.spec.differentiable for node in geometry.live_nodes):
            return energy
        detached_spec = replace(
            coordinates.node.spec, role="input", differentiable=False
        )
        gradient = input_array(source.gradient, detached_spec)
        snapshot = input_array(source.snapshot, detached_spec)
        # This exact local jet has value E at q=q0 and derivative g*dq.
        # g and q0 are immutable native snapshots, not differentiated SCF state.
        return energy + symbolic.sum(gradient * (coordinates - snapshot))


class _MolecularRuntime:
    def __init__(self) -> None:
        self.sources: dict[str, _MolecularSource] = {}
        self.calculators: dict[str, Any] = {}
        self.diagnostics: dict[str, dict[str, Any]] = {}
        self.bound_programs: dict[str, Program] = {}
        self.profile_enabled = os.environ.get("GENERATIVEQC_ARRAY_API_PROFILE") == "1"
        self.profile_totals_ns: dict[str, int] = {}
        self.profile_counts: dict[str, int] = {}

    def record_profile(self, stage: str, elapsed_ns: int) -> None:
        """Collect opt-in inclusive CPU wall times; never claim GPU kernel timing."""
        if not self.profile_enabled:
            return
        self.profile_totals_ns[stage] = (
            self.profile_totals_ns.get(stage, 0) + elapsed_ns
        )
        self.profile_counts[stage] = self.profile_counts.get(stage, 0) + 1

    def register(self, sources: list[_MolecularSource]) -> None:
        for source in sources:
            previous = self.sources.get(source.energy)
            if previous is not None and previous.payload() != source.payload():
                raise ValueError("molecular source identity collision")
            self.sources[source.energy] = source

    def bind_snapshots(self, program: Program) -> Program:
        """Evaluate jets at their bound point *after* first-order AD lowering.

        At q=q0 the jet displacement is identically zero. Substituting exact
        zero avoids duplicate geometry arithmetic and roundoff changing the
        native energy, including in nonlinear compositions. AD runs before
        this binding, so its g*dq term is retained. Reuse the canonical DAG
        rebuilder; no runtime-owned primitive or derivative rule is added.
        """
        from generativeqc_compiler.tensor import constant
        from generativeqc_compiler.tensor.ad_program import _rebuild

        if not self.sources:
            return program
        cached = self.bound_programs.get(program.logical_hash)
        if cached is not None:
            return cached
        snapshots = {source.snapshot for source in self.sources.values()}
        replacements = {}
        for node in program.nodes:
            if (
                node.op == "add"
                and node.attrs["coefficients"] == ((1, 1), (-1, 1))
                and len(node.inputs) == 2
                and node.inputs[1].op == "input"
                and node.inputs[1].attrs["name"] in snapshots
            ):
                replacements[node] = constant(
                    (0,) * node.spec.size,
                    replace(node.spec, role="constant", differentiable=False),
                )
        bound = _rebuild(program, replacements) if replacements else program
        self.bound_programs[program.logical_hash] = bound
        return bound

    def materialize(
        self, executor: Any, program: Program, feeds: dict[str, Any]
    ) -> None:
        profiling = self.profile_enabled
        mark = time.perf_counter_ns() if profiling else 0
        required = {
            node.attrs["name"] for node in program.live_nodes if node.op == "input"
        }
        if profiling:
            self.record_profile("source_input_scan", time.perf_counter_ns() - mark)
        for source in self.sources.values():
            needed = required.intersection(source.names)
            if not needed or needed <= feeds.keys():
                continue
            mark = time.perf_counter_ns() if profiling else 0
            coordinate_node = source.geometry.outputs["coordinates"]
            geometry = (
                feeds[coordinate_node.attrs["name"]]
                if coordinate_node.op == "input"
                else executor.execute(source.geometry, feeds)["coordinates"]
            )
            geometry = source.operator._coordinates(geometry)
            if profiling:
                self.record_profile("source_geometry", time.perf_counter_ns() - mark)
            mark = time.perf_counter_ns() if profiling else 0
            calculator = self.calculators.get(source.energy)
            if calculator is None:
                calculator = source.operator._calculator()
                self.calculators[source.energy] = calculator
            if profiling:
                self.record_profile(
                    "source_calculator_lookup", time.perf_counter_ns() - mark
                )
            derivative = bool({source.gradient, source.snapshot} & needed)
            mark = time.perf_counter_ns() if profiling else 0
            result = calculator.singlepoint(
                source.operator._atoms(geometry),
                charge=source.operator._charge,
                multiplicity=source.operator._multiplicity,
                properties=("energy", "forces") if derivative else ("energy",),
            )
            if profiling:
                self.record_profile(
                    "source_native_singlepoint", time.perf_counter_ns() - mark
                )
            mark = time.perf_counter_ns() if profiling else 0
            energy, gradient = source.operator._result_values(
                result, geometry, derivative
            )
            # Publish atomically only after the full requested endpoint succeeds.
            feeds[source.energy] = energy
            feeds[source.snapshot] = geometry
            if gradient is not None:
                feeds[source.gradient] = np.array(gradient, copy=True)
            if profiling:
                self.record_profile(
                    "source_result_publication", time.perf_counter_ns() - mark
                )
            mark = time.perf_counter_ns() if profiling else 0
            previous = self.diagnostics.get(source.energy, {})
            self.diagnostics[source.energy] = {
                "source": source.energy,
                "method": (
                    source.operator._method
                    if isinstance(source.operator._method, str)
                    else source.operator._method.identifier
                ),
                "calls": previous.get("calls", 0) + 1,
                "force_calls": previous.get("force_calls", 0) + int(derivative),
                "total_iterations": previous.get("total_iterations", 0)
                + result.iterations,
                "backend": result.executed_backend,
                "native_source_identity": (
                    calculator._library.generativeqc_get_source_identity().decode()
                    if hasattr(calculator, "_library")
                    else None
                ),
                "model_identity": getattr(result, "basis_metadata", {}).get(
                    "model_identity"
                ),
                "iterations": result.iterations,
                "energy_change": result.energy_change,
                "density_rms": result.density_rms,
                "nuclear_gradient": derivative,
                "geometry_hash": canonical_hash(geometry.tolist()),
            }
            if profiling:
                self.record_profile("source_diagnostics", time.perf_counter_ns() - mark)

    def close(self) -> None:
        for calculator in self.calculators.values():
            calculator.clear_cache()
        self.calculators.clear()
        self.sources.clear()
        self.bound_programs.clear()
