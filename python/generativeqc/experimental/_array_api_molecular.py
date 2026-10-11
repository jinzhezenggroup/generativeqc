"""Explicit native molecular source cuts for first-order Array API capture.

The native method owns SCF, integrals, and analytic nuclear derivatives. TensorIR
owns only the surrounding expression and its chain rule. A source's first-order
jet is bound at each actual geometry, never a differentiated iteration history
or a numerical surrogate for the method's energy away from that geometry.
"""

from __future__ import annotations

import math
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from generativeqc_compiler.array_api import VibeArray, input_array
from generativeqc_compiler.array_api import namespace as symbolic
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.tensor import Program, TensorSpec
from typing_extensions import Self

from . import _array_api_eager as eager

_PREFIX = "__gqc_molecular_"
_CAPTURE: ContextVar[_MolecularCapture | None] = ContextVar(
    "generativeqc_molecular_capture", default=None
)


@dataclass(frozen=True, init=False)
class RHF:
    """Immutable, all-electron closed-shell RHF energy operator.

    Call with float64 coordinates shaped ``(len(elements), 3)`` in Bohr. The
    result is a scalar Hartree Array API array. Inside ``compile``/first-order
    AD, the existing native RHF endpoint supplies the converged total energy
    and, when needed, its complete analytic nuclear gradient (including Pulay
    and nuclear repulsion terms), not a density derivative or finite difference.

    CPU and CUDA are explicit selections. Compiled surrounding algebra must
    select the same backend and logical device. CUDA currently stages host
    coordinates/results; no resident GPU tensor or performance parity is
    implied. Native admission, nonconvergence, and resource failures propagate
    without fallback. ``resource_budget`` bounds the native endpoint separately
    from the TensorIR compiler's algebra/workspace limits.
    """

    _numbers: tuple[int, ...]
    _basis: str
    _charge: int
    _backend: str
    _device: int
    _controls: tuple[tuple[str, Any], ...]
    _resource_budget: Any

    def __init__(
        self,
        elements: tuple[str | int, ...],
        *,
        basis: str = "sto-3g",
        charge: int = 0,
        backend: str = "cpu",
        device: int = 0,
        max_iterations: int = 100,
        diis_history: int = 8,
        energy_tolerance: float = 1e-12,
        density_tolerance: float = 1e-10,
        screening_tolerance: float = 1e-14,
        resource_budget: Any = None,
    ) -> None:
        """Bind immutable nuclei/basis/solver policy without loading native code.

        ``elements`` is a nonempty tuple of symbols or atomic numbers, at most
        256 atoms. Only positive even all-electron populations are admitted;
        multiplicity is fixed to one. Named all-electron basis admission is
        delegated to the native calculator. ECP and fitted approximations are
        not represented by this first molecular source contract.
        """
        from generativeqc_compiler.common.resources import ResourceBudget

        from generativeqc.elements import atomic_number

        if not isinstance(elements, tuple) or not 1 <= len(elements) <= 256:
            raise ValueError("RHF elements must be a tuple of 1 to 256 nuclei")
        numbers = tuple(atomic_number(element) for element in elements)
        if type(charge) is not int or not -(2**31) <= charge < 2**31:
            raise ValueError("RHF charge must be an int32")
        electrons = sum(numbers) - charge
        if electrons <= 0 or electrons % 2:
            raise ValueError("RHF requires a positive even electron count")
        if not isinstance(basis, str) or not basis.strip():
            raise ValueError("RHF basis must be a nonempty all-electron basis name")
        if backend not in ("cpu", "cuda"):
            raise ValueError("RHF backend must be 'cpu' or 'cuda'")
        if type(device) is not int or not 0 <= device < 2**31:
            raise ValueError("RHF device must be a nonnegative int32")
        if any(
            type(value) is not int or not 0 < value < 2**32 - 1
            for value in (max_iterations, diis_history)
        ):
            raise ValueError("RHF iteration counts must be positive uint32 values")
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
                raise ValueError(f"RHF {name} must be finite and positive")
        if resource_budget is not None and not isinstance(
            resource_budget, ResourceBudget
        ):
            raise TypeError("RHF resource_budget must be a ResourceBudget")
        object.__setattr__(self, "_numbers", numbers)
        object.__setattr__(self, "_basis", basis)
        object.__setattr__(self, "_charge", charge)
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

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "generativeqc.array-api.rhf-source/v1",
            "native_method": "rhf",
            "atomic_numbers": self._numbers,
            "basis": self._basis,
            "charge": self._charge,
            "backend": self._backend,
            "device": self._device,
            "controls": dict(self._controls),
            "precision": "fp64",
            "approximation": "all-electron-exact-jk",
            "derivative": "native-analytic-nuclear-first-order",
            "resource_budget": (
                None if self._resource_budget is None else asdict(self._resource_budget)
            ),
        }

    def _calculator(self) -> Any:
        from generativeqc import Calculator

        calculator = Calculator(
            method="rhf",
            basis=self._basis,
            device=self._backend,
            device_id=self._device,
            precision="fp64",
            density_fitting="none",
            initial_guess=None,
            resource_budget=self._resource_budget,
            **dict(self._controls),
        )
        # Reject ECP metadata before executing the all-electron source contract.
        try:
            calculator.resolved_model(
                self._atoms(np.zeros((len(self._numbers), 3))), charge=self._charge
            )
        except BaseException:
            calculator.clear_cache()
            raise
        return calculator

    def _atoms(self, coordinates: np.ndarray) -> tuple[Any, ...]:
        return tuple(zip(self._numbers, coordinates, strict=True))

    def _coordinates(self, value: Any) -> np.ndarray:
        coordinates = np.asarray(eager.call("asarray", value))
        if coordinates.dtype != np.float64:
            raise TypeError("RHF coordinates must be float64")
        if coordinates.shape != (len(self._numbers), 3):
            raise ValueError("RHF coordinates must have shape (number of nuclei, 3)")
        if not np.isfinite(coordinates).all():
            raise ValueError("RHF coordinates must be finite")
        return np.array(coordinates, copy=True, order="C")

    def _result_values(
        self, result: Any, geometry: np.ndarray, derivative: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        expected_backend = "cpu_reference" if self._backend == "cpu" else "cuda"
        if not result.converged or result.executed_backend != expected_backend:
            raise RuntimeError("RHF source did not converge on the selected backend")
        energy = np.asarray(result.energy, dtype=np.float64)
        gradient = None
        if derivative:
            if result.forces is None:
                raise ArithmeticError(
                    "RHF source returned no analytic nuclear gradient"
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
            raise ArithmeticError("RHF source returned invalid energy/nuclear gradient")
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
                    "RHF sources require public compile/AD capture"
                )
            return capture.bind(self, coordinates)
        from generativeqc_compiler.array_api.trace import active_capture

        if active_capture():
            raise TypeError("RHF capture requires symbolic coordinates")
        coordinates = self._coordinates(coordinates)
        calculator = self._calculator()
        try:
            result = calculator.singlepoint(
                self._atoms(coordinates), charge=self._charge, properties=("energy",)
            )
            return eager.wrap(self._result_values(result, coordinates, False)[0])
        finally:
            calculator.clear_cache()


@dataclass(frozen=True)
class _MolecularSource:
    operator: RHF
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

    def bind(self, operator: RHF, coordinates: VibeArray) -> VibeArray:
        if operator._backend != self.backend or operator._device != self.device:
            raise ValueError(
                "RHF source and compiled algebra must use the same backend/device"
            )
        if coordinates.dtype != "float64":
            raise TypeError("RHF coordinates must be float64")
        if coordinates.shape != (len(operator._numbers), 3):
            raise ValueError("RHF coordinates must have shape (number of nuclei, 3)")
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
                    "RHF scientific coordinates require complete atom/cartesian domains"
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
        required = {
            node.attrs["name"] for node in program.live_nodes if node.op == "input"
        }
        for source in self.sources.values():
            needed = required.intersection(source.names)
            if not needed or needed <= feeds.keys():
                continue
            coordinate_node = source.geometry.outputs["coordinates"]
            geometry = (
                feeds[coordinate_node.attrs["name"]]
                if coordinate_node.op == "input"
                else executor.execute(source.geometry, feeds)["coordinates"]
            )
            geometry = source.operator._coordinates(geometry)
            calculator = self.calculators.get(source.energy)
            if calculator is None:
                calculator = source.operator._calculator()
                self.calculators[source.energy] = calculator
            derivative = bool({source.gradient, source.snapshot} & needed)
            result = calculator.singlepoint(
                source.operator._atoms(geometry),
                charge=source.operator._charge,
                properties=("energy", "forces") if derivative else ("energy",),
            )
            energy, gradient = source.operator._result_values(
                result, geometry, derivative
            )
            # Publish atomically only after the full requested endpoint succeeds.
            feeds[source.energy] = energy
            feeds[source.snapshot] = geometry
            if gradient is not None:
                feeds[source.gradient] = np.array(gradient, copy=True)
            previous = self.diagnostics.get(source.energy, {})
            self.diagnostics[source.energy] = {
                "source": source.energy,
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

    def close(self) -> None:
        for calculator in self.calculators.values():
            calculator.clear_cache()
        self.calculators.clear()
        self.sources.clear()
        self.bound_programs.clear()
