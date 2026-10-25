"""Molecular energy/gradient sources with qualified directional response capture.

The default native source owns SCF, integrals, and analytic nuclear derivatives.
The explicit integral-ad source generates the integral/normalization pullback
for geometry and basis inputs and solves its primal with the shared RHF driver.
TensorIR owns the surrounding expression and its chain rule. A source's first-order
jet is bound at each actual geometry, never a differentiated iteration history
or a numerical surrogate for the method's energy away from that geometry.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping
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
    from generativeqc_compiler.method import MethodIR

    from generativeqc.ks import KsOptions


_PREFIX = "__gqc_molecular_"
_SOURCE_ORDER: ContextVar[int] = ContextVar(
    "generativeqc_molecular_source_order", default=1
)
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
    and its chain rule. Native second order requires the calculator's qualified
    HVP endpoint; integral second order retains one converged state. No new
    electronic method, numerical-gradient fallback or SCF-history derivative is
    invented. Coordinates are FP64 (natoms, 3) in Bohr; energy is in Hartree
    and complete native analytic gradients are in Hartree/Bohr.

    ``differentiation="integral-ad"`` instead selects FP64 conventional RHF/UHF
    through generated Gaussian integrals, normalization and the stationary SCF
    Lagrangian. Its optional flat ``exponents`` and ``coefficients`` call inputs
    follow shell/primitive order. This mode is unscreened; JVP of grad supports second order, with
    the default IntegralRHF byte/work/compiler bounds; it calls no force endpoint.

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
    _differentiation: str
    _integral_options: tuple[tuple[str, Any], ...]

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
        differentiation: str = "native",
        integral_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Bind a method selector and immutable derivative source contract.

        Admission validates static nuclei, spin, resource and solver settings
        without loading native code. Unsupported method/basis/device or missing
        analytic-gradient capabilities fail through the existing Calculator
        at execution, with no reference/CPU or finite-difference fallback.
        Compiled owners must be closed to release native workspaces.
        """
        from generativeqc_compiler.common.resources import ResourceBudget
        from generativeqc_compiler.method import MethodIR

        from generativeqc.elements import atomic_number

        if differentiation not in ("native", "integral-ad"):
            raise ValueError("differentiation must be 'native' or 'integral-ad'")
        if integral_options is not None and not isinstance(integral_options, Mapping):
            raise TypeError("integral_options must be a mapping")
        options = dict(integral_options or {})
        if options and differentiation != "integral-ad":
            raise ValueError("integral_options require differentiation='integral-ad'")
        allowed = {
            "max_bytes",
            "max_work",
            "compiler_work",
            "compile_timeout",
            "integral_storage",
        }
        if options.keys() - allowed:
            raise ValueError(
                "unknown integral_options: "
                + ", ".join(sorted(options.keys() - allowed))
            )
        for name in ("max_bytes", "max_work", "compiler_work"):
            if name in options and (
                type(options[name]) is not int or not 0 < options[name] < 2**63
            ):
                raise ValueError(name + " must be a positive int64")
        if "compile_timeout" in options and (
            type(options["compile_timeout"]) not in (int, float)
            or not math.isfinite(options["compile_timeout"])
            or options["compile_timeout"] <= 0
        ):
            raise ValueError("compile_timeout must be finite and positive")
        if options.get("integral_storage", "dense") not in ("dense", "direct"):
            raise ValueError("integral_storage must be dense or direct")
        if options.get("integral_storage") == "direct" and backend != "cpu":
            raise ValueError("direct integral AD storage currently requires CPU")
        object.__setattr__(self, "_integral_options", tuple(sorted(options.items())))
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
                or value < 0
                or (
                    value == 0
                    and not (
                        name == "screening_tolerance"
                        and differentiation == "integral-ad"
                    )
                )
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
        if differentiation == "integral-ad":
            if (
                method not in ("rhf", "uhf")
                or density_fitting != "none"
                or (method == "rhf" and multiplicity != 1)
                or ks_options is not None
                or resource_budget is not None
            ):
                raise ValueError(
                    "integral-ad requires direct RHF/UHF without KS/resource overrides; "
                    "use IntegralHF for explicit integral execution budgets"
                )
            if max_iterations > 10000 or diis_history > 32:
                raise ValueError(
                    "integral-ad permits at most 10000 iterations and 32 DIIS vectors"
                )
            if screening_tolerance not in (0, 1e-14):
                raise ValueError(
                    "integral-ad is unscreened; screening_tolerance must be zero or default"
                )
            # The shared default is accepted for convenience, but the explicit
            # mode always executes the complete unscreened differentiable sum.
            screening_tolerance = 0.0
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
        object.__setattr__(self, "_differentiation", differentiation)

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
        if self._differentiation == "integral-ad":
            payload["derivative"] = f"stationary-{self._method}-through-integral-ad-v2"
            payload["integral_options"] = dict(self._integral_options)
        if not isinstance(self._method, str):
            # Canonical science, not a user label, determines deduplication/cache IDs.
            payload["method_ir_identity"] = self._method.identity
        if self._ks_options is not None:
            payload["ks_options"] = asdict(self._ks_options)
        return payload

    def _calculator(self) -> Any:
        if self._differentiation == "integral-ad":
            return _IntegralRHFSource(self)
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

    def __call__(
        self, coordinates: Any, *, exponents: Any = None, coefficients: Any = None
    ) -> Any:
        """Evaluate native total energy, or bind a symbolic molecular source.

        Eager calls release their native owner before returning detached CPU
        storage. Compiled calls retain owners until the compiled function is
        closed; lowering alone never loads a native library or runs SCF.
        Integral AD accepts optional finite FP64 rank-one basis parameters;
        all other modes reject them. Omitted parameters use the named basis.
        """
        parameters = {
            name: value
            for name, value in (
                ("exponents", exponents),
                ("coefficients", coefficients),
            )
            if value is not None
        }
        if parameters and self._differentiation != "integral-ad":
            raise ValueError(
                "continuous basis inputs require differentiation='integral-ad'"
            )
        if isinstance(coordinates, VibeArray):
            capture = _CAPTURE.get()
            if capture is None:
                raise NotImplementedError(
                    "molecular sources require public compile/AD capture"
                )
            return capture.bind(self, coordinates, parameters)
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
                **parameters,
            )
            return eager.wrap(self._result_values(result, coordinates, False)[0])
        finally:
            calculator.clear_cache()


class _IntegralRHFSource:
    """Adapt the explicit integral AD chain to the existing capture lifetime.

    This is a source execution adapter, not a delegation to Calculator forces.
    It retains raw basis parameters and differentiates the actual generated
    integrals consumed by the shared native SCF solve.
    """

    def __init__(self, operator: MolecularModel, *, derivative_order: int = 1) -> None:
        from .integral_autodiff import IntegralHF

        controls = dict(operator._controls)
        self.owner = IntegralHF(
            operator._numbers,
            basis=operator._basis,
            charge=operator._charge,
            method=operator._method,
            spin_multiplicity=operator._multiplicity,
            backend=operator._backend,
            device=operator._device,
            derivative_order=derivative_order,
            max_iterations=controls["max_iterations"],
            diis_history=controls["diis_history"],
            energy_tolerance=controls["energy_tolerance"],
            density_tolerance=controls["density_tolerance"],
            **dict(operator._integral_options),
        )
        self._library = self.owner._calculator._library

    def singlepoint(
        self,
        atoms: Any,
        *,
        charge: int,
        multiplicity: int,
        properties: tuple[str, ...],
        **parameters: Any,
    ) -> Any:
        from types import SimpleNamespace

        if charge != self.owner.charge or multiplicity != self.owner.spin_multiplicity:
            raise ValueError("integral AD source molecular identity changed")
        coordinates = np.array([position for _, position in atoms], dtype=np.float64)
        derivative = "forces" in properties
        directions = parameters.pop("directions", None)
        with self.owner.prepare(coordinates, **parameters) as state:
            energy = state.energy
            gradient = state.gradient() if derivative else None
            gradient_direction = None
            if directions is not None:
                from .integral_autodiff import IntegralGradient

                gradient_direction = state.hvp(
                    IntegralGradient(
                        directions["coordinates"],
                        directions.get(
                            "exponents", np.zeros_like(self.owner.exponents)
                        ),
                        directions.get(
                            "coefficients", np.zeros_like(self.owner.coefficients)
                        ),
                    )
                )
            diagnostic = state.diagnostics()
        return SimpleNamespace(
            gradient_direction=gradient_direction,
            energy=energy,
            forces=None if gradient is None else -gradient.coordinates,
            basis_gradients=gradient,
            integral_ad_diagnostics=diagnostic,
            converged=True,
            executed_backend="cpu_reference" if self.owner.backend == "cpu" else "cuda",
            iterations=diagnostic["iterations"],
            energy_change=diagnostic["energy_change"],
            density_rms=diagnostic["density_rms"],
        )

    def clear_cache(self) -> None:
        self.owner.close()


@dataclass(frozen=True)
class _MolecularSource:
    operator: MolecularModel
    geometry: Program
    energy: str
    gradient: str
    snapshot: str
    parameter_names: tuple[str, ...] = ()
    derivative_order: int = 1

    @property
    def names(self) -> tuple[str, ...]:
        return (
            self.energy,
            self.gradient,
            self.snapshot,
            *self.response_names,
            *(
                name + "_" + parameter
                for parameter in self.parameter_names
                for name in (self.gradient, self.snapshot)
            ),
        )

    @property
    def gradient_names(self) -> tuple[str, ...]:
        return (
            self.gradient,
            *(self.gradient + "_" + name for name in self.parameter_names),
        )

    @property
    def response_names(self) -> tuple[str, ...]:
        return (
            tuple("d_" + name for name in self.gradient_names)
            if self.derivative_order == 2
            else ()
        )

    @property
    def snapshot_names(self) -> tuple[str, ...]:
        return (
            self.snapshot,
            *(self.snapshot + "_" + name for name in self.parameter_names),
        )

    def payload(self) -> dict[str, Any]:
        return {
            **self.operator._payload(),
            "geometry": self.geometry.logical_hash,
            "derivative_order": self.derivative_order,
        }


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

    def bind(
        self,
        operator: MolecularModel,
        coordinates: VibeArray,
        parameters: dict[str, Any] | None = None,
    ) -> VibeArray:
        parameters = parameters or {}
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
        for name, value in parameters.items():
            if (
                not isinstance(value, VibeArray)
                or value.dtype != "float64"
                or len(value.shape) != 1
                or not symbolic._is_generic_array(value)
            ):
                raise ValueError(
                    f"{name} must be a generic rank-one float64 captured array"
                )
        geometry = Program(
            {
                "coordinates": coordinates.node,
                **{name: value.node for name, value in parameters.items()},
            }
        )
        if any(
            node.op == "input" and node.attrs["name"].startswith(_PREFIX)
            for node in geometry.live_nodes
        ):
            raise NotImplementedError(
                "nested molecular geometry dependencies are unsupported"
            )
        order = _SOURCE_ORDER.get()
        payload = {
            **operator._payload(),
            "geometry": geometry.logical_hash,
            "derivative_order": order,
        }
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
                tuple(parameters),
                order,
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
        # The second transform differentiates the gradient with an explicitly
        # supplied HVP seed. Snapshot coordinates stay fixed execution data.
        gradient_spec = (
            replace(detached_spec, role="parameter", differentiable=True)
            if order == 2
            else detached_spec
        )
        gradient = input_array(source.gradient, gradient_spec)
        snapshot = input_array(source.snapshot, detached_spec)
        # This exact local jet has value E at q=q0 and derivative g*dq.
        # g and q0 are immutable native snapshots, not differentiated SCF state.
        result = energy + symbolic.sum(gradient * (coordinates - snapshot))
        for name, value in parameters.items():
            spec = replace(value.node.spec, role="input", differentiable=False)
            gradient_spec = (
                replace(spec, role="parameter", differentiable=True)
                if order == 2
                else spec
            )
            basis_gradient = input_array(source.gradient + "_" + name, gradient_spec)
            basis_snapshot = input_array(source.snapshot + "_" + name, spec)
            result = result + symbolic.sum(basis_gradient * (value - basis_snapshot))
        return result


class _MolecularRuntime:
    def __init__(self) -> None:
        self.sources: dict[str, _MolecularSource] = {}
        self.calculators: dict[str, Any] = {}
        self.diagnostics: dict[str, dict[str, Any]] = {}
        self.bound_programs: dict[str, Program] = {}
        self.directions: dict[str, np.ndarray] = {}
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
        """Evaluate source expansions at their snapshot after all requested AD transforms.

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
        snapshots = {
            name for source in self.sources.values() for name in source.snapshot_names
        }
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
            parameter_values = {}
            if source.parameter_names:
                parameter_values = executor.execute(source.geometry, feeds)
            geometry = (
                parameter_values["coordinates"]
                if parameter_values
                else feeds[coordinate_node.attrs["name"]]
                if coordinate_node.op == "input"
                else executor.execute(source.geometry, feeds)["coordinates"]
            )
            geometry = source.operator._coordinates(geometry)
            if profiling:
                self.record_profile("source_geometry", time.perf_counter_ns() - mark)
            mark = time.perf_counter_ns() if profiling else 0
            calculator = self.calculators.get(source.energy)
            if calculator is None:
                calculator = (
                    source.operator._calculator()
                    if source.derivative_order == 1
                    or source.operator._differentiation != "integral-ad"
                    else _IntegralRHFSource(source.operator, derivative_order=2)
                )
                self.calculators[source.energy] = calculator
            if profiling:
                self.record_profile(
                    "source_calculator_lookup", time.perf_counter_ns() - mark
                )
            derivative = bool((set(source.names) - {source.energy}) & needed)
            second = bool(set(source.response_names) & needed)
            mark = time.perf_counter_ns() if profiling else 0
            directions = None
            if second:
                if (
                    source.operator._differentiation != "integral-ad"
                    and "hvp" not in calculator.second_order_capabilities
                ):
                    raise NotImplementedError(
                        "this molecular context has no qualified second-order response"
                    )
                from generativeqc_compiler.tensor import linearize

                active = tuple(
                    node.attrs["name"]
                    for node in source.geometry.nodes
                    if node.op == "input" and node.attrs["name"] in self.directions
                )
                active = tuple(dict.fromkeys(active))
                motion = linearize(source.geometry, active)
                motion_feeds = dict(feeds)
                for seed, name in motion.input_map.items():
                    motion_feeds[seed] = self.directions[name]
                values = executor.execute(motion.program, motion_feeds)
                directions = {
                    name: values[generated]
                    for generated, name in motion.output_map.items()
                }
            arguments = {
                name: parameter_values[name] for name in source.parameter_names
            }
            if (
                directions is not None
                and source.operator._differentiation == "integral-ad"
            ):
                arguments["directions"] = directions
            result = calculator.singlepoint(
                source.operator._atoms(geometry),
                charge=source.operator._charge,
                multiplicity=source.operator._multiplicity,
                properties=("energy", "forces") if derivative else ("energy",),
                **arguments,
            )
            native_hvp = None
            native_response = None
            if second and source.operator._differentiation != "integral-ad":
                native_response = calculator.hessian_vector_product(
                    source.operator._atoms(geometry),
                    directions["coordinates"],
                    charge=source.operator._charge,
                    multiplicity=source.operator._multiplicity,
                )
                native_hvp = native_response.value
            if profiling:
                self.record_profile(
                    "source_native_singlepoint", time.perf_counter_ns() - mark
                )
            mark = time.perf_counter_ns() if profiling else 0
            energy, gradient = source.operator._result_values(
                result, geometry, derivative
            )
            # Publish atomically only after the full requested endpoint succeeds.
            publication = {source.energy: energy, source.snapshot: geometry}
            if gradient is not None:
                publication[source.gradient] = np.array(gradient, copy=True)
            for name in source.parameter_names:
                publication[source.snapshot + "_" + name] = np.array(
                    parameter_values[name], copy=True
                )
                if derivative:
                    value = np.asarray(getattr(result.basis_gradients, name))
                    if (
                        value.dtype != np.float64
                        or value.shape != parameter_values[name].shape
                        or not np.isfinite(value).all()
                    ):
                        raise ArithmeticError(
                            "integral AD source returned invalid basis derivatives"
                        )
                    publication[source.gradient + "_" + name] = np.array(
                        value, copy=True
                    )
            if second:
                for name, generated in zip(
                    ("coordinates", *source.parameter_names),
                    source.response_names,
                    strict=True,
                ):
                    value = (
                        native_hvp
                        if native_hvp is not None
                        else getattr(result.gradient_direction, name)
                    )
                    value = np.asarray(value)
                    expected = (
                        geometry.shape
                        if name == "coordinates"
                        else parameter_values[name].shape
                    )
                    if (
                        value.shape != expected
                        or value.dtype != np.float64
                        or not np.isfinite(value).all()
                    ):
                        raise ArithmeticError(
                            "molecular source returned invalid Hessian-vector product"
                        )
                    publication[generated] = np.array(value, copy=True)
            feeds.update(publication)
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
                "hvp_calls": previous.get("hvp_calls", 0) + int(second),
                "iteration_count_scope": "energy-and-gradient-source",
                "native_response_diagnostics": None
                if native_response is None
                else native_response.diagnostics,
                "force_calls": previous.get("force_calls", 0)
                + int(derivative and source.operator._differentiation == "native"),
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
            if source.operator._differentiation == "integral-ad":
                self.diagnostics[source.energy]["integral_ad"] = dict(
                    result.integral_ad_diagnostics
                )
            if profiling:
                self.record_profile("source_diagnostics", time.perf_counter_ns() - mark)

    def close(self) -> None:
        for calculator in self.calculators.values():
            calculator.clear_cache()
        self.calculators.clear()
        self.sources.clear()
        self.bound_programs.clear()
