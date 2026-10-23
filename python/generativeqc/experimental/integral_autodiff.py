"""End-to-end HF AD through normalized Gaussian integrals.

This explicit FP64 endpoint evaluates compiler-owned integral expressions,
solves their RHF/UHF problem with the shared native SCF driver, differentiates its
stationary Lagrangian, and pulls the resulting source weights through those
same integral expressions. Coordinates, primitive exponents and raw contraction
coefficients are continuous inputs. No force endpoint, PySCF, numerical
differentiation, or differentiated SCF iteration history is used at runtime.
"""

from __future__ import annotations

import ctypes as ct
import math
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.common.compiler_work import compiler_work_budget
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.common.native_runtime import compile_runtime
from generativeqc_compiler.common.paths import asset_path
from generativeqc_compiler.common.source_cache import cache_source
from generativeqc_compiler.integral.parameter_ad import emit_integral_program
from generativeqc_compiler.integral.shell_spec import cartesian_components
from generativeqc_compiler.method.rhf_integral_ad import (
    emit_spin_integral_weights,
)

from generativeqc import Atom, BasisSet, Calculator, Primitive, Shell, _native
from generativeqc.elements import atomic_number, checked_integer

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "IntegralGradient",
    "IntegralHF",
    "IntegralRHF",
    "IntegralUHF",
    "IntegralValues",
]

_DOUBLE = ct.POINTER(ct.c_double)


def _pointer(array: np.ndarray) -> Any:
    return array.ctypes.data_as(_DOUBLE)


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value)
    if (
        result.dtype != np.float64
        or result.shape != shape
        or not np.isfinite(result).all()
    ):
        raise ValueError(f"{name} must be finite float64 with shape {shape}")
    return np.array(result, copy=True, order="C")


@dataclass(frozen=True)
class IntegralValues:
    """Ordered AO sources: S, h=T+V, chemists' ERIs, and nuclear energy.

    Matrix shapes are (nao, nao), ERIs are (nao, nao, nao, nao); all are FP64.
    Returned arrays are immutable and outlive their owner. Values use atomic
    units and Cartesian CCA order. User-constructed records also accept arbitrary
    source cotangents; integral_vjp validates and copies their arrays.
    """

    overlap: np.ndarray
    hcore: np.ndarray
    eri: np.ndarray
    nuclear: float


@dataclass(frozen=True)
class IntegralGradient:
    """Immutable first derivatives in the caller's unnormalized input convention.

    Coordinates have shape (natoms, 3), in Hartree/Bohr for an energy objective.
    Exponents/coefficients are flat in shell then primitive order. Their units
    are the objective's units per input unit; gradients include contraction and
    primitive normalization. Coefficient scale directions are therefore null.
    """

    coordinates: np.ndarray
    exponents: np.ndarray
    coefficients: np.ndarray


@dataclass(frozen=True, init=False, repr=False, eq=False)
class IntegralHF:
    """Bounded native RHF/UHF differentiated from its Gaussian integral inputs.

    The topology fixes elements, charge and a Cartesian all-electron basis.
    Runtime coordinates (Bohr), positive exponents and signed raw contraction
    coefficients are FP64 inputs. Shell/primitive order is preserved. Each call
    solves a fresh unscreened stationary problem; nonconvergence, invalid
    normalization, singular overlap or exceeded budgets raise before publication.

    Set derivative_order=2 for integral JVPs, energy HVPs and full joint Hessians.
    CPU and CUDA use the same generated scalar algebra. CUDA evaluates integrals,
    J/K and eigensystems on device, with host SCF/Krylov control and small response
    matrix algebra. It stores dense primal ERIs under max_bytes;
    pullbacks stream primitive derivatives without a molecular Jacobian. The
    numeric-array bound excludes compiler memory, Python objects and opaque
    library scratch. max_work bounds primitive evaluations, while compiler_work
    bounds symbolic generation. Backend selection is explicit and never retries with another backend.
    Call close() when finished; results and returned snapshots remain valid.
    Configuration and topology are immutable after construction.
    """

    def __init__(
        self,
        elements: Sequence[str | int],
        *,
        basis: str | BasisSet | Sequence[Shell] = "sto-3g",
        charge: int = 0,
        method: str = "rhf",
        spin_multiplicity: int = 1,
        derivative_order: int = 1,
        backend: str = "cpu",
        device: int = 0,
        cuda_arch: str = "sm_120",
        cache: str | Path = ".artifacts/integral-ad",
        max_bytes: int = 64 << 20,
        max_work: int = 100_000_000,
        compiler_work: int = 5_000_000,
        compile_timeout: float = 300.0,
        max_iterations: int = 200,
        diis_history: int = 8,
        energy_tolerance: float = 1e-13,
        density_tolerance: float = 1e-11,
    ) -> None:
        """Validate the domain and budgets before generating native AD kernels."""
        object.__setattr__(
            self, "numbers", tuple(atomic_number(element) for element in elements)
        )
        object.__setattr__(
            self,
            "charge",
            checked_integer(charge, "ionic charge", low=-(2**31), high=2**31 - 1),
        )
        electrons = sum(self.numbers) - self.charge
        if method not in ("rhf", "uhf"):
            raise ValueError("integral AD method must be rhf or uhf")
        spin = (
            checked_integer(
                spin_multiplicity, "spin multiplicity", low=1, high=2**31 - 1
            )
            - 1
        )
        if (
            not 1 <= len(self.numbers) <= 256
            or electrons <= 0
            or spin > electrons
            or (electrons - spin) % 2
            or (method == "rhf" and spin != 0)
        ):
            raise ValueError("integral AD requires compatible HF spin occupations")
        if type(derivative_order) is not int or derivative_order not in (1, 2):
            raise ValueError("derivative_order must be 1 or 2")
        if backend not in ("cpu", "cuda"):
            raise ValueError("backend must be cpu or cuda")
        checked_integer(device, "device", low=0, high=2**31 - 1)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "spin_multiplicity", spin_multiplicity)
        object.__setattr__(self, "derivative_order", derivative_order)
        object.__setattr__(self, "max_work", max_work)
        for name, value in (
            ("max_bytes", max_bytes),
            ("max_work", max_work),
            ("compiler_work", compiler_work),
            ("max_iterations", max_iterations),
            ("diis_history", diis_history),
        ):
            if type(value) is not int or not 0 < value < 2**63:
                raise ValueError(f"{name} must be a positive int64")
        if max_iterations > 10000:
            raise ValueError("max_iterations must not exceed 10000")
        if diis_history > 32:
            raise ValueError("diis_history must not exceed 32")
        for name, value in (
            ("energy_tolerance", energy_tolerance),
            ("density_tolerance", density_tolerance),
            ("compile_timeout", compile_timeout),
        ):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        object.__setattr__(self, "max_bytes", max_bytes)
        object.__setattr__(self, "max_iterations", max_iterations)
        object.__setattr__(self, "diis_history", diis_history)
        object.__setattr__(self, "energy_tolerance", energy_tolerance)
        object.__setattr__(self, "density_tolerance", density_tolerance)
        object.__setattr__(self, "_closed", False)
        object.__setattr__(
            self,
            "_calculator",
            Calculator(
                method=method,
                basis=basis,
                device=backend,
                device_id=device,
                initial_guess=None,
            ),
        )
        # Basis lookup uses only element/topology data, not an SCF evaluation.
        atoms = tuple(Atom(z, (2.0 * i, 0.0, 0.0)) for i, z in enumerate(self.numbers))
        selected_basis = self._calculator._basis_for_atoms(atoms)
        # Expanding into mutable primitive inputs discards basis-level metadata.
        # Reject unsupported Hamiltonians/representations before that boundary,
        # rather than silently interpreting ECP shells as all-electron data.
        if isinstance(selected_basis, BasisSet):
            if selected_basis.representation != "cartesian":
                raise ValueError("integral AD requires a Cartesian basis")
            if any(
                selected_basis.by_element[z].ecp_core_electrons for z in self.numbers
            ):
                raise ValueError(
                    "integral AD requires an all-electron basis; ECPs are unsupported"
                )
            shells = selected_basis.shells_for(atoms)
        else:
            shells = tuple(selected_basis)
        object.__setattr__(self, "shells", tuple(shells))
        topology = tuple(
            (s.atom_index, s.angular_momentum, len(s.primitives)) for s in self.shells
        )
        if not topology or any(
            type(a) is not int
            or not 0 <= a < len(self.numbers)
            or type(l) is not int
            or not 0 <= l <= 4
            or count < 1
            for a, l, count in topology
        ):
            raise ValueError("integral AD requires nonempty Cartesian s/p/d/f/g shells")
        object.__setattr__(
            self, "nao", sum(len(cartesian_components(l)) for _, l, _ in topology)
        )
        if (electrons + spin) // 2 > self.nao or self.nao > 256:
            raise ValueError(
                "integral AD occupation/AO count is outside the admitted domain"
            )
        object.__setattr__(
            self,
            "exponents",
            immutable(
                np.array(
                    [p.exponent for s in self.shells for p in s.primitives],
                    dtype=np.float64,
                )
            ),
        )
        object.__setattr__(
            self,
            "coefficients",
            immutable(
                np.array(
                    [p.coefficient for s in self.shells for p in s.primitives],
                    dtype=np.float64,
                )
            ),
        )
        object.__setattr__(self, "nprimitive", self.exponents.size)
        n, p, a = self.nao, self.nprimitive, len(self.numbers)
        object.__setattr__(
            self,
            "numeric_peak_bytes",
            8 * (8 * n**4 + 600 * n**2 + 64 * n + 32 * p + 64 * a)
            + (
                (8 << 20) + 8 * (4 * n**4 + 200 * n**2 + 1024 * p + 384 * a)
                if backend == "cuda"
                else 0
            ),
        )
        primitive_aos = sum(
            len(cartesian_components(l)) * count for _, l, count in topology
        )
        object.__setattr__(
            self,
            "primitive_evaluations",
            primitive_aos**4 + (2 + a) * primitive_aos**2 + a * (a - 1) // 2,
        )
        if self.numeric_peak_bytes > max_bytes:
            raise ValueError("integral AD exceeds max_bytes before generation")
        if 2 * self.primitive_evaluations > max_work:
            raise ValueError("integral AD value and pullback exceed max_work")
        metadata: dict[str, int] = {}
        with compiler_work_budget(compiler_work):
            source = (
                emit_integral_program(
                    self.numbers,
                    topology,
                    second_order=derivative_order == 2,
                    backend=backend,
                    metadata=metadata,
                )
                + "\n"
                + emit_spin_integral_weights()
            )
        object.__setattr__(
            self,
            "numeric_peak_bytes",
            self.numeric_peak_bytes
            + 32 * metadata["tape_nodes"] * (32 if backend == "cuda" else 1),
        )
        if self.numeric_peak_bytes > max_bytes:
            raise ValueError(
                "integral AD tape scratch exceeds max_bytes before compilation"
            )
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / (
            sha256(source.encode()).hexdigest()
            + (".cu" if backend == "cuda" else ".cpp")
        )
        cache_source(path, source)
        header = asset_path("src/integrals/parameter_ad_runtime.hpp")
        headers = (header,)
        if derivative_order == 2 or backend == "cuda":
            headers += (asset_path("src/integrals/parameter_ad_tape.hpp"),)
        compiler: Any = CppCompilerAdapter(
            Path(os.environ.get("CXX", "c++")), compile_timeout
        )
        options = ("-ffp-contract=off", f"-I{header.parent}")
        if backend == "cuda":
            from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
            from generativeqc_compiler.common.cuda_target import cuda_target_info

            from generativeqc.profiles import find_nvcc

            nvcc = find_nvcc()
            if nvcc is None:
                raise RuntimeError("CUDA integral AD requires an NVCC toolchain")
            compiler = CudaCompilerAdapter(
                nvcc, cuda_target_info(cuda_arch), compile_timeout
            )
            options = ("--fmad=false", "-O1", f"-I{header.parent}")
            headers += (asset_path("src/integrals/parameter_ad_cuda_runtime.cuh"),)
        artifact = compile_runtime(
            compiler, cache, path, headers=headers, options=options
        )
        object.__setattr__(self, "_library", ct.CDLL(str(artifact.library)))
        object.__setattr__(self, "_execute", self._library.integral_ad_execute)
        self._execute.argtypes = ([ct.c_int] if backend == "cuda" else []) + [
            _DOUBLE
        ] * 5
        self._execute.restype = ct.c_int
        object.__setattr__(self, "_weights", self._library.integral_ad_hf_weights)
        self._weights.argtypes = [ct.c_size_t, _DOUBLE, _DOUBLE, _DOUBLE, _DOUBLE]
        self._weights.restype = ct.c_int
        if derivative_order == 2:
            object.__setattr__(
                self, "_directional", self._library.integral_ad_directional
            )
            self._directional.argtypes = ([ct.c_int] if backend == "cuda" else []) + [
                _DOUBLE
            ] * 7
            self._directional.restype = ct.c_int
        # Runtime scheduling is execution provenance too; changing the template
        # must be visible even if the generated scientific expressions match.
        object.__setattr__(
            self,
            "identity",
            sha256(
                source.encode()
                + b"\0"
                + b"".join(item.read_bytes() for item in headers)
            ).hexdigest(),
        )

    def _inputs(
        self, coordinates: Any, exponents: Any, coefficients: Any
    ) -> tuple[np.ndarray, ...]:
        if self._closed:
            raise RuntimeError("integral AD owner is closed")
        xyz = _array(coordinates, (len(self.numbers), 3), "coordinates")
        alpha = _array(
            self.exponents if exponents is None else exponents,
            (self.nprimitive,),
            "exponents",
        )
        raw = _array(
            self.coefficients if coefficients is None else coefficients,
            (self.nprimitive,),
            "coefficients",
        )
        if np.any(alpha <= 0):
            raise ValueError("exponents must be positive")
        return xyz, alpha, raw

    def _evaluate(
        self, inputs: tuple[np.ndarray, ...], seeds: np.ndarray | None = None
    ) -> np.ndarray:
        n = self.nao
        count = (
            2 * n**2 + n**4 + 1
            if seeds is None
            else 3 * len(self.numbers) + 2 * self.nprimitive
        )
        output = np.empty(count)
        status = self._execute(
            *([self.device] if self.backend == "cuda" else []),
            *map(_pointer, inputs),
            None if seeds is None else _pointer(seeds),
            _pointer(output),
        )
        if status or not np.isfinite(output).all():
            raise ArithmeticError(
                "generated integral AD failed: invalid normalization, geometry or arithmetic"
            )
        return output

    def _gradient(self, values: np.ndarray) -> IntegralGradient:
        xyz = 3 * len(self.numbers)
        return IntegralGradient(
            immutable(values[:xyz].reshape(-1, 3)),
            immutable(values[xyz : xyz + self.nprimitive]),
            immutable(values[xyz + self.nprimitive :]),
        )

    def integrals(
        self, coordinates: Any, *, exponents: Any = None, coefficients: Any = None
    ) -> IntegralValues:
        """Evaluate differentiable integral sources without solving SCF.

        Inputs follow the constructor's units/order; invalid data fail before
        native evaluation. Returned FP64 arrays are immutable owned snapshots.
        """
        values = self._evaluate(self._inputs(coordinates, exponents, coefficients))
        n, n2 = self.nao, self.nao**2
        return IntegralValues(
            immutable(values[:n2].reshape(n, n)),
            immutable(values[n2 : 2 * n2].reshape(n, n)),
            immutable(values[2 * n2 : -1].reshape((n,) * 4)),
            float(values[-1]),
        )

    def integral_vjp(
        self,
        coordinates: Any,
        cotangents: IntegralValues,
        *,
        exponents: Any = None,
        coefficients: Any = None,
    ) -> IntegralGradient:
        """Pull arbitrary full ordered integral cotangents back to all inputs.

        No symmetry of source weights is assumed. Weights must match the
        integral layouts, be finite FP64 and include the nuclear scalar seed.
        Repeated physical centers and zero contraction coefficients accumulate
        correctly; invalid input or execution failure publishes no gradient.
        """
        inputs = self._inputs(coordinates, exponents, coefficients)
        n = self.nao
        seeds = np.concatenate(
            (
                _array(cotangents.overlap, (n, n), "overlap cotangent").ravel(),
                _array(cotangents.hcore, (n, n), "hcore cotangent").ravel(),
                _array(cotangents.eri, (n,) * 4, "ERI cotangent").ravel(),
                _array(np.asarray(cotangents.nuclear), (), "nuclear cotangent").reshape(
                    1
                ),
            )
        )
        return self._gradient(self._evaluate(inputs, seeds))

    def _solve(
        self,
        inputs: tuple[np.ndarray, ...],
        values: np.ndarray,
        direction: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray, dict[str, Any], np.ndarray | None]:
        xyz, alpha, raw = inputs
        shells, begin = [], 0
        for shell in self.shells:
            count = len(shell.primitives)
            shells.append(
                Shell(
                    shell.atom_index,
                    shell.angular_momentum,
                    tuple(
                        Primitive(float(alpha[i]), float(raw[i]))
                        for i in range(begin, begin + count)
                    ),
                )
            )
            begin += count
        calculator = self._calculator
        library = calculator._library
        function = library.generativeqc_integral_ad_hf_v2
        function.argtypes = [
            ct.c_void_p,
            _DOUBLE,
            _DOUBLE,
            ct.c_size_t,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_uint,
            ct.c_uint,
            ct.c_double,
            ct.c_double,
            ct.c_size_t,
            _DOUBLE,
            _DOUBLE,
            _DOUBLE,
            ct.c_char_p,
            ct.c_size_t,
        ]
        function.restype = ct.c_int
        context, system = ct.c_void_p(), ct.c_void_p()
        n, n2 = self.nao, self.nao**2
        state, state_direction, receipt = (
            np.empty(4 * n2),
            np.empty(4 * n2),
            np.empty(8),
        )
        detail = ct.create_string_buffer(1024)
        try:
            _native.check(
                library,
                library.generativeqc_context_create(
                    ct.byref(calculator._context_descriptor()), ct.byref(context)
                ),
            )
            system = calculator._create_native_system(
                context,
                tuple(
                    Atom(z, tuple(r)) for z, r in zip(self.numbers, xyz, strict=True)
                ),
                self.charge,
                self.spin_multiplicity,
                basis=tuple(shells),
            )
            status = function(
                system,
                _pointer(values),
                None if direction is None else _pointer(direction),
                n,
                int(self.method == "uhf"),
                int(self.backend == "cuda"),
                self.device,
                self.max_iterations,
                self.diis_history,
                self.energy_tolerance,
                self.density_tolerance,
                self.max_bytes,
                _pointer(state),
                _pointer(state_direction),
                _pointer(receipt),
                detail,
                len(detail),
            )
            if status:
                raise RuntimeError(detail.value.decode())
        finally:
            if system.value:
                library.generativeqc_system_destroy(system)
            if context.value:
                library.generativeqc_context_destroy(context)
        seeds = np.empty_like(values)
        seed_direction = None if direction is None else np.empty_like(values)
        if self._weights(
            n,
            _pointer(state),
            None if direction is None else _pointer(state_direction),
            _pointer(seeds),
            None if seed_direction is None else _pointer(seed_direction),
        ):
            raise ArithmeticError("generated stationary integral pullback failed")
        diagnostics = {
            "derivative": f"stationary-{self.method}-through-integral-ad-v2",
            "derivative_order": 1 if direction is None else 2,
            "response_residual": float(receipt[6]),
            "response_operator_actions": int(receipt[7]),
            "integral_identity": self.identity,
            "backend": f"native-{self.backend}-fp64",
            "solver_control": "host-scf-and-krylov",
            "integral_execution": self.backend,
            "fock_execution": self.backend,
            "eigen_execution": self.backend,
            "iterations": int(receipt[1]),
            "physical_residual": float(receipt[2]),
            "fock_builds": int(receipt[3]),
            "energy_change": float(receipt[4]),
            "density_rms": float(receipt[5]),
            "primitive_value_evaluations": self.primitive_evaluations,
            "primitive_pullback_evaluations": self.primitive_evaluations,
            "numeric_peak_bytes_bound": self.numeric_peak_bytes,
        }
        return float(receipt[0]), seeds, diagnostics, seed_direction

    def value_and_grad(
        self, coordinates: Any, *, exponents: Any = None, coefficients: Any = None
    ) -> tuple[float, IntegralGradient]:
        """Solve HF and differentiate energy through every continuous integral input.

        Return (Hartree energy, immutable gradients). The native solver consumes
        these generated integrals, and its stationary source VJP is generated
        before the integral/normalization pullback. No analytic force endpoint
        is called. Only a converged physical state is eligible for publication.
        """
        inputs = self._inputs(coordinates, exponents, coefficients)
        values = self._evaluate(inputs)
        energy, seeds, diagnostics, _ = self._solve(inputs, values)
        gradient = self._gradient(self._evaluate(inputs, seeds))
        object.__setattr__(self, "_diagnostics", diagnostics)
        return energy, gradient

    def energy(
        self, coordinates: Any, *, exponents: Any = None, coefficients: Any = None
    ) -> float:
        """Return the converged scalar energy using generated integral values only."""
        inputs = self._inputs(coordinates, exponents, coefficients)
        value, _, diagnostics, _ = self._solve(inputs, self._evaluate(inputs))
        diagnostics["primitive_pullback_evaluations"] = 0
        object.__setattr__(self, "_diagnostics", diagnostics)
        return value

    def _direction(self, direction: IntegralGradient) -> np.ndarray:
        if self.derivative_order < 2:
            raise ValueError("construct with derivative_order=2 for integral JVP/HVP")
        return np.concatenate(
            (
                _array(
                    direction.coordinates,
                    (len(self.numbers), 3),
                    "coordinate direction",
                ).ravel(),
                _array(direction.exponents, (self.nprimitive,), "exponent direction"),
                _array(
                    direction.coefficients, (self.nprimitive,), "coefficient direction"
                ),
            )
        )

    def _evaluate_direction(
        self,
        inputs: tuple[np.ndarray, ...],
        direction: np.ndarray,
        seeds: np.ndarray | None = None,
        seed_direction: np.ndarray | None = None,
    ) -> np.ndarray:
        count = (2 * self.nao**2 + self.nao**4 + 1) if seeds is None else direction.size
        result = np.empty(count)
        status = self._directional(
            *([self.device] if self.backend == "cuda" else []),
            *map(_pointer, inputs),
            _pointer(direction),
            None if seeds is None else _pointer(seeds),
            None if seed_direction is None else _pointer(seed_direction),
            _pointer(result),
        )
        if status or not np.isfinite(result).all():
            raise ArithmeticError("generated integral directional AD failed")
        return result

    def integral_jvp(
        self,
        coordinates: Any,
        direction: IntegralGradient,
        *,
        exponents: Any = None,
        coefficients: Any = None,
    ) -> IntegralValues:
        """Differentiate all ordered integral values along a joint parameter direction.

        Requires derivative_order=2. The direction includes geometry, exponents
        and raw coefficients, so mixed paths through normalization remain live.
        """
        values = self._evaluate_direction(
            self._inputs(coordinates, exponents, coefficients),
            self._direction(direction),
        )
        n, n2 = self.nao, self.nao**2
        return IntegralValues(
            immutable(values[:n2].reshape(n, n)),
            immutable(values[n2 : 2 * n2].reshape(n, n)),
            immutable(values[2 * n2 : -1].reshape((n,) * 4)),
            float(values[-1]),
        )

    def hvp(
        self,
        coordinates: Any,
        direction: IntegralGradient,
        *,
        exponents: Any = None,
        coefficients: Any = None,
    ) -> IntegralGradient:
        """Analytic energy Hessian-vector product in all continuous inputs.

        Forward-over-reverse integral AD composes with the converged coupled
        orbital response. This includes mixed geometry/basis derivatives and
        requires derivative_order=2; it never differences energies or gradients.
        """
        inputs = self._inputs(coordinates, exponents, coefficients)
        result, diagnostics = self._hvp(inputs, direction)
        object.__setattr__(self, "_diagnostics", diagnostics)
        return result

    def _hvp(
        self,
        inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
        direction: IntegralGradient,
    ) -> tuple[IntegralGradient, dict[str, Any]]:
        """Stage an HVP without publishing diagnostics of a partial Hessian."""
        vector = self._direction(direction)
        if 4 * self.primitive_evaluations > self.max_work:
            raise ValueError("integral AD Hessian-vector product exceeds max_work")
        values = self._evaluate(inputs)
        motion = self._evaluate_direction(inputs, vector)
        _, seeds, diagnostics, seed_direction = self._solve(inputs, values, motion)
        result = self._gradient(
            self._evaluate_direction(inputs, vector, seeds, seed_direction)
        )
        diagnostics["primitive_value_evaluations"] = 2 * self.primitive_evaluations
        diagnostics["primitive_pullback_evaluations"] = self.primitive_evaluations
        return result, diagnostics

    def hessian(
        self,
        coordinates: Any,
        *,
        exponents: Any = None,
        coefficients: Any = None,
        parameters: tuple[str, ...] = ("coordinates", "exponents", "coefficients"),
    ) -> np.ndarray:
        """Return the full analytic joint Hessian, ordered xyz, exponents, coefficients.

        Each column uses the bounded HVP endpoint. The total primitive work and
        retained matrix are admitted before execution. No symmetrization hides
        response errors; each returned column is the actual differentiated result.
        """
        inputs = self._inputs(coordinates, exponents, coefficients)
        names = ("coordinates", "exponents", "coefficients")
        if (
            not isinstance(parameters, tuple)
            or not parameters
            or len(set(parameters)) != len(parameters)
            or any(name not in names for name in parameters)
        ):
            raise ValueError(
                "Hessian parameters must select distinct continuous inputs"
            )
        offsets = np.cumsum([0, *(array.size for array in inputs)])
        selected = np.concatenate(
            [
                np.arange(offsets[names.index(name)], offsets[names.index(name) + 1])
                for name in parameters
            ]
        )
        size = selected.size
        if self.numeric_peak_bytes + 8 * size * size > self.max_bytes:
            raise ValueError("integral AD Hessian exceeds max_bytes")
        if 4 * size * self.primitive_evaluations > self.max_work:
            raise ValueError("integral AD Hessian exceeds max_work")
        result = np.empty((size, size))
        diagnostics: dict[str, Any] = {}
        counters = (
            "iterations",
            "fock_builds",
            "response_operator_actions",
            "primitive_value_evaluations",
            "primitive_pullback_evaluations",
        )
        for column in range(size):
            vector = np.zeros(offsets[-1])
            vector[selected[column]] = 1
            gradient, evidence = self._hvp(inputs, self._gradient(vector))
            if not diagnostics:
                diagnostics = evidence
            else:
                for name in counters:
                    diagnostics[name] += evidence[name]
                for name in ("physical_residual", "response_residual"):
                    diagnostics[name] = max(diagnostics[name], evidence[name])
            result[:, column] = np.concatenate(
                (
                    gradient.coordinates.ravel(),
                    gradient.exponents,
                    gradient.coefficients,
                )
            )[selected]
        # Commit evidence only after every column succeeded, just like the
        # native state publication. Work counts describe the complete endpoint.
        result = immutable(result)
        diagnostics["hessian_columns"] = int(size)
        diagnostics["hessian_parameters"] = parameters
        diagnostics["numeric_peak_bytes_bound"] += 8 * size * size
        object.__setattr__(self, "_diagnostics", diagnostics)
        return result

    def diagnostics(self) -> dict[str, Any]:
        """Return detached work/provenance evidence for the last successful call."""
        return dict(getattr(self, "_diagnostics", {}))

    def close(self) -> None:
        """Close the owner; immutable results outlive it and later calls raise."""
        self._calculator.clear_cache()
        object.__setattr__(self, "_closed", True)


@dataclass(frozen=True, init=False, repr=False, eq=False)
class IntegralRHF(IntegralHF):
    """Restricted closed-shell specialization of :class:`IntegralHF`."""

    def __init__(self, elements: Sequence[str | int], **options: Any) -> None:
        """Construct an RHF owner with the shared integral/response controls."""
        super().__init__(elements, method="rhf", **options)


@dataclass(frozen=True, init=False, repr=False, eq=False)
class IntegralUHF(IntegralHF):
    """Unrestricted spin-resolved specialization of :class:`IntegralHF`."""

    def __init__(self, elements: Sequence[str | int], **options: Any) -> None:
        """Construct a UHF owner; specify spin_multiplicity for open shells."""
        super().__init__(elements, method="uhf", **options)
