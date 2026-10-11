"""CPU/CUDA binding for generated range-separated exchange first derivatives."""

import typing
from pathlib import Path

import numpy as np
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.integral.derivative_aot_registry import (
    select_packaged_derivative_aot,
)
from generativeqc_compiler.integral.ir import four_center_eri_operator
from generativeqc_compiler.integral.range_separation import CoulombKernel
from generativeqc_compiler.integral.shell_spec import cartesian_components
from generativeqc_compiler.integral.weighted_eri import build_weighted_eri_ir
from generativeqc_compiler.integral.weighted_eri_execute import (
    PreparedWeightedEri,
    compile_weighted_eri,
    packaged_weighted_eri,
)
from generativeqc_compiler.method.spec import RangeSeparatedExchangePrimitive


class RangeExchangeExecutor:
    """Bind generated SR/LR ERI derivatives to native AO tuples.

    The mathematical path is backend-neutral: the selected explicit compiler
    adapter determines whether the generated weighted-ERI provider executes on
    CPU or CUDA. Packaged AOT selection is also backend/operator based and does
    not inspect a functional or method name. This stationary-consumer binding
    uses a bounded shared Cartesian/AD recurrence on CPU through f, including
    real-spherical expansions. The explicit-subset CUDA diagnostic retains
    its s/p contract; the public CUDA endpoint uses its resident shell owner.
    """

    def __init__(
        self,
        basis: typing.Any,
        cache: typing.Any,
        primitive_tile: typing.Any,
        compiler: typing.Any,
        *,
        device_id: int = 0,
        aot_library: typing.Any = None,
    ) -> None:
        if isinstance(compiler, CppCompilerAdapter):
            backend = "cpu"
        elif isinstance(compiler, CudaCompilerAdapter):
            backend = "cuda"
        else:
            raise TypeError(
                "RSH stationary gradients require an explicit CPU C++ or CUDA compiler adapter"
            )
        if type(device_id) is not int or not 0 <= device_id < 2**31:
            raise ValueError("RSH stationary CUDA device ordinal must fit int32")
        if backend == "cpu" and device_id != 0:
            raise ValueError("CPU RSH stationary gradients require device_id=0")
        maximum_angular = 3 if backend == "cpu" else 1
        if any(shell.angular_momentum > maximum_angular for shell in basis.shells):
            raise NotImplementedError(
                f"{backend} RSH stationary gradients support angular momentum <= {maximum_angular}"
            )
        start = 3 * basis.natom
        self.primitives = basis.packed[start : start + 2 * basis.nprimitive].reshape(
            -1, 2
        )
        self.aos = basis.packed[start + 2 * basis.nprimitive :].reshape(-1, 16)
        self.centers = basis.packed[:start].reshape(-1, 3)
        if backend == "cuda" and any(int(row[3]) != 1 for row in self.aos):
            raise NotImplementedError(
                "RSH stationary gradients require one Cartesian component per public AO"
            )
        self.cache = Path(cache)
        self.basis = basis
        self.primitive_tile = primitive_tile
        self.compiler = compiler
        self.backend = backend
        self.device_id = device_id
        self.aot_library = aot_library
        self._plans: dict[tuple[typing.Any, ...], typing.Any] = {}
        self.records = 0
        self.recurrence_evaluations = 0
        self.packaged_aot_plans = 0
        self.runtime_compilations = 0

    @staticmethod
    def _kernel(primitive: typing.Any) -> CoulombKernel:
        if not isinstance(primitive, RangeSeparatedExchangePrimitive):
            raise TypeError(
                "range exchange execution requires a MethodIR range primitive"
            )
        family = {
            "short-range": "short_range",
            "long-range": "long_range",
        }.get(primitive.operator)
        if family is None:
            raise ValueError("unsupported range-exchange operator")
        return CoulombKernel(family, float(primitive.omega))

    @staticmethod
    def _component(row: typing.Any) -> tuple[int, int]:
        powers = tuple(int(value) for value in row[4:7])
        angular = sum(powers)
        label = "".join(axis * power for axis, power in zip("xyz", powers, strict=True))
        return angular, cartesian_components(angular).index(label)

    def _primitive_shell(self, row: typing.Any) -> tuple[tuple[float, float], ...]:
        begin, count = int(row[1]), int(row[2])
        return tuple(
            (float(exponent), float(coefficient))
            for exponent, coefficient in self.primitives[begin : begin + count]
        )

    def integral(
        self,
        primitive: typing.Any,
        indices: typing.Any,
        weight: typing.Any,
    ) -> typing.Any:
        rows = self.aos[list(indices)]
        if self.backend == "cpu" and (
            self.aot_library is None
            or any(shell.angular_momentum > 1 for shell in self.basis.shells)
        ):
            result, visits, evaluations = self._cpu_plan(primitive).execute(
                (indices,), (weight,), independent_slots=True
            )
            self.records += visits
            self.recurrence_evaluations += evaluations
            return [int(row[0]) for row in rows], result
        angular_and_component = tuple(self._component(row) for row in rows)
        angular = tuple(item[0] for item in angular_and_component)
        shape = tuple(len(cartesian_components(value)) for value in angular)
        component = int(
            np.ravel_multi_index(
                tuple(item[1] for item in angular_and_component), shape
            )
        )
        radial = self._kernel(primitive)
        aot_selection = select_packaged_derivative_aot(
            self.aot_library,
            backend=self.backend,
            radial=radial,
            angular=angular,
            component=component,
            target=(
                "native-host"
                if self.backend == "cpu"
                else self.compiler.target.architecture
            ),
        )
        if self.backend == "cpu" and aot_selection is None:
            result, visits, evaluations = self._cpu_plan(primitive).execute(
                (indices,), (weight,), independent_slots=True
            )
            self.records += visits
            self.recurrence_evaluations += evaluations
            return [int(row[0]) for row in rows], result
        key = (
            (
                radial.family.value,
                radial.omega,
                angular,
                "aot",
                aot_selection.key.group_index,
            )
            if aot_selection is not None
            else (radial.family.value, radial.omega, angular, "jit", component)
        )
        plan = self._plans.get(key)
        if plan is None:
            integral = build_weighted_eri_ir(
                angular, operator=four_center_eri_operator(radial)
            )
            if aot_selection is not None:
                artifact = packaged_weighted_eri(
                    integral,
                    self.aot_library,
                    component_indices=aot_selection.component_indices,
                    entry_prefix=aot_selection.entry_prefix,
                )
                self.packaged_aot_plans += 1
            else:
                artifact = compile_weighted_eri(
                    integral,
                    self.compiler,
                    self.cache,
                    component_indices=(component,),
                )
                self.runtime_compilations += 1
            if artifact.backend != self.backend:
                raise ValueError("range exchange compiled backend identity mismatch")
            plan = PreparedWeightedEri(
                artifact,
                record_capacity=self.primitive_tile,
                tile_capacity=1,
                device_id=self.device_id,
            )
            self._plans[key] = plan
        owners = [int(row[0]) for row in rows]
        primitives = tuple(self._primitive_shell(row) for row in rows)
        centers = tuple(
            tuple(float(value) for value in self.centers[owner]) for owner in owners
        )
        result = plan.raw(primitives, centers, (component,))
        self.records += int(result.diagnostics["records"])
        self.recurrence_evaluations += int(result.diagnostics["records"])
        return owners, float(weight) * result.values[0, 1:].reshape(4, 3)

    def _cpu_plan(self, primitive: typing.Any) -> typing.Any:
        """Retain one scalar schedule per frozen radial operator, not per AO."""
        from ._stationary_range_cpu import PreparedCpuRangeExchange

        radial = self._kernel(primitive)
        key = (radial.family.value, radial.omega)
        if key not in self._plans:
            self._plans[key] = PreparedCpuRangeExchange(
                self.basis, radial, self.compiler, self.cache
            )
            self.runtime_compilations += 1
        return self._plans[key]

    def integral_batch(
        self, primitive: typing.Any, indices: typing.Any, weights: typing.Any
    ) -> typing.Any:
        """Contract a bounded quartet tile directly into physical atoms on CPU."""
        if self.backend != "cpu":
            raise ValueError("compact host quartet batches require the CPU backend")
        output, visits, evaluations = self._cpu_plan(primitive).execute(
            indices, weights
        )
        self.records += visits
        self.recurrence_evaluations += evaluations
        return output

    def close(self) -> None:
        for plan in self._plans.values():
            plan.close()
        self._plans.clear()
